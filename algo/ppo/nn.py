import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd

from core.module import Module, Ensemble
from nn.func import Encoder, mlp


class Actor(Module):
    def __init__(self, config, action_dim, is_action_discrete, name='actor'):
        super().__init__(name=name)
        config = config.copy()

        self.action_dim = action_dim
        self.is_action_discrete = is_action_discrete
        self.eval_act_temp = config.pop('eval_act_temp', 1)
        assert self.eval_act_temp >= 0, self.eval_act_temp

        self._init_std = config.pop('init_std', 1)
        if not self.is_action_discrete:
            self.logstd = tf.Variable(
                initial_value=np.log(self._init_std)*np.ones(action_dim), 
                dtype='float32', 
                trainable=True, 
                name=f'actor/logstd')
        config.setdefault('out_gain', .01)
        self._layers = mlp(**config, 
                        out_size=action_dim, 
                        out_dtype='float32',
                        name=name)

    def call(self, x, evaluation=False):
        actor_out = self._layers(x)
        if self.is_action_discrete:
            logits = actor_out / self.eval_act_temp \
                if evaluation and self.eval_act_temp else actor_out
            act_dist = tfd.Categorical(logits)
        else:
            std = tf.exp(self.logstd)
            if evaluation and self.eval_act_temp:
                std = std * self.eval_act_temp
            act_dist = tfd.MultivariateNormalDiag(actor_out, std)
        return act_dist

    def action(self, dist, evaluation):
        return dist.mode() if evaluation and self.eval_act_temp == 0 \
            else dist.sample()


class Value(Module):
    def __init__(self, config, name='value'):
        super().__init__(name=name)
        config = config.copy()
        
        config.setdefault('out_gain', 1)
        self._layers = mlp(**config,
                          out_size=1,
                          out_dtype='float32',
                          name=name)

    def call(self, x):
        value = self._layers(x)
        value = tf.squeeze(value, -1)
        return value


def create_components(config, env):
    action_dim = env.action_dim
    is_action_discrete = env.is_action_discrete

    return dict(
        encoder=Encoder(config['encoder']), 
        actor=Actor(config['actor'], action_dim, is_action_discrete),
        value=Value(config['value'])
    )


class PPO(Ensemble):
    def __init__(self, config, env, model_fn=create_components, **kwargs):
        super().__init__(
            model_fn=model_fn, 
            config=config,
            env=env,
            **kwargs)

    @tf.function
    def action(self, obs, evaluation=False, return_value=True, **kwargs):
        x = self.encode(obs)
        act_dist = self.actor(x, evaluation=evaluation)
        action = self.actor.action(act_dist, evaluation)

        if evaluation:
            return action
        else:
            logpi = act_dist.log_prob(action)
            if return_value:
                if hasattr(self, 'value_encoder'):
                    x = self.value_encoder(obs)
                value = self.value(x)
                terms = {'logpi': logpi, 'value': value}
            else:
                terms = {'logpi': logpi}
            return action, terms    # keep the batch dimension for later use

    @tf.function
    def compute_value(self, x):
        if hasattr(self, 'value_encoder'):
            x = self.value_encoder(x)
        else:
            x = self.encoder(x)
        value = self.value(x)
        return value

    def encode(self, x):
        return self.encoder(x)


def create_model(config, env, **kwargs):
    return PPO(config, env, create_components, **kwargs)
