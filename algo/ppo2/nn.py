import tensorflow as tf

from utility.tf_utils import assert_rank
from core.module import Ensemble
from nn.func import rnn
from algo.ppo.nn import Encoder, Actor, Value


class PPO(Ensemble):
    def __init__(self, config, env, **kwargs):
        super().__init__(
            model_fn=create_components, 
            config=config,
            env=env,
            **kwargs)
    
    @tf.function
    def action(self, x, state, mask, evaluation=False, 
            prev_action=None, prev_reward=None, **kwargs):
        assert x.shape.ndims % 2 == 0, x.shape
        x, state = self.encode(
            x, state, mask, prev_action, prev_reward)
        act_dist = self.actor(x, evaluation=evaluation)
        action = self.actor.action(act_dist, evaluation)
        if evaluation:
            return action, state
        else:
            value = self.value(x)
            logpi = act_dist.log_prob(action)
            terms = {'logpi': logpi, 'value': value}
            # intend to keep the batch dimension for later use
            out = (action, terms)
            return out, state

    @tf.function(experimental_relax_shapes=True)
    def compute_value(self, x, state, mask, 
                    prev_action=None, prev_reward=None, return_state=False):
        x, state = self.encode(
            x, state, mask, prev_action, prev_reward)
        value = self.value(x)
        if return_state:
            return value, state
        else:
            return value

    def encode(self, x, state, mask, prev_action=None, prev_reward=None):
        if x.shape.ndims % 2 == 0:
            x = tf.expand_dims(x, 1)
        if mask.shape.ndims < 2:
            mask = tf.reshape(mask, (-1, 1))
        assert_rank(mask, 2)
        x = self.encoder(x)
        if hasattr(self, 'rnn'):
            additional_rnn_input = self._process_additional_input(
                x, prev_action, prev_reward)
            x, state = self.rnn(x, state, mask, 
                additional_input=additional_rnn_input)
        else:
            state = None
        if x.shape[1] == 1:
            x = tf.squeeze(x, 1)
        return x, state

    def _process_additional_input(self, x, prev_action, prev_reward):
        results = []
        if prev_action is not None:
            if self.actor.is_action_discrete:
                if prev_action.shape.ndims < 2:
                    prev_action = tf.reshape(prev_action, (-1, 1))
                prev_action = tf.one_hot(prev_action, self.actor.action_dim, dtype=x.dtype)
            else:
                if prev_action.shape.ndims < 3:
                    prev_action = tf.reshape(prev_action, (-1, 1, self.actor.action_dim))
            assert_rank(prev_action, 3)
            results.append(prev_action)
        if prev_reward is not None:
            if prev_reward.shape.ndims < 2:
                prev_reward = tf.reshape(prev_reward, (-1, 1, 1))
            elif prev_reward.shape.ndims == 2:
                prev_reward = tf.expand_dims(prev_reward, -1)
            assert_rank(prev_reward, 3)
            results.append(prev_reward)
        assert_rank(results, 3)
        return results

    def reset_states(self, states=None):
        if hasattr(self, 'rnn'):
            self.rnn.reset_states(states)

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        return self.rnn.get_initial_state(
            inputs, batch_size=batch_size, dtype=dtype) \
                if hasattr(self, 'rnn') else None


def create_components(config, env):
    action_dim = env.action_dim
    is_action_discrete = env.is_action_discrete

    if 'cnn_name' in config['encoder']:
        config['encoder']['time_distributed'] = True
    models = dict(
        encoder=Encoder(config['encoder']), 
        rnn=rnn(config['rnn']),
        actor=Actor(config['actor'], action_dim, is_action_discrete),
        value=Value(config['value'])
    )

    return models

def create_model(config, env, **kwargs):
    return PPO(config, env, **kwargs)
