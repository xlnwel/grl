import tensorflow as tf

from core.module import Ensemble
from algo.ppo.nn import Encoder, Actor, Value


class DAAC(Ensemble):
    def __init__(self, config, env, **kwargs):
        super().__init__(
            model_fn=create_components, 
            config=config,
            env=env,
            **kwargs)

    @tf.function
    def action(self, obs, evaluation=False, **kwargs):
        x = self.encoder(obs)
        act_dist = self.actor(x, evaluation=evaluation)
        action = self.actor.action(act_dist, evaluation)

        if evaluation:
            return action
        else:
            logpi = act_dist.log_prob(action)
            if hasattr(self, 'value_encoder'):
                x = self.value_encoder(obs)
            value = self.value(x)
            terms = {'logpi': logpi, 'value': value}
            return action, terms    # keep the batch dimension for later use

    @tf.function
    def compute_value(self, x):
        if hasattr(self, 'value_encoder'):
            x =self.value_encoder(x)
        else:
            x = self.encoder(x)
        value = self.value(x)
        return value

    @tf.function
    def compute_aux_data(self, obs):
        x = self.encoder(obs)
        logits = self.actor(x).logits
        if hasattr(self, 'value_encoder'):
            x =self.value_encoder(obs)
        value = self.value(x)
        return logits, value

    def reset_states(self, **kwargs):
        return

    @property
    def state_keys(self):
        return None


def create_components(config, env):
    action_dim = env.action_dim
    is_action_discrete = env.is_action_discrete

    models = dict(
        encoder=Encoder(config['encoder']), 
        actor=Actor(config['actor'], action_dim, is_action_discrete),
        advantage=Value(config['advantage'], name='advantage'),
        value_encoder=Encoder(config['encoder'], name='value_encoder'), 
        value=Value(config['value']),
        aux_value=Value(config['value'], name='aux_value'),
    )
    
    return models

def create_model(config, env, **kwargs):
    return DAAC(config, env, **kwargs)
