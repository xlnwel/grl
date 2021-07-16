import tensorflow as tf

from algo.ppo.nn import Encoder, Actor, Value, PPO


class PPG(PPO):
    @tf.function
    def compute_aux_data(self, obs):
        x = self.encoder(obs)
        logits = self.actor(x).logits
        if hasattr(self, 'value_encoder'):
            x =self.value_encoder(obs)
        value = self.value(x)
        return logits, value

def create_components(config, env):
    action_dim = env.action_dim
    is_action_discrete = env.is_action_discrete

    if config['architecture'] == 'shared':
        models = dict(
            encoder=Encoder(config['encoder']), 
            actor=Actor(config['actor'], action_dim, is_action_discrete),
            value=Value(config['value'])
        )
    elif config['architecture'] == 'dual':
        models = dict(
            encoder=Encoder(config['actor_encoder'], name='actor_encoder'), 
            value_encoder=Encoder(config['value_encoder'], name='value_encoder'), 
            actor=Actor(config['actor'], action_dim, is_action_discrete),
            value=Value(config['value']),
            aux_value=Value(config['value'], name='aux_value'),
        )
    else:
        raise ValueError(f'Unknown architecture: {config["architecture"]}')
    return models

def create_model(config, env, **kwargs):
    return PPG(config, env, create_components, **kwargs)
