import tensorflow as tf

from core.module import Module
from nn.func import Encoder, mlp
from algo.ppo.nn import Actor, Value, PPO


class Discriminator(Module):
    def __init__(self, config, name='discriminator'):
        super().__init__(name)
        config = config.copy()

        self._layers = mlp(**config, out_size=1)
        
    def call(self, obs, action):
        x = tf.concat([obs, action], axis=-1)
        x = self._layers(x)
        x = tf.squeeze(x)
        return x
    
    def compute_reward(self, obs, action):
        logits = self(obs, action)
        return -tf.math.log_sigmoid(-logits)

def create_components(config, env):
    action_dim = env.action_dim
    is_action_discrete = env.is_action_discrete

    return dict(
        encoder=Encoder(config['encoder']), 
        actor=Actor(config['actor'], action_dim, is_action_discrete),
        value=Value(config['value']),
        discriminator=Discriminator(config['discriminator']),
    )

def create_model(config, env, **kwargs):
    return PPO(config, env, create_components, **kwargs)
