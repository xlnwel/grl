import tensorflow as tf

from core.module import Module
from nn.func import Encoder, mlp
from algo.ppo.nn import Actor, Value, PPO


class Discriminator(Module):
    def __init__(self, config, name='discriminator'):
        super().__init__(name)
        config = config.copy()

        self._obs_action = config.pop('obs_action', True)
        self._g = mlp(**config, out_size=1, name='g')   # the AIRL paper use a linear model here, we use the same model for g and h for simplicity
        self._h = mlp(**config, out_size=1, name='h')
    
    def call(self, obs, action, discount, logpi, next_obs):
        f = self.f(obs, action, discount, next_obs)
        return f - logpi
    
    def f(self, obs, action, discount, next_obs):
        if self._obs_action:
            obs_action = tf.concat([obs, action], -1)
        else:
            obs_action = obs
        g = tf.squeeze(self._g(obs_action), -1)
        h = tf.squeeze(self._h(obs), -1)
        next_h = tf.squeeze(self._h(next_obs), axis=-1)
        return g + discount * next_h - h

    def compute_reward(self, obs, action, discount, next_obs, logpi):
        logits = self(obs, action, discount, next_obs, logpi)
        # return tf.math.log_sigmoid(logits) - tf.math.log_sigmoid(-logits)
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
