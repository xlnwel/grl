import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd
from tensorflow.keras import layers

from core.module import Module, Ensemble
from core.decorator import config
from utility.rl_utils import logpi_correction, epsilon_greedy, q_log_prob
from utility.schedule import TFPiecewiseSchedule
from nn.func import mlp
from nn.utils import get_initializer


class Actor(Module):
    def __init__(self, config, action_dim, name='actor'):
        super().__init__(name=name)
        config = config.copy()

        self._action_dim = action_dim
        self.LOG_STD_MIN = config.pop('LOG_STD_MIN', -20)
        self.LOG_STD_MAX = config.pop('LOG_STD_MAX', 2)
        self._tsallis_q = config.pop('tsallis_q', 1)

        out_size = 2*action_dim
        self._layers = mlp(**config, out_size=out_size, name=name)

    def call(self, x, evaluation=False, epsilon=0, temp=1):
        x = self._layers(x)

        mu, logstd = tf.split(x, 2, -1)
        logstd = tf.clip_by_value(logstd, self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = tf.exp(logstd)
        dist = tfd.MultivariateNormalDiag(mu, std*temp)
        raw_action = dist.mode() if evaluation else dist.sample()
        action = tf.tanh(raw_action)
        if evaluation:
            raw_logpi = dist.log_prob(raw_action)
            logpi = logpi_correction(raw_action, raw_logpi, is_action_squashed=False)
            return action, {'logpi': logpi}
        if isinstance(epsilon, tf.Tensor) or epsilon:
            action = epsilon_greedy(action, epsilon, False)

        return action

    def train_step(self, x):
        x = self._layers(x)
        mu, logstd = tf.split(x, 2, -1)
        logstd = tf.clip_by_value(logstd, self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = tf.exp(logstd)
        dist = tfd.MultivariateNormalDiag(mu, std)
        raw_action = dist.sample()
        raw_logpi = dist.log_prob(raw_action)
        action = tf.tanh(raw_action)
        logpi = logpi_correction(raw_action, raw_logpi, is_action_squashed=False)
        
        if self._tsallis_q != 1:
            logpi = q_log_prob(tf.exp(logpi), self._tsallis_q)
        terms = dict(
            raw_act_std=std,
        )

        return action, logpi, terms

class Q(Module):
    def __init__(self, config, name='q'):
        super().__init__(name=name)
        config = config.copy()
        
        self._layers = mlp(**config, out_size=1, name=name)

    def call(self, x, a):
        x = tf.concat([x, a], axis=-1)
        x = self._layers(x)
        x = tf.squeeze(x, -1)
            
        return x


class Temperature(Module):
    @config
    def __init__(self, name='temperature'):
        super().__init__(name=name)

        if self._temp_type == 'state':
            kernel_initializer = get_initializer('orthogonal', gain=.01)
            self._layer = layers.Dense(1, 
                kernel_initializer=kernel_initializer, name=name)
        elif self._temp_type == 'variable':
            self._log_temp = tf.Variable(np.log(self._value), dtype=tf.float32, name=name)
        elif self._temp_type == 'constant':
            self._temp = tf.Variable(self._value, trainable=False)
        elif self._temp_type == 'schedule':
            self._temp = TFPiecewiseSchedule(self._value)
        else:
            raise NotImplementedError(f'Error temp type: {self._temp_type}')
    
    @property
    def type(self):
        return self._temp_type

    def is_trainable(self):
        return self.type in ('state', 'variable')

    def call(self, x=None):
        if self._temp_type == 'state':
            x = self._layer(x)
            log_temp = -tf.nn.softplus(x)
            log_temp = tf.squeeze(log_temp)
            temp = tf.exp(log_temp)
        elif self._temp_type == 'variable':
            log_temp = self._log_temp
            temp = tf.exp(log_temp)
        elif self._temp_type == 'constant':
            temp = self._temp
            log_temp = tf.math.log(temp)
        elif self._temp_type == 'schedule':
            assert isinstance(x, int) or (
                isinstance(x, tf.Tensor) and x.shape == ())
            temp = self._temp(x)
            log_temp = tf.math.log(temp)
        else:
            raise ValueError(self._temp_type)
    
        return log_temp, temp


class SAC(Ensemble):
    def __init__(self, config, env, **kwargs):
        super().__init__(
            model_fn=create_components, 
            config=config,
            env=env,
            **kwargs)

    @tf.function
    def action(self, x, evaluation=False, epsilon=0, temp=1, **kwargs):
        if x.shape.ndims % 2 != 0:
            x = tf.expand_dims(x, axis=0)
        assert x.shape.ndims == 2, x.shape
        
        action = self.actor(x, evaluation=evaluation, epsilon=epsilon)
        action = tf.nest.map_structure(lambda x: tf.squeeze(x), action)

        return action

    @tf.function
    def value(self, x):
        if x.shape.ndims % 2 != 0:
            x = tf.expand_dims(x, axis=0)
        assert x.shape.ndims == 2, x.shape
        
        value = self.q(x)
        
        return value


def create_components(config, env):
    action_dim = env.action_dim

    actor_config = config['actor']
    q_config = config['q']
    temperature_config = config['temperature']
        
    return dict(
        actor=Actor(actor_config, action_dim),
        q=Q(q_config, 'q'),
        q2=Q(q_config, 'q2'),
        target_q=Q(q_config, 'target_q'),
        target_q2=Q(q_config, 'target_q2'),
        temperature=Temperature(temperature_config),
    )

def create_model(config, env, **kwargs):
    return SAC(config, env, **kwargs)