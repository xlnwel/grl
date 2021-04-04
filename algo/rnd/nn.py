import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd

from core.module import Module, Ensemble
from core.decorator import config


def ortho_init(scale):
    # it seems that tf.orthogonal slows training. 
    # return tf.keras.initializers.orthogonal(gain=scale)
    def _ortho_init(shape, dtype, partition_info=None):
        #lasagne ortho init for tf
        shape = tuple(shape)
        if len(shape) == 2:
            flat_shape = shape
        elif len(shape) == 4: # assumes NHWC
            flat_shape = (np.prod(shape[:-1]), shape[-1])
        else:
            raise NotImplementedError
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v # pick the one with the correct shape
        q = q.reshape(shape)
        return (scale * q[:shape[0], :shape[1]]).astype(np.float32)
    return _ortho_init

class ACCNN(Module):
    def __init__(self, name):
        super().__init__(name)

        kwargs = {
            'kernel_initializer': ortho_init(np.sqrt(2)),
            'activation': 'relu'
        }
        self._layers = [
            tf.keras.layers.Conv2D(32, 8, 4, **kwargs, name=f'{name}1'),
            tf.keras.layers.Conv2D(64, 4, 2, **kwargs, name=f'{name}2'),
            tf.keras.layers.Conv2D(64, 4, 1, **kwargs, name=f'{name}3'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, **kwargs, name=f'{name}4'),
            tf.keras.layers.Dense(448, **kwargs, name=f'{name}5'),
        ]

    def call(self, x):
        x = tf.cast(x, tf.float32)
        x = super().call(x)
        return x


class AC(Module):
    @config
    def __init__(self, action_dim, name='ac'):
        super().__init__(name=name)
        
        """ Network definition """
        self._cnn = ACCNN(name)
        
        self._eval_act_temp = getattr(self, '_eval_act_temp', 0)
        
        self._mlps = [
            tf.keras.layers.Dense(448, activation='relu', 
                kernel_initializer=ortho_init(.1),
                name='fc2act'),
            tf.keras.layers.Dense(448, activation='relu', 
                kernel_initializer=ortho_init(.1),
                name='fc2val'),
        ]
        self._actor = tf.keras.layers.Dense(action_dim, 
            kernel_initializer=ortho_init(.01),
            name='actor', dtype='float32')
        self._value_int = tf.keras.layers.Dense(1, 
            kernel_initializer=ortho_init(.01),
            name='value_int', dtype='float32')
        self._value_ext = tf.keras.layers.Dense(1, 
            kernel_initializer=ortho_init(.01),
            name='value_ext', dtype='float32')

    def call(self, x, return_value=False):
        x = self._cnn(x)
        ax, vx = x, x
        ax = self._mlps[0](ax) + ax
        actor_out = self._actor(ax)

        act_dist = tfd.Categorical(actor_out)

        if return_value:
            vx = self._mlps[1](vx) + vx
            value_int = tf.squeeze(self._value_int(vx), -1)
            value_ext = tf.squeeze(self._value_ext(vx), -1)
            return act_dist, value_int, value_ext
        else:
            return act_dist

    def compute_value(self, x):
        x = self._cnn(x)
        x = self._mlps[1](x) + x
        value_int = tf.squeeze(self._value_int(x), -1)
        value_ext = tf.squeeze(self._value_ext(x), -1)
        return value_int, value_ext
        
    def reset_states(self, **kwargs):
        return
    
    def action(self, dist, evaluation):
        return dist.mode() if evaluation and self._eval_act_temp == 0 \
            else dist.sample()


class RandomNet(Module):
    def __init__(self, name):
        super().__init__(name)

        kwargs = {
            'kernel_initializer': ortho_init(np.sqrt(2)),
            'activation': tf.keras.layers.LeakyReLU()
        }
        self._layers = [
            tf.keras.layers.Conv2D(32, 8, 4, **kwargs, name=f'{name}1'),
            tf.keras.layers.Conv2D(64, 4, 2, **kwargs, name=f'{name}2'),
            tf.keras.layers.Conv2D(64, 3, 1, **kwargs, name=f'{name}3'),
            tf.keras.layers.Flatten()
        ]
    
    def call(self, x):
        assert x.shape.rank == 5, x
        assert x.shape[-3:] == (84, 84, 1), x
        assert x.dtype == tf.float32, x
        t = x.shape[1]
        x = tf.reshape(x, (-1, *x.shape[-3:]))
        x = super().call(x)
        x = tf.reshape(x, (-1, t, x.shape[-1]))
        return x


class Target(Module):
    def __init__(self, name='target'):
        super().__init__(name=name)

        kwargs = {
            'kernel_initializer': ortho_init(np.sqrt(2)),
        }
        self._layers = [
            RandomNet(name),
            tf.keras.layers.Dense(512, **kwargs, dtype='float32')
        ]


class Predictor(Module):
    def __init__(self, name='predictor'):
        super().__init__(name=name)

        kwargs = {
            'kernel_initializer': ortho_init(np.sqrt(2)),
        }
        self._layers = [
            RandomNet(name),
            tf.keras.layers.Dense(512, activation='relu', **kwargs),
            tf.keras.layers.Dense(512, activation='relu', **kwargs),
            tf.keras.layers.Dense(512, **kwargs, dtype='float32'),
        ]

class RND(Ensemble):
    def __init__(self, config, env, **kwargs):
        super().__init__(
            model_fn=create_components, 
            config=config,
            env=env,
            **kwargs)

    @tf.function
    def action(self, x, evaluation=False, **kwargs):
        if evaluation:
            act_dist = self.ac(x, return_value=False)
            return act_dist.mode()
        else:
            act_dist, value_int, value_ext = self.ac(x, return_value=True)
            action = act_dist.sample()
            logpi = act_dist.log_prob(action)
            return action, dict(logpi=logpi, value_int=value_int, value_ext=value_ext)

    @tf.function
    def compute_value(self, x):
        return self.ac.compute_value(x)

    def reset_states(self, **kwargs):
        return

    @property
    def state_keys(self):
        return None

def create_components(config, env):
    action_dim = env.action_dim

    return dict(ac=AC(config['ac'], action_dim),
                target=Target(),
                predictor=Predictor())

def create_model(config, env, **kwargs):
    return RND(config, env, **kwargs)