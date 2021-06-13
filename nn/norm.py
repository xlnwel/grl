import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, initializers


EVONORM_B0 = 'evonorm_b0'
EVONORM_S0 = 'evonorm_s0'
LAYER_TYPES = (EVONORM_B0, EVONORM_S0)


class EvoNorm(layers.Layer):
    def __init__(self, 
                 name='evonorm',
                 layer_type=EVONORM_B0,
                 nonlinear=True,
                 num_groups=32,
                 decay=.9,
                 epsilon=1e-5):
        super().__init__(name=name)
        assert layer_type in LAYER_TYPES, f'Expected layer type({LAYER_TYPES}), but get {layer_type}'
        self._layer_type = layer_type
        self._num_groups = num_groups
        self._decay = decay
        self._epsilon = epsilon

    def build(self, input_shape):
        var_shape = np.ones_like(input_shape, dtype=int)
        var_shape[-1] = input_shape[-1]
        var_shape = tuple(var_shape)

        self.beta = self.add_weight(
            'beta',
            shape=var_shape,
            initializer=initializers.zeros,
            dtype=self._compute_dtype
        )
        self.gamma = self.add_weight(
            'gamma',
            shape=var_shape, 
            initializer=initializers.ones,
            dtype=self._compute_dtype
        )
        self.v = self.add_weight(
            'v',
            shape=var_shape,
            initializer=initializers.ones,
            dtype=self._compute_dtype
        )
        if self._layer_type == EVONORM_B0:
            self.moving_variance = self.add_weight(
                'moving_variance',
                shape=var_shape,
                initializer=initializers.ones,
                dtype=tf.float32,
                trainable=False
            )

    def call(self, x, training=True):
        if self._layer_type == EVONORM_S0:
            std = self._group_std(x)
            x = x * tf.nn.sigmoid(self.v * x) / std
        elif self._layer_type == EVONORM_B0:
            left = self._batch_std(x, training=training)
            right = self.v * x + self._instance_std(x)
            x = x / tf.maximum(left, right)
        else:
            raise ValueError(f'Unkown EvoNorm layer: {self._layer_type}')
        
        return x * self.gamma + self.beta

    def _batch_std(self, x, training=True):
        axes = tuple(range(len(x.shape)-1))
        if training:
            _, variance = tf.nn.moments(x, axes, keepdims=True)
            variance = tf.cast(variance, tf.float32)
            self.moving_variance.assign_sub((self.moving_variance - variance) * (1 - self._decay))
        else:
            variance = self.moving_variance
        std = tf.sqrt(variance+self._epsilon)
        return tf.cast(std, x.dtype)

    def _group_std(self, x):
        n = self._num_groups
        while n > 1:
            if x.shape[-1] % n == 0:
                break
            n -= 1
        x_shape = (-1,) + tuple(x.shape[1:])
        h, w, c = x.shape[-3:]
        g = c // n
        grouped_shape = (-1, ) + tuple(x.shape[1:-1]) + (n, g)
        x = tf.reshape(x, grouped_shape)
        _, variance = tf.nn.moments(x, [1, 2, 4], keepdims=True)
        std = tf.sqrt(variance + self._epsilon)
        std = tf.tile(std, [1, h, w, 1, g])
        std = tf.reshape(std, x_shape)
        return std

    def _instance_std(self, x):
        _, variance = tf.nn.moments(x, [-3, -2], keepdims=True)
        std = tf.sqrt(variance + self._epsilon)
        return std

if __name__ == '__main__':
    tf.random.set_seed(0)
    x = tf.random.normal((2, 4, 4, 32))
    net = EvoNorm(layer_type=EVONORM_S0, num_groups=4)
    print(net(x))