import logging

from core.module import Module
from nn.registry import cnn_registry
from nn.utils import *


logger = logging.getLogger(__name__)

@cnn_registry.register('nature')
class NatureCNN(Module):
    def __init__(self, 
                 *, 
                 time_distributed=False, 
                 obs_range=[0, 1], 
                 name='nature', 
                 kernel_initializer='glorot_uniform',
                 activation='relu',
                 out_size=512,
                 padding='valid',
                 **kwargs):
        super().__init__(name=name)
        self._obs_range = obs_range
        self._time_distributed = time_distributed

        gain = kwargs.pop('gain', calculate_gain(activation))
        logger.debug(f'{self.name} gain: {gain}')
        kernel_initializer = get_initializer(kernel_initializer, gain=gain)
        kwargs['kernel_initializer'] = kernel_initializer
        activation = get_activation(activation)
        kwargs['activation'] = activation
        kwargs['padding'] = padding

        self._conv_layers = [
            layers.Conv2D(32, 8, 4, **kwargs),
            layers.Conv2D(64, 4, 2, **kwargs),
            layers.Conv2D(64, 3, 1, **kwargs),
        ]
        self._flat = layers.Flatten()
        self.out_size = out_size
        if out_size:
            self._dense = layers.Dense(self.out_size, activation=activation, 
                kernel_initializer=kernel_initializer)

    def call(self, x):
        x = convert_obs(x, self._obs_range, global_policy().compute_dtype)
        if self._time_distributed:
            t = x.shape[1]
            x = tf.reshape(x, [-1, *x.shape[2:]])
        for l in self._conv_layers:
            x = l(x)
        x = self._flat(x)
        if self.out_size:
            x = self._dense(x)
        if self._time_distributed:
            x = tf.reshape(x, [-1, t, *x.shape[1:]])
        return x