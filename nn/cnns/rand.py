from core.module import Module
from nn.registry import cnn_registry
from nn.utils import *


@cnn_registry.register('rand')
class RandCNN(Module):
    def __init__(self,
                 *,
                 time_distributed=False,
                 obs_range=[0, 1],
                 kernel_size=3,
                 strides=1,
                 kernel_initializer='glorot_normal',
                 name='rand'):
        super().__init__(name=name)
        self._obs_range = obs_range
        self._kernel_size = kernel_size
        self._kernel_initializer = kernel_initializer
        self._time_distributed = time_distributed

    def build(self, input_shape):
        filters = input_shape[-1]
        self._layer = layers.Conv2D(
            filters, 
            self._kernel_size, 
            padding='same', 
            kernel_initializer=self._kernel_initializer,
            trainable=False,
            use_bias=False,
            time_distributed=self._time_distributed)
    
    def call(self, x):
        x = convert_obs(x, self._obs_range, global_policy().compute_dtype)
        x = self._layer(x)
        return x