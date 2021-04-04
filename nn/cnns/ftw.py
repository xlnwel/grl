from core.module import Module
from nn.registry import cnn_registry
from nn.utils import *


relu = activations.relu

@cnn_registry.register('ftw')
class FTWCNN(Module):
    def __init__(self, 
                 *, 
                 time_distributed=False, 
                 name='ftw', 
                 obs_range=[0, 1], 
                 kernel_initializer='glorot_uniform',
                 out_size=256,
                 **kwargs):
        super().__init__(name=name)
        self._obs_range = obs_range
        self._time_distributed = time_distributed

        gain = kwargs.pop('gain', calculate_gain('relu'))
        kernel_initializer = get_initializer(kernel_initializer, gain=gain)
        kwargs['kernel_initializer'] = kernel_initializer
        
        self._conv1 = layers.Conv2D(32, 8, strides=4, padding='same', **kwargs)
        self._conv2 = layers.Conv2D(64, 4, strides=2, padding='same', **kwargs)
        self._conv3 = layers.Conv2D(64, 3, strides=1, padding='same', **kwargs)
        self._conv4 = layers.Conv2D(64, 3, strides=1, padding='same', **kwargs)

        self._flat = layers.Flatten()

        self.out_size = out_size
        if self.out_size:
            self._dense = layers.Dense(self.out_size, activation=relu,
                            kernel_initializer=kernel_initializer)

    def call(self, x):
        x = convert_obs(x, self._obs_range, global_policy().compute_dtype)
        if self._time_distributed:
            t = x.shape[1]
            x = tf.reshape(x, [-1, *x.shape[2:]])
        x = relu(self._conv1(x))
        x = self._conv2(x)
        y = relu(x)
        y = self._conv3(y)
        x = x + y
        y = relu(x)
        y = self._conv4(y)
        x = x + y
        x = relu(x)
        if self._time_distributed:
            x = tf.reshape(x, [-1, t, *x.shape[1:]])
        x = self._flat(x)
        if self.out_size:
            x = self._dense(x)

        return x