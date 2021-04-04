from tensorflow.keras.mixed_precision import global_policy

from core.module import Module
from nn.registry import cnn_registry, subsample_registry, block_registry
from nn.utils import *


@cnn_registry.register('efficientnet')
class EfficientNet(Module):
    def __init__(self, 
                 *, 
                 time_distributed=False, 
                 obs_range=[0, 1], 
                 kernel_initializer='en_conv',
                 stem_type=None,
                 stem_kwargs={},
                 block_kwargs: dict(
                    expansion_ratios=[1, 6, 6, 6],
                    kernel_sizes=[3, 3, 3, 3],
                    strides=[2, 2, 2, 2],
                    out_filters=[16, 32, 32, 64],
                    num_repeats=[2, 2, 2, 1],
                    norm='batch',
                    norm_kwargs={},
                    am='se',
                    am_kwargs={},
                 ),
                 out_activation='relu',
                 out_size=None,
                 name='efficientnet',
                 rezero=False,
                 subsample_type='strided_mb',
                 subsample_kwargs={},
                 **kwargs):
        super().__init__(name=name)
        self._obs_range = obs_range
        self._time_distributed = time_distributed
        self._stem_type = stem_type
        self._stem_kwargs = stem_kwargs

        # kwargs specifies general kwargs for conv2d
        kwargs['time_distributed'] = time_distributed
        kwargs['kernel_initializer'] = get_initializer(kernel_initializer)
        assert 'activation' not in kwargs, kwargs
        self._kwargs = kwargs

        self._block_cls = block_registry.get('mb')
        block_kwargs.update(kwargs)
        self._block_kwargs = block_kwargs

        self._subsample_type = subsample_type
        self._subsample_cls = subsample_registry(subsample_type)
        subsample_kwargs.update(kwargs.copy())
        self._subsample_kwargs = subsample_kwargs

        self._out_act = out_activation
        self.out_size = out_size

    def build(self, input_shape):
        self._convs = []
        block_expansion_ratios = self._block_kwargs.pop('expansion_ratios')
        block_kernel_sizes = self._block_kwargs.pop('kernel_sizes')
        block_num_repeats = self._block_kwargs.pop('num_repeats')
        block_strides = self._block_kwargs.pop('strides')
        block_out_filters = self._block_kwargs.pop('out_filters')
        block_kwargs = self._block_kwargs
        subsample_kwargs = self._subsample_kwargs

        # stem part
        prefix = f'{self.scope_name}/'
        if self._stem_type:
            self._convs += [
                subsample_registry(self._stem_type)(filters=3, **subsample_kwargs)
            ]

        for i, (er, ks, nr, s, of) in enumerate(
                zip(block_expansion_ratios, block_kernel_sizes, 
                    block_num_repeats, block_strides, block_out_filters)):
            block_kwargs['expansion_ratio'] = er
            block_kwargs['kernel_size']= ks
            block_kwargs['out_filters']= of
            subsample_kwargs['strides']= s
            if self._subsample_type == 'strided_mb':
                subsample_kwargs['expansion_ratio'] = er
                subsample_kwargs['kernel_size']= ks
                subsample_kwargs['out_filters']= of
            name_fn = lambda cls_name, suffix='': prefix+f'{cls_name}_{i}_e{er}_o{of}'+suffix
            assert nr > 0, nr
            for n in range(nr):
                if n == 0 and (s > 1):
                    self._convs.append(
                        self._subsample_cls(name=name_fn(self._subsample_type), **self._subsample_kwargs))
                else:
                    block_kwargs['strides'] = 1
                    self._convs.append(
                        self._block_cls(name=name_fn(f'mb_{n}'), **block_kwargs))
            self._flat = layers.Flatten(name=prefix+'flatten')
            out_act_cls = get_activation(self._out_act, return_cls=True)
            self._out_act = out_act_cls(name=prefix+self._out_act)

            if self.out_size:
                self._dense = layers.Dense(self.out_size, activation=self._out_act, name=prefix+'out')
    
    def call(self, x, training=True, return_cnn_out=False):
        x = convert_obs(x, self._obs_range, global_policy().compute_dtype)
        if self._time_distributed:
            t = x.shape[1]
            x = tf.reshape(x, [-1, *x.shape[2:]])
        for l in self._convs:
            x = l(x)
        x = self._out_act(x)
        if self._time_distributed:
            x = tf.reshape(x, [-1, t, *x.shape[1:]])
        z = self._flat(x)
        if self.out_size:
            z = self._dense(z)
        if return_cnn_out:
            return z, x
        else:
            return z
