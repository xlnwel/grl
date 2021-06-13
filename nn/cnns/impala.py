import functools
from tensorflow.keras.mixed_precision import global_policy

from core.module import Module
from nn.registry import cnn_registry, subsample_registry, block_registry
from nn.utils import *


@cnn_registry.register('impala')
class IMPALACNN(Module):
    def __init__(self, 
                 *, 
                 time_distributed=False, 
                 obs_range=[0, 1], 
                 filters=[16, 32, 32],
                 kernel_initializer='glorot_uniform',   # there is no need to use orthogonal here
                 subsample_type='conv_maxpool',
                 subsample_kwargs={},
                 block='resv2',
                 block_kwargs=dict(
                    filter_coefs=[],
                    kernel_sizes=[3, 3],
                    norm=None,
                    norm_kwargs={},
                    activation='relu',
                    am=None,
                    am_kwargs={},
                    dropout_rate=0.,
                    rezero=False,
                 ),
                 out_activation='relu',
                 out_size=None,
                 deter_stoch=False,
                 name='impala',
                 **kwargs):
        super().__init__(name=name)
        self._obs_range = obs_range
        self._time_distributed = time_distributed

        # kwargs specifies general kwargs for conv2d
        kwargs['kernel_initializer'] = get_initializer(kernel_initializer)
        assert 'activation' not in kwargs, kwargs
        
        block_cls = block_registry.get(block)
        block_kwargs.update(kwargs.copy())

        subsample_cls = subsample_registry.get(subsample_type)
        subsample_kwargs.update(kwargs.copy())

        self._layers = []
        prefix = f'{self.scope_name}/'
        with self.name_scope:
            fs = filters[:-1] if deter_stoch else filters
            for i, f in enumerate(fs):
                if 'conv' in subsample_type:
                    subsample_kwargs['filters'] = f

                name_fn = lambda cls_name, suffix='': prefix+f'{cls_name}_f{f}_{i}'+suffix
                self._layers += [
                    subsample_cls(name=name_fn(subsample_type), **subsample_kwargs),
                    block_cls(name=name_fn(block, '_1'), **block_kwargs),
                    block_cls(name=name_fn(block, '_2'), **block_kwargs),
                ]
            
            if deter_stoch:
                f = filters[-1]
                subsample_kwargs['filters'] = f
                ds_cls = block_registry.get('dsl')
                self._ds_layer = ds_cls(
                    n_blocks=2,
                    subsample='conv_maxpool', 
                    subsample_kwargs=subsample_kwargs, 
                    block=block,
                    block_kwargs=block_kwargs,
                    name=f'{prefix}ds')
                self._layers += [self._ds_layer]
                self._training_cls.append(ds_cls)
            out_act_cls = get_activation(out_activation, return_cls=True)
            self._layers += [
                out_act_cls(name=prefix+out_activation)
            ]
            self._flat = layers.Flatten(name=prefix+'flatten')
            self.out_size = out_size
            if self.out_size:
                if out_activation is None:
                    self._layers.append[layers.ReLU()]

                self._dense = layers.Dense(
                    self.out_size, 
                    activation=out_act_cls(), 
                    name=f'{prefix}out')
        
        self._training_cls += [subsample_cls, block_cls]
    
    def call(self, x, training=False, return_cnn_out=False):
        x = convert_obs(x, self._obs_range, global_policy().compute_dtype)
        if self._time_distributed:
            t = x.shape[1]
            x = tf.reshape(x, [-1, *x.shape[2:]])
        x = super().call(x, training=training)
        self.cnn_out = x
        x = self._flat(x)
        if self._time_distributed:
            x = tf.reshape(x, [-1, t, *x.shape[1:]])
        
        if self.out_size:
            x = self._dense(x)

        return x

IMPALA64 = functools.partial(IMPALACNN, filters=[32, 64, 64], name='impala64')
cnn_registry.register('impala64')(IMPALA64)
