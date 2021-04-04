import functools
import tensorflow as tf
from tensorflow.keras import layers

from core.module import Module
from nn.registry import am_registry, layer_registry, block_registry, subsample_registry
from nn.utils import *


class ResidualBase(Module):
    def __init__(self, 
                 *,
                 name='resv1', 
                 conv='conv2d',
                 filters=None,      # output filters. "filters" here is to be consistent with subsample kwargs
                 filter_coefs=[],
                 kernel_sizes=[3, 3],
                 strides=1,
                 norm=None,
                 norm_kwargs={},
                 activation='relu',
                 act_kwargs={},
                 am=None,
                 am_kwargs={},
                 skip=True,
                 dropout_rate=0,
                 rezero=False,
                 subsample_type='maxblurpool',
                 subsample_kwargs={},
                 out_activation=None,
                 **kwargs):
        super().__init__(name=name)
        self._conv = conv
        self._out_filters = filters
        self._filter_coefs = filter_coefs
        self._kernel_sizes = kernel_sizes
        self._strides = strides
        self._norm = norm
        self._norm_kwargs = norm_kwargs.copy()
        self._activation = activation
        self._act_kwargs = act_kwargs.copy()
        self._am = am
        self._am_kwargs = am_kwargs.copy()
        self._skip = skip
        self._dropout_rate = dropout_rate
        self._use_rezero = rezero
        assert 'conv' not in subsample_type, subsample_type # if conv is involved in subsample, we have to take extra care of "training" when using indirect batch norm, which only complicates code. On the other hand directly calling batch norm is handled implicitly by Module
        self._subsample_type = subsample_type
        self._subsample_kwargs = subsample_kwargs.copy()
        self._out_act = out_activation
        self._kwargs = kwargs

    def build(self, input_shape):
        kwargs = self._kwargs.copy()
        out_filters = self._out_filters or input_shape[-1]
        filter_coefs = self._filter_coefs or [1 for _ in self._kernel_sizes]
        filters = [int(out_filters * fc) for fc in filter_coefs]
        if self._out_filters:
            filters[-1] = self._out_filters
        if isinstance(self._strides, int):
            strides = [1 for _ in self._kernel_sizes]
            strides[0] = self._strides # TODO: strided in the beginning
        else:
            assert isinstance(self._strides, (list, tuple)), self._strides
            strides = self._strides
        am_kwargs = self._am_kwargs.copy()
        am_kwargs.update(kwargs)

        self._layers = []
        conv_cls = layer_registry.get(self._conv)
        self._norm_cls = get_norm(self._norm)
        act_cls = get_activation(self._activation, return_cls=True)
        subsample_cls = subsample_registry.get(self._subsample_type)

        prefix = f'{self.scope_name}/'
        assert len(filters) == len(self._kernel_sizes) == len(strides) <= 3, \
            (filters, self._kernel_sizes, strides)
        self._build_residual_branch(
            filters, 
            self._kernel_sizes, 
            strides, 
            prefix, 
            subsample_cls, 
            conv_cls, 
            act_cls, 
            kwargs)

        am_cls = am_registry.get(self._am)
        self._layers.append(am_cls(name=prefix+f'{self._am}', **am_kwargs))

        if self._skip:
            if self._strides > 1:
                self._subsample = [
                    subsample_cls(name=prefix+f'identity_{self._subsample_type}', **self._subsample_kwargs),
                    conv_cls(filters[-1], 1, name=prefix+f'identity_{self._conv}'),
                    self._norm_cls(**self._norm_kwargs, name=prefix+f'identity_{self._norm}')
                ]
            if self._dropout_rate != 0:
                noise_shape = (None, 1, 1, 1)
                # Drop the entire residual branch with certain probability, https://arxiv.org/pdf/1603.09382.pdf
                # TODO: recalibrate the output at test time
                self._layers.append(
                    layers.Dropout(self._dropout_rate, noise_shape, name=prefix+'dropout'))
            if self._use_rezero:
                self._rezero = tf.Variable(0., trainable=True, dtype=tf.float32, name=prefix+'rezero')
        
        out_act_cls = get_activation(self._out_act, return_cls=True)
        self._out_act = out_act_cls(name=prefix+self._out_act if self._out_act else '')
        self._training_cls += [subsample_cls, am_cls]

    def call(self, x, training=False):
        y = super().call(x, training=training)

        if self._skip:
            if self._strides > 1:
                for l in self._subsample:
                    x = l(x)
            if self._use_rezero:
                y = self._rezero * y
            return self._out_act(x + y)
        else:
            return self._out_act(y)

    def _build_residual_branch(self, filters, kernel_size, strides, prefix, subsample_cls, conv_cls, act_cls, kwargs):
        pass

@block_registry.register('resv1')
class ResidualV1(ResidualBase):
    def __init__(self, name='resv1', **kwargs):
        super().__init__(name=name, **kwargs)

    def _build_residual_branch(self, filters, kernel_size, strides, prefix, subsample_cls, conv_cls, act_cls, kwargs):
        for i, (f, k, s) in enumerate(zip(filters, kernel_size, strides)):
            name_fn = lambda cls_name: prefix+f'{cls_name}_f{f}_k{k}_{i}'
            if s > 1 and self._subsample_type:
                self._layers.append(
                    subsample_cls(name=name_fn(self._subsample_type)), **self._subsample_kwargs)
                s = 1
            self._layers += [
                conv_cls(f, k, strides=s, padding='same', 
                        name=name_fn(self._conv), **kwargs),
                self._norm_cls(**self._norm_kwargs, name=name_fn(self._norm))]
            if i != len(filters)-1:
                self._layers.append(act_cls(name=name_fn(self._activation), **self._act_kwargs))

@block_registry.register('resv2')
class ResidualV2(ResidualBase):
    def __init__(self, name='resv2', **kwargs):
        super().__init__(name=name, **kwargs)

    def _build_residual_branch(self, filters, kernel_size, strides, prefix, subsample_cls, conv_cls, act_cls, kwargs):
        for i, (f, k, s) in enumerate(zip(filters, kernel_size, strides)):
            name_fn = lambda cls_name: prefix+f'{cls_name}_f{f}_k{k}_{i}'
            if s > 1 and self._subsample_type:
                self._layers.append(
                    subsample_cls(name=name_fn(self._subsample_type)), **self._subsample_kwargs)
                s = 1
            self._layers += [
                self._norm_cls(**self._norm_kwargs, name=name_fn(self._norm)),
                act_cls(name=name_fn(self._activation), **self._act_kwargs),
                conv_cls(f, k, strides=s, padding='same', 
                        name=name_fn(self._conv), **kwargs)]
        

strided_resv1 = functools.partial(ResidualV1, strides=2, name='strided_resv1')
strided_resv2 = functools.partial(ResidualV2, strides=2, name='strided_resv2')

subsample_registry.register('strided_resv1')(strided_resv1)
subsample_registry.register('strided_resv2')(strided_resv2)

if __name__ == '__main__':
    x = layers.Input((64, 64, 12))
    res = ResidualV1()
    y = res(x)
    model = tf.keras.Model(x, y)
    model.summary(200)