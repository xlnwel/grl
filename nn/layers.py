import functools
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, constraints, regularizers

from core.module import Module
from nn.registry import layer_registry, register_all
from nn.utils import *
from utility import tf_utils


@layer_registry.register('layer')
class Layer(Module):
    def __init__(self, *args, layer_type=layers.Dense, norm=None, 
                activation=None, kernel_initializer='glorot_uniform', 
                name=None, norm_after_activation=False, 
                norm_kwargs={}, **kwargs):
        super().__init__(name=name)
        if isinstance(layer_type, str):
            layer_type = layer_registry.get(layer_type)

        gain = kwargs.pop('gain', calculate_gain(activation))
        kernel_initializer = get_initializer(kernel_initializer, gain=gain)

        self._layer = layer_type(
            *args, kernel_initializer=kernel_initializer, name=name, **kwargs)
        self._norm = norm
        self._norm_cls = get_norm(norm)
        if self._norm:
            self._norm_layer = self._norm_cls(**norm_kwargs, name=f'{self.scope_name}/norm')
        self._norm_after_activation = norm_after_activation
        self.activation = get_activation(activation)

    def call(self, x, training=True, **kwargs):
        x = self._layer(x, **kwargs)
        
        if self._norm is not None and not self._norm_after_activation:
            x = call_norm(self._norm, self._norm_layer, x, training=training)
        if self.activation is not None:
            x = self.activation(x)
        if self._norm is not None and self._norm_after_activation:
            x = call_norm(self._norm, self._norm_layer, x, training=training)

        return x
    
    def reset(self):
        # reset noisy layer
        if isinstance(self._layer, Noisy):
            self._layer.reset()

@layer_registry.register('noisy')
class Noisy(layers.Dense):
    def __init__(self, units, name=None, **kwargs):
        if 'noisy_sigma' in kwargs:
            self.noisy_sigma = kwargs['noisy_sigma']
            del kwargs['noisy_sigma']
        else:
            self.noisy_sigma = .5
        super().__init__(units, name=name, **kwargs)

    def build(self, input_shape):
        self.last_dim = input_shape[-1]
        self.noisy_w = self.add_weight(
            'noise_kernel',
            shape=(self.last_dim, self.units),
            initializer=get_initializer('glorot_normal'),
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True)
        if self.use_bias:
            self.noisy_b = self.add_weight(
                'noise_bias',
                shape=[self.units],
                initializer=tf.constant_initializer(self.noisy_sigma / np.sqrt(self.units)),
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True)
        else:
            self.bias = None
        self.eps_w_in = self.add_weight(
            'eps_w_in', 
            shape=(self.last_dim, 1),
            initializer=get_initializer('zeros'),
            trainable=False,
            dtype=self._compute_dtype)
        self.eps_w_out = self.add_weight(
            'eps_w_out', 
            shape=(1, self.units),
            initializer=get_initializer('zeros'),
            trainable=False,
            dtype=self._compute_dtype)
        self.eps_b = tf.reshape(self.eps_w_out, [self.units])
        super().build(input_shape)

    def noisy_layer(self, inputs):
        eps_w_in = tf.math.sign(self.eps_w_in) * tf.math.sqrt(tf.math.abs(self.eps_w_in))
        eps_w_out = tf.math.sign(self.eps_w_out) * tf.math.sqrt(tf.math.abs(self.eps_w_out))
        eps_w = tf.matmul(eps_w_in, eps_w_out)
        return tf.matmul(inputs, self.noisy_w * eps_w) + self.noisy_b * self.eps_b

    def call(self, inputs, reset=True, noisy=True):
        y = super().call(inputs)
        if noisy:
            if reset:
                self.reset()
            noise = self.noisy_layer(inputs)
            y = y + noise
        return y

    def det_step(self, inputs):
        return super().call(inputs)

    def reset(self):
        self.eps_w_in.assign(tf.random.truncated_normal(
            [self.last_dim, 1], 
            stddev=self.noisy_sigma, 
            dtype=self._compute_dtype))
        self.eps_w_out.assign(tf.random.truncated_normal(
            [1, self.units], 
            stddev=self.noisy_sigma,
            dtype=self._compute_dtype))


@layer_registry.register('glu')
class GLU(Module):
    def __init__(self, *args, layer_type=layers.Dense, activation='sigmoid',
                kernel_initializer='glorot_uniform', name=None, **kwargs):
        super().__init__(name=name)
        if isinstance(layer_type, str):
            layer_type = layer_registry.get(layer_type)

        gain = kwargs.pop('gain', 1)
        kernel_initializer = get_initializer(kernel_initializer, gain=gain)

        self._layer = layer_type(
            *args, kernel_initializer=kernel_initializer, name=name, **kwargs)
        self.activation = get_activation(activation)

    def call(self, x):
        x = self._layer(x)
        x, gate = tf.split(x, 2, axis=-1)
        gate = self.activation(gate)
        x = x * gate
        return x
    
    def reset(self):
        # reset noisy layer
        if isinstance(self._layer, Noisy):
            self._layer.reset()


@layer_registry.register('sndense')
class SNDense(layers.Layer):
    def __init__(self,
                 units,
                 activation=None,
                 use_bias=True,
                 iterations=1,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 name='sndense',
                 **kwargs):
        super().__init__(
            name=name,
            activity_regularizer=regularizers.get(activity_regularizer),
            **kwargs)

        self.units = units
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.iterations = iterations
        self.kernel_initializer = get_initializer(kernel_initializer)
        self.bias_initializer = get_initializer(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

    def build(self, input_shape):
        last_dim = input_shape[-1]

        self.kernel = self.add_weight(
            name='kernel',
            shape=(last_dim, self.units),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True,
            dtype=self.dtype)
        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(self.units,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True,
                dtype=self.dtype)
        else:
            self.bias = None
        
        self.u = self.add_weight(
            name='u',
            shape=(1, self.units),
            initializer='truncated_normal',
            trainable=False,
            dtype=self.dtype)
        
        super().build(input_shape)

    def call(self, x):
        w = tf_utils.spectral_norm(self.kernel, self.u, self.iterations)
        x = tf.matmul(x, w)
        if self.use_bias:
            x = tf.nn.bias_add(x, self.bias)
        return x


@layer_registry.register('snconv2d')
class SNConv2D(layers.Layer):
    def __init__(self,
                 filters,
                 kernel_size=3,
                 strides=1,
                 padding='valid',
                #  data_format='channels_last',
                #  dilation_rate=1,
                 activation=None,
                 use_bias=True,
                 iterations=1,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 name='snconv2d',
                 **kwargs):
        super().__init__(
            name=name,
            activity_regularizer=regularizers.get(activity_regularizer),
            **kwargs)

        self.filters = filters
        self.kernel_size = self.get_kernel_size(kernel_size)
        self.strides = self.get_strides(strides)
        self.padding = padding.upper()
        self.activation = get_activation(activation)
        self.use_bias = use_bias
        self.iterations = iterations
        self.kernel_initializer = get_initializer((kernel_initializer))
        self.bias_initializer = get_initializer(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
    
    def build(self, input_shape):
        c = input_shape[-1]

        self.kernel = self.add_weight(
            name='kernel',
            shape=self.kernel_size + (c, self.filters),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True,
            dtype=self.dtype)
        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(self.filters,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True,
                dtype=self.dtype)
        else:
            self.bias = None
        
        self.u = self.add_weight(
            name='u',
            shape=(1, self.filters),
            initializer='truncated_normal',
            trainable=False,
            dtype=self.dtype)
        
        super().build(input_shape)

    def call(self, x):
        w = tf_utils.spectral_norm(self.kernel, self.u, self.iterations)
        x = tf.nn.conv2d(x, w, strides=self.strides, padding=self.padding)
        if self.use_bias:
            x = tf.nn.bias_add(x, self.bias)
        return x

    def get_kernel_size(self, kernel_size):
        if isinstance(kernel_size, int):
            return (kernel_size, kernel_size)
        else:
            assert isinstance(kernel_size, (list, tuple)) and len(kernel_size) == 2, kernel_size
            return kernel_size

    def get_strides(self, strides):
        if isinstance(strides, int):
            return (1, strides, strides, 1)
        else: 
            assert isinstance(strides, (list, tuple)) and len(strides) == 2, strides
            return (1,) + tuple(strides) + (1,)

layer_registry.register('global_avgpool2d')(layers.GlobalAvgPool2D)
layer_registry.register('global_maxpool2d')(layers.GlobalMaxPool2D)
layer_registry.register('reshape')(layers.Reshape)
layer_registry.register('flatten')(layers.Flatten)
layer_registry.register('dense')(layers.Dense)
layer_registry.register('conv2d')(layers.Conv2D)
layer_registry.register('dwconv2d')(layers.DepthwiseConv2D)
layer_registry.register('depthwise_conv2d')(layers.DepthwiseConv2D)
layer_registry.register('maxpool2d')(layers.MaxPool2D)
layer_registry.register('avgpool2d')(layers.AvgPool2D)


if __name__ == '__main__':
    tf.random.set_seed(0)
    shape = (1, 2, 3)
    x = tf.random.normal(shape)
    # print('x', x[0, 0, :, 0])
    print(layer_registry.get_all())
    l = layer_registry.get('layer')(2, name='Layer')
    y = l(x)
    print(y)
    y = l(x)
    print(y)
    y = l(x)
    print(y)
    x = tf.random.normal(shape)
    y = l(x)
    print(y)