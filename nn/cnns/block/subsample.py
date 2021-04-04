import functools
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from core.module import Module
from nn.registry import layer_registry, subsample_registry, block_registry, register_all
from nn.utils import get_norm, get_activation


class BlurPool(layers.Layer):
    def __init__(self,
                 filter_size=3,
                 strides=2,
                 pad_mode='REFLECT',
                 name='blurpool'):
        super().__init__(name=name)
        self.filter_size = filter_size
        self.paddings = np.zeros((4, 2), dtype=np.int32)
        self.paddings[1:-1] = int((filter_size - 1) / 2.)
        self.strides = self.get_strides(strides)
        self.pad_mode = pad_mode

    def build(self, input_shape):
        filters = input_shape[-1]
        filter_size = self.filter_size

        if(filter_size==1):
            a = np.array([1.,])
        elif(filter_size==2):
            a = np.array([1., 1.])
        elif(filter_size==3):
            a = np.array([1., 2., 1.])
        elif(filter_size==4):    
            a = np.array([1., 3., 3., 1.])
        elif(filter_size==5):    
            a = np.array([1., 4., 6., 4., 1.])
        elif(filter_size==6):    
            a = np.array([1., 5., 10., 10., 5., 1.])
        elif(filter_size==7):    
            a = np.array([1., 6., 15., 20., 15., 6., 1.])
        else:
            raise ValueError(f'Unkown filter size: {filter_size}')
        
        filt = a[:, None]*a[None]
        filt = filt / np.sum(filt)
        filt = filt[..., None, None]
        filt_shape = list(filt.shape)
        filt_shape[-2] = filters
        filt = np.broadcast_to(filt, filt_shape)
        self._filter = tf.convert_to_tensor(filt, self._compute_dtype)

    def call(self, x):
        x = tf.pad(x, self.paddings, mode=self.pad_mode)
        x = tf.nn.depthwise_conv2d(x, self._filter, self.strides, padding='VALID')
        return x

    def get_strides(self, strides):
        if isinstance(strides, int):
            return (1, strides, strides, 1)
        else: 
            assert isinstance(strides, (list, tuple)) and len(strides) == 2, strides
            return (1,) + tuple(strides) + (1,)


blurpool = functools.partial(BlurPool, strides=2, name='blurpool')


class ConvPool(Module):
    def __init__(self, 
                 filters=None, 
                 filter_size=3, 
                 strides=2,
                 conv=layers.Conv2D,
                 pool_type='max', 
                 pad_mode='REFLECT', # for blur pooling layer
                 norm=None,
                 norm_kwargs={},
                 activation=None,
                 act_kwargs={},
                 name='conv_maxpool',
                 **kwargs):
        super().__init__(name=name)
        self._filters = filters
        self._filter_size = filter_size
        self._conv = conv
        self._pool_type = pool_type
        self._pad_mode = pad_mode
        self._strides = strides
        self._norm = norm
        self._norm_kwargs = norm_kwargs
        self._activation = activation
        self._act_kwargs = act_kwargs
        self._kwargs = kwargs

    def build(self, input_shape):
        filters = self._filters or input_shape[-1]
        conv_name = f'{self.scope_name}/conv'
        norm_cls = get_norm(self._norm)
        norm_name = f'{self.scope_name}/{self._norm}'
        act_name = f'{self.scope_name}/{self._activation}'
        act_cls = get_activation(self._activation, return_cls=True)
        pool_name = f'{self.scope_name}/{self._pool_type}'

        self._layers = []
        if self._conv is not None:
            self._layers += [
                self._conv(filters, self._filter_size, strides=1, padding='same', 
                    name=conv_name, **self._kwargs),
                norm_cls(**self._norm_kwargs, name=norm_name),
                act_cls(name=act_name, **self._act_kwargs),
            ]
        if self._pool_type == 'max':
            self._layers += [
                layers.MaxPool2D(self._filter_size, strides=self._strides,
                    padding='same', name=pool_name)]
        elif self._pool_type == 'avg':
            self._layers += [
                layers.AvgPool2D(self._filter_size, strides=self._strides, 
                    padding='same', name=pool_name)]
        elif self._pool_type == 'maxblur':
            self._layers += [
                layers.MaxPool2D(self._filter_size, strides=1, padding='same', 
                    name=f'{self.scope_name}/max'),
                blurpool(self._filter_size, strides=self._strides, pad_mode=self._pad_mode,
                    name=f'{self.scope_name}/blur')]
        elif self._pool_type  == 'blur':
            self._layers +=[
                blurpool(self._filter_size, strides=self._strides, pad_mode=self._pad_mode,
                    name=pool_name)]
        else:
            raise ValueError(f'Unkonwn pool type: {self._pool_type}')

maxblurpool = functools.partial(ConvPool, conv=None, pool_type='maxblur', name='maxblurpool')
conv_maxpool = functools.partial(ConvPool, pool_type='max', name='conv_maxpool')
conv_avgpool = functools.partial(ConvPool, pool_type='avg', name='conv_avgpool')
conv_maxblurpool = functools.partial(ConvPool, pool_type='maxblur', name='conv_maxblurpool')
conv_blurpool = functools.partial(ConvPool, pool_type='blur', name='conv_blurpool')


@subsample_registry.register('strided_conv')
class StridedConv(Module):
    def __init__(self, 
                 filters=None, 
                 filter_size=3, 
                 conv=layers.Conv2D,
                 strides=2,
                 norm=None,
                 norm_kwargs={},
                 name='strided_conv', 
                 activation='relu',
                 **kwargs):
        super().__init__(name=name)
        self._filters = filters
        self._filter_size = filter_size
        self._conv = conv
        self._strides = strides
        self._norm = norm
        self._norm_kwargs = norm_kwargs
        self._activation = activation
        self._kwargs = kwargs

    def build(self, input_shape):
        filters = self._filters or input_shape[-1]
        conv_name = f'{self.scope_name}/conv_s{self._strides}'
        norm_cls = get_norm(self._norm)
        norm_name = f'{self.scope_name}/{self._norm}'
        kwargs = self._kwargs.copy()
        kwargs.update(dict(
            strides=self._strides,
            padding='same',
        ))

        with self.name_scope:
            self._layers = [
                self._conv(filters, self._filter_size, 
                    name=conv_name, **kwargs),
                norm_cls(**self._norm_kwargs, name=norm_name),
                get_activation(self._activation),
            ]

register_all(subsample_registry, globals())


if __name__ == '__main__':
    def f(x, **kwargs):
            l = maxblurpool(**kwargs)
            inp = tf.convert_to_tensor(x[None], tf.float32)
            blur_img = l(inp)
            blur_img = blur_img.numpy().astype(np.uint8)[0]
            return blur_img
    shape = (64, 64, 12)
    x = tf.random.normal(shape)
    img = f(x, filters=12)
    assert img.shape[-3:] == (32, 32, 12), img.shape
    import gym
    import matplotlib.pyplot as plt
    env = gym.make('BreakoutNoFrameskip-v4')
    x = env.reset()
    # blur_img = f(x, strides=1, filters=3)
    # blur_img2 = f(x, strides=2, filters=3)

    # fig = plt.figure()
    # ax = fig.add_subplot(131)
    # ax.imshow(x)
    # ax2 = fig.add_subplot(132)
    # ax2.imshow(blur_img)
    # ax3 = fig.add_subplot(133)
    # ax3.imshow(blur_img2)
    # print(blur_img.shape, blur_img2.shape)
    # plt.show()

    subsamples = subsample_registry.get_all()
    for name, subsample_cls in subsamples.items():
        if name:
            print(name, subsample_cls)
            x = tf.keras.layers.Input((4, 4, 2))
            l = subsample_cls()
            y = l(x)
            model = tf.keras.Model(x, y)
            model.summary()
