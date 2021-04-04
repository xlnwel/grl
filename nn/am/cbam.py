from tensorflow.keras import layers

from core.module import Module
from nn.registry import am_registry
from nn.utils import *


@am_registry.register('cbam')
class CBAM(Module):
    def __init__(self, 
                 ratio=1,
                 kernel_size=7,
                 excitation_type='2l',
                 ca_on=True,
                 sa_on=False,
                 out_activation='sigmoid',
                 name='cbam', 
                 **kwargs):
        super().__init__(name=name)
        self._ratio = ratio   # the inverse of the reduction ratio
        self._kernel_size = kernel_size
        self._ca_on = ca_on
        self._sa_on = sa_on
        self._out_activation = out_activation
        self._excitation_type = excitation_type
        self._kwargs = kwargs

    def build(self, input_shape):
        kwargs = self._kwargs.copy()    # we cannot modify attribute of the layer in build, which will emit an error when save the model
        kernel_initializer = kwargs.get('kernel_initializer', 'glorot_uniform')
        filters = input_shape[-1]

        if self._ca_on:
            self._c_avg, self._c_max, self._c_exc = \
                self._channel_attention(filters, kernel_initializer)
        if self._sa_on:
            self._s_avg, self._s_max, self._s_exc = \
                self._spatial_attention(filters, self._kernel_size, kernel_initializer)
    
    def _channel_attention(self, filters, kernel_initializer):
        name_fn = lambda name: f'{self.scope_name}/channel/{name}'
        avg_squeeze = layers.Lambda(
            lambda x: tf.reduce_mean(x, axis=[1, 2], keepdims=True), 
            name=name_fn('avg_squeeze'))
        max_squeeze = layers.Lambda(
            lambda x: tf.reduce_max(x, axis=[1, 2], keepdims=True), 
                name=name_fn('max_squeeze'))

        if self._excitation_type == '1l':
            excitation = [
                layers.Dense(filters,
                    activation=self._out_activation,
                    name=name_fn('excitation'),
                    use_bias=False)
            ]
        elif self._excitation_type == '2l':
            reduced_filters = max(int(filters * self._ratio), 1)
            excitation = [
                layers.Dense(reduced_filters, 
                    kernel_initializer=kernel_initializer, activation='relu',
                    name=name_fn('reduce'),
                    use_bias=False),
                layers.Dense(filters,
                    activation=self._out_activation,
                    name=name_fn('expand'),
                    use_bias=False)
            ]
        else:
            raise ValueError(f'Unkown excitation type: {self._exitation_type}')
        
        return avg_squeeze, max_squeeze, excitation
        
    def _spatial_attention(self, filters, kernel_size, kernel_initializer):
        name_fn = lambda name: f'{self.scope_name}/spatial/{name}'
        avg_squeeze = layers.Lambda(
            lambda x: tf.reduce_mean(x, axis=-1, keepdims=True), 
            name=name_fn('avg_squeeze'))
            
        max_squeeze = layers.Lambda(
            lambda x: tf.reduce_max(x, axis=-1, keepdims=True), 
                name=name_fn('max_squeeze'))

        excitation = layers.Conv2D(1, kernel_size, 
                            strides=1, padding='same',
                            kernel_initializer=kernel_initializer, 
                            activation=self._out_activation,
                            use_bias=False,
                            name=name_fn('excitation'))
        
        return avg_squeeze, max_squeeze, excitation

    def call(self, x, **kwargs):
        if self._ca_on:
            c_avg = self._c_avg(x)
            c_max = self._c_max(x)
            y = tf.concat([c_avg, c_max], -1)
            for l in self._c_exc:
                y = l(y)
            x = x * y
        
        if self._sa_on:
            s_avg = self._s_avg(x)
            s_max = self._s_max(x)
            y = tf.concat([s_avg, s_max], -1)
            y = self._s_exc(y)
            x = x * y
        
        return x

if __name__ == "__main__":
    for et in ['2l']:
        cbam = CBAM(2, name='scope/cbam', excitation_type=et)
        x = tf.keras.layers.Input(shape=(64, 64, 12))
        y = cbam(x)
        m = tf.keras.Model(x, y)
        m.summary()
        tf.random.set_seed(0)
        x = tf.ones((2, 64, 64, 12))
        print(cbam(x)[0, 0, 0])