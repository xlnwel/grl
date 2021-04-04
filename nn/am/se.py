from tensorflow.keras import layers

from core.module import Module
from nn.registry import am_registry
from nn.utils import *


@am_registry.register('se')
class SE(Module):
    def __init__(self, 
                 ratio=1,
                 out_activation='sigmoid',
                 name='se', 
                 **kwargs):
        super().__init__(name=name)
        self._ratio = ratio   # the inverse of the reduction ratio
        self._out_activation = out_activation
        self._kwargs = kwargs

    def build(self, input_shape):
        kwargs = self._kwargs.copy()    # we cannot modify attribute of the layer in build, which will emit an error when save the model
        kernel_initializer = kwargs.get('kernel_initializer', 'glorot_uniform')
        filters = input_shape[-1]

        out_activation = get_activation(self._out_activation)
        squeeze = [
            layers.GlobalAvgPool2D(name=f'{self.scope_name}/squeeze'),
            layers.Reshape((1, 1, filters), name=f'{self.scope_name}/reshape'),
        ]
        reduced_filters = max(int(filters * self._ratio), 1)
        excitation = [
            layers.Dense(reduced_filters, 
                kernel_initializer=kernel_initializer, activation='relu',
                name=f'{self.scope_name}/reduce'),
            layers.Dense(filters, activation=out_activation,
                name=f'{self.scope_name}/expand')
        ]
        self._layers = squeeze + excitation
        
    def call(self, x, **kwargs):
        y = super().call(x, **kwargs)
        return x * y

if __name__ == "__main__":
    shape = (3, 3, 2)
    se = SE(2, name='scope/se')
    x = tf.keras.layers.Input(shape=shape)
    y = se(x)
    m = tf.keras.Model(x, y)
    m.summary()