import logging
import tensorflow as tf

from core.module import Module
from nn.registry import layer_registry
from nn.utils import get_initializer


logger = logging.getLogger(__name__)


class MLP(Module):
    def __init__(self, units_list, out_size=None, layer_type='dense', 
                norm=None, activation=None, kernel_initializer='glorot_uniform', 
                name=None, out_dtype=None, out_gain=1, norm_after_activation=False, 
                norm_kwargs={}, **kwargs):
        super().__init__(name=name)
        layer_cls = layer_registry.get(layer_type)
        Layer = layer_registry.get('layer')
        logger.debug(f'{self.name} gain: {kwargs.get("gain", None)}')
        self._out_dtype = out_dtype
        if activation is None and (len(units_list) > 1 or (units_list and out_size)):
            logger.warning(f'MLP({name}) with units_list({units_list}) and out_size({out_size}) has no activation.')

        self._layers = [
            Layer(u, layer_type=layer_cls, norm=norm, 
                activation=activation, kernel_initializer=kernel_initializer, 
                norm_after_activation=norm_after_activation, norm_kwargs=norm_kwargs,
                name=f'{name}/{layer_type}_{i}', **kwargs)
            for i, u in enumerate(units_list)]
        if out_size:
            kwargs.pop('gain', None)
            logger.debug(f'{self.name} out gain: {out_gain}')
            kernel_initializer = get_initializer(kernel_initializer, gain=out_gain)
            self._layers.append(layer_cls(
                out_size, kernel_initializer=kernel_initializer, 
                dtype=out_dtype, name=f'{name}/out', **kwargs))

    def reset(self):
        for l in self._layers:
            l.reset()

    def call(self, x, **kwargs):
        x = super().call(x, **kwargs)
        if self._out_dtype is not None \
                and self._out_dtype != tf.float32 \
                and self._out_dtype != 'float32':
            x = tf.cast(x, self._out_dtype)
        return x

if __name__ == '__main__':
    config = {
        'units_list': [64, 64, 64],
        'activation': 'relu',
        'norm': 'layer',
        'norm_after_activation': True,
        'norm_kwargs': {
            'epsilon': 1e-5
        }
    }
    from tensorflow.keras import layers
    x = layers.Input((64))
    net = MLP(**config)
    y = net(x)

    model = tf.keras.Model(x, y)
    model.summary(200)