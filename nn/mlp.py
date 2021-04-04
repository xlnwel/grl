import logging

from core.module import Module
from nn.registry import layer_registry
from nn.utils import get_initializer


logger = logging.getLogger(__name__)

class MLP(Module):
    def __init__(self, units_list, out_size=None, layer_type='dense', 
                norm=None, activation=None, kernel_initializer='glorot_uniform', 
                name=None, out_dtype='float32', out_gain=1, **kwargs):
        super().__init__(name=name)
        layer_cls = layer_registry.get(layer_type)
        Layer = layer_registry.get('layer')
        logger.debug(f'{self.name} gain: {kwargs.get("gain", None)}')

        self._layers = [
            Layer(u, layer_type=layer_cls, norm=norm, 
                activation=activation, kernel_initializer=kernel_initializer, 
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