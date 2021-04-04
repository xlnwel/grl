import functools
from nn.utils import Dummy


class Registry:
    def __init__(self, name):
        self._mapping = {None: Dummy}
    
    def register(self, name):
        def _thunk(func):
            self._mapping[name] = func
            return func
        return _thunk
    
    def get(self, name):
        if isinstance(name, str) or name is None:
            return self._mapping[name]
        return name
    
    def contain(self, name):
        return name in self._mapping
    
    def get_all(self):
        return self._mapping


layer_registry = Registry(name='layer')
am_registry = Registry(name='am') # convolutional attention modules
block_registry = Registry(name='block')
subsample_registry = Registry(name='subsample')
cnn_registry = Registry(name='cnn')

def register_all(registry, globs):
    for k, v in globs.items():
        if isinstance(v, functools.partial):
            registry.register(k)(v)
