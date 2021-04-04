from core.module import Module
from nn.registry import am_registry, block_registry
from nn.utils import *
from nn.am.se import SE


@block_registry.register('mb')
class MobileBottleneck(Module):
    """ Mobile Inverted Residual Bottleneck. """
    def __init__(self, 
                 *,
                 name='mb', 
                 conv='conv2d',
                 expansion_ratio=1,
                 kernel_size=3,
                 strides=1,
                 out_filters=None,
                 norm=None,
                 norm_kwargs={},
                 activation='relu',
                 act_kwargs={},
                 am='se',
                 am_kwargs={},
                 dropout_rate=.2,   # increase this for large net, see EfficientNet: https://arxiv.org/abs/1905.11946
                 rezero=False,
                 **kwargs):
        super().__init__(name=name)
        self._conv = conv
        self._expansion_ratio = expansion_ratio
        self._kernel_size = kernel_size
        self._strides = strides
        self._out_filters = out_filters
        self._norm = norm
        self._norm_kwargs = norm_kwargs.copy()
        self._activation = activation
        self._act_kwargs = act_kwargs.copy()
        self._am = am
        self._am_kwargs = am_kwargs.copy()
        self._dropout_rate = dropout_rate
        self._use_rezero = rezero
        self._kwargs = kwargs

    def build(self, input_shape):
        kwargs = self._kwargs.copy()
        time_distributed = kwargs.get('time_distributed', False)
        am_kwargs = self._am_kwargs.copy()
        am_kwargs.update(kwargs)
        out_filters = self._out_filters or input_shape[-1]

        self._skip = self._strides == 1 and input_shape[-1] == out_filters

        self._layers = []
        self._norm_cls = get_norm(self._norm)
        act_cls = get_activation(self._activation, return_cls=True)

        filters = input_shape[-1] * self._expansion_ratio
        prefix = self.name
        if self._expansion_ratio != 1:
            self._layers += [
                layers.Conv2D(filters, 1, padding='same', name=prefix+'/expand_conv', **kwargs),
                self._norm_cls(**self._norm_kwargs, name=prefix+'/expand_bn'),
                act_cls(name=prefix+f'/expand_{self._activation}', **self._act_kwargs),
            ]

        self._layers += [
            layers.DepthwiseConv2D(self._kernel_size, self._strides, padding='same', 
                    use_bias=False, name=prefix+'/dwconv', **kwargs),
            self._norm_cls(**self._norm_kwargs, name=prefix+'/bn'),
            act_cls(name=prefix+f'/{self._activation}', **self._act_kwargs),
        ]

        am_cls = am_registry.get(self._am)
        self._layers.append(am_cls(name=f'{self.scope_name}/{self._am}', **am_kwargs))

        self._layers += [
            layers.Conv2D(out_filters, 1, padding='same', use_bias=False, 
                    name=prefix+'/project_conv', **kwargs),
            self._norm_cls(**self._norm_kwargs, name=prefix+'/project_bn')
        ]
        
        if self._skip:
            if self._dropout_rate != 0:
                noise_shape = (None, None, 1, 1, 1) if time_distributed else (None, 1, 1, 1)
                # Drop the entire residual branch with certain probability, https://arxiv.org/pdf/1603.09382.pdf
                self._layers.append(layers.Dropout(self._dropout_rate, noise_shape))
            if self._use_rezero:
                self._rezero = tf.Variable(0., trainable=True, dtype=tf.float32, name='rezero')

    def call(self, x, training=True):
        y = super().call(x)

        if self._skip:
            if self._use_rezero:
                y = self._rezero * y
            return x + y
        else:
            return y

if __name__ == "__main__":
    x = layers.Input(shape=(64, 64, 3))
    net = MobileBottleneck()
    y = net(x)
    m = tf.keras.Model(x, y)
    m.summary()