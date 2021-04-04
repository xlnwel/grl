import tensorflow as tf
from tensorflow.keras import layers

from core.module import Module
from nn.registry import layer_registry, block_registry
from nn.utils import get_norm, call_norm


@layer_registry.register('att')
class Attention(layers.Layer):
    def __init__(self,
                 name='attention',):
        super().__init__(name=name)

    def call(self, q, k, v, mask=None):
        # softmax(QK^T/)V
        dot_product = tf.matmul(q, k, transpose_b=True)
        if mask is not None:
            dot_product *= mask
        weights = tf.nn.softmax(dot_product)
        x = tf.matmul(weights, v)
        return x


@block_registry.register('mhsa')
@layer_registry.register('mhsa')
class MultiHeadSelfAttention(layers.Layer):
    def __init__(self,
                 key_size,
                 val_size,
                 num_heads,
                 scale_logits=True,
                 out_size=None,
                 pre_norm=False,
                 norm='layer',
                 norm_kwargs={},
                 drop_rate=0,
                 use_rezero=False,
                 name='sa',
                 **kwargs):
        super().__init__(name=name)
        self._key_size = key_size
        self._val_size = val_size
        self._num_heads = num_heads
        self._scale_logits = scale_logits
        self._out_size = out_size
        self._pre_norm = pre_norm
        self._norm = norm
        self._norm_kwargs = norm_kwargs
        self._drop_rate = drop_rate
        self._use_rezero = use_rezero
        kwargs.setdefault('use_bias', False)
        self._kwargs = kwargs

    def build(self, input_shape):
        assert len(input_shape) == 3, input_shape
        seqlen, out_size = input_shape[1:]
        qkv_size = 2 * self._key_size + self._val_size
        total_size = qkv_size * self._num_heads
        out_size = self._out_size or out_size

        prefix = f'{self.name}/'
        self._embed = layers.Dense(total_size, **self._kwargs, name=prefix+'embed')
        self._att = Attention(prefix+'att')

        self._group_heads = layers.Reshape((seqlen, self._num_heads, qkv_size), name=prefix+'group_heads')
        self._concat = layers.Reshape((seqlen, self._num_heads * self._val_size), name=prefix+'concat')
        self._out = layers.Dense(out_size, **self._kwargs, name=prefix+'out')
        if self._drop_rate > 0:
            self._drop = layers.Dropout(self._drop_rate, (None, None, 1), name=prefix+'drop')
        
        norm_cls = get_norm(self._norm)
        self._norm_layer = norm_cls(**self._norm_kwargs, name=prefix+self._norm)
        if self._use_rezero:
            self._rezero = tf.Variable(0., trainable=True, dtype=tf.float32, name=prefix+'rezero')
        
        super().build(input_shape)

    def call(self, x, training=False, mask=None):
        y = call_norm(self._norm, self._norm_layer, x, training) \
            if self._pre_norm else x
        qkv = self._embed(y)
        qkv = self._group_heads(qkv)                    # [B, N, F] -> [B, N, H, F/H]
        qkv = tf.transpose(qkv, [0, 2, 1, 3])           # [B, N, H, F/H] -> [B, H, N, F/H]

        q, k, v = tf.split(qkv, [self._key_size, self._key_size, self._val_size], -1)
        
        # softmax(QK^T/(d**2))V
        if self._scale_logits:
            q *= self._key_size ** -.5
        out = self._att(q, k, v, mask)
        # equivalence using einsum
        # dot_product = tf.einsum('bqhf,bkhf->bqhk', q, k)
        # if mask is not None:
        #     dot_product *= mask
        # weights = tf.nn.softmax(dot_product)
        # out = tf.einsum('bqhk,bkhn->bqhn', weights, v)

        # [B, H, N, V] -> [B, N, H, V]
        out = tf.transpose(out, [0, 2, 1, 3])
        # [B, N, H, V] -> [B, N, H * V]
        y = self._concat(out)
        y = self._out(y)

        if self._drop_rate > 0:
            y = self._drop(y, training=training)
        if self._use_rezero:
            y = self._rezero * y
        x = x + y
        x = x if self._pre_norm else \
            call_norm(self._norm, self._norm_layer, x, training)

        return x


@block_registry.register('conv_sa')
class ConvSelfAttention(Module):
    """ Convolutional Self-Attention Module, 
    following SAGAN: https://arxiv.org/abs/1805.08318
    """
    def __init__(self,
                 key_size=None,
                 val_size=None,
                 key_ratio=8,
                 val_ratio=2,
                 scale_logits=False,
                 conv='conv2d',
                 downsample_ratio=2,
                 out_size=None,
                 pre_norm=False,
                 norm=None,
                 norm_kwargs={},
                 drop_rate=0,
                 use_rezero=True,
                 name='conv_sa',
                 **kwargs):
        super().__init__(name=name)
        self._key_size = key_size
        self._val_size = val_size
        self._key_ratio = key_ratio
        self._val_ratio = val_ratio
        self._scale_logits = scale_logits
        self._conv = conv
        self._downsample_ratio = downsample_ratio
        self._out_size = out_size
        self._pre_norm = pre_norm
        self._norm = norm
        self._norm_kwargs = norm_kwargs
        self._drop_rate = drop_rate
        self._use_rezero = use_rezero
        kwargs.setdefault('use_bias', False)
        self._kwargs = kwargs
    
    def build(self, input_shape):
        H, W, C = input_shape[1:]
        q_seqlen = kv_seqlen = H * W
        
        key_size, val_size = self._compute_sizes(C)
        self._key_size, self._val_size = key_size, val_size
        
        conv_cls = layer_registry.get(self._conv)
        prefix = f'{self.scope_name}/'

        self._q_conv = conv_cls(key_size, 1, **self._kwargs, name=prefix+'q')
        self._k_conv = conv_cls(key_size, 1, **self._kwargs, name=prefix+'k')
        self._v_conv = conv_cls(val_size, 1, **self._kwargs, name=prefix+'v')

        if self._downsample_ratio > 1:
            self._k_downsample = layers.MaxPool2D(
                self._downsample_ratio, self._downsample_ratio, 
                padding='same', name=prefix+'k_pool')
            self._v_downsample = layers.MaxPool2D(
                self._downsample_ratio, self._downsample_ratio, 
                padding='same', name=prefix+'v_pool')
            kv_seqlen //= self._downsample_ratio**2

        self._q_reshape = layers.Reshape((q_seqlen, key_size), name=prefix+'q_reshape')
        self._k_reshape = layers.Reshape((kv_seqlen, key_size), name=prefix+'k_reshape')
        self._v_reshape = layers.Reshape((kv_seqlen, val_size), name=prefix+'v_reshape')

        self._att = Attention(prefix+'attention')
        self._o_reshape = layers.Reshape((H, W, val_size), name=prefix+'o_reshape')
        self._o_conv = conv_cls(C, 1, **self._kwargs, name=prefix+'o')

        norm_cls = get_norm(self._norm)
        self._norm_layer = norm_cls(**self._norm_kwargs, name=prefix+f'{self._norm}')

        if self._use_rezero:
            self._rezero = tf.Variable(0., trainable=True, dtype=tf.float32, name=prefix+'rezero')
        
        super().build(input_shape)

    def call(self, x, training=False):
        y = call_norm(self._norm, self._norm_layer, x, training) \
            if self._pre_norm else x
        q = self._q_conv(y)
        k = self._k_conv(y)
        v = self._v_conv(y)
        
        if self._downsample_ratio > 1:
            k = self._k_downsample(k)
            v = self._v_downsample(v)

        q = self._q_reshape(q)
        k = self._k_reshape(k)
        v = self._v_reshape(v)

        if self._scale_logits:
            q *= self._key_size ** -.5
        o = self._att(q, k, v)
        o = self._o_reshape(o)
        o = self._o_conv(o)

        if self._drop_rate > 0:
            y = self._drop(y, training=training)
        if self._use_rezero:
            o = self._rezero * o
        x = o + x
        x = x if self._pre_norm else \
            call_norm(self._norm, self._norm_layer, x, training)

        return x

    def _compute_sizes(self, C):
        if self._key_size is None or self._val_size is None:
            assert self._key_ratio is not None and self._val_ratio is not None
            key_size = C // self._key_ratio
            val_size = C // self._val_ratio
        else:
            key_size = self._key_size
            val_size = self._val_size
        return key_size, val_size


if __name__ == "__main__":
    shape = (3, 4, 4, 2)
    # x = layers.Input(shape)
    tf.random.set_seed(0)
    x = tf.random.normal(shape)
    sa = ConvSelfAttention()
    y = sa(x)
    # import time
    # start = time.time()
    # for _ in range(100):
    #     x = tf.random.normal(shape)
    #     y = sa(x)
    # print(time.time() - start)
    print(sa.variables)
    # model = tf.keras.Model(x, y)
    # model.summary(200)
