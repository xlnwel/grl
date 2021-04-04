import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, activations, initializers

from nn.norm import EvoNorm

logger = logging.getLogger(__name__)


class Dummy:
    def __init__(self, **kwargs):
        pass

    def __call__(self, x, **kwargs):
        return x

def get_activation(act_name, return_cls=False, **kwargs):
    custom_activations = {
        None: Dummy,
        'relu': layers.ReLU,
        'leaky_relu': layers.LeakyReLU,
        'lrelu': layers.LeakyReLU,
        'hsigmoid': lambda name='hsigmoid': layers.Lambda(lambda x: tf.nn.relu6(x+3) / 6, name=name), # see MobileNet3
        'hswish': lambda name='hswish': layers.Lambda(lambda x: x * (tf.nn.relu6(x+3) / 6), name=name), # see MobileNet3
    }
    if isinstance(act_name, str):
        act_name = act_name.lower()
    if act_name in custom_activations:
        act_cls = custom_activations[act_name]
        if return_cls:
            return act_cls
        else:
            return act_cls(**kwargs)
    else:
        if return_cls:
            return lambda name: layers.Activation(act_name, name=name)
        else:
            return activations.get(act_name)


def get_norm(name):
    norm_layers = {
        None: Dummy,
        'layer': layers.LayerNormalization,
        'batch': layers.BatchNormalization,
        'evonorm': EvoNorm,
    }
    """ Return a normalization """
    if isinstance(name, str):
        name = name.lower()
    if name in norm_layers:
        return norm_layers[name]
    else:
        # assume name is an normalization layer class
        return name

def calculate_gain(name, param=None):
    """ a replica of torch.nn.init.calculate_gain """
    m = {
        None: 1, 
        'sigmoid': 1, 
        'tanh': 5./3., 
        'relu': np.sqrt(2.), 
        'leaky_relu': np.sqrt(2./(1+(param or 0)**2)),
        # the followings are I make up
        'elu': np.sqrt(2.),
        'relu6': np.sqrt(2.),
        'hswish': np.sqrt(2.), 
        'selu': np.sqrt(2.),
    }
    return m[name]

def constant_initializer(val):
    return initializers.Constant(val)

def get_initializer(name, **kwargs):
    """ 
    Return a kernel initializer by name
    """
    custom_inits = {
        # initializers for EfficientNet
        'en_conv': initializers.VarianceScaling(scale=2., mode='fan_out', distribution='untruncated_normal'),
        'en_dense': initializers.VarianceScaling(scale=1./2., mode='fan_out', distribution='uniform')
    }
    if isinstance(name, str):
        name = name.lower()
        if name in custom_inits:
            return custom_inits[name]
        elif name.lower() == 'orthogonal':
            gain = kwargs.get('gain', 1.)
            return initializers.orthogonal(gain)
        return initializers.get(name)
    else:
        return name

def ortho_init(scale=1.0):
    """ 
    A reproduction of tf...Orthogonal, originally from openAI baselines
    """
    def _ortho_init(shape, dtype, partition_info=None):
        #lasagne ortho init for tf
        shape = tuple(shape)
        if len(shape) == 2:
            flat_shape = shape
        elif len(shape) == 4: # assumes NHWC
            flat_shape = (np.prod(shape[:-1]), shape[-1])
        else:
            raise NotImplementedError
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v # pick the one with the correct shape
        q = q.reshape(shape)
        return (scale * q[:shape[0], :shape[1]]).astype(np.float32)
    return _ortho_init

def convert_obs(x, obs_range, dtype=tf.float32):
    if x.dtype != np.uint8:
        logger.info(f'Observations({x.shape}, {x.dtype}) are already converted to {x.dtype}, no further process is performed')
        return x
    dtype = dtype or tf.float32 # dtype is None when global policy is not unspecified, override it
    logger.info(f'Observations({x.shape}, {x.dtype}) are converted to range {obs_range} of dtype {dtype}')
    if obs_range == [0, 1]:
        return tf.cast(x, dtype) / 255.
    elif obs_range == [-.5, .5]:
        return tf.cast(x, dtype) / 255. - .5
    elif obs_range == [-1, 1]:
        return tf.cast(x, dtype) / 127.5 - 1.
    else:
        raise ValueError(obs_range)

# def flatten(x):
#     shape = tf.concat([tf.shape(x)[:-3], [tf.reduce_prod(x.shape[-3:])]], 0)
#     x = tf.reshape(x, shape)
#     return x

def call_norm(norm_type, norm_layer, x, training):
    if norm_type == 'batch':
        y = norm_layer(x, training=training)
    else:
        y = norm_layer(x)
    return y
