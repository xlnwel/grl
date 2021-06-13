import argparse
import tensorflow as tf
from tensorflow.keras import layers

from nn.registry import *


def run_module(registry, name, keras_summary=True, shape=(64, 64, 12), **kwargs):
    if keras_summary:
        x = layers.Input(shape)
        block_cls = registry.get(name)
        block = block_cls(**kwargs)
        y = block(x)
        model = tf.keras.Model(x, y)
        model.summary(200)
    else:
        logdir = f'temp-logs/{name}'
        writer = tf.summary.create_file_writer(logdir)
        block_cls = registry.get(name)
        block = block_cls(**kwargs)
        @tf.function
        def fn(x):
            return block(x)
        tf.summary.trace_on(graph=True, profiler=True)
        y = fn(tf.random.normal((4,)+shape))
        with writer.as_default():
            tf.summary.trace_export(name=logdir, step=0, profiler_outdir=logdir)

def run_cnn(*, keras_summary=True, **new_kwargs):
    kwargs = {
        'cnn_name': 'impala',
        # 'obs_range': [0, 1],
        # 'filters': [32, 64, 64, 64],
        # 'n_blocks': [1, 1, 1, 1],
        # 'kernel_initializer': 'glorot_uniform',
        # 'stem': 'strided_resv1',
        # 'stem_kwargs': {
        #     # 'am': 'cbam',
        #     # 'am_kwargs': {
        #     #     'kernel_size': 1,
        #     #     'excitation_type': '2l',
        #     #     'sa_on': False,
        #     # }
        # },
        # 'subsample': 'strided_resv1',
        # 'subsample_kwargs': {
        #     # 'am': 'cbam',
        #     # 'am_kwargs': {
        #     #     'kernel_size': 1,
        #     #     'excitation_type': '2l',
        #     #     'sa_on': False,
        #     # }
        # },
        # 'block': 'resv1',
        # 'block_kwargs': {
        #     'conv': 'conv2d',
        #     'filter_coefs': [],
        #     'kernel_sizes': [3, 3],
        #     'norm': None,
        #     'norm_kwargs': {},
        #     'activation': 'relu',
        #     'am': 'cbam',
        #     # 'am_kwargs': {
        #     #     'kernel_size': 3,
        #     #     'excitation_type': '2l',
        #     #     'sa_on': False,
        #     #     'out_activation': 'sigmoid',
        #     # },
        #     'rezero': True,
        #     'dropout_rate': 0,
        # },
        # 'sa': 'conv_sa',
        # 'sa_kwargs': {
        #     'key_ratio': 8,
        #     'val_ratio': 2,
        #     'downsample_ratio': 2,
        # },
        # 'time_distributed': False,
        # 'out_activation': 'relu',
        # 'out_size': None,
    }
    from nn.cnn import cnn
    if keras_summary:
        kwargs.update(new_kwargs)
        shape = (64, 64, 3)
        x = layers.Input(shape)
        net = cnn(**kwargs)
        y = net(x)
        model = tf.keras.Model(x, y)
        model.summary(200)
    else:
        logdir = 'temp-logs'
        writer = tf.summary.create_file_writer(logdir)
        net = cnn(**kwargs)
        @tf.function
        def fn(x):
            return net(x)
        tf.summary.trace_on(graph=True, profiler=True)
        y = fn(tf.random.normal((4, 64, 64, 3)))
        with writer.as_default():
            tf.summary.trace_export(name=logdir, step=0, profiler_outdir=logdir)


def parse_args():
    parser = argparse.ArgumentParser()
    _, unknown = parser.parse_known_args()
    for arg in unknown:
        if arg.startswith(('-', '--')):
            name, val = arg.split('=')
            try:
                val = eval(val)
            except:
                pass
            parser.add_argument(name, type=type(val))
    args = parser.parse_args()
    return vars(args)


if __name__ == "__main__":
    kwargs = parse_args()
    run_cnn(keras_summary=True, **kwargs)
    # run_module(
    #     am_registry, 
    #     name='se', 
    #     keras_summary=True, 
    #     shape=(64, 64, 12),
    #     **kwargs)
    # import yaml
    # import os
    # path = os.path.abspath('algo/sacdiqn/procgen_config.yaml')
    # with open(path) as f:
    #     config = yaml.safe_load(f)
    # config = config['model']
    # kwargs = config['encoder']
    # for k, v in kwargs.items():
    #     print(k, v)
    # run_cnn(keras_summary=True, **kwargs)
