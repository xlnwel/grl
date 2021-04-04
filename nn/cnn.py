"""
This file defines general CNN architectures used in RL
"""
import os, glob
import types
import importlib
from nn.registry import cnn_registry


def cnn(cnn_name, **kwargs):
    if cnn_name is None:
        return None
    cnn_name = cnn_name.lower()
    kwargs.setdefault('name', cnn_name)
    return cnn_registry.get(cnn_name)(**kwargs)


# TODO: move these to utility
def source_file(_file_path):
    """
    Dynamically "sources" a provided file
    """
    basename = os.path.basename(_file_path)
    filename = basename.replace(".py", "")
    # Load the module
    loader = importlib.machinery.SourceFileLoader(filename, _file_path)
    mod = types.ModuleType(loader.name)
    loader.exec_module(mod)


def load_files(path="."):
    """
    This function takes a path to a local directory
    and imports all the available files in there.
    """
    # for _file_path in glob.glob(os.path.join(local_dir, "*.py")):
    #     source_file(_file_path)
    for f in glob.glob(f'{path}/*'):
        if os.path.isdir(f):
            load_files(f)
        elif f.endswith('.py') and f != os.path.realpath(__file__):
            source_file(f)


def load_nn():
    load_files(os.path.dirname(os.path.realpath(__file__)))
    
load_nn()

if __name__ == '__main__':
    import tensorflow as tf
    kwargs = {
        'cnn_name': 'impala',
        'obs_range': [0, 1],
        'filters': [16, 32, 32, 32],
        'kernel_initializer': 'glorot_uniform',
        'block': 'resv1',
        'block_kwargs': {
            'conv': 'snconv2d',
            'filter_coefs': [],
            'kernel_sizes': [3, 3],
            'norm': 'batch',
            'norm_kwargs': {},
            'activation': 'lrelu',
            'am': 'cbam',
            'am_kwargs': {
                'ratio': 1,
                'sa_on': False,
                'out_activation': 'sigmoid',
            },
            'rezero': False,
            'dropout_rate': 0.9,
        },
        'time_distributed': False,
        'out_activation': 'relu',
        'out_size': None,
        'subsample_type': 'conv_maxblurpool',
        'subsample_kwargs': {
            'norm': None,
            # 'activation': 'relu'
        },
    }
    inp = tf.keras.layers.Input((64, 64, 12))
    from nn.registry import block_registry
    res_cls = block_registry.get('resv2')
    res = res_cls(norm='batch', rezero=True, dropout_rate=.1)
    out = res(inp)
    model = tf.keras.Model(inp, out)
    model.summary(200)
    # net = cnn(**kwargs)
    # out = net(inp)
    # model = tf.keras.Model(inp, out)
    # model.summary(200)
    # logdir = 'temp-logs'
    # writer = tf.summary.create_file_writer(logdir)
    # net = cnn(**kwargs)
    # @tf.function
    # def fn(x):
    #     return net(x)
    # tf.summary.trace_on(graph=True, profiler=True)
    # y = fn(tf.random.normal((4, 64, 64, 12)))
    # with writer.as_default():
    #     tf.summary.trace_export(name=logdir, step=0, profiler_outdir=logdir)
    # kwargs = {
    #     'cnn_name': 'efficientnet',
    #     'obs_range': [0, 1],
    #     'kernel_initializer': 'glorot_uniform',
    #     'block_kwargs': {
    #         'expansion_ratios': [1, 6, 6, 6],
    #         'kernel_sizes': [3, 3, 3, 3],
    #         'strides': [2, 2, 2, 2],
    #         'out_filters': [16, 32, 32, 64],
    #         'num_repeats': [2, 2, 2, 1],
    #         'norm': 'batch',
    #         'norm_kwargs': {},
    #         'am_kwargs': {
    #             'ratio': 1,
    #             'out_activation': 'sigmoid',
    #         },
    #         'rezero': False,
    #         'dropout_rate': .2,
    #         'activation': 'relu'
    #     },
    #     'out_activation': 'relu',
    #     'out_size': None,
    #     'subsample_type': 'strided_mb',
    #     'subsample_kwargs': {
    #         # 'filters': 3,
    #         'norm': 'batch'
    #     },
    # }
    # inp = tf.keras.layers.Input((64, 64, 3))
    # net = cnn(**kwargs)
    # out = net(inp)
    # model = tf.keras.Model(inp, out)
    # model.summary(200)