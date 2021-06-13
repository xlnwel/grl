from nn.registry import cnn_registry


def cnn(cnn_name, **kwargs):
    if cnn_name is None:
        return None
    cnn_name = cnn_name.lower()
    kwargs.setdefault('name', cnn_name)
    return cnn_registry.get(cnn_name)(**kwargs)


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