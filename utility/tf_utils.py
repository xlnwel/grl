import numpy as np
import tensorflow as tf


def upsample(x):
    h, w = x.get_shape().as_list()[1:-1]
    x = tf.image.resize_nearest_neighbor(x, [2 * h, 2 * w])
    return x

def safe_norm(m, axis=None, keepdims=None, epsilon=1e-6):
    """The gradient-safe version of tf.norm(...)
    it avoid nan gradient when m consists of zeros
    """
    squared_norms = tf.reduce_sum(m * m, axis=axis, keepdims=keepdims)
    return tf.sqrt(squared_norms + epsilon)

def standard_normalization(x):
    with tf.name_scope('Normalization'):
        n_dims = len(x.shape.as_list())
        mean, var = tf.nn.moments(x, list(range(n_dims-1)), keep_dims=True)
        std = tf.sqrt(var)

        x = (x - mean) / std
    
    return x

def explained_variance(y, pred):
    if None in y.shape:
        assert y.shape.ndims == pred.shape.ndims, (y.shape, pred.shape)
    else:
        assert y.shape == pred.shape, (y.shape, pred.shape)
    y_var = tf.math.reduce_variance(y, axis=0)
    diff_var = tf.math.reduce_variance(y - pred, axis=0)
    return tf.maximum(-1., 1-(diff_var / y_var))

def softmax(x, tau, axis=-1):
    """ sfotmax(x / tau) """
    x_max = tf.reduce_max(x, axis=axis, keepdims=True)
    x = x - x_max
    return tf.nn.softmax(x / tau, axis=axis)

def logsumexp(x, tau, axis=None, keepdims=False):
    """ reimplementation of tau * tf.logsumexp(x / tau), it turns out 
    that tf.logsumexp is numerical stable """
    x /= tau
    x_max = tf.reduce_max(x, axis=axis, keepdims=True)
    x = x - x_max    # for numerical stability
    if keepdims is False:
        x_max = tf.squeeze(x_max)
    y = x_max + tf.math.log(tf.reduce_sum(
        tf.exp(x), axis=axis, keepdims=keepdims))
    return tau * y

def log_softmax(x, tau, axis=-1):
    """ tau * log_softmax(x / tau) """
    x_max = tf.reduce_max(x, axis=axis, keepdims=True)
    x = x - x_max
    x = x - tau * tf.reduce_logsumexp(x / tau, axis=axis, keepdims=True)
    return x

def square_sum(x):
    return 2 * tf.nn.l2_loss(x)

def padding(x, kernel_size, strides, mode='constant', name=None):
    """ This function pads x so that a convolution with the same args downsamples x by a factor of strides.
    It achieves it using the following equation:
    W // S = (W - k_w + 2P) / S + 1
    """
    assert mode.lower() == 'constant' or mode.lower() == 'reflect' or mode.lower() == 'symmetric', \
        f'Padding should be "constant", "reflect", or "symmetric", but got {mode}.'
    H, W = x.shape.as_list()[1:3]
    if isinstance(kernel_size, list) and len(kernel_size) == 2:
        k_h, k_w = kernel_size
    else:
        k_h = k_w = kernel_size
    p_h1 = int(((H / strides - 1) * strides - H + k_h) // strides)
    p_h2 = int(((H / strides - 1) * strides - H + k_h) - p_h1)
    p_w1 = int(((W / strides - 1) * strides - W + k_w) // strides)
    p_w2 = int(((W / strides - 1) * strides - W + k_w) -p_w1)
    return tf.pad(x, [[0, 0], [p_h1, p_h2], [p_w1, p_w2], [0, 0]], mode, name=name)

def spectral_norm(w, u_var, iterations=1):
    w_shape = w.shape
    if len(w_shape) != 2:
        w = tf.reshape(w, [-1, w_shape[-1]])    # [N, M]

    u = u_var
    assert u.shape == [1, w_shape[-1]]
    # power iteration
    for i in range(iterations):
        v = tf.nn.l2_normalize(tf.matmul(u, w, transpose_b=True))           # [1, N]
        u = tf.nn.l2_normalize(tf.matmul(v, w))                             # [1, M]

    sigma = tf.squeeze(tf.matmul(tf.matmul(v, w), u, transpose_b=True))     # scalar
    w = w / sigma

    u_var.assign(u)
    w = tf.reshape(w, w_shape)

    return w

def positional_encoding(indices, max_idx, dim, name='positional_encoding'):
    with tf.name_scope(name):
        # exp(-2i / d_model * log(10000))
        vals = np.array([pos * np.exp(- np.arange(0, dim, 2) / dim * np.log(10000)) for pos in range(max_idx)])
        
        params = np.zeros((max_idx, dim))
        params[:, 0::2] = np.sin(vals)    # 2i
        params[:, 1::2] = np.cos(vals)    # 2i + 1
        params = tf.convert_to_tensor(params, tf.float32)

        v = tf.nn.embedding_lookup(params, indices)

    return v

def static_scan(fn, start, inputs, reverse=False):
    """ Sequentially apply fn to inputs, with starting state start.
    inputs are expected to be time-major, and the outputs of fn are expected
    to have the same structure as start. 
    This function is equivalent to 
    tf.scan(
        fn=fn
        elems=inputs, 
        initializer=start,
        parallel_iterations=1,
        reverse=reverse
    )
    In practice, we find it's faster than tf.scan
    """
    last = start
    outputs = [[] for _ in tf.nest.flatten(start)]
    indices = range(len(tf.nest.flatten(inputs)[0]))
    if reverse:
        indices = reversed(indices)
    for index in indices:
        # extract inputs at step index
        inp = tf.nest.map_structure(lambda x: x[index], inputs)
        last = fn(last, inp)
        # distribute outputs
        [o.append(l) for o, l in zip(outputs, tf.nest.flatten(last))]
    if reverse:
        outputs = [list(reversed(x)) for x in outputs]
    outputs = [tf.stack(x) for x in outputs]
    # reconstruct outputs to have the same structure as start
    return tf.nest.pack_sequence_as(start, outputs)

class TFRunningMeanStd:
    """ Different from PopArt, this is only for on-policy training, """
    def __init__(self, axis, shape=(), clip=None, epsilon=1e-2, dtype=tf.float32):
        # use tf.float64 to avoid overflow
        self._sum = tf.Variable(np.zeros(shape), trainable=False, dtype=tf.float64, name='sum')
        self._sumsq = tf.Variable(np.zeros(shape), trainable=False, dtype=tf.float64, name='sum_squares')
        self._count = tf.Variable(np.zeros(shape), trainable=False, dtype=tf.float64, name='count')
        self._mean = None
        self._std = None
        self._axis = axis
        self._clip = clip
        self._epsilon = epsilon
        self._dtype = dtype

    def update(self, x):
        x = tf.cast(x, tf.float64)
        self._sum.assign_add(tf.reduce_sum(x, axis=self._axis))
        self._sumsq.assign_add(tf.cast(tf.reduce_sum(x**2, axis=self._axis), self._sumsq.dtype))
        self._count.assign_add(tf.cast(tf.math.reduce_prod(tf.shape(x)[:len(self._axis)]), self._count.dtype))
        mean = self._sum / self._count
        std = tf.sqrt(tf.maximum(
            self._sumsq / self._count - mean**2, self._epsilon))
        self._mean = tf.cast(mean, self._dtype)
        self._std = tf.cast(std, self._dtype)

    def normalize(self, x, zero_center=True):
        if zero_center:
            x = x - self._mean
        x = x / self._std
        if self._clip is not None:
            x = tf.clip_by_value(x, -self._clip, self._clip)
        return x
    
    def denormalize(self, x, zero_center=True):
        x = x * self._std
        if zero_center:
            x = x + self._mean
        return x
        

def get_stoch_state(x, min_std):
    mean, std = tf.split(x, 2, -1)
    std = tf.nn.softplus(std) + min_std
    stoch = mean + tf.random.normal(tf.shape(mean)) * std
    return mean, std, stoch

def assert_rank(tensors, rank=None):
    if not isinstance(tensors, (list, tuple)):
        tensors = [tensors]

    rank = rank or tensors[0].shape.ndims
    for tensor in tensors:
        tensor_shape = tf.TensorShape(tensor.shape)
        tensor_shape.assert_has_rank(rank)

def assert_shape_compatibility(tensors):
    assert isinstance(tensors, (list, tuple)), tensors
    union_of_shapes = tf.TensorShape(None)
    for tensor in tensors:
        tensor_shape = tf.TensorShape(tensor.shape)
        union_of_shapes = union_of_shapes.merge_with(tensor_shape)

def assert_rank_and_shape_compatibility(tensors, rank=None):
    """Asserts that the tensors have the correct rank and compatible shapes.

    Shapes (of equal rank) are compatible if corresponding dimensions are all
    equal or unspecified. E.g. `[2, 3]` is compatible with all of `[2, 3]`,
    `[None, 3]`, `[2, None]` and `[None, None]`.

    Args:
        tensors: List of tensors.
        rank: A scalar specifying the rank that the tensors passed need to have.

    Raises:
        ValueError: If the list of tensors is empty or fail the rank and mutual
        compatibility asserts.
    """
    if not isinstance(tensors, (list, tuple)):
        tensors = [tensors]

    rank = rank or tensors[0].shape.ndims
    union_of_shapes = tf.TensorShape(None)
    for tensor in tensors:
        tensor_shape = tf.TensorShape(tensor.shape)
        tensor_shape.assert_has_rank(rank)
        union_of_shapes = union_of_shapes.merge_with(tensor_shape)
