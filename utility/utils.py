import os, random
import itertools
import collections
import ast
import os.path as osp
import math
import multiprocessing
import numpy as np


class AttrDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

def deep_update(source, target):
    for k, v in target.items():
        if isinstance(v, collections.abc.Mapping):
            assert k in source, f'{k} does not exist in {source}'
            assert isinstance(source[k], collections.abc.Mapping), \
                f'Inconsistent types: {type(v)} vs {type(source[k])}'
            source[k] = deep_update(source.get(k, {}), v)
        else:
            source[k] = v
    return source

def config_attr(obj, config):
    for k, v in config.items():
        if k.islower():
            k = f'_{k}'
        if isinstance(v, str):
            try:
                v = float(v)
            except:
                pass
        if isinstance(v, float) and v == int(v):
            v = int(v)
        setattr(obj, k, v)

def to_int(s):
    return int(float(s))
    
def isscalar(x):
    return isinstance(x, (int, float))
    
def step_str(step):
    if step < 1000:
        return f'{step}'
    elif step < 1000000:
        return f'{step/1000:.3g}k'
    else:
        return f'{step/1000000:.3g}m'

def expand_dims_match(x, target):
    """ Expands dimensions of x to match target,
    an efficient way of the following process 
        while len(x.shape) < len(target.shape):
            x = np.expand_dims(x, -1)
    """
    assert x.shape == target.shape[:x.ndim], (x.shape, target.shape)
    return x[(*[slice(None) for _ in x.shape], *(None,)*(target.ndim - x.ndim))]

def moments(x, axis=None, mask=None):
    if x.dtype == np.uint8:
        x = x.astype(np.int32)
    if mask is None:
        x_mean = np.mean(x, axis=axis)
        x2_mean = np.mean(x**2, axis=axis)
    else:
        if axis is None:
            axis = tuple(range(x.ndim))
        else:
            axis = (axis,) if isinstance(axis, int) else tuple(axis)
        assert mask.ndim == len(axis), (mask.shape, axis)
        # the following process is about 5x faster than np.nan*
        # expand mask to match the dimensionality of x
        mask = expand_dims_match(mask, x)
        # compute valid entries in x corresponding to True in mask
        n = np.sum(mask)
        for i in axis:
            if mask.shape[i] != 1:
                assert mask.shape[i] == x.shape[i], (
                    f'{i}th dimension of mask({mask.shape[i]}) does not match'
                    f'that of x({x.shape[i]})')
            else:
                n *= x.shape[i]
        # compute x_mean and x_std from entries in x corresponding to True in mask
        x_mask = x * mask
        x_mean = np.sum(x_mask, axis=axis) / n
        x2_mean = np.sum(x_mask**2, axis=axis) / n
    x_var = x2_mean - x_mean**2

    return x_mean, x_var
    
def standardize(x, mask=None, axis=None, epsilon=1e-8):
    if mask is not None:
        mask = expand_dims_match(mask, x)
    x_mean, x_var = moments(x, axis=axis, mask=mask)
    x_std = np.sqrt(x_var + epsilon)
    y = (x - x_mean) / x_std
    if mask is not None:
        y = np.where(mask == 1, y, x)
    return y

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ValueError('Boolean value expected.')

def eval_str(val):
    try:
        val = ast.literal_eval(val)
    except ValueError:
        pass
    return val

def is_main_process():
    return multiprocessing.current_process().name == 'MainProcess'

def set_global_seed(seed=42, tf=None):
    os.environ['PYTHONHASHSEED']=str(seed)
    random.seed(seed)
    np.random.seed(seed)
    if tf:
        tf.random.set_seed(seed)

def timeformat(t):
    return f'{t:.2e}'

def get_and_unpack(x):
    """
    This function is used to decompose a list of remote objects 
    that corresponds to a tuple of lists.

    For example:
    @ray.remote
    def f():
        return ['a', 'a'], ['b', 'b']

    get_and_unpack(ray.get([f.remote() for _ in range(2)]))
    >>> [['a', 'a', 'a', 'a'], ['b', 'b', 'b', 'b']]
    """
    list_of_lists = list(zip(*x))
    results = []
    for item_list in list_of_lists:
        tmp = []
        for item in item_list:
            tmp += item
        results.append(tmp)

    return results

def squarest_grid_size(n, more_on_width=True):
    """Calculates the size of the most squared grid for n.

    Calculates the largest integer divisor of n less than or equal to
    sqrt(n) and returns that as the width. The height is
    n / width.

    Args:
        n: The total number of images.
        more_on_width: If cannot fit in a square, put more cells on width
    Returns:
        A tuple of (height, width) for the image grid.
    """
    # the following code is useful for large n, but it is not compatible with tf.numpy_function
    # import sympy
    # divisors = sympy.divisors(n)
    # square_root = math.sqrt(n)
    # for d in divisors:
    #     if d > square_root:
    #         break

    square_root = math.ceil(np.sqrt(n))
    for d in range(square_root, n+1):
        if n // d * d == n:
            break
    h, w = int(n // d), d
    if not more_on_width:
        h, w = w, h

    return h, w

def check_make_dir(path):
    _, ext = osp.splitext(path)
    if ext: # if path is a file path, extract its directory path
        path, _ = osp.split(path)

    if not os.path.isdir(path):
        os.mkdir(path)

def zip_pad(*args):
    list_len = None
    for x in args:
        if isinstance(x, list) or isinstance(x, tuple):
            list_len = len(x)
            break
    assert list_len is not None
    new_args = []
    for i, x in enumerate(args):
        if not isinstance(x, list) and not isinstance(x, tuple):
            new_args.append([x] * list_len)
        else:
            new_args.append(x)

    return list(zip(*new_args))
    
def convert_indices(indices, *args):
    """ 
    convert 1d indices to a tuple of for ndarray index
    args specify the size of the first len(args) dimensions
    e.g.
    x = np.array([[['a0', 'b0'], ['c0', 'd0']],
                [['a1', 'b1'], ['c1', 'd1']]])
    print(x.shape)
    >>> (2, 2, 2)
    indices = np.random.randint(7, size=5)
    print(indices)
    >>> [6 6 0 3 1]
    indices = convert_shape(indices, *x.shape)
    print(indices)
    >>> (array([1, 1, 0, 0, 0]), array([1, 1, 0, 1, 0]), array([0, 0, 0, 1, 1]))
    print(x[indices])
    >>> array(['b0', 'c1', 'b1', 'a1', 'c0'])
    """
    res = []
    v = indices
    for i in range(1, len(args)):
        prod = np.prod(args[i:])
        res.append(v // prod)
        v = v % prod
    res.append(v)

    return tuple(res)

def infer_dtype(dtype, precision=None):
    if precision is None:
        return dtype
    elif np.issubdtype(dtype, np.floating):
        dtype = {16: np.float16, 32: np.float32, 64: np.float64}[precision]
    elif np.issubdtype(dtype, np.signedinteger):
        dtype = {16: np.int16, 32: np.int32, 64: np.int64}[precision]
    elif np.issubdtype(dtype, np.uint8):
        dtype = np.uint8
    elif dtype == np.bool:
        dtype = np.bool
    else:
        dtype = None
    return dtype

def convert_dtype(value, precision=32, dtype=None, **kwargs):
    value = np.array(value, copy=False, **kwargs)
    if dtype is None:
        dtype = infer_dtype(value.dtype, precision)
    return value.astype(dtype)

def flatten_dict(**kwargs):
    """ Flatten a dict of lists into a list of dicts
    For example
    f(lr=[1, 2], a=[10,3], b=dict(c=[2, 4], d=np.arange(3)))
    >>>
    [{'lr': 1, 'a': 10, 'b': {'c': 2, 'd': 0}},
    {'lr': 1, 'a': 10, 'b': {'c': 2, 'd': 1}},
    {'lr': 1, 'a': 10, 'b': {'c': 2, 'd': 2}},
    {'lr': 1, 'a': 10, 'b': {'c': 4, 'd': 0}},
    {'lr': 1, 'a': 10, 'b': {'c': 4, 'd': 1}},
    {'lr': 1, 'a': 10, 'b': {'c': 4, 'd': 2}},
    {'lr': 1, 'a': 3, 'b': {'c': 2, 'd': 0}},
    {'lr': 1, 'a': 3, 'b': {'c': 2, 'd': 1}},
    {'lr': 1, 'a': 3, 'b': {'c': 2, 'd': 2}},
    {'lr': 1, 'a': 3, 'b': {'c': 4, 'd': 0}},
    {'lr': 1, 'a': 3, 'b': {'c': 4, 'd': 1}},
    {'lr': 1, 'a': 3, 'b': {'c': 4, 'd': 2}},
    {'lr': 2, 'a': 10, 'b': {'c': 2, 'd': 0}},
    {'lr': 2, 'a': 10, 'b': {'c': 2, 'd': 1}},
    {'lr': 2, 'a': 10, 'b': {'c': 2, 'd': 2}},
    {'lr': 2, 'a': 10, 'b': {'c': 4, 'd': 0}},
    {'lr': 2, 'a': 10, 'b': {'c': 4, 'd': 1}},
    {'lr': 2, 'a': 10, 'b': {'c': 4, 'd': 2}},
    {'lr': 2, 'a': 3, 'b': {'c': 2, 'd': 0}},
    {'lr': 2, 'a': 3, 'b': {'c': 2, 'd': 1}},
    {'lr': 2, 'a': 3, 'b': {'c': 2, 'd': 2}},
    {'lr': 2, 'a': 3, 'b': {'c': 4, 'd': 0}},
    {'lr': 2, 'a': 3, 'b': {'c': 4, 'd': 1}},
    {'lr': 2, 'a': 3, 'b': {'c': 4, 'd': 2}}]
    """
    ks, vs = [], []
    for k, v in kwargs.items():
        ks.append(k)
        if isinstance(v, dict):
            vs.append(flatten_dict(**v))
        elif isinstance(v, (int, float)):
            vs.append([v])
        else:
            vs.append(v)

    result = []
    for k, v in itertools.product([ks], itertools.product(*vs)):
        result.append(dict(zip(k, v)))

    return result

def batch_dicts(x, func=np.stack):
    keys = x[0].keys()
    vals = [o.values() for o in x]
    vals = [func(v) for v in zip(*vals)]
    x = {k: v for k, v in zip(keys, vals)}
    return x


class Every:
    def __init__(self, period, start=0):
        self._period = period
        self._next = start
    
    def __call__(self, step):
        if step >= self._next:
            while step >= self._next:
                self._next += self._period
            return True
        return False

    def step(self):
        return self._next - self._period

class RunningMeanStd:
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, axis, epsilon=1e-8, clip=None, name=None, ndim=None):
        """ Computes running mean and std from data
        A reimplementation of RunningMeanStd from OpenAI's baselines

        Args:
            axis: axis along which we compute mean and std from incoming data. 
                If it's None, we only receive at a time a sample without batch dimension
            ndim: expected number of dimensions for the stats, useful for debugging
        """
        self.name = name

        if isinstance(axis, int):
            axis = (axis, )
        elif isinstance(axis, (tuple, list)):
            axis = tuple(axis)
        elif axis is None:
            pass
        else:
            raise ValueError(f'Invalid axis({axis}) of type({type(axis)})')

        if isinstance(axis, tuple):
            assert axis == tuple(range(len(axis))), \
                f'Axis should only specifies leading axes so that '\
                f'mean and var can be broadcasted automatically when normalizing. '\
                f'But receving axis = {axis}'
        self._axis = axis
        if self._axis is not None:
            self._shape_slice = np.s_[: max(self._axis)+1]
        self._mean = None
        self._var = None
        self._epsilon = epsilon
        self._count = epsilon
        self._clip = clip
        self._ndim = ndim # expected number of dimensions o

    @property
    def axis(self):
        return self._axis

    def get_stats(self):
        Stats = collections.namedtuple('RMS', 'mean var count')
        return Stats(self._mean, self._var, self._count)

    def update(self, x, mask=None):
        x = x.astype(np.float64)
        if self._axis is None:
            assert mask is None, mask
            batch_mean, batch_var, batch_count = x, np.zeros_like(x), 1
        else:
            batch_mean, batch_var = moments(x, self._axis, mask)
            batch_count = np.prod(x.shape[self._shape_slice]) \
                if mask is None else np.sum(mask)
        if batch_count > 0:
            if self._ndim is not None:
                assert batch_mean.ndim == self._ndim, (batch_mean.shape, self._ndim)
            self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        if self._count == self._epsilon:
            self._mean = np.zeros_like(batch_mean, 'float64')
            self._var = np.ones_like(batch_var, 'float64')
        assert self._mean.shape == batch_mean.shape
        assert self._var.shape == batch_var.shape

        delta = batch_mean - self._mean
        total_count = self._count + batch_count

        new_mean = self._mean + delta * batch_count / total_count
        # no minus one here to be consistent with np.std
        m_a = self._var * self._count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self._count * batch_count / total_count
        assert np.all(np.isfinite(M2)), f'M2: {M2}'
        new_var = M2 / total_count

        self._mean = new_mean
        self._var = new_var
        self._std = np.sqrt(self._var)
        self._count = total_count
        assert np.all(self._var > 0), self._var[self._var <= 0]

    def normalize(self, x, zero_center=True, mask=None):
        assert not np.isinf(np.std(x)), f'{np.min(x)}\t{np.max(x)}'
        assert self._var is not None, (self._mean, self._var, self._count)
        assert x.ndim == self._var.ndim + (0 if self._axis is None else len(self._axis)), \
            (x.shape, self._var.shape, self._axis)
        if mask is not None:
            assert mask.ndim == len(self._axis), (mask.shape, self._axis)
            old = x.copy()
        if zero_center:
            x -= self._mean
        x /= self._std
        if self._clip:
            x = np.clip(x, -self._clip, self._clip)
        if mask is not None:
            mask = expand_dims_match(mask, x)
            x = np.where(mask, x, old)
        x = x.astype(np.float32)
        return x

class TempStore:
    def __init__(self, get_fn, set_fn):
        self._get_fn = get_fn
        self._set_fn = set_fn

    def __enter__(self):
        self.state = self._get_fn()
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self._set_fn(self.state)
