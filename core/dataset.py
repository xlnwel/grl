import logging
import functools
import collections
import numpy as np
import tensorflow as tf


DataFormat = collections.namedtuple('DataFormat', ('shape', 'dtype'))
logger = logging.getLogger(__name__)


class Dataset:
    def __init__(self, 
                 buffer, 
                 data_format, 
                 process_fn=None, 
                 batch_size=False, 
                 print_data_format=True,
                 **kwargs):
        """ Create a tf.data.Dataset for data retrieval
        
        Args:
            buffer: buffer, a callable object that stores data
            data_format: dict, whose keys are keys of returned data
            values are tuple (type, shape) that passed to 
            tf.data.Dataset.from_generator
        """
        self._buffer = buffer
        assert isinstance(data_format, dict)
        data_format = {k: DataFormat(*v) for k, v in data_format.items()}
        self.data_format = data_format
        if print_data_format:
            logger.info('Dataset info:')
            for k, v in data_format.items():
                logger.info(f'\t{k} {v}')
        self.types = {k: v.dtype for k, v in self.data_format.items()}
        self.shapes = {k: v.shape for k, v in self.data_format.items()}
        self._iterator = self._prepare_dataset(process_fn, batch_size, **kwargs)

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError("attempted to get missing private attribute '{}'".format(name))
        return getattr(self._buffer, name)

    def sample(self):
        return next(self._iterator)

    def update_priorities(self, priorities, indices):
        self._buffer.update_priorities(priorities, indices)

    def _prepare_dataset(self, process_fn, batch_size, **kwargs):
        with tf.name_scope('data'):
            ds = tf.data.Dataset.from_generator(
                self._sample, self.types, self.shapes)
            # batch data if the data has not been batched yet
            if batch_size:
                ds = ds.batch(batch_size, drop_remainder=True)
            # apply processing function to a batch of data
            if process_fn:
                ds = ds.map(map_func=process_fn, 
                            num_parallel_calls=tf.data.experimental.AUTOTUNE)
            prefetch = kwargs.get('prefetch', tf.data.experimental.AUTOTUNE)
            ds = ds.prefetch(prefetch)
            iterator = iter(ds)
        return iterator

    def _sample(self):
        while True:
            yield self._buffer.sample()


def process_with_env(data, env, obs_range=None, one_hot_action=True, dtype=tf.float32):
    with tf.device('cpu:0'):
        if env.obs_dtype == np.uint8 and obs_range is not None:
            if obs_range == [0, 1]:
                for k in data:
                    if 'obs' in k:
                        data[k] = tf.cast(data[k], dtype) / 255.
            elif obs_range == [-.5, .5]:
                for k in data:
                    if 'obs' in k:
                        data[k] = tf.cast(data[k], dtype) / 255. - .5
            else:
                raise ValueError(obs_range)
        if env.is_action_discrete and one_hot_action:
            for k in data:
                if k.endswith('action'):
                    data[k] = tf.one_hot(data[k], env.action_dim, dtype=dtype)
    return data


def create_dataset(replay, env, data_format=None, use_ray=False, one_hot_action=True):
    process = functools.partial(process_with_env, 
        env=env, one_hot_action=one_hot_action)
    if use_ray:
        from core.ray_dataset import RayDataset
        DatasetClass = RayDataset
    else:
        DatasetClass = Dataset
    dataset = DatasetClass(replay, data_format, process)
    return dataset
