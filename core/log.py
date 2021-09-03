import logging
import os, atexit, shutil
from collections import defaultdict
import numpy as np
import tensorflow as tf

from utility.utils import isscalar
from utility.display import pwc
from utility.graph import image_summary, video_summary
from utility import yaml_op

logger = logging.getLogger(__name__)


""" Logging """
def log(logger, writer, model_name, prefix, step, print_terminal_info=True, **kwargs):
    stats = dict(
        model_name=f'{model_name}',
        steps=step
    )
    stats.update(logger.get_stats(**kwargs))
    scalar_summary(writer, stats, prefix=prefix, step=step)
    logger.log_stats(stats, print_terminal_info=print_terminal_info)
    writer.flush()

def log_stats(logger, stats, print_terminal_info=True):
    [logger.log_tabular(k, v) for k, v in stats.items()]
    logger.dump_tabular(print_terminal_info=print_terminal_info)

def set_summary_step(step):
    tf.summary.experimental.set_step(step)

def scalar_summary(writer, stats, prefix=None, step=None):
    if step is not None:
        tf.summary.experimental.set_step(step)
    prefix = prefix or 'stats'
    with writer.as_default():
        for k, v in stats.items():
            if isinstance(v, str):
                continue
            if '/' not in k:
                k = f'{prefix}/{k}'
            # print(k, np.array(v).dtype)
            tf.summary.scalar(k, tf.reduce_mean(v), step=step)

def histogram_summary(writer, stats, prefix=None, step=None):
    if step is not None:
        tf.summary.experimental.set_step(step)
    prefix = prefix or 'stats'
    with writer.as_default():
        for k, v in stats.items():
            if isinstance(v, (str, int, float)):
                continue
            tf.summary.histogram(f'{prefix}/{k}', v, step=step)

def graph_summary(writer, sum_type, args, step=None):
    """ This function should only be called inside a tf.function """
    fn = {'image': image_summary, 'video': video_summary}[sum_type]
    if step is None:
        step = tf.summary.experimental.get_step()
    def inner(*args):
        tf.summary.experimental.set_step(step)
        with writer.as_default():
            fn(*args)
    return tf.numpy_function(inner, args, [])

def store(logger, **kwargs):
    logger.store(**kwargs)

def get_raw_item(logger, key):
    return logger.get_raw_item(key)

def get_item(logger, key, mean=True, std=False, min=False, max=False):
    return logger.get_item(key, mean=mean, std=std, min=min, max=max)

def get_raw_stats(logger):
    return logger.get_raw_stats()

def get_stats(logger, mean=True, std=False, min=False, max=False):
    return logger.get_stats(mean=mean, std=std, min=min, max=max)

def contains_stats(logger, key):
    return key in logger
    
def save_code(root_dir, model_name):
    """ Saves the code so that we can check the chagnes latter """
    dest_dir = f'{root_dir}/{model_name}/src'
    if os.path.isdir(dest_dir):
        shutil.rmtree(dest_dir)
    
    shutil.copytree('.', dest_dir, 
        ignore=shutil.ignore_patterns(
            '*logs*', 'data*', '*data*' '*/data/*', 
            '.*', '*pycache*', '*.md', '*test*',
            '*results*'))

def simplify_datatype(config):
    """ Converts ndarray to list, useful for saving config as a yaml file """
    for k, v in config.items():
        if isinstance(v, dict):
            config[k] = simplify_datatype(v)
        elif isinstance(v, tuple):
            config[k] = list(v)
        elif isinstance(v, np.ndarray):
            config[k] = v.tolist()
        else:
            config[k] = v
    return config

def save_config(root_dir, model_name, config):
    config = simplify_datatype(config)
    yaml_op.save_config(config, filename=f'{root_dir}/{model_name}/config.yaml')

""" Functions for setup logging """                
def setup_logger(root_dir, model_name):
    log_dir = root_dir and f'{root_dir}/{model_name}'
    # logger save stats in f'{root_dir}/{model_name}/logs/log.txt'
    logger = Logger(log_dir)
    return logger

def setup_tensorboard(root_dir, model_name):
    # writer for tensorboard summary
    # stats are saved in directory f'{root_dir}/{model_name}'
    writer = tf.summary.create_file_writer(
        f'{root_dir}/{model_name}', max_queue=1000, flush_millis=20000)
    writer.set_as_default()
    return writer

class Logger:
    def __init__(self, log_dir=None, log_file='log.txt'):
        """
        Initialize a Logger.

        Args:
            log_dir (string): A directory for saving results to. If 
                `None/False`, Logger only serves as a storage but doesn't
                write anything to the disk.

            log_file (string): Name for the tab-separated-value file 
                containing metrics logged throughout a training run. 
                Defaults to "log.txt". 
        """
        log_file = log_file if log_file.endswith('log.txt') \
            else log_file + '/log.txt'
        self._log_dir = log_dir
        if self._log_dir:
            path = os.path.join(self._log_dir, log_file)
            if os.path.exists(path) and os.stat(path).st_size != 0:
                i = 1
                name, suffix = path.rsplit('.', 1)
                while os.path.exists(name + f'{i}.' + suffix):
                    i += 1
                pwc(f'Warning: Log file "{path}" already exists!', 
                    f'Data will be logged to "{name + f"{i}." + suffix}" instead.',
                    color='magenta')
                path = name + f"{i}." + suffix
            if not os.path.isdir(self._log_dir):
                os.makedirs(self._log_dir)
            self._out_file = open(path, 'w')
            atexit.register(self._out_file.close)
            pwc(f'Logging data to "{self._out_file.name}"', color='green')
        else:
            self._out_file = None
            pwc(f'Log directory is not specified, '
                'no data will be logged to the disk',
                color='magenta')

        self._first_row=True
        self._log_headers = []
        self._log_current_row = {}
        self._store_dict = defaultdict(list)

    def __contains__(self, item):
        return self._store_dict[item] != []
        
    def store(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, tf.Tensor):
                v = v.numpy()
            if v is None:
                return
            elif isinstance(v, (list, tuple)):
                self._store_dict[k] += list(v)
            else:
                self._store_dict[k].append(v)

    """ All get functions below will remove the corresponding items from the store """
    def get_raw_item(self, key):
        if key in self._store_dict:
            v = self._store_dict[key]
            del self._store_dict[key]
            return {key: v}
        return None
        
    def get_item(self, key, mean=True, std=False, min=False, max=False):
        stats = {}
        if key not in self._store_dict:
            return stats
        v = self._store_dict[key]
        if isscalar(v):
            stats[key] = v
            return
        if mean:
            stats[f'{key}'] = np.mean(v).astype(np.float32)
        if std:
            stats[f'{key}_std'] = np.std(v).astype(np.float32)
        if min:
            stats[f'{key}_min'] = np.min(v).astype(np.float32)
        if max:
            stats[f'{key}_max'] = np.max(v).astype(np.float32)
        del self._store_dict[key]
        return stats

    def get_raw_stats(self):
        stats = self._store_dict.copy()
        self._store_dict.clear()
        return stats

    def get_stats(self, mean=True, std=False, min=False, max=False):
        stats = {}
        for k in sorted(self._store_dict):
            v = self._store_dict[k]
            k_std, k_min, k_max = std, min, max
            if k.startswith('train/'):
                k_std = k_min = k_max = True
            if isscalar(v):
                stats[k] = v
                continue
            if mean:
                stats[f'{k}'] = np.mean(v).astype(np.float32)
            if k_std:
                stats[f'{k}_std'] = np.std(v).astype(np.float32)
            if k_min:
                stats[f'{k}_min'] = np.min(v).astype(np.float32)
            if k_max:
                stats[f'{k}_max'] = np.max(v).astype(np.float32)
        self._store_dict.clear()
        return stats

    def get_count(self, name):
        return len(self._store_dict[name])

    def log_stats(self, stats, print_terminal_info=True):
        if not self._first_row and not set(stats).issubset(set(self._log_headers)):
            if self._first_row:
                logger.warning(f'All previous loggings are erased because stats does not match first row\n'
                    f'stats = {set(stats)}\nfirst row = {set(self._log_headers)}')
            self._out_file.seek(0)
            self._out_file.truncate()
            self._log_headers.clear()
            self._first_row = True
        [self.log_tabular(k, v) for k, v in stats.items()]
        self.dump_tabular(print_terminal_info=print_terminal_info)

    def _log_tabular(self, key, val):
        """
        Log a value of some diagnostic.

        Call this only once for each diagnostic quantity, each iteration.
        After using ``log_tabular`` to store values for each diagnostic,
        make sure to call ``dump_tabular`` to write them out to file and
        stdout (otherwise they will not get saved anywhere).
        """
        if self._first_row:
            self._log_headers.append(key)
        else:
            assert key in self._log_headers, \
                f"Trying to introduce a new key {key} " \
                "that you didn't include in the first iteration"
        assert key not in self._log_current_row, \
            f"You already set {key} this iteration. " \
            "Maybe you forgot to call dump_tabular()"
        self._log_current_row[key] = val
    
    def log_tabular(self, key, val=None, mean=True, std=False, min=False, max=False):
        """
        Log a value or possibly the mean/std/min/max values of a diagnostic.
        """
        if val is not None:
            self._log_tabular(key, val)
        else:
            v = np.asarray(self._store_dict[key])
            if mean:
                self._log_tabular(f'{key}_mean', np.mean(v))
            if std:
                self._log_tabular(f'{key}_std', np.std(v))
            if min:
                self._log_tabular(f'{key}_min', np.min(v))
            if max:
                self._log_tabular(f'{key}_max', np.max(v))
        self._store_dict[key] = []

    def dump_tabular(self, print_terminal_info=True):
        """
        Write all of the diagnostics from the current iteration.
        """
        vals = []
        key_lens = [len(key) for key in self._log_headers]
        max_key_len = max(15,max(key_lens))
        n_slashes = 22 + max_key_len
        if print_terminal_info:
            print("-"*n_slashes)
        for key in self._log_headers:
            val = self._log_current_row.get(key, "")
            # print(key, np.array(val).dtype)
            valstr = f"{val:8.3g}" if hasattr(val, "__float__") else val
            if print_terminal_info:
                print(f'| {key:>{max_key_len}s} | {valstr:>15s} |')
            vals.append(val)
        if print_terminal_info:
            print("-"*n_slashes)
        if self._out_file is not None:
            if self._first_row:
                self._out_file.write("\t".join(self._log_headers)+"\n")
            self._out_file.write("\t".join(map(str,vals))+"\n")
            self._out_file.flush()
        self._log_current_row.clear()
        self._store_dict.clear()
        self._first_row=False
