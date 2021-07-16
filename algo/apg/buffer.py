import time
import collections
from threading import Lock
import numpy as np

from core.decorator import config
from utility.utils import batch_dicts, to_array32
from replay.utils import *


class APGBuffer:
    @config
    def __init__(self, buffer, agent):
        self._buff = buffer
        self._buff.reshape_to_store = lambda: None
        self._agent = agent

        self._cache = []
        self._batch_idx = 0
        self._train_step = 0

        self._n_batch = self._n_trajs // self._n_envs   # #batch expected to received for training
        self._sample_wait_time = 0
        self._sleep_time = 0.025
        self._diag_stats = collections.defaultdict(list)

        # to avoid contention caused by multi-thread parallelism
        self._lock = Lock()

    @property
    def batch_size(self):
        return self._buff.batch_size

    def is_full(self):
        return self._batch_idx >= self._n_batch

    def set_train_step(self, train_step):
        self._train_step = train_step

    def merge(self, data):
        with self._lock:
            self._cache.append(data)
            self._batch_idx += 1

    def sample(self):
        if not self._buff.ready():
            self._diag_stats['sample_wait_time'].append(
                self._sample_wait_time)
            self._sample_wait_time = 0
            self._wait_to_sample()
        sample = self._buff.sample()
        return sample

    def _wait_to_sample(self):
        while not self.is_full():
            time.sleep(self._sleep_time)
            self._sample_wait_time += self._sleep_time
        
        n, train_step = self._fill_buffer()
        self._record_async_stats(n, train_step)
        self._update_agent_rms()

        self._buff.compute_advantage_return_in_memory()
        self._buff.reshape_to_sample()

    def _fill_buffer(self):
        # assert self._memory is None, self._memory
        with self._lock:
            data = self._cache[-self._n_batch:]
            self._cache = []
            n = self._batch_idx
            self._batch_idx = 0
        data = batch_dicts(data, np.concatenate)
        train_step = data.pop('train_step')
        self._buff.replace_memory(data)

        return n, train_step

    def _record_async_stats(self, n, train_step):
        self._diag_stats['trajs_dropped'].append(
            n * self._n_envs - self._n_trajs)
        self._diag_stats['policy_version_min_diff'].append(
            self._train_step - train_step[:, -1].max())
        self._diag_stats['policy_version_max_diff'].append(
            self._train_step - train_step[:, 0].min())
        self._diag_stats['policy_version_avg_diff'].append(
            self._train_step - train_step.mean())

    def _update_agent_rms(self):
        self._agent.update_obs_rms(np.concatenate(self._buff['obs']))
        self._agent.update_reward_rms(
            self._buff['reward'], self._buff['discount'])
        self._buff.update('reward', 
            self._agent.normalize_reward(self._buff['reward']), field='all')

    def get_async_stats(self):
        stats = self._diag_stats
        self._diag_stats = collections.defaultdict(list)
        return stats


class LocalBuffer:
    @config
    def __init__(self):
        self.reset()

    def is_full(self):
        return self._idx == self.N_STEPS

    def reset(self):
        self._idx = 0
        self._memory = collections.defaultdict(list)

    def add(self, **data):
        for k, v in data.items():
            self._memory[k].append(v)
        self._idx += 1

    def sample(self):
        data = {}

        # make data batch-major
        for k, v in self._memory.items():
            v = to_array32(v)
            data[k] = np.swapaxes(v, 0, 1) if v.ndim > 1 else v

        return data

    def finish(self, last_value=None,
            last_obs=None, last_mask=None):
        """ Add last value to memory. 
        Leave advantage and return computation to the learner 
        """
        assert self._idx == self.N_STEPS, self._idx
        if last_value is not None:
            self._memory['last_value'] = last_value
        if last_obs is not None:
            assert last_mask is not None, 'last_mask is required'
            self._memory['obs'].append(last_obs)
            self._memory['mask'].append(last_mask)
