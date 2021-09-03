import time
import logging
import collections
import numpy as np

from core.decorator import config
from algo.ppo.buffer import compute_indices
from utility.utils import standardize

logger = logging.getLogger(__name__)


def compute_gae(reward, discount, value, last_value, gamma, 
                gae_discount, norm_adv=False, mask=None, epsilon=1e-8):
    if last_value is not None:
        last_value = np.expand_dims(last_value, 0)
        next_value = np.concatenate([value[1:], last_value], axis=0)
    else:
        next_value = value[1:]
        value = value[:-1]
    assert value.shape == next_value.shape, (value.shape, next_value.shape)
    advs = delta = (reward + discount * gamma * next_value - value)
    next_adv = 0
    for i in reversed(range(advs.shape[0])):
        advs[i] = next_adv = (delta[i] 
            + discount[i] * gae_discount * next_adv)
    traj_ret = advs + value
    if norm_adv:
        advs = standardize(advs, mask=mask, epsilon=epsilon)
    return advs, traj_ret


class Buffer:
    @config
    def __init__(self):
        self._add_attributes()

    def _add_attributes(self):
        self._size = self._batch_size * self._sample_size
        self._mb_size = self._batch_size // self.N_MBS
        self._mb_idx = 0
        self._idxes = np.arange(self._batch_size)
        self._shuffled_idxes = np.arange(self._batch_size)
        self._curr_idxes = np.arange(self._mb_size)
        self._gae_discount = self._gamma * self._lam
        self._buffer = [collections.defaultdict(list) for _ in range(self._n_envs)]
        self._memory = collections.defaultdict(list)
        
        self._is_store_shape = True
        self._norm_adv = getattr(self, '_norm_adv', 'minibatch')
        self._epsilon = 1e-5

        # sleep for dataset to have data
        self._sleep_time = 0.025
        self._sample_wait_time = 0
        self._epoch_idx = 0
        self._ready = False

        # debug stats
        self._buffer_size = [0 for _ in range(self._n_envs)]
        self._n_transitions = 0
        self._mem_len = 0
        self._invalid_episodes = set()
        self._batch_history = collections.deque(maxlen=10)
        self._curr_batch_size = 0

        self.reset()
    
    def n_samples(self):
        return np.mean(self._batch_history)
    
    @property
    def batch_size(self):
        return self._mb_size
    
    def __getitem__(self, k):
        return self._memory[k]
    
    def __contains__(self, k):
        return k in self._memory
    
    def ready(self):
        return self._ready
    
    def reset(self):
        for traj in self._buffer:
            traj.clear()
        
        self.clear_buffer()
        self._n_transitions = 0
        self._memory.clear()
        self._mem_len = 0
        self._ready = False
    
    def clear_buffer(self):
        [traj.clear() for traj in self._buffer]
        self._invalid_episodes.clear()
        self._buffer_size = [0 for _ in range(self._n_envs)]
        
    def add(self, i, **data):
        assert i < len(self._buffer), i
        assert i not in self._invalid_episodes, (i, self._invalid_episodes)

        if i in self._invalid_episodes:
            logger.warning('Adding data to invalid episodes...')
            return
        n_agents = next(iter(data.values())).shape[0]
        self._buffer_size[i] += n_agents
        self._n_transitions += n_agents
        for k, v in data.items():
            assert v.shape[0] == n_agents, (k, v.shape)
            self._buffer[i][k].append(v)
    
    def remove(self, i):
        if not self._buffer[i]:
            assert self._buffer_size[i] == 0, self._buffer_size[i]
            return
        self._n_transitions -= self._buffer_size[i]
        self._buffer[i].clear()
        self._buffer_size[i] = 0
        self._invalid_episodes.add(i)
    
    def finish(self, last_values):
        def compute_advantages_and_returns(buffer, last_values):
            for i, (traj, lv) in enumerate(zip(buffer, last_values)):
                if not traj:
                    assert self._buffer_size[i] == 0, (i, self._buffer_size)
                    assert i in self._invalid_episodes, (i, self._invalid_episodes)
                    continue
                assert i not in self._invalid_episodes, (i, self._invalid_episodes)
                traj['advantage'], traj['traj_ret'] = \
                    compute_gae(
                        reward=np.array(traj['reward']), 
                        discount=np.array(traj['discount']),
                        value=np.array(traj['value']),
                        last_value=lv,
                        gamma=self._gamma,
                        gae_discount=self._gae_discount,
                        norm_adv=self._norm_adv == 'batch',
                        epsilon=self._epsilon)
            return buffer
        
        def move_data_to_memory(buffer):
            n = self._n_transitions - self._mem_len
            if self._n_transitions > self._size:
                n -= self._n_transitions % self._sample_size
            self._mem_len += n
            for k in self._sample_keys:
                v = np.concatenate([traj[k] for traj in buffer if traj])    # merge env axis into sequential axis
                v = np.swapaxes(v, 0, 1)    # swap sequential and agent axes
                v = np.concatenate(v)[:n]
                self._memory[k].append(v)
                # debugging assert
                lens = [v.shape[0] for v in self._memory[k]]
                assert sum(lens) == self._mem_len, (lens, self._mem_len, n)

        def reshape_memory(memory):
            for k, v in memory.items():
                v = np.concatenate(v)
                assert v.shape[0] == self._mem_len, (v.shape, self._mem_len)
                memory[k] = v.reshape(-1, self._sample_size, *v.shape[1:])
            self._curr_batch_size = self._mem_len // self._sample_size
            self._batch_history.append(self._curr_batch_size)
        
        if self._n_transitions > self._mem_len:
            compute_advantages_and_returns(self._buffer, last_values)
            move_data_to_memory(self._buffer)
            self.clear_buffer()
            self._ready = self._n_transitions > self._size
            if self._ready:
                reshape_memory(self._memory)
            return self._ready
        else:
            return False

    def sample(self, sample_keys=None):
        def shuffule_indices():
            assert self._mem_len % self._sample_size == 0, (self._mem_len, self._sample_size)
            if self._mb_idx == 0:
                if self._shuffled_idxes.size != self._curr_batch_size:
                    self._shuffled_idxes = np.arange(self._curr_batch_size)
            np.random.shuffle(self._shuffled_idxes)
        
        def post_process_for_dataset():
            if self._mb_idx == 0:
                self._epoch_idx += 1
                if self._epoch_idx == self.N_EPOCHS:
                    self.reset()
        if not self._ready:
            self.wait_to_sample()
        
        shuffule_indices()
        sample = self._sample(sample_keys)
        post_process_for_dataset()

        return sample
    
    def _wait_to_sample(self):
        while not self._ready:
            time.sleep(self._sleep_time)
            self._sample_wait_time += self._sleep_time
    
    def _sample(self, sample_keys=None):
        def get_sample(sample_keys, idxes):
            return {
                k: self._memory[k][idxes, 0]
                if k in self._state_keys else self._memory[k][idxes]
                for k in sample_keys
            }
        
        def process_sample(sample):
            if self._norm_adv == 'minibatch':
                sample['advantage'] = standardize(
                    sample['advantage'], epsilon=self._epsilon
                )
            return sample
        
        self._mb_idx, self._curr_idxes = compute_indices(
            self._shuffled_idxes, self._mb_idx, self._mb_size, self.N_MBS
        )
        sample_keys = sample_keys or self._sample_keys
        sample = get_sample(sample_keys, self._curr_idxes)
        sample = process_sample(sample)

        return sample
    
    def compute_mean_max_std(self, name):
        stats = self._memory[name]
        return {
            name: np.mean(stats),
            f'{name}_max': np.max(stats),
            f'{name}_min': np.min(stats),
            f'{name}_std': np.std(stats),
        }
    
    def compute_fraction(self, name):
        stats = self._memory[name]
        return {
            f'{name}_frac': np.sum(stats) / np.prod(stats.shape)
        }
    
    def get(self, i, k):
        return np.array(self._buffer[i][k])
    
    def update_buffer(self, i, k, v):
        self._buffer[i][k] = v
    
    def is_valid_traj(self, i):
        return bool(self._buffer[i])
