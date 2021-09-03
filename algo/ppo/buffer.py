import time
import logging
import numpy as np

from core.decorator import config
from utility.utils import batch_dicts, moments, standardize
from replay.utils import init_buffer, print_buffer


logger = logging.getLogger(__name__)


def compute_nae(reward, discount, value, last_value, 
                gamma, mask=None, epsilon=1e-8):
    next_return = last_value
    traj_ret = np.zeros_like(reward)
    for i in reversed(range(reward.shape[1])):
        traj_ret[:, i] = next_return = (reward[:, i]
            + discount[:, i] * gamma * next_return)

    # Standardize traj_ret and advantages
    traj_ret_mean, traj_ret_var = moments(traj_ret)
    traj_ret_std = np.maximum(np.sqrt(traj_ret_var), 1e-8)
    value = standardize(value, mask=mask, epsilon=epsilon)
    # To have the same mean and std as trajectory return
    value = (value + traj_ret_mean) / traj_ret_std     
    advantage = standardize(traj_ret - value, mask=mask, epsilon=epsilon)
    traj_ret = standardize(traj_ret, mask=mask, epsilon=epsilon)

    return advantage, traj_ret

def compute_gae(reward, discount, value, last_value, gamma, 
                gae_discount, norm_adv=False, mask=None, epsilon=1e-8):
    if last_value is not None:
        last_value = np.expand_dims(last_value, 1)
        next_value = np.concatenate([value[:, 1:], last_value], axis=1)
    else:
        next_value = value[:, 1:]
        value = value[:, :-1]
    assert value.shape == next_value.shape, (value.shape, next_value.shape)
    advs = delta = (reward + discount * gamma * next_value - value)
    next_adv = 0
    for i in reversed(range(advs.shape[1])):
        advs[:, i] = next_adv = (delta[:, i] 
            + discount[:, i] * gae_discount * next_adv)
    traj_ret = advs + value
    if norm_adv:
        advs = standardize(advs, mask=mask, epsilon=epsilon)
    return advs, traj_ret

def compute_indices(idxes, mb_idx, mb_size, N_MBS):
    start = mb_idx * mb_size
    end = (mb_idx + 1) * mb_size
    mb_idx = (mb_idx + 1) % N_MBS
    curr_idxes = idxes[start: end]
    return mb_idx, curr_idxes

def reshape_to_store(memory, n_envs, n_steps, sample_size=None):
    start_dim = 2 if sample_size else 1
    memory = {k: v.reshape(n_envs, n_steps, *v.shape[start_dim:])
        for k, v in memory.items()}

    return memory

def reshape_to_sample(memory, n_envs, n_steps, sample_size=None):
    batch_size = n_envs * n_steps
    if sample_size is not None:
        batch_size //= sample_size
        leading_dims = (batch_size, -1)
    else:
        leading_dims = (batch_size,)
    memory = {k: v.reshape(*leading_dims, *v.shape[2:])
            for k, v in memory.items()}

    return memory


class Buffer:
    @config
    def __init__(self):
        self._add_attributes()

    def _add_attributes(self):
        self._use_dataset = getattr(self, '_use_dataset', False)
        if self._use_dataset:
            logger.info(f'Dataset is used for data pipline')

        self._sample_size = getattr(self, '_sample_size', None)
        self._state_keys = getattr(self, '_state_keys', [])
        if self._sample_size:
            assert self._n_envs * self.N_STEPS % self._sample_size == 0, \
                f'{self._n_envs} * {self.N_STEPS} % {self._sample_size} != 0'
            size = self._n_envs * self.N_STEPS // self._sample_size
            logger.info(f'Sample size: {self._sample_size}')
        else:
            size = self._n_envs * self.N_STEPS
        self._size = size
        self._mb_size = size // self.N_MBS
        self._idxes = np.arange(size)
        self._shuffled_idxes = np.arange(size)
        self._gae_discount = self._gamma * self._lam
        self._memory = {}
        self._is_store_shape = True
        self._inferred_sample_keys = False
        self._norm_adv = getattr(self, '_norm_adv', 'minibatch')
        self._epsilon = 1e-5
        if hasattr(self, 'N_VALUE_EPOCHS'):
            self.N_EPOCHS += self.N_VALUE_EPOCHS
        self.reset()
        logger.info(f'Batch size: {size}')
        logger.info(f'Mini-batch size: {self._mb_size}')

        self._sleep_time = 0.025
        self._sample_wait_time = 0
        self._epoch_idx = 0

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
        self.reshape_to_store()
        self._is_store_shape = True
        self._idx = 0
        self._mb_idx = 0
        self._epoch_idx = 0
        self._ready = False

    def add(self, **data):
        if self._memory == {}:
            self._init_buffer(data)
            
        for k, v in data.items():
            self._memory[k][:, self._idx] = v

        self._idx += 1

    def update(self, key, value, field='mb', mb_idxes=None):
        if field == 'mb':
            mb_idxes = self._curr_idxes if mb_idxes is None else mb_idxes
            self._memory[key][mb_idxes] = value
        elif field == 'all':
            assert self._memory[key].shape == value.shape, (self._memory[key].shape, value.shape)
            self._memory[key] = value
        else:
            raise ValueError(f'Unknown field: {field}. Valid fields: ("all", "mb")')

    def update_value_with_func(self, fn):
        assert self._mb_idx == 0, f'Unfinished sample: self._mb_idx({self._mb_idx}) != 0'
        mb_idx = 0

        for start in range(0, self._size, self._mb_size):
            end = start + self._mb_size
            curr_idxes = self._idxes[start:end]
            obs = self._memory['obs'][curr_idxes]
            if self._sample_size:
                state = tuple([self._memory[k][curr_idxes, 0] 
                    for k in self._state_keys])
                mask = self._memory['mask'][curr_idxes]
                value, state = fn(obs, state=state, mask=mask, return_state=True)
                self.update('value', value, mb_idxes=curr_idxes)
                next_idxes = curr_idxes + self._mb_size
                self.update('state', state, mb_idxes=next_idxes)
            else:
                value = fn(obs)
                self.update('value', value, mb_idxes=curr_idxes)
        
        assert mb_idx == 0, mb_idx

    def sample(self, sample_keys=None):
        if not self._ready:
            self._wait_to_sample()

        self._shuffle_indices()
        sample = self._sample(sample_keys)
        self._post_process_for_dataset()

        return sample

    def _wait_to_sample(self):
        while not self._ready:
            time.sleep(self._sleep_time)
            self._sample_wait_time += self._sleep_time

    def _shuffle_indices(self):
        if self.N_MBS > 1 and self._mb_idx == 0:
            np.random.shuffle(self._shuffled_idxes)
        
    def _sample(self, sample_keys=None):
        sample_keys = sample_keys or self._sample_keys
        self._mb_idx, self._curr_idxes = compute_indices(
            self._shuffled_idxes, self._mb_idx, 
            self._mb_size, self.N_MBS)

        sample = self._get_sample(sample_keys, self._curr_idxes)
        sample = self._process_sample(sample)

        return sample

    def _get_sample(self, sample_keys, idxes):
        return {k: self._memory[k][idxes, 0]
            if k in self._state_keys 
            else self._memory[k][idxes] 
            for k in sample_keys}
    
    def _process_sample(self, sample):
        if 'advantage' in sample and self._norm_adv == 'minibatch':
            sample['advantage'] = standardize(
                sample['advantage'], mask=sample['life_mask'], epsilon=self._epsilon)
        return sample
    
    def _post_process_for_dataset(self):
        if self._mb_idx == 0:
            self._epoch_idx += 1
            if self._epoch_idx == self.N_EPOCHS:
                # resetting here is especially important 
                # if we use tf.data as sampling is done 
                # in a background thread
                self.reset()

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

    def finish(self, last_value):
        assert self._idx == self.N_STEPS, self._idx
        self.reshape_to_store()
        self.compute_advantage_return_in_memory(last_value)
        self.reshape_to_sample()

    def clear(self):
        self._memory = {}
        self.reset()

    def reshape_to_store(self):
        if not self._is_store_shape:
            self._memory = reshape_to_store(
                self._memory, self._n_envs, 
                self.N_STEPS, self._sample_size)
            self._is_store_shape = True

    def reshape_to_sample(self):
        if self._is_store_shape:
            self._memory = reshape_to_sample(
                self._memory, self._n_envs, 
                self.N_STEPS, self._sample_size)
            self._is_store_shape = False
        self._ready = True

    def _init_buffer(self, data):
        init_buffer(self._memory, pre_dims=(self._n_envs, self.N_STEPS), **data)
        self._memory['traj_ret'] = np.zeros(
            (self._n_envs, self.N_STEPS), dtype=np.float32)
        self._memory['advantage'] = np.zeros(
            (self._n_envs, self.N_STEPS), dtype=np.float32)
        print_buffer(self._memory)
        if self._inferred_sample_keys or getattr(self, '_sample_keys', None) is None:
            self._sample_keys = set(self._memory.keys()) - set(('discount', 'reward'))
            self._inferred_sample_keys = True

    def compute_advantage_return_in_memory(self, last_value=None):
        if self._adv_type != 'vtrace' and last_value is None:
            last_value = self._memory.pop('last_value')
        if self._adv_type == 'nae':
            assert self._norm_adv == 'batch', self._norm_adv
            self._memory['advantage'], self._memory['traj_ret'] = \
                compute_nae(
                reward=self._memory['reward'], 
                discount=self._memory['discount'],
                value=self._memory['value'],
                last_value=last_value,
                gamma=self._gamma,
                mask=self._memory.get('life_mask'),
                epsilon=self._epsilon)
        elif self._adv_type == 'gae':
            self._memory['advantage'], self._memory['traj_ret'] = \
                compute_gae(
                reward=self._memory['reward'], 
                discount=self._memory['discount'],
                value=self._memory['value'],
                last_value=last_value,
                gamma=self._gamma,
                gae_discount=self._gae_discount,
                norm_adv=self._norm_adv == 'batch',
                mask=self._memory.get('life_mask'),
                epsilon=self._epsilon)
        elif self._adv_type == 'vtrace':
            pass
        else:
            raise NotImplementedError

    def replace_memory(self, data):
        self._memory = data
        self._is_store_shape = True
        for v in self._memory.values():
            assert v.shape[0] == self._n_envs, (v.shape, self._n_envs)

    def clear_memory(self):
        self._memory = None
