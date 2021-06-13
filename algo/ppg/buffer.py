import functools
import logging
import numpy as np

from utility.utils import config_attr
from algo.ppo.buffer import Buffer as PPOBuffer, reshape_to_sample, reshape_to_store, compute_indices


logger = logging.getLogger(__name__)

class Buffer:
    def __init__(self, config, BufferBase=PPOBuffer):
        config_attr(self, config)
        self._buff = BufferBase(config)
        self._add_attributes()

    def _add_attributes(self):
        assert self.N_PI >= self.N_SEGS, (self.N_PI, self.N_SEGS)
        self.TOTAL_STEPS = self.N_STEPS * self.N_PI
        buff_size = self._n_envs * self.N_STEPS
        self._size = buff_size * self.N_SEGS
        self._mb_size = buff_size // self.N_AUX_MBS_PER_SEG
        self._n_aux_mbs = self._size / self._mb_size
        self._shuffled_idxes = np.arange(self._size)
        self._idx = 0
        self._mb_idx = 0
        self._ready = False

        self._gae_discount = self._gamma * self._lam
        self._memory = {}
        self._is_store_shape = True
        logger.info(f'Memory size: {self._size}')
        logger.info(f'Aux mini-batch size: {self._mb_size}')

    def __getitem__(self, k):
        return self._buff[k]

    def ready(self):
        return self._ready
    
    def set_ready(self):
        self._ready = True

    def add(self, **data):
        self._buff.add(**data)

    def update(self, key, value, field='mb', mb_idx=None):
        self._buff.update(key, value, field, mb_idx)
    
    def aux_update(self, key, value, field='mb', mb_idxes=None):
        if field == 'mb':
            mb_idxes = self._curr_idxes if mb_idxes is None else mb_idxes
            self._memory[key][mb_idxes] = value
        elif field == 'all':
            assert self._memory[key].shape == value.shape, (self._memory[key].shape, value.shape)
            self._memory[key] = value
        else:
            raise ValueError(f'Unknown field: {field}. Valid fields: ("all", "mb")')

    def update_value_with_func(self, fn):
        self.buff.update_value_with_func(fn)
    
    def compute_aux_data_with_func(self, fn):
        assert self._idx == 0, self._idx
        value_list = []
        logits_list = []
        self.reshape_to_sample()
        for start in range(0, self._size, self._mb_size):
            end = start + self._mb_size
            obs = self._memory['obs'][start:end]
            logits, value = fn(obs)
            value_list.append(value)
            logits_list.append(logits)
        self._memory['value'] = np.concatenate(value_list, axis=0)
        self._memory['logits'] = np.concatenate(logits_list, axis=0)
    
    def transfer_data(self):
        assert self._buff.ready()
        self._buff.reshape_to_store()
        if self._idx >= self.N_PI - self.N_SEGS:
            for k in self._transfer_keys:
                v = self._buff[k]
                if k in self._memory:
                    # NOTE: we concatenate segments along the sequential dimension,
                    # which increases the horizon when computing targets and advantages
                    # The effect is unclear and may correlate with the value of 𝝀
                    self._memory[k] = np.concatenate(
                        [self._memory[k], v], axis=1
                    )
                else:
                    self._memory[k] = v.copy()
        self._idx = (self._idx + 1) % self.N_PI

    def sample(self):
        return self._buff.sample()
    
    def sample_aux_data(self):
        assert self._ready, self._idx
        if self._mb_idx == 0:
            np.random.shuffle(self._shuffled_idxes)
        self._mb_idx, self._curr_idxes = compute_indices(
            self._shuffled_idxes, self._mb_idx, self._mb_size, self.N_SEGS)
        return {k: self._memory[k][self._curr_idxes] for k in self._aux_sample_keys}

    def compute_mean_max_std(self, stats='reward'):
        return self._buff.compute_mean_max_std(stats)
    
    def finish(self, last_value):
        self._buff.finish(last_value)
    
    def aux_finish(self, last_value):
        assert self._idx == 0, self._idx
        self.reshape_to_store()
        _, self._memory['traj_ret'] = \
            self._buff._compute_advantage_return(
                self._memory['reward'], self._memory['discount'], 
                self._memory['value'], last_value
            )
        
        self.reshape_to_sample()
        self._ready = True

    def reset(self):
        if self._buff.ready():
            self.transfer_data()
        self._buff.reset()
    
    def aux_reset(self):
        assert self._ready, self._idx
        self._memory.clear()
        self._is_store_shape = True
        self._idx = 0
        self._mb_idx = 0
        self._ready = False

    def reshape_to_store(self):
        if not self._is_store_shape:
            self._memory = reshape_to_store(self._memory, self._n_envs, self.TOTAL_STEPS)
            self._is_store_shape = True

    def reshape_to_sample(self):
        if self._is_store_shape:
            self._memory = reshape_to_sample(self._memory, self._n_envs, self.TOTAL_STEPS)
            self._is_store_shape = False

    def clear(self):
        self._idx = 0
        self._ready = False
        self._buff.clear()
        self._memory.clear()
