from abc import ABC, abstractmethod
from datetime import datetime
import logging
from pathlib import Path
import numpy as np

from core.decorator import config
from replay.utils import *

logger = logging.getLogger(__name__)


class Replay(ABC):
    """ Interface """
    @config
    def __init__(self, **kwargs):
        # params for general replay buffer
        self._save = getattr(self, '_save', False)
        self._save_stats = getattr(self, '_save_stats', None)
        self._save_temp = getattr(self, '_save_temp', False)
        self._dir = getattr(self, '_dir', '')
        self._dir = Path(self._dir).expanduser()
        if self._save:
            self._dir.mkdir(parents=True, exist_ok=True)
        self._transition_per_file = getattr(self, '_transition_per_file',
            self._capacity // 10)

        self._precision = getattr(self, '_precision', 32)
        
        self._memory = {}
        self._mem_idx = 0
        self._min_size = max(self._min_size, self._batch_size*10)
        self._n_envs = getattr(self, '_n_envs', 1)
        self._is_full = False

        self._add_attributes(**kwargs)
        self._construct_temp_buff()

    def _add_attributes(self):
        self._pre_dims = (self._capacity, )

    def _construct_temp_buff(self):
        if self._n_envs > 1:
            from replay.func import create_local_buffer
            self._tmp_buf = create_local_buffer({
                'replay_type': self._replay_type,
                'n_envs': self._n_envs,
                'seqlen': self._seqlen,
                'n_steps': self._n_steps,
                'max_steps': getattr(self, '_max_steps', self._n_steps),
                'gamma': self._gamma,
            })

    def name(self):
        return self._replay_type

    def good_to_learn(self):
        return len(self) >= self._min_size
    
    def load_data(self):
        if self._memory == {}:
            for filename in self._dir.glob('*.npz'):
                data = load_data(filename)
                if data is not None:
                    self.merge(data)
        else:
            logger.warning(f'There are already {len(self)} transitions in the memory. No further loading is performed')

    def save(self):
        if self._save:
            if self._save_temp and hasattr(self, '_tmp_buf'):
                data = self._tmp_buf.retrieve()
                self.merge(data)
            size = len(self)
            tpf = self._transition_per_file

            for start in np.arange(0, size, tpf):
                end = min(start + tpf, size)
                timestamp = datetime.now().strftime('%Y%m%dT%H%M%S')
                filename = self._dir / f'{timestamp}-{start}-{end}.npz'
                data = {k: self._memory[k][start: end] for k in self._save_stats or self._memory.keys()}
                save_data(filename, data)
                print(f'{end-start} transitions are saved at {filename}')

    def __len__(self):
        return self._capacity if self._is_full else self._mem_idx
    
    def size(self):
        return len(self)

    def __call__(self):
        while True:
            yield self.sample()

    @abstractmethod
    def sample(self, batch_size=None):
        raise NotImplementedError

    def merge(self, local_buffer):
        """ Merge a local buffer to the replay buffer, 
        useful for distributed algorithms """
        length = len(next(iter(local_buffer.values())))
        assert length < self._capacity, (
            f'Local buffer cannot be largeer than the replay: {length} vs. {self._capacity}')
        self._merge(local_buffer, length)

    def add(self, **kwargs):
        if self._n_envs > 1:
            self._tmp_buf.add(**kwargs)
            if self._tmp_buf.is_full():
                data = self._tmp_buf.sample()
                self.merge(data)
                self._tmp_buf.reset()
        else:
            """ Add a single transition to the replay buffer """
            next_obs = kwargs['next_obs']
            if self._memory == {}:
                if not self._has_next_obs:
                    del kwargs['next_obs']
                init_buffer(self._memory, 
                            pre_dims=self._pre_dims, 
                            has_steps=self._n_steps>1, 
                            precision=self._precision,
                            **kwargs)
                print_buffer(self._memory)

            if not self._is_full and self._mem_idx == self._capacity - 1:
                self._is_full = True
            
            add_buffer(
                self._memory, self._mem_idx, self._n_steps, self._gamma, cycle=self._is_full, **kwargs)
            self._mem_idx = (self._mem_idx + 1) % self._capacity
            if 'next_obs' not in self._memory:
                self._memory['obs'][self._mem_idx] = next_obs

    """ Implementation """
    def _sample(self, batch_size=None):
        raise NotImplementedError

    def _merge(self, local_buffer, length):
        if self._memory == {}:
            if not self._has_next_obs and 'next_obs' in local_buffer:
                del local_buffer['next_obs']
            init_buffer(self._memory, 
                        pre_dims=self._pre_dims, 
                        has_steps=self._n_steps>1, 
                        precision=self._precision,
                        **local_buffer)
            print_buffer(self._memory)

        end_idx = self._mem_idx + length

        if end_idx > self._capacity:
            first_part = self._capacity - self._mem_idx
            second_part = length - first_part
            
            copy_buffer(self._memory, self._mem_idx, self._capacity, local_buffer, 0, first_part)
            copy_buffer(self._memory, 0, second_part, local_buffer, first_part, length)
        else:
            copy_buffer(self._memory, self._mem_idx, end_idx, local_buffer, 0, length)

        # memory is full, recycle buffer according to FIFO
        if not self._is_full and end_idx >= self._capacity:
            self._is_full = True
        
        self._mem_idx = end_idx % self._capacity

    def _get_samples(self, idxes):
        """ retrieve samples from replay memory """
        results = {}
        idxes = np.array(idxes, copy=False, dtype=np.int32)
        for k, v in self._memory.items():
            if isinstance(v, np.ndarray):
                results[k] = v[idxes]
            else:
                results[k] = np.array([np.array(v[i], copy=False) for i in idxes])
            
        if 'next_obs' not in self._memory:
            steps = results.get('steps', 1)
            next_idxes = (idxes + steps) % self._capacity
            if isinstance(self._memory['obs'], np.ndarray):
                results['next_obs'] = self._memory['obs'][next_idxes]
            else:
                results['next_obs'] = np.array(
                    [np.array(self._memory['obs'][i], copy=False) 
                    for i in next_idxes])
        
        if 'steps' in results:
            results['steps'] = results['steps'].astype(np.float32)

        return results
