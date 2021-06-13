import logging
import collections
import numpy as np

from replay.uniform import UniformReplay


logger = logging.getLogger(__name__)

class SequentialBase:
    """ Construction """
    def _add_attributes(self, state_keys=None):
        self._state_keys = state_keys or getattr(self, '_state_keys', [])
        self._memory = collections.deque(maxlen=self._capacity)
        self._first = True

    def _construct_temp_buff(self):
        from replay.func import create_local_buffer
        self._tmp_buf = create_local_buffer({
            'replay_type': self._replay_type,
            'n_envs': self._n_envs,
            'sample_size': self._sample_size,
            'state_keys': self._state_keys,
            'extra_keys': self._extra_keys,
        })

    def __len__(self):
        return len(self._memory)

    def clear_temp_buffer(self):
        self._tmp_buf.clear()

    """ Sequential Replay Methods"""
    def add(self, **data):
        self._tmp_buf.add(**data)
        if self._tmp_buf.is_full():
            self.merge(self._tmp_buf.sample())
            self._tmp_buf.reset()

    def merge(self, local_buffer):
        """ Adds local_buffer to memory 
        Args:
            local_buffer: either a list/tuple of dicts across multiple timesteps
                or a dict of data at a single timestep
        """
        if isinstance(local_buffer, (list, tuple)):
            length = len(local_buffer)
            mem_idxes = np.arange(self._mem_idx, self._mem_idx + length) % self._capacity
            if self._replay_type.endswith('per'):
                priorities = np.array([b.pop('priority', self._top_priority) for b in local_buffer])
                np.testing.assert_array_less(0, priorities)
                self._data_structure.batch_update(mem_idxes, priorities)
            self._memory.extend(local_buffer)
            self._mem_idx = self._mem_idx + length
        else:
            if self._replay_type.endswith('per'):
                priority = local_buffer.pop('priority', self._top_priority)
                assert priority > 0, priority
                self._data_structure.update(self._mem_idx, priority)
            self._memory.append(local_buffer)
            self._mem_idx = self._mem_idx + 1
        if self._first:
            logger.info('First sample')
            for k, v in self._memory[0].items():
                logger.info(f'\t{k}, {v.shape}, {v.dtype}')
            self._first = False
        
        if not self._is_full and self._mem_idx >= self._capacity:
            print(f'Memory is full({len(self)})')
            self._is_full = True
        self._mem_idx = self._mem_idx % self._capacity

    def _get_samples(self, idxes):
        assert len(idxes) == self._batch_size, idxes
        results = collections.defaultdict(list)
        [results[k].append(v) for i in idxes for k, v in self._memory[i].items()]
        results = {k: np.stack(v) for k, v in results.items()}
        for k, v in results.items():
            if k in self._state_keys:
                assert v.shape[0] == self._batch_size, (k, v.shape)
            elif k in self._extra_keys:
                assert v.shape[:2] == (self._batch_size, self._sample_size+1), (k, v.shape)
            else:
                assert v.shape[:2] == (self._batch_size, self._sample_size), (k, v.shape)

        return results

class SequentialReplay(SequentialBase, UniformReplay):
    def __init__(self, config, state_keys=None):
        super().__init__(config, state_keys=state_keys)
