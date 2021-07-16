import logging
import numpy as np

from algo.ppo.buffer import Buffer as BufferBase


logger = logging.getLogger(__name__)


class Buffer(BufferBase):
    def update_value_with_func(self, fn):
        assert self._mb_idx == 0, f'Unfinished sample: self._mb_idx({self._mb_idx}) != 0'
        mb_idx = 0
        for start in range(0, self._size, self._mb_size):
            end = start + self._mb_size
            curr_idxes = self._idxes[start:end]
            global_state = self._memory['global_state'][curr_idxes]
            state = tuple([self._memory[k][curr_idxes, 0] for k in self._state_keys])
            mask = self._memory['mask'][curr_idxes]
            value = fn(global_state, state=state, mask=mask)
            self.update('value', value, mb_idxes=curr_idxes)
            # TODO: you may want to update states as well
        
        assert mb_idx == 0, mb_idx

    def _init_buffer(self, data):
        self._n_envs = data['reward'].shape[0]
        self._size = self._n_envs * self.N_STEPS // self._sample_size
        self._mb_size = self._size // self.N_MBS
        self._idxes = np.arange(self._size)
        self._shuffled_idxes = np.arange(self._size)

        super()._init_buffer(data)
