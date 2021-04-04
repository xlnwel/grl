import logging
import numpy as np

from env.wrappers import EnvOutput
from algo.ppo.buffer import compute_indices
from algo.ppo.buffer import Buffer as BufferBase


logger = logging.getLogger(__name__)

class Buffer(BufferBase):
    def update_value_with_func(self, fn):
        assert self._mb_idx == 0, f'Unfinished sample: self._mb_idx({self._mb_idx}) != 0'
        mb_idx = 0
        for start in range(0, self._size, self._mb_size):
            end = start + self._mb_size
            curr_idxes = self._idxes[start:end]
            obs = self._memory['obs'][curr_idxes]
            state = (self._memory[k][curr_idxes, 0] for k in self._state_keys)
            mask = self._memory['mask'][curr_idxes]
            value, state = fn(obs, state=state, mask=mask, return_state=True)
            self.update('value', value, mb_idxes=curr_idxes)
            next_idxes = curr_idxes + self._mb_size
            self.update('state', state, mb_idxes=next_idxes)
        
        assert mb_idx == 0, mb_idx

    def sample(self):
        assert self._ready
        if self._shuffle and self._mb_idx == 0:
            np.random.shuffle(self._shuffled_idxes)
        self._mb_idx, self._curr_idxes = compute_indices(
            self._shuffled_idxes, self._mb_idx, self._mb_size, self.N_MBS)

        sample = {k: self._memory[k][self._curr_idxes, 0]
            if k in self._state_keys 
            else self._memory[k][self._curr_idxes] 
            for k in self._sample_keys}
        
        return sample
