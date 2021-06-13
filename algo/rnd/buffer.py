import logging
import numpy as np

from replay.utils import init_buffer, print_buffer
from algo.ppo.buffer import Buffer as BufferBase


logger = logging.getLogger(__name__)

class Buffer(BufferBase):
    def _add_attributes(self):
        super()._add_attributes()
        self._gae_discount_int = self._gamma_int * self._lam

    def update_value_with_func(self, fn):
        assert self._mb_idx == 0, f'Unfinished sample: self._mb_idx({self._mb_idx}) != 0'
        mb_idx = 0
        for start in range(0, self._size, self._mb_size):
            end = start + self._mb_size
            curr_idxes = self._idxes[start:end]
            obs = self._memory['obs'][curr_idxes]
            value_int, value_ext = fn(obs)
            self.update('value_int', value_int, mb_idxes=curr_idxes)
            self.update('value_ext', value_ext, mb_idxes=curr_idxes)
        
        assert mb_idx == 0, mb_idx

    def get_obs(self, last_obs):
        assert self._idx == self.N_STEPS, self._idx
        return np.concatenate(
            [self._memory['obs'], np.expand_dims(last_obs, 1)], axis=1)

    def finish(self, reward_int, obs_norm, last_value_int, last_value_ext):
        assert self._idx == self.N_STEPS, self._idx
        assert obs_norm.shape == self._memory['obs_norm'].shape, obs_norm.shape
        self._memory['obs_norm'] = obs_norm
        self.reshape_to_store()

        adv_int, self._memory['traj_ret_int'] = self._compute_advantage_return(
            reward_int, self._memory['discount_int'], 
            self._memory['value_int'], last_value_int
        )
        adv_ext, self._memory['traj_ret_ext'] = self._compute_advantage_return(
            self._memory['reward'], self._memory['discount'], 
            self._memory['value_ext'], last_value_ext
        )

        self._memory['advantage'] = self._int_coef*adv_int + self._ext_coef*adv_ext

        self.reshape_to_sample()
        self._ready = True

    def _init_buffer(self, data):
        init_buffer(self._memory, pre_dims=(self._n_envs, self.N_STEPS), **data)
        self._memory['discount_int'] = np.ones_like(self._memory['discount'])   # non-episodic
        norm_obs_shape = self._memory['obs'].shape[:-1] + (1, )
        self._memory['obs_norm'] = np.zeros(norm_obs_shape, dtype=np.float32)
        self._memory['traj_ret_int'] = np.zeros((self._n_envs, self.N_STEPS), dtype=np.float32)
        self._memory['traj_ret_ext'] = np.zeros((self._n_envs, self.N_STEPS), dtype=np.float32)
        self._memory['advantage'] = np.zeros((self._n_envs, self.N_STEPS), dtype=np.float32)
        print_buffer(self._memory)
        if self._inferred_sample_keys or getattr(self, '_sample_keys', None) is None:
            self._sample_keys = set(self._memory.keys()) - set(('discount', 'reward'))
            self._inferred_sample_keys = True
