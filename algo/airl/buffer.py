import numpy as np

from algo.ppo.buffer import Buffer as PPOBuffer

class Buffer(PPOBuffer):
    def sample_for_disc(self, batch_size):
        idx = np.random.randint(0, self._size, size=batch_size)
        return {k: self._memory[k][idx] for k in 
            ['obs', 'action', 'discount', 'logpi', 'next_obs']}

    def compute_reward_with_func(self, fn):
        assert self._mb_idx == 0, f'Unfinished sample: self._mb_idx({self._mb_idx}) != 0'
        mb_idx = 0
        for start in range(0, self._size, self._mb_size):
            end = start + self._mb_size
            curr_idxes = self._idxes[start:end]
            obs = self._memory['obs'][curr_idxes]
            action = self._memory['action'][curr_idxes]
            discount = self._memory['discount'][curr_idxes]
            next_obs = self._memory['next_obs'][curr_idxes]
            logpi = self._memory['logpi'][curr_idxes]
            reward = fn(obs, action, discount, logpi, next_obs)
            self.update('reward', reward, mb_idxes=curr_idxes)

        assert mb_idx == 0, mb_idx