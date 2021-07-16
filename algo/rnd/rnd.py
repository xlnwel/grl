import logging
import cloudpickle
import numpy as np
import tensorflow as tf

from core.base import backward_discounted_sum
from utility.utils import RunningMeanStd


logger = logging.getLogger(__name__)

class RND:
    def __init__(self, model, gamma_int, rms_path):
        self.predictor = model['predictor']
        self.target = model['target']
        self._gamma_int = gamma_int
        axis = (0, 1)
        self._obs_rms = RunningMeanStd(axis=axis, epsilon=1e-4, clip=5)
        self._prev_int_return = 0
        self._int_return_rms = RunningMeanStd(axis=axis, epsilon=1e-4)
        self._rms_path = rms_path
        self._rms_restored = False

    def compute_int_reward(self, next_obs):
        """ next_obs is expected to be normalized """
        assert len(next_obs.shape) == 5, next_obs.shape
        assert next_obs.dtype == np.float32, next_obs.dtype
        assert next_obs.shape[-1] == 1, next_obs.shape
        int_reward = self._intrinsic_reward(next_obs).numpy()
        return int_reward

    def update_int_reward_rms(self, int_reward):
        self._prev_int_return, int_return = backward_discounted_sum(
            self._prev_int_return, int_reward, np.ones_like(int_reward), self._gamma_int)
        self._int_return_rms.update(int_return)

    def normalize_int_reward(self, reward):
        norm_reward = self._int_return_rms.normalize(reward, zero_center=False)
        return norm_reward

    @tf.function
    def _intrinsic_reward(self, next_obs):
        target_feat = self.target(next_obs)
        pred_feat = self.predictor(next_obs)
        int_reward = tf.reduce_mean(tf.square(target_feat - pred_feat), axis=-1)
        return int_reward

    def update_obs_rms(self, obs):
        if obs.dtype == np.uint8 and obs.shape[-1] > 1:
            # for stacked frames, we only use
            # the most recent one for rms update
            obs = obs[..., -1:]
        assert len(obs.shape) == 5 and obs.shape[-1] == 1, obs.shape
        assert obs.dtype == np.uint8, obs.dtype
        self._obs_rms.update(obs)

    def normalize_obs(self, obs):
        assert len(obs.shape) == 5, obs.shape
        next_obs = self._obs_rms.normalize(obs[..., -1:])
        assert not np.any(np.isnan(next_obs))
        return next_obs

    def restore(self):
        import os
        if os.path.exists(self._rms_path):
            with open(self._rms_path, 'rb') as f:
                self._obs_rms, self._int_return_rms, self._prev_int_return = \
                    cloudpickle.load(f)
            logger.info('RMSs are restored')
            self._rms_restored = True

    def save(self):
        with open(self._rms_path, 'wb') as f:
            cloudpickle.dump(
                (self._obs_rms, self._int_return_rms, self._prev_int_return), f)

    def rms_restored(self):
        return self._rms_restored
    
    def get_rms_stats(self):
        return self._obs_rms.get_rms_stats(), self._int_return_rms.get_rms_stats()
