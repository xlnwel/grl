import numpy as np
import tensorflow as tf

from core.tf_config import build
from core.decorator import override
from algo.ppo.base import PPOBase, collect


def get_data_format(*, env, **kwargs):
    obs_dtype = tf.uint8 if len(env.obs_shape) == 3 else tf.float32
    action_dtype = tf.int32 if env.is_action_discrete else tf.float32
    data_format = dict(
        obs=((None, *env.obs_shape), obs_dtype),
        action=((None, *env.action_shape), action_dtype),
        value=((None, ), tf.float32), 
        traj_ret=((None, ), tf.float32),
        advantage=((None, ), tf.float32),
        logpi=((None, ), tf.float32),
    )

    return data_format


class Agent(PPOBase):
    """ Initialization """
    @override(PPOBase)
    def _build_learn(self, env):
        # Explicitly instantiate tf.function to avoid unintended retracing
        dtype = tf.float32
        obs_dtype = dtype if np.issubdtype(env.obs_dtype, np.floating) else env.obs_dtype
        action_dtype = dtype if np.issubdtype(env.action_dtype, np.floating) else env.action_dtype
        TensorSpecs = dict(
            obs=(env.obs_shape, obs_dtype, 'obs'),
            action=(env.action_shape, action_dtype, 'action'),
            value=((), tf.float32, 'value'),
            traj_ret=((), tf.float32, 'traj_ret'),
            advantage=((), tf.float32, 'advantage'),
            logpi=((), tf.float32, 'logpi'),
        )
        self.learn = build(self._learn, TensorSpecs)

    # @override(PPOBase)
    # def _summary(self, data, terms):
    #     tf.summary.histogram('sum/value', data['value'], step=self._env_step)
    #     tf.summary.histogram('sum/logpi', data['logpi'], step=self._env_step)
