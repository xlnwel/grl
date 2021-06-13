import tensorflow as tf

from utility.tf_utils import tensor2numpy
from core.tf_config import build
from core.decorator import override
from core.base import Memory
from env.wrappers import EnvOutput
from algo.ppo.base import PPOBase, collect


class Agent(Memory, PPOBase):
    """ Initialization """
    @override(PPOBase)
    def _add_attributes(self, env, dataset):
        super()._add_attributes(env, dataset)
        self._setup_memory_state_record()

    @override(PPOBase)
    def _build_learn(self, env):
        # Explicitly instantiate tf.function to avoid unintended retracing
        TensorSpecs = dict(
            obs=((self._sample_size, *env.obs_shape), env.obs_dtype, 'obs'),
            action=((self._sample_size, *env.action_shape), env.action_dtype, 'action'),
            value=((self._sample_size,), tf.float32, 'value'),
            traj_ret=((self._sample_size,), tf.float32, 'traj_ret'),
            advantage=((self._sample_size,), tf.float32, 'advantage'),
            logpi=((self._sample_size,), tf.float32, 'logpi'),
            mask=((self._sample_size,), tf.float32, 'mask'),
        )
        if self._store_state:
            state_type = type(self.model.state_size)
            TensorSpecs['state'] = state_type(*[((sz, ), self._dtype, name) 
                for name, sz in self.model.state_size._asdict().items()])
        if self._additional_rnn_inputs:
            if 'prev_action' in self._additional_rnn_inputs:
                TensorSpecs['prev_action'] = ((self._sample_size, *env.action_shape), env.action_dtype, 'prev_action')
            if 'prev_reward' in self._additional_rnn_inputs:
                TensorSpecs['prev_reward'] = ((self._sample_size,), self._dtype, 'prev_reward')    # this reward should be unnormlaized
        self.learn = build(self._learn, TensorSpecs)

    """ Call """
    # @override(PPOBase)
    def _process_input(self, env_output, evaluation):
        obs, kwargs = super()._process_input(env_output, evaluation)
        mask = 1. - env_output.reset
        kwargs = self._add_memory_state_to_kwargs(obs, mask, kwargs=kwargs)
        return obs, kwargs

    # @override(PPOBase)
    def _process_output(self, obs, kwargs, out, evaluation):
        out = self._add_tensors_to_terms(obs, kwargs, out, evaluation)
        out = super()._process_output(obs, kwargs, out, evaluation)
        out = self._add_non_tensors_to_terms(out, kwargs, evaluation)
        return out

    """ PPO methods """
    @override(PPOBase)
    def record_last_env_output(self, env_output):
        self._env_output = EnvOutput(
            self.normalize_obs(env_output.obs), *env_output[1:])

    @override(PPOBase)
    def compute_value(self, obs=None, state=None, mask=None, prev_reward=None, return_state=False):
        # be sure obs is normalized if obs normalization is required
        if obs is None:
            obs = self._env_output.obs
        if state is None:
            state = self._state
        if mask is None:
            mask = 1. - self._env_output.reset
        kwargs = self._add_memory_state_to_kwargs(
            obs, mask, state=state, prev_reward=prev_reward)
        kwargs['return_state'] = return_state
        out = self.model.compute_value(obs, **kwargs)
        return tensor2numpy(out)
