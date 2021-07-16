import logging
import itertools
import numpy as np
import tensorflow as tf

from utility.utils import Every, RunningMeanStd
from utility.schedule import PiecewiseSchedule
from utility.rl_utils import compute_act_temp, compute_act_eps


logger = logging.getLogger(__name__)


class ActionScheduler:
    def _setup_action_schedule(self, env):
        # eval action epsilon and temperature
        self._eval_act_eps = tf.convert_to_tensor(
            getattr(self, '_eval_act_eps', 0), tf.float32)
        self._eval_act_temp = tf.convert_to_tensor(
            getattr(self, '_eval_act_temp', .5), tf.float32)

        self._schedule_act_eps = getattr(self, '_schedule_act_eps', False)
        self._schedule_act_temp = getattr(self, '_schedule_act_temp', False)
        
        self._schedule_act_epsilon(env)
        self._schedule_act_temperature(env)

    def _schedule_act_epsilon(self, env):
        """ Schedules action epsilon """
        if self._schedule_act_eps:
            if isinstance(self._act_eps, (list, tuple)):
                logger.info(f'Schedule action epsilon: {self._act_eps}')
                self._act_eps = PiecewiseSchedule(self._act_eps)
            else:
                self._act_eps = compute_act_eps(
                    self._act_eps_type, 
                    self._act_eps, 
                    getattr(self, '_id', None), 
                    getattr(self, '_n_workers', getattr(env, 'n_workers', 1)), 
                    env.n_envs)
                if env.action_shape != ():
                    self._act_eps = self._act_eps.reshape(-1, 1)
                self._schedule_act_eps = False  # not run-time scheduling
        print('Action epsilon:', np.reshape(self._act_eps, -1))
        if not isinstance(getattr(self, '_act_eps', None), PiecewiseSchedule):
            self._act_eps = tf.convert_to_tensor(self._act_eps, tf.float32)

    def _schedule_act_temperature(self, env):
        """ Schedules action temperature """
        if self._schedule_act_temp:
            self._act_temp = compute_act_temp(
                self._min_temp,
                self._max_temp,
                getattr(self, '_n_exploit_envs', 0),
                getattr(self, '_id', None),
                getattr(self, '_n_workers', getattr(env, 'n_workers', 1)), 
                env.n_envs)
            self._act_temp = self._act_temp.reshape(-1, 1)
            self._schedule_act_temp = False         # not run-time scheduling    
        else:
            self._act_temp = getattr(self, '_act_temp', 1)
        print('Action temperature:', np.reshape(self._act_temp, -1))
        self._act_temp = tf.convert_to_tensor(self._act_temp, tf.float32)

    def _get_eps(self, evaluation):
        """ Gets action epsilon """
        if evaluation:
            eps = self._eval_act_eps
        else:
            if self._schedule_act_eps:
                eps = self._act_eps.value(self.env_step)
                self.store(act_eps=eps)
                eps = tf.convert_to_tensor(eps, tf.float32)
            else:
                eps = self._act_eps
        return eps
    
    def _get_temp(self, evaluation):
        """ Gets action temperature """
        return self._eval_act_temp if evaluation else self._act_temp


class RMS:
    def _setup_rms_stats(self):
        # by default, we update reward stats once every N steps so we normalize long the first two axis
        self._reward_normalized_axis = tuple(
            getattr(self, '_reward_normalized_axis', (0, 1)))
        # by default, we update obs stats every step so we normalize along the first axis
        self._obs_normalized_axis = tuple(
            getattr(self, '_obs_normalized_axis', (0,)))
        self._normalize_obs = getattr(self, '_normalize_obs', False)
        self._normalize_reward = getattr(self, '_normalize_reward', False)
        self._normalize_reward_with_reversed_return = \
            getattr(self, '_normalize_reward_with_reversed_return', True)
        
        self._obs_names = getattr(self, '_obs_names', ['obs'])
        if self._normalize_obs:
            # we use dict to track a set of observation features
            self._obs_rms = {}
            for k in self._obs_names:
                self._obs_rms[k] = RunningMeanStd(
                    self._obs_normalized_axis, 
                    clip=getattr(self, '_obs_clip', 5), 
                    name=f'{k}_rms', ndim=1)
        else:
            self._obs_rms = None
        self._reward_rms = self._normalize_reward \
            and RunningMeanStd(self._reward_normalized_axis, 
                clip=getattr(self, '_rew_clip', 10), 
                name='reward_rms', ndim=0)
        if self._normalize_reward_with_reversed_return:
            self._reverse_return = 0
        else:
            self._reverse_return = -np.inf
        self._rms_path = f'{self._root_dir}/{self._model_name}/rms.pkl'

        logger.info(f'Observation normalization: {self._normalize_obs}')
        logger.info(f'Normalized observation names: {self._obs_names}')
        logger.info(f'Reward normalization: {self._normalize_reward}')
        logger.info(f'Reward normalization with reversed return: '
                    f'{self._normalize_reward_with_reversed_return}')

    def _process_obs(self, obs, update_rms=True, mask=None):
        """ Do obs normalization if required
        Args:
            mask: life mask, implying if the agent is still alive,
                useful for multi-agent environments, where 
                some agents might be dead before others.
        """
        if isinstance(obs, dict):
            for k in self._obs_names:
                v = obs[k]
                if update_rms:
                    self.update_obs_rms(v, k, mask=mask)
                # mask is important here as the value function still matters
                # even after the agent is dead
                obs[k] = self.normalize_obs(v, k, mask=mask)
        else:
            self.update_obs_rms(obs, mask=mask)
            obs = self.normalize_obs(obs, mask=mask)
    
    """ Functions for running mean and std """
    def set_rms_stats(self, obs_rms={}, rew_rms=None):
        if obs_rms:
            for k, v in obs_rms.items():
                self._obs_rms[k].set_rms_stats(*v)
        if rew_rms:
            self._reward_rms.set_rms_stats(*rew_rms)

    def get_rms_stats(self):
        return self.get_obs_rms_stats(), self.get_rew_rms_stats()

    def get_obs_rms_stats(self):
        obs_rms = {k: v.get_rms_stats() for k, v in self._obs_rms.items()} \
            if self._normalize_obs else {}
        return obs_rms

    def get_rew_rms_stats(self):
        rew_rms = self._reward_rms.get_rms_stats() if self._normalize_reward else ()
        return rew_rms

    @property
    def is_obs_or_reward_normalized(self):
        return self._normalize_obs or self._normalize_reward
    
    @property
    def is_obs_normalized(self):
        return self._normalize_obs

    @property
    def is_reward_normalized(self):
        return self._normalize_reward

    def update_obs_rms(self, obs, name='obs', mask=None):
        if self._normalize_obs:
            if obs.dtype == np.uint8 and \
                    getattr(self, '_image_normalization_warned', False):
                logger.warning('Image observations are normalized. Make sure you intentionally do it.')
                self._image_normalization_warned = True
            self._obs_rms[name].update(obs, mask=mask)

    def update_reward_rms(self, reward, discount=None, mask=None):
        if self._normalize_reward:
            assert len(reward.shape) == len(self._reward_normalized_axis), \
                (reward.shape, self._reward_normalized_axis)
            if self._normalize_reward_with_reversed_return:
                """
                Pseudocode can be found in https://arxiv.org/pdf/1811.02553.pdf
                section 9.3 (which is based on our Baselines code, haha)
                Motivation is that we'd rather normalize the returns = sum of future rewards,
                but we haven't seen the future yet. So we assume that the time-reversed rewards
                have similar statistics to the rewards, and normalize the time-reversed rewards.

                Quoted from
                https://github.com/openai/phasic-policy-gradient/blob/master/phasic_policy_gradient/reward_normalizer.py
                Yeah, you may not find the pseudocode. That's why I quote:-)
                """
                assert discount is not None, \
                    f"Normalizing rewards with reversed return requires environment's reset signals"
                assert reward.ndim == discount.ndim == len(self._reward_rms.axis), \
                    (reward.shape, discount.shape, self._reward_rms.axis)
                self._reverse_return, ret = backward_discounted_sum(
                    self._reverse_return, reward, discount, self._gamma)
                self._reward_rms.update(ret, mask=mask)
            else:
                self._reward_rms.update(reward, mask=mask)

    def normalize_obs(self, obs, name='obs', mask=None):
        """ Normalize obs using obs RMS """
        return self._obs_rms[name].normalize(obs, mask=mask) \
            if self._normalize_obs else obs

    def normalize_reward(self, reward, mask=None):
        """ Normalize obs using reward RMS """
        return self._reward_rms.normalize(reward, zero_center=False, mask=mask) \
            if self._normalize_reward else reward

    def update_from_rms_stats(self, obs_rms, rew_rms):
        for k, v in obs_rms.items():
            if v:
                self._obs_rms[k].update_from_moments(
                    batch_mean=v.mean,
                    batch_var=v.var,
                    batch_count=v.count)
        if rew_rms:
            self._reward_rms.update_from_moments(
                batch_mean=rew_rms.mean,
                batch_var=rew_rms.var,
                batch_count=rew_rms.count)

def backward_discounted_sum(prev_ret, reward, discount, gamma):
    """ Compute the discounted sum of rewards in the reverse order"""
    assert reward.ndim == discount.ndim, (reward.shape, discount.shape)
    if reward.ndim == 1:
        prev_ret = reward + gamma * prev_ret
        ret = prev_ret.copy()
        prev_ret *= discount
        return prev_ret, ret
    else:
        nstep = reward.shape[1]
        ret = np.zeros_like(reward)
        for t in range(nstep):
            ret[:, t] = prev_ret = reward[:, t] + gamma * prev_ret
            prev_ret *= discount[:, t]
        return prev_ret, ret


""" According to Python's MRO, the following classes should be positioned 
before AgentBase to override the corresponding functions. """
class Memory:
    def _setup_memory_state_record(self):
        """ Setups attributes for RNNs """
        self._state = None
        # do specify additional_rnn_inputs in *config.yaml. Otherwise, 
        # no additional rnn input is expected.
        # additional_rnn_inputs is expected to be a dict of (name, dtypes)
        # NOTE: additional rnn inputs are not tested yet.
        self._additional_rnn_inputs = getattr(self, '_additional_rnn_inputs', {})
        self._default_additional_rnn_inputs = self._additional_rnn_inputs.copy()
        self._squeeze_batch = False
        logger.info(f'Additional rnn inputs: {self._additional_rnn_inputs}')
    
    def _add_memory_state_to_kwargs(self, 
            obs, mask, state=None, kwargs={}, prev_reward=None, batch_size=None):
        """ Adds memory state to kwargs. Call this in self._process_input 
        when introducing sequential memory.
        """
        if state is None and self._state is None:
            batch_size = batch_size or (tf.shape(obs[0])[0] 
                if isinstance(obs, dict) else tf.shape(obs)[0])
            self._state = self.model.get_initial_state(batch_size=batch_size)
            for k, v in self._additional_rnn_inputs.items():
                assert v in ('float32', 'int32', 'float16'), v
                if k == 'prev_action':
                    self._additional_rnn_inputs[k] = tf.zeros(
                        (batch_size, *self._action_shape), dtype=v)
                else:
                    self._additional_rnn_inputs[k] = tf.zeros(batch_size, dtype=v)
            self._squeeze_batch = batch_size == 1

        if 'prev_reward' in self._additional_rnn_inputs:
            # by default, we do not process rewards. However, if you want to use
            # rewards as additional rnn inputs, you need to make sure it has 
            # the batch dimension
            assert self._additional_rnn_inputs['prev_reward'].ndims == prev_reward.ndim, prev_reward
            self._additional_rnn_inputs['prev_reward'] = tf.convert_to_tensor(
                prev_reward, self._additional_rnn_inputs['prev_reward'].dtype)

        if state is None:
            state = self._state

        state = self._apply_mask_to_state(state, mask)
        kwargs.update({
            'state': state,
            'mask': mask,   # mask is applied in RNN
            **self._additional_rnn_inputs
        })
        
        return kwargs
    
    def _add_tensors_to_terms(self, obs, kwargs, out, evaluation):
        """ Adds tensors to terms, which will be subsequently stored in the replay,
        call this before converting tensors to np.ndarray """
        out, self._state = out

        if not evaluation:
            # out is (action, terms), we add necessary stats to terms
            if self._store_state:
                out[1].update(kwargs['state']._asdict())
            if 'prev_action' in self._additional_rnn_inputs:
                out[1]['prev_action'] = self._additional_rnn_inputs['prev_action']
            if 'prev_reward' in self._additional_rnn_inputs:
                out[1]['prev_reward'] = self._additional_rnn_inputs['prev_reward']
            if self._squeeze_batch:
                # squeeze the batch dimension
                for k, v in out[1].items():
                    if len(out[1][k].shape) > 0 and out[1][k].shape[0] == 1:
                        out[1][k] = tf.squeeze(v, 0)
        
        if 'prev_action' in self._additional_rnn_inputs:
            self._additional_rnn_inputs['prev_action'] = \
                out[0] if isinstance(out, tuple) else out

        return out
    
    def _add_non_tensors_to_terms(self, out, kwargs, evaluation):
        """ Adds additional input terms, which are of non-Tensor type """
        if not evaluation:
            out[1]['mask'] = kwargs['mask']
        return out

    def _get_mask(self, reset):
        return np.float32(1. - reset)

    def _apply_mask_to_state(self, state, mask):
        if state is not None:
            mask_exp = np.expand_dims(mask, -1)
            if isinstance(state, (list, tuple)):
                state_type = type(state)
                state = state_type(*[v * mask_exp for v in state])
            else:
                state = state * mask_exp
        return state

    def reset_states(self, state=None):
        if state is None:
            self._state, self._additional_rnn_inputs = None, self._default_additional_rnn_inputs.copy()
        else:
            self._state, self._additional_rnn_inputs = state

    def get_states(self):
        return self._state, self._additional_rnn_inputs


class TargetNetOps:
    def _setup_target_net_sync(self):
        self._to_sync = Every(self._target_update_period) \
            if hasattr(self, '_target_update_period') else None

    @tf.function
    def _sync_nets(self):
        """ Synchronizes the target net with the online net """
        ons = self.get_online_nets()
        tns = self.get_target_nets()
        logger.info(f"Sync Networks | Online networks: {[n.name for n in ons]}")
        logger.info(f"Sync Networks | Target networks: {[n.name for n in tns]}")
        ovars = list(itertools.chain(*[v.variables for v in ons]))
        tvars = list(itertools.chain(*[v.variables for v in tns]))
        logger.info(f"Sync Networks | Online network parameters:\n\t" 
            + '\n\t'.join([f'{n.name}, {n.shape}' for n in ovars]))
        logger.info(f"Sync Networks | Target network parameters:\n\t" 
            + '\n\t'.join([f'{n.name}, {n.shape}' for n in tvars]))
        assert len(tvars) == len(ovars), f'{tvars}\n{ovars}'
        [tvar.assign(ovar) for tvar, ovar in zip(tvars, ovars)]

    @tf.function
    def _update_nets(self):
        """ Updates the target net towards online net using exponentially moving average """
        ons = self.get_online_nets()
        tns = self.get_target_nets()
        logger.info(f"Update Networks | Online networks: {[n.name for n in ons]}")
        logger.info(f"Update Networks | Target networks: {[n.name for n in tns]}")
        ovars = list(itertools.chain(*[v.variables for v in ons]))
        tvars = list(itertools.chain(*[v.variables for v in tns]))
        logger.info(f"Update Networks | Online network parameters:\n" 
            + '\n\t'.join([f'{n.name}, {n.shape}' for n in ovars]))
        logger.info(f"Update Networks | Target network parameters:\n" 
            + '\n\t'.join([f'{n.name}, {n.shape}' for n in tvars]))
        assert len(tvars) == len(ovars), f'{tvars}\n{ovars}'
        [tvar.assign(self._polyak * tvar + (1. - self._polyak) * mvar) 
            for tvar, mvar in zip(tvars, ovars)]

    def get_online_nets(self):
        """ Gets the online networks """
        return [getattr(self, f'{k}') for k in self.model 
            if f'target_{k}' in self.model]

    def get_target_nets(self):
        """ Gets the target networks """
        return [getattr(self, f'target_{k}') for k in self.model 
            if f'target_{k}' in self.model]
