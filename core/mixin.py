import logging
import itertools
import numpy as np
import tensorflow as tf

from utility.utils import Every
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


""" According to Python's MRO, the following classes should be positioned 
before AgentBase to override the corresponding functions. """
class Memory:
    def _setup_memory_state_record(self):
        """ Setups attributes for RNNs """
        self._state = None
        # do specify additional_rnn_inputs in *config.yaml. Otherwise, 
        # no additional rnn input is expected.
        # additional_rnn_inputs is expected to be a dict of name-dtypes
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
