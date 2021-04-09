from abc import ABC, abstractmethod
import os
import logging
import itertools
import cloudpickle

from utility.display import pwc
from utility.utils import Every
from utility.timer import Timer
from utility.schedule import PiecewiseSchedule, TFPiecewiseSchedule
from utility.rl_utils import compute_act_temp, compute_act_eps
from core.log import *
from core.optimizer import Optimizer
from core.checkpoint import *
from core.decorator import override, agent_config


logger = logging.getLogger(__name__)

class AgentImpl(ABC):
    """ Restore & save """
    def restore(self):
        """ Restore the latest parameter recorded by ckpt_manager

        Args:
            ckpt_manager: An instance of tf.train.CheckpointManager
            ckpt: An instance of tf.train.Checkpoint
            ckpt_path: The directory in which to write checkpoints
            name: optional name for print
        """
        restore(self._ckpt_manager, self._ckpt, self._ckpt_path, self._model_name)
        self.env_step = self._env_step.numpy()
        self.train_step = self._train_step.numpy()

    def save(self, print_terminal_info=False):
        """ Save Model
        
        Args:
            ckpt_manager: An instance of tf.train.CheckpointManager
        """
        self._env_step.assign(self.env_step)
        self._train_step.assign(self.train_step)
        save(self._ckpt_manager, print_terminal_info=print_terminal_info)

    """ Logging """
    def save_config(self, config):
        save_config(self._root_dir, self._model_name, config)

    def log(self, step, prefix=None, print_terminal_info=True):
        log(self._logger, self._writer, self._model_name, prefix=prefix, 
            step=step, print_terminal_info=print_terminal_info)

    def log_stats(self, stats, print_terminal_info=True):
        log_stats(self._logger, stats, print_terminal_info=print_terminal_info)

    def set_summary_step(self, step):
        set_summary_step(step)

    def scalar_summary(self, stats, prefix=None, step=None):
        scalar_summary(self._writer, stats, prefix=prefix, step=step)

    def histogram_summary(self, stats, prefix=None, step=None):
        histogram_summary(self._writer, stats, prefix=prefix, step=step)

    def graph_summary(self, sum_type, *args, step=None):
        """
        Args:
            sum_type str: either "video" or "image"
            args: Args passed to summary function defined in utility.graph,
                of which the first must be a str to specify the tag in Tensorboard
            
        """
        assert isinstance(args[0], str), f'args[0] is expected to be a name string, but got "{args[0]}"'
        args = list(args)
        args[0] = f'{self.name}/{args[0]}'
        graph_summary(self._writer, sum_type, args, step=step)

    def store(self, **kwargs):
        store(self._logger, **kwargs)

    def get_raw_item(self, key):
        return get_raw_item(self._logger, key)

    def get_item(self, key, mean=True, std=False, min=False, max=False):
        return get_item(self._logger, key, mean=mean, std=std, min=min, max=max)

    def get_raw_stats(self):
        return get_raw_stats(self._logger)

    def get_stats(self, mean=True, std=False, min=False, max=False):
        return get_stats(self._logger, mean=mean, std=std, min=min, max=max)
    
    def contains_stats(self, key):
        return contains_stats(self._logger, key)

    def print_construction_complete(self):
        pwc(f'{self.name.upper()} is constructed...', color='cyan')


class AgentBase(AgentImpl):
    """ Initialization """
    @agent_config
    def __init__(self, *, env, dataset):
        super().__init__()
        self.dataset = dataset

        self._obs_shape = env.obs_shape
        self._action_shape = env.action_shape
        self._action_dim = env.action_dim

        # intervals between calling self._summary
        self._to_summary = Every(self.LOG_PERIOD, self.LOG_PERIOD)

        self._add_attributes(env, dataset)
        self._construct_optimizers()
        self._build_learn(env)
        self._sync_nets()
    
    def _add_attributes(self, env, dataset):
        self._sample_timer = Timer('sample')
        self._train_timer = Timer('train')

        self._return_stats = getattr(self, '_return_stats', False)

        self.RECORD = getattr(self, 'RECORD', False)
        self.N_EVAL_EPISODES = getattr(self, 'N_EVAL_EPISODES', 1)

    @abstractmethod
    def _construct_optimizers(self):
        self._optimizer = self._construct_opt()

    def _construct_opt(self, models=None, lr=None, opt=None, l2_reg=None,
            weight_decay=None, clip_norm=None, epsilon=None):
        lr = lr or self._lr
        opt = opt or getattr(self, '_optimizer', 'adam')
        l2_reg = l2_reg or getattr(self, '_l2_reg', None)
        weight_decay = weight_decay or getattr(self, '_weight_decay', None)
        clip_norm = clip_norm or getattr(self, '_clip_norm', None)
        epsilon = epsilon or getattr(self, '_opt_eps', 1e-7)
        if isinstance(lr, (tuple, list)):
            lr = TFPiecewiseSchedule(lr)
        models = models or [v for k, v in self.model.items() if 'target' not in k]
        opt = Optimizer(
            opt, models, lr, 
            l2_reg=l2_reg,
            weight_decay=weight_decay,
            clip_norm=clip_norm,
            epsilon=epsilon
        )
        return opt

    @abstractmethod
    def _build_learn(self, env):
        raise NotImplementedError

    def _sync_nets(self):
        pass

    def reset_states(self, states=None):
        pass

    def get_states(self):
        pass

    def _summary(self, data, terms):
        """ Add non-scalar summaries """
        pass 

    """ Call """
    def __call__(self, env_output=(), evaluation=False, return_eval_stats=False):
        """ Call the agent to interact with the environment
        Args:
            obs: Observation(s), we keep a separate observation to for legacy reasons
            evaluation bool: evaluation mode or not
            env_output tuple: (obs, reward, discount, reset)
        """
        obs = env_output.obs
        if obs.ndim % 2 != 0:
            obs = np.expand_dims(obs, 0)    # add batch dimension
        assert obs.ndim in (2, 4), obs.shape

        obs, kwargs = self._process_input(obs, evaluation, env_output)
        out = self._compute_action(
            obs, **kwargs, 
            evaluation=evaluation, 
            return_stats=self._return_stats,
            return_eval_stats=return_eval_stats)
        out = self._process_output(obs, kwargs, out, evaluation)

        return out

    def _process_input(self, obs, evaluation, env_output):
        """Do necessary pre-process and produce inputs to model
        Args:
            obs: Observations with added batch dimension
        Returns: 
            obs: Pre-processed observations
            kwargs, dict: kwargs necessary to model  
        """
        return obs, {}
    
    def _compute_action(self, obs, **kwargs):
        return self.model.action(obs, **kwargs)
        
    def _process_output(self, obs, kwargs, out, evaluation):
        """Post-process output
        Args:
            obs: Pre-processed observations
            kwargs, dict: kwargs necessary to model  
            out (action, terms): Model output
        Returns:
            out: results supposed to return by __call__
        """
        return tf.nest.map_structure(lambda x: x.numpy(), out)

    @abstractmethod
    def _learn(self):
        raise NotImplementedError


class RMSAgentBase(AgentBase):
    @override(AgentBase)
    def _add_attributes(self, env, dataset):
        super()._add_attributes(env, dataset)

        from utility.utils import RunningMeanStd
        self._normalized_axis = getattr(self, '_normalized_axis', (0, 1))
        self._normalize_obs = getattr(self, '_normalize_obs', False)
        self._normalize_reward = getattr(self, '_normalize_reward', False)
        self._normalize_reward_with_reversed_return = \
            getattr(self, '_normalize_reward_with_reversed_return', True)
        
        axis = tuple(self._normalized_axis)
        self._obs_rms = self._normalize_obs and RunningMeanStd(axis, clip=getattr(self, '_obs_clip', None))
        self._reward_rms = self._normalize_reward and RunningMeanStd(axis, clip=getattr(self, '_rew_clip', 10))
        if self._normalize_reward_with_reversed_return:
            self._reverse_return = 0
        else:
            self._reverse_return = -np.inf
        self._rms_path = f'{self._root_dir}/{self._model_name}/rms.pkl'

        logger.info(f'Observation normalization: {self._normalize_obs}')
        logger.info(f'Reward normalization: {self._normalize_reward}')
        logger.info(f'Reward normalization with reversed return: {self._normalize_reward_with_reversed_return}')

    # @override(AgentBase)
    def _process_input(self, obs, evaluation, env_output):
        obs = self.normalize_obs(obs)
        return obs, {}
    
    @override(AgentBase)
    def _process_output(self, obs, kwargs, out, evaluation):
        out = super()._process_output(obs, kwargs, out, evaluation)        
        if self._normalize_obs and not evaluation:
            terms = out[1]
            terms['obs'] = obs
        return out

    """ Functions for running mean and std """
    def get_running_stats(self):
        obs_rms = self._obs_rms.get_stats() if self._normalize_obs else ()
        rew_rms = self._reward_rms.get_stats() if self._normalize_reward else ()
        return obs_rms, rew_rms

    @property
    def is_obs_or_reward_normalized(self):
        return self._normalize_obs or self._normalize_reward
    
    @property
    def is_obs_normalized(self):
        return self._normalize_obs

    @property
    def is_reward_normalized(self):
        return self._normalize_reward

    def update_obs_rms(self, obs):
        if self._normalize_obs:
            if obs.dtype == np.uint8 and \
                    getattr(self, '_image_normalization_warned', False):
                logger.warning('Image observations are normalized. Make sure you intentionally do it.')
                self._image_normalization_warned = True
            self._obs_rms.update(obs)

    def update_reward_rms(self, reward, discount=None):
        if self._normalize_reward:
            assert len(reward.shape) == len(self._normalized_axis), (reward.shape, self._normalized_axis)
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
                self._reward_rms.update(ret)
            else:
                self._reward_rms.update(reward)

    def normalize_obs(self, obs):
        return self._obs_rms.normalize(obs) if self._normalize_obs else obs

    def normalize_reward(self, reward):
        return self._reward_rms.normalize(reward, zero_center=False) \
            if self._normalize_reward else reward

    @override(AgentBase)
    def restore(self):
        if os.path.exists(self._rms_path):
            with open(self._rms_path, 'rb') as f:
                self._obs_rms, self._reward_rms, self._reverse_return = cloudpickle.load(f)
                logger.info(f'rms stats are restored from {self._rms_path}')
        super().restore()

    @override(AgentBase)
    def save(self, print_terminal_info=False):
        with open(self._rms_path, 'wb') as f:
            cloudpickle.dump((self._obs_rms, self._reward_rms, self._reverse_return), f)
        super().save(print_terminal_info=print_terminal_info)


def backward_discounted_sum(prev_ret, reward, discount, gamma):
    assert reward.ndim == discount.ndim, (reward.shape, discount.shape)
    if reward.ndim == 1:
        prev_ret = reward + gamma * prev_ret
        ret = prev_ret.copy()
        prev_ret *= discount
        return prev_ret, ret
    elif reward.ndim == 2:
        _nenv, nstep = reward.shape
        ret = np.zeros_like(reward)
        for t in range(nstep):
            ret[:, t] = prev_ret = reward[:, t] + gamma * prev_ret
            prev_ret *= discount[:, t]
        return prev_ret, ret
    else:
        raise ValueError(f'Unknown reward shape: {reward.shape}')


class Memory:
    """ According to Python's MRO, this class should be positioned 
    before AgentBase to override reset&get state operations. """
    def _setup_memory_state_record(self):
        self._state = None
        self._additional_rnn_inputs = getattr(self, '_additional_rnn_inputs', {})
        self._default_additional_rnn_inputs = self._additional_rnn_inputs.copy()
        self._squeeze_batch = False
    
    def _add_memory_state_to_kwargs(self, obs, env_output, kwargs, state=None):
        if self._state is None:
            B = tf.shape(obs)[0]
            self._state = self.model.get_initial_state(batch_size=B)
            for k, v in self._additional_rnn_inputs.items():
                assert v in ('float32', 'int32', 'float16'), v
                if k == 'prev_action':
                    self._additional_rnn_inputs[k] = tf.zeros((B, *self._action_shape), dtype=v)
                else:
                    self._additional_rnn_inputs[k] = tf.zeros(B, dtype=v)
            self._squeeze_batch = B == 1

        if 'prev_reward' in self._additional_rnn_inputs:
            self._additional_rnn_inputs['prev_reward'] = tf.convert_to_tensor(
                env_output.reward, self._additional_rnn_inputs['prev_reward'].dtype)

        kwargs.update({
            'state': state or self._state,
            'mask': 1. - env_output.reset,   # mask is applied in LSTM
            **self._additional_rnn_inputs
        })
        return obs, kwargs
    
    def _add_tensor_memory_state_to_terms(self, obs, kwargs, out, evaluation):
        out, self._state = out
        
        if not evaluation:
            if self._store_state:
                out[1].update(kwargs['state']._asdict())
            if 'prev_action' in self._additional_rnn_inputs:
                out[1]['prev_action'] = self._additional_rnn_inputs['prev_action']
            if 'prev_reward' in self._additional_rnn_inputs:
                out[1]['prev_reward'] = self._additional_rnn_inputs['prev_reward']
            if self._squeeze_batch:
                for k, v in out[1].items():
                    if len(out[1][k].shape) > 0 and out[1][k].shape[0] == 1:
                        out[1][k] = tf.squeeze(v, 0)
        
        if 'prev_action' in self._additional_rnn_inputs:
            self._additional_rnn_inputs['prev_action'] = \
                out[0] if isinstance(out, tuple) else out

        return out
    
    def _add_non_tensor_memory_states_to_terms(self, out, kwargs, evaluation):
        """ add additional input terms, which are of non-Tensor type """
        if not evaluation:
            out[1]['mask'] = kwargs['mask']
        return out

    def reset_states(self, state=None):
        if state is None:
            self._state, self._additional_rnn_inputs = None, self._default_additional_rnn_inputs.copy()
        else:
            self._state, self._additional_rnn_inputs = state

    def get_states(self):
        return self._state, self._additional_rnn_inputs


class ActionScheduler:
    def _setup_action_schedule(self, env):
        self._eval_act_eps = tf.convert_to_tensor(
            getattr(self, '_eval_act_eps', 0), tf.float32)
        self._eval_act_temp = tf.convert_to_tensor(
            getattr(self, '_eval_act_temp', .5), tf.float32)
        self._schedule_act_eps = getattr(self, '_schedule_act_eps', False)
        self._schedule_act_temp = getattr(self, '_schedule_act_temp', False)
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
        return self._eval_act_temp if evaluation else self._act_temp

class TargetNetOps:
    """ According to Python's MRO, this class should be positioned 
    before AgentBase to override _sync_nets. Otherwise, you have to 
    explicitly call TargetNetOps._sync_nets(self) """
    def _setup_target_net_sync(self):
        self._to_sync = Every(self._target_update_period) \
            if hasattr(self, '_target_update_period') else None

    @tf.function
    def _sync_nets(self):
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
        return [getattr(self, f'{k}') for k in self.model 
            if f'target_{k}' in self.model]

    def get_target_nets(self):
        return [getattr(self, f'target_{k}') for k in self.model 
            if f'target_{k}' in self.model]
