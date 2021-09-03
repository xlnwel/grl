from abc import ABC, abstractmethod
import os
import logging
import cloudpickle

from utility.display import pwc
from utility.utils import Every
from utility.tf_utils import tensor2numpy
from utility.timer import Timer
from utility.schedule import TFPiecewiseSchedule
from core.log import *
from core.optimizer import Optimizer
from core.checkpoint import *
from core.decorator import override, agent_config, step_track
from core.mixin import RMS

logger = logging.getLogger(__name__)


class AgentImpl(ABC):
    def get_env_train_steps(self):
        return self.env_step, self.train_step

    """ Restore & save """
    def restore(self):
        """ Restore the latest parameter recorded by ckpt_manager """
        restore(self._ckpt_manager, self._ckpt, self._ckpt_path, self._model_name)
        self.env_step = self._env_step.numpy()
        self.train_step = self._train_step.numpy()

    def save(self, print_terminal_info=False):
        """ Save Model """
        self._env_step.assign(self.env_step)
        self._train_step.assign(self.train_step)
        save(self._ckpt_manager, print_terminal_info=print_terminal_info)

    """ Logging """
    def save_config(self, config):
        """ Save config.yaml """
        save_config(self._root_dir, self._model_name, config)

    def log(self, step, prefix=None, print_terminal_info=True, **kwargs):
        """ Log stored data to disk and tensorboard """
        log(self._logger, self._writer, self._model_name, prefix=prefix, 
            step=step, print_terminal_info=print_terminal_info, **kwargs)

    def log_stats(self, stats, print_terminal_info=True):
        """ Save stats to disk """
        log_stats(self._logger, stats, print_terminal_info=print_terminal_info)

    def set_summary_step(self, step):
        """ Sets tensorboard step """
        set_summary_step(step)

    def scalar_summary(self, stats, prefix=None, step=None):
        """ Adds scalar summary to tensorboard """
        scalar_summary(self._writer, stats, prefix=prefix, step=step)

    def histogram_summary(self, stats, prefix=None, step=None):
        """ Adds histogram summary to tensorboard """
        histogram_summary(self._writer, stats, prefix=prefix, step=step)

    def graph_summary(self, sum_type, *args, step=None):
        """ Adds graph summary to tensorboard
        Args:
            sum_type str: either "video" or "image"
            args: Args passed to summary function defined in utility.graph,
                of which the first must be a str to specify the tag in Tensorboard
        """
        assert isinstance(args[0], str), f'args[0] is expected to be a name string, but got "{args[0]}"'
        args = list(args)
        args[0] = f'{self.name}/{args[0]}'
        graph_summary(self._writer, sum_type, args, step=step)

    def store(self, **stats):
        """ Stores stats to self._logger """
        store(self._logger, **stats)

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

        self._add_attributes(env, dataset)
        models = self._construct_optimizers()
        # we assume all models not starting with 'target'
        # are trainable.
        all_models = set([v for k, v in self.model.items() 
            if not k.startswith('target')])
        assert set(models) == all_models, f'{models}\n{all_models}'
        self._build_learn(env)
        self._sync_nets()
    
    def _add_attributes(self, env, dataset):
        """ Adds attributes to Agent """
        self._sample_timer = Timer('sample')
        self._learn_timer = Timer('train')

        self._return_stats = getattr(self, '_return_stats', False)

        self.RECORD = getattr(self, 'RECORD', False)
        self.N_EVAL_EPISODES = getattr(self, 'N_EVAL_EPISODES', 1)

        # intervals between calling self._summary
        self._to_summary = Every(self.LOG_PERIOD, self.LOG_PERIOD)

    def _construct_optimizers(self):
        """ Constructs optimizers for training
        Returns models to check if any model components is missing
        """
        self._optimizer = self._construct_opt()
        models = [v for k, v in self.model.items() 
            if not k.startswith('target')]
        logger.info(f'{self.name} model: {models}')
        return models

    def _construct_opt(self, models=None, lr=None, opt=None, l2_reg=None,
            weight_decay=None, clip_norm=None, opt_kwargs={}):
        """ Constructs an optimizer """
        if lr is None:
            if getattr(self, '_schedule_lr', False):
                assert isinstance(self._lr, (list, tuple)), self._lr
                lr = TFPiecewiseSchedule(self._lr)
            else:
                lr = self._lr
        opt = opt or getattr(self, '_optimizer', 'adam')
        l2_reg = l2_reg or getattr(self, '_l2_reg', None)
        weight_decay = weight_decay or getattr(self, '_weight_decay', None)
        clip_norm = clip_norm or getattr(self, '_clip_norm', None)
        if opt_kwargs == {}:
            opt_kwargs = getattr(self, '_opt_kwargs', {})
        models = models or [v for k, v in self.model.items() 
            if not k.startswith('target')]
        opt = Optimizer(
            opt, models, lr, 
            l2_reg=l2_reg,
            weight_decay=weight_decay,
            clip_norm=clip_norm,
            **opt_kwargs
        )
        return opt

    @abstractmethod
    def _build_learn(self, env):
        """ Builds @tf.function for model training """
        raise NotImplementedError

    def _sync_nets(self):
        pass

    def reset_states(self, states=None):
        pass

    def get_states(self):
        pass

    def _summary(self, data, terms):
        """ Adds non-scalar summaries here """
        pass 

    """ Call """
    def __call__(self, env_output=(), evaluation=False, return_eval_stats=False):
        """ Call the agent to interact with the environment
        Args:
            env_output namedtuple: (obs, reward, discount, reset)
            evaluation bool: evaluation mode or not
            return_eval_stats bool: if return evaluation stats
        Return:
            action and terms.
        """
        env_output = self._reshape_env_output(env_output)
        obs, kwargs = self._process_input(env_output, evaluation)
        kwargs['evaluation'] = kwargs.get('evaluation', evaluation)
        out = self._compute_action(
            obs, **kwargs, 
            return_stats=self._return_stats,
            return_eval_stats=return_eval_stats)
        out = self._process_output(obs, kwargs, out, evaluation)

        return out

    def _reshape_env_output(self, env_output):
        """ Adds the batch dimension if it's missing """
        if np.shape(env_output.reward) == ():
            env_output = tf.nest.map_structure(
                lambda x: np.expand_dims(x, 0), env_output)
        return env_output

    def _process_input(self, env_output, evaluation):
        """ Does necessary pre-process and produces inputs to model
        Args:
            env_output tuple: (obs, reward, discount, reset)
            evaluation bool: evaluation mode or not
        Returns: 
            obs: Pre-processed observations
            kwargs, dict: kwargs necessary for model to compute actions 
        """
        return env_output.obs, {}
    
    def _compute_action(self, obs, **kwargs):
        return self.model.action(obs, **kwargs)
        
    def _process_output(self, obs, kwargs, out, evaluation):
        """Post-processes output
        Args:
            obs: Pre-processed observations
            kwargs, dict: kwargs necessary to model  
            out (action, terms): Model output
        Returns:
            out: results returned to the environment
        """
        return tensor2numpy(out)

    """ Train """
    @step_track
    def learn_log(self, step):
        n = self._sample_learn()
        self._store_buffer_stats()

        return n

    def _sample_learn(self):
        raise NotImplementedError
    
    def _store_buffer_stats(self):
        pass

    @abstractmethod
    def _learn(self):
        raise NotImplementedError


class RMSAgentBase(RMS, AgentBase):
    """ AgentBase with Running Mean Std(RMS) """
    @override(AgentBase)
    def _add_attributes(self, env, dataset):
        super()._add_attributes(env, dataset)

        self._setup_rms_stats()

    def _process_input(self, env_output, evaluation):
        self._process_obs(env_output.obs, update_rms=not evaluation)
        return env_output.obs, {}

    # @override(AgentBase)
    def _process_output(self, obs, kwargs, out, evaluation):
        out = super()._process_output(obs, kwargs, out, evaluation)        
        if self._normalize_obs and not evaluation:
            terms = out[1]
            terms['obs'] = obs
        return out

    @step_track
    def learn_log(self, step):
        n = self._sample_learn()
        self._store_buffer_stats()
        self._store_rms_stats()

        return n
    
    def _store_rms_stats(self):
        obs_rms, rew_rms = self.get_rms_stats()
        if rew_rms:
            self.store(**{
                'train/reward_rms_mean': rew_rms.mean,
                'train/reward_rms_var': rew_rms.var
            })
        if obs_rms:
            for k, v in obs_rms.items():
                self.store(**{
                    f'train/{k}_rms_mean': v.mean,
                    f'train/{k}_rms_var': v.var,
                })

    @override(AgentBase)
    def restore(self):
        """ Restore the RMS and the model """
        if os.path.exists(self._rms_path):
            with open(self._rms_path, 'rb') as f:
                self._obs_rms, self._reward_rms, self._return = cloudpickle.load(f)
                logger.info(f'rms stats are restored from {self._rms_path}')
        assert self._reward_rms.axis == self._reward_normalized_axis, (self._reward_rms.axis, self._reward_normalized_axis)
        for v in self._obs_rms.values():
            assert v.axis == self._obs_normalized_axis, (v.axis, self._obs_normalized_axis)
        super().restore()

    @override(AgentBase)
    def save(self, print_terminal_info=False):
        """ Save the RMS and the model """
        with open(self._rms_path, 'wb') as f:
            cloudpickle.dump((self._obs_rms, self._reward_rms, self._return), f)
        super().save(print_terminal_info=print_terminal_info)
