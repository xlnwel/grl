import logging
import numpy as np
import tensorflow as tf

from utility.tf_utils import log_softmax
from utility.schedule import TFPiecewiseSchedule
from core.tf_config import build
from core.base import AgentBase
from core.mixin import ActionScheduler, TargetNetOps
from core.decorator import override, step_track


logger = logging.getLogger(__name__)

def get_data_format(*, env, replay_config, **kwargs):
    is_per = replay_config['replay_type'].endswith('per')
    n_steps = replay_config['n_steps']
    obs_dtype = tf.uint8 if len(env.obs_shape) == 3 else tf.float32
    action_dtype = tf.int32 if env.is_action_discrete else tf.float32
    data_format = dict(
        obs=((None, *env.obs_shape), obs_dtype),
        action=((None, *env.action_shape), action_dtype),
        reward=((None, ), tf.float32), 
        next_obs=((None, *env.obs_shape), obs_dtype),
        discount=((None, ), tf.float32),
    )
    if is_per:
        data_format['idxes'] = ((None, ), tf.int32)
        if replay_config.get('use_is_ratio', is_per):
            data_format['IS_ratio'] = ((None, ), tf.float32)
    if n_steps > 1:
        data_format['steps'] = ((None, ), tf.float32)

    return data_format

def collect(replay, env, env_step, reset, next_obs, **kwargs):
    # if reset:
    #     # we reset noisy every episode. Theoretically, 
    #     # this follows the guide of deep exploration.
    #     # More importantly, it saves time!
    #     if hasattr(agent, 'reset_noisy'):
    #         agent.reset_noisy()
    if isinstance(reset, np.ndarray):
        if np.any(reset):
            for i, r in enumerate(reset):
                if r:
                    next_obs[i] = env.prev_obs(i)[0]
    elif reset:
        next_obs = env.prev_obs()
    kwargs['next_obs'] = next_obs
    replay.add(**kwargs)


class DQNBase(TargetNetOps, ActionScheduler, AgentBase):
    """ Initialization """
    @override(AgentBase)
    def _add_attributes(self, env, dataset):
        super()._add_attributes(env, dataset)

        self.MUNCHAUSEN = getattr(self, 'MUNCHAUSEN', False)
        self._probabilistic_regularization = getattr(
            self, '_probabilistic_regularization', None)

        self._is_per = False if dataset is None else dataset.name().endswith('per')
        self._double = getattr(self, '_double', False)
        logger.info(f'Prioritized replay: {self._is_per}')
        logger.info(f'Double Q-learning: {self._double}')

        self._return_stats = getattr(self, '_return_stats', False)

        self._setup_action_schedule(env)
        self._setup_target_net_sync()

    @override(AgentBase)
    def _construct_optimizers(self):
        actor_models = []
        if [k for k in self.model.keys() if 'actor' in k]:
            actor_models = [v for k, v in self.model.items() 
                if 'actor' in k and 'target' not in k]
            logger.info(f'Actor model: {actor_models}')
            self._actor_opt = super()._construct_opt(
                actor_models, lr=self._actor_lr)

        value_models = [v for k, v in self.model.items() \
            if k != 'temperature' and 'actor' not in k
            and 'target' not in k]
        logger.info(f'Value model: {value_models}')
        self._value_opt = super()._construct_opt(
            value_models, lr=self._value_lr)

        temp_models = []
        if hasattr(self, 'temperature'):
            temp_models = [self.temperature]
            logger.info(f'Temperature model: {temp_models}')
            if self.temperature.is_trainable():
                self._temp_opt = super()._construct_opt(
                    temp_models, lr=self._temp_lr)
                if isinstance(getattr(
                    self, '_target_entropy_coef', None), (list, tuple)):
                    self._target_entropy_coef = TFPiecewiseSchedule(self._target_entropy_coef)
        
        return actor_models + value_models + temp_models

    @override(AgentBase)
    def _build_learn(self, env):
        # Explicitly instantiate tf.function to initialize variables
        TensorSpecs = dict(
            obs=(env.obs_shape, env.obs_dtype, 'obs'),
            action=((env.action_dim,), tf.float32, 'action'),
            reward=((), tf.float32, 'reward'),
            next_obs=(env.obs_shape, env.obs_dtype, 'next_obs'),
            discount=((), tf.float32, 'discount'),
        )
        if self._is_per and getattr(self, '_use_is_ratio', self._is_per):
            TensorSpecs['IS_ratio'] = ((), tf.float32, 'IS_ratio')
        if self._n_steps > 1:
            TensorSpecs['steps'] = ((), tf.float32, 'steps')
        self.learn = build(self._learn, TensorSpecs, batch_size=self._batch_size)

    """ Call """
    def _process_input(self, env_output, evaluation):
        obs, kwargs = super()._process_input(env_output, evaluation)
        kwargs['epsilon'] = self._get_eps(evaluation)
        kwargs['temp'] = self._get_temp(evaluation)
        return obs, kwargs

    @step_track
    def learn_log(self, step):
        for _ in range(self.N_UPDATES):
            with self._sample_timer:
                data = self.dataset.sample()

            if self._is_per:
                idxes = data.pop('idxes').numpy()

            with self._learn_timer:
                terms = self.learn(**data)

            if self._to_sync is not None:
                if self._to_sync(self.train_step):
                    self._sync_nets()
            else:
                self._update_nets()

            terms = {f'train/{k}': v.numpy() for k, v in terms.items()}
            if self._is_per:
                self.dataset.update_priorities(terms['train/priority'], idxes)

            self.store(**terms)

        if self._to_summary(step):
            self._summary(data, terms)
        
        self.store(**{
            'time/sample_mean': self._sample_timer.average(),
            'time/learn_mean': self._learn_timer.average()
        })

        return self.N_UPDATES
    
    def _compute_reward(self, reward, qs, action):
        terms = {}
        if self.MUNCHAUSEN:
            logpi = log_softmax(qs, self._tau, axis=-1)
            logpi_a = tf.reduce_sum(logpi * action, axis=-1)
            logpi_a = tf.clip_by_value(logpi_a, self._clip_logpi_min, 0)
            terms['reward'] = reward
            terms['logpi_a'] = logpi_a
            reward = reward + self._alpha * logpi_a
        return reward, terms

    def _compute_priority(self, priority):
        """ p = (p + 𝝐)**𝛼 """
        priority += self._per_epsilon
        priority **= self._per_alpha
        return priority

    def reset_noisy(self):
        pass
