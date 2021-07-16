import numpy as np
import tensorflow as tf

from utility.tf_utils import reduce_mean, explained_variance
from utility.rl_loss import n_step_target, huber_loss
from core.mixin import Memory
from core.tf_config import build
from core.base import override
from algo.dqn.base import DQNBase
from replay.local import LocalBuffer


def get_data_format(*, env, replay_config, agent_config, **kwargs):
    sample_size = env.max_episode_steps
    data_format = dict(
        obs=((None, sample_size+1, env.n_agents, *env.obs_shape), tf.float32),
        global_state=((None, sample_size+1, *env.shared_state_shape), tf.float32),
        action_mask=((None, sample_size+1, env.n_agents, env.action_dim), tf.bool),
        episodic_mask=((None, sample_size), tf.float32),
        action=((None, sample_size, env.n_agents, *env.action_shape), tf.int32),
        reward=((None, sample_size), tf.float32), 
        discount=((None, sample_size), tf.float32),
    )
    return data_format


def _add_last_obs(buffer, env):
    # retrieve the last obs and add it to the buffer accordingly
    data = env.prev_obs().copy()
    data.pop('episodic_mask')
    buffer.add(**data)


def _remove_bad_episodes(buffer, idxes):
    # in case that some environments(e.g., smac) throw an error 
    # and restart, we drop bad episodes in the temporary buffer
    if isinstance(buffer, LocalBuffer):
        assert buffer.is_full()
        buffer.reset(idxes)
    else:
        assert buffer.is_local_buffer_full()
        buffer.reset_local_buffer(idxes)


def collect(buffer, env, env_step, reset, next_obs, **data):
    obs = data.pop('obs')
    data.update(obs)
    buffer.add(**data)

    if np.any(reset):
        assert np.all(reset), reset
        _add_last_obs(buffer, env)
        if isinstance(buffer, LocalBuffer):
            assert buffer.is_full(), (buffer._idx, buffer._memlen)
        else:
            assert buffer.is_local_buffer_full()

        # remove bad data from the buffer 
        infos = env.info()
        if isinstance(infos, dict):
            bad_episode_idxes = [0] if infos.get('bad_episode', False) else []
        else:
            bad_episode_idxes = [i for i, info in enumerate(infos) 
                if info.get('bad_episode', False)]

        if bad_episode_idxes:
            print('bad episodes indexes:', bad_episode_idxes)
            _remove_bad_episodes(buffer, bad_episode_idxes)


class Agent(Memory, DQNBase):
    @override(DQNBase)
    def _add_attributes(self, env, dataset):
        super()._add_attributes(env, dataset)
        self._setup_memory_state_record()

        self._sample_size = env.max_episode_steps
        self._n_agents = env.n_agents

    @override(DQNBase)
    def _build_learn(self, env):
        # Explicitly instantiate tf.function to avoid unintended retracing
        TensorSpecs = dict(
            obs=((self._sample_size+1, self._n_agents, *env.obs_shape), env.obs_dtype, 'obs'),
            global_state=((self._sample_size+1, *env.shared_state_shape), env.shared_state_dtype, 'global_state'),
            action_mask=((self._sample_size+1, self._n_agents, env.action_dim), tf.bool, 'action_mask'),
            episodic_mask=((self._sample_size,), tf.float32, 'episodic_mask'),
            action=((self._sample_size, self._n_agents, env.action_dim), tf.float32, 'action'),
            reward=((self._sample_size,), tf.float32, 'reward'),
            discount=((self._sample_size,), tf.float32, 'discount'),
        )
        
        self.learn = build(self._learn, TensorSpecs, batch_size=self._batch_size)

    def _process_input(self, env_output, evaluation):
        mask = self._get_mask(env_output.reset)
        mask = np.tile(np.expand_dims(mask, -1), [1, self._n_agents]).reshape(-1)
        obs, kwargs = self._divide_obs(env_output.obs)
        kwargs = self._add_memory_state_to_kwargs(
            obs, mask=mask, kwargs=kwargs,
            batch_size=obs.shape[0] * self._n_agents)
        kwargs.pop('mask')  # no mask is applied to RNNs

        kwargs['epsilon'] = self._get_eps(evaluation)
        kwargs['temp'] = self._get_temp(evaluation)

        return obs, kwargs

    def _divide_obs(self, obs):
        # TODO: consider using life mask to block gradients from invalid data
        kwargs = {
            'global_state': obs['global_state'].astype(np.float32),
            'action_mask': obs['action_mask'].astype(np.bool),
            'episodic_mask': np.float32(obs['episodic_mask']),
        }
        obs = obs['obs'].astype(np.float32)
        return obs, kwargs

    # @override(DQNBase)
    def _process_output(self, obs, kwargs, out, evaluation):
        out = self._add_tensors_to_terms(obs, kwargs, out, evaluation)
        out = super()._process_output(obs, kwargs, out, evaluation)
        out = self._add_non_tensors_to_terms(obs, kwargs, out, evaluation)
        return out
    
    def _add_non_tensors_to_terms(self, obs, kwargs, out, evaluation):
        return out

    @tf.function
    def _learn(self, obs, global_state, action_mask, 
            action, reward, discount, episodic_mask):
        loss_fn = dict(
            huber=huber_loss, mse=lambda x: x**2)[self._loss_type]
        target, terms = self._compute_target(
            obs, global_state, action_mask, reward, discount)

        obs, _ = tf.split(obs, [self._sample_size, 1], axis=1)
        global_state, _ = tf.split(global_state, [self._sample_size, 1], axis=1)

        with tf.GradientTape() as tape:
            utils = self.model.compute_utils(obs, online=True)
            q = self.model.compute_joint_q(
                utils, global_state, online=True, action=action)
            error = target - q
            loss = reduce_mean(loss_fn(error), mask=episodic_mask)
        tf.debugging.assert_shapes([
            [target, (self._batch_size, self._sample_size)],
            [q, (self._batch_size, self._sample_size)],
        ])

        terms['norm'] = self._value_opt(tape, loss)
        terms.update(dict(
            reward=tf.reduce_mean(reward),
            eps_reward=tf.reduce_sum(reward * episodic_mask, axis=1),
            q=q,
            target=target,
            mask=episodic_mask,
            discount=discount,
            loss=loss,
            explained_variance_q=explained_variance(target, q, mask=episodic_mask),
        ))

        return terms

    def _compute_target(self, obs, global_state, action_mask, 
            reward, discount):
        terms = {}

        # TODO: Consider other improvements, the main challenge here is to tackle joint acitons.
        utils = self.model.compute_utils(obs, online=self._double)
        action = self.q.compute_greedy_action(utils, action_mask, one_hot=True)
        if self._double:
            # we recompute utils using the target networks for DDQN
            utils = self.model.compute_utils(obs, online=False)

        _, next_utils = tf.split(utils, [1, self._sample_size], axis=1)
        _, next_shared_state = tf.split(global_state, [1, self._sample_size], axis=1)
        _, next_action = tf.split(action, [1, self._sample_size], axis=1)
        tf.debugging.assert_shapes([
            [next_utils, (self._batch_size, self._sample_size, self._n_agents, self._action_dim)],
            [next_action, (self._batch_size, self._sample_size, self._n_agents, self._action_dim)],
        ])
        next_q = self.model.compute_joint_q(
            next_utils, next_shared_state, online=False, action=next_action)
        tf.debugging.assert_shapes([
            [reward, (self._batch_size, self._sample_size)],
            [discount, (self._batch_size, self._sample_size)],
            [next_q, (self._batch_size, self._sample_size)],
        ])
        
        target = n_step_target(reward, next_q, discount, self._gamma)

        return target, terms
