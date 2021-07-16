import numpy as np
import ray

from core.tf_config import *
from core.mixin import RMS
from core.dataset import create_dataset
from core.decorator import step_track
from utility import pkg
from algo.ppo.buffer import Buffer
from algo.seed.actor import \
    get_actor_class as get_actor_base_class, \
    get_learner_class as get_learner_base_class, \
    get_worker_class as get_worker_base_class, \
    get_evaluator_class
from .buffer import APGBuffer, LocalBuffer


def get_actor_class(AgentBase):
    ActorBase = get_actor_base_class(AgentBase)
    class Actor(ActorBase):
        def __init__(self, actor_id, model_fn, config, 
                model_config, env_config):
            super().__init__(actor_id, model_fn, config, model_config, env_config)

        def _process_output(self, obs, kwargs, out, evaluation):
            out = super()._process_output(obs, kwargs, out, evaluation)
            out[1]['train_step'] = np.ones(obs.shape[0]) * self.train_step
            return out

        def start(self, workers, learner, monitor):
            super().start(workers, learner, monitor)
            self._workers = workers

    return Actor


def get_learner_class(AgentBase):
    LearnerBase = get_learner_base_class(AgentBase)
    class Learner(LearnerBase):
        def _add_attributes(self, env, dataset):
            super()._add_attributes(env, dataset)

            if not hasattr(self, '_push_names'):
                self._push_names = [
                    k for k in self.model.keys() if 'target' not in k]

        def push_weights(self):
            train_step_weights = self.get_train_step_weights(
                name=self._push_names)
            train_step_weights_id = ray.put(train_step_weights)
            if self._normalize_obs:
                obs_rms, _ = self.get_rms_stats()
                obs_rms_id = ray.put(obs_rms)
                for q in self._param_queues:
                    q.put((obs_rms_id, train_step_weights_id))
            else:
                for q in self._param_queues:
                    q.put(train_step_weights_id)

        def _create_dataset(self, replay, model, env, config, replay_config):
            buff_config = replay_config.copy()
            buff_config['state_keys'] = model.state_keys
            buff_config['n_agents'] = getattr(env, 'n_agents', 1)
            buff_config['n_envs'] = config['n_trajs'] * buff_config['n_agents']
            buffer = Buffer(buff_config)
            self.replay = APGBuffer(replay_config, buffer, self)
            if replay_config.get('use_dataset', True):
                am = pkg.import_module('agent', config=config, place=-1)
                data_format = am.get_data_format(
                    env=env, batch_size=self.replay.batch_size, 
                    sample_size=config.get('sample_size', None),
                    store_state=config.get('store_state', True),
                    state_size=model.state_size)
                dataset = create_dataset(self.replay, env, 
                    data_format=data_format, one_hot_action=False)
            else:
                dataset = self.replay
            return dataset

        @step_track
        def learn_log(self, step):
            self._sample_learn()
            self._store_buffer_stats()
            self._store_rms_stats()

            return 0

        def _after_train_step(self):
            self.train_step += 1
            self.replay.set_train_step(self.train_step)
            self.push_weights()

        def _store_buffer_stats(self):
            # we do not store buffer stats because 
            # it may be empty due to the use of tf.data
            self.store(**self.replay.get_async_stats())

    return Learner


def get_worker_class():
    """ A Worker is only responsible for resetting&stepping environment """
    WorkerBase = get_worker_base_class()
    class Worker(WorkerBase, RMS):
        def __init__(self, worker_id, config, env_config, buffer_config):
            super().__init__(worker_id, config, env_config, buffer_config)

            self._setup_rms_stats()
            self._counters = {f'env_step_{i}': 0 
                for i in range(self._n_envvecs)}

        def env_step(self, eid, action, terms):
            self._counters[f'env_step_{eid}'] += 1

            # TODO: consider using a queue here
            env_output = self._envvecs[eid].step(action)
            kwargs = dict(
                obs=self._obs[eid], 
                action=action, 
                reward=env_output.reward,
                discount=env_output.discount, 
                next_obs=env_output.obs)
            kwargs.update(terms)
            self._obs[eid] = env_output.obs

            if self._buffs[eid].is_full():
                # Adds the last value/obs to buffer for gae computation. 
                if self._buffs[eid]._adv_type == 'vtrace':
                    self._buffs[eid].finish(
                        last_obs=env_output.obs, 
                        last_mask=1-env_output.reset)
                else:
                    self._buffs[eid].finish(last_value=terms['value'])
                self._send_data(self._replay, self._buffs[eid])

            self._collect(
                self._buffs[eid], self._envvecs[eid], env_step=None,
                reset=env_output.reset, **kwargs)

            done_env_ids = [i for i, r in enumerate(env_output.reset) if np.all(r)]
            if done_env_ids:
                self._info['score'] += self._envvecs[eid].score(done_env_ids)
                self._info['epslen'] += self._envvecs[eid].epslen(done_env_ids)
                if len(self._info['score']) > 10:
                    self._send_episodic_info(self._monitor)

            return env_output

        def random_warmup(self, steps):
            rewards = []
            discounts = []

            for e in self._envvecs:
                for _ in range(steps // e.n_envs):
                    env_output = e.step(e.random_action())
                    if e.is_multiagent:
                        env_output = tf.nest.map_structure(np.concatenate, env_output)
                        life_mask = env_output.obs.get('life_mask')
                    else:
                        life_mask = None
                    self._process_obs(env_output.obs, mask=life_mask)
                    rewards.append(env_output.reward)
                    discounts.append(env_output.discount)

            rewards = np.swapaxes(rewards, 0, 1)
            discounts = np.swapaxes(discounts, 0, 1)
            self.update_reward_rms(rewards, discounts)

            return self.get_rms_stats()

        def _create_buffer(self, buffer_config, n_envvecs):
            buffer_config['force_envvec'] = True
            return {eid: LocalBuffer(buffer_config) 
                for eid in range(n_envvecs)}

        def _send_episodic_info(self, monitor):
            """ Sends episodic info to monitor for bookkeeping """
            if self._info:
                monitor.record_episodic_info.remote(
                    self._id, **self._info, **self._counters)
                self._info.clear()

    return Worker
