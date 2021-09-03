import functools
import itertools
import collections
import threading
import numpy as np
import psutil
import tensorflow as tf
import ray

from utility.ray_setup import cpu_affinity
from utility.utils import Every, config_attr, batch_dicts
from utility.timer import Timer
from utility.typing import EnvOutput
from utility import pkg
from core.tf_config import *
from env.func import create_env
from replay.func import create_local_buffer
from algo.apex.actor import config_actor, get_learner_class, \
    get_worker_base_class, get_evaluator_class, \
    get_actor_base_class


def get_actor_class(AgentBase):
    """ An Actor is responsible for inference only """
    ActorBase = get_actor_base_class(AgentBase)
    class Actor(ActorBase):
        def __init__(self,
                    actor_id,
                    model_fn,
                    config,
                    model_config,
                    env_config):
            self._id = actor_id
            name = f'Actor_{self._id}'
            config_actor(name, config)

            psutil.Process().nice(config.get('default_nice', 0)+2)

            # avoids additional workers created by RayEnvVec
            env_config['n_workers'] = 1
            # create env to get state&action spaces
            self._n_envvecs = env_config['n_envvecs']
            self._n_envs = env_config['n_envs']
            env = create_env(env_config)

            models = model_fn(config=model_config, env=env)

            super().__init__(
                name=name,
                config=config, 
                models=models,
                dataset=None,
                env=env)

            # number of workers per actor
            self._wpa = self._n_workers // self._n_actors

            # number of env(vec) instances for each inference pass
            self._action_batch = int(
                self._wpa * self._n_envvecs * self._action_frac)

            # agent's state
            if 'rnn' in self.model:
                self._state_mapping = collections.defaultdict(lambda:
                    self.model.get_initial_state(
                        batch_size=env.n_envs, dtype=self._dtype))
                self._prev_action_mapping = collections.defaultdict(lambda:
                    tf.zeros((env.n_envs, *self._action_shape), self._dtype))

            if not hasattr(self, '_pull_names'):
                self._pull_names = [
                    k for k in self.model.keys() if 'target' not in k]

            self._to_sync = Every(self.SYNC_PERIOD) \
                if hasattr(self, 'SYNC_PERIOD') else lambda _: None

            env.close()

        def __call__(self, wids, eids, env_output):
            if 'rnn' in self.model:
                raw_state = [tf.concat(s, 0)
                    for s in zip(*
                    [tf.nest.flatten(self._state_mapping[(wid, eid)]) 
                        for wid, eid in zip(wids, eids)])]
                self._state = tf.nest.pack_sequence_as(
                    self.model.state_keys, raw_state)
                # self._prev_action = tf.concat(
                #     [self._prev_action_mapping[(wid, eid)] 
                #     for wid, eid in zip(wids, eids)], 0)
                # print(self._prev_action)

            action, terms = super().__call__(env_output, evaluation=False)

            # store states
            if 'rnn' in self.model:
                state = zip(*tf.nest.map_structure(
                    lambda s: tf.split(s, self._action_batch), self._state))
                # action = zip(*tf.nest.map_structure(
                #     lambda a: tf.split(a, self._action_batch), action))
                for wid, eid, s, a in zip(wids, eids, state, action):
                    self._state_mapping[(wid, eid)] = \
                        tf.nest.pack_sequence_as(self.model.state_keys,
                        ([tf.reshape(x, (-1, tf.shape(x)[-1])) for x in s]))
                    # self._prev_action_mapping[(wid, eid)] = a

            return action, terms

        def start(self, workers, learner, monitor):
            self._act_thread = threading.Thread(
                target=self._act_loop, 
                args=[workers, learner, monitor], 
                daemon=True)
            # run the act loop in a background thread to provide the
            # flexibility to allow the learner to push weights
            self._act_thread.start()

        def _act_loop(self, workers, learner, monitor):
            # retrieve the last env_output
            objs = {workers[wid].env_output.remote(eid): (wid, eid)
                for wid in range(self._wpa) 
                for eid in range(self._n_envvecs)}

            self.env_step = 0
            q_size = []
            while True:
                q_size, fw = self._fetch_weights(q_size)

                # retrieve ready objs
                with Timer(f'{self.name} wait') as wt:
                    ready_objs, _ = ray.wait(
                        list(objs), num_returns=self._action_batch)
                assert self._action_batch == len(ready_objs), \
                    (self._action_batch, len(ready_objs))

                # prepare data
                wids, eids = zip(*[objs.pop(i) for i in ready_objs])
                assert len(wids) == len(eids) == self._action_batch, \
                    (len(wids), len(eids), self._action_batch)
                env_output = list(zip(*ray.get(ready_objs)))
                assert len(env_output) == 4, env_output
                if isinstance(env_output[0][0], dict):
                    # if obs is a dict
                    env_output = EnvOutput(*[
                        batch_dicts(x, np.concatenate)
                        if isinstance(x[0], dict) else np.concatenate(x, 0)
                        for x in env_output])
                else:
                    env_output = EnvOutput(*[
                        np.concatenate(x, axis=0) 
                        for x in env_output])
                # do inference
                with Timer(f'{self.name} call') as ct:
                    actions, terms = self(wids, eids, env_output)

                # distribute action and terms
                actions = np.split(actions, self._action_batch)
                terms = [list(itertools.product([k], np.split(v, self._action_batch))) 
                    for k, v in terms.items()]
                terms = [dict(v) for v in zip(*terms)]

                # step environments
                objs.update({
                    workers[wid].env_step.remote(eid, a, t): (wid, eid)
                    for wid, eid, a, t in zip(wids, eids, actions, terms)})

                self.env_step += self._action_batch * self._n_envs

                if self._to_sync(self.env_step):
                    monitor.record_run_stats.remote(
                        worker_name=self._id,
                        **{
                        'time/wait_env': wt.average(),
                        'time/agent_call': ct.average(),
                        'time/fetch_weights': fw.average(),
                        'n_ready': self._action_batch,
                        'param_queue_size': np.mean(q_size)
                    })
                    q_size = []

        def _fetch_weights(self, q_size):
            obs_rms, train_step_weights = None, None

            q_size.append(self._param_queue.qsize())

            with Timer(f'{self.name} fetch weights') as ft:
                while not self._param_queue.empty():
                    if self._normalize_obs:
                        obs_rms, train_step_weights = \
                            self._param_queue.get(block=False)
                    else:
                        train_step_weights = self._param_queue.get(block=False)

                if train_step_weights is not None:
                    if self._normalize_obs:
                        self.set_rms_stats(obs_rms=ray.get(obs_rms))
                        self.set_train_step_weights(
                            *ray.get(train_step_weights))
                    else:
                        self.set_train_step_weights(
                            *train_step_weights)
            
            return q_size, ft

    return Actor


def get_worker_class():
    """ A Worker is only responsible for resetting&stepping environment """
    WorkerBase = get_worker_base_class(object)
    class Worker(WorkerBase):
        def __init__(self, worker_id, config, env_config, buffer_config):
            config_attr(self, config)
            cpu_affinity(f'Worker_{worker_id}')

            psutil.Process().nice(config.get('default_nice', 0)+10)
            
            self._id = worker_id
            self.name = f'Worker_{self._id}'
            
            # avoids additional workers created by RayEnvVec
            env_config['n_workers'] = 1
            self._n_envvecs, self._envvecs = self._create_envvec(env_config)
            
            collect_fn = pkg.import_module(
                'agent', config=config, place=-1).collect
            self._collect = functools.partial(
                collect_fn, env_step=None)

            self._buffs = self._create_buffer(
                buffer_config, self._n_envvecs)

            self._obs = {eid: e.output().obs 
                for eid, e in enumerate(self._envvecs)}
            self._info = collections.defaultdict(list)

        def random_warmup(self, steps):
            for e in self._envvecs:
                for _ in range(steps):
                    e.step(e.random_action())

        def set_handler(self, **kwargs):
            config_attr(self, kwargs)
        
        def env_output(self, eid):
            return self._envvecs[eid].output()

        def env_step(self, eid, action, terms):
            env_output = self._envvecs[eid].step(action)
            kwargs = dict(
                obs=self._obs[eid], 
                action=action, 
                reward=env_output.reward,
                discount=env_output.discount, 
                next_obs=env_output.obs)
            kwargs.update(terms)
            self._obs[eid] = env_output.obs

            self._collect(
                self._buffs[eid], self._envvecs[eid], env_step=None,
                reset=env_output.reset, **kwargs)
            if self._buffs[eid].is_full():
                self._send_data(self._replay, self._buffs[eid])

            done_env_ids = [i for i, r in enumerate(env_output.reset) if r]
            if done_env_ids:
                self._info['score'] += self._envvecs[eid].score(done_env_ids)
                self._info['epslen'] += self._envvecs[eid].epslen(done_env_ids)
                if len(self._info['score']) > 10:
                    self._send_episodic_info(self._monitor)

            return env_output

        def _send_data(self, replay, buffer):
            data = buffer.sample()
            replay.merge.remote(data)
            buffer.reset()

        def _create_envvec(self, env_config):
            n_envvecs = env_config.pop('n_envvecs')
            env_config.pop('n_workers', None)
            envvecs = [
                create_env(env_config, force_envvec=True) 
                for _ in range(n_envvecs)]
            return n_envvecs, envvecs

        def _create_buffer(self, buffer_config, n_envvecs):
            buffer_config['force_envvec'] = True
            return {eid: create_local_buffer(buffer_config) 
                for eid in range(n_envvecs)}

    return Worker
