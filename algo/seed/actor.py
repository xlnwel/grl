import functools
import itertools
import collections
import threading
import numpy as np
import tensorflow as tf
import ray

from utility.ray_setup import cpu_affinity
from utility.utils import Every, config_attr
from utility.rl_utils import compute_act_eps
from utility.timer import Timer
from utility import pkg
from core.tf_config import *
from env.func import create_env
from env.cls import EnvOutput
from replay.func import create_local_buffer
from algo.apex.actor import config_actor, get_learner_class, \
    get_worker_base_class, get_evaluator_class


def get_actor_class(AgentBase):
    """ An Actor is responsible for inference """
    class Actor(AgentBase):
        def __init__(self,
                    actor_id,
                    model_fn,
                    config,
                    model_config,
                    env_config):
            config_actor('Actor', config)

            self._id = actor_id

            self._n_envvecs = env_config['n_envvecs']
            self._n_envs = env_config['n_envs']
            env = create_env(env_config)

            models = model_fn(model_config, env)

            super().__init__(
                name=f'Actor_{actor_id}',
                config=config, 
                models=models,
                dataset=None,
                env=env)

            # number of workers per actor
            self._wpa = self._n_workers // self._n_actors

            self._action_batch = int(
                self._n_workers * self._n_envvecs * self._action_frac)
            if 'act_eps' in config:
                act_eps = compute_act_eps(
                    config['act_eps_type'], 
                    config['act_eps'], 
                    None, 
                    config['n_workers'], 
                    self._n_envvecs * self._n_envs)
                self._act_eps_mapping = act_eps.reshape(
                    config['n_workers'], self._n_envvecs, self._n_envs)
                print(self.name, self._act_eps_mapping)
            else:
                self._act_eps_mapping = None

            # agent's state
            if 'rnn' in self.model:
                self._state_mapping = collections.defaultdict(lambda:
                    self.model.get_initial_state(batch_size=env.n_envs, dtype=self._dtype))
                self._prev_action_mapping = collections.defaultdict(lambda:
                    tf.zeros((env.n_envs, *self._action_shape), self._dtype))
            
            if not hasattr(self, '_pull_names'):
                self._pull_names = [k for k in self.model.keys() if 'target' not in k]
            
            self._to_sync = Every(self.SYNC_PERIOD) if getattr(self, 'SYNC_PERIOD') else None

        def pull_weights(self, learner):
            weights = ray.get(learner.get_weights.remote(self._pull_names))
            self.model.set_weights(weights)

        def set_weights(self, weights):
            self.model.set_weights(weights)

        def __call__(self, wids, eids, env_output):
            if 'rnn' in self.model:
                raw_state = [tf.concat(s, 0)
                    for s in zip(*
                    [tf.nest.flatten(self._state_mapping[(wid, eid)]) 
                        for wid, eid in zip(wids, eids)])]
                self._state = tf.nest.pack_sequence_as(
                    self.model.state_keys, raw_state)
                self._prev_action = tf.concat(
                    [self._prev_action_mapping[(wid, eid)] 
                    for wid, eid in zip(wids, eids)], 0)
                self._prev_reward = env_output.reward

            action, terms = super().__call__(env_output, evaluation=False)

            # store states
            if 'rnn' in self.model:
                for wid, eid, s, a in zip(wids, eids, zip(*self.state), action):
                    self._state_mapping[(wid, eid)] = \
                        tf.nest.pack_sequence_as(self.model.state_keys,
                        ([tf.reshape(x, (-1, tf.shape(x)[-1])) for x in s]))
                    self._prev_action_mapping[(wid, eid)] = a

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
            objs = {workers[wid].env_output.remote(eid): (wid, eid)
                for wid in range(self._wpa) 
                for eid in range(self._n_envvecs)}

            self.env_step = 0
            while True:
                # retrieve ready objs
                with Timer('wait') as wt:
                    ready_objs, _ = ray.wait(
                        list(objs), num_returns=self._action_batch)
                n_ready = len(ready_objs)
                wids, eids = zip(*[objs.pop(i) for i in ready_objs])
                env_output = EnvOutput(*[np.concatenate(v, axis=0) 
                    for v in zip(*ray.get(ready_objs))])
                if self._act_eps_mapping is not None:
                    self._act_eps = np.reshape(
                        self._act_eps_mapping[wids, eids], 
                        (-1) if self._action_shape == () else (-1, 1))
                assert len(wids) == len(eids) == n_ready, \
                    (len(wids), len(eids), n_ready)
                
                actions, terms = self(wids, eids, env_output)
                
                # distribute action and terms
                actions = np.split(actions, n_ready)
                terms = [list(itertools.product([k], np.split(v, n_ready))) 
                    for k, v in terms.items()]
                terms = [dict(v) for v in zip(*terms)]

                # environment step
                objs.update({
                    workers[wid].env_step.remote(eid, a, t): (wid, eid)
                    for wid, eid, a, t in zip(wids, eids, actions, terms)})
                
                self.env_step += n_ready * self._n_envs
                if self._to_sync(self.env_step):
                    self.pull_weights(learner)
                    monitor.record_run_stats.remote(**{
                        'time/wait_env': wt.average(),
                        'n_ready': n_ready
                    })

    return Actor


def get_worker_class():
    """ A Worker is only responsible for resetting&stepping environment """
    WorkerBase = get_worker_base_class(object)
    class Worker(WorkerBase):
        def __init__(self, worker_id, config, env_config, buffer_config):
            config_attr(self, config)
            cpu_affinity(f'Worker_{worker_id}')
            self._id = worker_id

            self._n_envvecs = env_config.pop('n_envvecs')
            env_config.pop('n_workers', None)
            self._envvecs = [
                create_env(env_config, force_envvec=True) 
                for _ in range(self._n_envvecs)]
            
            collect_fn = pkg.import_module(
                'agent', config=config, place=-1).collect
            self._collect = functools.partial(
                collect_fn, env=None, step=None, reset=None)

            buffer_config['force_envvec'] = True
            self._buffs = {eid: create_local_buffer(buffer_config) 
                for eid in range(self._n_envvecs)}

            self._obs = {eid: e.output().obs 
                for eid, e in enumerate(self._envvecs)}
            self._info = collections.defaultdict(list)
        
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
                next_obs=env_output.obs
            )
            kwargs.update(terms)
            self._obs[eid] = env_output.obs

            self._collect(self._buffs[eid], **kwargs)
            if self._buffs[eid].is_full():
                self._send_data(self._replay, self._buffs[eid])
            
            done_env_ids = [i for i, r in enumerate(env_output.reset) if r]
            if np.any(done_env_ids):
                self._info['score'] += self._envvecs[eid].score(done_env_ids)
                self._info['epslen'] += self._envvecs[eid].epslen(done_env_ids)
                if len(self._info['score']) > 10:
                    self._send_episodic_info(self._monitor)

            return env_output
    
    return Worker
