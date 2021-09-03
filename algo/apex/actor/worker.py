import collections
import functools
import numpy as np
import ray

from utility import pkg
from utility.ray_setup import config_actor
from utility.run import Runner, RunMode
from utility.timer import Timer
from env.func import create_env
from algo.apex.actor.actor import get_actor_base_class


def get_worker_base_class(AgentBase):
    ActorBase = get_actor_base_class(AgentBase)
    class WorkerBase(ActorBase):
        """ Only implements minimal functionality for workers """
        def _send_data(self, replay, buffer=None, data=None):
            """ Sends data to replay and resets buffer """
            if buffer is None:
                buffer = self.dataset
            if data is None:
                data = buffer.sample()
            
            if data is None:
                print(f"Worker {self._id}: no data is retrieved")
                return

            if isinstance(data, dict):
                # regular dqn families
                if self._worker_side_prioritization:
                    data['priority'] = self._compute_priorities(**data)
                # drops q-values as we don't need them for training
                data.pop('q', None)
                data.pop('next_q', None)
            elif isinstance(data, (list, tuple)):
                # recurrent dqn families
                q = [d.pop('q', None) for d in data]
                next_q = [d.pop('next_q', None) for d in data]
                if self._worker_side_prioritization:
                    kwargs = {
                        'reward': np.array([d['reward'] for d in data]),
                        'discount': np.array([d['discount'] for d in data]),
                        'steps': np.array([d['steps'] for d in data]),
                        'q': np.array(q),
                        'next_q': np.array(next_q)
                    }
                    priority = self._compute_priorities(**kwargs)
                    for d, p in zip(data, priority):
                        d['priority'] = p
            else:
                raise ValueError(f'Unknown data of type: {type(data)}')
            
            replay.merge.remote(data)
            buffer.reset()

        def _send_episodic_info(self, monitor):
            """ Sends episodic info to monitor for bookkeeping """
            if self._info:
                monitor.record_episodic_info.remote(self._id, **self._info)
                self._info.clear()

        def _compute_priorities(self, reward, discount, steps, q, next_q, **kwargs):
            """ Computes priorities for prioritized replay"""
            target_q = reward + discount * self._gamma**steps * next_q
            priority = np.abs(target_q - q)
            if priority.ndim == 2:
                priority = self._per_eta*priority.max(axis=1) \
                    + (1-self._per_eta)*priority.mean(axis=1)
            assert priority.ndim == 1, priority.shape
            priority += self._per_epsilon
            priority **= self._per_alpha

            return priority

    return WorkerBase


def get_worker_class(AgentBase):
    WorkerBase = get_worker_base_class(AgentBase)
    class Worker(WorkerBase):
        def __init__(self,
                    *,
                    worker_id,
                    config,
                    model_config, 
                    env_config, 
                    buffer_config,
                    model_fn,
                    buffer_fn):
            self._id = worker_id
            name = f'Worker_{self._id}'

            config_actor(name, config)
            
            # avoids additional workers created by RayEnvVec
            env_config['n_workers'] = 1
            self.env = create_env(env_config)

            buffer_config['n_envs'] = self.env.n_envs
            if buffer_config.get('seqlen', 0) == 0:
                buffer_config['seqlen'] = self.env.max_episode_steps
            buffer = buffer_fn(buffer_config)

            models = model_fn(config=model_config, env=self.env)

            super().__init__(
                name=name,
                config=config,
                models=models,
                dataset=buffer,
                env=self.env)
            
            # setups runner
            em = pkg.import_module(self.env.name.split("_")[0], pkg='env')
            info_func = em.info_func if hasattr(em, 'info_func') else None
            self._run_mode = getattr(self, '_run_mode', RunMode.NSTEPS)
            assert self._run_mode in [RunMode.NSTEPS, RunMode.TRAJ]
            self.runner = Runner(
                self.env, self, 
                nsteps=self.SYNC_PERIOD if self._run_mode == RunMode.NSTEPS else None,
                run_mode=self._run_mode,
                record_envs=getattr(self, '_record_envs', None),
                info_func=info_func)

            # worker side prioritization
            self._worker_side_prioritization = getattr(
                self, '_worker_side_prioritization', False)
            self._return_stats = self._worker_side_prioritization \
                or buffer_config.get('max_steps', 0) > buffer_config.get('n_steps', 1)

            # setups self._collect using <collect> function from the algorithm module
            collect_fn = pkg.import_module('agent', algo=self._algorithm, place=-1).collect
            self._collect = functools.partial(collect_fn, buffer)

            # the names of network modules that should be in sync with the learner
            if not hasattr(self, '_pull_names'):
                self._pull_names = [k for k in self.model.keys() if 'target' not in k]

            # used for recording worker side info 
            self._info = collections.defaultdict(list)

        """ Worker Methods """
        def prefill_replay(self, replay):
            while not ray.get(replay.good_to_learn.remote()):
                self._run(replay)

        def run(self, learner, replay, monitor):
            while True:
                self.pull_weights(learner)
                self._run(replay)
                self._send_episodic_info(monitor)

        def store(self, **kwargs):
            for k, v in kwargs.items():    
                if isinstance(v, (int, float)):
                    self._info[k].append(v)
                else:
                    self._info[k] += list(v)

        def _run(self, replay):
            def collect(*args, **kwargs):
                self._collect(*args, **kwargs)
                if self.dataset.is_full():
                    self._send_data(replay)

            start_step = self.runner.step
            with Timer('run') as rt:
                self.env_step = self.runner.run(step_fn=collect)
            self._info['time/run'] = rt.average()

            return self.env_step - start_step
    
    return Worker