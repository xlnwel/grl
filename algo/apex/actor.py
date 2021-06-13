import time
import threading
import functools
import collections
import numpy as np
import ray

from core.tf_config import *
from utility.utils import Every
from utility.timer import Timer
from utility.ray_setup import cpu_affinity, get_num_cpus
from utility.run import Runner, evaluate, RunMode
from utility import pkg
from env.func import create_env
from core.dataset import create_dataset


def config_actor(name, config):
    cpu_affinity(name)
    silence_tf_logs()
    num_cpus = get_num_cpus()
    configure_threads(num_cpus, num_cpus)
    use_gpu = configure_gpu()
    if not use_gpu and 'precision' in config:
        config['precision'] = 32
    configure_precision(config.get('precision', 32))

def get_learner_base_class(AgentBase):
    class LearnerBase(AgentBase):
        """ Only implements minimal functionality for learners """
        def start_learning(self):
            self._learning_thread = threading.Thread(
                target=self._learning, daemon=True)
            self._learning_thread.start()
            
        def _learning(self):
            # waits for enough data to learn
            while not self.dataset.good_to_learn():
                time.sleep(1)
            print(f'{self.name} starts learning...')

            while True:
                self.learn_log()
                
        def get_weights(self, name=None):
            return self.model.get_weights(name=name)

        def get_stats(self):
            """ retrieve traning stats for the monotor to record """
            return self.train_step, super().get_stats()

    return LearnerBase

def get_learner_class(AgentBase):
    LearnerBase = get_learner_base_class(AgentBase)
    class Learner(LearnerBase):
        def __init__(self,
                    model_fn,
                    replay,
                    config, 
                    model_config,
                    env_config,
                    replay_config):
            config_actor('Learner', config)

            env = create_env(env_config)

            model = model_fn(config=model_config, env=env)

            am = pkg.import_module('agent', config=config, place=-1)
            data_format = am.get_data_format(
                env=env, replay_config=replay_config, 
                agent_config=config, model=model)
            dataset = create_dataset(
                replay, 
                env, 
                data_format=data_format, 
                use_ray=True)
            
            super().__init__(
                name='Learner',
                config=config, 
                models=model,
                dataset=dataset,
                env=env,
            )

    return Learner


def get_worker_base_class(AgentBase):
    class WorkerBase(AgentBase):
        """ Only implements minimal functionality for workers """
        def _send_data(self, replay, buffer=None, data=None):
            """ Sends data to replay and resets buffer """
            if buffer is None:
                buffer = self.buffer
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
                pass
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
            config_actor(f'Worker_{worker_id}', config)
            self._id = worker_id

            self.env = create_env(env_config)

            buffer_config['n_envs'] = self.env.n_envs
            if 'seqlen' not in buffer_config:
                buffer_config['seqlen'] = self.env.max_episode_steps
            self.buffer = buffer_fn(buffer_config)

            models = model_fn( 
                config=model_config, 
                env=self.env)

            super().__init__(
                name=f'Worker_{worker_id}',
                config=config,
                models=models,
                dataset=self.buffer,
                env=self.env)
            
            # setup runner
            import importlib
            em = importlib.import_module(
                f'env.{env_config["name"].split("_")[0]}')
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
            self._collect = functools.partial(collect_fn, self.buffer)

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
                weights = self._pull_weights(learner)
                self.model.set_weights(weights)
                self._run(replay)
                self._send_episodic_info(monitor)

        def store(self, **kwargs):
            for k, v in kwargs.items():    
                if isinstance(v, (int, float)):
                    self._info[k].append(v)
                else:
                    self._info[k] += list(v)

        def _pull_weights(self, learner):
            return ray.get(learner.get_weights.remote(name=self._pull_names))

        def _run(self, replay):
            def collect(*args, **kwargs):
                self._collect(*args, **kwargs)
                if self.buffer.is_full():
                    self._send_data(replay)

            start_step = self.runner.step
            with Timer('run') as rt:
                self.env_step = self.runner.run(step_fn=collect)
            self._info['time/run'] = rt.average()

            return self.env_step - start_step
    
    return Worker


def get_evaluator_class(AgentBase):
    class Evaluator(AgentBase):
        """ Initialization """
        def __init__(self, 
                    *,
                    config,
                    name='Evaluator',
                    model_config,
                    env_config,
                    model_fn):
            config_actor(name, config)

            env_config.pop('reward_clip', False)
            self.env = env = create_env(env_config)

            model = model_fn(
                config=model_config, 
                env=env)
            
            super().__init__(
                name=name,
                config=config, 
                models=model,
                dataset=None,
                env=env,
            )
        
            # the names of network modules that should be in sync with the learner
            if not hasattr(self, '_pull_names'):
                self._pull_names = [k for k in self.model.keys() if 'target' not in k]
            
            # used for recording evaluator side info 
            self._info = collections.defaultdict(list)

        """ Evaluator Methods """
        def run(self, learner, monitor):
            step = 0
            if getattr(self, 'RECORD_PERIOD', False):
                # how often to record videos
                to_record = Every(self.RECORD_PERIOD)
            else:
                to_record = lambda x: False 

            while True:
                step += 1
                weights = self._pull_weights(learner)
                self.model.set_weights(weights)
                self._run(record=to_record(step))
                self._send_episodic_info(monitor)

        def _pull_weights(self, learner):
            return ray.get(learner.get_weights.remote(name=self._pull_names))

        def _run(self, record):
            score, epslen, video = evaluate(self.env, self, 
                record=record, n=self.N_EVALUATION)
            self.store(score, epslen, video)

        def store(self, score, epslen, video):
            self._info['eval_score'] += score
            self._info['eval_epslen'] += epslen
            if video is not None:
                self._info['video'] = video

        def _send_episodic_info(self, monitor):
            if self._info:
                monitor.record_episodic_info.remote(**self._info)
                self._info.clear()
    
    return Evaluator
