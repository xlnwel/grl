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

def get_base_learner_class(AgentBase):
    class LearnerBase(AgentBase):
        def start_learning(self):
            self._learning_thread = threading.Thread(
                target=self._learning, daemon=True)
            self._learning_thread.start()
            
        def _learning(self):
            while not self.dataset.good_to_learn():
                time.sleep(1)
            print(f'{self.name} starts learning...')

            while True:
                self.learn_log()
                
        def get_weights(self, name=None):
            return self.model.get_weights(name=name)

        def get_stats(self):
            return self.train_step, super().get_stats()

    return LearnerBase

def get_learner_class(AgentBase):
    LearnerBase = get_base_learner_class(AgentBase)
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


def get_base_worker_class(AgentBase):
    class WorkerBase(AgentBase):
        def _send_data(self, replay, buffer=None):
            buffer = buffer or self.buffer
            data = buffer.sample()

            if isinstance(data, dict):
                # regular dqn families
                if self._worker_side_prioritization:
                    data['priority'] = self._compute_priorities(**data)
                data.pop('q', None)
                data.pop('next_q', None)
            elif isinstance(data, (list, tuple)):
                # recurrent dqn families
                pass
            else:
                raise ValueError(f'Unknown data of type: {type(data)}')
            replay.merge.remote(data)
            buffer.reset()

        def _send_episode_info(self, monitor):
            if self._info:
                monitor.record_episode_info.remote(self._id, **self._info)
                self._info.clear()

        def _compute_priorities(self, reward, discount, steps, q, next_q, **kwargs):
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
    WorkerBase = get_base_worker_class(AgentBase)
    class Worker(WorkerBase):
        """ Initialization """
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
            self.n_envs = self.env.n_envs

            buffer_config['n_envs'] = self.n_envs
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
            
            self._run_mode = getattr(self, '_run_mode', RunMode.NSTEPS)
            assert self._run_mode in [RunMode.NSTEPS, RunMode.TRAJ]
            self.runner = Runner(
                self.env, self, 
                nsteps=self.SYNC_PERIOD if self._run_mode == RunMode.NSTEPS else None,
                run_mode=self._run_mode)

            self._worker_side_prioritization = getattr(self, '_worker_side_prioritization', False)
            self._return_stats = self._worker_side_prioritization \
                or buffer_config.get('max_steps', 0) > buffer_config.get('n_steps', 1)
            collect_fn = pkg.import_module('agent', algo=self._algorithm, place=-1).collect
            self._collect = functools.partial(collect_fn, self.buffer)

            if not hasattr(self, '_pull_names'):
                self._pull_names = [k for k in self.model.keys() if 'target' not in k]
            self._info = collections.defaultdict(list)
            
        """ Call """
        def _process_input(self, obs, evaluation, env_output):
            obs, kwargs = super()._process_input(obs, evaluation, env_output)
            return obs, kwargs

        """ Worker Methods """
        def prefill_replay(self, replay):
            while not ray.get(replay.good_to_learn.remote()):
                self._run(replay)

        def run(self, learner, replay, monitor):
            while True:
                weights = self._pull_weights(learner)
                self.model.set_weights(weights)
                self._run(replay)
                self._send_episode_info(monitor)

        def store(self, score, epslen):
            if isinstance(score, (int, float)):
                self._info['score'].append(score)
                self._info['epslen'].append(epslen)
            else:
                self._info['score'] += list(score)
                self._info['epslen'] += list(epslen)

        def _pull_weights(self, learner):
            return ray.get(learner.get_weights.remote(name=self._pull_names))

        def _run(self, replay):
            def collect(*args, **kwargs):
                self._collect(*args, **kwargs)
                if self.buffer.is_full():
                    self._send_data(replay)
            start_step = self.runner.step
            
            with Timer('run') as rt:
                end_step = self.runner.run(step_fn=collect)
            self._info[f'time/run_{self._id}'] = rt.average()

            return end_step - start_step
    
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
            self.n_envs = self.env.n_envs

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
        
            if not hasattr(self, '_pull_names'):
                self._pull_names = [k for k in self.model.keys() if 'target' not in k]
            self._info = collections.defaultdict(list)

        """ Evaluator Methods """
        def run(self, learner, monitor):
            step = 0
            if getattr(self, 'RECORD_PERIOD', False):
                to_record = Every(self.RECORD_PERIOD)
            else:
                to_record = lambda x: False 
            while True:
                step += 1
                weights = self._pull_weights(learner)
                self.model.set_weights(weights)
                self._run(record=to_record(step))
                self._send_episode_info(monitor)

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

        def _send_episode_info(self, monitor):
            if self._info:
                monitor.record_episode_info.remote(**self._info)
                self._info.clear()
    
    return Evaluator
