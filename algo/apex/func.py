import logging
import tensorflow as tf
import ray

from replay.func import create_local_buffer
from algo.apex.monitor import Monitor

logger = logging.getLogger(__name__)


def create_monitor(config):
    config = config.copy()
    RayMonitor = ray.remote(Monitor)
    monitor = RayMonitor.remote(config=config)

    return monitor

def create_learner(Learner, model_fn, replay, config, model_config, env_config, replay_config):
    config = config.copy()
    model_config = model_config.copy()
    env_config = env_config.copy()
    replay_config = replay_config.copy()
    
    n_cpus = config.setdefault('n_learner_cpus', 3)
    config['writer'] = False
    config['logger'] = False

    if tf.config.list_physical_devices('GPU'):
        n_gpus = config.setdefault('n_learner_gpus', 1)
        RayLearner = ray.remote(num_cpus=n_cpus, num_gpus=n_gpus)(Learner)
    else:
        RayLearner = ray.remote(num_cpus=n_cpus)(Learner)

    learner = RayLearner.remote( 
        model_fn=model_fn, 
        replay=replay,
        config=config, 
        model_config=model_config, 
        env_config=env_config,
        replay_config=replay_config)
    ray.get(learner.save_config.remote(dict(
        env=env_config,
        model=model_config,
        agent=config,
        replay=replay_config
    )))

    return learner


def create_worker(
        Worker, worker_id, model_fn, 
        config, model_config, 
        env_config, buffer_config):
    config = config.copy()
    model_config = model_config.copy()
    env_config = env_config.copy()
    buffer_config = buffer_config.copy()

    buffer_fn = create_local_buffer

    if 'seed' in env_config:
        env_config['seed'] += worker_id * 100
    
    config['display_var'] = False
    config['save_code'] = False
    config['logger'] = False
    config['writer'] = False

    n_cpus = config.get('n_worker_cpus', 1)
    n_gpus = config.get('n_worker_gpus', 0)
    RayWorker = ray.remote(num_cpus=n_cpus, num_gpus=n_gpus)(Worker)
    worker = RayWorker.remote(
        worker_id=worker_id, 
        config=config, 
        model_config=model_config, 
        env_config=env_config, 
        buffer_config=buffer_config,
        model_fn=model_fn, 
        buffer_fn=buffer_fn)

    return worker

def create_evaluator(Evaluator, model_fn, config, model_config, env_config):
    config = config.copy()
    model_config = model_config.copy()
    env_config = env_config.copy()

    config['display_var'] = False
    config['save_code'] = False
    config['logger'] = False
    config['writer'] = False

    config['schedule_act_eps'] = False
    config['schedule_act_temp'] = False

    if 'seed' in env_config:
        env_config['seed'] += 999
    env_config['n_workers'] = 1
    env_config['n_envs'] = 4 if 'procgen' in env_config['name'] else 1

    RayEvaluator = ray.remote(num_cpus=1)(Evaluator)
    evaluator = RayEvaluator.remote(
        config=config,
        model_config=model_config,
        env_config=env_config,
        model_fn=model_fn)

    return evaluator
