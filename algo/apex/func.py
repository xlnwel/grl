import logging
import tensorflow as tf
import ray

from replay.func import create_local_buffer
from algo.apex.actor import Monitor

logger = logging.getLogger(__name__)


def disable_info_logging(config, 
        display_var=False, save_code=False,
        logger=False, writer=False):
    config['display_var'] = display_var
    config['save_code'] = save_code
    config['logger'] = logger
    config['writer'] = writer

    return config

def ray_remote_config(config, name, 
        default_cpus=None, default_gpus=None):
    ray_config = {}
    if config.setdefault(f'n_{name}_cpus', default_cpus):
        ray_config['num_cpus'] = config[f'n_{name}_cpus']
    if name.lower() == 'learner':
        # for learner, we set the default number of gpus 
        # to the maximum number of gpus available if 
        # default_gpus is not specified
        n_gpus = config.setdefault(f'n_{name}_gpus', 
            default_gpus or len(tf.config.list_physical_devices('GPU')))
    else:
        n_gpus = config.setdefault(f'n_{name}_gpus', default_gpus)
    if n_gpus:
        ray_config['num_gpus'] = n_gpus
    return ray_config

def create_monitor(config):
    config = config.copy()
    RayMonitor = Monitor.as_remote()
    monitor = RayMonitor.remote(config=config)

    return monitor

def create_learner(
        Learner, model_fn, replay, config, 
        model_config, env_config, replay_config):
    config = config.copy()
    model_config = model_config.copy()
    env_config = env_config.copy()
    replay_config = replay_config.copy()
    
    config = disable_info_logging(config, display_var=True)
    # avoids additional workers created by RayEnvVec
    env_config['n_workers'] = 1

    ray_config = ray_remote_config(config, 'learner')
    RayLearner = Learner.as_remote(**ray_config)
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

    config = disable_info_logging(config)

    buffer_fn = create_local_buffer

    if 'seed' in env_config:
        env_config['seed'] += worker_id * 100
    # avoids additional workers created by RayEnvVec
    env_config['n_workers'] = 1

    ray_config = ray_remote_config(config, 'worker')
    RayWorker = Worker.as_remote(**ray_config)
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

    config = disable_info_logging(config)

    config['schedule_act_eps'] = False
    config['schedule_act_temp'] = False

    if 'seed' in env_config:
        env_config['seed'] += 999
    env_config['n_workers'] = 1
    env_config['n_envs'] = env_config.pop('n_eval_envs', 4)

    RayEvaluator = Evaluator.as_remote(num_cpus=1)
    evaluator = RayEvaluator.remote(
        config=config,
        model_config=model_config,
        env_config=env_config,
        model_fn=model_fn)

    return evaluator
