import tensorflow as tf
import ray

from algo.apex.func import create_learner, create_monitor, create_evaluator


def create_worker(
        Worker, worker_id, config, 
        env_config, buffer_config):
    config = config.copy()
    env_config = env_config.copy()
    buffer_config = buffer_config.copy()

    if 'seed' in env_config:
        env_config['seed'] += worker_id * 100
    
    config['display_var'] = False
    config['save_code'] = False
    config['logger'] = False
    config['writer'] = False

    RayWorker = ray.remote(num_cpus=1)(Worker)
    worker = RayWorker.remote(
        worker_id=worker_id, 
        config=config, 
        env_config=env_config, 
        buffer_config=buffer_config)

    return worker

def create_actor(Actor, actor_id, model_fn, config, model_config, env_config):
    config = config.copy()
    model_config = model_config.copy()
    env_config = env_config.copy()

    config['display_var'] = False
    config['save_code'] = False
    config['logger'] = False
    config['writer'] = False

    if tf.config.list_physical_devices('GPU'):
        n_gpus = config.setdefault('n_actor_gpus', .1)
        RayActor = ray.remote(num_cpus=1, num_gpus=n_gpus)(Actor)
    else:
        RayActor = ray.remote(num_cpus=2)(Actor)

    actor = RayActor.remote(
        actor_id=actor_id,
        model_fn=model_fn, 
        config=config, 
        model_config=model_config, 
        env_config=env_config)
    
    return actor
