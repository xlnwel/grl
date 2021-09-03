from algo.apex.func import disable_info_logging, ray_remote_config, \
    create_learner, create_monitor, create_evaluator


def create_worker(
        Worker, worker_id, config, 
        env_config, buffer_config):
    config = config.copy()
    env_config = env_config.copy()
    buffer_config = buffer_config.copy()

    config = disable_info_logging(config)

    if 'seed' in env_config:
        env_config['seed'] += worker_id * 100
    # avoids additional workers created by RayEnvVec
    env_config['n_workers'] = 1

    ray_config = ray_remote_config(config, 'worker', default_cpus=1)
    RayWorker = Worker.as_remote(**ray_config)
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

    config = disable_info_logging(config)

    ray_config = ray_remote_config(config, 'actor')
    RayActor = Actor.as_remote(**ray_config)
    actor = RayActor.remote(
        actor_id=actor_id,
        model_fn=model_fn, 
        config=config, 
        model_config=model_config, 
        env_config=env_config)
    
    return actor
