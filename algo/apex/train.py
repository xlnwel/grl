import os
import time
import tensorflow as tf
import ray

from utility.ray_setup import sigint_shutdown_ray
from utility import pkg
from replay.func import create_replay_center


default_agent_config = {    
    'MAX_STEPS': 1e8,
    'LOG_PERIOD': 60,
    'N_UPDATES': 1000,
    'SYNC_PERIOD': 1000,
    'RECORD_PERIOD': 100,
    'N_EVALUATION': 10,

    # distributed algo params
    'n_learner_cpus': 1,
    'n_learner_gpus': 1,
    'n_workers': 5,
    'n_worker_cpus': 1,
    'n_worker_gpus': 0,
}

def main(env_config, model_config, agent_config, replay_config):
    gpus = tf.config.list_physical_devices('GPU')
    ray.init(num_cpus=os.cpu_count(), num_gpus=len(gpus))
    
    sigint_shutdown_ray()

    default_agent_config.update(agent_config)
    agent_config = default_agent_config

    replay = create_replay_center(replay_config) \
        if agent_config.get('use_central_buffer', True) else None

    model_fn, Agent = pkg.import_agent(config=agent_config)
    am = pkg.import_module('actor', config=agent_config)
    fm = pkg.import_module('func', config=agent_config)

    monitor = fm.create_monitor(config=agent_config)

    Learner = am.get_learner_class(Agent)
    learner = fm.create_learner(
        Learner=Learner, 
        model_fn=model_fn, 
        replay=replay, 
        config=agent_config, 
        model_config=model_config, 
        env_config=env_config,
        replay_config=replay_config)
    ray.get(monitor.sync_env_train_steps.remote(learner))

    Worker = am.get_worker_class(Agent)
    workers = []
    for wid in range(agent_config['n_workers']):
        worker = fm.create_worker(
            Worker=Worker, 
            worker_id=wid, 
            model_fn=model_fn,
            config=agent_config, 
            model_config=model_config, 
            env_config=env_config, 
            buffer_config=replay_config)
        worker.prefill_replay.remote(
            learner if replay is None else replay)
        workers.append(worker)

    if agent_config.get('has_evaluator', True):
        Evaluator = am.get_evaluator_class(Agent)
        evaluator = fm.create_evaluator(
            Evaluator=Evaluator,
            model_fn=model_fn,
            config=agent_config,
            model_config=model_config,
            env_config=env_config)
        evaluator.run.remote(learner, monitor)

    learner.start_learning.remote()
    [w.run.remote(learner, replay, monitor) for w in workers]
    
    elapsed_time = 0
    interval = 10
    while not ray.get(monitor.is_over.remote()):
        time.sleep(interval)
        elapsed_time += interval
        if elapsed_time % agent_config['LOG_PERIOD'] == 0:
            monitor.record_train_stats.remote(learner)
    ray.get(monitor.record_train_stats.remote(learner))

    ray.shutdown()
