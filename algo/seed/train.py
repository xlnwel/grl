import os
import time
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
    ray.init(num_cpus=os.cpu_count(), num_gpus=1)

    sigint_shutdown_ray()

    default_agent_config.update(agent_config)
    agent_config = default_agent_config

    replay = create_replay_center(replay_config)

    model_fn, Agent = pkg.import_agent(config=agent_config)
    am = pkg.import_module('actor', config=agent_config)
    fm = pkg.import_module('func', config=agent_config)

    monitor = fm.create_monitor(config=agent_config)

    Worker = am.get_worker_class()
    workers = []
    for wid in range(agent_config['n_workers']):
        worker = fm.create_worker(
            Worker=Worker, 
            worker_id=wid, 
            config=agent_config, 
            env_config=env_config, 
            buffer_config=replay_config)
        worker.set_handler.remote(replay=replay)
        worker.set_handler.remote(monitor=monitor)
        workers.append(worker)

    Learner = am.get_learner_class(Agent)
    learner = fm.create_learner(
        Learner=Learner, 
        model_fn=model_fn, 
        replay=replay, 
        config=agent_config, 
        model_config=model_config, 
        env_config=env_config,
        replay_config=replay_config)
    learner.start_learning.remote()

    Evaluator = am.get_evaluator_class(Agent)
    evaluator = fm.create_evaluator(
        Evaluator=Evaluator,
        model_fn=model_fn,
        config=agent_config,
        model_config=model_config,
        env_config=env_config)
    evaluator.run.remote(learner, monitor)
    
    Actor = am.get_actor_class(Agent)
    actors = []
    na = agent_config['n_actors']
    nw = agent_config['n_workers']
    assert nw % na == 0, f"n_workers({nw}) is not divisible by n_actors({na})"
    wpa = nw // na
    for aid in range(agent_config['n_actors']):
        actor = fm.create_actor(
            Actor=Actor, 
            actor_id=aid,
            model_fn=model_fn, 
            config=agent_config, 
            model_config=model_config, 
            env_config=env_config)
        actor.start.remote(workers[aid*wpa:(aid+1)*wpa], learner, monitor)
        actors.append(actor)
    

    elapsed_time = 0
    interval = 10
    while not ray.get(monitor.is_over.remote()):
        time.sleep(interval)
        elapsed_time += interval
        if elapsed_time % agent_config['LOG_PERIOD'] == 0:
            monitor.record_train_stats.remote(learner)

    ray.get(learner.save.remote())

    ray.shutdown()
