import os
import time
import ray

from utility.ray_setup import sigint_shutdown_ray
from utility import pkg
from replay.func import create_replay_center


def main(env_config, model_config, agent_config, replay_config):
    ray.init(num_cpus=os.cpu_count(), num_gpus=1)

    sigint_shutdown_ray()

    replay = create_replay_center(replay_config) \
        if agent_config.get('use_central_buffer', True) else None

    model_fn, Agent = pkg.import_agent(config=agent_config)
    am = pkg.import_module('actor', config=agent_config)
    fm = pkg.import_module('func', config=agent_config)

    # create the monitor
    monitor = fm.create_monitor(config=agent_config)

    # create the learner
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

    # create workers
    Worker = am.get_worker_class()
    workers = []
    for wid in range(agent_config['n_workers']):
        worker = fm.create_worker(
            Worker=Worker, 
            worker_id=wid, 
            config=agent_config, 
            env_config=env_config, 
            buffer_config=replay_config)
        worker.set_handler.remote(
            replay=learner if replay is None else replay)
        worker.set_handler.remote(monitor=monitor)
        workers.append(worker)

    # create the evaluator
    if agent_config.get('has_evaluator', True):
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
    learner.start_learning.remote()
    
    elapsed_time = 0
    interval = 10
    # put the main thead into sleep 
    # the monitor records training stats once in a while
    while not ray.get(monitor.is_over.remote()):
        time.sleep(interval)
        elapsed_time += interval
        if elapsed_time % agent_config['LOG_PERIOD'] == 0:
            monitor.record_train_stats.remote(learner)

    ray.get(learner.save.remote())

    ray.shutdown()
