import os
import time
import functools
import numpy as np
import tensorflow as tf
from tensorflow.keras.mixed_precision import global_policy
import ray

from core.tf_config import configure_gpu, configure_precision, silence_tf_logs
from utility.ray_setup import sigint_shutdown_ray
from utility.graph import video_summary
from utility.utils import Every, TempStore
from utility.run import Runner, evaluate
from utility import pkg
from env.func import create_env
from replay.func import create_replay
from core.dataset import Dataset, process_with_env
from algo.dreamer.env import make_env


def train(agent, env, eval_env, replay):
    def collect(env, step, reset, **kwargs):
        if np.any(reset):
            if env.n_envs == 1:
                episodes = env.prev_episode()
            else:
                episodes = [e.prev_episode() for e, d in zip(env.envs, reset) if d]
            replay.merge(episodes)
    _, step = replay.count_episodes()
    step = max(agent.env_step, step)

    runner = Runner(env, agent, step=step)
    while not replay.good_to_learn():
        step = runner.run(action_selector=env.random_action, step_fn=collect)
        
    to_log = Every(agent.LOG_PERIOD)
    to_eval = Every(agent.EVAL_PERIOD)
    print('Training starts...')
    while step < int(agent.MAX_STEPS):
        start_step = step
        start_t = time.time()
        agent.learn_log(step)
        step = runner.run(step_fn=collect, nsteps=agent.TRAIN_PERIOD)
        duration = time.time() - start_t
        agent.store(
            fps=(step-start_step) / duration,
            tps=agent.N_UPDATES / duration)

        if to_eval(step):
            with TempStore(agent.get_states, agent.reset_states):
                score, epslen, video = evaluate(eval_env, agent, 
                    record=agent.RECORD, size=(64, 64))
                if agent.RECORD:
                    video_summary(f'{agent.name}/sim', video, step=step)
                agent.store(eval_score=score, eval_epslen=epslen)
            
        if to_log(step):
            agent.log(step)
            agent.save()

def main(env_config, model_config, agent_config, replay_config):
    silence_tf_logs()
    configure_gpu()
    configure_precision(agent_config['precision'])

    use_ray = env_config.get('n_workers', 0) > 1
    if use_ray:
        ray.init()
        sigint_shutdown_ray()

    env = create_env(env_config, make_env, force_envvec=True)
    eval_env_config = env_config.copy()
    eval_env_config['n_envs'] = 1
    eval_env_config['n_workers'] = 1
    eval_env_config['log_episode'] = False
    if 'reward_hack' in eval_env_config:
        del eval_env_config['reward_hack']
    eval_env = create_env(eval_env_config, make_env)

    replay_config['dir'] = agent_config['root_dir'].replace('logs', 'data')
    replay = create_replay(replay_config)
    replay.load_data()
    dtype = global_policy().compute_dtype
    data_format = pkg.import_module('agent', config=agent_config).get_data_format(
        env=env, 
        batch_size=agent_config['batch_size'], 
        sample_size=agent_config['sample_size'], 
        dtype=dtype)
    process = functools.partial(process_with_env, 
        env=env, 
        obs_range=[-.5, .5], 
        one_hot_action=True, 
        dtype=dtype)
    dataset = Dataset(replay, data_format, process)

    create_model, Agent = pkg.import_agent(config=agent_config)
    models = create_model(model_config, env)

    agent = Agent(
        name=env.name,
        config=agent_config,
        models=models, 
        dataset=dataset,
        env=env)

    agent.save_config(dict(
        env=env_config,
        model=model_config,
        agent=agent_config,
        replay=replay_config
    ))

    train(agent, env, eval_env, replay)
