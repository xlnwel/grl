import functools
import numpy as np

from core.tf_config import *
from utility.utils import Every, TempStore
from utility.graph import video_summary
from utility.timer import Timer
from utility.run import Runner, evaluate, RunMode
from utility import pkg
from env.func import create_env
from replay.func import create_replay
from core.dataset import create_dataset


def train(agent, env, eval_env, replay):
    collect_fn = pkg.import_module('agent', algo=agent.name).collect
    collect = functools.partial(collect_fn, replay)
    
    em = pkg.import_module(env.name.split("_")[0], pkg='env')
    info_func = em.info_func if hasattr(em, 'info_func') else None

    env_step = agent.env_step
    runner = Runner(env, agent, step=env_step,
        run_mode=RunMode.TRAJ, info_func=info_func)
    agent.TRAIN_PERIOD = env.max_episode_steps
    while not replay.good_to_learn():
        env_step = runner.run(step_fn=collect)
        replay.finish_episodes()

    to_eval = Every(agent.EVAL_PERIOD)
    to_log = Every(agent.LOG_PERIOD, agent.LOG_PERIOD)
    to_eval = Every(agent.EVAL_PERIOD)
    to_record = Every(agent.EVAL_PERIOD*10)
    rt = Timer('run')
    tt = Timer('train')
    # et = Timer('eval')
    lt = Timer('log')
    print('Training starts...')
    while env_step <= int(agent.MAX_STEPS):
        with rt:
            env_step = runner.run(step_fn=collect)
        assert np.all(runner.env_output.reset), runner.env_output.reset
        replay.finish_episodes()

        with tt:
            agent.learn_log(env_step)

        # if to_eval(env_step):
        #     with TempStore(agent.get_states, agent.reset_states):
        #         with et:
        #             record = agent.RECORD and to_record(env_step)
        #             eval_score, eval_epslen, video = evaluate(
        #                 eval_env, agent, n=agent.N_EVAL_EPISODES, 
        #                 record=agent.RECORD, size=(64, 64))
        #             if record:
        #                 video_summary(f'{agent.name}/sim', video, step=env_step)
        #             agent.store(
        #                 eval_score=eval_score, 
        #                 eval_epslen=eval_epslen)

        if to_log(env_step):
            with lt:
                fps = rt.average() * agent.TRAIN_PERIOD
                tps = tt.average() * agent.N_UPDATES
                
                agent.store(
                    env_step=agent.env_step,
                    train_step=agent.train_step,
                    fps=fps, 
                    tps=tps,
                )
                agent.store(**{
                    'train_step': agent.train_step,
                    'time/run': rt.total(), 
                    'time/train': tt.total(),
                    # 'time/eval': et.total(),
                    'time/log': lt.total(),
                    'time/run_mean': rt.average(), 
                    'time/train_mean': tt.average(),
                    # 'time/eval_mean': et.average(),
                    'time/log_mean': lt.average(),
                })
                agent.log(env_step)
                agent.save()

def main(env_config, model_config, agent_config, replay_config):
    silence_tf_logs()
    configure_gpu()
    configure_precision(agent_config.get('precision', 32))

    use_ray = env_config.get('n_workers', 1) > 1
    if use_ray:
        import ray
        from utility.ray_setup import sigint_shutdown_ray
        ray.init()
        sigint_shutdown_ray()

    env = create_env(env_config)
    # eval_env_config = env_config.copy()
    # eval_env_config['n_workers'] = 1
    # eval_env_config['n_envs'] = 1
    # reward_key = [k for k in eval_env_config.keys() if 'reward' in k]
    # [eval_env_config.pop(k) for k in reward_key]
    # eval_env = create_env(eval_env_config, force_envvec=True)
    eval_env = None

    agent_config['N_UPDATES'] *= env_config['n_workers'] * env_config['n_envs']
    create_model, Agent = pkg.import_agent(config=agent_config)
    models = create_model(model_config, env)

    n_workers = env_config.get('n_workers', 1)
    n_envs = env_config.get('n_envs', 1)
    replay_config['n_envs'] = n_workers * n_envs
    replay_config['seqlen'] = env.max_episode_steps
    if getattr(models, 'state_keys', ()):
        replay_config['state_keys'] = list(models.state_keys)
    replay = create_replay(replay_config)
    replay.load_data()

    am = pkg.import_module('agent', config=agent_config)
    data_format = am.get_data_format(
        env=env, replay_config=replay_config, 
        agent_config=agent_config, model=models)
    dataset = create_dataset(replay, env, data_format=data_format)
    
    agent = Agent(
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

    if use_ray:
        ray.shutdown()
