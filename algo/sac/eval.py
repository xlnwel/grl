import numpy as np

from core.tf_config import *
from utility.display import pwc
from utility.ray_setup import sigint_shutdown_ray
from utility.run import evaluate
from utility import pkg
from utility.graph import save_video
from env.func import create_env
from replay.func import create_replay


def main(env_config, model_config, agent_config, replay_config,
        n, record=False, size=(128, 128), video_len=1000, 
        force_envvec=False, fps=30, save=False):
    silence_tf_logs()
    configure_gpu()
    configure_precision(agent_config.get('precision', 32))

    use_ray = env_config.get('n_workers', 0) > 1
    if use_ray:
        import ray
        ray.init()
        sigint_shutdown_ray()
    
    algo_name = agent_config['algorithm']
    env_name = env_config['name']

    if record:
        env_config['log_episode'] = True
        env_config['n_workers'] = env_config['n_envs'] = 1

    env = create_env(env_config, force_envvec=force_envvec)

    create_model, Agent = pkg.import_agent(config=agent_config)

    models = create_model(model_config, env)
    
    agent = Agent( 
        config=agent_config, 
        models=models, 
        dataset=None, 
        env=env)

    if save:
        n_workers = env_config.get('n_workers', 1)
        n_envs = env_config.get('n_envs', 1)
        replay_config['n_envs'] = n_workers * n_envs
        replay_config['replay_type'] = 'uniform'
        replay_config['dir'] = f'data/{agent.name.lower()}-{env.name.lower()}'
        replay_config['n_steps'] = 1
        replay_config['save'] = True
        replay_config['save_temp'] = True
        replay_config['capacity'] = int(1e6)
        replay_config['has_next_obs'] = True
        replay = create_replay(replay_config)
        def collect(obs, action, reward, discount, next_obs, logpi, **kwargs):
            replay.add(obs=obs, action=action, reward=reward, 
                discount=discount, next_obs=next_obs, logpi=logpi)
    else:
        def collect(**kwargs):
            pass

    if n < env.n_envs:
        n = env.n_envs
    scores, epslens, video = evaluate(env, agent, n, 
        record=record, size=size, 
        video_len=video_len, step_fn=collect)
    pwc(f'After running {n} episodes',
        f'Score: {np.mean(scores):.3g}\tEpslen: {np.mean(epslens):.3g}', color='cyan')
    
    if save:
        replay.save()

    if record:
        save_video(f'{algo_name}-{env_name}', video, fps=fps)
    if use_ray:
        ray.shutdown()
