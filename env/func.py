import numpy as np

from env.cls import Env, EnvVec, make_env


def create_env(config, env_fn=None, force_envvec=False):
    """ Creates an Env/EnvVec from config """
    config = config.copy()
    env_fn = env_fn or make_env
    if config.get('n_workers', 1) <= 1:
        EnvType = EnvVec if force_envvec or config.get('n_envs', 1) > 1 else Env
        env = EnvType(config, env_fn)
    else:
        from env.ray_env import RayEnvVec
        EnvType = EnvVec if config.get('n_envs', 1) > 1 else Env
        env = RayEnvVec(EnvType, config, env_fn)

    return env


if __name__ == '__main__':
    def run(config):
        env = create_env(config)
        env.reset()
        st = time.time()
        for _ in range(10000):
            a = env.random_action()
            _, _, d, _ = env.step(a)
            if np.any(d == 0):
                idx = [i for i, dd in enumerate(d) if dd == 0]
                # print(idx)
                env.reset(idx)
        return time.time() - st
    import ray
    # performance test
    config = dict(
        name='procgen_coinrun',
        frame_stack=4,
        life_done=False,
        np_obs=False,
        seed=0,
    )
    import time
    ray.init()
    config['n_envs'] = 4
    config['n_workers'] = 8
    duration = run(config)
    print(f'RayEnvVec({config["n_workers"]}, {config["n_envs"]})', duration)
    
    ray.shutdown()
    config['n_envs'] = config['n_workers'] * config['n_envs']
    config['n_workers'] = 1
    duration = run(config)
    print(f'EnvVec({config["n_workers"]}, {config["n_envs"]})', duration)
