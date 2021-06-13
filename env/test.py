from env.func import create_env
import time
import numpy as np
import ray

from env import procgen

if __name__ == '__main__':
    config = dict(
        name='procgen_coinrun',
        n_envs=10,
    )
    def make_env(config):
        env = procgen.make_procgen_env(config)
        return env

    ray.init()
    env = create_env(config, make_env)

    print('Env', env)
    
    def run(env):
        st = time.time()
        for _ in range(10000):
            a = env.random_action()
            env.step(a)
        return time.time() - st
    
    print("Ray env:", run(env))

