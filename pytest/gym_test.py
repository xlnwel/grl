import random
import numpy as np
import ray

from env.func import create_env
from utility.timer import Timer


default_config = dict(
    name='BipedalWalkerHardcore-v3',
    auto_reset=True,
    frame_stack=4,
    frame_skip=4,
    seed=0
)

class TestClass:
    def test_Env(self):
        for name in ['atari_pong', 'atari_breakout', 'BipedalWalkerHardcore-v3']:
            for life_done in [False, True]:
                for _ in range(2):
                    config = default_config.copy()
                    config['name'] = name
                    config['n_envs'] = 1
                    config['life_done'] = life_done
                    env = create_env(config)
                    cr = 0
                    n = 0
                    re = 0
                    for i in range(2000):
                        a = env.random_action()
                        s, r, d, re = env.step(a)
                        cr += r
                        if r != 0:
                            print(name, i, r, cr, env.score())
                        n += env.info().get('frame_skip', 1)
                        np.testing.assert_equal(cr, env.score())
                        np.testing.assert_equal(n, env.epslen())
                        if env.info().get('game_over'):
                            cr = 0
                            n = 0

    def test_EnVec(self):
        for name in ['atari_pong', 'atari_breakout', 'BipedalWalkerHardcore-v3']:
            for _ in range(3):
                config = default_config.copy()
                config['name'] = name
                config['n_envs'] = 2
                env = create_env(config)
                cr = np.zeros(env.n_envs)
                n = np.zeros(env.n_envs)
                for _ in range(2000):
                    a = env.random_action()
                    s, r, d, re = env.step(a)
                    cr += r
                    n += np.array([i.get('frame_skip', 1) for i in env.info()])
                    np.testing.assert_allclose(cr, env.score(), rtol=1e-5, atol=1e-5)
                    np.testing.assert_equal(n, env.epslen())
                    if np.any(re):
                        info = env.info()
                        for k, i in enumerate(info):
                            if i.get('game_over'):
                                cr[k] = 0
                                n[k] = 0

    def test_RayEnvVec(self):
        for name in ['atari_pong', 'atari_breakout', 'BipedalWalkerHardcore-v3']:
            for _ in range(3):
                config = default_config.copy()
                config['name'] = name
                ray.init()
                config['n_envs'] = 2
                config['n_workers'] = 2
                env = create_env(config)
                cr = np.zeros(env.n_envs)
                n = np.zeros(env.n_envs)
                for _ in range(2000):
                    a = env.random_action()
                    s, r, d, re = env.step(a)
                    cr += r
                    n += np.array([i.get('frame_skip', 1) for i in env.info()])
                    np.testing.assert_allclose(cr, env.score())
                    np.testing.assert_equal(n, env.epslen())
                    if np.any(re):
                        info = env.info()
                        for k, i in enumerate(info):
                            if i.get('game_over'):
                                cr[k] = 0
                                n[k] = 0

                ray.shutdown()

    # def test_frame_stack(self):
    #     ray.init()
    #     config = default_config.copy()
    #     config['frame_skip'] = True
    #     for n_workers in [1, 2]:
    #         for n_envs in [1, 2]:
    #             config['n_workers'] = n_workers
    #             config['n_envs'] = n_envs
    #             for name in ['BipedalWalkerHardcore-v3']:
    #                 config['name'] = name
    #                 env = create_env(config)
    #                 has_done = False
    #                 s = env.reset()
    #                 cr = np.zeros(env.n_envs)
    #                 n = np.zeros(env.n_envs)
    #                 for _ in range(1000):
    #                     frame_skip = np.random.randint(1, 10, size=(n_workers, n_envs))
    #                     frame_skip = np.squeeze(frame_skip)
    #                     a = env.random_action()
    #                     s, r, d, info = env.step(a, frame_skip=frame_skip)
    #                     cr += np.where(env.mask(), r, 0)
    #                     if env.n_envs == 1:
    #                         n += info['frame_skip']
    #                     else:
    #                         for k, i in enumerate(info):
    #                             if 'frame_skip' in i:
    #                                 n[k] += i['frame_skip']
                            
    #                     if np.all(env.game_over()):
    #                         break
                            
    #                     np.testing.assert_equal(env.epslen(), n)
    #                     np.testing.assert_allclose(env.score(), cr, atol=1e-5, rtol=1e-5)
    #                     if np.all(d):
    #                         break
            
    #     ray.shutdown()
