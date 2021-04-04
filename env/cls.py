import itertools
import numpy as np
import cv2
import gym

from env import wrappers, atari, pg, dmc

EnvOutput = wrappers.EnvOutput


def make_env(config):
    config = config.copy()
    env_name = config['name'].lower()
    if env_name.startswith('atari'):
        env = atari.make_atari_env(config)
    else:
        if env_name.startswith('procgen'):
            env = pg.make_procgen_env(config)
        elif env_name.startswith('dmc'):
            env = dmc.make_dmc_env(config)
        else:
            env = gym.make(config['name']).env
            env = wrappers.DummyEnv(env)    # useful for hidding unexpected frame_skip
            config.setdefault('max_episode_steps', env.spec.max_episode_steps)
    if config.get('reward_scale') or config.get('reward_clip'):
        env = wrappers.RewardHack(env, **config)
    frame_stack = config.setdefault('frame_stack', 1)
    np_obs = config.setdefault('np_obs', False)
    if frame_stack > 1:
        env = wrappers.FrameStack(env, frame_stack, np_obs)
    frame_diff = config.setdefault('frame_diff', False)
    assert not (frame_diff and frame_stack > 1), f"Don't support using FrameStack and FrameDiff at the same time"
    if frame_diff:
        gray_scale_residual = config.setdefault('gray_scale_residual', False)
        distance = config.setdefault('distance', 1)
        env = wrappers.FrameDiff(env, gray_scale_residual, distance)
    env = wrappers.post_wrap(env, config)
    
    return env


class Env(gym.Wrapper):
    def __init__(self, config, env_fn=make_env):
        self.env = env_fn(config)
        if 'seed' in config and hasattr(self.env, 'seed'):
            self.env.seed(config['seed'])
        self.name = config['name']
        self.max_episode_steps = self.env.max_episode_steps
        self.n_envs = 1
        self.env_type = 'Env'
        super().__init__(self.env)

    def reset(self, idxes=None):
        return self.env.reset()

    def random_action(self, *args, **kwargs):
        action = self.env.action_space.sample()
        return action
        
    def step(self, action, **kwargs):
        output = self.env.step(action, **kwargs)
        return output

    """ the following code is needed for ray """
    def score(self, *args):
        return self.env.score()

    def epslen(self, *args):
        return self.env.epslen()

    def mask(self, *args):
        return self.env.mask()

    def prev_obs(self):
        return self.env.prev_obs()

    def info(self):
        return self.env.info()

    def game_over(self):
        return self.env.game_over()

    def get_screen(self, size=None):
        if hasattr(self.env, 'get_screen'):
            img = self.env.get_screen()
        else:
            img = self.env.render(mode='rgb_array')

        if size is not None and size != img.shape[:2]:
            # cv2 receive size of form (width, height)
            img = cv2.resize(img, size[::-1], interpolation=cv2.INTER_AREA)
            
        return img


class EnvVecBase(gym.Wrapper):
    def __init__(self):
        self.env_type = 'EnvVec'
        super().__init__(self.env)

    def _convert_batch_obs(self, obs):
        if obs != []:
            if isinstance(obs[0], np.ndarray):
                obs = np.reshape(obs, [-1, *self.obs_shape])
            else:
                obs = list(obs)
        return obs

    def _get_idxes(self, idxes):
        if idxes is None:
            idxes = list(range(self.n_envs))
        elif isinstance(idxes, int):
            idxes = [idxes]
        return idxes


class EnvVec(EnvVecBase):
    def __init__(self, config, env_fn=make_env):
        self.n_envs = n_envs = config.pop('n_envs', 1)
        self.name = config['name']
        self.envs = [env_fn(config) for i in range(n_envs)]
        self.env = self.envs[0]
        if 'seed' in config:
            [env.seed(config['seed'] + i) 
                for i, env in enumerate(self.envs)
                if hasattr(env, 'seed')]
        self.max_episode_steps = self.env.max_episode_steps
        super().__init__()

    def random_action(self, *args, **kwargs):
        return np.array([env.action_space.sample() for env in self.envs], copy=False)

    def reset(self, idxes=None, **kwargs):
        idxes = self._get_idxes(idxes)
        obs, reward, done, reset = zip(*[self.envs[i].reset() for i in idxes])

        return EnvOutput(
            self._convert_batch_obs(obs),
            np.array(reward), 
            np.array(done),
            np.array(reset))

    def step(self, actions, **kwargs):
        return self._envvec_op('step', action=actions, **kwargs)

    def score(self, idxes=None):
        idxes = self._get_idxes(idxes)
        return [self.envs[i].score() for i in idxes]

    def epslen(self, idxes=None):
        idxes = self._get_idxes(idxes)
        return [self.envs[i].epslen() for i in idxes]
    
    def mask(self, idxes=None):
        idxes = self._get_idxes(idxes)
        return np.array([self.envs[i].mask() for i in idxes])

    def game_over(self):
        return np.array([env.game_over() for env in self.envs], dtype=np.bool)

    def prev_obs(self, idxes=None):
        idxes = self._get_idxes(idxes)
        return [self.envs[i].prev_obs() for i in idxes]

    def info(self, idxes=None):
        idxes = self._get_idxes(idxes)
        return [self.envs[i].info() for i in idxes]

    def output(self, idxes=None):
        idxes = self._get_idxes(idxes)
        obs, reward, done, reset = zip(*[self.envs[i].output() for i in idxes])

        return EnvOutput(
            self._convert_batch_obs(obs),
            np.array(reward), 
            np.array(done),
            np.array(reset))

    def get_screen(self, size=None):
        if hasattr(self.env, 'get_screen'):
            imgs = np.array([env.get_screen() for env in self.envs], copy=False)
        else:
            imgs = np.array([env.render(mode='rgb_array') for env in self.envs],
                            copy=False)

        if size is not None:
            # cv2 receive size of form (width, height)
            imgs = np.array([cv2.resize(i, size[::-1], interpolation=cv2.INTER_AREA) 
                            for i in imgs])
        
        return imgs

    def _envvec_op(self, name, **kwargs):
        method = lambda e: getattr(e, name)
        if kwargs:
            kwargs = [dict(x) for x in zip(*[itertools.product([k], v) 
                for k, v in kwargs.items()])]
            obs, reward, done, reset = \
                zip(*[method(env)(**kw) for env, kw in zip(self.envs, kwargs)])
        else:
            obs, reward, done, reset = \
                zip(*[method(env)() for env in self.envs])

        return EnvOutput(
            self._convert_batch_obs(obs),
            np.array(reward), 
            np.array(done),
            np.array(reset))

