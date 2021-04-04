import numpy as np
from PIL import Image
import gym
import threading

from env import wrappers
from env import procgen
from env.func import create_env


def make_env(config):
    config = config.copy()
    if '_' not in config['name']:
        config['name'] = f"_{config['name']}"
    suite, task = config['name'].split('_', 1)
    if suite == 'dmc':
        env = DeepMindControl(task, config.get('size', (64, 64)))
        env = wrappers.FrameSkip(env, config['frame_skip'])
        env = wrappers.NormalizeActions(env)
        config['max_episode_steps'] = 1000
        if config.get('frame_stack', 1) > 1:
            env = wrappers.FrameStack(env, config['frame_stack'], np_obs=False)
    elif suite == 'atari':
        env = Atari(
            task, config['frame_skip'], (64, 64), grayscale=False,
            life_done=True, sticky_actions=True)
        config['max_episode_steps'] = 108000
    elif 'procgen' in config['name'].lower():
        env = procgen.make_procgen_env(config)
        config['max_episode_steps'] = env.spec.max_episode_steps or 1000
    else:
        raise NotImplementedError(suite)
    env = wrappers.post_wrap(env, config)

    return env
    
class DeepMindControl(gym.Env):

    def __init__(self, name, size=(64, 64), camera=None):
        domain, task = name.split('_', 1)
        if domain == 'cup':  # Only domain with multiple words.
            domain = 'ball_in_cup'
        if isinstance(domain, str):
            from dm_control import suite
            self._env = suite.load(domain, task)
        else:
            assert task is None
            self._env = domain()
        self._size = tuple(size)
        if camera is None:
            camera = dict(quadruped=2).get(domain, 0)
        self._camera = camera

        self.observation_space = gym.spaces.Box(
            0, 255, self._size + (3,), dtype=np.uint8)

        spec = self._env.action_spec()
        self.action_space = gym.spaces.Box(spec.minimum, spec.maximum, dtype=np.float32)

    def step(self, action):
        time_step = self._env.step(action)
        # obs = dict(time_step.observation)
        self._obs = self.render()
        reward = time_step.reward or 0
        done = time_step.last()
        info = {'discount': np.array(time_step.discount, np.float32)}
        return self._obs, reward, done, info

    def reset(self):
        time_step = self._env.reset()
        self._obs = self.render()
        return self._obs

    def get_screen(self):
        return self._obs

    def render(self, *args, size=None, **kwargs):
        size = size or self._size
        if kwargs.get('mode', 'rgb_array') != 'rgb_array':
            raise ValueError("Only render mode 'rgb_array' is supported.")
        return self._env.physics.render(*size, camera_id=self._camera)

class Atari(gym.Env):

    LOCK = threading.Lock()

    def __init__(
        self, name, action_repeat=4, size=(84, 84), grayscale=True, noops=30,
        life_done=False, sticky_actions=True):
        import gym
        version = 0 if sticky_actions else 4
        name = ''.join(word.title() for word in name.split('_'))
        with self.LOCK:
            self._env = gym.make('{}NoFrameskip-v{}'.format(name, version))
        self.frame_skip = action_repeat
        self._size = size
        self._grayscale = grayscale
        self._noops = noops
        self._life_done = life_done
        self._lives = None
        shape = self._env.observation_space.shape[:2] + (() if grayscale else (3,))
        self._buffers = [np.empty(shape, dtype=np.uint8) for _ in range(2)]
        self._random = np.random.RandomState(seed=None)

    @property
    def observation_space(self):
        shape = self._size + (1 if self._grayscale else 3,)
        return gym.spaces.Box(low=0, high=255, shape=shape, dtype=np.uint8)
        
    @property
    def action_space(self):
        return self._env.action_space

    def close(self):
        return self._env.close()

    def reset(self):
        with self.LOCK:
            self._env.reset()
        noops = self._random.randint(1, self._noops + 1)
        for _ in range(noops):
            done = self._env.step(0)[2]
            if done:
                with self.LOCK:
                    self._env.reset()
        self._lives = self._env.ale.lives()
        if self._grayscale:
            self._obs = self._env.ale.getScreenGrayscale(self._buffers[0])
        else:
            self._obs = self._env.ale.getScreenRGB2(self._buffers[0])
        self._buffers[1].fill(0)
        return self._get_obs()

    def step(self, action):
        total_reward = 0.0
        for step in range(self.frame_skip):
            _, reward, done, info = self._env.step(action)
            total_reward += reward
            if self._life_done:
                lives = self._env.ale.lives()
                done = done or lives < self._lives
                self._lives = lives
            if done:
                break
            elif step >= self.frame_skip - 2:
                index = step - (self.frame_skip - 2)
                if self._grayscale:
                    self._obs = self._env.ale.getScreenGrayscale(self._buffers[index])
                else:
                    self._obs = self._env.ale.getScreenRGB2(self._buffers[index])
        obs = self._get_obs()
        return obs, total_reward, done, info

    def render(self, mode):
        return self._env.render(mode)

    def get_screen(self):
        return self._obs

    def _get_obs(self):
        if self.frame_skip > 1:
            np.maximum(self._buffers[0], self._buffers[1], out=self._buffers[0])
        image = np.array(Image.fromarray(self._buffers[0]).resize(
            self._size, Image.BILINEAR))
        image = np.clip(image, 0, 255).astype(np.uint8)
        image = image[:, :, None] if self._grayscale else image
        return image


if __name__ == '__main__':
    config= dict(
        name='dmc_walker_walk',
        n_workers=1,
        n_envs=1,
        log_episode=True,
        auto_reset=True,
        frame_skip=2,
        max_episode_steps=1000,
    )
    env = create_env(config, make_env)
    s = env.reset()
    for _ in range(1010):
        s, r, d, i = env.step(env.random_action())
        if d: print(d)
