import numpy as np
import os
os.environ.setdefault('PATH', '')

import gym
from gym.spaces.box import Box
import cv2

from env.utils import process_single_agent_env

cv2.ocl.setUseOpenCL(False)



def make_atari(config):
    assert 'atari' in config['name'], config['name']
    env = Atari(**config)
    config.setdefault('max_episode_steps', 108000)    # 30min
    env = process_single_agent_env(env, config)
    
    return env

class Atari:
    """ This class originally from Google's Dopamine,
    Adapted for more general cases
    """
    def __init__(self, name, *, frame_skip=4, life_done=False,
                image_size=(84, 84), noop=30, 
                sticky_actions=True, gray_scale=True, 
                np_obs=False, **kwargs):
        version = 0 if sticky_actions else 4
        name = name.split('_', 1)[-1]
        name = name[0].capitalize() + name[1:]
        name = f'{name}NoFrameskip-v{version}'
        env = gym.make(name)
        """
        Original comments:
            Strip out the TimeLimit wrapper from Gym, which caps us at 100k frames. 
            We handle this time limit internally instead, which lets us cap at 108k 
            frames (30 minutes). The TimeLimit wrapper also plays poorly with 
            saving and restoring states.
        Interestingly, I found in breakout, images are distorted after 100k... 
        """
        self.env = env.env
        self.life_done = life_done
        self.frame_skip = frame_skip
        self.gray_scale = gray_scale
        self.noop = noop
        self.image_size = (image_size, image_size) \
            if isinstance(image_size, int) else tuple(image_size)
        self.np_obs = np_obs

        assert self.frame_skip > 0, \
            f'Frame skip should be strictly positive, got {self.frame_skip}'
        assert np.all([s > 0 for s in self.image_size]), \
            f'Screen size should be strictly positive, got {image_size}'

        obs_shape = self.env.observation_space.shape
        # Stores temporary observations used for pooling over two successive
        # frames.
        shape = obs_shape[:2]
        if not gray_scale:
            shape += (3,)
        self._buffer = [np.empty(shape, dtype=np.uint8) for _ in range(2)]

        self.lives = 0  # Will need to be set by reset().
        self._game_over = True
        self._frames_in_step = 0    # count the frames elapsed in a single step

    @property
    def observation_space(self):
        # Return the observation space adjusted to match the shape of the processed
        # observations.
        c = 1 if self.gray_scale else 3
        shape = self.image_size + (c, )
        return Box(low=0, high=255, shape=shape,
                dtype=np.uint8)

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def reward_range(self):
        return self.env.reward_range

    @property
    def metadata(self):
        return self.env.metadata

    def seed(self, seed=0):
        self.env.seed(seed)

    def close(self):
        return self.env.close()

    def get_screen(self):
        return self.env.ale.getScreenRGB2()

    def game_over(self):
        return self._game_over

    def set_game_over(self):
        self._game_over = True

    def reset(self, hard_reset=True, **kwargs):
        self.env.reset(**kwargs)
        if 'FIRE' in self.env.get_action_meanings():
            action = self.env.get_action_meanings().index('FIRE')
            for _ in range(self.frame_skip):
                self.env.step(action)
        noop = np.random.randint(1, self.noop + 1)
        for _ in range(noop):
            d = self.env.step(0)[2]
            if d:
                self.env.reset(**kwargs)
        
        self.lives = self.env.ale.lives()
        self._get_screen(self._buffer[0])
        self._buffer[1].fill(0)
        obs = self._pool_and_resize()

        self._game_over = False
        return obs

    def _fake_reset(self):
        if 'FIRE' in self.env.get_action_meanings():
            action = self.env.get_action_meanings().index('FIRE')
        else:
            action = 0
        obs = self.step(action)[0]
        
        return obs

    def render(self, mode):
        """Renders the current screen, before preprocessing.

        This calls the Gym API's render() method.

        Args:
            mode: Mode argument for the environment's render() method.
                Valid values (str) are:
                'rgb_array': returns the raw ALE image.
                'human': renders to display via the Gym renderer.

        Returns:
            if mode='rgb_array': numpy array, the most recent screen.
            if mode='human': bool, whether the rendering was successful.
        """
        return self.env.render(mode)

    def step(self, action):
        total_reward = 0.

        for step in range(1, self.frame_skip+1):
            # We bypass the Gym observation altogether and directly fetch
            # the image from the ALE. This is a little faster.
            _, reward, done, info = self.env.step(action)
            total_reward += reward
            is_terminal = done
            if self.life_done:
                new_lives = self.env.ale.lives()
                if new_lives < self.lives and new_lives > 0:
                    self.lives = new_lives
                    is_terminal = True
                    self._fake_reset()
                    info['reset'] = True

            if is_terminal:
                info['game_over'] = done
                break
            elif step >= self.frame_skip - 1:
                i = step - (self.frame_skip - 1)
                self._get_screen(self._buffer[i])

        # Pool the last two observations.
        obs = self._pool_and_resize()

        self._game_over = done
        info['frame_skip'] = step
        return obs, total_reward, is_terminal, info

    def _pool_and_resize(self):
        """Transforms two frames into a Nature DQN observation.

        For efficiency, the transformation is done in-place in self._buffer.

        Returns:
            transformed_screen: numpy array, pooled, resized image.
        """
        # Pool if there are enough screens to do so.
        if self.frame_skip > 1:
            np.maximum(self._buffer[0], self._buffer[1],
                    out=self._buffer[0])

        img = cv2.resize(
            self._buffer[0], self.image_size, interpolation=cv2.INTER_AREA)
        img = np.asarray(img, dtype=np.uint8)
        return np.expand_dims(img, axis=2) if self.gray_scale else img

    def _get_screen(self, output):
        if self.gray_scale:
            self.env.ale.getScreenGrayscale(output)
        else:
            self.env.ale.getScreenRGB2(output)

    @property
    def is_multiagent(self):
        return False
