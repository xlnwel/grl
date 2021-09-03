import numpy as np
import gym

from env.utils import process_single_agent_env


def make_dmc(config):
    assert 'dmc' in config['name']
    task = config['name'].split('_', 1)[-1]
    env = DeepMindControl(
        task, 
        size=config.setdefault('size', (84, 84)), 
        frame_skip=config.setdefault('frame_skip', 1))
    config.setdefault('max_episode_steps', 1000)
    env = process_single_agent_env(env, config)

    return env


""" original code from https://github.com/denisyarats/dmc2gym """
def _spec_to_box(spec):
    from dm_env import specs
    def extract_min_max(s):
        assert s.dtype == np.float64 or s.dtype == np.float32
        dim = np.int(np.prod(s.shape))
        if type(s) == specs.Array:
            bound = np.inf * np.ones(dim, dtype=np.float32)
            return -bound, bound
        elif type(s) == specs.BoundedArray:
            zeros = np.zeros(dim, dtype=np.float32)
            return s.minimum + zeros, s.maximum + zeros

    mins, maxs = [], []
    for s in spec:
        mn, mx = extract_min_max(s)
        mins.append(mn)
        maxs.append(mx)
    low = np.concatenate(mins, axis=0)
    high = np.concatenate(maxs, axis=0)
    assert low.shape == high.shape
    return gym.spaces.Box(low, high, dtype=np.float32)


def _flatten_obs(obs):
    obs_pieces = []
    for v in obs.values():
        flat = np.array([v]) if np.isscalar(v) else v.ravel()
        obs_pieces.append(flat)
    return np.concatenate(obs_pieces, axis=0)


class DeepMindControl(gym.Env):
    def __init__(
        self, 
        name,
        task_kwargs={},
        visualize_reward=False,
        from_pixels=True,
        size=(84, 84),
        camera_id=0,
        frame_skip=1,
        environment_kwargs=None,
        channels_first=False
    ):
        self._from_pixels = from_pixels
        self._size = tuple(size)
        self._camera_id = camera_id
        self.frame_skip = frame_skip
        self._channels_first = channels_first

        domain, task = name.split('_')

        # create task
        from dm_control import suite
        self._env = suite.load(
            domain_name=domain,
            task_name=task,
            task_kwargs=task_kwargs,
            visualize_reward=visualize_reward,
            environment_kwargs=environment_kwargs
        )

        # true and normalized action gym.spaces
        self._true_action_space = _spec_to_box([self._env.action_spec()])
        self._norm_action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=self._true_action_space.shape,
            dtype=np.float32
        )

        # create observation space
        if from_pixels:
            shape = [3, *size] if channels_first else [*size, 3]
            self._observation_space = gym.spaces.Box(
                low=0, high=255, shape=shape, dtype=np.uint8
            )
        else:
            self._observation_space = _spec_to_box(
                self._env.observation_spec().values()
            )
            
        self._state_space = _spec_to_box(
                self._env.observation_spec().values()
        )
        
        self.current_state = None

        # set seed
        self.seed(seed=task_kwargs.setdefault('random', 1))

    def __getattr__(self, name):
        return getattr(self._env, name)

    def _get_obs(self, time_step):
        if self._from_pixels:
            obs = self.render(size=self._size, camera_id=self._camera_id)
            if self._channels_first:
                obs = obs.transpose(2, 0, 1).copy()
        else:
            obs = _flatten_obs(time_step.observation)
        return obs

    def _convert_action(self, action):
        action = action.astype(np.float64)
        true_delta = self._true_action_space.high - self._true_action_space.low
        norm_delta = self._norm_action_space.high - self._norm_action_space.low
        action = (action - self._norm_action_space.low) / norm_delta
        action = action * true_delta + self._true_action_space.low
        action = action.astype(np.float32)
        return action

    @property
    def is_multiagent(self):
        return False

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def state_space(self):
        return self._state_space

    @property
    def action_space(self):
        return self._norm_action_space

    def seed(self, seed):
        self._true_action_space.seed(seed)
        self._norm_action_space.seed(seed)
        self._observation_space.seed(seed)

    def step(self, action):
        assert self._norm_action_space.contains(action), f'{action} not in {self._norm_action_space}'
        action = self._convert_action(action)
        assert self._true_action_space.contains(action), f'{action} not in {self._true_action_space}'
        reward = 0
        info = {'internal_state': self._env.physics.get_state().copy()}

        for t in range(self.frame_skip):
            time_step = self._env.step(action)
            reward += time_step.reward or 0
            done = time_step.last()
            if done:
                break
        info['frame_skip'] = t+1
        obs = self._get_obs(time_step)
        self.current_state = _flatten_obs(time_step.observation)
        info['discount'] = time_step.discount
        return obs, reward, done, info

    def reset(self):
        time_step = self._env.reset()
        self.current_state = _flatten_obs(time_step.observation)
        obs = self._get_obs(time_step)
        return obs

    def render(self, mode='rgb_array', size=None, camera_id=0):
        assert mode == 'rgb_array', 'only support rgb_array mode, given %s' % mode
        size = size or self._size
        camera_id = camera_id or self._camera_id
        return self._env.physics.render(
            *size, camera_id=camera_id
        )