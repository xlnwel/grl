import gym

from env.utils import process_single_agent_env


class DummyEnv:
    """ Useful to break the inheritance of unexpected attributes.
    For example, control tasks in gym by default use frame_skip=4,
    but we usually don't count these frame skips in practice.
    """
    def __init__(self, env):
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.spec = env.spec
        self.reward_range = env.reward_range
        self.metadata = env.metadata

        self.reset = env.reset
        self.step = env.step
        self.render = env.render
        self.close = env.close
        self.seed = env.seed

    @property
    def is_multiagent(self):
        return getattr(self.env, 'is_multiagent', False)


def make_built_in_gym(config):
    env = gym.make(config['name']).env
    env = DummyEnv(env)    # useful for hidding unexpected frame_skip
    config.setdefault('max_episode_steps', 
        env.spec.max_episode_steps)
    env = process_single_agent_env(env, config)
    
    return env
