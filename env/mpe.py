from env.mpe_env.MPE_env import MPEEnv
from env import wrappers


def make_mpe(config):
    assert 'mpe' in config['name'], config['name']
    env = MPEEnv(config)
    env = wrappers.DataProcess(env)
    env = wrappers.MAEnvStats(env)

    return env
