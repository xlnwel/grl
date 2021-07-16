from env.mpe_env.MPE_env import MPEEnv


def make_mpe_env(config):
    assert 'mpe' in config['name'], config['name']
    env = MPEEnv(config)

    return env
