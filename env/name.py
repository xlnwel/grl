import itertools
import atari_py as atari
from procgen.env import ENV_NAMES

atari_envs = set(atari.list_games())
procgen_envs = set(ENV_NAMES)

suite_env = dict(
    atari=atari_envs,
    procgen=procgen_envs,
)

env2suite = dict(
    list(itertools.product(atari_envs, ['atari']))
    + list(itertools.product(procgen_envs, ['procgen']))
)

def is_atari(name):
    return env2suite.get(name) == 'atari'

def is_procgen(name):
    return env2suite.get(name) == 'procgen'

if __name__ == '__main__':
    for v in suite_env['procgen']:
        print(v)