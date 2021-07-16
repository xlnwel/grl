from replay.uniform import UniformReplay
from replay.per import ProportionalPER
from replay.eps import EpisodicReplay
from replay.seq import SequentialReplay
from replay.sper import SequentialPER
from replay.local import *


replay_type = dict(
    uniform=UniformReplay,
    per=ProportionalPER,
    episodic=EpisodicReplay,
    seq=SequentialReplay,
    seqper=SequentialPER
)

def create_local_buffer(config):
    config = config.copy()

    if config.get('local_buffer_type', None) is not None:
        buffer_name = config.pop('local_buffer_type')
        # use the vectorized buffer if n_envs > 1
        if config.get('n_envs', 1) > 1 and not buffer_name.startswith('vec_'):
            buffer_name = 'vec_' + buffer_name

        buffer_type = {
            'nstep': EnvNStepBuffer,
            'vec_nstep': EnvVecNStepBuffer,
            'seq': EnvSequentialBuffer,
            'vec_seq': EnvVecSequentialBuffer,
            'fixed_eps': EnvFixedEpisodicBuffer,
            'vec_fixed_eps': EnvVecFixedEpisodicBuffer,
        }[buffer_name]

        return buffer_type(config)
    else:
        n_envs = config.get('n_envs', 1)
        is_sequential = config['replay_type'].startswith('seq')
        is_envvec = config.pop('force_envvec', False) or n_envs > 1
        buffer_type = {
            (False, False): EnvNStepBuffer, 
            (False, True): EnvVecNStepBuffer, 
            (True, False): EnvSequentialBuffer, 
            (True, True): EnvVecSequentialBuffer, 
        }[(is_sequential, is_envvec)]

        return buffer_type(config)

def create_replay(config, **kwargs):
    config = config.copy()
    return replay_type[config['replay_type']](config, **kwargs)

def create_replay_center(config, **kwargs):
    config = config.copy()
    
    plain_type = replay_type[config['replay_type']]
    import ray
    ray_type = ray.remote(plain_type)
    return ray_type.remote(config, **kwargs)
    
if __name__ == '__main__':
    config = dict(
        type='sper',                      # per or uniform
        precision=32,
        # arguments for PER
        beta0=0.4,
        to_update_top_priority=False,

        # arguments for general replay
        batch_size=0,
        sample_size=7,
        burn_in_size=2,
        min_size=2,
        capacity=4,
    )
    import threading
    import numpy as np
    replay = create_replay(config, state_keys=['h', 'c'])
    def sample():
        for _ in range(100000):
            data = replay.sample()
    while not replay.good_to_learn():
        replay.add(
            o=np.random.normal(size=4), 
            h=np.random.normal(size=2), 
            c=np.random.normal(size=2)) 
    print('start update')
    for i in range(100):
        replay.add(
            o=np.random.normal(size=4), 
            h=np.random.normal(size=2), 
            c=np.random.normal(size=2))
        priority = np.random.uniform(.1, 2, size=3)
        replay.update_priorities(priority, np.ones(len(priority), dtype=np.int32))
        # replay.update_priorities(priority, np.random.randint(0, len(replay), len(priority))) 