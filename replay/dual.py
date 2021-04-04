import logging
import numpy as np

from utility.utils import to_int
from replay.base import Replay
from replay.uniform import UniformReplay
from replay.per import ProportionalPER

logger = logging.getLogger(__name__)

class DualReplay(Replay):
    def __init__(self, config):
        self._replay_type = config['replay_type']
        self._capacity = to_int(config['capacity'])
        self._min_size = to_int(config['min_size'])
        self._batch_size = config['batch_size']

        BufferType = ProportionalPER if self._replay_type.endswith('per') else UniformReplay
        config['replay_type'] = 'per' if self._replay_type.endswith('per') else 'Uniform'
        config['capacity'] = int(self._capacity * config['cap_frac'])
        config['min_size'] = self._min_size
        config['batch_size'] = int(self._batch_size * config['bs_frac'])
        logger.debug(f'Fast replay capacity({config["capacity"]})')
        logger.debug(f'Fast replay batch size({config["batch_size"]})')
        self._fast_replay = BufferType(config)
        
        config['capacity'] = self._capacity - config['capacity']
        config['min_size'] = self._min_size - config['min_size']
        config['batch_size'] = self._batch_size - config['batch_size']
        logger.debug(f'Slow replay capacity({config["capacity"]})')
        logger.debug(f'Slow replay batch size({config["batch_size"]})')
        self._slow_replay = BufferType(config)

    def buffer_type(self):
        return self._replay_type
        
    def good_to_learn(self):
        return self._fast_replay.good_to_learn()

    def __len__(self):
        return self._capacity if self._is_full else len(self._fast_replay) + len(self._fast_replay)

    def sample(self, batch_size=None):
        assert self._good_to_learn()
        batch_size = batch_size or self._batch_size
        if self._slow_replay.good_to_learn():
            regular_samples = self._fast_replay.sample()
            additional_samples = self._slow_replay.sample()
            return self._combine_samples(regular_samples, additional_samples)
        else:
            regular_samples = self._fast_replay.sample(batch_size)
            return regular_samples

    def combine_samples(self, samples1, samples2):
        samples = {}
        assert len(samples1) == len(samples2)
        for k in samples1.keys():
            samples[k] = np.concatenate([samples1[k], samples2[k]])
            assert samples[k].shape[0] == self._batch_size

        return samples

    def merge(self, local_buffer, target_replay):
        if target_replay == 'fast_replay':
            self._fast_replay.merge(local_buffer)
        elif target_replay == 'slow_replay':
            self._slow_replay.merge(local_buffer)
        else:
            raise NotImplementedError
