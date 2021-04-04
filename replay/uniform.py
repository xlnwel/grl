import numpy as np

from core.decorator import override
from replay.base import Replay


class UniformReplay(Replay):
    @override(Replay)
    def sample(self, batch_size=None):
        assert self.good_to_learn(), (
            'There are not sufficient transitions to start learning --- '
            f'transitions in buffer({len(self)}) vs '
            f'minimum required size({self.min_size})')

        samples = self._sample(batch_size)

        return samples

    """ Implementation """
    @override(Replay)
    def _sample(self, batch_size=None):
        batch_size = batch_size or self._batch_size
        size = self._capacity if self._is_full else self._mem_idx
        idxes = np.random.randint(size, size=batch_size)
        # the following code avoids repetitive sampling, 
        # but it takes significant more time to run(around 1000x).
        # idxes = np.random.choice(size, size=batch_size, replace=False)
        
        samples = self._get_samples(idxes)

        return samples
