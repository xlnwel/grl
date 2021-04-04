import numpy as np

from core.decorator import override
from utility.schedule import PiecewiseSchedule
from replay.base import Replay
from replay.ds.sum_tree import SumTree


class PERBase(Replay):
    """ Base class for PER, left in case one day I implement rank-based PER """
    def _add_attributes(self):
        super()._add_attributes()
        self._top_priority = 1.
        self._data_structure = None            
        self._use_is_ratio = getattr(self, '_use_is_ratio', True)
        self._beta = float(getattr(self, 'beta0', .4))
        if getattr(self, '_beta_schedule', None):
            assert isinstance(self._beta_schedule, list)
            self._beta_schedule = PiecewiseSchedule(self._beta_schedule)
        self._sample_i = 0   # count how many times self._sample is called

    @override(Replay)
    def sample(self, batch_size=None):
        assert self.good_to_learn(), (
            'There are not sufficient transitions to start learning --- '
            f'transitions in buffer({len(self)}) vs '
            f'minimum required size({self._min_size})')
        samples = self._sample(batch_size=batch_size)
        self._sample_i += 1
        if hasattr(self, '_beta_schedule'):
            self._update_beta()
        return samples

    @override(Replay)
    def add(self, **kwargs):
        super().add(**kwargs)
        # super().add updates self._mem_idx 
        if self._n_envs == 1:
            self._data_structure.update(self._mem_idx - 1, self._top_priority)

    def update_priorities(self, priorities, idxes):
        assert not np.any(np.isnan(priorities)), priorities
        np.testing.assert_array_less(0, priorities)
        if self._to_update_top_priority:
            self._top_priority = max(self._top_priority, np.max(priorities))
        self._data_structure.batch_update(idxes, priorities)

    """ Implementation """
    def _update_beta(self):
        self._beta = self._beta_schedule.value(self._sample_i)

    @override(Replay)
    def _merge(self, local_buffer, length):    
        priority = local_buffer.pop('priority')[:length] \
            if 'priority' in local_buffer else self._top_priority * np.ones(length)
        np.testing.assert_array_less(0, priority)
        # update sum tree
        mem_idxes = np.arange(self._mem_idx, self._mem_idx + length) % self._capacity
        self._data_structure.batch_update(mem_idxes, priority)
        # update memory
        super()._merge(local_buffer, length)
        
    def _compute_IS_ratios(self, probabilities):
        """
        w = (N * p)**(-beta)
        max(w) = max(N * p)**(-beta) = (N * min(p))**(-beta)
        norm_w = w / max(w) = (N*p)**(-beta) / (N * min(p))**(-beta)
               = (min(p) / p)**beta
        """
        IS_ratios = (np.min(probabilities) / probabilities)**self._beta

        return IS_ratios


class ProportionalPER(PERBase):
    """ Interface """
    def _add_attributes(self):
        super()._add_attributes()
        self._data_structure = SumTree(self._capacity)        # mem_idx    -->     priority

    """ Implementation """
    @override(PERBase)
    def _sample(self, batch_size=None):
        batch_size = batch_size or self._batch_size
        total_priorities = self._data_structure.total_priorities

        intervals = np.linspace(0, total_priorities, batch_size+1)
        values = np.random.uniform(intervals[:-1], intervals[1:])
        priorities, idxes = self._data_structure.batch_find(values)
        assert np.max(idxes) < len(self), f'idxes: {idxes}\nvalues: {values}\npriorities: {priorities}\ntotal: {total_priorities}, len: {len(self)}'
        assert np.min(priorities) > 0, f'idxes: {idxes}\nvalues: {values}\npriorities: {priorities}\ntotal: {total_priorities}, len: {len(self)}'

        probabilities = priorities / total_priorities

        # compute importance sampling ratios
        samples = self._get_samples(idxes)
        samples['idxes'] = idxes
        if self._use_is_ratio:
            IS_ratios = self._compute_IS_ratios(probabilities)
            samples['IS_ratio'] = IS_ratios.astype(np.float32)

        return samples
