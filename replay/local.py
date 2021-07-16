from abc import ABC, abstractmethod
import logging
import collections
import math
import numpy as np

from core.decorator import config
from utility.utils import batch_dicts, flatten_dict
from replay.utils import *

logger = logging.getLogger(__name__)


class LocalBuffer(ABC):
    @config
    def __init__(self):
        self._memory = {}
        self._idx = 0

        self._add_attributes()
    
    def _add_attributes(self):
        self._memlen = self._seqlen

    def name(self):
        return self._replay_type

    def is_full(self):
        assert self._idx <= self._memlen, (self._idx, self._memlen)
        return self._idx == self._memlen

    @property
    def seqlen(self):
        return self._seqlen

    @abstractmethod
    def sample(self):
        raise NotImplementedError

    @abstractmethod
    def add(self, **data):
        raise NotImplementedError


class NStepBuffer(LocalBuffer):
    def _add_attributes(self):
        self._max_steps = getattr(self, '_max_steps', 0)
        self._extra_len = max(self._n_steps, self._max_steps)
        self._memlen = self._seqlen + self._extra_len


class EnvNStepBuffer(NStepBuffer):
    """ Local memory only stores one episode of transitions from each of n environments """
    def reset(self):
        assert self.is_full(), self._idx
        self._idx = self._extra_len
        for v in self._memory.values():
            v[:self._extra_len] = v[self._seqlen:]

    def add(self, **data):
        """ Add experience to local memory """
        if self._memory == {}:
            del data['next_obs']
            init_buffer(self._memory, pre_dims=self._memlen, has_steps=self._n_steps>1, **data)
            print_buffer(self._memory, 'Local')
            
        add_buffer(self._memory, self._idx, self._n_steps, self._gamma, **data)
        self._idx = self._idx + 1

    def sample(self):
        assert self.is_full(), self._idx
        return self.retrieve(self._seqlen)

    def retrieve(self, seqlen=None):
        seqlen = seqlen or self._idx
        results = {}
        for k, v in self._memory.items():
            results[k] = v[:seqlen]
        if 'next_obs' not in self._memory:
            idxes = np.arange(seqlen)
            steps = results.get('steps', 1)
            next_idxes = idxes + steps
            if isinstance(self._memory['obs'], np.ndarray):
                results['next_obs'] = self._memory['obs'][next_idxes]
            else:
                results['next_obs'] = [np.array(self._memory['obs'][i], copy=False) 
                    for i in next_idxes]
        if 'steps' in results:
            results['steps'] = results['steps'].astype(np.float32)

        return results


class EnvVecNStepBuffer(NStepBuffer):
    """ Local memory only stores one episode of transitions from n environments """
    def reset(self):
        assert self.is_full(), self._idx
        self._idx = self._extra_len
        for v in self._memory.values():
            v[:, :self._extra_len] = v[:, self._seqlen:]

    def add(self, env_ids=None, **data):
        """ Add experience to local memory """
        if self._memory == {}:
            # initialize memory
            init_buffer(self._memory, pre_dims=(self._n_envs, self._memlen), 
                        has_steps=self._extra_len>1, **data)
            print_buffer(self._memory, 'Local Buffer')

        idx = self._idx
        
        for k, v in data.items():
            if isinstance(self._memory[k], np.ndarray):
                self._memory[k][:, idx] = v
            else:
                for i in range(self._n_envs):
                    self._memory[k][i][idx] = v[i]
        if self._extra_len > 1:
            self._memory['steps'][:, idx] = 1

        self._idx += 1

    def sample(self):
        assert self.is_full(), self._idx
        return self.retrieve(self._seqlen)
    
    def retrieve(self, seqlen=None):
        seqlen = seqlen or self._idx
        results = adjust_n_steps_envvec(self._memory, seqlen, 
            self._n_steps, self._max_steps, self._gamma)
        value = None
        for k, v in results.items():
            if k in ('q', 'v'):
                value = results[k]
                pass
            else:
                results[k] = v[:, :seqlen].reshape(-1, *v.shape[2:])
        if value:
            idx = np.broadcast_to(np.arange(seqlen), (self._n_envs, seqlen))
            results['q'] = value[idx]
            results['next_q'] = value[idx + results.get('steps', 1)]
        if 'mask' in results:
            mask = results.pop('mask')
            results = {k: v[mask] for k, v in results.items()}
        if 'steps' in results:
            results['steps'] = results['steps'].astype(np.float32)

        return results


class SequentialBuffer(LocalBuffer):
    def reset(self):
        self._idx = self._memlen - self._reset_shift

    def _add_attributes(self):
        if not hasattr(self, '_reset_shift'):
            self._reset_shift = getattr(self, '_burn_in_size', 0) or self._sample_size
        self._extra_len = 1
        self._memlen = self._sample_size + self._extra_len

    def add(self, **data):
        if self._memory == {}:
            for k in data:
                if k in self._state_keys:
                    self._memory[k] = collections.deque(
                        maxlen=math.ceil(self._memlen / self._reset_shift))
                else:
                    self._memory[k] = collections.deque(maxlen=self._memlen)

        for k, v in data.items():
            if k not in self._state_keys or self._idx % self._reset_shift == 0:
                self._memory[k].append(v)
        
        self._idx += 1
    
    def clear(self):
        self._idx = 0


class EnvSequentialBuffer(SequentialBuffer):
    def sample(self):
        assert self.is_full(), self._idx
        results = {}
        for k, v in self._memory.items():
            if k in self._state_keys:
                results[k] = v[0]
            elif k in self._extra_keys:
                results[k] = np.array(v)
            else:
                results[k] = np.array(v)[:self._sample_size]

        return results


class EnvVecSequentialBuffer(SequentialBuffer):
    def sample(self):
        assert self.is_full(), self._idx
        results = {}
        for k, v in self._memory.items():
            if k in self._state_keys:
                results[k] = v[0]
            elif k in self._extra_keys:
                results[k] = np.swapaxes(np.array(v), 0, 1)
            else:
                results[k] = np.swapaxes(np.array(list(v)[:self._sample_size]), 0, 1)
        
        results = [{k: v[i] for k, v in results.items()} for i in range(self._n_envs)] 
        for seq in results:
            for k, v in seq.items():
                if k in self._state_keys:
                    pass
                elif k in self._extra_keys:
                    assert v.shape[0] == self._sample_size + self._extra_len, (k, v.shape)
                else:
                    assert v.shape[0] == self._sample_size, (k, v.shape)
        assert len(results) == self._n_envs, results
        
        return results


class EnvEpisodicBuffer(LocalBuffer):
    def _add_attributes(self):
        super()._add_attributes()
        self._memory = collections.defaultdict(list)
    
    def reset(self):
        self._memory.clear()

    def sample(self):
        results = {k: np.array(v) for k, v in self._memory.items()}
        self.reset()
        return results

    def add(self, **data):
        for k, v in data.items():
            self._memory[k].append(v)


class EnvFixedEpisodicBuffer(EnvEpisodicBuffer):
    def _add_attributes(self):
        super()._add_attributes()
        self._memlen = self._seqlen + 1 # one for the last observation

    def reset(self):
        super().reset()
        self._idx = 0
        assert len(self._memory) == 0, self._memory

    def add(self, **data):
        super().add(**data)
        self._idx += 1

    def sample(self):
        assert self.is_full(), self._idx
        results = {k: np.array(v) for k, v in self._memory.items()}
        for k, v in results.items():
            assert v.shape[0] >= self._seqlen, [
                (kk, vv.shape) for kk, vv in results.items()
            ]
        return results


class EnvVecFixedEpisodicBuffer(EnvFixedEpisodicBuffer):
    def _add_attributes(self):
        super()._add_attributes()
        self.reset()

    def reset(self, idxes=None):
        if idxes is None:
            self._memory = [
                collections.defaultdict(list) 
                for _ in range(self._n_envs)]
            self._idx = 0
        elif isinstance(idxes, (list, tuple)):
            # do not reset self._idx here; 
            # we only drop the bad episodes given by idxes
            [self._memory[i].clear() for i in idxes]
        else:
            raise ValueError(idxes)

    def add(self, **data):
        # we do not add all data at once since 
        # some environments throw errors and 
        # we need to drop these data accordingly
        for k, v in data.items():
            for i in range(self._n_envs):
                self._memory[i][k].append(v[i])
        self._idx += 1

    def sample(self, batch_data=False):
        assert self.is_full(), (self._idx, self._memlen)
        results = [d for d in self._memory if d]
        if batch_data:
            results = batch_dicts(results)
            for k, v in results.items():
                assert v.shape[1] >= self._seqlen, [
                    (kk, vv.shape) for kk, vv in results.items()
                ]

        return results
