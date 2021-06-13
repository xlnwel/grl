from datetime import datetime
import logging
from pathlib import Path
import random
import uuid
import numpy as np
from numpy.lib.arraysetops import isin

from core.decorator import config
from replay.local import EnvEpisodicBuffer, EnvFixedEpisodicBuffer
from replay.utils import load_data, save_data

logger = logging.getLogger(__name__)


class EpisodicReplay:
    @config
    def __init__(self, state_keys=[]):
        self._dir = Path(self._dir).expanduser()
        self._save = getattr(self, '_save', False)
        if self._save:
            self._dir.mkdir(parents=True, exist_ok=True)
        self._memory = {}
        # Store and retrieve entire episodes if sample_size is None
        self._sample_size = getattr(self, '_sample_size', None)
        self._state_keys = state_keys
        self._tmp_bufs = []

        self._local_buffer_type = getattr(self, '_local_buffer_type', 'eps')
        self.TempBufferType = {
            'eps': EnvEpisodicBuffer, 
            'fixed_eps': EnvFixedEpisodicBuffer
        }.get(self._local_buffer_type)
    
    def name(self):
        return self._replay_type

    def good_to_learn(self):
        return len(self._memory) >= self._min_episodes

    def __len__(self):
        return len(self._memory)

    def add(self, idxes=None, **data):
        if self._n_envs > 1:
            if self._n_envs != len(self._tmp_bufs):
                logger.info(f'Initialize {self._n_envs} temporary buffer')
                self._tmp_bufs = [
                    self.TempBufferType({'seqlen': self._seqlen}) 
                    for _ in range(self._n_envs)]
            if idxes is None:
                idxes = range(self._n_envs)
            for i in idxes:
                d = {k: v[i] for k, v in data.items()}
                self._tmp_bufs[i].add(**d)
        else:
            if self._tmp_bufs == []:
                self._tmp_bufs = self.TempBufferType({'seqlen': self._seqlen})
            self._tmp_bufs.add(**data)

    def reset_local_buffer(self, i=None):
        if i is None:
            if self._n_envs > 1:
                [buf.reset() for buf in self._tmp_bufs]
            else:
                self.merge(self._tmp_bufs.reset())
        elif isinstance(i, (list, tuple)):
            [self._tmp_bufs[ii].reset() for ii in range(i)]
        elif isinstance(i, int):
            self._tmp_bufs[i].reset()
        else:
            raise ValueError(f'{i} of type {type(i)} is not supported')
        
    def is_local_buffer_full(self, i=None):
        if i is None:
            if self._n_envs > 1:
                is_full = [buf.is_full() for buf in self._tmp_bufs]
            else:
                is_full = self._tmp_bufs.is_full()
        elif isinstance(i, (list, tuple)):
            is_full = [self._tmp_bufs[ii].is_full() for ii in i]
        elif isinstance(i, int):
            is_full = self._tmp_bufs[i].is_full()
        else:
            raise ValueError(f'{i} of type {type(i)} is not supported')
        return is_full

    def finish_episodes(self, i=None):
        if i is None:
            if self._n_envs > 1:
                [self.merge(buf.sample()) for buf in self._tmp_bufs]
            else:
                self.merge(self._tmp_bufs.sample())
        elif isinstance(i, (list, tuple)):
            [self.merge(self._tmp_bufs[ii].sample()) for ii in range(i)]
        elif isinstance(i, int):
            self.merge(self._tmp_bufs[i].sample())
        else:
            raise ValueError(f'{i} of type {type(i)} is not supported')
        
    def merge(self, episodes):
        if episodes is None:
            return
        if isinstance(episodes, dict):
            episodes = [episodes]
        timestamp = datetime.now().strftime('%Y%m%dT%H%M%S')
        for eps in episodes:
            if self._sample_size and len(next(iter(eps.values()))) < self._sample_size:
                continue    # ignore short episodes
            identifier = str(uuid.uuid4().hex)
            length = len(eps['reward'])
            filename = self._dir / f'{timestamp}-{identifier}-{length}.npz'
            self._memory[filename] = eps
            if self._save:
                save_data(filename, eps)
        if self._save:
            self._remove_files()

    def count_episodes(self):
        """ count the total number of episodes and transitions in the directory """
        if self._save:
            filenames = self._dir.glob('*.npz')
            # subtract 1 as we don't take into account the terminal state
            lengths = [int(n.stem.rsplit('-', 1)[-1]) - 1 for n in filenames]
            episodes, steps = len(lengths), sum(lengths)
            return episodes, steps
        else:
            return 0, 0
    
    def count_steps(self):
        filenames = self._dir.glob('*.npz')
        # subtract 1 as we don't take into account the terminal state
        lengths = [int(n.stem.rsplit('-', 1)[-1]) - 1 for n in filenames]
        episodes, steps = len(lengths), sum(lengths)
        return episodes, steps

    def load_data(self):
        if self._memory == {}:
            # load data from files
            for filename in self._dir.glob('*.npz'):
                if filename not in self._memory:
                    data = load_data(filename)
                    if data is not None:
                        self._memory[filename] = data
            logger.info(f'{len(self)} episodes are loaded')
        else:
            logger.warning(f'There are already {len(self)} episodes in the memory. No further loading is performed')

    def sample(self, batch_size=None):
        batch_size = batch_size or self._batch_size
        if batch_size > 1:
            samples = [self._sample() for _ in range(batch_size)]
            data = {k: np.stack([t[k] for t in samples], 0)
                for k in samples[0].keys()}
        else:
            data = self._sample()
        
        return data

    def _sample(self):
        filename = random.choice(list(self._memory))
        episode = self._memory[filename]
        if self._sample_size:
            total = len(next(iter(episode.values())))
            available = total - self._sample_size
            assert available > 0, f'Skipped short episode of length {total}.' \
                f'{[(k, np.array(v).shape) for e in self._memory.values() for k, v in e.items()]}'
                
            i = int(random.randint(0, available))
            episode = {k: v[i] if k in self._state_keys 
                        else v[i: i + self._sample_size] 
                        for k, v in episode.items()}
        return episode

    def _remove_files(self):
        if getattr(self, '_max_episodes', 0) > 0 and len(self._memory) > self._max_episodes:
            # remove some oldest files if the number of files stored exceeds maximum size
            filenames = sorted(self._memory)
            start = int(.1 * self._max_episodes)
            for filename in filenames[:start]:
                filename.unlink()
                if filename in self._memory:
                    del self._memory[filename]
            filenames = filenames[start:]
            logger.info(f'{start} files are removed')

    def clear_temp_bufs(self):
        self._tmp_bufs = []