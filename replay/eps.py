import collections
from datetime import datetime
import logging
from pathlib import Path
import random
import uuid
import numpy as np

from core.decorator import config
from replay.local import EnvEpisodicBuffer, EnvFixedEpisodicBuffer, \
    EnvVecFixedEpisodicBuffer
from replay.utils import load_data, print_buffer, save_data

logger = logging.getLogger(__name__)


class EpisodicReplay:
    @config
    def __init__(self, state_keys=[]):
        self._dir = Path(self._dir).expanduser()
        self._save = getattr(self, '_save', False)
        if self._save:
            self._dir.mkdir(parents=True, exist_ok=True)

        self._filenames = collections.deque()
        self._memory = {}

        self._max_episodes = getattr(self, '_max_episodes', 1000)

        # Store and retrieve entire episodes if sample_size is None
        self._sample_size = getattr(self, '_sample_size', None)
        self._state_keys = state_keys
        self._tmp_bufs = []

        self._local_buffer_type = getattr(
            self, '_local_buffer_type', 'eps')
        # if self._n_envs > 1 and self._local_buffer_type.startswith('env_'):
        #     self._local_buffer_type = 'vec_' + self._local_buffer_type
        self.TempBufferType = {
            'eps': EnvEpisodicBuffer, 
            'fixed_eps': EnvFixedEpisodicBuffer,
            # 'vec_fixed_eps': EnvVecFixedEpisodicBuffer,
        }.get(self._local_buffer_type)

        self._info_printed = False

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
                self._tmp_bufs.reset()
        elif isinstance(i, (list, tuple)):
            [self._tmp_bufs[ii].reset() for ii in range(i)]
        elif isinstance(i, int):
            self._tmp_bufs[i].reset()
        else:
            raise ValueError(f'{i} of type {type(i)} is not supported')

    def is_local_buffer_full(self, i=None):
        """ Returns if all local buffers are full """
        if i is None:
            if self._n_envs > 1:
                is_full = np.all([buf.is_full() for buf in self._tmp_bufs])
            else:
                is_full = self._tmp_bufs.is_full()
        elif isinstance(i, (list, tuple)):
            is_full = np.all([self._tmp_bufs[ii].is_full() for ii in i])
        elif isinstance(i, int):
            is_full = self._tmp_bufs[i].is_full()
        else:
            raise ValueError(f'{i} of type {type(i)} is not supported')
        return is_full

    def finish_episodes(self, i=None):
        """ Adds episodes in local buffers to memory """
        if i is None:
            if self._n_envs > 1:
                episodes = [buf.sample() for buf in self._tmp_bufs]
                [buf.reset() for buf in self._tmp_bufs]
            else:
                episodes = self._tmp_bufs.sample()
                self._tmp_bufs.reset()
        elif isinstance(i, (list, tuple)):
            episodes = [self._tmp_bufs[ii].sample() for ii in i]
            [self._tmp_bufs[ii].reset() for ii in i]
        elif isinstance(i, int):
            episodes = self._tmp_bufs[i].sample()
            self._tmp_bufs[i].reset()
        else:
            raise ValueError(f'{i} of type {type(i)} is not supported')
        self.merge(episodes)
        
    def merge(self, episodes):
        if episodes is None:
            return
        if isinstance(episodes, dict):
            episodes = [episodes]
        if not self._info_printed and episodes[0] is not None:
            print_buffer(episodes[0], 'Episodict')
            self._info_printed = True
        timestamp = datetime.now().strftime('%Y%m%dT%H%M%S')
        for eps in episodes:
            if eps is None or (self._sample_size 
                and len(next(iter(eps.values()))) < self._sample_size):
                continue    # ignore None/short episodes
            identifier = str(uuid.uuid4().hex)
            length = len(eps['reward'])
            filename = self._dir / f'{timestamp}-{identifier}-{length}.npz'
            self._memory[filename] = eps
            if self._save:
                save_data(filename, eps)
            self._filenames.append(filename)
        if self._save:
            self._remove_file()
        else:
            self._pop_episode()

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
        """ Samples a sequence """
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
        else:
            episode = {k: v[0] if k in self._state_keys
                        else v for k, v in episode.items()}

        return episode

    def _pop_episode(self):
        if len(self._memory) > self._max_episodes:
            filename = self._filenames.popleft()
            assert(filename in self._memory)
            del self._memory[filename]

    def _remove_file(self):
        if len(self._memory) > self._max_episodes:
            filename = self._filenames.popleft()
            assert(filename in self._memory)
            del self._memory[filename]
            filename.unlink()
            
    def clear_temp_bufs(self):
        self._tmp_bufs = []