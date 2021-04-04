import ray

from core.dataset import *


class RayDataset(Dataset):
    def name(self):
        return ray.get(self._buffer.name.remote())

    def good_to_learn(self):
        return ray.get(self._buffer.good_to_learn.remote())

    def _sample(self):
        while True:
            yield ray.get(self._buffer.sample.remote())

    def update_priorities(self, priorities, indices):
        self._buffer.update_priorities.remote(priorities, indices)


def get_dataformat(replay):
    import time
    i = 0
    while not ray.get(replay.good_to_learn.remote()):
        time.sleep(1)
        i += 1
        if i % 60 == 0:
            size = ray.get(replay.size.remote())
            if size == 0:
                import sys
                print('Replay does not collect any data in 60s. Specify data_format for dataset construction explicitly')
                sys.exit()
            print(f'Dataset Construction: replay size = {size}')
    data = ray.get(replay.sample.remote())
    data_format = {k: (v.shape, v.dtype) for k, v in data.items()}
    print('data format')
    for k, v in data_format.items():
        print('\t', k, v)
    return data_format