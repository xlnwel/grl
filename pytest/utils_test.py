import numpy as np

from utility.utils import *


class TestClass:
    def test_moments(self):
        # moments along all dimensions
        for x in [np.random.randn(10), np.random.randn(10, 3)]:
            mask = np.random.randint(2, size=x.shape[:1])
            mean, std = moments(x, mask=mask)
            x2 = []
            for i, m in enumerate(mask):
                if m == 1:
                    x2.append(x[i])
            x2 = np.array(x2)
            print(np.sum(x2, axis=0), np.sum(mask))
            np.testing.assert_allclose(mean, np.mean(x2))
            np.testing.assert_allclose(std, np.std(x2))

        for x in [np.random.randn(10), np.random.randn(10, 3)]:
            mask = np.random.randint(2, size=x.shape[:1])
            mean, std = moments(x, axis=0, mask=mask)
            x2 = []
            for i, m in enumerate(mask):
                if m == 1:
                    x2.append(x[i])
            x2 = np.array(x2)
            print(np.sum(x2, axis=0), np.sum(mask))
            np.testing.assert_allclose(mean, np.mean(x2, axis=0))
            np.testing.assert_allclose(std, np.std(x2, axis=0))

    def test_standardize(self):
        epsilon = 1e-5
        x = np.arange(1000).reshape((2, 50, 2, 5))
        mask = np.ones(100).reshape((2, 50, 1))
        mask[1, 14:] = 0
        result = standardize(x, epsilon=epsilon, mask=mask).flatten()
        idx = sum(result != 0)
        result = result[:idx]
        x_masked = x.flatten()[:idx]
        base = standardize(x_masked, epsilon=epsilon)

        np.testing.assert_allclose(result, base)

    def test_runningmeanstd(self):
        for (x1, x2, x3) in [
            (np.random.randn(), np.random.randn(), np.random.randn()),
            (np.random.randn(2), np.random.randn(2), np.random.randn(2)),
            ]:
            stats = RunningMeanStd(axis=None, epsilon=0)
            stats.update(x1)
            stats.update(x2)
            stats.update(x3)
            x = np.stack([x1, x2, x3], axis=0)
            mean, std = np.mean(x, axis=0), np.std(x, axis=0)
            var = np.square(std)
            np.testing.assert_allclose(mean, stats._mean)
            np.testing.assert_allclose(var, stats._var)

        for (x1, x2, x3) in [
            (np.random.randn(3), np.random.randn(4), np.random.randn(5)),
            (np.random.randn(3,2), np.random.randn(4,2), np.random.randn(5,2)),
            ]:
            stats = RunningMeanStd(axis=0, epsilon=0)
            stats.update(x1)
            stats.update(x2)
            stats.update(x3)
            x = np.concatenate([x1, x2, x3], axis=0)
            mean, std = moments(x, axis=0)
            var = np.square(std)
            np.testing.assert_allclose(mean, stats._mean)
            np.testing.assert_allclose(var, stats._var)

        # test mask
        for (x1, x2, x3) in [
            (np.random.randn(3), np.random.randn(3), np.random.randn(3)),
            (np.random.randn(3,2), np.random.randn(3,2), np.random.randn(3,2)),
            ]:
            while True:
                m = [np.random.randint(2, size=x.shape[:1]) for x in (x1, x2, x3)]
                if not np.any([np.all(x == 0) for x in m]):
                    break
            x = np.stack([x1, x2, x3], axis=0)
            m = np.stack(m, axis=0)
            stats = RunningMeanStd(axis=tuple(range(m.ndim)), epsilon=0)
            stats.update(x, mask=m)
            mean, std = moments(x, axis=tuple(range(m.ndim)), mask=m)
            var = np.square(std)
            np.testing.assert_allclose(mean, stats._mean)
            np.testing.assert_allclose(var, stats._var)

            x_2 = np.random.randn(*x.shape)
            m_2 = np.random.randint(2, size=m.shape)
            stats.update(x_2, m_2)
            x = np.concatenate([x, x_2], axis=0)
            mean, std = moments(x, axis=tuple(range(m.ndim)), mask=np.concatenate([m, m_2], axis=0))
            var = np.square(std)
            np.testing.assert_allclose(mean, stats._mean)
            np.testing.assert_allclose(var, stats._var)

        for (x1, x2, x3) in [
            (np.random.randn(3), np.random.randn(4), np.random.randn(5)),
            (np.random.randn(3,2), np.random.randn(4,2), np.random.randn(5,2)),
            ]:
            while True:
                m = [np.random.randint(2, size=x.shape[:1]) for x in (x1, x2, x3)]
                if not np.any([np.all(x == 0) for x in m]):
                    break
            stats = RunningMeanStd(axis=0, epsilon=0)
            stats.update(x1, mask=m[0])
            stats.update(x2, mask=m[1])
            stats.update(x3, mask=m[2])
            x = np.concatenate([x1, x2, x3], axis=0)
            m = np.concatenate(m, axis=0)
            mean, std = moments(x, axis=tuple(range(m.ndim)), mask=m)
            var = np.square(std)
            np.testing.assert_allclose(mean, stats._mean)
            np.testing.assert_allclose(var, stats._var)