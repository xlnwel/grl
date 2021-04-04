import numpy as np
import gym

from replay.func import create_replay, create_local_buffer
from replay.ds.sum_tree import SumTree
from utility.utils import set_global_seed


n_steps = 1
bs = 100
gamma = .99
capacity = 3000
config = dict(
    replay_type='uniform',
    n_steps=n_steps,
    gamma=gamma,
    batch_size=bs,
    min_size=7,
    capacity=capacity,
    has_next_obs=True,
)

class ReplayBuffer:
    def __init__(self, config):
        self.min_size = max(10000, config['batch_size']*10)
        self.batch_size = config['batch_size']
        self.ptr, self.size, self.max_size = 0, 0, int(float(config['capacity']))
        self._type = 'uniform'
        print('spinup')

    def name(self):
        return self._type

    def good_to_learn(self):
        return self.ptr >= self.min_size

    def add(self, obs, action, reward, next_obs, done, **kwargs):
        if not hasattr(self, 'obs'):
            obs_dim = obs.shape[0]
            act_dim = action.shape[0]
            self.obs = np.zeros([self.max_size, obs_dim], dtype=np.float32)
            self.next_obs = np.zeros([self.max_size, obs_dim], dtype=np.float32)
            self.action = np.zeros([self.max_size, act_dim], dtype=np.float32)
            self.reward = np.zeros(self.max_size, dtype=np.float32)
            self.done = np.zeros(self.max_size, dtype=np.bool)
            self.n_steps = np.ones(self.max_size, dtype=np.uint8)
        self.obs[self.ptr] = obs
        self.next_obs[self.ptr] = next_obs
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.done[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample(self):
        idxs = np.random.randint(0, self.size, size=self.batch_size)
        return dict(obs=self.obs[idxs],
                    next_obs=self.next_obs[idxs],
                    action=self.action[idxs],
                    reward=self.reward[idxs],
                    done=self.done[idxs],
                    steps=self.n_steps[idxs])


class TestClass:
    def test_buffer_op(self):
        replay = create_replay(config)
        simp_replay = ReplayBuffer(config)

        env = gym.make('BipedalWalkerHardcore-v3')

        s = env.reset()
        for i in range(10000):
            a = env.action_space.sample()
            ns, r, d, _ = env.step(a)
            if d:
                ns = env.reset()
            replay.add(obs=s.astype(np.float32), action=a.astype(np.float32), reward=np.float32(r), next_obs=ns.astype(np.float32), done=d)
            simp_replay.add(obs=s, action=a, reward=r, next_obs=ns, done=d)
            s = ns

            if i > 1000:
                set_global_seed(i)
                sample1 = replay.sample()
                set_global_seed(i)
                sample2 = simp_replay.sample()

                for k in sample1.keys():
                    np.testing.assert_allclose(sample1[k], sample2[k], err_msg=f'{k}')

    def test_sum_tree(self):
        for i in range(10):
            cap = np.random.randint(10, 20)
            st1 = SumTree(cap)
            st2 = SumTree(cap)

            # test update
            sz = np.random.randint(5, cap+1)
            priorities = np.random.uniform(size=sz)

            [st1.update(i, p) for i, p in enumerate(priorities)]
            st2.batch_update(np.arange(sz), priorities)
            np.testing.assert_allclose(st1._container, st2._container)
        
            # test find
            bs = np.random.randint(2, sz)
            intervals = np.linspace(0, st1.total_priorities, bs+1)
            values = np.random.uniform(intervals[:-1], intervals[1:])

            p1, idx1 = list(zip(*[st1.find(v) for v in values]))
            p2, idx2 = st2.batch_find(values)

            np.testing.assert_allclose(p1, p2)
            np.testing.assert_equal(idx1, idx2)
            np.testing.assert_array_less(0, p1)
            np.testing.assert_array_less(idx2, sz, err_msg=f'{values}\n{idx2}\n{st1.total_priorities}')

            # validate sum tree's internal structure
            nodes = np.arange(st1._tree_size)
            left, right = 2 * nodes + 1, 2 * nodes + 2
            np.testing.assert_allclose(st1._container[nodes], st1._container[left] + st1._container[right])
            np.testing.assert_allclose(st2._container[nodes], st2._container[left] + st2._container[right])

            # test update with the same indices
            sz = np.random.randint(5, cap+1)
            idxes = np.ones(sz) * (i % cap)
            priorities = np.random.uniform(size=sz)
            [st1.update(i, p) for i, p in enumerate(priorities)]
            st2.batch_update(np.arange(sz), priorities)
            np.testing.assert_allclose(st1._container, st2._container)
    
    def test_sequential_buffer(self):
        config = dict(
            replay_type='seqper',                      # per or uniform
            # arguments for general replay
            n_envs=32,
            seqlen=16,
            reset_shift=2,
            state_keys=['h', 'c', 'prev_reward'],
            extra_keys=['obs', 'action', 'mu', 'mask']
        )
        obs_shape = (4,)
        for n_envs in np.arange(1, 3):
            config['n_envs'] = n_envs
            for reset_shift in np.arange(0, config['seqlen']):
                config['reset_shift'] = reset_shift
                buff = create_local_buffer(config)
                start_value = 0
                for i in range(start_value, 100):
                    if n_envs == 1:
                        o = np.ones(obs_shape) * i
                        a = r = i
                    else:
                        o = np.ones((n_envs, *obs_shape)) * i
                        a = r = np.ones(n_envs) * i
                    buff.add(obs=o, action=a, reward=r, h=r)
                    if buff.is_full():
                        data = buff.sample()
                        if isinstance(data, list):
                            data = data[0]
                        np.testing.assert_equal(data['obs'][:, 0], np.arange(start_value, start_value + config['seqlen']+1))
                        np.testing.assert_equal(data['action'], np.arange(start_value, start_value + config['seqlen']+1))
                        np.testing.assert_equal(data['reward'], np.arange(start_value, start_value + config['seqlen']))
                        np.testing.assert_equal(data['h'], start_value)
                        buff.reset()
                        start_value += reset_shift or config['seqlen']

    def test_sequential_buffer_random(self):
        config = dict(
            replay_type='seqper',                      # per or uniform
            # arguments for general replay
            n_envs=32,
            seqlen=16,
            reset_shift=2,
            state_keys=['h', 'c', 'prev_reward'],
            extra_keys=['obs', 'action', 'mu', 'mask']
        )
        env_config = dict(
            n_envs=1,
            name='dummy'
        )
        from env.dummy import DummyEnv
        from env import wrappers
        from env.func import create_env
        def mkenv(config):
            env = DummyEnv(**config)
            env = wrappers.post_wrap(env, config)
            return env
        for n_envs in np.arange(2, 3):
            config['n_envs'] = n_envs
            env_config['n_envs'] = n_envs
            for burn_in_size in np.arange(0, config['seqlen']):
                config['burn_in_size'] = burn_in_size
                buff = create_local_buffer(config)
                env = create_env(env_config, mkenv)
                out = env.output()
                o, prev_reward, d, reset = out
                for i in range(1, 1000):
                    a = np.random.randint(0, 10, n_envs)
                    no, r, d, reset = env.step(a)
                    print(r)
                    if n_envs == 1:
                        h = np.ones(2) * r
                        c = np.ones(2) * r
                    else:
                        h = np.ones((n_envs, 2)) * r[:, None]
                        c = np.ones((n_envs, 2)) * r[:, None]
                    buff.add(obs=o, reward=r, discount=d, h=h, c=c, mask=1-reset, prev_reward=prev_reward)
                    if buff.is_full():
                        data_list = buff.sample()
                        if n_envs == 1:
                            data_list = [data_list]
                        for data in data_list:
                            np.testing.assert_equal(data['reward'][0], data['h'][0])
                            np.testing.assert_equal(data['obs'][0, 0], data['c'][0])
                        buff.reset()
                    prev_reward = r
                    o = no
    
    def test_sper(self):
        config = dict(
            replay_type='seqper',                      # per or uniform
            precision=32,
            # arguments for PER
            beta0=0.4,
            to_update_top_priority=False,

            # arguments for general replay
            batch_size=2,
            sample_size=7,
            burn_in_size=2,
            min_size=2,
            capacity=10000,
            state_keys=['h', 'c', 'prev_reward'],
            extra_keys=['obs', 'action', 'mu', 'mask']
        )
        env_config = dict(
            n_envs=1,
            name='dummy'
        )
        from env.dummy import DummyEnv
        from env import wrappers
        from env.func import create_env
        def mkenv(config):
            env = DummyEnv(**config)
            env = wrappers.post_wrap(env, config)
            return env
        for n_envs in np.arange(2, 3):
            config['n_envs'] = n_envs
            env_config['n_envs'] = n_envs
            for burn_in_size in np.arange(0, config['sample_size']):
                config['burn_in_size'] = burn_in_size
                replay = create_replay(config)
                env = create_env(env_config, mkenv)
                out = env.output()
                o, prev_reward, d, reset = out
                for i in range(1, 10000):
                    a = np.random.randint(0, 10, n_envs)
                    no, r, d, reset = env.step(a)
                    if n_envs == 1:
                        h = np.ones(2) * r
                        c = np.ones(2) * r
                    else:
                        h = np.ones((n_envs, 2)) * r[:, None]
                        c = np.ones((n_envs, 2)) * r[:, None]
                    replay.add(obs=o, reward=r, discount=d, h=h, c=c, mask=1-reset, prev_reward=prev_reward)
                    if replay.good_to_learn():
                        data = replay.sample()
                        np.testing.assert_equal(data['reward'][:, 0], data['h'][:, 0])
                        np.testing.assert_equal(data['obs'][:, 0, 0], data['c'][:, 0])
                    o = no
                    prev_reward = r
