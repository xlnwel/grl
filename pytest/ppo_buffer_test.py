import random
import numpy as np

from utility.utils import standardize


gamma = .99
lam = .95
gae_discount = gamma * lam
config = dict(
    gamma=gamma,
    lam=lam,
    adv_type='gae',
    n_minibatches=2,
    n_envs=8, 
    N_STEPS=1000,
    N_MBS=4, 
    sample_size=10
)

class TestClass:
    def test_gae0(self):
        from algo.ppo.buffer import Buffer
        buffer = Buffer(config)
        n_envs = config['n_envs']
        n_steps = config['N_STEPS']
        d = np.zeros(n_envs)
        for i in range(n_steps):
            r = np.random.rand(n_envs)
            v = np.random.rand(n_envs)
            if np.random.randint(2):
                d[np.random.randint(n_envs)] = 1
            buffer.add(reward=r, value=v, discount=1-d)
        last_value = np.random.rand(n_envs)
        buffer.finish(last_value)

        memory = {k: v.copy().reshape(n_envs, -1) for k, v in buffer._memory.items()}
        mb_advs = np.zeros_like(memory['reward'])
        lastgaelam = 0
        for t in reversed(range(buffer._idx)):
            if t == buffer._idx - 1:
                nextdiscount = memory['discount'][:, t]
                nextvalues = last_value
            else:
                nextdiscount = memory['discount'][:, t]
                nextvalues = memory['value'][:, t+1]
            delta = memory['reward'][:, t] + gamma * nextvalues * nextdiscount - memory['value'][:, t]
            mb_advs[:, t] = lastgaelam = delta + gae_discount * nextdiscount * lastgaelam
        mb_advs = standardize(mb_advs)

        np.testing.assert_allclose(mb_advs, memory['advantage'], atol=1e-5)
        
    def test_gae2(self):
        from algo.ppo2.buffer import Buffer
        for prec in [16, 32]:
            config['precision'] = prec
            buffer = Buffer(config, state_keys=['h', 'c'])
            n_envs = config['n_envs']
            n_steps = config['N_STEPS']
            d = np.zeros(n_envs)
            m = np.ones(n_envs)
            for i in range(n_steps):
                r = np.random.rand(n_envs)
                v = np.random.rand(n_envs)
                h = np.random.rand(n_envs, 32)
                c = np.random.rand(n_envs, 32)
                if np.random.randint(2):
                    d[np.random.randint(n_envs)] = 1
                buffer.add(reward=r, value=v, discount=1-d, mask=m, c=c, h=h)
                m = 1 - d
            last_value = np.random.rand(n_envs)
            buffer.finish(last_value)

            memory = {k: v.copy().reshape((n_envs, -1)) for k, v in buffer._memory.items()}
            mb_advs = np.zeros_like(memory['reward'])
            lastgaelam = 0
            for t in reversed(range(buffer._idx)):
                if t == buffer._idx - 1:
                    nextdiscount = memory['discount'][:, t]
                    nextvalues = last_value
                else:
                    nextdiscount = memory['discount'][:, t]
                    nextvalues = memory['value'][:, t+1]
                delta = memory['reward'][:, t] + gamma * nextvalues * nextdiscount - memory['value'][:, t]
                mb_advs[:, t] = lastgaelam = delta + gae_discount * nextdiscount * lastgaelam
            # no mask is used here
            mb_advs = standardize(mb_advs)

            np.testing.assert_allclose(mb_advs, memory['advantage'], atol=1e-5)
            