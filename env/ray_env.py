import atexit
import numpy as np
import ray

from env.cls import *


class RayEnvVec(EnvVecBase):
    def __init__(self, EnvType, config, env_fn=make_env):
        self.name = config['name']
        self.n_workers= config.get('n_workers', 1)
        self.envsperworker = config.get('n_envs', 1)
        self.n_envs = self.envsperworker * self.n_workers
        RayEnvType = ray.remote(EnvType)
        # leave the name "envs" for consistency, albeit workers seems more appropriate
        if 'seed' in config:
            self.envs = [config.update({'seed': 100*i}) or RayEnvType.remote(config, env_fn) 
                    for i in range(self.n_workers)]
        else:
            self.envs = [RayEnvType.remote(config, env_fn) 
                    for i in range(self.n_workers)]

        self.env = EnvType(config, env_fn)
        self.max_episode_steps = self.env.max_episode_steps
        self._combine_func = np.stack if isinstance(self.env, Env) else np.concatenate
        self.env.close()

        super().__init__()

        atexit.register(self.close)

    def reset(self, idxes=None):
        out = self._remote_call('reset', idxes, single_output=False)
        return EnvOutput(*out)

    def random_action(self, *args, **kwargs):
        return self._combine_func(ray.get([env.random_action.remote() for env in self.envs]))

    def step(self, actions, **kwargs):
        actions = [np.squeeze(a) for a in np.split(actions, self.n_workers)]
        if kwargs:
            kwargs = {k: [np.squeeze(x) for x in np.split(v, self.n_workers)] 
                for k, v in kwargs.items()}
            kwargs = [dict(x) for x in zip(*[itertools.product([k], v) 
                for k, v in kwargs.items()])]
            out = ray.get([env.step.remote(a, **kw) 
                for env, a, kw in zip(self.envs, actions, kwargs)])
        else:
            out = ray.get([env.step.remote(a) 
                for env, a in zip(self.envs, actions)])
        # for i, os in enumerate(zip(*out)):
        #     for j, o in enumerate(os):
        #         print(i, j)
        #         if isinstance(o, dict):
        #             print(list(o))
        #         else:
        #             print(o.shape, o.dtype)
        out = [self._convert_batch(o, self._combine_func) for o in zip(*out)]
        return EnvOutput(*out)

    def score(self, idxes=None):
        return self._remote_call('score', idxes)

    def epslen(self, idxes=None):
        return self._remote_call('epslen', idxes)

    def mask(self, idxes=None):
        return self._remote_call('mask', idxes)

    def game_over(self, idxes=None):
        return self._remote_call('game_over', idxes)

    def prev_obs(self, idxes=None):
        return self._remote_call('prev_obs', idxes)

    def info(self, idxes=None, convert_batch=False):
        return self._remote_call(
            'info', idxes, convert_batch=convert_batch)
    
    def output(self, idxes=None):
        out = self._remote_call('output', idxes, single_output=False)
        
        return EnvOutput(*out)

    def _remote_call(self, name, idxes, single_output=True, convert_batch=True):
        """
        single_output: if the call produces only one output
        """
        method = lambda e: getattr(e, name)
        if idxes is None:
            out = ray.get([method(e).remote() for e in self.envs])
        else:
            if isinstance(self.env, Env):
                out = ray.get([method(self.envs[i]).remote() for i in idxes])
            else:
                new_idxes = [[] for _ in range(self.n_workers)]
                for i in idxes:
                    new_idxes[i // self.envsperworker].append(i % self.envsperworker)
                out = ray.get([method(self.envs[i]).remote(j) 
                    for i, j in enumerate(new_idxes) if j])

        if single_output:
            if isinstance(self.env, Env):
                return self._convert_batch(out) if convert_batch else out
            # for these outputs, we expect them to be of form [[out*], [out*]]
            # and we chain them into [out*]
            out = list(itertools.chain(*out))
            if convert_batch:
                # always stack as chain has flattened the data
                out = self._convert_batch(out)
            return out
        else:
            out = list(zip(*out))
            if convert_batch:
                out = [self._convert_batch(o, self._combine_func) for o in out]
            return out

    def close(self):
        ray.get([env.close.remote() for env in self.envs])
        self.env.close()
        del self
