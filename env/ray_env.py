import numpy as np
import ray

from env.cls import *


class RayEnvVec(EnvVecBase):
    def __init__(self, EnvType, config, env_fn=make_env):
        self.name = config['name']
        config['np_obs'] = True
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
        super().__init__()

    def reset(self, idxes=None):
        output = self._remote_call('reset', idxes, single_output=False)

        if isinstance(self.env, Env):
            output = [np.stack(o, 0) for o in output]
        else:
            output = [np.concatenate(o, 0) for o in output]
        return EnvOutput(*output)

    def random_action(self, *args, **kwargs):
        return np.reshape(ray.get([env.random_action.remote() for env in self.envs]), 
                          (self.n_envs, *self.action_shape))

    def step(self, actions, **kwargs):
        actions = np.squeeze(actions.reshape(self.n_workers, self.envsperworker, *self.action_shape))
        if kwargs:
            kwargs = dict([(k, np.squeeze(v.reshape(self.n_workers, self.envsperworker, -1))) for k, v in kwargs.items()])
            kwargs = [dict(x) for x in zip(*[itertools.product([k], v) for k, v in kwargs.items()])]
            output = ray.get([env.step.remote(a, **kw) 
                for env, a, kw in zip(self.envs, actions, kwargs)])
        else:
            output = ray.get([env.step.remote(a) 
                for env, a in zip(self.envs, actions)])
        output = list(zip(*output))

        if isinstance(self.env, Env):
            output = [np.stack(o, 0) for o in output]
        else:
            output = [np.concatenate(o, 0) for o in output]
        return EnvOutput(*output)

    def score(self, idxes=None):
        return self._remote_call('score', idxes)

    def epslen(self, idxes=None):
        return self._remote_call('epslen', idxes)
        
    def game_over(self, idxes=None):
        return self._remote_call('game_over', idxes)

    def prev_obs(self, idxes=None):
        return self._remote_call('prev_obs', idxes)

    def info(self, idxes=None):
        return self._remote_call('info', idxes)
    
    def output(self, idxes=None):
        output = self._remote_call('output', idxes, single_output=False)

        if isinstance(self.env, Env):
            output = [np.stack(o, 0) for o in output]
        else:
            output = [np.concatenate(o, 0) for o in output]
        return EnvOutput(*output)

    def _remote_call(self, name, idxes, single_output=True):
        """
        single_output: if the call produces only one output
        """
        method = lambda e: getattr(e, name)
        if idxes is None:
            output = ray.get([method(e).remote() for e in self.envs])
        else:
            if isinstance(self.env, Env):
                output = ray.get([method(self.envs[i]).remote() for i in idxes])
            else:
                new_idxes = [[] for _ in range(self.n_workers)]
                for i in idxes:
                    new_idxes[i // self.envsperworker].append(i % self.envsperworker)
                output = ray.get([method(self.envs[i]).remote(j) 
                    for i, j in enumerate(new_idxes) if j])
        if single_output:
            if isinstance(self.env, Env):
                return output
            # for these outputs, we expect them to be of form [[output*], [output*]]
            # and we chain them into [output*]
            return list(itertools.chain(*output))
        else:
            return list(zip(*output))

    def close(self):
        del self
