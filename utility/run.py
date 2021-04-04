import collections
import logging
import numpy as np

from env.wrappers import get_wrapper_by_name

logger = logging.getLogger(__name__)

class RunMode:
    NSTEPS='nsteps'
    TRAJ='traj'


class Runner:
    def __init__(self, env, agent, step=0, nsteps=None, run_mode=RunMode.NSTEPS):
        self.env = env
        if env.max_episode_steps == int(1e9):
            logger.info(f'Maximum episode steps is not specified'
                f'and is by default set to {self.env.max_episode_steps}')
            # assert nsteps is not None
        self.agent = agent
        self.step = step
        if run_mode == RunMode.TRAJ and env.env_type == 'EnvVec':
            logger.warning('Runner.step is not the actual environment steps '
                f'as run_mode == {RunMode.TRAJ} and env_type == EnvVec')
        self.env_output = self.env.output()
        self.episodes = np.zeros(env.n_envs)
        assert get_wrapper_by_name(self.env, 'EnvStats').auto_reset
        self.run = {
            f'{RunMode.NSTEPS}-Env': self._run_env,
            f'{RunMode.NSTEPS}-EnvVec': self._run_envvec,
            f'{RunMode.TRAJ}-Env': self._run_traj_env,
            f'{RunMode.TRAJ}-EnvVec': self._run_traj_envvec,
        }[f'{run_mode}-{self.env.env_type}']

        self._frame_skip = getattr(env, 'frame_skip', 1)
        self._frames_per_step = self.env.n_envs * self._frame_skip
        self._default_nsteps = nsteps or env.max_episode_steps // self._frame_skip

    def _run_env(self, *, action_selector=None, step_fn=None, nsteps=None):
        action_selector = action_selector or self.agent
        nsteps = nsteps or self._default_nsteps
        obs = self.env_output.obs
        reset = self.env_output.reset
        
        for t in range(nsteps):
            action = action_selector(
                self.env_output,
                evaluation=False)
            obs, reset = self.step_env(obs, action, step_fn)

            # logging when env is reset 
            if reset:
                info = self.env.info()
                if 'score' in info:
                    self.agent.store(
                        score=info['score'], epslen=info['epslen'])
                    self.episodes += 1

        return self.step

    def _run_envvec(self, *, action_selector=None, step_fn=None, nsteps=None):
        action_selector = action_selector or self.agent
        nsteps = nsteps or self._default_nsteps
        obs = self.env_output.obs
        reset = self.env_output.reset
        
        for t in range(nsteps):
            action = action_selector(
                self.env_output,
                evaluation=False)
            obs, reset = self.step_env(obs, action, step_fn)
            
            # logging when any env is reset 
            done_env_ids = [i for i, r in enumerate(reset) if r]
            if done_env_ids:
                info = self.env.info(done_env_ids)
                # further filter done caused by life loss
                done_env_ids = [k for k, i in enumerate(info) if i.get('game_over')]
                info = [info[i] for i in done_env_ids]
                score = [i['score'] for i in info]
                epslen = [i['epslen'] for i in info]
                self.agent.store(score=score, epslen=epslen)
                self.episodes[done_env_ids] += 1

        return self.step

    def _run_traj_env(self, action_selector=None, step_fn=None):
        action_selector = action_selector or self.agent
        obs = self.env_output.obs
        reset = self.env_output.reset
        
        for t in range(self._default_nsteps):
            action = action_selector(
                self.env_output,
                evaluation=False)
            obs, reset = self.step_env(obs, action, step_fn)

            if reset:
                break
        
        info = self.env.info()
        self.agent.store(
            score=info['score'], epslen=info['epslen'])
        self.episodes += 1
                
        return self.step

    def _run_traj_envvec(self, action_selector=None, step_fn=None):
        action_selector = action_selector or self.agent
        self.env_output = self.env.reset()  # explicitly reset envvect to turn off auto-reset
        obs = self.env_output.obs
        
        for t in range(self._default_nsteps):
            action = action_selector(
                self.env_output,
                evaluation=False)
            obs, _ = self.step_env(obs, action, step_fn, mask=True)

            # logging when any env is reset 
            if np.all(self.env_output.discount == 0):
                break
        info = self.env.info()
        score = [i['score'] for i in info]
        epslen = [i['epslen'] for i in info]
        self.agent.store(score=score, epslen=epslen)
        self.episodes += 1

        return self.step

    def step_env(self, obs, action, step_fn, mask=False):
        if isinstance(action, tuple):
            if len(action) == 2:
                action, terms = action
                self.env_output = self.env.step(action)
                self.step += self._frames_per_step
            elif len(action) == 3:
                action, frame_skip, terms = action
                frame_skip += 1     # plus 1 as values returned start from zero
                self.env_output = self.env.step(action, frame_skip=frame_skip)
                self.step += np.sum(frame_skip)
            else:
                raise ValueError(f'Invalid action "{action}"')
        else:
            self.env_output = self.env.step(action)
            self.step += self._frames_per_step
            terms = {}
        next_obs, reward, discount, reset = self.env_output
        
        if step_fn:
            kwargs = dict(obs=obs, action=action, reward=reward,
                discount=discount, next_obs=next_obs)
            if mask:
                kwargs['mask'] = self.env.mask()
            assert 'reward' not in terms, 'reward in terms is from the preivous timestep and should not be used to override here'
            # allow terms to overwrite the values in kwargs
            kwargs.update(terms)
            step_fn(self.env, self.step, reset, **kwargs)

        return next_obs, reset

def evaluate(env, 
             agent, 
             n=1, 
             record=False, 
             size=None, 
             video_len=1000, 
             step_fn=None, 
             record_stats=False,
             n_windows=4):
    assert get_wrapper_by_name(env, 'EnvStats') is not None
    scores = []
    epslens = []
    max_steps = env.max_episode_steps // getattr(env, 'frame_skip', 1)
    maxlen = min(video_len, max_steps)
    frames = [collections.deque(maxlen=maxlen) 
        for _ in range(min(n_windows, env.n_envs))]
    if hasattr(agent, 'reset_states'):
        agent.reset_states()
    env_output = env.reset()
    n_run_eps = env.n_envs  # count the number of episodes that has begun to run
    n = max(n, env.n_envs)
    n_done_eps = 0
    frame_skip = None
    obs = env_output.obs
    while n_done_eps < n:
        for k in range(max_steps):
            if record:
                img = env.get_screen(size=size)
                if env.env_type == 'Env':
                    frames[0].append(img)
                else:
                    for i in range(len(frames)):
                        frames[i].append(img[i])
                    
            action = agent(
                env_output, 
                evaluation=True, 
                return_eval_stats=record_stats)
            terms = {}
            if isinstance(action, tuple):
                if len(action) == 2:
                    action, terms = action
                elif len(action) == 3:
                    action, frame_skip, terms = action
                else:
                    raise ValueError(f'Unkown model return: {action}')
            if frame_skip is not None:
                frame_skip += 1     # plus 1 as values returned start from zero
                env_output = env.step(action, frame_skip=frame_skip)
            else:
                env_output = env.step(action)
            next_obs, reward, discount, reset = env_output

            if step_fn:
                step_fn(obs=obs, action=action, reward=reward, 
                    discount=discount, next_obs=next_obs, 
                    reset=reset, **terms)
            obs = next_obs
            if env.env_type == 'Env':
                if env.game_over():
                    scores.append(env.score())
                    epslens.append(env.epslen())
                    n_done_eps += 1
                    if n_run_eps < n:
                        n_run_eps += 1
                        env_output = env.reset()
                        if hasattr(agent, 'reset_states'):
                            agent.reset_states()
                    break
            else:
                done_env_ids = [i for i, (d, m) in enumerate(zip(env.game_over(), env.mask())) if d and m]
                n_done_eps += len(done_env_ids)
                if done_env_ids:
                    score = env.score(done_env_ids)
                    epslen = env.epslen(done_env_ids)
                    scores += score
                    epslens += epslen
                    if n_run_eps < n:
                        reset_env_ids = done_env_ids[:n-n_run_eps]
                        n_run_eps += len(reset_env_ids)
                        eo = env.reset(reset_env_ids)
                        for t, s in zip(env_output, eo):
                            for i, ri in enumerate(reset_env_ids):
                                t[ri] = s[i]
                    elif n_done_eps == n:
                        break

    if record:
        max_len = np.max([len(f) for f in frames])
        # padding to make all sequences of the same length
        for i, f in enumerate(frames):
            while len(f) < max_len:
                f.append(f[-1])
            frames[i] = np.array(f)
        frames = np.array(frames)
        return scores, epslens, frames
    else:
        return scores, epslens, None
