import time
import numpy as np
import ray

from core.decorator import record
from core.base import AgentImpl
from utility.graph import video_summary


class Monitor(AgentImpl):
    @record
    def __init__(self, config):
        self._ready = np.zeros(config['n_workers'])
        
        self.time = time.time()
        self.env_step = 0
        self.last_env_step = 0
        self.last_train_step = 0
        self.MAX_STEPS = int(float(config['MAX_STEPS']))
        self._print_logs = getattr(self, '_print_logs', False)

    def sync_env_train_steps(self, learner):
        self.env_step, self.train_step = ray.get(
            learner.get_env_train_steps.remote())
        self.last_env_step = self.env_step

    def record_episodic_info(self, worker_name=None, **stats):
        video = stats.pop('video', None)
        if 'epslen' in stats:
            self.env_step += np.sum(stats['epslen'])
        if worker_name is not None:
            stats = {f'{k}_{worker_name}': v for k, v in stats.items()}
        self.store(**stats)
        if video is not None:
            video_summary(f'{self.name}/sim', video, step=self.env_step)

    def record_run_stats(self, worker_name=None, **stats):
        if worker_name is not None:
            stats = {f'{k}_{worker_name}': v for k, v in stats.items()}
        self.store(**stats)

    def record_train_stats(self, learner):
        train_step, stats = ray.get(learner.get_stats.remote())
        if train_step == 0:
            return
        duration = time.time() - self.time
        env_steps = self.env_step - self.last_env_step
        train_steps = train_step - self.last_train_step
        self.store(
            train_step=train_step, 
            env_step=self.env_step, 
            fps=env_steps / duration,
            tps=train_steps / duration,
            fpt=env_steps / train_steps,
            tpf=train_steps / env_steps,
            **stats)
        self.log(self.env_step, std=True, max=True, 
            print_terminal_info=self._print_logs)
        self.last_train_step = train_step
        self.last_env_step = self.env_step
        self.time = time.time()
        learner.save.remote(self.env_step)

    def is_over(self):
        return self.env_step > self.MAX_STEPS
