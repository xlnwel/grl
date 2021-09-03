import collections

from utility.ray_setup import config_actor
from utility.run import evaluate
from utility.utils import Every
from env.func import create_env
from algo.apex.actor.actor import get_actor_base_class


def get_evaluator_class(AgentBase):
    ActorBase = get_actor_base_class(AgentBase)
    class Evaluator(ActorBase):
        """ Initialization """
        def __init__(self, 
                    *,
                    config,
                    name='Evaluator',
                    model_config,
                    env_config,
                    model_fn):
            config_actor(name, config)

            for k in list(env_config.keys()):
                # pop reward hacks
                if 'reward' in k:
                    env_config.pop(k)
            self.env = create_env(env_config)

            model = model_fn(
                config=model_config, 
                env=self.env)
            
            super().__init__(
                name=name,
                config=config, 
                models=model,
                dataset=None,
                env=self.env,
            )

            # the names of network modules that should be in sync with the learner
            if not hasattr(self, '_pull_names'):
                self._pull_names = [k for k in self.model.keys() if 'target' not in k]

            # used for recording evaluator side info 
            self._info = collections.defaultdict(list)

        """ Evaluator Methods """
        def run(self, learner, monitor):
            step = 0
            if getattr(self, 'RECORD_PERIOD', False):
                # how often to record videos
                to_record = Every(self.RECORD_PERIOD)
            else:
                to_record = lambda x: False 

            while True:
                step += 1
                self.pull_weights(learner)
                self._run(record=to_record(step))
                self._send_episodic_info(monitor)

        def _run(self, record):
            score, epslen, video = evaluate(self.env, self, 
                record=record, n=self.N_EVALUATION)
            self.store(score, epslen, video)

        def store(self, score, epslen, video):
            self._info['eval_score'] += score
            self._info['eval_epslen'] += epslen
            if video is not None:
                self._info['video'] = video

        def _send_episodic_info(self, monitor):
            if self._info:
                monitor.record_episodic_info.remote(**self._info)
                self._info.clear()
    
    return Evaluator
def get_evaluator_class(AgentBase):
    ActorBase = get_actor_base_class(AgentBase)
    class Evaluator(ActorBase):
        """ Initialization """
        def __init__(self, 
                    *,
                    config,
                    name='Evaluator',
                    model_config,
                    env_config,
                    model_fn):
            config_actor(name, config)

            for k in list(env_config.keys()):
                # pop reward hacks
                if 'reward' in k:
                    env_config.pop(k)
            self.env = create_env(env_config)

            model = model_fn(
                config=model_config, 
                env=self.env)
            
            super().__init__(
                name=name,
                config=config, 
                models=model,
                dataset=None,
                env=self.env,
            )

            # the names of network modules that should be in sync with the learner
            if not hasattr(self, '_pull_names'):
                self._pull_names = [k for k in self.model.keys() if 'target' not in k]

            # used for recording evaluator side info 
            self._info = collections.defaultdict(list)

        """ Evaluator Methods """
        def run(self, learner, monitor):
            step = 0
            if getattr(self, 'RECORD_PERIOD', False):
                # how often to record videos
                to_record = Every(self.RECORD_PERIOD)
            else:
                to_record = lambda x: False 

            while True:
                step += 1
                self.pull_weights(learner)
                self._run(record=to_record(step))
                self._send_episodic_info(monitor)

        def _run(self, record):
            score, epslen, video = evaluate(self.env, self, 
                record=record, n=self.N_EVALUATION)
            self.store(score, epslen, video)

        def store(self, score, epslen, video):
            self._info['eval_score'] += score
            self._info['eval_epslen'] += epslen
            if video is not None:
                self._info['video'] = video

        def _send_episodic_info(self, monitor):
            if self._info:
                monitor.record_episodic_info.remote(**self._info)
                self._info.clear()
    
    return Evaluator
