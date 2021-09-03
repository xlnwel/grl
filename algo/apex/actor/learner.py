import time
import threading
import psutil

from core.tf_config import *
from core.dataset import create_dataset
from utility.utils import config_attr
from utility.ray_setup import config_actor
from utility import pkg
from replay.func import create_replay
from env.func import create_env
from algo.apex.actor.actor import get_actor_base_class


def get_learner_base_class(AgentBase):
    ActorBase = get_actor_base_class(AgentBase)
    class LearnerBase(ActorBase):
        """ Only implements minimal functionality for learners """
        def start_learning(self):
            self._learning_thread = threading.Thread(
                target=self._learning, daemon=True)
            self._learning_thread.start()
            
        def _learning(self):
            # waits for enough data to learn
            while hasattr(self.dataset, 'good_to_learn') \
                    and not self.dataset.good_to_learn():
                time.sleep(1)
            print(f'{self.name} starts learning...')

            while True:
                self.learn_log()

        def get_weights(self, name=None):
            return self.model.get_weights(name=name)

        def get_train_step_weights(self, name=None):
            return self.train_step, self.model.get_weights(name=name)

        def get_stats(self):
            """ retrieve training stats for the monitor to record """
            return self.train_step, super().get_stats()

        def set_handler(self, **kwargs):
            config_attr(self, kwargs)
        
        def save(self, env_step):
            self.env_step = env_step
            super().save()

    return LearnerBase

def get_learner_class(AgentBase):
    LearnerBase = get_learner_base_class(AgentBase)
    class Learner(LearnerBase):
        def __init__(self,
                    model_fn,
                    replay,
                    config, 
                    model_config,
                    env_config,
                    replay_config):
            name = 'Learner'
            psutil.Process().nice(config.get('default_nice', 0))

            config_actor(name, config)

            # avoids additional workers created by RayEnvVec
            env_config['n_workers'] = 1
            env_config['n_envs'] = 1
            env = create_env(env_config)

            model = model_fn(config=model_config, env=env)

            dataset = self._create_dataset(
                replay, model, env, config, replay_config) 
            
            super().__init__(
                name=name,
                config=config, 
                models=model,
                dataset=dataset,
                env=env,
            )

            env.close()

        def merge(self, data):
            assert hasattr(self, 'replay'), f'There is no replay in {self.name}.\nDo you use a central replay?'
            self.replay.merge(data)
        
        def good_to_learn(self):
            assert hasattr(self, 'replay'), f'There is no replay in {self.name}.\nDo you use a central replay?'
            return self.replay.good_to_learn()

        def _create_dataset(self, replay, model, env, config, replay_config):
            am = pkg.import_module('agent', config=config, place=-1)
            data_format = am.get_data_format(
                env=env, replay_config=replay_config, 
                agent_config=config, model=model)
            if not getattr(self, 'use_central_buffer', True):
                assert replay is None, f'Replay({replay}) is not None for non-central buffer'
                self.replay = replay = create_replay(replay_config)
            dataset = create_dataset(
                replay, env, 
                data_format=data_format, 
                use_ray=getattr(self, '_use_central_buffer', True))
            
            return dataset

    return Learner
