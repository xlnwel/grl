import ray

from utility.utils import config_attr


def get_actor_base_class(AgentBase):
    """" Mixin that defines some basic operations for remote actor """
    class ActorBase(AgentBase):
        def pull_weights(self, learner):
            if getattr(self, '_normalize_obs', False):
                obs_rms = ray.get(learner.get_obs_rms_stats.remote())
                self.set_rms_stats(obs_rms=obs_rms)
            train_step, weights = ray.get(
                learner.get_train_step_weights.remote(self._pull_names))
            self.train_step = train_step
            self.model.set_weights(weights)
        
        def set_train_step_weights(self, train_step, weights):
            self.train_step = train_step
            self.model.set_weights(weights)

        def set_handler(self, **kwargs):
            config_attr(self, kwargs)
        
        @classmethod
        def as_remote(cls, **kwargs):
            return ray.remote(**kwargs)(cls)

    return ActorBase
