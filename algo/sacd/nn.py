import logging
import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd

from utility.rl_utils import epsilon_greedy, epsilon_scaled_logits
from core.module import Module, Ensemble
from nn.func import mlp
from algo.sac.nn import Temperature
from algo.dqn.nn import Encoder, Q

logger = logging.getLogger(__name__)


class Actor(Module):
    def __init__(self, config, action_dim, name='actor'):
        super().__init__(name=name)
        config = config.copy()
        
        self._action_dim = action_dim
        prior = np.ones(action_dim, dtype=np.float32)
        prior /= np.sum(prior)
        self.prior = tf.Variable(prior, trainable=False, name='prior')
        self._epsilon_scaled_logits = config.pop('epsilon_scaled_logits', False)

        self._layers = mlp(
            **config, 
            out_size=action_dim,
            out_dtype='float32',
            name=name)
    
    @property
    def action_dim(self):
        return self._action_dim

    def call(self, x, evaluation=False, epsilon=0, temp=1):
        self.logits = logits = self._layers(x)

        if evaluation:
            temp = tf.where(temp == 0, 1e-9, temp)
            logits = logits / temp
            dist = tfd.Categorical(logits)
            action = dist.sample()
        else:
            if self._epsilon_scaled_logits and \
                    (isinstance(epsilon, tf.Tensor) or epsilon):
                self.logits = epsilon_scaled_logits(logits, epsilon, temp)
            else:
                self.logits = logits / temp

            dist = tfd.Categorical(self.logits)
            action = dist.sample()
            if not self._epsilon_scaled_logits and \
                    (isinstance(epsilon, tf.Tensor) or epsilon):
                action = epsilon_greedy(action, epsilon, True, action_dim=self.action_dim)

        self._dist = dist
        self._action = action

        return action

    def train_step(self, x):
        x = self._layers(x)
        pi = tf.nn.softmax(x)
        logpi = tf.math.log(tf.maximum(pi, 1e-8))    # bound logpi to avoid numerical instability
        return pi, logpi

    def update_prior(self, x, lr):
        self.prior.assign_add(lr * (x - self.prior))
    
    def compute_prob(self):
        # do not take into account epsilon and temperature to reduce variance
        prob = self._dist.prob(self._action)
        return prob


class SAC(Ensemble):
    def __init__(self, config, *, model_fn=None, env, **kwargs):
        model_fn = model_fn or create_components
        super().__init__(
            model_fn=model_fn, 
            config=config,
            env=env,
            **kwargs)

    @tf.function
    def action(self, x, evaluation=False, epsilon=0, temp=1., return_stats=False, return_eval_stats=False, **kwargs):
        if x.shape.ndims % 2 != 0:
            x = tf.expand_dims(x, axis=0)

        x = self.encoder(x)
        action = self.actor(x, evaluation=evaluation, epsilon=epsilon, temp=temp)
        terms = {}
        if return_eval_stats:
            action, terms = action
            q = self.q(x)
            q = tf.squeeze(q)
            idx = tf.stack([tf.range(action.shape[0]), action], -1)
            q = tf.gather_nd(q, idx)

            action_best_q = tf.argmax(q, 1)
            action_best_q = tf.squeeze(action_best_q)
            terms = {
                'action': action,
                'action_best_q': action_best_q,
                'q': q,
            }
        elif return_stats:
            q = self.q(x, action=action)
            q = tf.squeeze(q)
            terms['q'] = q
            if self.reward_kl:
                kl = -tfd.Categorical(self.actor.logits).entropy()
                if self.temperature.type == 'schedule':
                    _, temp = self.temperature(self._train_step)
                elif self.temperature.type == 'state':
                    raise NotImplementedError
                else:
                    _, temp = self.temperature()
                kl = temp * kl
                terms['kl'] = kl
        action = tf.squeeze(action)
        return action, terms

    @tf.function
    def value(self, x):
        if x.shape.ndims % 2 != 0:
            x = tf.expand_dims(x, axis=0)
        assert x.shape.ndims == 4, x.shape
        
        x = self.encoder(x)
        value = self.q(x)
        value = tf.squeeze(value)
        
        return value


def create_components(config, env):
    assert env.is_action_discrete
    config = config.copy()
    action_dim = env.action_dim
    encoder_config = config['encoder']
    actor_config = config['actor']
    q_config = config['q']
    temperature_config = config['temperature']
        
    models = dict(
        encoder=Encoder(encoder_config, name='encoder'),
        actor=Actor(actor_config, action_dim),
        q=Q(q_config, action_dim, name='q'),
        target_encoder=Encoder(encoder_config, name='target_encoder'),
        target_actor=Actor(actor_config, action_dim, name='target_actor'),
        target_q=Q(q_config, action_dim, name='target_q'),
        temperature=Temperature(temperature_config),
    )

    return models

def create_model(config, env, **kwargs):
    return SAC(config=config, env=env, **kwargs)
