import tensorflow as tf

from utility.rl_loss import n_step_target
from utility.tf_utils import explained_variance
from core.decorator import override
from algo.dqn.base import DQNBase, get_data_format, collect
from algo.sacd.base import TempLearner

class Agent(DQNBase, TempLearner):
    # @tf.function
    # def summary(self, data, terms):
    #     tf.summary.histogram('learn/entropy', terms['entropy'], step=self._env_step)
    #     tf.summary.histogram('learn/reward', data['reward'], step=self._env_step)

    @override(DQNBase)
    @tf.function
    def _learn(self, obs, action, reward, next_obs, discount, steps=1, IS_ratio=1):
        terms = {}
        if self.temperature.type == 'schedule':
            _, temp = self.temperature(self._train_step)
        elif self.temperature.type == 'state':
            raise NotImplementedError
        else:
            _, temp = self.temperature()

        next_x = self.target_encoder(next_obs)
        next_pi, next_logpi = self.target_actor.train_step(next_x)
        next_qs = self.target_q(next_x)
        if self._soft_target:
            next_value = tf.reduce_sum(next_pi 
                * (next_qs - temp * next_logpi), axis=-1)
        else:
            next_value = tf.reduce_sum(next_pi * next_qs, axis=-1)
        q_target = n_step_target(reward, next_value, discount, 
            self._gamma, steps, tbo=self._tbo)
        tf.debugging.assert_shapes([
            [next_value, (None,)],
            [reward, (None,)],
            [next_pi, (None, self._action_dim)],
            [next_logpi, (None, self._action_dim)],
            [next_qs, (None, self._action_dim)],
            [q_target, (None)],
        ])
        with tf.GradientTape() as tape:
            x = self.encoder(obs)
            q, qs, error, value_losss = self._compute_value_losss(
                self.q, x, action, q_target, IS_ratio)
        terms['value_norm'] = self._value_opt(tape, value_losss)

        with tf.GradientTape() as tape:
            pi, logpi = self.actor.train_step(x)
            v = tf.reduce_sum(pi * qs, axis=-1)
            entropy = - tf.reduce_sum(pi * logpi, axis=-1)
            actor_loss = -(v + temp * entropy)
            tf.debugging.assert_shapes([[actor_loss, (None, )]])
            actor_loss = tf.reduce_mean(IS_ratio * actor_loss)
        terms['actor_norm'] = self._actor_opt(tape, actor_loss)

        temp_terms = self._learn_temp(x, entropy, IS_ratio)

        if self._is_per:
            priority = self._compute_priority(error)
            terms['priority'] = priority
            
        terms.update(dict(
            steps=steps,
            reward_min=tf.reduce_min(reward),
            actor_loss=actor_loss,
            v=v, 
            entropy=entropy,
            entropy_max=tf.reduce_max(entropy),
            entropy_min=tf.reduce_min(entropy),
            value_loss=value_losss, 
            temp=temp,
            explained_variance_q=explained_variance(q_target, q),
            **temp_terms
        ))

        return terms

    def _compute_value_losss(self, q_fn, x, action, returns, IS_ratio):
        qs = q_fn(x)
        q = tf.reduce_sum(qs * action, axis=-1)
        q_error = returns - q
        value_losss = .5 * tf.reduce_mean(IS_ratio * q_error**2)
        return q, qs, tf.abs(q_error), value_losss
