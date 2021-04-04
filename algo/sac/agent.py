import tensorflow as tf

from utility.rl_loss import n_step_target
from utility.tf_utils import explained_variance
from algo.dqn.base import DQNBase, get_data_format, collect


class Agent(DQNBase):
    @tf.function
    def _learn(self, obs, action, reward, next_obs, discount, steps=1, IS_ratio=1):
        next_action, next_logpi, _ = self.actor.train_step(next_obs)
        next_q_with_actor = self.target_q(next_obs, next_action)
        next_q2_with_actor = self.target_q2(next_obs, next_action)
        next_q_with_actor = tf.minimum(next_q_with_actor, next_q2_with_actor)
        if self.temperature.type == 'schedule':
            _, temp = self.temperature(self._train_step)
        elif self.temperature.type == 'state':
            _, temp = self.temperature(next_obs)
        else:
            _, temp = self.temperature()
        next_value = next_q_with_actor - temp * next_logpi
        q_target = n_step_target(reward, next_value, discount, 
            self._gamma, steps, tbo=self._tbo)

        terms = {}
        with tf.GradientTape() as tape:
            q = self.q(obs, action)
            q2 = self.q2(obs, action)
            q_error = q_target - q
            q2_error = q_target - q2
            q_losss = .5 * tf.reduce_mean(IS_ratio * q_error**2)
            q2_loss = .5 * tf.reduce_mean(IS_ratio * q2_error**2)
            value_loss = q_losss + q2_loss
        terms['value_norm'] = self._value_opt(tape, value_loss)

        with tf.GradientTape() as actor_tape:
            action, logpi, actor_terms = self.actor.train_step(obs)
            terms.update(actor_terms)
            q_with_actor = self.q(obs, action)
            q2_with_actor = self.q2(obs, action)
            q_with_actor = tf.minimum(q_with_actor, q2_with_actor)
            actor_loss = tf.reduce_mean(IS_ratio * (temp * logpi - q_with_actor))
        self._actor_opt(actor_tape, actor_loss)

        if self.temperature.is_trainable():
            target_entropy = getattr(self, '_target_entropy', -self._action_dim)
            with tf.GradientTape() as temp_tape:
                log_temp, temp = self.temperature(obs)
                temp_loss = -tf.reduce_mean(IS_ratio * log_temp 
                    * tf.stop_gradient(logpi + target_entropy))
            self._temp_opt(temp_tape, temp_loss)
            terms.update(dict(
                temp=temp,
                temp_loss=temp_loss,
            ))

        if self._is_per:
            priority = self._compute_priority((tf.abs(q_error) + tf.abs(q2_error)) / 2.)
            terms['priority'] = priority
            
        terms.update(dict(
            actor_loss=actor_loss,
            q=q, 
            q2=q2,
            logpi=logpi,
            q_target=q_target,
            value_loss=value_loss, 
            explained_variance_q=explained_variance(q_target, q),
        ))

        return terms
