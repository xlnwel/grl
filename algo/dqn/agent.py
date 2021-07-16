import tensorflow as tf

from utility.tf_utils import softmax, log_softmax, explained_variance
from utility.rl_loss import n_step_target, huber_loss
from algo.dqn.base import DQNBase, get_data_format, collect


class Agent(DQNBase):
    @tf.function
    def _learn(self, obs, action, reward, next_obs, discount, steps=1, IS_ratio=1):
        loss_fn = dict(
            huber=huber_loss, mse=lambda x: .5 * x**2)[self._loss_type]
        target, terms = self._compute_target(
            obs, action, reward, next_obs, discount, steps)

        with tf.GradientTape() as tape:
            q = self._compute_qs(obs, action=action)
            error = target - q
            loss = tf.reduce_mean(IS_ratio * loss_fn(error))
        tf.debugging.assert_shapes([
            [target, (None)],
            [q, (None)],
        ])

        if self._is_per:
            priority = self._compute_priority(tf.abs(error))
            terms['priority'] = priority
        
        terms['norm'] = self._value_opt(tape, loss)
        
        terms.update(dict(
            q=q,
            target=target,
            loss=loss,
            explained_variance_q=explained_variance(target, q),
        ))

        return terms

    def reset_noisy(self):
        self.q.reset_noisy()
    
    def _compute_qs(self, obs, online=True, action=None):
        if online:
            x = self.encoder(obs)
            qs = self.q(x, action)
        else:
            x = self.target_encoder(obs)
            qs = self.target_q(x, action)
        if action is None:
            tf.debugging.assert_shapes([[qs, (None, self.q.action_dim)]])
        else:
            tf.debugging.assert_shapes([[qs, (None,)]])
        return qs
    
    def _compute_target(self, obs, action, reward, next_obs, discount, steps):
        terms = {}
        if self.MUNCHAUSEN:
            qs = self._compute_qs(obs, False)
            reward, terms = self._compute_reward(reward, qs, action)
        
        next_qs = self._compute_qs(next_obs, False)
        if self._probabilistic_regularization is None:
            if self._double:
                next_online_qs = self._compute_qs(next_obs)
                next_action = self.q.compute_greedy_action(
                    next_online_qs, one_hot=True)
            else:
                next_action = self.target_q.compute_greedy_action(
                    next_qs, one_hot=True)
            next_v = tf.reduce_sum(next_qs * next_action, axis=-1)
        elif self._probabilistic_regularization == 'prob':
            next_pi = softmax(next_qs, self._tau)
            next_v = tf.reduce_sum(next_qs * next_pi, axis=-1)
        elif self._probabilistic_regularization == 'entropy':
            next_pi = softmax(next_qs, self._tau)
            next_logpi = log_softmax(next_qs, self._tau)
            terms['next_entropy'] = - tf.reduce_sum(
                next_pi * next_logpi / self._tau, axis=-1)
            next_v = tf.reduce_sum(
                (next_qs - next_logpi) * next_pi, axis=-1)
        else:
            raise ValueError(self._probabilistic_regularization)

        target = n_step_target(
            reward, next_v, discount, self._gamma, steps, self._tbo)

        return target, terms
