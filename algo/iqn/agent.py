import tensorflow as tf

from utility.tf_utils import softmax, log_softmax
from utility.rl_loss import n_step_target, quantile_regression_loss
from core.decorator import override
from algo.dqn.base import DQNBase, get_data_format, collect
from algo.iqn.base import IQNOps


class Agent(DQNBase, IQNOps):
    @override(DQNBase)
    @tf.function
    def _learn(self, obs, action, reward, next_obs, discount, steps=1, IS_ratio=1):
        target, terms = self._compute_target(obs, action, reward, next_obs, discount, steps)

        with tf.GradientTape() as tape:
            tau_hat, qtv = self._compute_qtvs_from_obs(obs, self.N, action=action)
            qtv = tf.expand_dims(qtv, axis=-1)  # [B, N, 1]
            error, qr_loss = quantile_regression_loss(
                qtv, target, tau_hat, kappa=self.KAPPA, return_error=True)
            loss = tf.reduce_mean(IS_ratio * qr_loss)

        terms['norm'] = self._value_opt(tape, loss)

        if self._is_per:
            error = self._compute_error(error)
            priority = self._compute_priority(error)
            terms['priority'] = priority
        
        terms.update(dict(
            IS_ratio=IS_ratio,
            target=target,
            q=tf.reduce_mean(qtv),
            qr_loss=qr_loss,
            loss=loss,
        ))

        return terms

    def _compute_target(self, obs, action, reward, next_obs, discount, steps):
        terms = {}
        if self.MUNCHAUSEN:
            _, qtvs = self._compute_qtvs_from_obs(obs, self.N_PRIME, False)
            qs = self.q.value(qtvs)
            reward, terms = self._compute_reward(reward, qs, action)
        reward = reward[:, None]

        _, next_qtvs = self._compute_qtvs_from_obs(next_obs, self.N_PRIME, False)
        if self._probabilistic_regularization is None:
            if self._double:
                _, next_online_qtvs = self._compute_qtvs_from_obs(next_obs, self.N_PRIME)
                next_qs = self.q.value(next_online_qtvs)
                next_action = self.q.compute_greedy_action(next_qs, one_hot=True)
            else:
                next_qs = self.target_q.value(next_qtvs)
                next_action = self.target_q.compute_greedy_action(next_qs, one_hot=True)
            next_action = tf.expand_dims(next_action, axis=1)
            tf.debugging.assert_shapes([
                [next_action, (None, 1, self.q.action_dim)],
            ])
            next_qtv = tf.reduce_sum(next_qtvs * next_action, axis=-1)
        elif self._probabilistic_regularization == 'prob':
            next_qs = self.target_q.value(next_qtvs)
            next_pi = softmax(next_qs, self._tau)
            next_pi = tf.expand_dims(next_pi, axis=1)
            tf.debugging.assert_shapes([
                [next_pi, (None, 1, self.q.action_dim)],
            ])
            next_qtv = tf.reduce_sum(next_qtvs * next_pi, axis=-1)
        elif self._probabilistic_regularization == 'entropy':
            next_qs = self.target_q.value(next_qtvs)
            next_pi = softmax(next_qs, self._tau)
            next_logpi = log_softmax(next_qs, self._tau)
            terms['next_entropy'] = - tf.reduce_sum(next_pi * next_logpi / self._tau, axis=-1)
            next_pi = tf.expand_dims(next_pi, axis=1)
            next_logpi = tf.expand_dims(next_logpi, axis=1)
            tf.debugging.assert_shapes([
                [next_pi, (None, 1, self.q.action_dim)],
                [next_logpi, (None, 1, self.q.action_dim)],
            ])
            next_qtv = tf.reduce_sum((next_qtvs - next_logpi) * next_pi, axis=-1)
        else:
            raise ValueError(self._probabilistic_regularization)
            
        discount = discount[:, None]
        if not isinstance(steps, int):
            steps = steps[:, None]
        target = n_step_target(reward, next_qtv, discount, self._gamma, steps)
        target = tf.expand_dims(target, axis=1)      # [B, 1, N']
        tf.debugging.assert_shapes([
            [next_qtv, (None, self.N_PRIME)],
            [target, (None, 1, self.N_PRIME)],
        ])

        return target, terms
