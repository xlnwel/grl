import numpy as np
import tensorflow as tf

from utility.tf_utils import softmax, log_softmax, explained_variance
from utility.rl_utils import *
from utility.rl_loss import quantile_regression_loss, retrace
from core.decorator import override
from algo.mrdqn.base import RDQNBase, get_data_format, collect
from algo.iqn.base import IQNOps


class Agent(RDQNBase, IQNOps):
    """ MRIQN methods """
    @tf.function
    def _learn(self, obs, action, reward, discount, mu, mask, 
                IS_ratio=1, state=None, prev_action=None, prev_reward=None):
        obs, action, mu, mask, target, state, add_inp, terms = \
            self._compute_target_and_process_data(
                obs, action, reward, discount, mu, mask, 
                state, prev_action, prev_reward)

        with tf.GradientTape() as tape:
            x, _ = self._compute_embed(obs, mask, state, add_inp)

            tau_hat, qtv = self._compute_qtvs(x, self.N, action=action)
            qtv = tf.expand_dims(qtv, axis=-1)
            error, qr_loss = quantile_regression_loss(
                qtv, target, tau_hat, kappa=self.KAPPA, return_error=True
            )
            loss = tf.reduce_mean(IS_ratio * qr_loss)
        tf.debugging.assert_shapes([
            [qtv, (None, self._sample_size, self.N, 1)],
            [target, (None, self._sample_size, 1, self.N_PRIME)],
            [error, (None, self._sample_size, self.N, self.N_PRIME)],
            [IS_ratio, (None,)],
            [loss, ()]
        ])
        if self._is_per:
            error = self._compute_error(error)
            priority = self._compute_priority(tf.abs(error))
            terms['priority'] = priority
        
        terms['norm'] = self._optimizer(tape, loss)
        
        q = tf.reshape(tf.reduce_mean(qtv, axis=(-1, -2)), (-1,))
        target = tf.reshape(tf.reduce_mean(target, axis=(-1, -2)), (-1,))
        terms.update(dict(
            q=tf.reduce_mean(qtv, axis=(-1, -2)),
            mu_min=tf.reduce_min(mu),
            mu=mu,
            mu_std=tf.math.reduce_std(mu),
            target=target,
            loss=loss,
            q_explained_variance=explained_variance(target, q)
        ))

        return terms
    
    @override(IQNOps)
    def _compute_qtvs(self, x, N, online=True, action=None):
        b, s, d = x.shape
        x = tf.reshape(x, (-1, d))
        tau_hat, qtv = super()._compute_qtvs(x, N, online)
        tau_hat = tf.reshape(tau_hat, (b, s, N, 1))
        qtv = tf.reshape(qtv, (b, s, N, self._action_dim))
        if action is not None:
            action = tf.expand_dims(action, axis=2)
            qtv = tf.reduce_sum(qtv * action, axis=-1)

        return tau_hat, qtv

    def _compute_target(self, obs, action, reward, discount, 
                        mu, mask, state, add_inp):
        terms = {}
        x, _ = self._compute_embed(obs, mask, state, add_inp, online=False)
        if self._burn_in_size:
            bis = self._burn_in_size
            ss = self._sample_size - bis
            _, reward = tf.split(reward, [bis, ss], 1)
            _, discount = tf.split(discount, [bis, ss], 1)
            _, next_mu_a = tf.split(mu, [bis+1, ss], 1)
            _, next_x = tf.split(x, [bis+1, ss], 1)
            _, next_action = tf.split(action, [bis+1, ss], 1)
        else:
            _, next_mu_a = tf.split(mu, [1, self._sample_size], 1)
            _, next_x = tf.split(x, [1, self._sample_size], 1)
            _, next_action = tf.split(action, [1, self._sample_size], 1)

        _, next_qtvs = self._compute_qtvs(next_x, self.N_PRIME, False)
        regularization = None
        if self._probabilistic_regularization is None:
            if self._double:
                online_x, _ = self._compute_embed(obs, mask, state, add_inp)
                next_online_x = tf.split(online_x, [bis+1, ss-1], 1)
                _, next_online_qtvs = self._compute_qtvs(next_online_x, self.N_PRIME)
                next_online_qs = self.q.value(next_online_qtvs, axis=2)
                next_pi = self.q.compute_greedy_action(next_online_qs, one_hot=True)
            else:
                next_qs = self.target_q.value(next_qtvs, axis=2)
                next_pi = self.target_q.compute_greedy_action(next_qs, one_hot=True)
        elif self._probabilistic_regularization == 'prob':
            next_qs = self.target_q.value(next_qtvs, axis=2)
            next_pi = softmax(next_qs, self._tau)
        elif self._probabilistic_regularization == 'entropy':
            next_qs = self.target_q.value(next_qtvs, axis=2)
            next_pi = softmax(next_qs, self._tau)
            next_logpi = log_softmax(next_qs, self._tau)
            neg_scaled_entropy = tf.reduce_sum(next_pi * next_logpi, axis=-1)
            regularization = tf.expand_dims(neg_scaled_entropy, 2)
            terms['next_entropy'] = - neg_scaled_entropy / self._tau
        else:
            raise ValueError(self._probabilistic_regularization)
        
        tf.debugging.assert_shapes([
            [next_pi, (None, self._sample_size, 15)]
        ])
            
        reward = tf.expand_dims(reward, axis=2)
        next_action = tf.expand_dims(next_action, axis=2)
        next_pi = tf.expand_dims(next_pi, axis=2)
        next_mu_a = tf.expand_dims(next_mu_a, axis=2)
        discount = discount * self._gamma
        discount = tf.expand_dims(discount, axis=2)

        target = retrace(
            reward, next_qtvs, next_action, 
            next_pi, next_mu_a, discount,
            lambda_=self._lambda, 
            axis=1, tbo=self._tbo,
            regularization=regularization)
        target = tf.expand_dims(target, axis=2)      # [B, S, 1, N']
        tf.debugging.assert_shapes([
            [next_qtvs, (None, self._sample_size, self.N_PRIME, self._action_dim)],
            [target, (None, self._sample_size, 1, self.N_PRIME)],
        ])

        return target, terms
