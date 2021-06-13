import tensorflow as tf

from utility.tf_utils import softmax, log_softmax, explained_variance
from utility.rl_utils import *
from utility.rl_loss import retrace
from core.decorator import override
from algo.mrdqn.base import RDQNBase, get_data_format, collect


class Agent(RDQNBase):
    """ MRDQN methods """
    @tf.function
    def _learn(self, obs, action, reward, discount, mu, mask, 
                IS_ratio=1, state=None, prev_action=None, prev_reward=None):
        obs, action, mu, mask, target, state, add_inp, terms = \
            self._compute_target_and_process_data(
                obs, action, reward, discount, mu, mask, 
                state, prev_action, prev_reward)

        with tf.GradientTape() as tape:
            x, _ = self._compute_embed(obs, mask, state, add_inp)
            
            qs = self.q(x)
            q = tf.reduce_sum(qs * action, -1)
            error = target - q
            value_loss = tf.reduce_mean(.5 * error**2, axis=-1)
            value_loss = tf.reduce_mean(IS_ratio * value_loss)
            terms['value_loss'] = value_loss
        tf.debugging.assert_shapes([
            [q, (None, self._sample_size)],
            [target, (None, self._sample_size)],
            [error, (None, self._sample_size)],
            [IS_ratio, (None,)],
            [value_loss, ()]
        ])
        terms['value_norm'] = self._value_opt(tape, value_loss)

        if 'actor' in self.model:
            with tf.GradientTape() as tape:
                pi, logpi = self.actor.train_step(x)
                pi_a = tf.reduce_sum(pi * action, -1)
                reinforce = tf.minimum(1. / mu, self._loo_c) * error * pi_a
                v = tf.reduce_sum(qs * pi, axis=-1)
                regularization = -tf.reduce_sum(pi * logpi, axis=-1)
                loo_loss = -(self._v_pi_coef * v + self._reinforce_coef * reinforce)
                tf.debugging.assert_shapes([
                    [pi, (None, self._sample_size, self._action_dim)],
                    [qs, (None, self._sample_size, self._action_dim)],
                    [v, (None, self._sample_size)],
                    [reinforce, (None, self._sample_size)],
                    [regularization, (None, self._sample_size)],
                ])
                loo_loss = tf.reduce_mean(loo_loss, axis=-1)
                regularization = tf.reduce_mean(regularization, axis=-1)
                actor_loss = loo_loss - self._tau * regularization
                actor_loss = tf.reduce_mean(IS_ratio * actor_loss)
                terms.update(dict(
                    reinforce=reinforce,
                    v=v,
                    loo_loss=loo_loss,
                    regularization=regularization,
                    actor_loss=actor_loss,
                    ratio=tf.reduce_mean(pi_a / mu),
                    pi_min=tf.reduce_min(pi),
                    pi_std=tf.math.reduce_std(pi)
                ))
            terms['actor_norm'] = self._actor_opt(tape, actor_loss)

        if self._is_per:
            priority = self._compute_priority(tf.abs(error))
            terms['priority'] = priority
        
        terms.update(dict(
            q=q,
            q_std=tf.math.reduce_std(q),
            error=error,
            error_std=tf.math.reduce_std(error),
            mu_min=tf.reduce_min(mu),
            mu=mu,
            mu_inv=tf.reduce_mean(1/mu),
            mu_std=tf.math.reduce_std(mu),
            target=target,
            explained_variance_q=explained_variance(target, q)
        ))

        return terms
    
    @override(RDQNBase)
    def _compute_target(self, obs, action, reward, discount, 
                        mu, mask, state, add_inp):
        terms = {}
        x, _ = self._compute_embed(obs, mask, state, add_inp, online=False)
        if self._burn_in_size:
            bis = self._burn_in_size
            ss = self._sample_size
            _, reward = tf.split(reward, [bis, ss], 1)
            _, discount = tf.split(discount, [bis, ss], 1)
            _, next_mu_a = tf.split(mu, [bis+1, ss], 1)
            _, next_x = tf.split(x, [bis+1, ss], 1)
            _, next_action = tf.split(action, [bis+1, ss], 1)
        else:
            _, next_mu_a = tf.split(mu, [1, self._sample_size], 1)
            _, next_x = tf.split(x, [1, self._sample_size], 1)
            _, next_action = tf.split(action, [1, self._sample_size], 1)

        next_qs = self.target_q(next_x)
        regularization = None
        if 'actor' in self.model:
            next_pi, next_logpi = self.target_actor.train_step(next_x)
            if self._probabilistic_regularization == 'entropy':
                regularization = tf.reduce_sum(
                    self._tau * next_pi * next_logpi, axis=-1)
        else:
            if self._probabilistic_regularization is None:
                if self._double:    # don't suggest to use double Q here, but implement it anyway
                    online_x, _ = self._compute_embed(obs, mask, state, add_inp)
                    next_online_x = tf.split(online_x, [bis+1, ss-1], 1)
                    next_online_qs = self.q(next_online_x)
                    next_pi = self.q.compute_greedy_action(next_online_qs, one_hot=True)
                else:
                    next_pi = self.target_q.compute_greedy_action(next_qs, one_hot=True)
            elif self._probabilistic_regularization == 'prob':
                next_pi = softmax(next_qs, self._tau)
            elif self._probabilistic_regularization == 'entropy':
                next_pi = softmax(next_qs, self._tau)
                next_logpi = log_softmax(next_qs, self._tau)
                regularization = tf.reduce_sum(next_pi * next_logpi, axis=-1)
                terms['next_entropy'] = - regularization / self._tau
            else:
                raise ValueError(self._probabilistic_regularization)

        discount = discount * self._gamma
        target = retrace(
            reward, next_qs, next_action, 
            next_pi, next_mu_a, discount,
            lambda_=self._lambda, 
            axis=1, tbo=self._tbo,
            regularization=regularization)

        return target, terms
