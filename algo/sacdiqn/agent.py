import tensorflow as tf

from utility.rl_loss import n_step_target, quantile_regression_loss
from utility.tf_utils import explained_variance
from core.decorator import override
from algo.dqn.base import DQNBase, get_data_format, collect
from algo.sacd.base import TempLearner
from algo.iqn.base import IQNOps


class Agent(DQNBase, IQNOps, TempLearner):
    """ Initialization """
    # @tf.function
    # def summary(self, data, terms):
    #     tf.summary.histogram('learn/regularization', terms['regularization'], step=self._env_step)
    #     tf.summary.histogram('learn/reward', data['reward'], step=self._env_step)
    
    """ Call """
    def _process_input(self, obs, evaluation, env_output):
        obs, kwargs = super()._process_input(obs, evaluation, env_output)
        return obs, kwargs

    """ SACIQN Methods"""
    @override(DQNBase)
    @tf.function
    def _learn(self, obs, action, reward, next_obs, discount, steps=1, IS_ratio=1):
        if self.temperature.type == 'schedule':
            _, temp = self.temperature(self._train_step)
        elif self.temperature.type == 'state':
            raise NotImplementedError
        else:
            _, temp = self.temperature()

        target, terms = self._compute_target(reward, next_obs, discount, steps, temp)

        with tf.GradientTape() as tape:
            x = self.encoder(obs, training=True)
            tau_hat, qt_embed = self.quantile(x, self.N)
            x_ext = tf.expand_dims(x, axis=1)
            action_ext = tf.expand_dims(action, axis=1)
            # q loss
            qtvs, qs = self.q(x_ext, qt_embed, return_value=True)
            qtv = tf.reduce_sum(qtvs * action_ext, axis=-1, keepdims=True)  # [B, N, 1]
            error, qr_loss = quantile_regression_loss(
                qtv, target, tau_hat, kappa=self.KAPPA, return_error=True)
            value_loss = tf.reduce_mean(IS_ratio * qr_loss)
        terms['value_norm'] = self._value_opt(tape, value_loss)

        with tf.GradientTape() as tape:
            pi, logpi = self.actor.train_step(x)
            regularization = -tf.reduce_sum(pi * logpi, axis=-1)
            q = tf.reduce_sum(pi * qs, axis=-1)
            actor_loss = -(q + temp * regularization)
            actor_loss = tf.reduce_mean(IS_ratio * actor_loss)
        terms['actor_norm'] = self._actor_opt(tape, actor_loss)

        pi = tf.reduce_mean(pi, 0)
        self.actor.update_prior(pi, self._prior_lr)

        temp_terms = self._learn_temp(x, regularization, IS_ratio)

        if self._is_per:
            error = tf.abs(error)
            error = tf.reduce_max(tf.reduce_mean(error, axis=-1), axis=-1)
            priority = self._compute_priority(error)
            terms['priority'] = priority
            
        target = tf.reduce_mean(target, axis=(1, 2))
        terms.update(dict(
            IS_ratio=IS_ratio,
            steps=steps,
            reward_min=tf.reduce_min(reward),
            actor_loss=actor_loss,
            q=q,
            regularization=regularization,
            regularization_max=tf.reduce_max(regularization),
            regularization_min=tf.reduce_min(regularization),
            value_loss=value_loss, 
            temp=temp,
            explained_variance_q=explained_variance(target, q),
            **temp_terms
        ))
        # for i in range(self.actor.action_dim):
        #     terms[f'prior_{i}'] = self.actor.prior[i]

        return terms

    def _compute_target(self, reward, next_obs, discount, steps, temp):
        terms = {}

        next_x = self.target_encoder(next_obs, training=False)
        next_pi, next_logpi = self.target_actor.train_step(next_x)
        _, qt_embed = self.target_quantile(next_x, self.N_PRIME)
        next_x_ext = tf.expand_dims(next_x, axis=1)
        next_qtv = self.target_q(next_x_ext, qt_embed)
        
        next_pi_ext = tf.expand_dims(next_pi, axis=-2)
        next_logpi_ext = tf.expand_dims(next_logpi, axis=-2)
        next_qtv_v = tf.reduce_sum(next_pi_ext * next_qtv, axis=-1)
        tf.debugging.assert_shapes([
            [next_qtv_v, (None, self.N_PRIME)],
            [next_pi_ext, (None, 1, self._action_dim)],
        ])
        if self._probabilistic_regularization is not None:
            next_qtv_v += temp * -tf.reduce_sum(next_pi_ext * next_logpi_ext, axis=-1)
            
        reward = reward[:, None]
        discount = discount[:, None]
        if not isinstance(steps, int):
            steps = steps[:, None]
        target = n_step_target(reward, next_qtv_v, discount, self._gamma, steps)
        target = tf.expand_dims(target, axis=1)      # [B, 1, N']
        tf.debugging.assert_shapes([
            [next_qtv_v, (None, self.N_PRIME)],
            [target, (None, 1, self.N_PRIME)],
        ])
        return target, terms
