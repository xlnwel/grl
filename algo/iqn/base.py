import tensorflow as tf


class IQNOps:
    def _compute_qtvs_from_obs(self, obs, N, online=True, action=None):
        if online:
            x = self.encoder(obs)
        else:
            x = self.target_encoder(obs)
        return self._compute_qtvs(x, N, online=online, action=action)

    def _compute_qtvs(self, x, N, online=True, action=None):
        if online:
            tau_hat, qt_embed = self.quantile(x, N)
            qtv = self.q(x, qt_embed, action=action)
        else:
            tau_hat, qt_embed = self.target_quantile(x, N)
            qtv = self.target_q(x, qt_embed, action=action)
        return tau_hat, qtv

    def _compute_error(self, error):
        error = tf.abs(error)
        error = tf.reduce_max(tf.reduce_mean(error, axis=-1), axis=-1)
        return error
