import tensorflow as tf
from tensorflow_probability import distributions as tfd

from core.module import Ensemble
from algo.sacd.nn import Encoder, Actor, Temperature
from algo.iqn.nn import Quantile, Value


class SACIQN(Ensemble):
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
        assert x.shape.ndims == 4, x.shape

        x = self.encoder(x)
        action = self.actor(x, evaluation=evaluation, epsilon=epsilon, temp=temp)
        terms = {}
        if return_eval_stats:
            action, terms = action
            _, qt_embed = self.quantile(x)
            _, qtv, q = self.q(x, qt_embed, return_value=True)
            qtv = tf.transpose(qtv, [0, 2, 1])
            idx = tf.stack([tf.range(action.shape[0]), action], -1)
            qtv_max = tf.reduce_max(qtv, 1)
            q_max = tf.reduce_max(q, 1)
            action_best_q = tf.argmax(q, 1)
            qtv = tf.gather_nd(qtv, idx)
            q = tf.gather_nd(q, idx)
            q = tf.squeeze(q)
            action = tf.squeeze(action)
            action_best_q = tf.squeeze(action_best_q)
            qtv_max = tf.squeeze(qtv_max)
            qtv = tf.squeeze(qtv)
            q_max = tf.squeeze(q_max)
            terms = {
                'action': action,
                'action_best_q': action_best_q,
                'qtv_max': qtv_max,
                'qtv': qtv,
                'q_max': q_max,
                'q': q,
            }
        elif return_stats:
            _, qt_embed = self.quantile(x)
            x_ext = tf.expand_dims(x, axis=1)
            _, q = self.q(x_ext, qt_embed, action=action, return_value=True)
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


def create_components(config, env, **kwargs):
    assert env.is_action_discrete, env.name
    action_dim = env.action_dim
    encoder_config = config['encoder']
    actor_config = config['actor']
    q_config = config['q']
    temperature_config = config['temperature']
    quantile_config = config['quantile']

    models = dict(
        encoder=Encoder(encoder_config, name='encoder'),
        quantile=Quantile(quantile_config, name='phi'),
        actor=Actor(actor_config, action_dim),
        q=Value(q_config, action_dim, name='q'),
        target_encoder=Encoder(encoder_config, name='target_encoder'),
        target_quantile=Quantile(quantile_config, name='target_phi'),
        target_actor=Actor(actor_config, action_dim, name='target_actor'),
        target_q=Value(q_config, action_dim, name='target_q'),
        temperature=Temperature(temperature_config),
    )

    return models

def create_model(config, env, **kwargs):
    return SACIQN(config=config, env=env, **kwargs)
