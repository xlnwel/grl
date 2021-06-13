import logging
import tensorflow as tf

from utility.schedule import TFPiecewiseSchedule
from utility.utils import Every
from utility.tf_utils import explained_variance
from utility.rl_loss import ppo_loss
from core.tf_config import build
from core.decorator import override
from algo.ppo.base import PPOBase, collect


logger = logging.getLogger(__name__)

class Agent(PPOBase):
    """ Initialization """
    def _add_attributes(self, env, dataset):
        super()._add_attributes(env, dataset)
        self._to_summary_value = Every(self.LOG_PERIOD, self.LOG_PERIOD)
        
    @override(PPOBase)
    def _construct_optimizers(self):
        if getattr(self, 'schedule_lr', False):
            assert isinstance(self._actor_lr, list)
            assert isinstance(self._value_lr, list)
            self._actor_lr = TFPiecewiseSchedule(self._actor_lr)
            self._value_lr = TFPiecewiseSchedule(self._value_lr)

        actor_models = [self.encoder, self.actor, self.advantage]
        if hasattr(self, 'actor_rnn'):
            actor_models.append(self.actor_rnn)
        self._actor_opt = self._construct_opt(actor_models, self._actor_lr)

        value_models = [self.value]
        if hasattr(self, 'value_encoder'):
            value_models.append(self.value_encoder)
        if hasattr(self, 'value_rnn'):
            value_models.append(self.value_rnn)
        self._value_opt = self._construct_opt(value_models, self._value_lr)
        
        return actor_models + value_models

    @override(PPOBase)
    def _build_learn(self, env):
        # Explicitly instantiate tf.function to avoid unintended retracing
        TensorSpecs = dict(
            obs=(env.obs_shape, env.obs_dtype, 'obs'),
            action=(env.action_shape, env.action_dtype, 'action'),
            advantage=((), tf.float32, 'advantage'),
            logpi=((), tf.float32, 'logpi'),
        )
        self._policy_data = ['obs', 'action', 'advantage', 'logpi']
        self.learn_policy = build(self._learn_policy, TensorSpecs, batch_size=self._batch_size)
        TensorSpecs = dict(
            obs=(env.obs_shape, env.obs_dtype, 'obs'),
            value=((), tf.float32, 'value'),
            traj_ret=((), tf.float32, 'traj_ret'),
        )
        self._value_data = ['obs', 'value', 'traj_ret']
        self.learn_value = build(self._learn_value, TensorSpecs, batch_size=self._batch_size)

    def _summary(self, data, terms):
        pass

    def _summary_value(self, data, terms):
        pass

    """ DAAC methods """
    @tf.function
    def _learn_policy(self, obs, action, advantage, logpi, state=None, mask=None):
        terms = {}
        with tf.GradientTape() as tape:
            x = self.encoder(obs)
            if state is not None:
                x, state = self.rnn(x, state, mask=mask)    
            act_dist = self.actor(x)
            new_logpi = act_dist.log_prob(action)
            entropy = act_dist.entropy()
            log_ratio = new_logpi - logpi
            policy_loss, entropy, kl, p_clip_frac = ppo_loss(
                log_ratio, advantage, self._clip_range, entropy)
            actor_loss = policy_loss - self._entropy_coef * entropy

            one_hot = tf.one_hot(action, self._action_dim)
            x_a = tf.concat([x, one_hot], axis=-1)
            adv = self.advantage(x_a)
            tf.debugging.assert_rank(adv, 1)
            adv_loss = .5 * tf.reduce_mean((adv - advantage)**2)
            loss = actor_loss + self._adv_coef * adv_loss

        actor_norm = self._actor_opt(tape, loss)
        terms.update(dict(
            ratio=tf.exp(log_ratio),
            adv=advantage,
            entropy=entropy,
            kl=kl,
            p_clip_frac=p_clip_frac,
            ppo_loss=policy_loss,
            actor_loss=actor_loss,
            adv_loss=adv_loss,
            loss=loss,
            actor_norm=actor_norm
        ))
        return terms
    
    @tf.function
    def _learn_value(self, obs, value, traj_ret, state=None, mask=None):
        old_value = value
        terms = {}
        with tf.GradientTape() as tape:
            x = self.value_encoder(obs)
            if state is not None:
                x, state = self.value_rnn(x, state, mask=mask)
            value = self.value(x)
            value_loss, v_clip_frac = self._compute_value_loss(
                value, traj_ret, old_value)
        value_norm = self._value_opt(tape, value_loss)
        terms.update(dict(
            value=value,
            value_norm=value_norm,
            value_loss=value_loss,
            explained_variance=explained_variance(traj_ret, value),
            v_clip_frac=v_clip_frac,
        ))
        return terms

    def _sample_learn(self):
        for i in range(self.N_EPOCHS):
            for j in range(1, self.N_MBS+1):
                with self._sample_timer:
                    data = self.dataset.sample()
                data = {k: tf.convert_to_tensor(data[k]) for k in self._policy_data}
                
                with self._learn_timer:
                    terms = self.learn_policy(**data)
                
                terms = {f'train/{k}': v.numpy() for k, v in terms.items()}
                self.store(**terms)
                kl = terms.pop('train/kl')
                if getattr(self, '_max_kl', None) and kl > self._max_kl:
                    break
            if getattr(self, '_max_kl', None) and kl > self._max_kl:
                logger.info(f'{self._model_name}: Eearly stopping after '
                    f'{i*self.N_MBS+j} update(s) due to reaching max kl.'
                    f'Current kl={kl:.3g}')
                break
        
        n = i * self.N_MBS + j
        if self._to_summary(n):
            self._summary(data, terms)

        for i in range(self.N_VALUE_EPOCHS):
            for j in range(1, self.N_MBS+1):
                with self._sample_timer:
                    data = self.dataset.sample()
                data = {k: tf.convert_to_tensor(data[k]) for k in self._value_data}

                terms = self.learn_value(**data)

                terms = {f'train/{k}': v.numpy() for k, v in terms.items()}
                value = terms.pop('train/value')
                self.store(**terms, **{'train/value': value.mean()})
                if self._value_update == 'reuse':
                    self.dataset.update('value', value)
            if self._value_update == 'once':
                self.dataset.update_value_with_func(self.compute_value)
            if self._value_update is not None:
                last_value = self.compute_value()
                self.dataset.finish(last_value)
        
        if self._to_summary_value(n):
            self._summary_value(data, terms)

        return n
