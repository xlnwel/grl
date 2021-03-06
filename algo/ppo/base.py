import logging
import tensorflow as tf

from utility.tf_utils import explained_variance
from utility.rl_loss import huber_loss, reduce_mean, ppo_loss, ppo_value_loss
from utility.typing import EnvOutput
from core.base import RMSAgentBase
from core.decorator import override


logger = logging.getLogger(__name__)


def collect(buffer, env, env_step, reset, next_obs, **kwargs):
    buffer.add(**kwargs)


class PPOBase(RMSAgentBase):
    """ Initialization """
    @override(RMSAgentBase)
    def _add_attributes(self, env, dataset):
        super()._add_attributes(env, dataset)

        self._huber_threshold = getattr(self, '_huber_threshold', None)

        self._last_obs = None   # we record last obs before training to compute the last value
        self._value_update = getattr(self, '_value_update', None)
        if not hasattr(self, '_batch_size'):
            self._batch_size = env.n_envs * self.N_STEPS // self.N_MBS

    """ Standard PPO methods """
    def before_run(self, env):
        pass

    def record_last_env_output(self, env_output):
        self._env_output = EnvOutput(
            self.normalize_obs(env_output.obs), env_output.reward,
            env_output.discount, env_output.reset)

    def compute_value(self, obs=None):
        # be sure you normalize obs first if obs normalization is required
        obs = self._env_output.obs if obs is None else obs
        return self.model.compute_value(obs).numpy()

    @tf.function
    def _learn(self, obs, action, value, traj_ret, advantage, logpi, 
                state=None, mask=None, prev_action=None, prev_reward=None):
        old_value = value
        terms = {}
        with tf.GradientTape() as tape:
            if hasattr(self.model, 'rnn'):
                x, state = self.model.encode(obs, state, mask,
                    prev_action=prev_action, prev_reward=prev_reward)
            else:
                x = self.encoder(obs)
            act_dist = self.actor(x)
            new_logpi = act_dist.log_prob(action)
            entropy = act_dist.entropy()
            # policy loss
            log_ratio = new_logpi - logpi
            policy_loss, entropy, kl, p_clip_frac = ppo_loss(
                log_ratio, advantage, self._clip_range, entropy)
            # value loss
            value = self.value(x)
            value_loss, v_clip_frac = self._compute_value_loss(
                value, traj_ret, old_value)

            actor_loss = (policy_loss - self._entropy_coef * entropy)
            value_loss = self._value_coef * value_loss
            ac_loss = actor_loss + value_loss

        terms['ac_norm'] = self._optimizer(tape, ac_loss)
        ratio = tf.exp(log_ratio)
        terms.update(dict(
            value=value,
            ratio=ratio, 
            entropy=entropy, 
            kl=kl, 
            p_clip_frac=p_clip_frac,
            ppo_loss=policy_loss,
            actor_loss=actor_loss,
            v_loss=value_loss,
            explained_variance=explained_variance(traj_ret, value),
            v_clip_frac=v_clip_frac
        ))

        return terms

    def _compute_value_loss(self, value, traj_ret, old_value, mask=None):
        value_loss_type = getattr(self, '_value_loss', 'mse')
        v_clip_frac = 0
        if value_loss_type == 'huber':
            value_loss = reduce_mean(
                huber_loss(value, traj_ret, threshold=self._huber_threshold), mask)
        elif value_loss_type == 'mse':
            value_loss = .5 * reduce_mean((value - traj_ret)**2, mask)
        elif value_loss_type == 'clip':
            value_loss, v_clip_frac = ppo_value_loss(
                value, traj_ret, old_value, self._clip_range, 
                mask=mask)
        elif value_loss_type == 'clip_huber':
            value_loss, v_clip_frac = ppo_value_loss(
                value, traj_ret, old_value, self._clip_range, 
                mask=mask, threshold=self._huber_threshold)
        else:
            raise ValueError(f'Unknown value loss type: {value_loss_type}')
        
        return value_loss, v_clip_frac

    def _sample_learn(self):
        for i in range(self.N_EPOCHS):
            for j in range(1, self.N_MBS+1):
                with self._sample_timer:
                    data = self._sample_data()

                data = {k: tf.convert_to_tensor(v) for k, v in data.items()}

                with self._learn_timer:
                    terms = self.learn(**data)

                kl = terms.pop('kl').numpy()
                value = terms.pop('value').numpy()
                terms = {f'train/{k}': v.numpy() for k, v in terms.items()}

                self.store(**terms)
                if getattr(self, '_max_kl', None) and kl > self._max_kl:
                    break
                if self._value_update == 'reuse':
                    self.dataset.update('value', value)

                self._after_train_step()

            if getattr(self, '_max_kl', None) and kl > self._max_kl:
                logger.info(f'{self._model_name}: Eearly stopping after '
                    f'{i*self.N_MBS+j} update(s) due to reaching max kl.'
                    f'Current kl={kl:.3g}')
                break
            
            if self._value_update == 'once':
                self.dataset.update_value_with_func(self.compute_value)
            if self._value_update is not None:
                last_value = self.compute_value()
                self.dataset.finish(last_value)
            
            self._after_train_epoch()

        n = i * self.N_MBS + j
        self.store(**{
            'stats/policy_updates': n,
            'train/kl': kl,
            'train/value': value,
            'time/sample_mean': self._sample_timer.average(),
            'time/learn_mean': self._learn_timer.average(),
        })

        if self._to_summary(self.train_step + n):
            self._summary(data, terms)

        return n
    
    def _sample_data(self):
        return self.dataset.sample()

    def _after_train_step(self):
        """ Does something after each training step """
        pass

    def _after_train_epoch(self):
        """ Does something after each training epoch """
        pass

    def _store_buffer_stats(self):
        self.store(**self.dataset.compute_mean_max_std('reward'))
        # self.store(**self.dataset.compute_mean_max_std('obs'))
        self.store(**self.dataset.compute_mean_max_std('advantage'))
        self.store(**self.dataset.compute_mean_max_std('value'))
        self.store(**self.dataset.compute_mean_max_std('traj_ret'))
