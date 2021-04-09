import logging
import tensorflow as tf

from utility.tf_utils import explained_variance
from utility.rl_loss import ppo_loss, ppo_value_loss
from utility.schedule import TFPiecewiseSchedule
from core.base import RMSAgentBase
from core.optimizer import Optimizer
from core.decorator import override, step_track


logger = logging.getLogger(__name__)

def collect(buffer, env, step, reset, next_obs, **kwargs):
    buffer.add(**kwargs)
    
class PPOBase(RMSAgentBase):
    """ Initialization """
    @override(RMSAgentBase)
    def _add_attributes(self, env, dataset):
        super()._add_attributes(env, dataset)

        self._last_obs = None   # we record last obs before training to compute the last value
        self._value_update = getattr(self, '_value_update', None)
        self._batch_size = env.n_envs * self.N_STEPS // self.N_MBS

    @override(RMSAgentBase)
    def _construct_optimizers(self):
        if getattr(self, 'schedule_lr', False):
            assert isinstance(self._lr, list), self._lr
            self._lr = TFPiecewiseSchedule(self._lr)
        models = list(self.model.values())
        self._ac_opt = Optimizer(
            self._optimizer, models, self._lr, 
            clip_norm=self._clip_norm, epsilon=self._opt_eps)

    """ Standard PPO methods """
    def before_run(self, env):
        pass
    
    def record_last_env_output(self, env_output):
        self._last_obs = self.normalize_obs(env_output.obs)

    def compute_value(self, obs=None):
        # be sure you normalize obs first if obs normalization is required
        obs = self._last_obs if obs is None else obs
        return self.model.compute_value(obs).numpy()

    @step_track
    def learn_log(self, step):
        for i in range(self.N_EPOCHS):
            for j in range(1, self.N_MBS+1):
                with self._sample_timer:
                    data = self.dataset.sample()
                data = {k: tf.convert_to_tensor(v) for k, v in data.items()}
                
                with self._train_timer:
                    terms = self.learn(**data)

                terms = {f'train/{k}': v.numpy() for k, v in terms.items()}
                kl = terms.pop('train/kl')
                value = terms.pop('train/value')
                self.store(**terms, **{'train/value': value.mean()})
                if getattr(self, '_max_kl', None) and kl > self._max_kl:
                    break
                if self._value_update == 'reuse':
                    self.dataset.update('value', value)
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
        
        self.store(**self.dataset.sample_stats('reward'))
        if self._to_summary(step):
            self._summary(data, terms)

        self.store(**{
            'train/kl': kl,
            'time/sample': self._sample_timer.average(),
            'time/train': self._train_timer.average()
        })

        _, rew_rms = self.get_running_stats()
        if rew_rms:
            self.store(**{
                'train/reward_rms_mean': rew_rms.mean,
                'train/reward_rms_var': rew_rms.var
            })

        return i * self.N_MBS + j

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
            value_loss, v_clip_frac = self._compute_value_loss(value, traj_ret, old_value)
            
            actor_loss = (policy_loss - self._entropy_coef * entropy)
            value_loss = self._value_coef * value_loss
            ac_loss = actor_loss + value_loss

        terms['ac_norm'] = self._ac_opt(tape, ac_loss)
        terms.update(dict(
            value=value,
            traj_ret=tf.reduce_mean(traj_ret), 
            advantage=tf.reduce_mean(advantage), 
            ratio=tf.exp(log_ratio), 
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

    def _compute_value_loss(self, value, traj_ret, old_value):
        value_loss_type = getattr(self, '_value_loss', 'mse')
        v_clip_frac = 0
        if value_loss_type == 'mse':
            value_loss = .5 * tf.reduce_mean((value - traj_ret)**2)
        elif value_loss_type == 'clip':
            value_loss, v_clip_frac = ppo_value_loss(
                value, traj_ret, old_value, self._clip_range)
        else:
            raise ValueError(f'Unknown value loss type: {value_loss_type}')
        return value_loss, v_clip_frac
