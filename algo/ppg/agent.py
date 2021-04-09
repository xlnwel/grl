import logging
import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd

from utility.schedule import TFPiecewiseSchedule
from utility.tf_utils import explained_variance
from utility.rl_loss import ppo_loss
from core.tf_config import build
from core.optimizer import Optimizer
from core.decorator import override
from algo.ppo.base import PPOBase, collect


logger = logging.getLogger(__name__)

class Agent(PPOBase):
    """ Initialization """
    @override(PPOBase)
    def _add_attributes(self, env, dateset):
        super()._add_attributes(env, dateset)
        assert self.N_SEGS <= self.N_PI, f'{self.N_SEGS} > {self.N_PI}'
        self.N_AUX_MBS = self.N_SEGS * self.N_AUX_MBS_PER_SEG
        self._aux_batch_size = env.n_envs * self.N_STEPS // self.N_AUX_MBS_PER_SEG

    @override(PPOBase)
    def _construct_optimizers(self):
        if getattr(self, 'schedule_lr', False):
            assert isinstance(self._actor_lr, list)
            assert isinstance(self._value_lr, list)
            assert isinstance(self._aux_lr, list)
            self._actor_lr = TFPiecewiseSchedule(self._actor_lr)
            self._value_lr = TFPiecewiseSchedule(self._value_lr)
            self._aux_lr = TFPiecewiseSchedule(self._aux_lr)

        actor_models = [self.encoder, self.actor]
        if hasattr(self, 'rnn'):
            actor_models.append(self.rnn)
        self._actor_opt = Optimizer(
            self._optimizer, actor_models, self._actor_lr, 
            clip_norm=self._clip_norm, epsilon=self._opt_eps)

        value_models = [self.value]
        if hasattr(self, 'value_encoder'):
            value_models.append(self.value_encoder)
        if hasattr(self, 'value_rnn'):
            value_models.append(self.value_rnn)
        self._value_opt = Optimizer(
            self._optimizer, value_models, self._value_lr, 
            clip_norm=self._clip_norm, epsilon=self._opt_eps)
        
        aux_models = list(self.model.values())
        self._aux_opt = Optimizer(
            self._optimizer, aux_models, self._aux_lr, 
            clip_norm=self._clip_norm, epsilon=self._opt_eps)

    @override(PPOBase)
    def _build_learn(self, env):
        # Explicitly instantiate tf.function to avoid unintended retracing
        TensorSpecs = dict(
            obs=(env.obs_shape, env.obs_dtype, 'obs'),
            action=(env.action_shape, env.action_dtype, 'action'),
            value=((), tf.float32, 'value'),
            traj_ret=((), tf.float32, 'traj_ret'),
            advantage=((), tf.float32, 'advantage'),
            logpi=((), tf.float32, 'logpi'),
        )
        self.learn = build(self._learn, TensorSpecs, batch_size=self._batch_size)
        TensorSpecs = dict(
            obs=(env.obs_shape, env.obs_dtype, 'obs'),
            logits=((env.action_dim,), tf.float32, 'logits'),
            value=((), tf.float32, 'value'),
            traj_ret=((), tf.float32, 'traj_ret'),
        )
        self.aux_learn = build(self._aux_learn, TensorSpecs, batch_size=self._aux_batch_size)

    """ PPG methods """
    def compute_aux_data(self, obs):
        out = self.model.compute_aux_data(obs)
        out = tf.nest.map_structure(lambda x: x.numpy(), out)
        return out

    def aux_learn_log(self, step):
        for i in range(self.N_AUX_EPOCHS):
            for j in range(1, self.N_AUX_MBS+1):
                data = self.dataset.sample_aux_data()
                data = {k: tf.convert_to_tensor(v) for k, v in data.items()}
                terms = self.aux_learn(**data)

                terms = {f'aux/{k}': v.numpy() for k, v in terms.items()}
                kl = terms.pop('aux/kl')
                value = terms.pop('aux/value')
                self.store(**terms, value=value.mean())
                if self._value_update == 'reuse':
                    self.dataset.aux_update('value', value)
            if self._value_update == 'once':
                self.dataset.compute_aux_data_with_func(self.compute_value)
            if self._value_update is not None:
                last_value = self.compute_value()
                self.dataset.aux_finish(last_value)
        if self.N_AUX_EPOCHS:
            self.store(**{'aux/kl': kl})
            
            if self._to_summary(step):
                self._summary(data, terms)

            return i * self.N_MBS + j
        else:
            return 0

    @tf.function
    def _learn(self, obs, action, value, traj_ret, advantage, logpi, state=None, mask=None):
        old_value = value
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
        terms['actor_norm'] = self._actor_opt(tape, actor_loss)

        with tf.GradientTape() as tape:
            x_value = self.value_encoder(obs) if hasattr(self, 'value_encoder') else x
            value = self.value(x_value)
            value_loss, v_clip_frac = self._compute_value_loss(
                value, traj_ret, old_value)
        terms['value_norm'] = self._value_opt(tape, value_loss)

        terms.update(dict(
            value=value,
            advantage=advantage, 
            ratio=tf.exp(log_ratio), 
            entropy=entropy, 
            kl=kl, 
            p_clip_frac=p_clip_frac,
            ppo_loss=policy_loss,
            v_loss=value_loss,
            explained_variance=explained_variance(traj_ret, value),
            v_clip_frac=v_clip_frac
        ))

        return terms

    @tf.function
    def _aux_learn(self, obs, logits, value, traj_ret, mask=None, state=None):
        old_value = value
        terms = {}
        with tf.GradientTape() as tape:
            x = self.encoder(obs)
            if state is not None:
                x, state = self.rnn(x, state, mask=mask)
            act_dist = self.actor(x)
            old_dist = tfd.Categorical(logits=logits)
            kl = tf.reduce_mean(old_dist.kl_divergence(act_dist))
            actor_loss = bc_loss = self._bc_coef * kl
            if hasattr(self, 'value_encoder'):
                aux_value = self.aux_value(x)
                aux_loss = .5 * tf.reduce_mean((aux_value - traj_ret)**2)
                terms['bc_loss'] = bc_loss
                terms['aux_loss'] = aux_loss
                actor_loss = aux_loss + bc_loss

                x_value = self.value_encoder(obs)
                value = self.value(x_value)
            else:
                # allow gradients from value head if using a shared encoder
                value = self.value(x)

            value_loss, v_clip_frac = self._compute_value_loss(
                value, traj_ret, old_value)
            loss = actor_loss + value_loss

        terms['actor_norm'] = self._aux_opt(tape, loss)

        terms.update(dict(
            value=value, 
            kl=kl, 
            actor_loss=actor_loss,
            v_loss=value_loss,
            explained_variance=explained_variance(traj_ret, value),
            v_clip_frac=v_clip_frac
        ))

        return terms
