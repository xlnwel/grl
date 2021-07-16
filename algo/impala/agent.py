import tensorflow as tf

from core.tf_config import build
from core.decorator import override
from core.base import RMSAgentBase
from core.mixin import Memory
from utility.tf_utils import explained_variance
from utility.rl_loss import v_trace_from_ratio, ppo_loss


def collect(buffer, env, env_step, reset, next_obs, **kwargs):
    buffer.add(**kwargs)


def get_data_format(*, env, batch_size, sample_size=None,
        store_state=False, state_size=None, **kwargs):
    obs_dtype = tf.uint8 if len(env.obs_shape) == 3 else tf.float32
    action_dtype = tf.int32 if env.is_action_discrete else tf.float32
    data_format = dict(
        obs=((None, sample_size+1, *env.obs_shape), obs_dtype),
        action=((None, sample_size, *env.action_shape), action_dtype),
        reward=((None, sample_size), tf.float32),
        discount=((None, sample_size), tf.float32),
        logpi=((None, sample_size), tf.float32),
        mask=((None, sample_size+1), tf.float32),
    )
    if store_state:
        dtype = tf.keras.mixed_precision.experimental.global_policy().compute_dtype
        data_format.update({
            k: ((batch_size, v), dtype)
                for k, v in state_size._asdict().items()
        })

    return data_format


class Agent(Memory, RMSAgentBase):
    """ Initialization """
    @override(RMSAgentBase)
    def _add_attributes(self, env, dataset):
        super()._add_attributes(env, dataset)
        self._setup_memory_state_record()

    @override(RMSAgentBase)
    def _build_learn(self, env):
        # Explicitly instantiate tf.function to avoid unintended retracing
        TensorSpecs = dict(
            obs=((self._sample_size+1, *env.obs_shape), env.obs_dtype, 'obs'),
            action=((self._sample_size, *env.action_shape), env.action_dtype, 'action'),
            reward=((self._sample_size,), tf.float32, 'reward'),
            discount=((self._sample_size,), tf.float32, 'discount'),
            logpi=((self._sample_size,), tf.float32, 'logpi'),
            mask=((self._sample_size+1,), tf.float32, 'mask'),
        )
        if self._store_state:
            state_type = type(self.model.state_size)
            TensorSpecs['state'] = state_type(*[((sz, ), self._dtype, name) 
                for name, sz in self.model.state_size._asdict().items()])
        if self._additional_rnn_inputs:
            if 'prev_action' in self._additional_rnn_inputs:
                TensorSpecs['prev_action'] = (
                    (self._sample_size, *env.action_shape), 
                    env.action_dtype, 'prev_action')
            if 'prev_reward' in self._additional_rnn_inputs:
                TensorSpecs['prev_reward'] = (
                    (self._sample_size,), self._dtype, 'prev_reward')    # this reward should be unnormlaized
        self.learn = build(self._learn, TensorSpecs)

    """ Call """
    # @override(PPOBase)
    def _process_input(self, env_output, evaluation):
        obs, kwargs = super()._process_input(env_output, evaluation)
        mask = 1. - env_output.reset
        kwargs = self._add_memory_state_to_kwargs(obs, mask, kwargs=kwargs)
        kwargs['return_value'] = False
        return obs, kwargs

    # @override(PPOBase)
    def _process_output(self, obs, kwargs, out, evaluation):
        out = self._add_tensors_to_terms(obs, kwargs, out, evaluation)
        out = super()._process_output(obs, kwargs, out, evaluation)
        out = self._add_non_tensors_to_terms(out, kwargs, evaluation)
        return out

    # @override(PPOBase)
    # def _summary(self, data, terms):
    #     tf.summary.histogram('sum/value', data['value'], step=self._env_step)
    #     tf.summary.histogram('sum/logpi', data['logpi'], step=self._env_step)

    @tf.function
    def _learn(self, obs, action, reward, 
                discount, logpi, mask=None, state=None, 
                prev_action=None, prev_reward=None):
        terms = {}
        with tf.GradientTape() as tape:
            if hasattr(self.model, 'rnn'):
                x, state = self.model.encode(obs, state, mask,
                    prev_action=prev_action, prev_reward=prev_reward)
            else:
                x = self.encoder(obs)
            curr_x, _ = tf.split(x, [self._sample_size, 1], 1)
            act_dist = self.actor(curr_x)
            new_logpi = act_dist.log_prob(action)
            log_ratio = new_logpi - logpi
            ratio = tf.exp(log_ratio)
            entropy = act_dist.entropy()
            value = self.value(x)
            value, next_value = value[:, :-1], value[:, 1:]
            discount = self._gamma * discount
            # policy loss
            target, advantage = v_trace_from_ratio(
                reward, value, next_value, ratio, discount, 
                lambda_=self._lambda, c_clip=self._c_clip, 
                rho_clip=self._rho_clip, rho_clip_pg=self._rho_clip_pg, axis=1)
            adv_mean = tf.reduce_mean(advantage)
            adv_var = tf.math.reduce_variance(advantage)
            if self._normalize_advantage:
                adv_std = tf.sqrt(adv_var + 1e-5)
                advantage = (advantage - adv_mean) / adv_std
            target = tf.stop_gradient(target)
            advantage = tf.stop_gradient(advantage)
            if self._policy_loss == 'ppo':
                policy_loss, entropy, kl, p_clip_frac = ppo_loss(
                    log_ratio, advantage, self._clip_range, entropy)
            elif self._policy_loss == 'reinforce':
                policy_loss = -tf.reduce_mean(advantage * log_ratio)
                entropy = tf.reduce_mean(entropy)
                kl = tf.reduce_mean(-log_ratio)
                p_clip_frac = 0
            else:
                raise NotImplementedError
            # value loss
            value_loss = .5 * tf.reduce_mean((value - target)**2)

            actor_loss = (policy_loss - self._entropy_coef * entropy)
            value_loss = self._value_coef * value_loss
            ac_loss = actor_loss + value_loss

        terms['ac_norm'] = self._optimizer(tape, ac_loss)
        terms.update(dict(
            value=value,
            target=tf.reduce_mean(target),
            adv=adv_mean,
            adv_var=adv_var,
            ratio_max=tf.reduce_max(ratio),
            ratio_min=tf.reduce_min(ratio),
            rho=tf.minimum(ratio, self._rho_clip),
            entropy=entropy, 
            kl=kl, 
            p_clip_frac=p_clip_frac,
            ppo_loss=policy_loss,
            actor_loss=actor_loss,
            v_loss=value_loss,
            explained_variance=explained_variance(target, value),
        ))

        return terms

    def _sample_learn(self):
        for i in range(self.N_EPOCHS):
            for j in range(1, self.N_MBS+1):
                with self._sample_timer:
                    data = self.dataset.sample()
                data = {k: tf.convert_to_tensor(v) for k, v in data.items()}
                with self._learn_timer:
                    terms = self.learn(**data)
                terms = {f'train/{k}': v.numpy() for k, v in terms.items()}
                self.store(**terms)

                self._after_train_step()

        self.store(**{
            'time/sample_mean': self._sample_timer.average(),
            'time/learn_mean': self._learn_timer.average(),
        })

        if self._to_summary(self.train_step):
            self._summary(data, terms)

        return 1

    def _after_train_step(self):
        pass

    def _store_buffer_stats(self):
        self.store(**self.dataset.compute_mean_max_std('reward'))
        self.store(**self.dataset.compute_mean_max_std('value'))
