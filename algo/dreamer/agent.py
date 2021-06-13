import functools
import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd

from utility.utils import AttrDict, Every
from utility.rl_loss import lambda_return
from utility.tf_utils import static_scan
from core.tf_config import build
from core.base import AgentBase
from core.mixin import Memory
from core.decorator import override, step_track
from algo.dreamer.nn import RSSMState


def get_data_format(*, env, batch_size, sample_size=None, 
        store_state=False, state_size=None, dtype=tf.float32, **kwargs):
    data_format = dict(
        obs=((batch_size, sample_size, *env.obs_shape), env.obs_dtype),
        prev_action=((batch_size, sample_size, *env.action_shape), env.action_dtype),
        reward=((batch_size, sample_size), dtype), 
        discount=((batch_size, sample_size), dtype),
    )
    if store_state:
        data_format.update({
            k: ((batch_size, v), dtype)
                for k, v in state_size._asdict().items()
        })
    return data_format


def collect(replay, env, env_step, reset, obs, 
            prev_action, reward, discount, next_obs, **kwargs):
    # print('obs', obs.shape, obs.dtype)
    # print('action', prev_action.shape)
    # print('reward', reward)
    # print('discount', discount)
    if isinstance(reset, np.ndarray):
        for i, r in enumerate(reset):
            replay.add(i, r, obs=obs[i], prev_action=prev_action[i],
                reward=reward[i], discount=discount[i])
            if r:
                zero_action = np.zeros_like(prev_action[i], dtype=prev_action[0].dtype) \
                    if isinstance(prev_action[0], np.ndarray) else np.float32(0)
                replay.add(i, False, obs=env.prev_obs(i)[0], prev_action=zero_action,
                    reward=np.float32(0), discount=np.float32(1))
    else:
        replay.add(0, reset, obs=obs, prev_action=prev_action,
            reward=reward, discount=discount)
        if reset:
            zero_action = np.zeros_like(prev_action, dtype=prev_action.dtype) \
                if isinstance(prev_action, np.ndarray) else np.float32(0)
            replay.add(0, False, obs=env.prev_obs()[0], prev_action=zero_action,
                reward=np.float32(0), discount=np.float32(1))


class Agent(Memory, AgentBase):
    @override(AgentBase)
    def _add_attributes(self, env, dataset):
        super()._add_attributes(env, dataset)
        self._to_log_images = Every(self.LOG_PERIOD)
        self._setup_memory_state_record()

    @override(AgentBase)
    def _construct_optimizers(self):
        dynamics_models = [self.encoder, self.rssm, self.decoder, self.reward]
        if hasattr(self, 'discount'):
            dynamics_models.append(self.discount)

        self._model_opt = self._construct_opt(
            models=dynamics_models, lr=self._model_lr)
        self._actor_opt = self._construct_opt(
            models=self.actor, lr=self._actor_lr)
        self._value_opt = self._construct_opt(
            models=self.value, lr=self._value_lr)

        return dynamics_models + [self.actor, self.value]

    @override(AgentBase)
    def _build_learn(self, env):
        # time dimension must be explicitly specified here
        # otherwise, InaccessibleTensorError arises when expanding rssm
        TensorSpecs = dict(
            obs=((self._sample_size, *self._obs_shape), self._dtype, 'obs'),
            prev_action=((self._sample_size, self._action_dim), self._dtype, 'prev_action'),
            reward=((self._sample_size,), self._dtype, 'reward'),
            discount=((self._sample_size,), self._dtype, 'discount'),
            log_images=(None, tf.bool, 'log_images')
        )
        if self._store_state:
            state_size = self.rssm.state_size
            TensorSpecs['state'] = (RSSMState(
               *[((sz, ), self._dtype, name) 
               for name, sz in zip(RSSMState._fields, state_size)]
            ))

        self.learn = build(self._learn, TensorSpecs, batch_size=self._batch_size)

    def _process_input(self, env_output, evaluation):
        obs, kwargs = super()._process_input(env_output, evaluation)
        mask = 1. - env_output.reset
        kwargs = self._add_memory_state_to_kwargs(obs, mask, kwargs)
        return obs, kwargs

    def _process_output(self, obs, kwargs, out, evaluation):
        out = self._add_tensors_to_terms(obs, kwargs, out, evaluation)
        if not evaluation:
            out[1]['prev_action'] = self._additional_rnn_inputs['prev_action']
        out = super()._process_output(obs, kwargs, out, evaluation)
        
        return out

    @step_track
    def learn_log(self, step):
        for i in range(self.N_UPDATES):
            data = self.dataset.sample()
            log_images = tf.convert_to_tensor(
                self._log_images and i == 0 and self._to_log_images(step), 
                tf.bool)
            terms = self.learn(**data, log_images=log_images)
            terms = {k: v.numpy() for k, v in terms.items()}
            self.store(**terms)
        return self.N_UPDATES

    @tf.function
    def _learn(self, obs, prev_action, reward, discount, log_images, state=None):
        with tf.GradientTape() as model_tape:
            embed = self.encoder(obs)
            if self._burn_in:
                bis = self._burn_in_size
                ss = self._sample_size - self._burn_in_size
                bi_embed, embed = tf.split(embed, [bis, ss], 1)
                bi_prev_action, prev_action = tf.split(prev_action, [bis, ss], 1)
                state, _ = self.rssm.observe(bi_embed, bi_prev_action, state)
                state = tf.nest.pack_sequence_as(state, 
                    tf.nest.map_structure(lambda x: tf.stop_gradient(x[:, -1]), state))
                
                _, obs = tf.split(obs, [bis, ss], 1)
                _, reward = tf.split(reward, [bis, ss], 1)
                _, discount = tf.split(discount, [bis, ss], 1)
            post, prior = self.rssm.observe(embed, prev_action, state)
            feature = self.rssm.get_feat(post)
            obs_pred = self.decoder(feature)
            reward_pred = self.reward(feature)
            likelihoods = AttrDict()
            likelihoods.obs_loss = -tf.reduce_mean(obs_pred.log_prob(obs))
            likelihoods.reward_loss = -tf.reduce_mean(reward_pred.log_prob(reward))
            if hasattr(self, 'discount'):
                disc_pred = self.discount(feature)
                disc_target = self._gamma * discount
                likelihoods.disc_loss = -(self._discount_scale 
                    * tf.reduce_mean(disc_pred.log_prob(disc_target)))
            prior_dist = self.rssm.get_dist(prior.mean, prior.std)
            post_dist = self.rssm.get_dist(post.mean, post.std)
            kl = tf.reduce_mean(tfd.kl_divergence(post_dist, prior_dist))
            kl = tf.maximum(kl, self._free_nats)
            model_loss = self._kl_scale * kl + sum(likelihoods.values())

        with tf.GradientTape() as actor_tape:
            imagined_feature = self._imagine_ahead(post)
            reward = self.reward(imagined_feature).mode()
            if hasattr(self, 'discount'):
                discount = self.discount(imagined_feature).mean()
            else:
                discount = self._gamma * tf.ones_like(reward)
            value = self.value(imagined_feature).mode()
            # compute lambda return at each imagined step
            returns = lambda_return(
                reward[:-1], value[:-1], discount[:-1], 
                self._lambda, value[-1], axis=0)
            # discount lambda returns based on their sequential order
            discount = tf.stop_gradient(tf.math.cumprod(tf.concat(
                [tf.ones_like(discount[:1]), discount[:-2]], 0), 0))
            actor_loss = -tf.reduce_mean(discount * returns)
            
        with tf.GradientTape() as value_tape:
            value_pred = self.value(imagined_feature)[:-1]
            target = tf.stop_gradient(returns)
            value_loss = -tf.reduce_mean(discount * value_pred.log_prob(target))

        model_norm = self._model_opt(model_tape, model_loss)
        actor_norm = self._actor_opt(actor_tape, actor_loss)
        value_norm = self._value_opt(value_tape, value_loss)
        
        act_dist, terms = self.actor(feature)
        terms = dict(
            prior_entropy=prior_dist.entropy(),
            post_entropy=post_dist.entropy(),
            kl=kl,
            value=value,
            returns=returns,
            action_entropy=act_dist.entropy(),
            **likelihoods,
            **terms,
            model_loss=model_loss,
            actor_loss=actor_loss,
            value_loss=value_loss,
            model_norm=model_norm,
            actor_norm=actor_norm,
            value_norm=value_norm,
        )

        if log_images:
            self._image_summaries(obs, prev_action, embed, obs_pred)
    
        return terms

    def _imagine_ahead(self, post):
        if hasattr(self, 'discount'):   # Omit the last step as it could be done
            post = RSSMState(*[v[:, :-1] for v in post])
        # we merge the time dimension into the batch dimension 
        # since we treat each state as a starting state when imagining
        flatten = lambda x: tf.reshape(x, [-1] + list(x.shape[2:]))
        start = RSSMState(*[flatten(x) for x in post])
        policy = lambda state: self.actor(
            tf.stop_gradient(self.rssm.get_feat(state)))[0].sample()
        states = static_scan(
            lambda prev_state, _: self.rssm.img_step(prev_state, policy(prev_state)),
            start, tf.range(self._horizon)
        )
        imagined_features = self.rssm.get_feat(states)
        return imagined_features

    def _image_summaries(self, obs, prev_action, embed, image_pred):
        truth = obs[:6] + 0.5
        recon = image_pred.mode()[:6]
        init, _ = self.rssm.observe(embed[:6, :5], prev_action[:6, :5])
        init = RSSMState(*[v[:, -1] for v in init])
        prior = self.rssm.imagine(prev_action[:6, 5:], init)
        openl = self.decoder(self.rssm.get_feat(prior)).mode()
        # join the first 5 reconstructed images to the imagined subsequent images
        model = tf.concat([recon[:, :5] + 0.5, openl + 0.5], 1)
        error = (model - truth + 1) / 2
        openl = tf.concat([truth, model, error], 2)
        self.graph_summary('video', 'comp', openl, (1, 6),
            step=self._env_step)
