import tensorflow as tf

from core.tf_config import build
from core.decorator import override
from core.mixin import Memory
from algo.dqn.base import DQNBase


def get_data_format(*, env, replay_config, agent_config,
        model, **kwargs):
    is_per = replay_config['replay_type'].endswith('per')
    store_state = agent_config['store_state']
    sample_size = agent_config['sample_size']
    obs_dtype = tf.uint8 if len(env.obs_shape) == 3 else tf.float32
    data_format = dict(
        obs=((None, sample_size+1, *env.obs_shape), obs_dtype),
        action=((None, sample_size+1, *env.action_shape), tf.int32),
        reward=((None, sample_size), tf.float32), 
        mu=((None, sample_size+1), tf.float32),
        discount=((None, sample_size), tf.float32),
        mask=((None, sample_size+1), tf.float32),
    )
    if is_per:
        data_format['idxes'] = ((None), tf.int32)
        if replay_config.get('use_is_ratio'):
            data_format['IS_ratio'] = ((None, ), tf.float32)
    if store_state:
        state_size = model.state_size
        from tensorflow.keras.mixed_precision import global_policy
        state_dtype = global_policy().compute_dtype
        data_format.update({
            k: ((None, v), state_dtype)
                for k, v in state_size._asdict().items()
        })

    return data_format

def collect(replay, env, step, reset, next_obs, **kwargs):
    replay.add(**kwargs)


class RDQNBase(Memory, DQNBase):
    """ Initialization """
    @override(DQNBase)
    def _add_attributes(self, env, dataset):
        super()._add_attributes(env, dataset)
        self._burn_in_size = 'rnn' in self.model and self._burn_in_size
        self._setup_memory_state_record()

    @override(DQNBase)
    def _build_learn(self, env):
        seqlen = self._sample_size + self._burn_in_size
        # Explicitly instantiate tf.function to initialize variables
        TensorSpecs = dict(
            obs=((seqlen+1, *env.obs_shape), env.obs_dtype, 'obs'),
            action=((seqlen+1, env.action_dim), tf.float32, 'action'),
            reward=((seqlen,), tf.float32, 'reward'),
            mu=((seqlen+1,), tf.float32, 'mu'),
            discount=((seqlen,), tf.float32, 'discount'),
            mask=((seqlen+1,), tf.float32, 'mask')
        )
        if self._is_per and getattr(self, '_use_is_ratio', self._is_per):
            TensorSpecs['IS_ratio'] = ((), tf.float32, 'IS_ratio')
        if self._store_state:
            state_type = type(self.model.state_size)
            TensorSpecs['state'] = state_type(*[((sz, ), self._dtype, name) 
                for name, sz in self.model.state_size._asdict().items()])
        if self._additional_rnn_inputs:
            if 'prev_action' in self._additional_rnn_inputs:
                TensorSpecs['prev_action'] = (
                    (seqlen, *env.action_shape), env.action_dtype, 'prev_action')
            if 'prev_reward' in self._additional_rnn_inputs:
                TensorSpecs['prev_reward'] = (
                    (seqlen,), self._dtype, 'prev_reward')    # this reward should be unnormlaized
        self.learn = build(self._learn, TensorSpecs, batch_size=self._batch_size)

    """ Call """
    # @override(DQNBase)
    def _process_input(self, env_output, evaluation):
        obs, kwargs = super()._process_input(env_output, evaluation)
        mask = 1. - env_output.reset
        kwargs = self._add_memory_state_to_kwargs(obs, mask, kwargs=kwargs)
        return obs, kwargs

    # @override(DQNBase)
    def _process_output(self, obs, kwargs, out, evaluation):
        out = self._add_tensors_to_terms(obs, kwargs, out, evaluation)
        out = super()._process_output(obs, kwargs, out, evaluation)
        out = self._add_non_tensors_to_terms(out, kwargs, evaluation)
        return out

    def _compute_priority(self, priority):
        """ p = (p + 𝝐)**𝛼 """
        priority = (self._per_eta*tf.math.reduce_max(priority, axis=1) 
                    + (1-self._per_eta)*tf.math.reduce_mean(priority, axis=1))
        priority += self._per_epsilon
        priority **= self._per_alpha
        return priority

    def _compute_embed(self, obs, mask, state, add_inp, online=True):
        encoder = self.encoder if online else self.target_encoder
        x = encoder(obs)
        if 'rnn' in self.model:
            rnn = self.rnn if online else self.target_rnn
            x, state = rnn(x, state, mask, additional_input=add_inp)
        return x, state

    def _compute_target(self):
        raise NotImplementedError

    def _compute_target_and_process_data(self, 
            obs, action, reward, 
            discount, mu, mask, state=None, 
            prev_action=None, prev_reward=None):
        add_inp = []
        if prev_action is not None:
            prev_action = tf.concat([prev_action, action[:, :-1]], axis=1)
            add_inp.append(prev_action)
        if prev_reward is not None:
            prev_reward = tf.concat([prev_reward, reward[:, :-1]], axis=1)
            add_inp.append(prev_reward)
        
        target, terms = self._compute_target(
            obs, action, reward, discount, 
            mu, mask, state, add_inp)
        if self._burn_in_size:
            bis = self._burn_in_size
            ss = self._sample_size
            bi_obs, obs, _ = tf.split(obs, [bis, ss, 1], 1)
            bi_mask, mask, _ = tf.split(mask, [bis, ss, 1], 1)
            _, mu, _ = tf.split(mu, [bis, ss, 1], 1)
            if add_inp:
                bi_add_inp, add_inp, _ = zip(*[tf.split(v, [bis, ss, 1]) for v in add_inp])
            else:
                bi_add_inp = []
            _, state = self._compute_embed(bi_obs, bi_mask, state, bi_add_inp)
        else:
            obs, _ = tf.split(obs, [self._sample_size, 1], 1)
            mask, _ = tf.split(mask, [self._sample_size, 1], 1)
            mu, _ = tf.split(mu, [self._sample_size, 1], 1)
        action, _ = tf.split(action, [self._sample_size, 1], 1)

        return obs, action, mu, mask, target, state, add_inp, terms