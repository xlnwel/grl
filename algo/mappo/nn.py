import collections
import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd

from utility.tf_utils import assert_rank
from core.module import Module, Ensemble
from nn.func import Encoder, rnn, mlp
from algo.ppo.nn import Value


class Actor(Module):
    def __init__(self, config, action_dim, is_action_discrete, name='actor'):
        super().__init__(name=name)
        config = config.copy()

        self.action_dim = action_dim
        self.is_action_discrete = is_action_discrete
        self.eval_act_temp = config.pop('eval_act_temp', 1)
        assert self.eval_act_temp >= 0, self.eval_act_temp

        self.attention_action = config.pop('attention_action', False)
        self.embed_dim = config.pop('embed_dim', 10)
        if self.attention_action:
            self.embed = tf.Variable(
                tf.random.uniform((action_dim, self.embed_dim), -0.01, 0.01), 
                dtype='float32',
                trainable=True)
        self._init_std = config.pop('init_std', 1)
        if not self.is_action_discrete:
            self.logstd = tf.Variable(
                initial_value=np.log(self._init_std)*np.ones(action_dim), 
                dtype='float32', 
                trainable=True, 
                name=f'actor/logstd')
        config.setdefault('out_gain', .01)
        self._layers = mlp(**config, 
                        out_size=self.embed_dim if self.attention_action else action_dim, 
                        out_dtype='float32',
                        name=name)

    def call(self, x, action_mask=None, evaluation=False):
        x = self._layers(x)
        if self.attention_action:
            # action_mask_exp = tf.expand_dims(action_mask, -1)
            # action_embed = tf.where(action_mask_exp, self.embed, 0)
            # if x.shape.ndims == 2:
            #     x = tf.einsum('be,bae->ba', x, action_embed)
            #     tf.debugging.assert_shapes(
            #         [[x, (None, self.action_dim)]])
            # else:
            #     x = tf.einsum('bse,bsae->bsa', x, action_embed)
            #     tf.debugging.assert_shapes(
            #         [[x, (None, None, self.action_dim)]])
            x = tf.matmul(x, self.embed, transpose_b=True)

        logits = x / self.eval_act_temp \
            if evaluation and self.eval_act_temp else x
        if action_mask is not None:
            assert logits.shape[1:] == action_mask.shape[1:], (logits.shape, action_mask.shape)
            logits = tf.where(action_mask, logits, -1e10)
        act_dist = tfd.Categorical(logits)

        return act_dist

    def action(self, dist, evaluation):
        if evaluation:
            action = dist.mode()
        else:
            action = dist.sample()
            # ensures all actions are valid. This is time-consuming, 
            # it turns out that tfd.Categorical ignore events with 
            # extremely low logits
            # def cond(a, x):
            #     i = tf.stack([tf.range(3), a], 1)
            #     return tf.reduce_all(tf.gather_nd(action_mask, i))
            # def body(a, x):
            #     d = tfd.Categorical(x)
            #     a = d.sample()
            #     return (a, x)
            # action = tf.while_loop(cond, body, [action, logits])[0]
        return action

def create_components(config, env):
    action_dim = env.action_dim
    is_action_discrete = env.is_action_discrete

    return dict(
        actor_encoder=Encoder(config['actor_encoder'], name='actor_encoder'), 
        actor_rnn=rnn(config['actor_rnn'], name='actor_rnn'), 
        actor=Actor(config['actor'], action_dim, is_action_discrete),
        value_encoder=Encoder(config['value_encoder'], name='value_encoder'),
        value_rnn=rnn(config['value_rnn'], name='value_rnn'),
        value=Value(config['value'])
    )


class PPO(Ensemble):
    def __init__(self, config, env, model_fn=create_components, **kwargs):
        state = {
            'lstm': 'actor_h actor_c value_h value_c',
            'mlstm': 'actor_h actor_c value_h value_c',
            'gru': 'actor_h value_h',
            'mgru': 'actor_h value_h',
        }
        self.State = collections.namedtuple(
            'State', state[config['actor_rnn']['rnn_name']])
        
        super().__init__(
            model_fn=model_fn, 
            config=config,
            env=env,
            **kwargs)

    @tf.function
    def action(self, obs, global_state, 
            state, mask, action_mask=None,
            evaluation=False, prev_action=None, 
            prev_reward=None, **kwargs):
        assert obs.shape.ndims % 2 == 0, obs.shape

        actor_state, value_state = self.split_state(state)
        x_actor, actor_state = self.encode(
            obs, actor_state, mask, 'actor', 
            prev_action, prev_reward)
        act_dist = self.actor(x_actor, action_mask, evaluation=evaluation)
        action = self.actor.action(act_dist, evaluation)

        if evaluation:
            # we do not compute the value state at evaluation 
            return action, self.State(*actor_state, *value_state)
        else:
            x_value, value_state = self.encode(
                global_state, value_state, mask, 'value', 
                prev_action, prev_reward)
            value = self.value(x_value)
            logpi = act_dist.log_prob(action)
            terms = {'logpi': logpi, 'value': value}
            out = (action, terms)
            return out, self.State(*actor_state, *value_state)

    @tf.function(experimental_relax_shapes=True)
    def compute_value(self, global_state, state, mask, 
            prev_action=None, prev_reward=None, return_state=False):
        x, state = self.encode(
            global_state, state, mask, 'value', prev_action, prev_reward)
        value = self.value(x)
        if return_state:
            return value, state
        else:
            return value

    def encode(self, x, state, mask, stream, prev_action=None, prev_reward=None):
        assert stream in ('actor', 'value'), stream
        if stream == 'actor':
            encoder = self.actor_encoder
            rnn = self.actor_rnn
        else:
            encoder = self.value_encoder
            rnn = self.value_rnn
        if x.shape.ndims % 2 == 0:
            x = tf.expand_dims(x, 1)
        if mask.shape.ndims < 2:
            mask = tf.reshape(mask, (-1, 1))
        assert_rank(mask, 2)

        x = encoder(x)
        additional_rnn_input = self._process_additional_input(
            x, prev_action, prev_reward)
        x, state = rnn(x, state, mask, 
            additional_input=additional_rnn_input)
        if x.shape[1] == 1:
            x = tf.squeeze(x, 1)
        return x, state

    def _process_additional_input(self, x, prev_action, prev_reward):
        results = []
        if prev_action is not None:
            if self.actor.is_action_discrete:
                if prev_action.shape.ndims < 2:
                    prev_action = tf.reshape(prev_action, (-1, 1))
                prev_action = tf.one_hot(
                    prev_action, self.actor.action_dim, dtype=x.dtype)
            else:
                if prev_action.shape.ndims < 3:
                    prev_action = tf.reshape(
                        prev_action, (-1, 1, self.actor.action_dim))
            assert_rank(prev_action, 3)
            results.append(prev_action)
        if prev_reward is not None:
            if prev_reward.shape.ndims < 2:
                prev_reward = tf.reshape(prev_reward, (-1, 1, 1))
            elif prev_reward.shape.ndims == 2:
                prev_reward = tf.expand_dims(prev_reward, -1)
            assert_rank(prev_reward, 3)
            results.append(prev_reward)
        assert_rank(results, 3)
        return results

    def split_state(self, state):
        mid = len(state) // 2
        actor_state, value_state = state[:mid], state[mid:]
        return actor_state, value_state

    def reset_states(self, state=None):
        actor_state, value_state = self.split_state(state)
        self.actor_rnn.reset_states(actor_state)
        self.value_rnn.reset_states(value_state)

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        actor_state = self.actor_rnn.get_initial_state(
            inputs, batch_size=batch_size, dtype=dtype)
        value_state = self.value_rnn.get_initial_state(
            inputs, batch_size=batch_size, dtype=dtype)
        return self.State(*actor_state, *value_state)

    @property
    def state_size(self):
        return self.State(*self.actor_rnn.state_size, *self.value_rnn.state_size)

    @property
    def actor_state_size(self):
        return self.actor_rnn.state_size

    @property
    def value_state_size(self):
        return self.value_rnn.state_size

    @property
    def state_keys(self):
        return self.State(*self.State._fields)


def create_model(config, env, **kwargs):
    return PPO(config, env, create_components, **kwargs)
