import tensorflow as tf
from tensorflow_probability import distributions as tfd

from utility.tf_utils import assert_rank
from utility.rl_utils import epsilon_greedy, epsilon_scaled_logits
from core.module import Module, Ensemble
from nn.func import Encoder, mlp, rnn
from nn.rnns import *
from algo.dqn.nn import Q as QBase


class Q(QBase):
    def action(self, x, action_mask, noisy=True, reset=True, 
            epsilon=0, temp=1):
        qs = self.call(x, noisy=noisy, reset=reset)
        
        action = self.compute_runtime_action(
            qs, action_mask, epsilon, temp)

        return action
    
    def compute_greedy_action(self, qs, action_mask, one_hot=False):
        qs = tf.where(action_mask, qs, -1e10)
        action = tf.argmax(qs, axis=-1, output_type=tf.int32)
        if one_hot:
            action = tf.one_hot(action, self.action_dim, dtype=qs.dtype)
        return action
    
    def compute_runtime_action(self, qs, action_mask, epsilon, temp=1):
        self._action_epsilon = epsilon
        if self._stoch_action:
            self._action_temp = temp
            self._raw_logits = logits = (
                qs - tf.reduce_max(qs, axis=-1, keepdims=True)) / self._tau
            if isinstance(epsilon, tf.Tensor) or epsilon:
                self._logits = logits = epsilon_scaled_logits(logits, epsilon, temp)
            logits = tf.where(action_mask, logits, -1e10)
            self._dist = tfd.Categorical(logits)
            self._action = action = self._dist.sample()
        else:
            self._raw_action = self.compute_greedy_action(qs, action_mask)
            self._action = action = epsilon_greedy(
                self._raw_action, epsilon,
                is_action_discrete=True, 
                action_mask=action_mask,
                action_dim=self.action_dim)
        return action


class QMixer(Module):
    def __init__(self, config, n_agents, name='qmixer'):
        super().__init__(name=name)

        config = config.copy()
        self.n_agents = n_agents
        self.hidden_dim = config.pop('hidden_dim')
        self.w1 = mlp(**config, out_size=n_agents * self.hidden_dim, 
            name=f'{self.name}/w1')
        self.w2 = mlp(**config, out_size=self.hidden_dim, 
            name=f'{self.name}/w2')
        self.b = mlp([], self.hidden_dim, 
            name=f'{self.name}/b')

        config['units_list'] = [self.hidden_dim]
        self.v = mlp(**config, out_size=1,
            name=f'{self.name}/v')

    def call(self, qs, state):
        assert_rank(qs, 3)
        assert_rank(state, 3)
        B, seqlen = qs.shape[:2]
        tf.debugging.assert_shapes([
            [qs, (B, seqlen, self.n_agents)],
            [state, (B, seqlen, None)],
        ])
        qs = tf.reshape(qs, (-1, self.n_agents))
        state = tf.reshape(state, (-1, state.shape[-1]))

        w1 = tf.math.abs(self.w1(state))
        w1 = tf.reshape(w1, (-1, self.n_agents, self.hidden_dim))
        b = self.b(state)
        h = tf.nn.elu(tf.einsum('ba,bah->bh', qs, w1) + b)
        tf.debugging.assert_shapes([
            [b, (B*seqlen, self.hidden_dim)], 
            [h, (B*seqlen, self.hidden_dim)], 
        ])
        w2 = tf.math.abs(self.w2(state))
        w2 = tf.reshape(w2, (-1, self.hidden_dim, 1))
        v = self.v(state)
        y = tf.einsum('bh,bho->bo', h, w2) + v
        y = tf.reshape(y, [-1, seqlen])
        tf.debugging.assert_shapes([
            [v, (B*seqlen, 1)], 
            [y, (B, seqlen)], 
        ])

        return y

class QMIX(Ensemble):
    def __init__(self, config, env, **kwargs):
        self.State = dict(
            lstm=LSTMState,
            mlstm=LSTMState,
            gru=GRUState,
            mgru=GRUState,
        )[config['q_rnn']['rnn_name']]

        super().__init__(
            model_fn=create_components, 
            config=config,
            env=env,
            **kwargs)

    @tf.function
    def action(self, obs, 
            action_mask, state, 
            evaluation=False, 
            epsilon=0,
            temp=1.,
            return_stats=False,
            return_eval_stats=False,
            **kwargs):
        x, state = self.encode(obs, state, online=True)
        action = self.q.action(
            x, action_mask, epsilon=epsilon, temp=temp)
        
        action = tf.squeeze(action)
        terms = {}
        # For now, we do not return stats, which requires to call QMIXER to compute the joint action-values

        return (action, terms), state

    def encode(self, obs, state, online=True):
        encoder = self.q_encoder if online else self.target_q_encoder
        rnn = self.q_rnn if online else self.target_q_rnn

        if obs.shape.ndims % 2 != 0:
            obs = tf.expand_dims(obs, 1)
        assert_rank(obs, 4)
        
        x = encoder(obs)                        # [B, S, A, F]
        seqlen, n_agents = x.shape[1:3]

        tf.debugging.assert_equal(n_agents, self.qmixer.n_agents)
        x = tf.transpose(x, [0, 2, 1, 3])       # [B, A, S, F]
        x = tf.reshape(x, [-1, *x.shape[2:]])   # [B * A, S, F]
        x = rnn(x, state)
        x, state = x[0], self.State(*x[1:])
        x = tf.reshape(x, (-1, n_agents, seqlen, x.shape[-1]))  # [B, A, S, F]
        x = tf.transpose(x, [0, 2, 1, 3])       # [B, S, A, F]

        if seqlen == 1:
            x = tf.squeeze(x, 1)

        return x, state

    def compute_utils(self, obs, state=None, online=True, action=None):
        x, state = self.encode(obs, state, online=online)
        
        q_cls = self.q if online else self.target_q
        utils = q_cls(x, action)

        return utils

    def compute_joint_q(self, utils, global_state, online=True, action=None):
        if action is None:
            q_cls = self.q if online else self.target_q
            action = q_cls.compute_greedy_action(utils, one_hot=True)
        util = tf.reduce_sum(action * utils, axis=-1)

        qmixer = self.qmixer if online else self.target_qmixer
        q = qmixer(util, global_state)

        return q

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        if inputs is None:
            assert batch_size is not None
            inputs = tf.zeros([batch_size, 1, 1])
        state = self.State(*self.q_rnn.get_initial_state(inputs))
        return state

    @property
    def state_size(self):
        return self.State(*self.q_rnn.state_size)
    
    @property
    def state_keys(self):
        return self.State._fields

def create_components(config, env, n_agents, **kwargs):
    return dict(
        q_encoder=Encoder(config['q_encoder'], name='encoder'),
        q_rnn=rnn(config['q_rnn'], name='q_rnn'),
        q=Q(config['q'], env.action_dim, name='q'),
        qmixer=QMixer(config['q_mixer'], n_agents, name='qmix'), 
        target_q_encoder=Encoder(config['q_encoder'], name='target_encoder'),
        target_q_rnn=rnn(config['q_rnn'], name='target_q_rnn'),
        target_q=Q(config['q'], env.action_dim, name='target_q'),
        target_qmixer=QMixer(config['q_mixer'], n_agents, name='target_qmix'), 
    )

def create_model(config, env, **kwargs):
    return QMIX(config, env, n_agents=env.n_agents, **kwargs)


if __name__ == '__main__':
    config = {
        'q_encoder': dict(
            units_list=[64],
            activation='relu',
            layer_type='dense'),
        'q_rnn': dict(
            rnn_name='gru',
            return_sequences=True,
            return_state=True,
            units=64),
        'q': dict(
            units_list=[],
            activation='relu',
            layer_type='dense',
            duel=False),
        'q_mixer': dict(
            units_list=[64],
            hidden_dim=32,
            activation='relu')
    }
    class Env:
        def __init__(self):
            self.action_dim = 5
            self.n_agents = 3
    B = 2
    env = Env()
    model = create_model(config, env)
    x = tf.random.normal((B, env.n_agents, 5))
    action_mask = tf.random.uniform((B, env.n_agents, env.action_dim), 0, 2, dtype=tf.int32)
    action_mask = tf.cast(action_mask, tf.bool)
    a = model.compute_utils(x)
    x = tf.keras.Input(shape=(env.n_agents, 5))
    action_mask = tf.keras.Input(shape=(env.n_agents, env.action_dim), dtype=tf.bool)
    model = tf.keras.Model(inputs=x, outputs=a)
    model.summary(200)
