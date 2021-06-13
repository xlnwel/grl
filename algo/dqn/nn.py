import tensorflow as tf
from tensorflow_probability import distributions as tfd

from utility.tf_utils import assert_shape_compatibility
from utility.rl_utils import epsilon_greedy, epsilon_scaled_logits
from core.module import Module, Ensemble
from nn.func import Encoder, mlp


class Q(Module):
    def __init__(self, config, action_dim, name='value'):
        super().__init__(name=name)
        config = config.copy()
        
        self._action_dim = action_dim
        self._duel = config.pop('duel', False)
        self._layer_type = config.get('layer_type', 'dense')
        self._stoch_action = config.pop('stoch_action', False)
        self._tau = config.pop('tau', 1)

        self._add_layer(config)

    def _add_layer(self, config):
        """ Network definition """
        if self._duel:
            self._v_layers = mlp(
                **config,
                out_size=1, 
                name=self.name+'/v',
                out_dtype='float32')
        self._layers = mlp(
            **config, 
            out_size=self.action_dim, 
            name=self.name,
            out_dtype='float32')

    @property
    def action_dim(self):
        return self._action_dim

    @property
    def stoch_action(self):
        return self._stoch_action

    def action(self, x, noisy=True, reset=True, epsilon=0, temp=1, return_stats=False):
        qs = self.call(x, noisy=noisy, reset=reset)
        
        action = self.compute_runtime_action(qs, epsilon, temp)

        if return_stats:
            if self._stoch_action:
                one_hot = tf.one_hot(action, qs.shape[-1])
                q = tf.reduce_sum(qs * one_hot, axis=-1)
            else:
                q = tf.reduce_max(qs, axis=-1)
            return action, {'q': q}
        else:
            return action
    
    def call(self, x, action=None, noisy=False, reset=False):
        kwargs = dict(noisy=noisy, reset=reset) if self._layer_type == 'noisy' else {}

        if self._duel:
            v = self._v_layers(x, **kwargs)
            a = self._layers(x, **kwargs)
            q = v + a - tf.reduce_mean(a, axis=-1, keepdims=True)
        else:
            q = self._layers(x, **kwargs)

        if action is not None:
            if action.dtype.is_integer:
                action = tf.one_hot(action, self.action_dim, dtype=q.dtype)
            assert_shape_compatibility([action, q])
            q = tf.reduce_sum(q * action, -1)
        
        return q

    def reset_noisy(self):
        if self._layer_type == 'noisy':
            if self._duel:
                self._v_layers.reset()
            self._layers.reset()
    
    def compute_greedy_action(self, qs, one_hot=False):
        action = tf.argmax(qs, axis=-1, output_type=tf.int32)
        if one_hot:
            action = tf.one_hot(action, self.action_dim, dtype=qs.dtype)
        return action
    
    def compute_runtime_action(self, qs, epsilon, temp=1):
        self._action_epsilon = epsilon
        if self._stoch_action:
            self._action_temp = temp
            self._raw_logits = logits = (qs - tf.reduce_max(qs, axis=-1, keepdims=True)) / self._tau
            if isinstance(epsilon, tf.Tensor) or epsilon:
                self._logits = logits = epsilon_scaled_logits(logits, epsilon, temp)
            self._dist = tfd.Categorical(logits)
            self._action = action = self._dist.sample()
        else:
            self._raw_action = self.compute_greedy_action(qs)
            self._action = action = epsilon_greedy(
                self._raw_action, epsilon,
                is_action_discrete=True, 
                action_dim=self.action_dim)
        return action

    def compute_prob(self):
        if self._stoch_action:
            # TODO: compute exact probs taking into account temp and epsilon
            prob = self._dist.prob(self._action)
        else:
            eps_prob = self._action_epsilon / self.action_dim
            max_prob = 1 - self._action_epsilon + eps_prob
            prob = tf.where(self._action == self._raw_action, max_prob, eps_prob)
        return prob


class DQN(Ensemble):
    def __init__(self, config, env, **kwargs):
        super().__init__(
            model_fn=create_components, 
            config=config,
            env=env,
            **kwargs)

    @tf.function
    def action(self, x, 
            evaluation=False, 
            epsilon=0,
            temp=1.,
            return_stats=False,
            return_eval_stats=False):
        assert x.shape.ndims in (2, 4), x.shape

        x = self.encoder(x)
        noisy = not evaluation
        action = self.q.action(
            x, noisy=noisy, reset=noisy, 
            epsilon=epsilon, temp=temp, 
            return_stats=return_stats)
        terms = {}
        action = tf.nest.map_structure(lambda x: tf.squeeze(x), action)
        if return_stats:
            action, terms = action

        return action, terms


def create_components(config, env, **kwargs):
    action_dim = env.action_dim
    return dict(
        encoder=Encoder(config['encoder'], name='encoder'),
        q=Q(config['q'], action_dim, name='q'),
        target_encoder=Encoder(config['encoder'], name='target_encoder'),
        target_q=Q(config['q'], action_dim, name='target_q'),
    )

def create_model(config, env, **kwargs):
    return DQN(config, env, **kwargs)
