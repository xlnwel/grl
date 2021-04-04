import numpy as np
import tensorflow as tf

from core.module import Module, Ensemble
from core.decorator import config
from utility.rl_utils import epsilon_greedy
from nn.func import Encoder, mlp
from algo.iqn.nn import Value


class FractionProposalNetwork(Module):
    @config
    def __init__(self, name='fqn'):
        super().__init__(name=name)
        kernel_initializer = tf.keras.initializers.VarianceScaling(
            1./np.sqrt(3.), distribution='uniform')
        self._layers = mlp(
            out_size=self.N,
            name=f'{self.name}/fpn',
            kernel_initializer=kernel_initializer)
    
    def build(self, x):
        _, self._cnn_out_size = x

    def call(self, x):
        x = self._layers(x)

        probs = tf.nn.softmax(x, axis=-1)

        tau_0 = tf.zeros([*probs.shape[:-1], 1], dtype=probs.dtype)
        tau_rest = tf.math.cumsum(probs, axis=-1)
        tau = tf.concat([tau_0, tau_rest], axis=-1)         # [B, N+1]
        tau_hat = (tau[..., :-1] + tau[..., 1:]) / 2.       # [B, N]
        tf.debugging.assert_shapes([
            [tau_0, (None, 1)],
            [probs, (None, self.N)],
            [tau, (None, self.N+1)],
            [tau_hat, (None, self.N)],
        ])

        return tau, tau_hat

class QuantileEmbed(Module):
    @config
    def __init__(self, name='qe'):
        super().__init__(name=name)

    def call(self, x, tau):
        _, cnn_out_size = x.shape
        pi = tf.convert_to_tensor(np.pi, tf.float32)
        degree = tf.cast(tf.range(1, self._tau_embed_size+1), tau.dtype) \
            * pi * tf.expand_dims(tau, -1)
        qt_embed = tf.math.cos(degree)              # [B, N, E]
        kwargs = dict(
            kernel_initializer=getattr(self, '_kernel_initializer', 'glorot_uniform'),
            activation=getattr(self, '_activation', 'relu'),
            out_dtype='float32',
        )
        qt_embed = self.mlp(
            qt_embed, 
            [cnn_out_size], 
            name=f'{self.name}/phi',
            **kwargs)  
        return qt_embed

class FQF(Ensemble):
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
        assert x.shape.ndims == 4, x.shape

        x = self.encoder(x)
        tau, tau_hat = self.fpn(x)
        qt_embed = self.qe(x, tau_hat)
        action = self.q.action(
            x, qt_embed, epsilon=epsilon, 
            temp=temp, return_stats=return_stats)
        terms = {}
        action = tf.nest.map_structure(lambda x: tf.squeeze(x), action)
        if return_stats:
            action, terms = action
        if isinstance(epsilon, tf.Tensor) or epsilon:
            
            action = epsilon_greedy(action, epsilon,
                is_action_discrete=True, 
                action_dim=self.q.action_dim)

        return action, terms

    @tf.function
    def value(self, x):
        if x.shape.ndims % 2 != 0:
            x = tf.expand_dims(x, axis=0)
        assert x.shape.ndims == 4, x.shape

        x = self.encoder(x)
        tau, tau_hat = self.fpn(x)
        qt_embed = self.qe(x, tau_hat)
        qtv, q = self.q(x, qt_embed, tau_range=tau)
        qtv = tf.squeeze(qtv)
        q = tf.squeeze(q)

        return qtv, q


def create_components(config, env, **kwargs):
    action_dim = env.action_dim
    return dict(
        encoder=Encoder(config['encoder'], name='cnn'),
        fpn=FractionProposalNetwork(config['fpn'], name='fpn'),
        qe=QuantileEmbed(config['qe'], name='qe'),
        q=Value(config['q'], action_dim, name='iqn'),
        target_encoder=Encoder(config['encoder'], name='target_cnn'),
        target_fpn=FractionProposalNetwork(config['fpn'], name='fpn'),
        target_qe=QuantileEmbed(config['qe'], name='qe'),
        target_q=Value(config['q'], action_dim, name='target_iqn'),
    )

def create_model(config, env, **kwargs):
    return FQF(config, env, **kwargs)
