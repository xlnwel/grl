import tensorflow as tf

from utility.tf_utils import assert_rank
from core.module import Ensemble
from nn.func import Encoder, rnn
from algo.dqn.nn import Q
from algo.sacd.nn import Actor


class RDQN(Ensemble):
    def __init__(self, config, env, **kwargs):
        super().__init__(
            model_fn=create_components, 
            config=config,
            env=env,
            **kwargs)
    
    @tf.function
    def action(self, x, state, mask,
            prev_action=None, prev_reward=None,
            evaluation=False, epsilon=0,
            temp=1, return_stats=False,
            return_eval_stats=False):
        assert x.shape.ndims in (2, 4), x.shape

        x, state = self._encode(
            x, state, mask, prev_action, prev_reward)
        if getattr(self, 'actor') is not None:
            action = self.actor(x, evaluation=evaluation, epsilon=epsilon, temp=temp)
        else:
            action = self.q.action(x, epsilon=epsilon, temp=temp, return_stats=return_stats)

        if evaluation:
            return tf.squeeze(action), state
        else:
            terms = {}
            action = tf.nest.map_structure(lambda x: tf.squeeze(x), action)
            if return_stats:
                action, terms = action
            terms.update({
                'mu': self.actor.compute_prob() if getattr(self, 'actor') \
                    else self.q.compute_prob()
            })
            out = tf.nest.map_structure(lambda x: tf.squeeze(x), (action, terms))
            return out, state

    def _encode(self, x, state, mask, prev_action=None, prev_reward=None):
        x = tf.expand_dims(x, 1)
        mask = tf.reshape(mask, (-1, 1))
        x = self.encoder(x)
        if hasattr(self, 'rnn'):
            additional_rnn_input = self._process_additional_input(
                x, prev_action, prev_reward)
            x, state = self.rnn(x, state, mask, 
                additional_input=additional_rnn_input)
        else:
            state = None
        x = tf.squeeze(x, 1)
        return x, state

    def _process_additional_input(self, x, prev_action, prev_reward):
        results = []
        if prev_action is not None:
            prev_action = tf.reshape(prev_action, (-1, 1))
            prev_action = tf.one_hot(prev_action, self.actor.action_dim, dtype=x.dtype)
            results.append(prev_action)
        if prev_reward is not None:
            prev_reward = tf.reshape(prev_reward, (-1, 1, 1))
            results.append(prev_reward)
        assert_rank(results, 3)
        return results

    def reset_states(self, states=None):
        if hasattr(self, 'rnn'):
            self.rnn.reset_states(states)

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        return self.rnn.get_initial_state(
            inputs, batch_size=batch_size, dtype=dtype) \
                if hasattr(self, 'rnn') else None

    @property
    def state_size(self):
        return self.rnn.state_size if hasattr(self, 'rnn') else None
        
    @property
    def state_keys(self):
        return self.rnn.state_keys if hasattr(self, 'rnn') else ()

def create_components(config, env):
    action_dim = env.action_dim
    encoder_config = config['encoder']
    q_config = config['q']

    if 'cnn_name' in encoder_config:
        encoder_config['time_distributed'] = True
    model = dict(
        encoder=Encoder(encoder_config, name='encoder'),
        q=Q(q_config, action_dim, name='q'),
        target_encoder=Encoder(encoder_config, name='target_encoder'),
        target_q=Q(q_config, action_dim, name='target_q'),
    )
    if config.get('rnn'):
        rnn_config = config['rnn']
        model.update({
            'rnn': rnn(rnn_config, name='rnn'),
            'target_rnn': rnn(rnn_config, name='target_rnn')
        })
    if config.get('actor'):
        actor_config = config['actor']
        model.update({
            'actor': Actor(actor_config, action_dim, name='actor'),
            'target_actor': Actor(actor_config, action_dim, name='target_actor')
        })

    return model

def create_model(config, env, **kwargs):
    return RDQN(config, env, **kwargs)
