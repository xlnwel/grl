import tensorflow as tf

from core.tf_config import build
from core.decorator import override
from core.optimizer import Optimizer
from algo.ppo.base import PPOBase


def collect(buffer, env, step, reset, **kwargs):
    buffer.add(**kwargs)

class Agent(PPOBase):
    @override(PPOBase)
    def _construct_optimizers(self):
        super()._construct_optimizers()
        self._disc_opt = Optimizer(
            self._optimizer, self.discriminator, self._disc_lr)
    
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
        self.learn = build(self._learn, TensorSpecs)
        TensorSpecs = dict(
            obs=(env.obs_shape, env.obs_dtype, 'obs'),
            action=(env.action_shape, env.action_dtype, 'action'),
            obs_exp=(env.obs_shape, env.obs_dtype, 'obs_exp'),
            action_exp=(env.action_shape, env.action_dtype, 'action_exp'),
        )
        self.learn_discriminator = build(self._learn_discriminator, TensorSpecs)
    
    def compute_reward(self, obs, action):
        return self.discriminator.compute_reward(obs, action)
    
    def disc_learn_log(self, exp_buffer):
        for i in range(self.N_DISC_EPOCHS):
            data = self.dataset.sample_for_disc(self._disc_batch_size)
            data_exp = exp_buffer.sample(self._disc_batch_size)
            data_exp = {f'{k}_exp': data_exp[k] for k in ['obs', 'action']}
            terms = self.learn_discriminator(**data, **data_exp)
            terms = {f'disc/{k}': v.numpy() for k, v in terms.items()}
            self.store(**terms)

        return self.N_DISC_EPOCHS

    @tf.function
    def _learn_discriminator(self, obs, action, obs_exp, action_exp):
        terms = {}
        with tf.GradientTape() as tape:
            logits = self.discriminator(obs, action)
            logits_exp = self.discriminator(obs_exp, action_exp)
            
            loss_pi = -tf.math.log_sigmoid(-logits)
            loss_exp = -tf.math.log_sigmoid(logits_exp)
            loss = tf.reduce_mean(loss_pi + loss_exp)
        
        terms['disc_norm'] = self._disc_opt(tape, loss)
        terms['loss'] = loss

        return terms
