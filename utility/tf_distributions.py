import numpy as np
import tensorflow as tf
from tensorflow.keras.mixed_precision import global_policy
import tensorflow_probability as tfp
tfd = tfp.distributions

EPSILON = 1e-8

""" Legacy code, use tfp instead """
def tf_scope(func):
    def name_scope(*args, **kwargs):
        with tf.name_scope(func.__name__):
            return func(*args, **kwargs)
    return name_scope

class Distribution(tf.Module):
    @tf_scope
    def log_prob(self, x):
        return -self._neg_log_prob(x)

    @tf_scope
    def neg_log_prob(self, x):
        return self._neg_log_prob(x)

    @tf_scope
    def sample(self, *args, **kwargs):
        return self._sample(*args, **kwargs)
        
    @tf_scope
    def entropy(self):
        return self._entropy()

    @tf_scope
    def kl(self, other):
        assert isinstance(other, type(self))
        return self._kl(other)

    @tf_scope
    def mean(self):
        return self._mean()

    @tf_scope
    def mode(self):
        return self._mode()

    def _neg_log_prob(self, x):
        raise NotImplementedError

    def _sample(self):
        raise NotImplementedError

    def _entropy(self):
        raise NotImplementedError

    def _kl(self, other):
        raise NotImplementedError

    def _mean(self):
        raise NotImplementedError

    def _mode(self):
        raise NotImplementedError


class Categorical(Distribution):
    """ An implementation of tfd.RelaxedOneHotCategorical """
    def __init__(self, logits, tau=None):
        self.logits = logits
        self.tau = tau  # tau in Gumbel-Softmax

    def _neg_log_prob(self, x):
        if x.shape.ndims == len(self.logits.shape) and x.shape[-1] == self.logits.shape[-1]:
            # when x is one-hot encoded
            return tf.nn.softmax_cross_entropy_with_logits(labels=x, logits=self.logits)
        else:
            return tf.nn.sparse_softmax_cross_entropy_with_logits(labels=x, logits=self.logits)

    def _sample(self, hard=True, one_hot=False):
        """
         A differentiable sampling method for categorical distribution
         reference paper: Categorical Reparameterization with Gumbel-Softmax
         original code: https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
        """
        if self.tau and one_hot:
            # # sample Gumbel(0, 1)
            # # U = tf.random.uniform(tf.shape(self.logits), minval=0, maxval=1)
            # # g = -tf.math.log(-tf.math.log(U+EPSILON)+EPSILON)
            # g = tfd.Gumbel(0, 1).sample(tf.shape(self.logits))
            # # Draw a sample from the Gumbel-Softmax distribution
            # y = tf.nn.softmax((self.logits + g) / self.tau)
            # # draw one-hot encoded sample from the softmax
            # if not one_hot:
            #     y = tf.cast(tf.argmax(y, -1), tf.int32)
            # elif hard:
            #     y_hard = tf.one_hot(tf.argmax(y, -1), self.logits.shape[-1])
            #     y = tf.stop_gradient(y_hard - y) + y
            dist = tfd.RelaxedOneHotCategorical(self.tau, logits=self.logits)
            y = dist.sample()
            if hard:
                y_hard = tf.one_hot(tf.argmax(y, -1), self.logits.shape[-1])
                y = tf.stop_gradient(y_hard - y) + y
        else:
            dist = tfd.Categorical(self.logits)
            y = dist.sample()
            if one_hot:
                y = tf.one_hot(y, self.logits.shape[-1], dtype=self.logits.dtype)
                probs = dist.probs_parameter()
                y += tf.cast(probs - tf.stop_gradient(probs), self.logits.dtype)

        return y

    def _entropy(self):
        probs = tf.nn.softmax(self.logits)
        log_probs = tf.math.log(probs + EPSILON)
        entropy = tf.reduce_sum(-probs * log_probs, axis=-1)

        return entropy

    def _kl(self, other):
        probs = tf.nn.softmax(self.logits)
        log_probs = tf.nn.log_softmax(self.logits)
        other_log_probs = tf.nn.log_softmax(other.logits)
        kl = tf.reduce_sum(probs * (log_probs - other_log_probs), axis=-1)

        return kl

    def _mode(self, one_hot=False):
        y = tf.argmax(self.logits, -1)
        if one_hot:
            y = tf.one_hot(y, self.logits.shape[-1])
        return y


class DiagGaussian(Distribution):
    def __init__(self, mean, logstd):
        self.mu, self.logstd = mean, logstd
        self.std = tf.exp(self.logstd)

    def _neg_log_prob(self, x):
        return .5 * tf.reduce_sum(np.log(2. * np.pi)
                                  + 2. * self.logstd
                                  + ((x - self.mu) / (self.std + EPSILON))**2, 
                                  axis=-1)

    def _sample(self):
        return self.mu + self.std * tf.random.normal(tf.shape(self.mu))

    def _entropy(self):
        return tf.reduce_sum(.5 * np.log(2. * np.pi) + self.logstd + .5, axis=-1)

    def _kl(self, other):
        return tf.reduce_sum(other.logstd - self.logstd - .5
                             + .5 * (self.std**2 + (self.mu - other.mean)**2)
                                / (other.std + EPSILON)**2, axis=-1)

    def _mean(self):
        return self.mu
    
    def _mode(self):
        return self.mu


class OneHotDist:

    def __init__(self, logits=None, probs=None):
        self._dist = tfd.Categorical(logits=logits, probs=probs)
        self._num_classes = self.mean().shape[-1]
        self._dtype = global_policy().compute_dtype

    @property
    def name(self):
        return 'OneHotDist'

    def __getattr__(self, name):
        return getattr(self._dist, name)

    def prob(self, events):
        indices = tf.argmax(events, axis=-1)
        return self._dist.prob(indices)

    def log_prob(self, events):
        indices = tf.argmax(events, axis=-1)
        return self._dist.log_prob(indices)

    def mean(self):
        return self._dist.probs_parameter()

    def mode(self):
        return self._one_hot(self._dist.mode())

    def sample(self, amount=None):
        amount = [amount] if amount else []
        indices = self._dist.sample(*amount)
        sample = self._one_hot(indices)
        probs = self._dist.probs_parameter()
        sample += tf.cast(probs - tf.stop_gradient(probs), self._dtype)
        return sample

    def _one_hot(self, indices):
        return tf.one_hot(indices, self._num_classes, dtype=self._dtype)

    
class SampleDist:
    """ useful to compute statistics after tfd.TransformedDistribution,
    originally from https://github.com/danijar/dreamer """
    def __init__(self, dist, samples=100):
        self._dist = dist
        self._samples = samples

    @property
    def name(self):
        return 'SampleDist'

    def __getattr__(self, name):
        return getattr(self._dist, name)

    def mean(self):
        samples = self._dist.sample(self._samples)
        return tf.reduce_mean(samples, 0)

    def mode(self):
        sample = self._dist.sample(self._samples)
        logprob = self._dist.log_prob(sample)
        return tf.gather(sample, tf.argmax(logprob))[0]

    def entropy(self):
        sample = self._dist.sample(self._samples)
        logprob = self.log_prob(sample)
        return -tf.reduce_mean(logprob, 0)


class TanhBijector(tfp.bijectors.Bijector):
    """ Tanh bijector, used to limit Gaussian action to range [-1, 1], 
    originally from https://github.com/danijar/dreamer"""
    def __init__(self, validate_args=False, name='tanh'):
        super().__init__(
            forward_min_event_ndims=0,
            validate_args=validate_args,
            name=name)

    def _forward(self, x):
        return tf.nn.tanh(x)

    def _inverse(self, y):
        dtype = y.dtype
        y = tf.cast(y, tf.float32)
        y = tf.where(
            tf.less_equal(tf.abs(y), 1.),
            tf.clip_by_value(y, -0.99999997, 0.99999997), y)
        y = tf.atanh(y)
        y = tf.cast(y, dtype)
        return y

    def _forward_log_det_jacobian(self, x):
        log2 = tf.math.log(tf.constant(2.0, dtype=x.dtype))
        return 2.0 * (log2 - x - tf.nn.softplus(-2.0 * x))

def compute_sample_mean_variance(samples, name='sample_mean_var'):
    """ Compute mean and covariance matrix from samples """
    sample_size = samples.shape.as_list()[0]
    with tf.name_scope(name):
        samples = tf.reshape(samples, [sample_size, -1])
        mean = tf.reduce_mean(samples, axis=0)
        samples_shifted = samples - mean
        # Following https://en.wikipedia.org/wiki/Estimation_of_covariance_matrices
        covariance = 1 / (sample_size - 1.) * tf.matmul(samples_shifted, samples_shifted, transpose_a=True)

        # Take into account case of zero covariance
        almost_zero_covariance = tf.fill(tf.shape(covariance), EPSILON)
        is_zero = tf.equal(tf.reduce_sum(tf.abs(covariance)), 0)
        covariance = tf.where(is_zero, almost_zero_covariance, covariance)

        return mean, covariance

def compute_kl_with_standard_gaussian(mean, covariance, name='kl_with_standard_gaussian'):
    """ Compute KL(N(mean, covariance) || N(0, I)) following 
    https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Kullback%E2%80%93Leibler_divergence
    """
    vec_dim = mean.shape[-1]
    with tf.name_scope(name):
        trace = tf.trace(covariance)
        squared_term = tf.reduce_sum(tf.square(mean))
        logdet = tf.linalg.logdet(covariance)
        result = 0.5 * (trace + squared_term - vec_dim - logdet)

    return result

if __name__ == '__main__':
    tf.random.set_seed(0)
    logits = tf.random.normal((2, 3))
    tf.random.set_seed(0)
    dist = Categorical(logits)
    a = dist.sample()
    print(a)
    tf.random.set_seed(0)
    dist = OneHotDist(logits)
    a = dist.sample()
    print(a)