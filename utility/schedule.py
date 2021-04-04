import numpy as np
import tensorflow as tf


def linear_interpolation(l_t, l_v, r_t, r_v, t, use_tf=False):
    to_float = lambda x: tf.cast(x, dtype=tf.float32) \
                            if use_tf else float(x)
    alpha = to_float(t - l_t) / to_float(r_t - l_t)
    return l_v + alpha * to_float(r_v - l_v)


def exponential_interpolation(l_t, l_v, r_t, r_v, t, use_tf=False):
    to_float = lambda x: tf.cast(x, dtype=tf.float32) \
                            if use_tf else float(x)
    alpha = to_float(t - l_t)
    q = r_v / l_v
    base = q**to_float(1/(r_t - l_t))
    return l_v * base**alpha

schedule_map = {
    'linear': linear_interpolation,
    'exp': exponential_interpolation
}


class PiecewiseSchedule:
    def __init__(self, endpoints, interpolation='linear'):
        """Piecewise schedule.
        endpoints: [(int, float)]
            list of pairs `(time, value)` meanining that schedule should output
            `value` when `t==time`. All the values for time must be sorted in
            an increasing order. When t is between two times, e.g. `(time_a, value_a)`
            and `(time_b, value_b)`, such that `time_a <= t < time_b` then value outputs
            `interpolation(value_a, value_b, alpha)` where alpha is a fraction of
            time passed between `time_a` and `time_b` for time `t`.
        interpolation: lambda float, float, float: float
            a function that takes value to the left and to the right of t according
            to the `endpoints`. Alpha is the fraction of distance from left endpoint to
            right endpoint that t has covered. See linear_interpolation for example.
        """
        idxes = [e[0] for e in endpoints]
        assert idxes == sorted(idxes)
        self._interpolation = schedule_map[interpolation]
        self._endpoints = [(int(t), float(v)) for t, v in endpoints]
        if interpolation == 'exp':
            self._exp_base = [(self._endpoints[i+1][1] / self._endpoints[i][1])\
                                **(1/(self._endpoints[i+1][0] - self._endpoints[i][0]))
                            for i in range(len(self._endpoints)-1)]
        self._outside_value = self._endpoints[-1][1]

    def value(self, t):
        if t < self._endpoints[0][0]:
            return self._endpoints[0][1]
        else:
            for (l_t, l), (r_t, r) in zip(self._endpoints[:-1], self._endpoints[1:]):
                if l_t <= t and t < r_t:
                    return self._interpolation(l_t, l, r_t, r, t)

        # t does not belong to any of the pieces, so doom.
        assert self._outside_value is not None
        return self._outside_value


class TFPiecewiseSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, endpoints, interpolation='linear', name=None):
        """Piecewise schedule.
        endpoints: [(int, int)]
            list of pairs `(time, value)` meanining that schedule should output
            `value` when `t==time`. All the values for time must be sorted in
            an increasing order. When t is between two times, e.g. `(time_a, value_a)`
            and `(time_b, value_b)`, such that `time_a <= t < time_b` then value outputs
            `interpolation(value_a, value_b, alpha)` where alpha is a fraction of
            time passed between `time_a` and `time_b` for time `t`.
        """
        super().__init__()
        idxes = [e[0] for e in endpoints]
        assert idxes == sorted(idxes)
        self._interpolation = schedule_map[interpolation]
        self._endpoints_t = tf.convert_to_tensor(
            [int(t) for t, _ in endpoints], tf.float32)
        self._endpoints_v = tf.convert_to_tensor(
            [float(v) for _, v in endpoints], tf.float32)
        self._outside_value = self._endpoints_v[-1]
        self.name=name

    @tf.function
    def __call__(self, step):
        if step.dtype != tf.float32:
            step = tf.cast(step, tf.float32)
        def compute_lr(step):
            lr = self._outside_value
            # for pair in tf.stack([self._endpoints[:-1], self._endpoints[1:]], axis=1):
            for i in tf.range(len(self._endpoints_t)-1):
                l_t, r_t = self._endpoints_t[i], self._endpoints_t[i+1]
                l, r = self._endpoints_v[i], self._endpoints_v[i+1]
                if l_t <= step and step < r_t:
                    lr = self._interpolation(l_t, l, r_t, r, step, use_tf=True)
                    break

            return lr

        if step < self._endpoints_t[0]:
            return self._endpoints_v[0]
        else:
            return compute_lr(step)

    def get_config(self):
        return dict(
            endpoints=self._endpoints,
            end_learning_rate=self._outside_value_v,
            name=self.name,
        )
