import numpy as np
import tensorflow as tf

from nn.func import mlp


class TestClass:
    def test_mlp(self):
        units_list = [10, 5]
        activation = 'relu'
        kernel_initializer = 'he_uniform'
        out_dim = 3
        layer_seed = 10

        tf.random.set_seed(0)
        x = tf.random.normal([1, 2])
        
        tf.random.set_seed(layer_seed)
        plain_layers = [tf.keras.layers.Dense(
            u, activation=activation, kernel_initializer=kernel_initializer)
            for u in units_list]
        plain_layers.append(tf.keras.layers.Dense(out_dim))
        plain_y = x
        for l in plain_layers:
            plain_y = l(plain_y)

        tf.random.set_seed(layer_seed)
        mlp_layers = mlp(units_list, out_dim, activation=activation, kernel_initializer=kernel_initializer)
        mlp_y = mlp_layers(x)

        np.testing.assert_allclose(plain_y.numpy(), mlp_y.numpy())
        plain_vars = []
        for l in plain_layers:
            plain_vars += l.variables
        for pv, mv in zip(plain_vars, mlp_layers.variables):
            np.testing.assert_allclose(pv.numpy(), mv.numpy())

