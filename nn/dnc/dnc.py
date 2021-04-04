# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""DNC Cores.

These modules create a DNC core. They take input, pass parameters to the memory
access module, and integrate the output of memory to form an output.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from nn.dnc import access

DNCState = collections.namedtuple('DNCState', ('access_output', 'access_state',
                                               'controller_state'))


class DNC(tf.Module):
    """DNC core module.

    Contains controller and memory access module.
    """

    def __init__(self,
                access_config,
                controller_config,
                output_size,
                clip_value=None,
                name='dnc'):
        """Initializes the DNC core.

        Args:
            access_config: dictionary of access module configurations.
            controller_config: dictionary of controller (LSTM) module configurations.
            output_size: output dimension size of core.
            clip_value: clips controller and core output values to between
                `[-clip_value, clip_value]` if specified.
            name: module name (default 'dnc').

        Raises:
            TypeError: if direct_input_size is not None for any access module other
                than KeyValueMemory.
        """
        super().__init__(name=name)

        self._controller = layers.LSTMCell(**controller_config)
        self._access = access.MemoryAccess(**access_config)

        self._output_size = output_size
        self._clip_value = clip_value or 0

        self.out_layer = layers.Dense(self._output_size, name='output_linear')

    def _clip_if_enabled(self, x):
        if self._clip_value > 0:
            return tf.clip_by_value(x, -self._clip_value, self._clip_value)
        else:
            return x

    def __call__(self, inputs, prev_state):
        return self.call(inputs, prev_state)
    
    def call(self, inputs, prev_state):
        """Connects the DNC core into the graph.

        Args:
            inputs: Tensor input.
            prev_state: A `DNCState` tuple containing the fields `access_output`,
                `access_state` and `controller_state`. `access_state` is a 3-D Tensor
                of shape `[batch_size, num_reads, word_size]` containing read words.
                `access_state` is a tuple of the access module's state, and
                `controller_state` is a tuple of controller module's state.

        Returns:
            A tuple `(output, next_state)` where `output` is a tensor and `next_state`
            is a `DNCState` tuple containing the fields `access_output`,
            `access_state`, and `controller_state`.
        """
        prev_state = DNCState(*prev_state)
        prev_access_output = prev_state.access_output
        prev_access_state = prev_state.access_state
        prev_controller_state = prev_state.controller_state

        # Flatten to have shape `[batch_size, dim]`
        batch_flatten = layers.Flatten()
        controller_input = tf.concat(
            [batch_flatten(inputs), batch_flatten(prev_access_output)], 1)

        # Call the controller(LSTM)
        controller_output, controller_state = self._controller(
            controller_input, prev_controller_state)

        # Clip controller's output if needed
        controller_output = self._clip_if_enabled(controller_output)
        controller_state = tf.nest.map_structure(self._clip_if_enabled, controller_state)

        # Call the access module
        access_output, access_state = self._access(controller_output,
                                                prev_access_state)

        # Concatenate controller output and current read vector
        # Doing so we combine W_y and W_r into a single layer
        output = tf.concat([controller_output, batch_flatten(access_output)], 1)
        output = self.out_layer(output)
        output = self._clip_if_enabled(output)

        return output, DNCState(
            access_output=access_output,
            access_state=access_state,
            controller_state=controller_state)

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        if inputs is not None:
            assert batch_size is None or batch_size == tf.shape(inputs)[0]
            batch_size = tf.shape(inputs)[0]
        if dtype is None:
            dtype = tf.keras.mixed_precision.global_policy().compute_dtype
        return DNCState(
            access_output=tf.zeros([batch_size, *self._access.output_size], dtype=dtype),
            access_state=self._access.get_initial_state(batch_size=batch_size, dtype=dtype),
            controller_state=self._controller.get_initial_state(batch_size=batch_size, dtype=dtype))

    @property
    def state_size(self):
        return DNCState(
            access_output=self._access.output_size,
            access_state=self._access.state_size,
            controller_state=self._controller.state_size)

    @property
    def output_size(self):
        return tf.TensorShape([self._output_size])

if __name__ == '__main__':
    BATCH_SIZE = 2
    MEMORY_SIZE = 20
    WORD_SIZE = 6
    NUM_READS = 2
    NUM_WRITES = 3
    TIME_STEPS = 4
    INPUT_SIZE = 10
    OUTPUT_SIZE = 2
    access_config = {
        "memory_size": MEMORY_SIZE,
        "word_size": WORD_SIZE,
        "num_reads": NUM_READS,
        "num_writes": NUM_WRITES,
    }
    controller_config = {
        "units": 4,
    }
    dnc_cell = DNC(access_config, controller_config, OUTPUT_SIZE)
    initial_state = dnc_cell.get_initial_state(batch_size=BATCH_SIZE)
    rnn = layers.RNN(dnc_cell, return_state=True, return_sequences=True)
    inputs = tf.random.normal([BATCH_SIZE, TIME_STEPS, INPUT_SIZE])

    targets = np.random.rand(BATCH_SIZE, TIME_STEPS, OUTPUT_SIZE)
    with tf.GradientTape() as tape:
        x = rnn(inputs=inputs, initial_state=initial_state)
        output, _ = x[0], x[1:]
        loss = tf.reduce_mean(tf.square(output - targets))
    grads = tape.gradient(loss, rnn.trainable_variables)
    optimizer = tf.keras.optimizers.SGD(1)
    optimizer.apply_gradients(zip(grads, rnn.trainable_variables))