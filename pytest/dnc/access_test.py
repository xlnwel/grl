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
"""Tests for memory access."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from nn.dnc import access
from core.tf_config import configure_gpu

BATCH_SIZE = 2
MEMORY_SIZE = 20
WORD_SIZE = 6
NUM_READS = 2
NUM_WRITES = 3
TIME_STEPS = 4
INPUT_SIZE = 10

configure_gpu()

module = access.MemoryAccess(MEMORY_SIZE, WORD_SIZE, NUM_READS, NUM_WRITES)
initial_state = module.get_initial_state(batch_size=BATCH_SIZE)
module_rnn = tf.keras.layers.RNN(module)

class TestClass:
    def test_build_and_train(self):
        inputs = tf.random.normal([BATCH_SIZE, TIME_STEPS, INPUT_SIZE])

        targets = np.random.rand(BATCH_SIZE, TIME_STEPS, NUM_READS, WORD_SIZE)
        with tf.GradientTape() as tape:
            output, _ = module_rnn(inputs=inputs, initial_state=initial_state)
            loss = tf.reduce_mean(tf.square(output - targets))
        grads = tape.gradient(loss, module_rnn.trainable_variables)
        optimizer = tf.keras.optimizers.SGD(1)
        optimizer.apply_gradients(zip(grads, module_rnn.trainable_variables))

    def test_valid_read_mode(self):
        inputs = module._read_inputs(
            tf.random.normal([BATCH_SIZE, INPUT_SIZE]))

        # Check that the read modes for each read head constitute a probability
        # distribution.
        np.testing.assert_allclose(inputs['read_mode'].numpy().sum(2),
                            np.ones([BATCH_SIZE, NUM_READS]), rtol=1e-6, atol=1e-6)
        assert inputs['read_mode'].numpy().min() >= 0

    def test_write_weights(self):
        memory = 10 * (np.random.rand(BATCH_SIZE, MEMORY_SIZE, WORD_SIZE) - 0.5)
        usage = np.random.rand(BATCH_SIZE, MEMORY_SIZE)

        allocation_gate = np.random.rand(BATCH_SIZE, NUM_WRITES)
        write_gate = np.random.rand(BATCH_SIZE, NUM_WRITES)
        write_content_keys = np.random.rand(BATCH_SIZE, NUM_WRITES, WORD_SIZE)
        write_content_strengths = np.random.rand(BATCH_SIZE, NUM_WRITES)

        # Check that turning on allocation gate fully brings the write gate to
        # the allocation weighting (which we will control by controlling the usage).
        usage[:, 3] = 0
        allocation_gate[:, 0] = 1
        write_gate[:, 0] = 1

        inputs = {
            'allocation_gate': tf.convert_to_tensor(allocation_gate, tf.float32),
            'write_gate': tf.convert_to_tensor(write_gate, tf.float32),
            'write_keys': tf.convert_to_tensor(write_content_keys, tf.float32),
            'write_strengths': tf.convert_to_tensor(write_content_strengths, tf.float32)
        }

        weights = module._write_weights(inputs,
                                        tf.convert_to_tensor(memory, tf.float32),
                                        tf.convert_to_tensor(usage, tf.float32))

        # Check the weights sum to their target gating.
        np.testing.assert_allclose(np.sum(weights, axis=2), write_gate, atol=5e-2)

        # Check that we fully allocated to the third row.
        weights_0_0_target = tf.one_hot(3, MEMORY_SIZE)
        np.testing.assert_allclose(weights[0, 0], weights_0_0_target, atol=1e-3)

    def test_read_weights(self):
        memory = 10 * (np.random.rand(BATCH_SIZE, MEMORY_SIZE, WORD_SIZE) - 0.5).astype(np.float32)
        prev_read_weights = np.random.rand(BATCH_SIZE, NUM_READS, MEMORY_SIZE).astype(np.float32)
        prev_read_weights /= prev_read_weights.sum(2, keepdims=True) + 1

        link = np.random.rand(BATCH_SIZE, NUM_WRITES, MEMORY_SIZE, MEMORY_SIZE).astype(np.float32)
        # Row and column sums should be at most 1:
        link /= np.maximum(link.sum(2, keepdims=True), 1)
        link /= np.maximum(link.sum(3, keepdims=True), 1)

        # We query the memory on the third location in memory, and select a large
        # strength on the query. Then we select a content-based read-mode.
        read_keys = np.random.rand(BATCH_SIZE, NUM_READS, WORD_SIZE)
        read_keys[0, 0] = memory[0, 3]
        read_strengths = tf.ones(shape=[BATCH_SIZE, NUM_READS]) * 100.
        read_mode = np.random.rand(BATCH_SIZE, NUM_READS, 1 + 2 * NUM_WRITES)
        read_mode[0, 0, :] = tf.one_hot(2 * NUM_WRITES, 1 + 2 * NUM_WRITES)
        inputs = {
            'read_keys': tf.convert_to_tensor(read_keys, tf.float32),
            'read_strengths': read_strengths,
            'read_mode': tf.convert_to_tensor(read_mode, tf.float32),
        }
        read_weights = module._read_weights(inputs, memory, prev_read_weights,
                                                link)

        # read_weights for batch 0, read head 0 should be memory location 3
        np.testing.assert_allclose(
            read_weights[0, 0, :], tf.one_hot(3, MEMORY_SIZE), atol=1e-3)

    def test_gradients(self):
        inputs = tf.convert_to_tensor(np.random.randn(BATCH_SIZE, INPUT_SIZE), tf.float32)
        def forward(inputs, memory, read_weights, link, precedence_weights):
            from nn.dnc.addressing import TemporalLinkageState
            from nn.dnc.access import AccessState
            output, _ = module(inputs, 
                AccessState(memory, read_weights, initial_state.write_weights,
                    TemporalLinkageState(link, precedence_weights), initial_state.usage))
            loss = tf.reduce_sum(output)
            return loss

        # We don't test grads for write_weights and usage since their grads is blocked 
        # when computing usage in Freeness.usage
        tensors_to_check = [
            inputs, initial_state.memory, 
            initial_state.read_weights, 
            initial_state.linkage.link,
            initial_state.linkage.precedence_weights
        ]

        theoretical, numerical = tf.test.compute_gradient(
            forward,
            tensors_to_check,
            delta=1e-5)
        err = 0
        for a1, a2 in zip(theoretical, numerical):
            err = np.maximum(err, np.max(np.abs(a1-a2)))
        assert err < 0.1

if __name__ == '__main__':
    test = TestClass()
    test.test_gradients()
