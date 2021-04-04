import numpy as np
import tensorflow as tf

from nn.dnc.dnc import DNC
from core.tf_config import configure_gpu

BATCH_SIZE = 2
MEMORY_SIZE = 20
WORD_SIZE = 6
NUM_READS = 2
NUM_WRITES = 3
TIME_STEPS = 4
INPUT_SIZE = 10
OUTPUT_SIZE = 2

configure_gpu()

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
rnn = tf.keras.layers.RNN(dnc_cell)

class TestClass:
    def test_build_and_train(self):
        inputs = tf.random.normal([BATCH_SIZE, TIME_STEPS, INPUT_SIZE])

        targets = np.random.rand(BATCH_SIZE, TIME_STEPS, OUTPUT_SIZE)
        with tf.GradientTape() as tape:
            output, _ = rnn(inputs=inputs, initial_state=initial_state)
            loss = tf.reduce_mean(tf.square(output - targets))
        grads = tape.gradient(loss, rnn.trainable_variables)
        optimizer = tf.keras.optimizers.SGD(1)
        optimizer.apply_gradients(zip(grads, rnn.trainable_variables))

    def test_gradients(self):
        inputs = tf.random.normal([BATCH_SIZE, INPUT_SIZE])
        def forward(inputs, access_output, memory, read_weights, link, precedence_weights, controller_state):
            from nn.dnc.addressing import TemporalLinkageState
            from nn.dnc.access import AccessState
            from nn.dnc.dnc import DNCState
            output, _ = dnc_cell(inputs, 
                DNCState(access_output,
                        AccessState(memory, read_weights, initial_state.access_state.write_weights,
                            TemporalLinkageState(link, precedence_weights), 
                            initial_state.access_state.usage), 
                        controller_state))
            loss = tf.reduce_sum(output)
            return loss

        # We don't test grads for write_weights and usage since grads is blocked 
        # when computing usage in Freeness.usage
        tensors_to_check = [
            inputs, initial_state.access_output, 
            initial_state.access_state.memory, 
            initial_state.access_state.read_weights,
            initial_state.access_state.linkage.link, 
            initial_state.access_state.linkage.precedence_weights,
            initial_state.controller_state
        ]

        theoretical, numerical = tf.test.compute_gradient(
            forward,
            tensors_to_check,
            delta=1e-5)
        err = 0
        for a1, a2 in zip(theoretical, numerical):
            err = np.maximum(err, np.max(np.abs(a1-a2)))

        assert err < .1

if __name__ == '__main__':
    from utility.timer import Timer
    writer = tf.summary.create_file_writer(f'logs/dnc_profiler')
    writer.set_as_default()
    test = TestClass()
    tf.summary.trace_on(profiler=True)
    with Timer('gradient', 1):
        test.test_gradients()
    tf.summary.trace_export('grad', step=0, profiler_outdir='logs/dnc_profiler')
    writer.flush()
