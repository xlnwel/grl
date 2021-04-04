import numpy as np
import tensorflow as tf

from nn.dnc import addressing


class TestClass:
    def test_batch_invert_permutation(self):
        # Tests that the _batch_invert_permutation function correctly inverts a
        # batch of permutations.
        batch_size = 5
        length = 7

        permutations = np.empty([batch_size, length], dtype=np.int32)

        for i in range(batch_size):
            permutations[i] = np.random.permutation(length)

        inverse = addressing.batch_invert_permutation(permutations)

        for i in range(batch_size):
            for j in range(length):
                assert permutations[i][inverse[i][j]] == j

    def test_batch_gather(self):
        values = np.array([[3, 1, 4, 1], [5, 9, 2, 6], [5, 3, 5, 7]])
        indexs = np.array([[1, 2, 0, 3], [3, 0, 1, 2], [0, 2, 1, 3]])
        target = np.array([[1, 4, 3, 1], [6, 5, 9, 2], [5, 5, 3, 7]])
        result = addressing.batch_gather(values, indexs)

        assert np.all(target == result)

    def test_temporal_linkage_module(self):
        batch_size = 7
        memory_size = 4
        num_reads = 11
        num_writes = 5
        module = addressing.TemporalLinkage(
            memory_size=memory_size, num_writes=num_writes)

        state = addressing.TemporalLinkageState(
            link=np.zeros([batch_size, num_writes, memory_size, memory_size], dtype=np.float32),
            precedence_weights=np.zeros([batch_size, num_writes, memory_size], dtype=np.float32))

        num_steps = 5
        for i in range(num_steps):
            write_weights = np.random.rand(batch_size, num_writes, memory_size).astype(np.float32)
            write_weights /= write_weights.sum(2, keepdims=True) + 1

            # Simulate (in final steps) link 0-->1 in head 0 and 3-->2 in head 1
            if i == num_steps - 2:
                write_weights[0, 0, :] = tf.one_hot(0, memory_size)
                write_weights[0, 1, :] = tf.one_hot(3, memory_size)
            elif i == num_steps - 1:
                write_weights[0, 0, :] = tf.one_hot(1, memory_size)
                write_weights[0, 1, :] = tf.one_hot(2, memory_size)

            state = module(write_weights,
                            addressing.TemporalLinkageState(
                                link=state.link,
                                precedence_weights=state.precedence_weights))

        # link should be bounded in range [0, 1]
        assert np.min(state.link) >= 0
        assert np.max(state.link) <= 1

        # link diagonal should be zero
        np.testing.assert_array_equal(
            state.link.numpy()[:, :, range(memory_size), range(memory_size)],
            np.zeros([batch_size, num_writes, memory_size]))

        # link rows and columns should sum to at most 1
        assert state.link.numpy().sum(2).max() <= 1
        assert state.link.numpy().sum(3).max() <= 1

        # records our transitions in batch 0: head 0: 0->1, and head 1: 3->2
        np.testing.assert_array_equal(state.link.numpy()[0, 0, :, 0], tf.one_hot(1, memory_size))
        np.testing.assert_array_equal(state.link.numpy()[0, 1, :, 3], tf.one_hot(2, memory_size))

        # Now test calculation of forward and backward read weights
        prev_read_weights = np.random.rand(batch_size, num_reads, memory_size)
        prev_read_weights[0, 5, :] = tf.one_hot(0, memory_size)  # read 5, posn 0
        prev_read_weights[0, 6, :] = tf.one_hot(2, memory_size)  # read 6, posn 2
        forward_read_weights = module.directional_read_weights(
            tf.convert_to_tensor(state.link),
            tf.convert_to_tensor(prev_read_weights, dtype=tf.float32),
            forward=True)
        backward_read_weights = module.directional_read_weights(
            tf.convert_to_tensor(state.link),
            tf.convert_to_tensor(prev_read_weights, dtype=tf.float32),
            forward=False)

        # Check directional weights calculated correctly.
        # read=5, write=0
        assert np.all(
            forward_read_weights[0, 5, 0, :] == tf.one_hot(1, memory_size))
        # read=6, write=1
        assert np.all(
            backward_read_weights[0, 6, 1, :] == tf.one_hot(3, memory_size))

    def test_temporal_linkage_precedence_weights(self):
        batch_size = 7
        memory_size = 3
        num_writes = 5
        module = addressing.TemporalLinkage(
            memory_size=memory_size, num_writes=num_writes)

        prev_precedence_weights = np.random.rand(batch_size, num_writes,
                                                memory_size)
        write_weights = np.random.rand(batch_size, num_writes, memory_size)

        # These should sum to at most 1 for each write head in each batch.
        write_weights /= write_weights.sum(2, keepdims=True) + 1
        prev_precedence_weights /= prev_precedence_weights.sum(2, keepdims=True) + 1

        write_weights[0, 1, :] = 0  # batch 0 head 1: no writing
        write_weights[1, 2, :] /= write_weights[1, 2, :].sum()  # b1 h2: all writing

        precedence_weights = module._precedence_weights(
            prev_precedence_weights=tf.convert_to_tensor(prev_precedence_weights),
            write_weights=tf.convert_to_tensor(write_weights))

        # precedence weights should be bounded in range [0, 1]
        assert np.min(precedence_weights) >= 0
        assert np.max(precedence_weights) <= 1

        # no writing in batch 0, head 1
        np.testing.assert_allclose(precedence_weights[0, 1, :],
                            prev_precedence_weights[0, 1, :])

        # all writing in batch 1, head 2
        np.testing.assert_allclose(precedence_weights[1, 2, :], write_weights[1, 2, :])

    def test_freeness_module(self):
        batch_size = 5
        memory_size = 11
        num_reads = 3
        num_writes = 7
        module = addressing.Freeness(memory_size)

        free_gate = np.random.rand(batch_size, num_reads)

        # Produce read weights that sum to 1 for each batch and head.
        prev_read_weights = np.random.rand(batch_size, num_reads, memory_size)
        prev_read_weights[1, :, 3] = 0  # no read at batch 1, position 3; see below
        prev_read_weights /= prev_read_weights.sum(2, keepdims=True)
        prev_write_weights = np.random.rand(batch_size, num_writes, memory_size)
        prev_write_weights /= prev_write_weights.sum(2, keepdims=True)
        prev_usage = np.random.rand(batch_size, memory_size)

        # Add some special values that allows us to test the behaviour:
        prev_write_weights[1, 2, 3] = 1  # full write in batch 1, head 2, position 3
        prev_read_weights[2, 0, 4] = 1  # full read at batch 2, head 0, position 4
        free_gate[2, 0] = 1  # can free up all locations for batch 2, read head 0

        usage = module.usage(
            tf.convert_to_tensor(prev_write_weights, tf.float32),
            tf.convert_to_tensor(free_gate, tf.float32),
            tf.convert_to_tensor(prev_read_weights, tf.float32), 
            tf.convert_to_tensor(prev_usage, tf.float32))

        # Check all usages are between 0 and 1.
        assert np.min(usage) >= 0
        assert np.max(usage) <= 1

        # Check that the full write at batch 1, position 3 makes it fully used.
        np.testing.assert_equal(usage.numpy()[1][3], 1)

        # Check that the full free at batch 2, position 4 makes it fully free.
        np.testing.assert_equal(usage.numpy()[2][4], 0)

    def test_freeness_write_allocation_weights(self):
        batch_size = 7
        memory_size = 23
        num_writes = 5
        module = addressing.Freeness(memory_size)

        usage = np.random.rand(batch_size, memory_size)
        write_gates = np.random.rand(batch_size, num_writes)

        # Turn off gates for heads 1 and 3 in batch 0. This doesn't scaling down the
        # weighting, but it means that the usage doesn't change, so we should get
        # the same allocation weightings for: (1, 2) and (3, 4) (but all others
        # being different).
        write_gates[0, 1] = 0
        write_gates[0, 3] = 0
        # and turn heads 0 and 2 on for full effect.
        write_gates[0, 0] = 1
        write_gates[0, 2] = 1

        # In batch 1, make one of the usages 0 and another almost 0, so that these
        # entries get most of the allocation weights for the first and second heads.
        usage[1] = usage[1] * 0.9 + 0.1  # make sure all entries are in [0.1, 1]
        usage[1][4] = 0  # write head 0 should get allocated to position 4
        usage[1][3] = 1e-4  # write head 1 should get allocated to position 3
        write_gates[1, 0] = 1  # write head 0 fully on
        write_gates[1, 1] = 1  # write head 1 fully on

        weights = module.write_allocation_weights(
            usage=tf.convert_to_tensor(usage, tf.float32),
            write_gates=tf.convert_to_tensor(write_gates, tf.float32),
            num_writes=num_writes)

        # Check that all weights are between 0 and 1
        assert np.min(weights) >= 0
        assert np.max(weights) <= 1

        # Check that weights sum to close to 1
        np.testing.assert_allclose(
            np.sum(weights, axis=2), np.ones([batch_size, num_writes]), atol=1e-3)

        # Check the same / different allocation weight pairs as described above.
        assert np.max(np.abs(weights[0, 0, :] - weights[0, 1, :])) > 0.1
        np.testing.assert_array_equal(weights[0, 1, :], weights[0, 2, :])
        assert np.max(np.abs(weights[0, 2, :] - weights[0, 3, :])) > 0.1
        np.testing.assert_array_equal(weights[0, 3, :], weights[0, 4, :])

        np.testing.assert_allclose(weights[1][0], tf.one_hot(4, memory_size), atol=1e-3)
        np.testing.assert_allclose(weights[1][1], tf.one_hot(3, memory_size), atol=1e-3)

    def test_freeness_write_allocation_weights_gradient(self):
        batch_size = 7
        memory_size = 5
        num_writes = 3
        module = addressing.Freeness(memory_size)

        usage = tf.convert_to_tensor(np.random.rand(batch_size, memory_size), tf.float32)
        write_gates = tf.convert_to_tensor(np.random.rand(batch_size, num_writes), tf.float32)
        weights = module.write_allocation_weights(usage, write_gates, num_writes)

        theoretical, numerical = tf.test.compute_gradient(
            module.write_allocation_weights,
            [usage, write_gates],
            delta=1e-5)
        err = 0
        for a1, a2 in zip(theoretical, numerical):
            err = np.maximum(err, np.max(np.abs(a1-a2)))
        assert err < 0.01

    def test_freeness_allocation(self):
        batch_size = 7
        memory_size = 13
        usage = np.random.rand(batch_size, memory_size)
        module = addressing.Freeness(memory_size)
        allocation = module._allocation(tf.convert_to_tensor(usage, tf.float32))

        # 1. Test that max allocation goes to min usage, and vice versa.
        np.testing.assert_array_equal(np.argmin(usage, axis=1), np.argmax(allocation, axis=1))
        np.testing.assert_array_equal(np.argmax(usage, axis=1), np.argmin(allocation, axis=1))

        # 2. Test that allocations sum to almost 1.
        np.testing.assert_allclose(np.sum(allocation, axis=1), np.ones(batch_size), 0.01)

    def test_freeness_allocation_gradient(self):
        batch_size = 1
        memory_size = 5
        usage = tf.convert_to_tensor(np.random.rand(batch_size, memory_size), tf.float32)
        module = addressing.Freeness(memory_size)
        allocation = module._allocation(usage)

        theoretical, numerical = tf.test.compute_gradient(
            module._allocation,
            [usage],
            delta=1e-5)
        err = 0
        for a1, a2 in zip(theoretical, numerical):
            err = np.maximum(err, np.max(np.abs(a1-a2)))
        assert err < 0.01


if __name__ == '__main__':
    from utility.timer import Timer
    writer = tf.summary.create_file_writer(f'logs/dnc_profiler')
    writer.set_as_default()
    test = TestClass()
    tf.summary.trace_on(profiler=True)
    with Timer('gradient', 1):
        test.test_freeness_allocation_gradient()
    tf.summary.trace_export('grad', step=0, profiler_outdir='logs/dnc_profiler')
    writer.flush()
