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
"""DNC access modules."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import activations

from nn.dnc import addressing


AccessState = namedtuple('AccessState', 
    ('memory', 'read_weights', 'write_weights', 'linkage', 'usage')
)

def _erase_and_write(memory, address, erase_weights, values):
    """Module to erase and write in the external memory.

    Erase operation:
        M_t'(i) = M_{t-1}(i) * (1 - w_t(i) * e_t)

    Add operation:
        M_t(i) = M_t'(i) + w_t(i) * v_t

    where e are the erase_weights, w the write weights and v the values.

    Args:
        memory: 3-D tensor of shape `[batch_size, memory_size, word_size]`.
        address: 3-D tensor `[batch_size, num_writes, memory_size]`.
        erase_weights: 3-D tensor `[batch_size, num_writes, word_size]`.
        values: 3-D tensor `[batch_size, num_writes, word_size]`.

    Returns:
        3-D tensor of shape `[batch_size, num_writes, word_size]`.
    """
    # Notice that we use different processes to compute 
    # E - w_t @ e_t^⊤ and w_t @ v_t^⊤
    # In the first one we compute the product over all write heads
    # In the second one, we compute the sum
    with tf.name_scope('erase_memory'):
        expand_address = tf.expand_dims(address, -1)
        erase_weights = tf.expand_dims(erase_weights, 2)
        # w_t @ e_t^⊤
        weighted_resets = expand_address * erase_weights
        # available weightings after N writes
        # for N write heads, this is 𝜫_{i=1}^{N}(E - w_t @ e_t^⊤)
        reset_gate = tf.reduce_prod(1 - weighted_resets, 1)
        memory = memory * reset_gate

    with tf.name_scope('additive_write'):
        # w_t @ v_t^⊤
        add_matrix = tf.matmul(address, values, adjoint_a=True)
        memory += add_matrix

    return memory

class MemoryAccess(tf.Module):
    """Access module of the Differentiable Neural Computer.

    This memory module supports multiple read and write heads. It makes use of:

    *   `addressing.TemporalLinkage` to track the temporal ordering of writes in
        memory for each write head.
    *   `addressing.FreenessAllocator` for keeping track of memory usage, where
        usage increase when a memory location is written to, and decreases when
        memory is read from that the controller says can be freed.

    Write-address selection is done by an interpolation between content-based
    lookup and using unused memory.

    Read-address selection is done by an interpolation of content-based lookup
    and following the link graph in the forward or backwards read direction.
    """

    def __init__(self,
                memory_size=128,
                word_size=20,
                num_reads=1,
                num_writes=1,
                memory_init_value=1e-6,
                name='memory_access',
                dtype=tf.float32):
        """Creates a MemoryAccess module.

        Args:
            memory_size: The number of memory slots (N in the DNC paper).
            word_size: The width of each memory slot (W in the DNC paper)
            num_reads: The number of read heads (R in the DNC paper).
            num_writes: The number of write heads (fixed at 1 in the paper).
            memory_init_value: The initial value for memory
            name: The name of the module.
        """
        super().__init__(name=name)
        self._memory_size = memory_size
        self._word_size = word_size
        self._num_reads = num_reads
        self._num_writes = num_writes
        self._memory_init_value = tf.convert_to_tensor(memory_init_value, dtype)

        self._write_content_weights_mod = addressing.CosineWeights(
            num_writes, word_size, name='write_content_weights')
        self._read_content_weights_mod = addressing.CosineWeights(
            num_reads, word_size, name='read_content_weights')

        self._linkage = addressing.TemporalLinkage(memory_size, num_writes)
        self._freeness = addressing.Freeness(memory_size)

        Dense = layers.Dense
        sigmoid = activations.sigmoid
        oneplus = lambda x: activations.softplus(x) + 1

        self._layers = dict(
            read_keys=Dense(self._num_reads*self._word_size, name='read_keys', dtype=dtype),
            read_strengths=Dense(self._num_reads, name='read_strengths', activation=oneplus, dtype=dtype),
            write_keys=Dense(self._num_writes*self._word_size, name='write_keys', dtype=dtype),
            write_strengths=Dense(self._num_writes, name='write_strengths', activation=oneplus, dtype=dtype),
            write_vectors=Dense(self._num_writes*self._word_size, name='write_vectors', dtype=dtype),
            erase_vectors=Dense(self._num_writes*self._word_size, activation=sigmoid, name='erase_vectors', dtype=dtype),
            free_gate=Dense(self._num_reads, activation=sigmoid, name='free_gate', dtype=dtype),
            allocation_gate=Dense(self._num_writes, activation=sigmoid, name='allocation_gate', dtype=dtype),
            write_gate=Dense(self._num_writes, activation=sigmoid, name='write_gate', dtype=dtype),
            read_mode=Dense(self._num_reads * (2 * self._num_writes + 1),name='read_mode', dtype=dtype),
        )

    def __call__(self, inputs, prev_state):
        return self.call(inputs, prev_state)
    
    def call(self, inputs, prev_state):
        """Connects the MemoryAccess module into the graph.

        Args:
        inputs: tensor of shape `[batch_size, input_size]`. This is used to
            control this access module.
        prev_state: Instance of `AccessState` containing the previous state.

        Returns:
        A tuple `(output, next_state)`, where `output` is a tensor of shape
        `[batch_size, num_reads, word_size]`, and `next_state` is the new
        `AccessState` named tuple at the current time t.
        """
        prev_state = AccessState(*prev_state)
        inputs = self._read_inputs(inputs)

        # Update p using inputs['free_gate'] and previous read & write weights.
        usage = self._freeness.usage(
            write_weights=prev_state.write_weights,
            free_gate=inputs['free_gate'],
            read_weights=prev_state.read_weights,
            prev_usage=prev_state.usage)

        # Write to memory.
        write_weights = self._write_weights(inputs, prev_state.memory, usage)
        memory = _erase_and_write(
            memory=prev_state.memory,
            address=write_weights,
            erase_weights=inputs['erase_vectors'],
            values=inputs['write_vectors'])

        linkage_state = self._linkage(write_weights, prev_state.linkage)

        # Read from memory.
        read_weights = self._read_weights(
            inputs,
            memory=memory,
            prev_read_weights=prev_state.read_weights,
            link=linkage_state.link)
        read_words = tf.matmul(read_weights, memory)

        return (read_words, AccessState(
            memory=memory,
            read_weights=read_weights,
            write_weights=write_weights,
            linkage=linkage_state,
            usage=usage))

    def _read_inputs(self, inputs):
        """
        Applies transformations to `inputs` to get control for this module.
        Computes elements in the interface vector 𝜉
        """

        def linear(name, first_dim=None, second_dim=None):
            """
            Returns a linear transformation of `inputs`. If first_dim and second_dim
            are provide, reshape the resulting Tensor
            """
            linear = self._layers[name](inputs)
            if first_dim and second_dim:
                linear = tf.reshape(linear, [-1, first_dim, second_dim])
            return linear

        # v_t^i - The vectors to write to memory, for each write head `i`.
        write_vectors = linear('write_vectors', self._num_writes, self._word_size)

        # e_t^i - Amount to erase the memory by before writing, for each write head.
        erase_vectors = linear('erase_vectors', self._num_writes, self._word_size)

        # f_t^j - Amount that the memory at the locations read from at the previous
        # time step can be declared unused, for each read head `j`.
        free_gate = linear('free_gate')

        # g_t^{a, i} - Interpolation between writing to unallocated memory and
        # content-based lookup, for each write head `i`. Note: `a` is simply used to
        # identify this gate with allocation vs writing (as defined below).
        allocation_gate = linear('allocation_gate')

        # g_t^{w, i} - Overall gating of write amount for each write head.
        write_gate = linear('write_gate')

        # 𝜋_t^j - Mixing between "backwards" and "forwards" positions (for
        # each write head), and content-based lookup, for each read head.
        num_read_modes = 1 + 2 * self._num_writes
        read_mode = linear('read_mode', self._num_reads, num_read_modes)
        read_mode = activations.softmax(read_mode)

        # Parameters for the (read / write) "weights by content matching" modules.
        write_keys = linear('write_keys', self._num_writes, self._word_size)
        write_strengths = linear('write_strengths')

        read_keys = linear('read_keys', self._num_reads, self._word_size)
        read_strengths = linear('read_strengths')
        
        result = dict(
            read_keys=read_keys,
            read_strengths=read_strengths,
            write_keys=write_keys,
            write_strengths=write_strengths,
            write_vectors=write_vectors,
            erase_vectors=erase_vectors,
            free_gate=free_gate,
            allocation_gate=allocation_gate,
            write_gate=write_gate,
            read_mode=read_mode,
        )
        return result

    def _write_weights(self, inputs, memory, usage):
        """Calculates the memory locations to write to.

        This uses a combination of content-based lookup and finding an unused
        location in memory, for each write head.

        Args:
        inputs: Collection of inputs to the access module, including controls for
            how to chose memory writing, such as the content to look-up and the
            weighting between content-based and allocation-based addressing.
        memory: A tensor of shape  `[batch_size, memory_size, word_size]`
            containing the current memory contents.
        usage: Current memory usage, which is a tensor of shape `[batch_size,
            memory_size]`, used for allocation-based addressing.

        Returns:
        tensor of shape `[batch_size, num_writes, memory_size]` indicating where
            to write to (if anywhere) for each write head.
        """
        with tf.name_scope('write_weights'):
            # c_t^{w, i} - The content-based weights for each write head.
            write_content_weights = self._write_content_weights_mod(
                memory, inputs['write_keys'],
                inputs['write_strengths'])

            # a_t^i - The allocation weights for each write head.
            write_allocation_weights = self._freeness.write_allocation_weights(
                usage=usage,
                write_gates=(inputs['allocation_gate'] * inputs['write_gate']),
                num_writes=self._num_writes)

            # Expands gates over memory locations.
            allocation_gate = inputs['allocation_gate'][..., None]
            write_gate = inputs['write_gate'][..., None]

            # w_t^{w, i} - The write weightings for each write head.
            return write_gate * (allocation_gate * write_allocation_weights +
                                (1 - allocation_gate) * write_content_weights)

    def _read_weights(self, inputs, memory, prev_read_weights, link):
        """Calculates read weights for each read head.

        The read weights are a combination of following the link graphs in the
        forward or backward directions from the previous read position, and doing
        content-based lookup. The interpolation between these different modes is
        done by `inputs['read_mode']`.

        Args:
        inputs: Controls for this access module. This contains the content-based
            keys to lookup, and the weightings for the different read modes.
        memory: A tensor of shape `[batch_size, memory_size, word_size]`
            containing the current memory contents to do content-based lookup.
        prev_read_weights: A tensor of shape `[batch_size, num_reads,
            memory_size]` containing the previous read locations.
        link: A tensor of shape `[batch_size, num_writes, memory_size,
            memory_size]` containing the temporal write transition graphs.

        Returns:
        A tensor of shape `[batch_size, num_reads, memory_size]` containing the
        read weights for each read head.
        """
        with tf.name_scope('read_weights'):
            # c_t^{r, i} - The content weightings for each read head.
            content_weights = self._read_content_weights_mod(
                memory, inputs['read_keys'], inputs['read_strengths'])

            # Calculates f_t^i and b_t^i.
            forward_weights = self._linkage.directional_read_weights(
                link, prev_read_weights, forward=True)
            backward_weights = self._linkage.directional_read_weights(
                link, prev_read_weights, forward=False)

            backward_mode = inputs['read_mode'][:, :, :self._num_writes]
            forward_mode = (
                inputs['read_mode'][:, :, self._num_writes:2 * self._num_writes])
            content_mode = inputs['read_mode'][:, :, 2 * self._num_writes]

            assert len(forward_mode.shape) == len(backward_mode.shape) == 3
            read_weights = (
                content_mode[..., None] * content_weights 
                + tf.reduce_sum(forward_mode[..., None] * forward_weights, 2) 
                + tf.reduce_sum(backward_mode[..., None] * backward_weights, 2))

        return read_weights

    @property
    def state_size(self):
        """Returns a tuple of the shape of the state tensors."""
        return AccessState(
            memory=tf.TensorShape([self._memory_size, self._word_size]),
            read_weights=tf.TensorShape([self._num_reads, self._memory_size]),
            write_weights=tf.TensorShape([self._num_writes, self._memory_size]),
            linkage=self._linkage.state_size,
            usage=self._freeness.state_size)

    def get_initial_state(self, inputs=None, batch_size=None, dtype=tf.float32):
        state_size = self.state_size
        if inputs is not None:
            assert batch_size is None or batch_size == tf.shape(inputs)[0]
            batch_size = tf.shape(inputs)[0]
        if dtype is None:
            dtype = tf.keras.mixed_precision.global_policy().compute_dtype
        # TODO: initialize memory with small values
        return AccessState(
            memory=tf.zeros([batch_size, *state_size.memory], dtype=dtype),
            read_weights=tf.zeros([batch_size, *state_size.read_weights], dtype=dtype),
            write_weights=tf.zeros([batch_size, *state_size.write_weights], dtype=dtype),
            linkage=self._linkage.get_initial_state(batch_size=batch_size, dtype=dtype),
            usage=self._freeness.get_initial_state(batch_size=batch_size, dtype=dtype),
        )

    @property
    def output_size(self):
        """Returns the output shape."""
        return tf.TensorShape([self._num_reads, self._word_size])


if __name__ == '__main__':
    BATCH_SIZE = 2
    MEMORY_SIZE = 20
    WORD_SIZE = 6
    NUM_READS = 2
    NUM_WRITES = 3
    TIME_STEPS = 4
    INPUT_SIZE = 10


    module = MemoryAccess(MEMORY_SIZE, WORD_SIZE, NUM_READS, NUM_WRITES)
    print(module.state_size)
    initial_state = module.get_initial_state(batch_size=BATCH_SIZE)
    module_rnn = tf.keras.layers.RNN(module, stateful=True)
    inputs = tf.random.normal([BATCH_SIZE, TIME_STEPS, INPUT_SIZE])
    output, _ = module_rnn(inputs=inputs, initial_state=initial_state)
    print(output)
    output, _ = module_rnn(inputs)
    print(output)
    output, _ = module_rnn(inputs, initial_state=initial_state)
    print(output)
    print(module.trainable_variables)