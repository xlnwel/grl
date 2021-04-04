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
"""DNC addressing modules."""
from collections import namedtuple
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# Ensure values are greater than epsilon to avoid numerical instability.
_EPSILON = 1e-6

TemporalLinkageState = namedtuple('TemporalLinkageState',
    ('link', 'precedence_weights')
)

def batch_invert_permutation(permutations):
    """Returns batched `tf.invert_permutation` for every row in `permutations`."""
    return tf.stack([tf.math.invert_permutation(indices) 
                    for indices in tf.unstack(permutations)])

def batch_gather(values, indices):
    return tf.stack([tf.gather(v, i) 
                    for v, i in zip(tf.unstack(values), tf.unstack(indices))])


class CosineWeights(tf.Module):
    """Cosine-weighted attention.

    Calculates the cosine similarity between a query and each word in memory, then
    applies a weighted softmax to return a sharp distribution.
    """
    def __init__(self,
                 num_heads,
                 word_size,
                 name='cosine_weights'):
        """Initializes the CosineWeights module.

        Args:
            num_heads: number of memory heads.
            word_size: memory word size.
            name: module name (default 'cosine_weights')
        """
        super().__init__(name=name)
        self._num_heads = num_heads
        self._word_size = word_size
        self._strength_op = tf.nn.softplus

    def __call__(self, memory, keys, strengths):
        """Connects the CosineWeights module into the graph.

        Args:
            memory: A 3-D tensor of shape `[batch_size, memory_size, word_size]`.
            keys: A 3-D tensor of shape `[batch_size, num_heads, word_size]`.
            strengths: A 2-D tensor of shape `[batch_size, num_heads]`.

        Returns:
            Weights tensor of shape `[batch_size, num_heads, memory_size]`.
        """
        # Calculates cosine similarity between the query vector and words in memory.
        normalized_keys = tf.math.l2_normalize(keys, axis=2, epsilon=1e-6)
        normalized_memory = tf.math.l2_normalize(memory, axis=2, epsilon=1e-6)
        similarity = tf.matmul(normalized_keys, normalized_memory, adjoint_b=True)

        assert strengths.shape.ndims == 2
        assert similarity.shape.ndims == 3
        assert strengths.shape == similarity.shape[:2]
        return keras.activations.softmax(tf.expand_dims(strengths, -1) * similarity)

        
class TemporalLinkage(tf.Module):
    """Keeps track of write order for forward and backward addressing.

    This is a pseudo-RNNCore module, whose state is a pair `(link,
    precedence_weights)`, where `link` is a (collection of) graphs for (possibly
    multiple) write heads (represented by a tensor with values in the range
    [0, 1]), and `precedence_weights` records the "previous write locations" used
    to build the link graphs.

    The function `directional_read_weights` computes addresses following the
    forward and backward directions in the link graphs.
    """

    def __init__(self, 
                memory_size, 
                num_writes, 
                name='temporal_linkage'):
        """Construct a TemporalLinkage module.

        Args:
            memory_size: The number of memory slots.
            num_writes: The number of write heads.
            name: Name of the module.
        """
        super().__init__(name=name)
        self._memory_size = memory_size
        self._num_writes = num_writes

    def __call__(self, write_weights, prev_state):
        """Calculate the updated linkage state given the write weights.

        Args:
            write_weights: A tensor of shape `[batch_size, num_writes, memory_size]`
                containing the memory addresses of the different write heads.
            prev_state: `TemporalLinkageState` tuple containg a tensor `link` of
                shape `[batch_size, num_writes, memory_size, memory_size]`, and a
                tensor `precedence_weights` of shape `[batch_size, num_writes,
                memory_size]` containing the aggregated history of recent writes.

        Returns:
            A `TemporalLinkageState` tuple `next_state`, which contains the updated
            link and precedence weights.
        """
        link = self._link(prev_state.link, prev_state.precedence_weights,
                          write_weights)
        precedence_weights = self._precedence_weights(prev_state.precedence_weights,
                                                    write_weights)

        return TemporalLinkageState(
            link=link, precedence_weights=precedence_weights)

    def directional_read_weights(self, link, prev_read_weights, forward):
        """Calculates the forward or the backward read weights (f/b).

        For each read head (at a given address), there are `num_writes` link graphs
        to follow. Thus this function computes a read address for each of the
        `num_reads * num_writes` pairs of read and write heads.

        Args:
            link: tensor of shape `[batch_size, num_writes, memory_size,
                memory_size]` representing the link graphs L_t.
            prev_read_weights: tensor of shape `[batch_size, num_reads,
                memory_size]` containing the previous read weights w_{t-1}^r.
            forward: Boolean indicating whether to follow the "future" direction in
                the link graph (True) or the "past" direction (False).

        Returns:
            tensor of shape `[batch_size, num_reads, num_writes, memory_size]`
        """

        with tf.name_scope('directional_read_weights'):
            # We calculate the forward and backward directions for each pair of
            # read and write heads; hence we need to tile the read weights and do a
            # sort of "outer product" to get this.
            # shape: `[batch_size, num_writes, num_reads, memory_size]`
            # we cannot stack using axis=2, which would result in inconsistent first
            # dimension in tf.matmul
            expanded_read_weights = tf.stack([prev_read_weights] * self._num_writes, 
                                            axis=1)
            result = tf.matmul(expanded_read_weights, link, adjoint_b=forward)
            # Swap dimensions 1, 2 so order is [batch, reads, writes, memory]:
            return tf.transpose(result, perm=[0, 2, 1, 3])

    def _link(self, prev_link, prev_precedence_weights, write_weights):
        """Calculates the new link graphs (L).
        L = (E - W @ 1^⊤ -1 @ W^⊤)L + w p_{t-1}^⊤
        L = L(E - I)

        Args:
            prev_link: A tensor of shape `[batch_size, num_writes, memory_size,
                memory_size]` representing the previous link graphs for each write
                head.
            prev_precedence_weights: A tensor of shape `[batch_size, num_writes,
                memory_size]` which is the previous "aggregated" write weights for
                each write head.
            write_weights: A tensor of shape `[batch_size, num_writes, memory_size]`
                containing the new locations in memory written to.

        Returns:
            A tensor of shape `[batch_size, num_writes, memory_size, memory_size]`
            containing the new link graphs for each write head.
        """
        with tf.name_scope('link'):
            batch_size = tf.shape(prev_link)[0]
            write_weights_i = tf.expand_dims(write_weights, 3)
            write_weights_j = tf.expand_dims(write_weights, 2)
            prev_precedence_weights_j = tf.expand_dims(prev_precedence_weights, 2)

            prev_link_scale = 1 - write_weights_i - write_weights_j
            new_link = write_weights_i * prev_precedence_weights_j
            link = prev_link_scale * prev_link + new_link

            zeros = tf.zeros((batch_size, self._num_writes, self._memory_size), 
                            dtype=link.dtype)
            return tf.linalg.set_diag(link, zeros)

    def _precedence_weights(self, prev_precedence_weights, write_weights):
        """Calculates the new precedence weights (p) given the current write weights.

        The precedence weights are the "aggregated write weights" for each write
        head, where write weights with sum close to zero will leave the precedence
        weights unchanged, but with sum close to one will replace the precedence
        weights.

        Args:
            prev_precedence_weights: A tensor of shape `[batch_size, num_writes,
                memory_size]` containing the previous precedence weights.
            write_weights: A tensor of shape `[batch_size, num_writes, memory_size]`
                containing the new write weights.

        Returns:
            A tensor of shape `[batch_size, num_writes, memory_size]` containing the
            new precedence weights.
        """
        with tf.name_scope('precedence_weights'):
            write_sum = tf.reduce_sum(write_weights, axis=2, keepdims=True)
            return (1 - write_sum) * prev_precedence_weights + write_weights
    
    @property
    def state_size(self):
        """Returns a `TemporalLinkageState` tuple of the state tensors' shapes."""
        return TemporalLinkageState(
            link=tf.TensorShape(
                [self._num_writes, self._memory_size, self._memory_size]),
            precedence_weights=tf.TensorShape([self._num_writes,
                                                self._memory_size])
        )

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        state_size = self.state_size
        if inputs is not None:
            assert batch_size is None or batch_size == tf.shape(inputs)[0]
            batch_size = tf.shape(inputs)[0]
        if dtype is None:
            dtype = tf.keras.mixed_precision.global_policy().compute_dtype
        return TemporalLinkageState(
            link=tf.zeros([batch_size, *state_size.link], dtype=dtype),
            precedence_weights=tf.zeros([batch_size, *state_size.precedence_weights], dtype=dtype),
        )

class Freeness(tf.Module):
    """Memory usage that is increased by writing and decreased by reading.

    This module is a pseudo-RNNCore whose state is a tensor with values in
    the range [0, 1] indicating the usage of each of `memory_size` memory slots.

    The usage is:

    *   Increased by writing, where usage is increased towards 1 at the write
        addresses.
    *   Decreased by reading, where usage is decreased after reading from a
        location when free_gate is close to 1.

    The function `write_allocation_weights` can be invoked to get free locations
    to write to for a number of write heads.
    """

    def __init__(self, memory_size, name='freeness'):
        """Creates a Freeness module.

        Args:
            memory_size: Number of memory slots.
            name: Name of the module.
        """
        super().__init__(name=name)
        self._memory_size = memory_size
    
    def usage(self, write_weights, free_gate, read_weights, prev_usage):
        """Calculates the new memory usage u_t.

        Memory that was written to in the previous time step will have its usage
        increased; memory that was read from and the controller says can be "freed"
        will have its usage decreased.

        Args:
            write_weights: tensor of shape `[batch_size, num_writes,
                memory_size]` giving write weights at previous time step.
            free_gate: tensor of shape `[batch_size, num_reads]` which indicates
                which read heads read memory that can now be freed.
            read_weights: tensor of shape `[batch_size, num_reads,
                memory_size]` giving read weights at previous time step.
            prev_usage: tensor of shape `[batch_size, memory_size]` giving
                usage u_{t - 1} at the previous time step, with entries in range
                [0, 1].

        Returns:
            tensor of shape `[batch_size, memory_size]` representing updated memory
            usage.
        """
        # Calculation of usage is not differentiable with respect to write weights.
        with tf.name_scope('usage'):
            write_weights = tf.stop_gradient(write_weights)
            usage = self._usage_after_write(prev_usage, write_weights)
            usage = self._usage_after_read(usage, free_gate, read_weights)

        return usage

    def write_allocation_weights(self, usage, write_gates, num_writes=3):
        """Calculates freeness-based locations for writing to.

        This finds unused memory by ranking the memory locations by usage, for each
        write head. (For more than one write head, we use a "simulated new usage"
        which takes into account the fact that the previous write head will increase
        the usage in that area of the memory.)

        Args:
            usage: A tensor of shape `[batch_size, memory_size]` representing
                current memory usage.
            write_gates: A tensor of shape `[batch_size, num_writes]` with values in
                the range [0, 1] indicating how much each write head does writing
                based on the address returned here (and hence how much usage
                increases).
            num_writes: The number of write heads to calculate write weights for.
                its default value only exists to pass the test code
        Returns:
            tensor of shape `[batch_size, num_writes, memory_size]` containing the
                freeness-based write locations. Note that this isn't scaled by
                `write_gate`; this scaling must be applied externally.
        """
        with tf.name_scope('write_allocation_weights'):
            # expand gatings over memory locations
            write_gates = tf.expand_dims(write_gates, -1)

            allocation_weights = []
            for i in range(num_writes):
                allocation_weights.append(self._allocation(usage))
                # update usage to take into account writing to this new allocation
                usage += ((1 - usage) * write_gates[:, i, :] * allocation_weights[i])

            # Pack the allocation weights for the write heads into one tensor.
            return tf.stack(allocation_weights, axis=1)


    def _usage_after_write(self, prev_usage, write_weights):
        """Calculates the new usage after writing to memory.
        u = u + w^w - uw^w

        Args:
            prev_usage: tensor of shape `[batch_size, memory_size]`.
            write_weights: tensor of shape `[batch_size, num_writes, memory_size]`.

        Returns:
            New usage, a tensor of shape `[batch_size, memory_size]`.
        """
        with tf.name_scope('usage_after_write'):
            # Calculate the aggregated effect of all write heads
            write_weights = 1 - tf.reduce_prod(1 - write_weights, 1)
            return prev_usage + (1 - prev_usage) * write_weights

    def _usage_after_read(self, prev_usage, free_gate, read_weights):
        """Calcualtes the new usage after reading and freeing from memory.
        u = u 𝜓
        Args:
            prev_usage: tensor of shape `[batch_size, memory_size]`.
            free_gate: tensor of shape `[batch_size, num_reads]` with entries in the
                range [0, 1] indicating the amount that locations read from can be
                freed.
            read_weights: tensor of shape `[batch_size, num_reads, memory_size]`.

        Returns:
            New usage, a tensor of shape `[batch_size, memory_size]`.
        """
        with tf.name_scope('usage_after_read'):
            # memory retention 𝜓 = 𝜫(1 - fw^r)
            free_gate = tf.expand_dims(free_gate, -1)
            free_read_weights = free_gate * read_weights
            psi = tf.reduce_prod(1 - free_read_weights, 1, name='psi')
            # new usage u = (u + w^w - uw^w)𝜓
            return prev_usage * psi

    def _allocation(self, usage):
        r"""Computes allocation by sorting `usage`.

        This corresponds to the value a = a_t[\phi_t[j]] in the paper.

        Args:
            usage: tensor of shape `[batch_size, memory_size]` indicating current
                memory usage. This is equal to u_t in the paper when we only have one
                write head, but for multiple write heads, one should update the usage
                while iterating through the write heads to take into account the
                allocation returned by this function.

        Returns:
            Tensor of shape `[batch_size, memory_size]` corresponding to allocation.
        """
        with tf.name_scope('allocation'):
            # Ensure values are not too small prior to cumprod.
            usage = usage + (1 - usage) * _EPSILON

            available = 1 - usage
            # (1-u[𝜙])
            sorted_available, indices = tf.nn.top_k(
                available, k=self._memory_size, name='sort')
            sorted_usage = 1 - sorted_available
            # 𝜫^{j-1} u[𝜙]
            prod_sorted_usage = tf.math.cumprod(sorted_usage, axis=1, exclusive=True)
            # a[𝜙]
            sorted_allocation = sorted_available * prod_sorted_usage

            # This final two lines "unsort" sorted_allocation, so that the indexing
            # corresponds to the original indexing of `usage`.
            inverse_indices = batch_invert_permutation(indices)
            
            return batch_gather(sorted_allocation, inverse_indices)

    @property
    def state_size(self):
        """Returns the shape of the state tensor."""
        return tf.TensorShape([self._memory_size])
    
    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        state_size = self.state_size
        if inputs is not None:
            assert batch_size is None or batch_size == tf.shape(inputs)[0]
            batch_size = tf.shape(inputs)[0]
        if dtype is None:
            dtype = tf.keras.mixed_precision.global_policy().compute_dtype
        return tf.zeros([batch_size, *state_size], dtype=dtype)
        
if __name__ == '__main__':
    import numpy as np
    batch_size = 7
    memory_size = 3
    num_writes = 5
    module = TemporalLinkage(
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
    print(module.trainable_variables)