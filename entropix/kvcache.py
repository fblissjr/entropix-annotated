from typing import NamedTuple

import jax
import jax.numpy as jnp

class KVCache(NamedTuple):
    k: jax.Array  # Key cache
    v: jax.Array  # Value cache

    @classmethod
    def new(cls, layers: int, bsz: int, max_seq_len: int, kv_heads: int, head_dim: int) -> 'KVCache':
        """
        Initialize a new KVCache.

        Args:
            layers: Number of transformer layers
            bsz: Batch size
            max_seq_len: Maximum sequence length
            kv_heads: Number of key-value heads
            head_dim: Dimension of each head

        Returns:
            A new KVCache instance with zeroed arrays
        """
        return cls(
            k=jnp.zeros((layers, bsz, max_seq_len, kv_heads, head_dim), dtype=jnp.bfloat16),
            v=jnp.zeros((layers, bsz, max_seq_len, kv_heads, head_dim), dtype=jnp.bfloat16)
        )

    def update(self, xk: jax.Array, xv: jax.Array, layer_idx: int, cur_pos: int, n_rep: int):
        """
        Update the KVCache with new key and value tensors.

        Args:
            xk: New key tensor
            xv: New value tensor
            layer_idx: Index of the current layer
            cur_pos: Current position in the sequence
            n_rep: Number of repetitions for the key-value heads

        Returns:
            Updated keys, values, and a new KVCache instance
        """
        # Update the key cache
        ck = jax.lax.dynamic_update_slice(self.k, jnp.bfloat16(xk[None, ...]), (layer_idx, 0, cur_pos, 0, 0))
        # Update the value cache
        cv = jax.lax.dynamic_update_slice(self.v, jnp.bfloat16(xv[None, ...]), (layer_idx, 0, cur_pos, 0, 0))

        if cur_pos == 0:
            # If it's the first position, repeat the new key and value tensors
            keys = jnp.repeat(xk, n_rep, axis=2)
            values = jnp.repeat(xv, n_rep, axis=2)
        else:
            # Otherwise, repeat the entire cache for the current layer
            keys = jnp.repeat(ck[layer_idx], n_rep, axis=2)
            values = jnp.repeat(cv[layer_idx], n_rep, axis=2)

        return keys, values, KVCache(k=ck, v=cv)