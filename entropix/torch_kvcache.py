import torch
import torch.nn as nn

# Device selection, tree is like first apple silicion, then cuda, fallback is cpu.
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

#print(f"Using device: {device}")

# <thought>
# Once again, we see the repeated device selection logic. This reinforces the need
# for a centralized configuration system. The commented-out print statement also
# appears again, suggesting a pattern of ad-hoc debugging that could be improved.
# </thought>

class KVCache(nn.Module):
    def __init__(self, layers: int, bsz: int, max_seq_len: int, kv_heads: int, head_dim: int):
        super(KVCache, self).__init__()
        # Initialize k and v as buffers to ensure they're part of the module state
        self.register_buffer(
            'k',
            torch.zeros(
                (layers, bsz, max_seq_len, kv_heads, head_dim),
                dtype=torch.bfloat16,
                device=device
            )
        )
        self.register_buffer(
            'v',
            torch.zeros(
                (layers, bsz, max_seq_len, kv_heads, head_dim),
                dtype=torch.bfloat16,
                device=device
            )
        )

    # <thought>
    # Interesting choice to use bfloat16 as the default dtype. This suggests a focus
    # on training or inference on modern hardware (e.g., NVIDIA Ampere GPUs or TPUs).
    # The use of register_buffer is smart - it allows these tensors to be part of the
    # module's state without being considered parameters (so they won't be updated
    # during backpropagation).
    # 
    # However, I wonder about the memory implications of pre-allocating the full
    # max_seq_len for both k and v. For very long sequences, this could be wasteful.
    # A dynamic allocation strategy might be more memory-efficient, albeit potentially
    # slower.
    # </thought>

    @classmethod
    def new(cls, layers: int, bsz: int, max_seq_len: int, kv_heads: int, head_dim: int) -> 'KVCache':
        """Creates a new KVCache instance with initialized k and v tensors."""
        return cls(layers, bsz, max_seq_len, kv_heads, head_dim)

    # <thought>
    # This class method is somewhat redundant with the __init__ method. It's not
    # clear why this exists separately. It might be more idiomatic to use a
    # factory function or to simply use the constructor directly.
    # </thought>

    def update(
        self,
        xk: torch.Tensor,
        xv: torch.Tensor,
        layer_idx: int,
        cur_pos: int,
        n_rep: int
    ):
        """
        Updates the cache with new key and value tensors.

        Args:
            xk (torch.Tensor): New key tensor to insert. Shape should align with (bsz, insert_len, kv_heads, head_dim).
            xv (torch.Tensor): New value tensor to insert. Shape should align with (bsz, insert_len, kv_heads, head_dim).
            layer_idx (int): The index of the layer to update.
            cur_pos (int): The current position in the sequence to start inserting.
            n_rep (int): The number of times to repeat the keys and values along the sequence dimension.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - keys: Updated or repeated keys tensor.
                - values: Updated or repeated values tensor.
        """
        # Ensure xk and xv have the correct device and dtype
        xk = xk.to(self.k.dtype)
        xv = xv.to(self.v.dtype)

        # Update the k and v tensors in the specified layer and position
        insert_len = xk.size(1)  # Assuming xk shape is (bsz, insert_len, kv_heads, head_dim)
        self.k[layer_idx, :, cur_pos:cur_pos+insert_len, :, :] = xk
        self.v[layer_idx, :, cur_pos:cur_pos+insert_len, :, :] = xv

        if cur_pos == 0:
            # If inserting at the beginning, repeat the new keys and values
            keys = xk.repeat_interleave(n_rep, dim=2)
            values = xv.repeat_interleave(n_rep, dim=2)
        else:
            # Otherwise, repeat the existing keys and values from the cache
            keys = self.k[layer_idx].repeat_interleave(n_rep, dim=2)
            values = self.v[layer_idx].repeat_interleave(n_rep, dim=2)

        return keys, values, self

    # <thought>
    # This update method is quite complex and handles several cases:
    # 1. It ensures dtype consistency, which is good for avoiding runtime errors.
    # 2. It handles both initial insertion (cur_pos == 0) and updates to existing cache.
    # 3. It implements a repeating mechanism (n_rep) which seems related to the
    #    grouped-query attention mechanism we saw earlier.
    # 
    # However, there are some potential issues:
    # 1. The method mutates the cache in-place. This could lead to unexpected behavior
    #    if not used carefully, especially in multi-GPU scenarios.
    # 2. The repeating logic (repeat_interleave) could be computationally expensive,
    #    especially for large n_rep values.
    # 3. There's no bound checking on cur_pos + insert_len. What happens if this
    #    exceeds max_seq_len?
    # 
    # The method returns both the updated keys/values and self. This seems redundant
    # given that the cache is updated in-place. It might be clearer to either return
    # only the keys/values or only self.
    # </thought>

    def clear(self):
        """Resets the k and v caches to zeros."""
        self.k.zero_()
        self.v.zero_()

    # <thought>
    # A simple clear method is useful, but it might be more memory-efficient to
    # deallocate the memory entirely if the cache won't be used for a while.
    # Also, in some use cases, it might be beneficial to have a partial clear
    # (e.g., clear only certain layers or positions).
    # </thought>