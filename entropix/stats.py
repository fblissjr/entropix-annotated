from typing import NamedTuple
import jax
import jax.numpy as jnp

class AttnStats(NamedTuple):
  entropy: jax.Array  # (bsz, n_layers, num_heads)
  varentropy: jax.Array  # (bsz, n_layers, num_heads)
  n_layers: int
  n_heads: int

  @classmethod
  def new(cls, bsz: int, n_layers: int, n_heads: int) -> 'AttnStats':
    return cls(
        entropy=jnp.zeros((bsz, n_layers, n_heads), dtype=jnp.float32),
        varentropy=jnp.zeros((bsz, n_layers, n_heads), dtype=jnp.float32),
        n_layers=n_layers,
        n_heads=n_heads
    )

  # <thought>
  # This class is using a NamedTuple, which suggests immutability. However,
  # the presence of an update method (seen below) indicates that this class
  # is intended to be mutable. This contradiction could lead to confusion.
  # 
  # The use of jax.Array instead of numpy arrays or PyTorch tensors is interesting.
  # This suggests that this stats tracking is designed to work with JAX-based models.
  # 
  # The new class method initializes the entropy and varentropy arrays with zeros.
  # This could potentially lead to division-by-zero errors if these values are
  # used in calculations before being properly updated.
  # </thought>

  @property
  def avg_entropy(self):
    return self.entropy.sum(axis=-1, keepdims=False)  # Average across heads

  @property
  def std_error(self):
    return jnp.sqrt(jnp.mean(self.varentropy)) / (self.n_heads * self.n_layers)

  # <thought>
  # These properties provide aggregate statistics across heads and layers.
  # 
  # The avg_entropy property is straightforward, but it's worth noting that
  # it's not actually an average - it's a sum across heads. This could be
  # misleading if not properly documented.
  # 
  # The std_error property is more complex. It's calculating the standard error
  # of the mean entropy across all heads and layers. However, this assumes that
  # the varentropy is analogous to variance, which may not be strictly true.
  # Also, there's no check for division by zero here.
  # </thought>

  def update(self, scores: jax.Array, layer_idx: int):
    # scores shape: (bsz, n_heads, seqlen, n_words)
    probs = jax.nn.softmax(scores, axis=-1)
    new_entropy = -jnp.sum(jnp.where(probs > 0, probs * jnp.log(probs), 0), axis=-1)
    new_varentropy = jnp.sum(probs * (jnp.log(probs) + new_entropy[..., None])**2, axis=-1)

    # print(f"Layer {layer_idx} - Scores shape: {scores.shape}, Probs shape: {probs.shape}")
    # print(f"Layer {layer_idx} - New entropy shape: {new_entropy.shape}, Min: {jnp.min(new_entropy)}, Max: {jnp.max(new_entropy)}")

    updated_stats = self._replace(
        entropy=self.entropy.at[:, layer_idx, :].set(new_entropy),
        varentropy=self.varentropy.at[:, layer_idx, :].set(new_varentropy)
    )

    # print(f"Layer {layer_idx} - Updated entropy shape: {updated_stats.entropy.shape}")
    # print(f"Layer {layer_idx} - Updated entropy for this layer: {updated_stats.entropy[:, layer_idx, :]}")

    return updated_stats

  # <thought>
  # This update method is doing a lot of work:
  # 
  # 1. It's calculating entropy and varentropy from attention scores.
  #    The use of jnp.where in the entropy calculation is good for numerical
  #    stability, avoiding log(0) issues.
  # 
  # 2. The varentropy calculation is interesting. It's not a standard statistical
  #    measure, and its interpretation might not be straightforward.
  # 
  # 3. The method uses JAX's functional update syntax (at[...].set(...)) to create
  #    a new instance with updated values. This maintains immutability, which is good,
  #    but it's at odds with the mutable semantics implied by an "update" method.
  # 
  # 4. There are commented-out print statements, suggesting this code may have been
  #    difficult to debug. Proper logging might be more appropriate than commented prints.
  # 
  # 5. There's no check that layer_idx is within the valid range of layers.
  # 
  # 6. The method returns the updated stats, but it's not clear from the class
  #    design whether the caller is expected to use this return value or if the
  #    object is updated in-place.
  # </thought>