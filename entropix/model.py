from typing import Optional, Tuple

import jax
import jax.numpy as jnp

from functools import partial

from entropix.config import ModelParams
from entropix.kvcache import KVCache
from entropix.stats import AttnStats
from entropix.weights import XfmrWeights, LayerWeights

# Default mask value for attention
DEFAULT_MASK_VALUE = -0.7 * float(jnp.finfo(jnp.dtype("float32")).max)

def rms_norm(x: jax.Array, w: jax.Array, eps: float = 1e-6) -> jax.Array:
    """
    Applies Root Mean Square (RMS) normalization to the input.
    
    Args:
        x: Input tensor
        w: Weight tensor
        eps: Small epsilon value for numerical stability
    
    Returns:
        Normalized tensor
    """
    return w * (x * jax.lax.rsqrt(jax.lax.pow(x, 2).mean(-1, keepdims=True) + eps))

def apply_rotary_emb(xq: jax.Array, xk: jax.Array, freqs_cis: jax.Array, dtype: jnp.dtype = jnp.float32) -> Tuple[jax.Array, jax.Array]:
    """
    Applies rotary positional embeddings to the query and key tensors.
    
    Args:
        xq: Query tensor of shape (batch_size, seq_len, n_heads, head_dim)
        xk: Key tensor of shape (batch_size, seq_len, n_heads, head_dim)
        freqs_cis: Pre-computed complex exponentials for rotary embeddings
        dtype: Data type for the output tensors
    
    Returns:
        Tuple of transformed query and key tensors
    """
    # Reshape query and key tensors to prepare for rotation
    # We split the last dimension (head_dim) into two halves
    # This is done because we'll treat each pair of values as a complex number
    # Shape becomes: (..., head_dim//2, 2)
    reshape_xq = xq.astype(jnp.float32).reshape(*xq.shape[:-1], -1, 2)
    reshape_xk = xk.astype(jnp.float32).reshape(*xk.shape[:-1], -1, 2)
    
    # Convert the reshaped tensors into complex numbers
    # The first value of each pair becomes the real part, the second the imaginary part
    # This effectively represents each vector as a complex number
    xq_ = jax.lax.complex(reshape_xq[..., 0], reshape_xq[..., 1])
    xk_ = jax.lax.complex(reshape_xk[..., 0], reshape_xk[..., 1])
    
    # Apply the rotation by multiplying with the pre-computed complex exponentials
    # freqs_cis contains the rotation factors for each position
    # The None and : are used to correctly broadcast the multiplication
    xq_out = xq_ * freqs_cis[None, :, None, :]
    xk_out = xk_ * freqs_cis[None, :, None, :]
    
    # Convert the rotated complex numbers back to real-valued tensors
    # We stack the real and imaginary parts along a new last axis
    # Then we reshape to flatten the last two dimensions back into head_dim
    xq_out = jnp.stack((jnp.real(xq_out), jnp.imag(xq_out)), axis=-1).reshape(*xq_out.shape[:-1], -1)
    xk_out = jnp.stack((jnp.real(xk_out), jnp.imag(xk_out)), axis=-1).reshape(*xk_out.shape[:-1], -1)
    
    # Convert back to the original dtype and return
    return xq_out.astype(dtype), xk_out.astype(dtype)

def attention(x: jax.Array, layer_weights: LayerWeights, model_params, cur_pos: int, layer_idx: int, freqs_cis: jax.Array, kvcache: KVCache, attn_mask: Optional[jax.Array] = None) -> Tuple[jax.Array, KVCache]:
    """
    Computes multi-head attention for a transformer layer.
    
    Args:
        x: Input tensor
        layer_weights: Weights for the current layer
        model_params: Model parameters
        cur_pos: Current position in the sequence
        layer_idx: Index of the current layer
        freqs_cis: Pre-computed complex exponentials for rotary embeddings
        kvcache: Key-value cache
        attn_mask: Optional attention mask
    
    Returns:
        Tuple of output tensor and updated KVCache
    """
    bsz, _, _ = x.shape
    n_rep = model_params.n_local_heads // model_params.n_local_kv_heads
    
    # Compute query, key, and value tensors
    xq = jnp.dot(x, layer_weights.wq.T).reshape(bsz, -1, model_params.n_local_heads, model_params.head_dim)
    xk = jnp.dot(x, layer_weights.wk.T).reshape(bsz, -1, model_params.n_local_kv_heads, model_params.head_dim)
    xv = jnp.dot(x, layer_weights.wv.T).reshape(bsz, -1, model_params.n_local_kv_heads, model_params.head_dim)
    
    # Apply rotary embeddings
    xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
    
    # Update key-value cache
    keys, values, kvcache = kvcache.update(xk, xv, layer_idx, cur_pos, n_rep)
    
    # Reshape tensors for attention computation
    xq = jnp.transpose(xq, (0, 2, 1, 3))  # (bs, n_heads, seqlen, head_dim)
    keys = jnp.transpose(keys, (0, 2, 3, 1))  # (bs, n_heads, head_dim, cache_len + seqlen)
    values = jnp.transpose(values, (0, 2, 1, 3))  # (bs, n_heads, cache_len + seqlen, head_dim)
    
    # Compute attention scores
    scores = jnp.matmul(xq, keys)
    pre_scores = scores / jnp.sqrt(model_params.head_dim)
    scores = pre_scores.astype(jnp.float32)  # Always do attention softmax at float32
    
    # Apply attention mask if provided
    if cur_pos == 0:
        scores = scores + attn_mask
    
    # Apply softmax to get attention weights
    mask = jnp.where(scores != 0.0, scores, DEFAULT_MASK_VALUE)
    padded_logits = jnp.where((mask >= DEFAULT_MASK_VALUE * 0.5), scores, DEFAULT_MASK_VALUE)
    scores = jax.nn.softmax(padded_logits, axis=-1).astype(x.dtype)
    
    # Compute weighted sum of values
    output = jnp.matmul(scores, values)
    output = jnp.swapaxes(output, 1, 2).reshape(xq.shape[0], xq.shape[2], -1)
    
    # Apply output projection
    out = jnp.dot(output, layer_weights.wo.T)
    
    return out, kvcache, pre_scores

def feed_forward(x: jax.Array, layer_weights: LayerWeights) -> jax.Array:
    """
    Applies the feed-forward network of a transformer layer.
    
    Args:
        x: Input tensor
        layer_weights: Weights for the current layer
    
    Returns:
        Output tensor after applying the feed-forward network
    """
    return jnp.dot(jax.nn.silu(jnp.dot(x, layer_weights.w1.T)) * jnp.dot(x, layer_weights.w3.T), layer_weights.w2.T)

def xfmr(xfmr_weights: XfmrWeights, model_params: ModelParams, tokens: jax.Array, cur_pos: int, freqs_cis: jax.Array, kvcache: KVCache, attn_mask: Optional[jax.Array]=None) -> Tuple[jax.Array, KVCache]:
    """
    Applies the entire transformer model to the input tokens.
    
    Args:
        xfmr_weights: Weights for the entire transformer model
        model_params: Model parameters
        tokens: Input token ids
        cur_pos: Current position in the sequence
        freqs_cis: Pre-computed complex exponentials for rotary embeddings
        kvcache: Key-value cache
        attn_mask: Optional attention mask
    
    Returns:
        Tuple of logits, updated KVCache, attention scores, and attention statistics
    """
    # Embed input tokens
    h = xfmr_weights.tok_embeddings[tokens]
    
    # Initialize attention statistics
    attn_stats = AttnStats.new(
        bsz=tokens.shape[0],
        n_layers=model_params.n_layers,
        n_heads=model_params.n_local_heads
    )
    
    # Apply transformer layers
    for i in range(model_params.n_layers):
        # Apply layer normalization
        norm_x = rms_norm(h, xfmr_weights.layer_weights[i].attention_norm)
        
        # Apply attention
        h_attn, kvcache, scores = attention(norm_x, xfmr_weights.layer_weights[i], model_params, cur_pos, i, freqs_cis, kvcache, attn_mask=attn_mask)
        
        # Update attention statistics
        attn_stats = attn_stats.update(scores[:,:,-1,:], i)
        
        # Apply residual connection
        h = h + h_attn
        
        # Apply feed-forward network with another residual connection
        h = h + feed_forward(rms_norm(h, xfmr_weights.layer_weights[i].ffn_norm), xfmr_weights.layer_weights[i])
    
    # Apply final layer normalization and compute logits
    logits = jnp.dot(rms_norm(h, xfmr_weights.norm), xfmr_weights.output.T)
    
    return logits, kvcache, scores, attn_stats