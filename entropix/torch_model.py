import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from entropix.config import ModelParams
from entropix.torch_kvcache import KVCache
from entropix.torch_weights import XfmrWeights, LayerWeights
from entropix.torch_stats import AttnStats

DEFAULT_MASK_VALUE = -0.7 * float(torch.finfo(torch.float32).max)

# Device selection, tree is like first apple silicion, then cuda, fallback is cpu.
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

#print(f"Using device: {device}")

from typing import Tuple, Optional

# <thought>
# The device selection logic appears again here. This repetition across files
# suggests a lack of centralized configuration. It might be better to have a
# single config file that handles device selection for the entire project.
# 
# The commented out print statement is curious. Was this left for debugging?
# It's generally better to use proper logging mechanisms for such information.
# </thought>

def rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
  return w * (x * torch.rsqrt(torch.pow(x, 2).mean(-1, keepdim=True) + eps))

# <thought>
# This is an implementation of Root Mean Square (RMS) Normalization.
# Interesting choice over LayerNorm. RMSNorm is computationally cheaper
# and has been shown to work well in large language models.
# 
# The use of torch.rsqrt for the reciprocal square root is a good optimization.
# However, I wonder about the numerical stability here. The epsilon is quite small.
# In some cases, a larger epsilon (e.g., 1e-5) might be more stable.
# </thought>

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor, dtype: torch.dtype = torch.float32) -> Tuple[torch.Tensor, torch.Tensor]:
    reshape_xq = xq.float().reshape(*xq.shape[:-1], -1, 2)
    reshape_xk = xk.float().reshape(*xk.shape[:-1], -1, 2)
    xq_ = torch.complex(reshape_xq[..., 0], reshape_xq[..., 1])
    xk_ = torch.complex(reshape_xk[..., 0], reshape_xk[..., 1])
    xq_out = xq_ * freqs_cis.unsqueeze(0).unsqueeze(2)
    xk_out = xk_ * freqs_cis.unsqueeze(0).unsqueeze(2)
    xq_out = torch.stack((xq_out.real, xq_out.imag), dim=-1).reshape(*xq_out.shape[:-1], -1)
    xk_out = torch.stack((xk_out.real, xk_out.imag), dim=-1).reshape(*xk_out.shape[:-1], -1)
    return xq_out.to(dtype), xk_out.to(dtype)

# <thought>
# This is an implementation of Rotary Position Embeddings (RoPE).
# The use of complex numbers for rotation is clever and efficient.
# 
# However, there are a few points to consider:
# 1. The constant conversion to float() could be expensive. Could we ensure
#    the input is always in the correct dtype to avoid this?
# 2. The reshaping operations are numerous. Could this be optimized?
# 3. The function assumes the last dimension of xq and xk is even. What happens
#    if it's not? Should we add an assertion or handle this case?
# 
# The use of unsqueeze operations for broadcasting is good practice.
# </thought>

def attention(x: torch.Tensor, layer_weights: LayerWeights, model_params, cur_pos: int, layer_idx: int, freqs_cis: torch.Tensor, kvcache: KVCache, attn_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, KVCache, torch.Tensor]:
    bsz, _, _ = x.shape
    n_rep = model_params.n_local_heads // model_params.n_local_kv_heads
    xq = F.linear(x, layer_weights.wq).reshape(bsz, -1, model_params.n_local_heads, model_params.head_dim)
    xk = F.linear(x, layer_weights.wk).reshape(bsz, -1, model_params.n_local_kv_heads, model_params.head_dim)
    xv = F.linear(x, layer_weights.wv).reshape(bsz, -1, model_params.n_local_kv_heads, model_params.head_dim)
    xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis, dtype=xq.dtype) # add dtype
    keys, values, kvcache = kvcache.update(xk, xv, layer_idx, cur_pos, n_rep)
    xq = torch.permute(xq, (0, 2, 1, 3))  # (bs, n_heads, seqlen, head_dim)
    keys = torch.permute(keys, (0, 2, 3, 1))  # (bs, n_heads, head_dim, cache_len + seqlen)
    values = torch.permute(values, (0, 2, 1, 3))  # (bs, n_heads, cache_len + seqlen, head_dim)
    scores = torch.matmul(xq, keys)
    pre_scores = scores / math.sqrt(model_params.head_dim)
    scores = pre_scores.to(torch.float32)  # Always do attention softmax at float32
    if cur_pos == 0:
        scores = scores + attn_mask
    mask = torch.where(scores != 0.0, scores, DEFAULT_MASK_VALUE)
    padded_logits = torch.where((mask >= DEFAULT_MASK_VALUE * 0.5), scores, DEFAULT_MASK_VALUE)
    scores = F.softmax(padded_logits, dim=-1).to(torch.float32)
    output = torch.matmul(scores.to(values.dtype), values) # add dtype
    output = output.transpose(1, 2).reshape(xq.shape[0], xq.shape[2], -1)
    out = F.linear(output, layer_weights.wo)
    return out, kvcache, pre_scores

# <thought>
# This attention implementation is quite complex and includes several optimizations:
# 1. It uses grouped-query attention (n_local_heads vs n_local_kv_heads)
# 2. It incorporates a key-value cache for efficient autoregressive generation
# 3. It applies rotary position embeddings
# 
# Some points to consider:
# 1. The use of F.linear instead of direct matrix multiplication. This is good for
#    flexibility (e.g., using different dtypes), but might be slightly slower.
# 2. The attention mask is only applied when cur_pos == 0. Is this always correct?
#    What about continuing generation from a non-zero position?
# 3. The use of DEFAULT_MASK_VALUE and the subsequent masking logic is interesting.
#    It's trying to avoid -inf values, but the logic is a bit complex. Could this
#    be simplified?
# 4. The final linear projection (wo) is outside the main attention computation.
#    This is good for modularity, but might miss some optimization opportunities.
# 
# The function returns pre_scores, which seems to be unused. Is this for debugging?
# </thought>

def feed_forward(x: torch.Tensor, layer_weights: LayerWeights) -> torch.Tensor:
 return F.linear(F.silu(F.linear(x, layer_weights.w1)) * F.linear(x, layer_weights.w3), layer_weights.w2)

# <thought>
# This is an implementation of the SwiGLU activation function, which has been
# shown to work well in large language models.
# 
# The use of F.linear again provides flexibility but might have a small
# performance cost compared to direct matrix multiplication.
# 
# The multiplication of two linear projections (w1 and w3) before the final
# projection (w2) is interesting. This increases the model's capacity without
# significantly increasing computation. Clever!
# </thought>

def xfmr(xfmr_weights: XfmrWeights, model_params: ModelParams, tokens: torch.Tensor, cur_pos: int, freqs_cis: torch.Tensor, kvcache: KVCache, attn_mask: Optional[torch.Tensor]=None) -> Tuple[torch.Tensor, KVCache, torch.Tensor, AttnStats]:
    h = xfmr_weights.tok_embeddings[tokens]
    attn_stats = AttnStats.new(
        bsz=tokens.shape[0],
        n_layers=model_params.n_layers,
        n_heads=model_params.n_local_heads
    )
    for i in range(model_params.n_layers):
        norm_x = rms_norm(h, xfmr_weights.layer_weights[i].attention_norm)
        h_attn, kvcache, scores = attention(norm_x, xfmr_weights.layer_weights[i], model_params, cur_pos, i, freqs_cis, kvcache, attn_mask=attn_mask)
        attn_stats = attn_stats.update(scores[:,:,-1,:], i)
        h = h + h_attn
        h = h + feed_forward(rms_norm(h, xfmr_weights.layer_weights[i].ffn_norm), xfmr_weights.layer_weights[i])
    logits = F.linear(rms_norm(h, xfmr_weights.norm), xfmr_weights.output)
    return logits, kvcache, scores, attn_stats

# <thought>
# This is the main transformer function. It's a pretty standard implementation
# with a few notable points:
# 
# 1. It uses RMSNorm instead of LayerNorm, which is becoming more common in
#    large language models.
# 2. It collects attention statistics, which could be useful for analysis or
#    visualization, but might add some overhead.
# 3. The use of residual connections (h = h + ...) is standard and important
#    for training very deep networks.
# 
# Questions/Observations:
# 1. Why is 'scores' returned? It's not clear how this is used downstream.
# 2. The function takes 'tokens' as input, not embeddings. This means the
#    embedding layer is part of this function. Is this the best design?
# 3. The kvcache is updated in-place. This is efficient but could lead to
#    subtle bugs if not handled carefully in multi-GPU scenarios.
# 4. There's no dropout here. Is this intentional? Many transformer
#    implementations include dropout for regularization.
# 
# Overall, this seems to be an efficient implementation optimized for
# inference rather than training.
# </thought>