from typing import List, NamedTuple

import torch
import jax
import jax.numpy as jnp
import numpy as np

import ml_dtypes

from pathlib import Path

# Device selection, tree is like first apple silicion, then cuda, fallback is cpu.
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

#print(f"Using device: {device}")

# <thought>
# Once again, we see the device selection logic. This repetition across files
# continues to be a concern and suggests a need for centralized configuration.
# 
# The imports are interesting here. We're seeing a mix of torch, jax, and numpy.
# This suggests some kind of interoperability or conversion between these libraries,
# which could be for benchmarking or compatibility reasons.
# </thought>

class LayerWeights(NamedTuple):
  wq: torch.Tensor
  wk: torch.Tensor
  wv: torch.Tensor
  wo: torch.Tensor
  w1: torch.Tensor
  w2: torch.Tensor
  w3: torch.Tensor
  ffn_norm: torch.Tensor
  attention_norm: torch.Tensor

class XfmrWeights(NamedTuple):
  tok_embeddings: torch.Tensor
  norm: torch.Tensor
  output: torch.Tensor
  layer_weights: List[LayerWeights]

# <thought>
# These NamedTuples define the structure of the weights for a transformer model.
# The use of NamedTuples provides a clean, immutable structure for organizing the weights.
# 
# The LayerWeights class includes weights for attention (wq, wk, wv, wo) and 
# feed-forward layers (w1, w2, w3), as well as layer normalization weights.
# 
# The presence of w3 suggests this might be using a variant of the feed-forward
# layer, possibly SwiGLU or a similar activation.
# </thought>

def compare_outputs(torch_output: torch.Tensor, jax_output: jax.Array, atol: float = 1e-5, rtol: float = 1e-8) -> None:
  jax_output_np = np.array(jax_output)
  torch_output_np = torch_output.cpu().view(dtype=torch.uint16).numpy().view(ml_dtypes.bfloat16)

  try:
    np.testing.assert_allclose(torch_output_np, jax_output_np, atol=atol, rtol=rtol)
  except AssertionError as e:
    print(f'JAX output (first 30): {jax_output_np.flatten()[:30]}')
    print(f'PyTorch output (first 30): {torch_output_np.flatten()[:30]}')
    raise e

# <thought>
# This function is comparing outputs between PyTorch and JAX implementations.
# It's converting the PyTorch tensor to bfloat16 for comparison, which suggests
# that the model is using bfloat16 precision, likely for performance reasons.
# 
# The use of assert_allclose with small tolerance values indicates that we're
# expecting very close agreement between the two implementations. This could be
# part of a verification process to ensure that a PyTorch implementation matches
# a reference JAX implementation.
# </thought>

def load_weights(ckpt_dir: Path = Path('weights/1B-Instruct'), n_layers: int = 16):
  w = {}
  layer_weights = []
  with torch.inference_mode():
    for file in ckpt_dir.glob("*.npy"):
      name = '.'.join(str(file).split('/')[-1].split('.')[:-1])
      jax_weight = jnp.load(file=file, mmap_mode='r', allow_pickle=True)
      #print(f'JAX output (first 30): {jax_weight.flatten()[:30]}')
      np_weight = np.array(jax_weight).astype(np.float32)
      weight = torch.from_numpy(np_weight).to(torch.bfloat16).to(device)
      compare_outputs(torch_output=weight, jax_output=jax_weight)
      w[name] = weight.to(device)
    for i in range(n_layers):
      layer_weights.append(LayerWeights(
        wq=w[f'layers.{i}.attention.wq.weight'],
        wk=w[f'layers.{i}.attention.wk.weight'],
        wv=w[f'layers.{i}.attention.wv.weight'],
        wo=w[f'layers.{i}.attention.wo.weight'],
        w1=w[f'layers.{i}.feed_forward.w1.weight'],
        w2=w[f'layers.{i}.feed_forward.w2.weight'],
        w3=w[f'layers.{i}.feed_forward.w3.weight'],
        ffn_norm=w[f'layers.{i}.ffn_norm.weight'],
        attention_norm=w[f'layers.{i}.attention_norm.weight'],
      ))

    xfmr_weights = XfmrWeights(
      tok_embeddings=w['tok_embeddings.weight'],
      norm=w['norm.weight'],
      output=w['output.weight'],
      layer_weights=layer_weights
    )

    return xfmr_weights

# <thought>
# This load_weights function is doing several interesting things:
#
# 1. It's loading weights from .npy files, which are typically associated with NumPy.
#    This suggests the weights were originally saved in a NumPy-compatible format.
# 
# 2. It's using JAX to load the weights initially (jnp.load), then converting to NumPy,
#    and finally to PyTorch tensors. This multi-step conversion process is unusual
#    and might be for ensuring compatibility or precision.
# 
# 3. The weights are being converted to bfloat16 precision. This is a common choice
#    for large language models as it offers a good balance of precision and memory efficiency.
# 
# 4. There's a comparison being made between the original JAX weights and the
#    converted PyTorch weights. This suggests a strong emphasis on ensuring
#    the conversion process doesn't introduce significant numerical discrepancies.
# 
# 5. The function is constructing a nested structure of weights (XfmrWeights containing
#    multiple LayerWeights) that matches the architecture of the transformer model.
# 
# 6. The use of torch.inference_mode() suggests this function is optimized for
#    inference, disabling gradient computation and certain PyTorch optimizations.
# 
# Potential issues or considerations:
# 1. The function assumes a specific directory structure and naming convention
#    for the weight files. This could be fragile if the weight saving format changes.
# 2. The hard-coded nature of the layer names (e.g., 'layers.{i}.attention.wq.weight')
#    might make it difficult to adapt this function to slightly different architectures.
# 3. Loading and converting each weight tensor individually could be slow for very
#    large models. A more batched approach might be more efficient.
# 4. There's no error handling for missing weight files or unexpected architectures.
# </thought>