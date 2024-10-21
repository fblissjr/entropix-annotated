# Weight Transformations in Entropix

## Overview

Entropix performs several transformations on the weights of LLaMA models (1B and 1B-Instruct) to optimize them for use in its own architecture. This document details these transformations, explaining what happens to the original weights, why these transformations are necessary, and how they affect the model's structure.

## Original LLaMA Weights vs. Entropix Weights

### High-Level Changes

1. **File Format**: Original LLaMA weights are typically stored in PyTorch's `.pth` or `.pt` format. Entropix converts these to NumPy's `.npy` format for each layer and component.

2. **Weight Naming**: While the overall structure remains similar, some weight names are changed to better reflect their function in the Entropix architecture.

3. **Precision**: Entropix typically uses bfloat16 precision for weights, which balances numerical stability and memory efficiency.

4. **Attention Mechanism**: Entropix may implement optimizations for attention computation, potentially affecting how the weights are structured and used.

### Detailed Transformations

#### 1. Attention Weights

- **Original**: `self_attn.{q,k,v}_proj`
- **Entropix**: `attention.w{q,k,v}`

The query, key, and value projection weights are renamed and possibly reshaped to accommodate Entropix's attention implementation.

#### 2. Feed-Forward Weights

- **Original**: `mlp.{gate,up,down}_proj`
- **Entropix**: `feed_forward.w{1,2,3}`

The feed-forward network weights are renamed. The `w3` weight likely corresponds to the SwiGLU activation used in LLaMA.

#### 3. Layer Normalization

- **Original**: `input_layernorm`, `post_attention_layernorm`
- **Entropix**: `attention_norm`, `ffn_norm`

Layer normalization weights are renamed to reflect their position in the Entropix architecture.

#### 4. Output Projection

- **Original**: `lm_head`
- **Entropix**: `output`

The final output projection is renamed.

### Why These Transformations?

1. **Optimization**: Some transformations may be done to optimize the model for Entropix's specific implementation, potentially improving inference speed or memory usage.

2. **Architectural Differences**: Entropix might have slight architectural differences from the original LLaMA, necessitating weight transformations.

3. **Precision**: Converting to bfloat16 can save memory while maintaining good numerical properties.

4. **Ease of Use**: The `.npy` format and renamed weights might be easier to work with in Entropix's codebase.

## Tokenizer Changes

The tokenizer is typically not transformed significantly. Entropix likely uses the original LLaMA tokenizer, possibly with some additional special tokens. The tokenizer model file is usually kept separate from the weights.

## Inspecting the Weights

Here's some Python code to inspect the `.npy` files:

```python
import numpy as np
import os

def inspect_weights(directory):
    for filename in os.listdir(directory):
        if filename.endswith('.npy'):
            path = os.path.join(directory, filename)
            arr = np.load(path)
            print(f"File: {filename}")
            print(f"Shape: {arr.shape}")
            print(f"Dtype: {arr.dtype}")
            print(f"Min: {arr.min()}, Max: {arr.max()}, Mean: {arr.mean()}")
            print("---")

# Inspect Base LLaMA weights
print("Base LLaMA Weights:")
inspect_weights('./1B-Base')

# Inspect Instruct LLaMA weights
print("\nInstruct LLaMA Weights:")
inspect_weights('./1B-Instruct')
