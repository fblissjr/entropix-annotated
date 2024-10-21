from typing import List, NamedTuple
import jax
import jax.numpy as jnp
from pathlib import Path

class LayerWeights(NamedTuple):
    # Query weights: shape (hidden_dim, n_heads * head_dim)
    wq: jax.Array
    # Key weights: shape (hidden_dim, n_kv_heads * head_dim)
    wk: jax.Array
    # Value weights: shape (hidden_dim, n_kv_heads * head_dim)
    wv: jax.Array
    # Output projection weights: shape (n_heads * head_dim, hidden_dim)
    wo: jax.Array
    # First feed-forward layer weights: shape (hidden_dim, ffn_dim)
    w1: jax.Array
    # Second feed-forward layer weights: shape (ffn_dim, hidden_dim)
    w2: jax.Array
    # Third feed-forward layer weights (for SwiGLU activation): shape (hidden_dim, ffn_dim)
    w3: jax.Array
    # Feed-forward layer normalization weights: shape (hidden_dim,)
    ffn_norm: jax.Array
    # Attention layer normalization weights: shape (hidden_dim,)
    attention_norm: jax.Array

class XfmrWeights(NamedTuple):
    # Token embedding weights: shape (vocab_size, hidden_dim)
    tok_embeddings: jax.Array
    # Final layer normalization weights: shape (hidden_dim,)
    norm: jax.Array
    # Output projection weights: shape (hidden_dim, vocab_size)
    output: jax.Array
    # List of LayerWeights for each transformer layer
    layer_weights: List[LayerWeights]

def load_weights(ckpt_dir: Path, n_layers: int = 16):
    w = {}
    layer_weights = []
    
    # Attempt to use GPU, fall back to CPU if not available
    try:
        # jax.devices("gpu") returns a list of all available GPUs
        device = jax.devices("gpu")[0]  # Select the first GPU
    except RuntimeError:
        print("GPU not found. Using CPU instead.")
        # jax.devices("cpu") returns a list of all available CPU devices
        device = jax.devices("cpu")[0]  # Select the first CPU device
    
    # Iterate through all .npy files in the checkpoint directory
    for file in ckpt_dir.glob("*.npy"):
        # Extract the weight name from the file name
        name = '.'.join(str(file).split('/')[-1].split('.')[:-1])
        
        # Load the weight array from the .npy file
        # mmap_mode='r' allows reading the file without loading it entirely into memory
        weight = jnp.load(file=file, mmap_mode='r', allow_pickle=True)
        
        # Move the weight to the selected device (GPU or CPU)
        # This creates a copy of the weight on the device, optimizing memory access
        w[name] = jax.device_put(weight, device)
    
    # Organize weights into LayerWeights objects for each layer
    for i in range(n_layers):
        layer_weights.append(LayerWeights(
            # Attention weights
            wq=w[f'layers.{i}.attention.wq.weight'],  # shape: (hidden_dim, n_heads * head_dim)
            wk=w[f'layers.{i}.attention.wk.weight'],  # shape: (hidden_dim, n_kv_heads * head_dim)
            wv=w[f'layers.{i}.attention.wv.weight'],  # shape: (hidden_dim, n_kv_heads * head_dim)
            wo=w[f'layers.{i}.attention.wo.weight'],  # shape: (n_heads * head_dim, hidden_dim)
            
            # Feed-forward network weights
            w1=w[f'layers.{i}.feed_forward.w1.weight'],  # shape: (hidden_dim, ffn_dim)
            w2=w[f'layers.{i}.feed_forward.w2.weight'],  # shape: (ffn_dim, hidden_dim)
            w3=w[f'layers.{i}.feed_forward.w3.weight'],  # shape: (hidden_dim, ffn_dim)
            
            # Layer normalization weights
            ffn_norm=w[f'layers.{i}.ffn_norm.weight'],          # shape: (hidden_dim,)
            attention_norm=w[f'layers.{i}.attention_norm.weight']  # shape: (hidden_dim,)
        ))

    # Create XfmrWeights object containing all model weights
    xfmr_weights = XfmrWeights(
        tok_embeddings=w['tok_embeddings.weight'],  # shape: (vocab_size, hidden_dim)
        norm=w['norm.weight'],                      # shape: (hidden_dim,)
        output=w['output.weight'],                  # shape: (hidden_dim, vocab_size)
        layer_weights=layer_weights                 # list of LayerWeights objects
    )

    return xfmr_weights