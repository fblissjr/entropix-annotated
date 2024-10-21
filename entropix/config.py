from typing import NamedTuple

# Define model parameters
params = {
    "dim": 2048,               # Dimension of the model's hidden states
    "n_layers": 16,            # Number of transformer layers
    "n_heads": 32,             # Number of attention heads
    "n_kv_heads": 8,           # Number of key-value heads (for grouped-query attention)
    "vocab_size": 128256,      # Size of the vocabulary
    "ffn_dim_multiplier": 1.5, # Multiplier for the feed-forward network dimension
    "multiple_of": 256,        # Ensures certain dimensions are multiples of this value
    "norm_eps": 1e-05,         # Epsilon value for layer normalization
    "rope_theta": 500000.0,    # Base value for rotary positional embeddings
    "use_scaled_rope": True,   # Whether to use scaled rotary positional embeddings
    "max_seq_len": 4096        # Maximum sequence length the model can handle
}

# Define a named tuple for easy access to model parameters
class ModelParams(NamedTuple):
    n_layers: int              # Number of transformer layers
    n_local_heads: int         # Number of attention heads
    n_local_kv_heads: int      # Number of key-value heads
    head_dim: int              # Dimension of each attention head
    max_seq_len: int           # Maximum sequence length
    rope_theta: float          # Base value for rotary positional embeddings
    use_scaled_rope: bool      # Whether to use scaled rotary positional embeddings

# Create an instance of ModelParams with the defined parameters
LLAMA_1B_PARAMS = ModelParams(
    n_layers=params["n_layers"],
    n_local_heads=params["n_heads"],
    n_local_kv_heads=params["n_kv_heads"],
    head_dim=params["dim"] // params["n_heads"],  # Calculate head dimension
    max_seq_len=params["max_seq_len"],
    rope_theta=params["rope_theta"],
    use_scaled_rope=params["use_scaled_rope"]
)