# Import necessary libraries and modules
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import tyro  # Library for creating command-line interfaces

# Import custom modules from the entropix package
from entropix.config import LLAMA_1B_PARAMS
from entropix.kvcache import KVCache
from entropix.model import xfmr
from entropix.sampler import SamplerConfig, sample
from entropix.prompts import create_prompts_from_csv, prompt
from entropix.tokenizer import Tokenizer
from entropix.weights import load_weights

# <thought>
# The imports suggest a modular design, which is good for maintainability.
# However, the use of 'import *' for some modules (e.g., sample from sampler)
# could lead to namespace pollution and make it harder to track where 
# functions are coming from. It might be better to use explicit imports.
# 
# The use of JAX instead of PyTorch or TensorFlow is interesting. This suggests
# a focus on high-performance computing, possibly leveraging JAX's JIT compilation
# and automatic differentiation capabilities.
# </thought>

# Define the default path for the model weights
DEFAULT_WEIGHTS_PATH = Path(__file__).parent / '../weights'

# <thought>
# Using a relative path for weights is good for portability, but it assumes
# a specific directory structure. This could be problematic if the project
# structure changes. Consider making this more configurable, perhaps through
# environment variables or a config file.
# </thought>

# Function to apply scaling to the rotary positional embedding frequencies
def apply_scaling(freqs: jax.Array):
    # Define scaling parameters
    SCALE_FACTOR = 8
    LOW_FREQ_FACTOR = 1
    HIGH_FREQ_FACTOR = 4
    OLD_CONTEXT_LEN = 8192  # original llama3 length

    # Calculate wavelength thresholds
    low_freq_wavelen = OLD_CONTEXT_LEN / LOW_FREQ_FACTOR
    high_freq_wavelen = OLD_CONTEXT_LEN / HIGH_FREQ_FACTOR

    # Define the scaling function for individual frequencies
    def scale_freq(freq):
        wavelen = 2 * math.pi / freq

        # Function to scale frequencies in the middle range
        def scale_mid(_):
            smooth = (OLD_CONTEXT_LEN / wavelen - LOW_FREQ_FACTOR) / (HIGH_FREQ_FACTOR - LOW_FREQ_FACTOR)
            return (1 - smooth) * freq / SCALE_FACTOR + smooth * freq

        # Apply different scaling based on the wavelength
        return jax.lax.cond(
            wavelen < high_freq_wavelen,
            lambda _: freq,
            lambda _: jax.lax.cond(wavelen > low_freq_wavelen, lambda _: freq / SCALE_FACTOR, scale_mid, None),
            None
        )

    # Apply the scaling function to all frequencies
    return jax.vmap(scale_freq)(freqs)

# <thought>
# This scaling function is quite complex and seems to be an implementation of
# RoPE (Rotary Position Embedding) scaling. The use of magic numbers
# (SCALE_FACTOR, LOW_FREQ_FACTOR, etc.) without clear explanation is concerning.
# These values significantly impact the model's behavior and should be well-documented.
#
# The nested conditional logic (jax.lax.cond) makes the function hard to read.
# It might be clearer to use a more explicit if-else structure, even if it's
# slightly less efficient.
#
# The use of vmap suggests an attempt at parallelization, which is good for
# performance but may make debugging more difficult.
# </thought>

# Function to precompute complex exponentials for rotary positional embeddings
def precompute_freqs_cis(dim: int, end: int, theta: float = 500000.0, use_scaled: bool = False, dtype: jnp.dtype = jnp.float32) -> jax.Array:
    # Generate frequency bands
    freqs = 1.0 / (theta ** (jnp.arange(0, dim, 2)[: (dim // 2)].astype(dtype) / dim))
    
    # Apply scaling if requested
    if use_scaled:
        freqs = apply_scaling(freqs)
    
    # Generate time steps
    t = jnp.arange(end, dtype=dtype)
    
    # Compute outer product of time steps and frequencies
    freqs = jnp.outer(t, freqs)
    
    # Return complex exponentials
    return jnp.exp(1j * freqs)

# <thought>
# This function implements the core of RoPE. The default theta value (500000.0)
# is much larger than the typical value (10000.0) used in many transformer models.
# This could significantly affect the model's ability to handle long sequences.
#
# The use of complex exponentials is mathematically elegant but may have
# performance implications. It's worth benchmarking this against a real-valued
# alternative.
#
# The optional scaling adds flexibility but also complexity. It's not clear
# when or why one would choose to use scaled frequencies.
# </thought>

# Function to build an attention mask for the transformer
def build_attn_mask(seqlen: int, start_pos: int) -> jax.Array:
    mask = jnp.zeros((seqlen, seqlen), dtype=jnp.float32)
    if seqlen > 1:
        # Create a lower triangular matrix with -inf values
        mask = jnp.full((seqlen, seqlen), float('-inf'))
        mask = jnp.triu(mask, k=1)
        # Prepend zeros to allow attention to all previous tokens
        mask = jnp.hstack([jnp.zeros((seqlen, start_pos)), mask], dtype=jnp.float32)
    return mask

# <thought>
# This attention mask implementation is standard for causal (left-to-right) attention.
# However, the use of float('-inf') could potentially lead to NaN issues in some edge cases.
# A very large negative number might be more numerically stable.
#
# The 'start_pos' parameter allows for flexible masking, which is useful for
# tasks like continued text generation. However, it's not clear how this interacts
# with the KV-cache implementation.
# </thought>

# Main function to set up the model and generate text
def main(weights_path: Path = DEFAULT_WEIGHTS_PATH.joinpath('1B-Instruct')):
    # Load model parameters and weights
    model_params = LLAMA_1B_PARAMS
    xfmr_weights = load_weights(weights_path.absolute())
    
    # Initialize the tokenizer
    tokenizer = Tokenizer('entropix/tokenizer.model')

    # Define the text generation function
    def generate(xfmr_weights, model_params, tokens):
        gen_tokens = None
        cur_pos = 0
        # Convert input tokens to a JAX array
        tokens = jnp.array([tokens], jnp.int32)
        bsz, seqlen = tokens.shape
        
        # Build the attention mask
        attn_mask = build_attn_mask(seqlen, cur_pos)
        
        # Precompute positional embedding frequencies
        freqs_cis = precompute_freqs_cis(model_params.head_dim, model_params.max_seq_len, model_params.rope_theta, model_params.use_scaled_rope)
        
        # Initialize the key-value cache
        kvcache = KVCache.new(model_params.n_layers, bsz, max_seq_len, model_params.n_local_kv_heads, model_params.head_dim)
        
        # Generate the initial logits
        logits, kvcache, _, _ = xfmr(xfmr_weights, model_params, tokens, cur_pos, freqs_cis[:seqlen], kvcache, attn_mask=attn_mask)
        
        # Select the next token
        next_token = jnp.argmax(logits[:, -1], axis=-1, keepdims=True).astype(jnp.int32)
        gen_tokens = next_token
        
        # Print the first generated token
        print(tokenizer.decode([next_token.item()]), end='', flush=True)
        
        # Update the current position
        cur_pos = seqlen
        
        # Define stop tokens
        stop = jnp.array([128001, 128008, 128009])
        
        # Initialize the sampler configuration
        sampler_cfg = SamplerConfig()
        
        # Main generation loop
        while cur_pos < 8192:
            cur_pos += 1
            # Generate next token logits
            logits, kvcache, scores, stats = xfmr(xfmr_weights, model_params, next_token, cur_pos, freqs_cis[cur_pos:cur_pos+1], kvcache)
            
            # Sample the next token
            next_token = sample(gen_tokens, logits, scores, cfg=sampler_cfg)
            
            # Append the new token to generated tokens
            gen_tokens = jnp.concatenate((gen_tokens, next_token))
            
            # Print the generated token
            print(tokenizer.decode(next_token.tolist()[0]), end='', flush=True)
            
            # Check if a stop token has been generated
            if jnp.isin(next_token, stop).any():
                break

    # <thought>
    # The generate function is complex and handles both the initial forward pass
    # and the token-by-token generation. This dual responsibility makes the function
    # harder to understand and maintain.
    #
    # The use of a fixed maximum sequence length (8192) could be limiting. It might
    # be better to make this configurable or dynamically determined.
    #
    # The printing of tokens as they're generated could be problematic for large
    # batches or when using this function as part of a larger system. Consider
    # making this optional or separating the generation logic from the output handling.
    #
    # The stop token logic is hardcoded. It would be more flexible to allow
    # configurable stop conditions.
    #
    # The function doesn't handle any error conditions (e.g., running out of memory,
    # unexpected model outputs). Adding proper error handling would make it more robust.
    # </thought>

    # Load prompts from a CSV file (commented out in this version)
    csv_path = Path('entropix/data/prompts2.csv')
    prompts = create_prompts_from_csv(csv_path)
    PROMPT_TEST = False

    # Choose between testing multiple prompts or using a single prompt
    if PROMPT_TEST:
        for p in prompts:
            print(p)
            tokens = tokenizer.encode(p,  bos=False, eos=False, allowed_special='all')
            generate(xfmr_weights, model_params, tokens)
    else:
        print(prompt)
        tokens = tokenizer.encode(prompt,  bos=False, eos=False, allowed_special='all')
        generate(xfmr_weights, model_params, tokens)

# <thought>
# The prompt handling logic seems to be set up for testing different prompts,
# but it's currently disabled (PROMPT_TEST = False). This suggests that the
# code might be in a development or debugging state.
#
# The tokenization parameters (bos=False, eos=False, allowed_special='all')
# are hardcoded. These could significantly affect the model's behavior and
# might need to be adjusted for different use cases.
#
# There's no error handling around the file operations (loading prompts from CSV).
# This could lead to crashes if the file is missing or malformed.
# </thought>

# Run the main function if this script is executed directly
if __name__ == '__main__':
    tyro.cli(main)

# <thought>
# The use of tyro for CLI handling is interesting. It's less common than
# argparse or click, which might make it harder for new contributors to
# understand the code. However, it does provide strong typing for CLI arguments,
# which can help prevent runtime errors.
#
# There's no apparent use of command-line arguments in the main function,
# which makes the use of tyro.cli seem unnecessary in this context.
# </thought>