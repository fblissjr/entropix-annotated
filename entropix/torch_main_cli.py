from typing import NamedTuple, Optional, Tuple

import torch
import torch.nn.functional as F

import math
import tyro

from pathlib import Path
from functools import partial

from entropix.config import LLAMA_1B_PARAMS, params
from entropix.tokenizer import Tokenizer
from entropix.torch_kvcache import KVCache
from entropix.torch_model import xfmr
from entropix.torch_weights import XfmrWeights, LayerWeights, load_weights
from entropix.torch_sampler import sample
from entropix.prompts import prompt, bp1, create_prompts_from_csv

# Added for the CliConfig below
from dataclasses import dataclass, field
import csv

# <thought>
# Importing from config.py and prompts.py allows for better organization and modularity.
# However, we might want to consider creating a unified configuration system that
# combines CLI arguments with the config file for more flexibility.
# </thought>
# <thought>
# Using tyro instead of argparse aligns with the project's existing practices.
# This change should make the CLI more consistent with other parts of the project.
# </thought>

@dataclass
class CliConfig:
    """Configuration for Entropix text generation CLI."""
    prompt_file: Optional[Path] = None
    """Path to a CSV file containing prompts."""
    
    weights_path: Path = Path("weights/1B-Instruct")
    """Path to model weights."""
    
    tokenizer_path: str = "entropix/tokenizer.model"
    """Path to tokenizer model."""
    
    max_length: int = params["max_seq_len"]
    """Maximum length of generated text."""
    
    device: str = field(default_factory=lambda: "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    """Device to use for computation."""
    
    prompt_test: bool = False
    """Whether to test multiple prompts from the CSV file."""
    
# <thought>
# Using a dataclass for CLI configuration allows for easy extension and modification.
# The default device selection logic is now part of the config, making it more visible and configurable.
# </thought>

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Device selection, tree is like first apple silicion, then cuda, fallback is cpu.
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

torch.set_float32_matmul_precision('high')
# <thought>
# The device selection logic seems redundant. We're defining DEVICE and then immediately
# redefining 'device'. This could lead to confusion. Why not consolidate this?
# Also, why is Apple Silicon (mps) prioritized over CUDA? This seems unusual.
# </thought>

def apply_scaling(freqs: torch.Tensor) -> torch.Tensor:
    SCALE_FACTOR = 8.0
    LOW_FREQ_FACTOR = 1.0
    HIGH_FREQ_FACTOR = 4.0
    OLD_CONTEXT_LEN = 8192  # original llama3 length

    low_freq_wavelen = OLD_CONTEXT_LEN / LOW_FREQ_FACTOR
    high_freq_wavelen = OLD_CONTEXT_LEN / HIGH_FREQ_FACTOR

    def scale_freq(freq: torch.Tensor) -> torch.Tensor:
        wavelen = 2 * torch.pi / freq

        # Calculate smooth factor
        smooth = (OLD_CONTEXT_LEN / wavelen - LOW_FREQ_FACTOR) / (HIGH_FREQ_FACTOR - LOW_FREQ_FACTOR)
        smooth = torch.clamp(smooth, 0.0, 1.0)  # Ensure smooth is between 0 and 1

        # Calculate scaled frequency
        scaled = (1 - smooth) * freq / SCALE_FACTOR + smooth * freq

        # Apply conditional scaling
        scaled = torch.where(
            wavelen < high_freq_wavelen,
            freq,  # No scaling
            torch.where(
                wavelen > low_freq_wavelen,
                freq / SCALE_FACTOR,  # Apply scaling factor
                scaled  # Apply smooth scaling
            )
        )
        return scaled

    scaled_freqs = torch.vmap(scale_freq)(freqs)
    
    return scaled_freqs

    # <thought>
    # This function implements RoPE scaling. Consider adding references to relevant papers
    # or explanations for the magic numbers used (SCALE_FACTOR, LOW_FREQ_FACTOR, etc.).
    # In the future, we might want to make the scaling factors configurable via the CliConfig for easier experimentation.
    # </thought>

# <thought>
# This scaling function seems to be an implementation of RoPE (Rotary Position Embedding) scaling.
# The use of wavelengths and smooth interpolation suggests an attempt to extend the context length.
# However, I'm curious about the choice of constants (8.0, 1.0, 4.0, 8192). Are these empirically derived?
# It would be helpful to have some references or explanations for these specific values.
# Also, the use of torch.vmap is interesting - it's not commonly used. Is this for performance reasons?
# </thought>

def precompute_freqs_cis(dim: int, end: int, theta: float = 500000.0, use_scaled: bool = False, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=dtype, device=device)[: (dim // 2)] / dim))
    if use_scaled:
        freqs = apply_scaling(freqs)

    t = torch.arange(end, dtype=dtype, device=device).unsqueeze(1)  # Shape: (end, 1)
    freqs = freqs.unsqueeze(0)  # Shape: (1, dim//2)
    freqs = t * freqs  # Broadcasting to shape: (end, dim//2)
    return torch.exp(1j * freqs)

# <thought>
# This is a standard implementation of RoPE frequency computation. The use of complex exponentials
# is typical for RoPE. However, I'm curious about the 'use_scaled' flag. It seems this implementation
# allows for both scaled and unscaled RoPE. What's the reasoning behind this flexibility?
# Also, the choice of theta=500000.0 is interesting. This is different from the original paper's
# value of 10000.0. What's the justification for this change?
# </thought>

# <thought>
# Using values from the config file increases flexibility. Consider adding these
# parameters to the CliConfig if frequent adjustments are needed.
# </thought>

#########################
# Code Comparison of build_attn_mask options

# - Key Differences
# Return Type: The return type of the function has changed from torch.Tensor to Optional[torch.Tensor]. This indicates that the function can now return None if certain conditions are met.
# Early Return: In the updated code, if seqlen is less than or equal to 1, the function immediately returns None. This is a more efficient way to handle this edge case.
# Device Specification: The device parameter is now specified when creating the tensor with torch.full and torch.zeros. This ensures that the tensor is created on the correct device (e.g., GPU or CPU).
# Type Conversion: The to(torch.float32) conversion is now applied after the torch.hstack operation. This doesn't change the functionality but makes the code slightly more efficient.

# - Minor Differences
# Variable Initialization: The mask variable is no longer initialized with None before the if statement.
# Code Organization: The updated code is slightly more concise and easier to read.

# -Functionality
# Both code snippets appear to be building an attention mask for a sequence of length seqlen, starting from position start_pos. The mask is used to prevent the model from attending to certain positions in the sequence. The updated code is slightly more efficient and easier to understand.
# Best Practices

# - The updated code follows best practices by:
# Using type hints for function parameters and return types
# Handling edge cases efficiently
# Specifying the device for tensor creation
# Keeping the code concise and readable
################################################

#1: og entropix code ---
def build_attn_mask(seqlen: int, start_pos: int) -> torch.Tensor:
  mask = None
  if seqlen > 1:
      mask = torch.full((seqlen, seqlen), float("-inf"))
      mask = torch.triu(mask, diagonal=1)
      mask = torch.hstack([torch.zeros((seqlen, start_pos)), mask]).to(torch.float32).to(device)
  return mask
#1: og entroipix code ---

#2: maybe better code ---
# def build_attn_mask(seqlen: int, start_pos: int) -> Optional[torch.Tensor]:
#     if seqlen <= 1:
#         return None
#     mask = torch.full((seqlen, seqlen), float("-inf"), device=device)
#     mask = torch.triu(mask, diagonal=1)
#     mask = torch.hstack([torch.zeros((seqlen, start_pos), device=device), mask]).to(torch.float32)
#     return mask
#2: maybe better code ---

    # <thought>
    # For performance optimization, consider implementing a caching mechanism for
    # frequently used mask sizes.
    # </thought>

    # <thought>
    # This function creates a causal attention mask. For performance, consider pre-computing
    # masks for common sequence lengths and caching them, especially if the same lengths
    # are used frequently during generation.
    # </thought>

# <thought>
# This is a standard causal attention mask, but with a twist. The 'start_pos' parameter allows
# for a partial causal mask. This could be useful for continued generation or fine-tuning scenarios.
# However, I wonder about the performance implications of creating this mask on every forward pass.
# Could this be precomputed for common sequence lengths?
# </thought>

# # Prompt Dataset Loading Logic
# def load_prompt_from_file(file_path: Path) -> List[str]:
#     if file_path.suffix.lower() == '.csv':
#         with open(file_path, 'r') as f:
#             reader = csv.reader(f)
#             return [row[0] for row in reader]  # Assumes prompt is in the first column
#     else:
#         with open(file_path, 'r') as f:
#             return [f.read().strip()]

# def load_dataset(file_path: Path) -> List[dict]:
#     ext = file_path.suffix.lower()
#     if ext == '.csv':
#         with open(file_path, 'r') as f:
#             return list(csv.DictReader(f))
#     elif ext == '.jsonl':
#         with open(file_path, 'r') as f:
#             return [json.loads(line) for line in f]
#     else:
#         raise ValueError(f"Unsupported file format: {ext}")

# <thought>
# These utility functions for loading prompts and datasets are now more type-annotated,
# which aligns well with tyro's focus on type safety.
# </thought>

# <thought>
# These utility functions for loading prompts and datasets are useful additions.
# In the future, we might want to add support for more file formats or streaming
# for very large datasets to reduce memory usage.
# </thought>

# def main():
#   with torch.inference_mode():
#     model_params = LLAMA_1B_PARAMS
#     xfmr_weights = load_weights()

#     tokenizer = Tokenizer('entropix/tokenizer.model')
#     raw_tokens1 = tokenizer.encode(prompt,  bos=False, eos=False, allowed_special='all')
#     #this is not used in this script, but can be used to generate base_raw_tokens1
#     base_raw_tokens1 = tokenizer.encode(bp1, bos=True, eos=False, allowed_special='all')

# def generate(xfmr_weights: XfmrWeights, model_params: NamedTuple, tokens: List[int], tokenizer: Tokenizer, device: str, max_length: int = params["max_seq_len"]) -> str:
#     gen_tokens = None
#     cur_pos = 0
#     tokens = torch.tensor([tokens], dtype=torch.long).to(device)
#     bsz, seqlen = tokens.shape
    
    # -- REMOVED FOR CLI TESTING ---
    # attn_mask = build_attn_mask(seqlen, cur_pos)
    # freqs_cis = precompute_freqs_cis(model_params.head_dim, model_params.max_seq_len, model_params.rope_theta, model_params.use_scaled_rope)
    # kvcache = KVCache.new(model_params.n_layers, bsz, model_params.max_seq_len, model_params.n_local_kv_heads, model_params.head_dim).to(device)
    # logits, kvcache, _, _ = xfmr(xfmr_weights, model_params, tokens, cur_pos, freqs_cis[:seqlen], kvcache, attn_mask=attn_mask)
    # -- REMOVED FOR CLI TESTING ---
    
    # -- ADDED FOR CLI TESTING ---
def generate(xfmr_weights, model_params, tokens, tokenizer, device, max_length=params["max_seq_len"]):
    gen_tokens = None
    cur_pos = 0
    tokens = torch.tensor([tokens], dtype=torch.long).to(device)
    bsz, seqlen = tokens.shape
    attn_mask = build_attn_mask(seqlen, cur_pos)
    freqs_cis = precompute_freqs_cis(model_params.head_dim, model_params.max_seq_len, model_params.rope_theta, model_params.use_scaled_rope)
    kvcache = KVCache.new(model_params.n_layers, bsz, model_params.max_seq_len, model_params.n_local_kv_heads, model_params.head_dim).to(device)
    
    xfmr_kwargs = {'attn_mask': attn_mask} if attn_mask is not None else {}
    logits, kvcache, _, _ = xfmr(xfmr_weights, model_params, tokens, cur_pos, freqs_cis[:seqlen], kvcache, **xfmr_kwargs)
    
    next_token = torch.argmax(logits[:, -1], dim=-1, keepdim=True).to(torch.int32)
    gen_tokens = next_token
    print(tokenizer.decode([next_token.item()]), end='', flush=True)
    
    cur_pos = seqlen
    stop = torch.tensor([128001, 128008, 128009], device=device, dtype=torch.int32)
    
    while cur_pos < max_length:
        cur_pos += 1
        logits, kvcache, scores, stats = xfmr(xfmr_weights, model_params, next_token, cur_pos, freqs_cis[cur_pos:cur_pos+1], kvcache)
        next_token = sample(gen_tokens, logits, scores)
        gen_tokens = torch.cat((gen_tokens, next_token), dim=1)
        print(tokenizer.decode(next_token.tolist()[0]), end='', flush=True)
        if torch.isin(next_token, stop).any():
            break

    print()  # New line after generation is complete

# <thought>
# The generate function now takes the device as a parameter, making it more flexible.
# Consider adding more parameters for fine-tuning the generation process, such as
# temperature or top-k/top-p values.
# </thought>

    # <thought>
    # The generate function now returns the output instead of printing it directly.
    # This makes it more flexible for different use cases (e.g., saving to file, further processing).
    # Consider adding more configurable options, such as different stopping criteria or
    # sampling parameters that can be adjusted via CLI arguments.
    # </thought>

    # <thought>
    # This generation function is doing a lot. It's handling the initial forward pass,
    # then entering a loop for autoregressive generation. A few questions/observations:
    # 1. Why is cur_pos initialized to 0, then immediately set to seqlen?
    # 2. The stop condition (cur_pos < 8192) seems arbitrary. Is this the max context length?
    # 3. The use of torch.isin for stop tokens is interesting. Are these hardcoded values?
    #    It might be more flexible to pass these as parameters.
    # 4. The printing of tokens as they're generated could be problematic for large batches.
    #    Is this intended for debugging?
    # </thought>

    # print(prompt)
    # generate(xfmr_weights, model_params, raw_tokens1)
    
def main(config: CliConfig) -> None:
    print(f"Using device: {config.device}")
    torch.set_float32_matmul_precision('high')

    model_params = LLAMA_1B_PARAMS
    xfmr_weights = load_weights(config.weights_path)
    tokenizer = Tokenizer(config.tokenizer_path)

    if config.prompt_file:
        prompts = create_prompts_from_csv(str(config.prompt_file))
        if config.prompt_test:
            for p in prompts:
                print(f"Prompt: {p}")
                tokens = tokenizer.encode(p, bos=False, eos=False, allowed_special='all')
                generate(xfmr_weights, model_params, tokens, tokenizer, config.device, config.max_length)
                print("-" * 50)
        else:
            print(f"Prompt: {prompts[0]}")
            tokens = tokenizer.encode(prompts[0], bos=False, eos=False, allowed_special='all')
            generate(xfmr_weights, model_params, tokens, tokenizer, config.device, config.max_length)
    else:
        print(f"Prompt: {prompt}")
        tokens = tokenizer.encode(prompt, bos=False, eos=False, allowed_special='all')
        generate(xfmr_weights, model_params, tokens, tokenizer, config.device, config.max_length)
            
# <thought>
# The main function now uses the CliConfig dataclass, which makes it easier to add
# new configuration options in the future. The different modes (prompt file, dataset,
# interactive) are clearly separated, improving readability and maintainability.
# </thought>

if __name__ == "__main__":
    config = tyro.cli(CliConfig)
    main(config)
    
# <thought>
# Using tyro.cli with the CliConfig dataclass provides a clean and type-safe way to
# handle command-line arguments. This approach aligns well with the project's
# existing practices and makes it easier to extend the CLI in the future.
# </thought>