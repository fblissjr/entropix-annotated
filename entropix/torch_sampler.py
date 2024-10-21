import torch
import torch.nn.functional as F
from typing import Tuple, Dict

# Device selection, tree is like first apple silicion, then cuda, fallback is cpu.
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

LN_2 = 0.69314718056  # ln(2) = 1.0 / LOG2_E

# <thought>
# Once again, we see the device selection logic. This repetition across files
# is becoming a clear pattern that needs addressing.
# 
# The LN_2 constant is interesting. It's used for converting between natural log
# and log base 2. This suggests we'll be doing some information theory calculations,
# likely related to entropy.
# </thought>

def calculate_varentropy_logsoftmax(logits: torch.Tensor, axis: int = -1) -> Tuple[torch.Tensor, torch.Tensor]:
    """Calculate the entropy and varentropy of the probability distribution using logsoftmax."""
    log_probs = F.log_softmax(logits, dim=axis)
    probs = torch.exp(log_probs)
    entropy = -torch.sum(probs * log_probs, axis=axis) / LN_2  # Convert to base-2
    varentropy = torch.sum(probs * (log_probs / LN_2 + entropy.unsqueeze(-1))**2, axis=axis)
    return entropy, varentropy

# <thought>
# This function calculates both entropy and varentropy (variance of entropy) of the logits.
# The use of log_softmax is numerically stable, which is good.
# 
# Converting to base-2 entropy is interesting - it means the entropy values will be in bits,
# which can be more intuitive for analysis.
# 
# The varentropy calculation is less common. It could be used to measure the uncertainty
# in the model's predictions, beyond just the average uncertainty (entropy).
# 
# One potential issue: if probs are very small, the exp(log_probs) could lead to numerical
# instability. It might be worth adding a small epsilon to prevent this.
# </thought>

def multinomial_sample_one(probs_sort: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Samples one token from a multinomial distribution with sorted probabilities."""
    # Use torch.rand instead of Exponential distribution
    q = torch.rand(probs_sort.shape, generator=generator, device=probs_sort.device)
    return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(torch.int32)

# <thought>
# This is an implementation of the Gumbel-Max trick for sampling from a categorical distribution.
# It's more efficient than the standard torch.multinomial because it avoids the CDF computation.
# 
# The use of a provided generator ensures reproducibility, which is great for debugging.
# 
# However, this method assumes the probabilities are already sorted. This could be
# problematic if the calling code forgets to sort, leading to subtle bugs.
# </thought>

def _sample(logits: torch.Tensor, temperature=0.666, top_p=0.90, top_k=27, min_p: float = 0.0, generator: torch.Generator = None) -> torch.Tensor:
    bsz = logits.shape[0]
    logit = logits[:, -1]
    probs = F.softmax(logit / temperature, dim=-1)

    # Apply min_p sampling
    if min_p > 0.0:
        p_max = torch.max(probs, dim=-1, keepdim=True).values
        indices_to_remove = probs < (min_p * p_max)
        logit = torch.where(indices_to_remove, torch.full_like(logit, float('-inf')), logit)

    # Apply top-k sampling
    top_k_probs, top_k_indices = torch.topk(probs, k=min(top_k, probs.shape[-1]))
    probs_sort = torch.flip(top_k_probs, dims=[-1])
    probs_idx = torch.flip(top_k_indices, dims=[-1])
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    # Apply top-p sampling
    mask = torch.where(probs_sum - probs_sort > top_p, torch.tensor(1.0, device=device), torch.tensor(0.0, device=device))
    probs_sort = probs_sort * (1 - mask)
    probs_sort = probs_sort / torch.sum(probs_sort, dim=-1, keepdim=True)
    next_token = multinomial_sample_one(probs_sort, generator)
    # Convert next_token to int64 before using it in gather
    next_token_g = torch.gather(probs_idx, -1, next_token.reshape(bsz, 1).to(torch.int64))
    return next_token_g.to(torch.int32)

# <thought>
# This is a comprehensive sampling function that combines temperature scaling,
# top-k sampling, top-p (nucleus) sampling, and minimum probability thresholding.
# 
# Observations:
# 1. The default temperature (0.666) is lower than the common 0.7 or 1.0. This will
#    make the distribution slightly more peaked, potentially reducing randomness.
# 2. The combination of top-k and top-p is interesting. Usually, one or the other is used.
# 3. The min_p sampling is an additional constraint that's not commonly seen. It ensures
#    that no probability is too small relative to the max probability.
# 
# Potential issues:
# 1. The function modifies 'logit' after computing 'probs'. This could lead to
#    inconsistencies if not careful.
# 2. The repeated use of torch.flip might be inefficient. Could be optimized.
# 3. The final conversion to int32 might not be necessary depending on the downstream use.
# 
# Overall, this function provides a lot of flexibility in sampling, but the combination
# of all these methods might be overkill for some use cases.
# </thought>

def calculate_metrics(logits: torch.Tensor, attention_scores: torch.Tensor) -> Dict[str, torch.Tensor]:
    entropy, varentropy = calculate_varentropy_logsoftmax(logits)
    attention_probs = F.softmax(attention_scores, dim=-1)
    attn_entropy = -torch.sum(attention_probs * torch.log2(torch.clamp(attention_probs, 1e-10, 1.0)), dim=-1)
    attn_varentropy = torch.var(attn_entropy, dim=-1)
    
    # Add a small epsilon to avoid NaN when all values are the same
    attn_varentropy = torch.where(torch.isnan(attn_varentropy), torch.zeros_like(attn_varentropy), attn_varentropy)
    mean_attention = torch.mean(attention_probs, dim=1)
    agreement = torch.mean(torch.abs(attention_probs - mean_attention.unsqueeze(1)), dim=(1, 2))

    interaction_strength = torch.mean(torch.abs(attention_scores), dim=(1, 2, 3))

    return {
        "logits_entropy": torch.mean(entropy),
        "logits_varentropy": torch.mean(varentropy),
        "attn_entropy": torch.mean(attn_entropy),
        "attn_varentropy": torch.mean(attn_varentropy),
        "agreement": torch.mean(agreement),
        "interaction_strength": interaction_strength
    }

# <thought>
# This function calculates various metrics related to the model's output and attention mechanism.
# It's quite comprehensive, considering both the output logits and the attention scores.
# 
# The metrics include:
# 1. Entropy and varentropy of the output distribution
# 2. Entropy and varentropy of the attention distribution
# 3. "Agreement" of attention heads (how similar their distributions are)
# 4. "Interaction strength" based on the magnitude of attention scores
# 
# These metrics could be very useful for analyzing model behavior, but computing them
# during inference might add significant overhead.
# 
# The use of torch.where to handle potential NaN values is a good practice for numerical stability.
# 
# One question: why are we taking the mean of most metrics, but not for interaction_strength?
# This inconsistency might lead to confusion when interpreting the results.
# </thought>

# <thought>
# The adaptive_sample and sample functions likely implement
# more sophisticated sampling strategies based on these metrics. This suggests a focus
# on controlling the generation process based on the model's confidence and attention patterns.
# 
# Such adaptive sampling could potentially improve the quality and diversity of generated text,
# but it also adds complexity and potential overhead to the generation process.
# </thought>

def adaptive_sample(logits: torch.Tensor, metrics: Dict[str, torch.Tensor],
                    gen_tokens: torch.Tensor, n_samples: int,
                    base_temp: float = 0.666, base_top_p: float = 0.90, base_top_k: int = 40, base_min_p: float = 0.03,
                    generator: torch.Generator = None) -> torch.Tensor:
    logits_uncertainty = metrics["logits_entropy"] + metrics["logits_varentropy"]
    attn_uncertainty = metrics["attn_entropy"] + metrics["attn_varentropy"]

    temperature = base_temp * (1 + 0.3 * logits_uncertainty + 0.2 * attn_uncertainty - 0.2 * metrics["agreement"])
    top_p = torch.clamp(base_top_p * (1 + 0.1 * metrics["attn_varentropy"]), 0.1, 1.0)
    top_k = int(torch.clamp(
        torch.round(torch.tensor(base_top_k) * (1 + 0.3 * metrics["interaction_strength"].item() - 0.2 * metrics["agreement"].item())),
        min=1,
        max=100
    ).item())
    min_p = torch.clamp(base_min_p * (1 - 0.5 * logits_uncertainty), 0.01, 0.5)

    samples = []
    for _ in range(n_samples):
        sample = _sample(logits, temperature=temperature, top_p=top_p, top_k=top_k, min_p=min_p, generator=generator)
        samples.append(sample)

    def score_sample(sample):
        # Flatten the sample tensor and convert to long (int64)
        sample_flat = sample.flatten().to(torch.long)
        
        # Create one-hot encoding
        one_hot = F.one_hot(sample_flat, logits.shape[-1])
        
        # Reshape log_softmax output to match one_hot
        log_probs = F.log_softmax(logits, dim=-1).view(-1, logits.shape[-1])
        
        # Calculate log probability
        log_prob = torch.sum(log_probs * one_hot)
        
        confidence_score = (
            (1 - metrics["logits_entropy"]) * 0.1 +
            (1 - metrics["attn_entropy"]) * 0.2 +
            (1 - metrics["logits_varentropy"]) * 0.3 +
            (1 - metrics["attn_varentropy"]) * 0.4 +
            metrics["agreement"] * 0.5 +
            metrics["interaction_strength"] * 0.6
        )
        return log_prob + confidence_score

    sample_scores = torch.stack([score_sample(sample) for sample in samples])
    best_sample_idx = torch.argmax(sample_scores)
    return samples[best_sample_idx]

# <thought>
# This adaptive_sample function is quite sophisticated. It adjusts sampling parameters
# based on the current state of the model (as reflected in the metrics). Some key points:
# 
# 1. It combines multiple uncertainty metrics (entropy and varentropy for both logits and attention)
#    to adjust the temperature. Higher uncertainty leads to higher temperature, encouraging exploration.
# 
# 2. The top_p and top_k parameters are also dynamically adjusted based on attention varentropy
#    and interaction strength. This allows for more or less focused sampling depending on the
#    model's current state.
# 
# 3. The min_p parameter is adjusted inversely to logits uncertainty, potentially preventing
#    the model from becoming too uncertain.
# 
# 4. It generates multiple samples and then scores them based on both their likelihood and
#    a confidence score derived from the metrics.
# 
# This approach could potentially lead to more controlled and high-quality sampling, but it's
# also quite complex and computationally expensive. The effectiveness would need to be
# empirically validated against simpler methods.
# 
# Potential issues:
# 1. The magic numbers (0.3, 0.2, etc.) in the adjustments lack clear justification.
# 2. The scoring function weights different metrics, but the choice of weights seems arbitrary.
# 3. Generating multiple samples and scoring them could be slow for real-time applications.
# </thought>

def sample(gen_tokens: torch.Tensor, logits: torch.Tensor, attention_scores: torch.Tensor,
           temperature=0.666, top_p=0.90, top_k=27, min_p: float = 0.0, 
           generator: torch.Generator = torch.Generator(device=device).manual_seed(1337)) -> torch.Tensor:
    metrics = calculate_metrics(logits, attention_scores)
    ent, vent = metrics["logits_entropy"], metrics["logits_varentropy"]
    attn_ent, attn_vent = metrics["attn_entropy"], metrics["attn_varentropy"]
    agreement = metrics["agreement"]
    interaction_strength = metrics["interaction_strength"]

    # Low Entropy, Low Varentropy: "flowing with unspoken intent"
    if ent < 0.1 and vent < 0.1:
        return torch.argmax(logits[:, -1], dim=-1, keepdim=True).to(torch.int32)

    # High Entropy, Low Varentropy: "treading carefully, asking clarifying questions"
    elif ent > 3.0 and vent < 0.1:
        # Insert a clarifying question token if not already present
        if not torch.isin(gen_tokens[:,-1], torch.tensor([2564], device=device)).any():
            return torch.tensor([[2564]], dtype=torch.int32, device=device)  # Assuming 2564 is our "ask clarifying question" token
        else:
            # If we've just asked a question, sample with slightly higher temperature
            temp_adj = 1.3 + 0.2 * attn_ent  # Increase temperature based on attention entropy
            return _sample(logits, temperature=min(1.5, temperature * temp_adj), top_p=top_p, top_k=top_k, min_p=min_p, generator=generator)

    # Low Entropy, High Varentropy: "exploring forks in the path"
    elif ent < 5.0 and vent > 5.0:
        temp_adj = 1.2 + 0.3 * interaction_strength  # Increase temperature based on interaction strength
        top_k_adj = max(5, int(top_k * (1 + 0.5 * (1 - agreement))))  # Increase top_k when agreement is low
        return _sample(logits, temperature=min(1.5, temperature * temp_adj), top_p=top_p, top_k=top_k_adj, min_p=min_p, generator=generator)

    # High Entropy, High Varentropy: "resampling in the mist"
    elif ent > 5.0 and vent > 5.0:
        # Use high temperature and adjusted top_p based on attention metrics
        temp_adj = 2.0 + 0.5 * attn_vent  # Increase temperature based on attention varentropy
        top_p_adj = max(0.5, top_p - 0.2 * attn_ent)  # Decrease top_p when attention entropy is high
        return _sample(logits, temperature=max(2.0, temperature * temp_adj), top_p=top_p_adj, top_k=top_k, min_p=min_p, generator=generator)

    # Middle ground: use adaptive sampling
    else:
        return adaptive_sample(
            logits,
            metrics,
            gen_tokens,
            n_samples=5,
            base_temp=temperature,
            base_top_p=top_p,
            base_top_k=top_k,
            generator=generator
        )

# <thought>
# This sample function is the heart of the sampling strategy. It uses the calculated metrics
# to decide between different sampling approaches based on the current state of the model.
# Key observations:
# 
# 1. It defines four distinct regimes based on entropy and varentropy, each with its own
#    sampling strategy. This is a novel approach that tries to adapt the sampling method
#    to the model's current "state of mind".
# 
# 2. In the "treading carefully" regime, it can actually force the model to ask a clarifying
#    question by returning a specific token (2564). This is an interesting way to try to
#    control the model's behavior at a high level.
# 
# 3. The "exploring forks" and "resampling in the mist" regimes adjust sampling parameters
#    based on interaction strength, agreement, and attention metrics. This could potentially
#    lead to more diverse or focused sampling as needed.
# 
# 4. If none of the specific regimes apply, it falls back to the adaptive_sample method,
#    which provides a more general-purpose adaptive sampling approach.
# 
# Potential issues and considerations:
# 1. The entropy and varentropy thresholds (0.1, 3.0, 5.0) seem arbitrary and might need
#    to be tuned for different models or tasks.
# 2. The clarifying question token (2564) is hardcoded, which might not be appropriate for
#    all models or tokenizers.
# 3. The complexity of this approach might make it difficult to reason about the model's
#    behavior and could potentially introduce unexpected biases in the generation process.
# 4. The computational overhead of calculating all these metrics for every sampling step
#    could be significant, potentially impacting generation speed.
# </thought>