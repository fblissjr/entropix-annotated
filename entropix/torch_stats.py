import torch

# Device selection, tree is like first apple silicion, then cuda, fallback is cpu.
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

#print(f"Using device: {device}")

from typing import NamedTuple

# <thought>
# Once again, we see the repeated device selection logic. This reinforces the need
# for a centralized configuration system. The commented-out print statement is also
# a recurring pattern, suggesting inconsistent debugging practices.
# 
# The use of NamedTuple suggests we're going for a lightweight, immutable data structure.
# This is generally a good practice for maintaining clean and predictable state.
# </thought>

class AttnStats(NamedTuple):
    entropy: torch.Tensor  # (bsz, n_layers, num_heads)
    varentropy: torch.Tensor  # (bsz, n_layers, num_heads)
    n_layers: int
    n_heads: int

    @classmethod
    def new(cls, bsz: int, n_layers: int, n_heads: int) -> 'AttnStats':
        return cls(
            entropy=torch.zeros((bsz, n_layers, n_heads), dtype=torch.float32, device=device),
            varentropy=torch.zeros((bsz, n_layers, n_heads), dtype=torch.float32, device=device),
            n_layers=n_layers,
            n_heads=n_heads
        )

    # <thought>
    # The use of a class method for construction is interesting. It allows for a more
    # descriptive way to create the object compared to direct instantiation. However,
    # it's worth noting that this pattern is more common in languages like Java or C++
    # than in Python.
    # 
    # The initialization of entropy and varentropy tensors with zeros is a good default,
    # but it might be worth considering whether this could lead to any division-by-zero
    # issues in downstream calculations.
    # </thought>

    @property
    def avg_entropy(self):
        return self.entropy.sum(dim=-1, keepdim=False)  # Average across heads

    @property
    def std_error(self):
        return torch.sqrt(torch.mean(self.varentropy)) / (self.n_heads * self.n_layers)

    # <thought>
    # These properties provide useful aggregate statistics. The avg_entropy is straightforward,
    # but the std_error calculation is more complex. It seems to be calculating the standard error
    # of the mean entropy across all heads and layers.
    # 
    # One potential issue: if n_heads or n_layers is 0, this will raise a division by zero error.
    # It might be worth adding a check for this edge case.
    # </thought>

    def update(self, scores: torch.Tensor, layer_idx: int):
        # scores shape: (bsz, n_heads, seqlen, n_words)
        probs = torch.nn.functional.softmax(scores, dim=-1)
        new_entropy = -torch.sum(torch.where(probs > 0, probs * torch.log(probs), torch.tensor(0.0)), dim=-1)
        new_varentropy = torch.sum(probs * (torch.log(probs) + new_entropy.unsqueeze(-1))**2, dim=-1)

        # Update entropy and varentropy tensors
        self.entropy[:, layer_idx, :] = new_entropy
        self.varentropy[:, layer_idx, :] = new_varentropy

        return self

    # <thought>
    # This update method is doing quite a bit of work:
    # 1. It's converting attention scores to probabilities.
    # 2. It's calculating entropy and varentropy for these probabilities.
    # 3. It's updating the stored statistics for a specific layer.
    # 
    # The use of torch.where to handle log(0) cases is good for numerical stability.
    # 
    # However, there are some potential issues:
    # 1. The method modifies the object in-place, which can be unexpected for a NamedTuple.
    #    This could lead to subtle bugs if not used carefully.
    # 2. The method always returns self, which is unnecessary for in-place operations and
    #    could be misleading.
    # 3. There's no check that layer_idx is within the valid range.
    # 
    # It might be worth considering making this class immutable and returning a new object
    # with updated values instead of modifying in-place.
    # </thought>