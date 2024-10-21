import torch
import torch.nn.functional as F
from typing import Tuple

from entropix.torch_main import calculate_varentropy_logsoftmax, _sample

class MCTSSearch:
    def __init__(self, cxfmr, xfmr_weights, model_params, freqs_cis, kvcache):
        self.cxfmr = cxfmr
        self.xfmr_weights = xfmr_weights
        self.model_params = model_params
        self.freqs_cis = freqs_cis
        self.kvcache = kvcache
        self.max_depth = 6
        self.n_branches = 5

    # <thought>
    # This initialization suggests that MCTS is being applied directly to the transformer model.
    # The inclusion of xfmr_weights, freqs_cis, and kvcache indicates that this MCTS
    # implementation is tightly coupled with the specific transformer architecture.
    # 
    # The max_depth and n_branches parameters are hardcoded. This lack of flexibility
    # could be limiting in different scenarios where we might want to adjust the
    # search depth or breadth.
    # </thought>

    def _is_normal_range(self, ent: float, vent: float) -> bool:
        return ent < 5.0 and vent < 5.0

    # <thought>
    # This method defines what's considered a "normal range" for entropy and varentropy.
    # The threshold of 5.0 for both seems arbitrary. It's not clear why this specific
    # value was chosen or if it's universally applicable across different models or tasks.
    # 
    # Using the same threshold for both entropy and varentropy might not be ideal,
    # as they measure different aspects of the distribution.
    # </thought>

    def simulate_path(self, token: torch.Tensor, cur_pos: int, depth: int = 0) -> Tuple[torch.Tensor, bool]:
        if depth >= self.max_depth:
            return token, False

        next_logits, _ = self.cxfmr(self.xfmr_weights, self.model_params, token.unsqueeze(0), 
                                    cur_pos + depth + 1, 
                                    self.freqs_cis[cur_pos + depth + 1:cur_pos + depth + 2], 
                                    self.kvcache)
        next_ent, next_vent = calculate_varentropy_logsoftmax(next_logits)
        
        if self._is_normal_range(next_ent.item(), next_vent.item()):
            return token, True

        next_token = _sample(next_logits, temperature=1.0)
        return self.simulate_path(next_token.squeeze(), cur_pos, depth + 1)

    # <thought>
    # This simulate_path method is the core of the MCTS implementation. Some observations:
    # 
    # 1. It's using recursion, which could lead to stack overflow for very deep searches.
    #    An iterative approach might be more robust.
    # 
    # 2. The method stops either when it reaches max_depth or when it finds a token
    #    with "normal" entropy/varentropy. This is an interesting termination condition
    #    that's trying to find stable or confident predictions.
    # 
    # 3. It's calling the transformer (cxfmr) directly for each step. This could be
    #    computationally expensive, especially for large models.
    # 
    # 4. The sampling temperature is fixed at 1.0, which might not always be optimal.
    #    Allowing this to be adjustable could provide more flexibility.
    # 
    # 5. The method doesn't accumulate any scores along the path, which is atypical
    #    for MCTS. This seems to be more of a depth-first search until a "good" token
    #    is found, rather than a true MCTS implementation.
    # </thought>

    def search(self, logits: torch.Tensor, cur_pos: int) -> torch.Tensor:
        # Select initial candidates
        candidates = []
        for _ in range(self.n_branches):
            candidates.append(_sample(logits, temperature=2))
        
        for candidate in candidates:
            # Remove extra dimensions to get a 1D tensor
            candidate_token = candidate.squeeze()
            final_token, success = self.simulate_path(candidate_token, cur_pos)
            if success:
                return final_token.unsqueeze(0).unsqueeze(0)

        # If no path leads to normal range, return the first candidate
        return candidates[0]

    # <thought>
    # This search method is the entry point for the MCTS process. However, it doesn't
    # really implement the core MCTS algorithm (selection, expansion, simulation, backpropagation).
    # Instead, it's more of a beam search with depth-first exploration. Observations:
    # 
    # 1. It samples initial candidates with a higher temperature (2.0) than the
    #    simulate_path method (1.0). This encourages more diversity in the initial set.
    # 
    # 2. It explores each candidate path fully before moving to the next. This is
    #    different from traditional MCTS which would interleave exploration of different paths.
    # 
    # 3. It returns as soon as it finds a successful path. This could lead to
    #    suboptimal choices if a better path exists among the unexplored candidates.
    # 
    # 4. If no successful path is found, it arbitrarily returns the first candidate.
    #    This fallback strategy might not be ideal.
    # 
    # 5. The method doesn't use any of the accumulated statistics typical in MCTS
    #    (visit counts, scores, etc.) to guide its search or final selection.
    # </thought>