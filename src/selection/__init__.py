from dataclasses import dataclass

import numpy as np
import torch

from .herding import select_maxherding
from .probcover import choose_delta, select_probcover
from .uncertainty import bootstrap_heads, image_scores, select_topb


@dataclass
class SelectionState:
    """Everything a selector may use. Masks are only those of L (returned by the Oracle)."""
    cfg: dict
    seed: int
    x: np.ndarray
    sigma: float
    feats: np.ndarray
    pool_rows: np.ndarray
    labeled_masks: np.ndarray = None
    model: torch.nn.Module = None
    delta: float = None


def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def maxherding(state, labeled_idx, candidate_idx, B):
    batch, gains = select_maxherding(state.x, labeled_idx, candidate_idx, B, state.sigma, device=device())
    return batch, {"gains": gains, "batch_scores": gains}


def probcover(state, labeled_idx, candidate_idx, B):
    al = state.cfg["al"]
    if state.delta is None:  # label-free and fixed for the whole run: depends only on the pool vectors
        state.delta, _ = choose_delta(state.x, al["probcover_k"], al["probcover_alpha"], state.seed)
    batch, gains = select_probcover(state.x, labeled_idx, candidate_idx, B, state.delta, state.seed, device=device())
    # per-pick score = newly covered points; the randomly filled tail (pool fully covered) scores 0
    return batch, {"gains": gains, "batch_scores": list(gains) + [0.0] * (len(batch) - len(gains))}


def uncertainty(state, labeled_idx, candidate_idx, B):
    scorer = state.cfg["al"]["scorer"]
    heads = [state.model] if scorer == "ripu" else bootstrap_heads(state, labeled_idx)
    scores = image_scores(heads, state.feats, state.pool_rows[candidate_idx], state.cfg, scorer)
    batch = select_topb(scores, candidate_idx, B)
    score_of = dict(zip(candidate_idx.tolist(), scores.tolist()))
    return batch, {"scores": scores, "batch_scores": [score_of[c] for c in batch]}


SELECTORS = {"maxherding": maxherding, "probcover": probcover, "uncertainty": uncertainty}
# Methods that need no task model select their own cold-start batch; model-based ones fall back to MaxHerding.
MODEL_FREE = {"maxherding", "probcover"}


def check_batch(batch, labeled_idx, B):
    assert len(batch) == B, f"expected {B} ids, got {len(batch)}"
    assert len(set(batch)) == B, "duplicate ids in batch"
    assert not set(batch) & set(labeled_idx), "batch contains labeled ids"
