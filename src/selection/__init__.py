from dataclasses import dataclass

import numpy as np
import torch

from .herding import select_maxherding
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


def maxherding(state, labeled_idx, candidate_idx, B):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    batch, gains = select_maxherding(state.x, labeled_idx, candidate_idx, B, state.sigma, device=dev)
    return batch, {"gains": gains}


def uncertainty(state, labeled_idx, candidate_idx, B):
    scorer = state.cfg["al"]["scorer"]
    heads = [state.model] if scorer == "ripu" else bootstrap_heads(state, labeled_idx)
    scores = image_scores(heads, state.feats, state.pool_rows[candidate_idx], state.cfg, scorer)
    return select_topb(scores, candidate_idx, B), {"scores": scores}


SELECTORS = {"maxherding": maxherding, "uncertainty": uncertainty}


def check_batch(batch, labeled_idx, B):
    assert len(batch) == B, f"expected {B} ids, got {len(batch)}"
    assert len(set(batch)) == B, "duplicate ids in batch"
    assert not set(batch) & set(labeled_idx), "batch contains labeled ids"
