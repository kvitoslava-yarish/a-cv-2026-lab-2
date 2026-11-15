"""ProbCover (Yehuda et al., NeurIPS 2022) on image-level vectors.

Graph G_delta: edge i -> c iff ||x_i - x_c|| < delta. Greedy max coverage: pick the candidate whose ball covers
the most still-uncovered points. This is exactly herding_core with a 0/1 kernel K[i, c] = [d(i, c) < delta]
(MaxHerding is the soft version with a Gaussian kernel), so the greedy step is reused from herding.py.
"""
import numpy as np
import torch
from sklearn.cluster import KMeans

from .herding import herding_core


def nearest_other_cluster(x, labels):
    """r_i = distance from x_i to the closest point with a different pseudo-label."""
    d = torch.cdist(torch.as_tensor(x, dtype=torch.float32), torch.as_tensor(x, dtype=torch.float32))
    lab = torch.as_tensor(labels)
    d[lab[:, None] == lab[None, :]] = float("inf")
    return d.min(1).values.numpy()


def purity(r, delta):
    """Fraction of points whose open delta-ball contains only their own pseudo-label: B_delta(x_i) pure iff delta <= r_i."""
    return float((np.asarray(r) >= delta).mean())


def choose_delta(x, k, alpha, seed):
    """Largest delta with purity >= alpha, purity measured against k-means pseudo-labels (no oracle labels).
    purity(delta) = mean(r_i >= delta) is non-increasing in delta, so the answer is an order statistic of r."""
    labels = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(np.asarray(x))
    r = np.sort(nearest_other_cluster(x, labels))
    delta = float(r[int(np.floor((1 - alpha) * len(r)))])
    assert purity(r, delta) >= alpha
    return delta, labels


def ball_kernel(x, cols, delta, device=None):
    x = torch.as_tensor(x, dtype=torch.float32, device=device)
    return (torch.cdist(x, x[list(cols)]) < delta).float()


def coverage(x, labeled_idx, delta, device=None):
    """P(C(L, delta)): fraction of the pool inside the union of delta-balls around L."""
    if not len(labeled_idx):
        return 0.0
    return float(ball_kernel(x, labeled_idx, delta, device).max(1).values.mean())


def select_probcover(x, labeled_idx, candidate_idx, B, delta, seed, device=None):
    """Returns pool indices and per-pick gains (number of newly covered points).
    Once everything reachable is covered the gain is 0 for every candidate; the rest of the batch is then random,
    as in the paper (otherwise argmax would just take the lowest indices)."""
    K = ball_kernel(x, candidate_idx, delta, device)
    m = ball_kernel(x, labeled_idx, delta, device).max(1).values if len(labeled_idx) else torch.zeros(len(K), device=K.device)
    picks, gains = herding_core(K, m, B)
    n_greedy = next((i for i, g in enumerate(gains) if g <= 0), B)
    batch = [int(candidate_idx[p]) for p in picks[:n_greedy]]
    if n_greedy < B:
        rest = np.setdiff1d(np.asarray(candidate_idx), batch)
        batch += np.random.default_rng(seed + len(labeled_idx)).choice(rest, B - n_greedy, replace=False).tolist()
    return [int(b) for b in batch], gains[:n_greedy]
