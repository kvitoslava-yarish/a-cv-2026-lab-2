import numpy as np
import torch

from src.selection.herding import herding_core
from src.selection.probcover import (ball_kernel, choose_delta, coverage, nearest_other_cluster, purity,
                                     select_probcover)

TOY = np.array([[0.0], [1.0], [2.0], [6.0], [7.0], [9.0]])


def random_pool(n=80, d=8, seed=0):
    x = np.random.default_rng(seed).normal(size=(n, d)).astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def brute_force_greedy(x, labeled, candidates, B, delta):
    d = np.abs(x[:, None, 0] - x[None, :, 0]) if x.shape[1] == 1 else np.linalg.norm(x[:, None] - x[None], axis=-1)
    covered = (d[:, labeled] < delta).any(1) if len(labeled) else np.zeros(len(x), bool)
    picks, available = [], list(candidates)
    for _ in range(B):
        gain = [int(((d[:, c] < delta) & ~covered).sum()) for c in available]
        c = available.pop(int(np.argmax(gain)))
        picks.append(c)
        covered |= d[:, c] < delta
    return picks


def test_toy_two_balls_cover_the_pool():
    # delta = 2.5: balls of 0, 1 and 2 all equal {0,1,2} (hard balls tie, unlike the Gaussian kernel where 1 wins);
    # on the right only ball(7) = {6,7,9} covers 3 points. The third pick has zero gain and is random.
    batch, gains = select_probcover(TOY, [], np.arange(6), B=3, delta=2.5, seed=0)
    assert TOY[batch[0], 0] in (0.0, 1.0, 2.0) and TOY[batch[1], 0] == 7.0
    assert gains == [3.0, 3.0]
    assert coverage(TOY, batch[:2], 2.5) == 1.0
    assert len(set(batch)) == 3


def test_matches_brute_force_max_coverage():
    x = random_pool()
    labeled = [0, 3]
    candidates = np.setdiff1d(np.arange(len(x)), labeled)
    delta = 0.9
    batch, _ = select_probcover(x, labeled, candidates, 6, delta, seed=0)
    assert batch == brute_force_greedy(x, labeled, candidates, 6, delta)


def test_is_herding_with_indicator_kernel():
    x = random_pool()
    K = ball_kernel(x, np.arange(len(x)), 0.8)
    picks, _ = herding_core(K, torch.zeros(len(x)), 5)
    batch, _ = select_probcover(x, [], np.arange(len(x)), 5, 0.8, seed=0)
    assert batch == picks


def test_full_coverage_falls_back_to_random_valid_batch():
    x = random_pool(n=30)
    labeled = [0, 1]
    candidates = np.setdiff1d(np.arange(len(x)), labeled)
    batch, gains = select_probcover(x, labeled, candidates, 20, delta=10.0, seed=0)  # one ball covers everything
    assert len(gains) == 0
    assert len(batch) == 20 and len(set(batch)) == 20 and not set(batch) & set(labeled)


def test_batch_is_valid_and_deterministic():
    x = random_pool()
    labeled = [2, 9, 11]
    candidates = np.setdiff1d(np.arange(len(x)), labeled)
    a, _ = select_probcover(x, labeled, candidates, 15, 0.7, seed=0)
    b, _ = select_probcover(x, labeled, candidates, 15, 0.7, seed=0)
    assert a == b
    assert len(set(a)) == 15 and not set(a) & set(labeled)


def test_purity_closed_form_matches_grid():
    x = random_pool(n=60)
    labels = np.random.default_rng(1).integers(0, 4, len(x))
    r = nearest_other_cluster(x, labels)
    d = np.linalg.norm(x[:, None] - x[None], axis=-1)
    for delta in np.linspace(0.05, 2.0, 25):
        ball = d < delta
        pure = ~(ball & (labels[:, None] != labels[None, :])).any(1)
        assert abs(purity(r, delta) - pure.mean()) < 1e-9


def test_choose_delta_is_largest_with_enough_purity():
    x = random_pool(n=100)
    alpha = 0.95
    delta, labels = choose_delta(x, k=5, alpha=alpha, seed=0)
    r = nearest_other_cluster(x, labels)
    assert purity(r, delta) >= alpha
    assert purity(r, delta + 1e-4) < alpha or delta >= r.max()


def test_coverage_grows_with_selection():
    x = random_pool()
    batch, _ = select_probcover(x, [], np.arange(len(x)), 10, 0.8, seed=0)
    covs = [coverage(x, batch[:k], 0.8) for k in range(1, 11)]
    assert all(b >= a for a, b in zip(covs, covs[1:]))
    assert coverage(x, [], 0.8) == 0.0
