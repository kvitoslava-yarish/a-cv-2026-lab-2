import numpy as np
import torch

from src.selection.herding import herding_core, herding_inputs, median_sigma, select_maxherding

TOY = np.array([[0.0], [1.0], [2.0], [6.0], [7.0], [9.0]])


def test_toy_exercise_picks_1_then_7():
    batch, gains = select_maxherding(TOY, [], np.arange(6), B=2, sigma=2.0)
    assert TOY[batch, 0].tolist() == [1.0, 7.0]
    assert gains[0] > gains[1] > 0


def test_toy_top2_by_first_gain_differs():
    K, m = herding_inputs(TOY, [], np.arange(6), sigma=2.0)
    first_gain = K.sum(0)
    assert TOY[first_gain.topk(2).indices.numpy(), 0].tolist() == [1.0, 2.0]


def random_pool(n=60, d=8, seed=0):
    x = np.random.default_rng(seed).normal(size=(n, d)).astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def test_batch_is_valid():
    x = random_pool()
    labeled = [0, 5, 7]
    candidates = np.setdiff1d(np.arange(len(x)), labeled)
    batch, _ = select_maxherding(x, labeled, candidates, B=10, sigma=median_sigma(x))
    assert len(batch) == 10 and len(set(batch)) == 10
    assert not set(batch) & set(labeled)


def test_deterministic():
    x = random_pool()
    sigma = median_sigma(x)
    a, _ = select_maxherding(x, [], np.arange(len(x)), 12, sigma)
    b, _ = select_maxherding(x, [], np.arange(len(x)), 12, sigma)
    assert a == b


def test_constant_weights_equal_maxherding():
    x = random_pool()
    candidates = np.arange(3, len(x))
    K, m = herding_inputs(x, [0, 1, 2], candidates, median_sigma(x))
    plain, _ = herding_core(K, m, 15)
    weighted, _ = herding_core(K, m, 15, weights=torch.full((len(x),), 0.37))
    assert plain == weighted


def test_coverage_update_uses_max():
    K = torch.tensor([[1.0, 0.9], [0.0, 0.1]])
    picks, gains = herding_core(K, torch.zeros(2), 2)
    assert picks == [0, 1]
    assert abs(gains[1] - 0.1) < 1e-6
