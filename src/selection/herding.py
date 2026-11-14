import torch


def gaussian_kernel(a, b, sigma):
    return torch.exp(-torch.cdist(a, b).pow(2) / sigma ** 2)


def median_sigma(x):
    """Label-free kernel width: median pairwise Euclidean distance between image vectors."""
    x = torch.as_tensor(x, dtype=torch.float32)
    d = torch.cdist(x, x)
    i, j = torch.triu_indices(len(x), len(x), offset=1)
    return float(d[i, j].median())


def herding_core(K, m, B, weights=None):
    """Greedy (weighted) herding. K: targets x candidates, m: current coverage of every target.
    gain(c) = sum_i w_i * max(K[i, c] - m_i, 0); coverage is updated by max."""
    w = torch.ones(K.shape[0], dtype=K.dtype, device=K.device) if weights is None else weights.to(K)
    m = m.clone()
    available = torch.ones(K.shape[1], dtype=torch.bool, device=K.device)
    picks, gains = [], []
    for _ in range(B):
        gain = (w[:, None] * (K - m[:, None]).clamp_min(0)).sum(0)
        gain[~available] = -float("inf")
        c = int(gain.argmax())
        picks.append(c)
        gains.append(float(gain[c]))
        available[c] = False
        m = torch.maximum(m, K[:, c])
    return picks, gains


def herding_inputs(x, labeled_idx, candidate_idx, sigma, device=None):
    x = torch.as_tensor(x, dtype=torch.float32, device=device)
    K = gaussian_kernel(x, x[list(candidate_idx)], sigma)
    if len(labeled_idx):
        m = gaussian_kernel(x, x[list(labeled_idx)], sigma).max(1).values
    else:
        m = torch.zeros(len(x), device=x.device)
    return K, m


def select_maxherding(x, labeled_idx, candidate_idx, B, sigma, device=None):
    """Rows of K = all pool images (targets), columns = unlabeled candidates. Returns pool indices and gains."""
    K, m = herding_inputs(x, labeled_idx, candidate_idx, sigma, device)
    picks, gains = herding_core(K, m, B)
    return [int(candidate_idx[p]) for p in picks], gains
