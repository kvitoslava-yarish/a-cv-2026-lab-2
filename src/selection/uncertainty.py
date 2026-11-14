import numpy as np
import torch
import torch.nn.functional as F

from ..train import iter_logits, train_head


def entropy(p, dim=1):
    return -(p * p.clamp_min(1e-12).log()).sum(dim)


def ripu_map(probs, window):
    """RIPU: region impurity of predicted labels x region-averaged prediction entropy."""
    onehot = F.one_hot(probs.argmax(1), probs.shape[1]).permute(0, 3, 1, 2).float()
    pool = dict(kernel_size=window, stride=1, padding=window // 2, count_include_pad=False)
    impurity = entropy(F.avg_pool2d(onehot, **pool))
    uncertainty = F.avg_pool2d(entropy(probs)[:, None], **pool)[:, 0]
    return impurity * uncertainty


def bald_map(probs_list):
    probs = torch.stack(probs_list)
    return entropy(probs.mean(0)) - entropy(probs, dim=2).mean(0)


def top_fraction_mean(maps, frac):
    flat = maps.flatten(1)
    k = max(1, int(frac * flat.shape[1]))
    return flat.topk(k, dim=1).values.mean(1)


def image_scores(heads, feats, rows, cfg, scorer):
    al = cfg["al"]
    size = (al["score_size"], al["score_size"])
    scores = []
    for logits in iter_logits(heads, feats, rows, cfg["train"]["eval_batch_size"], size=size):
        probs = [lg.softmax(1) for lg in logits]
        maps = ripu_map(probs[0], al["ripu_window"]) if scorer == "ripu" else bald_map(probs)
        scores.append(top_fraction_mean(maps, al["top_frac"]).cpu())
    return torch.cat(scores).numpy()


def bootstrap_heads(state, labeled_idx):
    rng = np.random.default_rng(state.seed)
    rows = state.pool_rows[labeled_idx]
    heads = []
    for k in range(state.cfg["al"]["bald_heads"]):
        pick = rng.integers(0, len(rows), len(rows))
        head, _ = train_head(state.feats, state.labeled_masks[pick], rows[pick], state.cfg, seed=state.seed + 1000 + k)
        heads.append(head)
    return heads


def select_topb(scores, candidate_idx, B):
    order = np.argsort(-scores, kind="stable")[:B]
    return [int(candidate_idx[i]) for i in order]
