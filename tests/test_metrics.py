import torch
import torch.nn.functional as F

from src.head import SegHead
from src.metrics import SegMetrics


def test_perfect_prediction_and_ignore():
    target = torch.tensor([[0, 1], [2, 255]])
    probs = F.one_hot(torch.tensor([[0, 1], [2, 5]]), 21).permute(2, 0, 1).float()
    m = SegMetrics()
    m.update(probs, target)
    res = m.compute()
    assert res["miou"] == 1.0 and res["ece"] < 1e-9 and res["pixel_acc"] == 1.0


def test_wrong_prediction():
    target = torch.zeros(2, 2, dtype=torch.long)
    probs = F.one_hot(torch.ones(2, 2, dtype=torch.long), 21).permute(2, 0, 1).float()
    m = SegMetrics()
    m.update(probs, target)
    assert m.compute()["miou"] == 0.0


def test_head_shapes():
    head = SegHead(n_layers=2, dim=32, hidden=16)
    out = head(torch.randn(3, 2, 32, 28, 28), size=(448, 448))
    assert out.shape == (3, 21, 448, 448)

