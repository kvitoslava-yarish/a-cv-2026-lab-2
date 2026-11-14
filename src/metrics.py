import torch

from .data import IGNORE


class SegMetrics:
    def __init__(self, num_classes=21, n_bins=15, device="cpu"):
        self.nc, self.n_bins = num_classes, n_bins
        self.cm = torch.zeros(num_classes * num_classes, dtype=torch.float64, device=device)
        self.bin_count = torch.zeros(n_bins, dtype=torch.float64, device=device)
        self.bin_conf = torch.zeros_like(self.bin_count)
        self.bin_acc = torch.zeros_like(self.bin_count)
        self.nll = torch.zeros((), dtype=torch.float64, device=device)

    @torch.no_grad()
    def update(self, probs, target):
        """probs: C x H x W (softmax), target: H x W with 255 = ignore."""
        valid = target != IGNORE
        t = target[valid].long()
        p = probs.permute(1, 2, 0)[valid].double()
        conf, pred = p.max(1)
        self.cm += torch.bincount(t * self.nc + pred, minlength=self.nc ** 2).double()
        b = (conf * self.n_bins).long().clamp(max=self.n_bins - 1)
        self.bin_count += torch.bincount(b, minlength=self.n_bins).double()
        self.bin_conf += torch.bincount(b, weights=conf, minlength=self.n_bins)
        self.bin_acc += torch.bincount(b, weights=(pred == t).double(), minlength=self.n_bins)
        self.nll += -p.gather(1, t[:, None]).clamp_min(1e-12).log().sum()

    def compute(self):
        cm = self.cm.view(self.nc, self.nc)
        inter = cm.diag()
        union = cm.sum(0) + cm.sum(1) - inter
        iou = torch.where(union > 0, inter / union.clamp_min(1), torch.full_like(inter, float("nan")))
        n = self.bin_count.sum().clamp_min(1)
        return {
            "miou": float(iou.nanmean()),
            "per_class_iou": [float(v) for v in iou],
            "ece": float((self.bin_acc - self.bin_conf).abs().sum() / n),
            "pixel_acc": float(inter.sum() / n),
            "loss": float(self.nll / n),
        }
