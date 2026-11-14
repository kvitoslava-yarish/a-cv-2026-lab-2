import torch
import torch.nn as nn
import torch.nn.functional as F


class SegHead(nn.Module):
    def __init__(self, n_layers=2, dim=768, hidden=256, num_classes=21, dropout=0.1):
        super().__init__()
        self.norms = nn.ModuleList(nn.LayerNorm(dim) for _ in range(n_layers))
        self.projs = nn.ModuleList(nn.Linear(dim, hidden) for _ in range(n_layers))
        self.fuse = nn.Sequential(
            nn.Linear(n_layers * hidden, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, num_classes)
        )

    def forward(self, x, size=None):
        # x: B x n_layers x C x h x w; 1x1 convs are applied as Linear on channels-last tensors
        x = x.permute(0, 1, 3, 4, 2)
        z = torch.cat([proj(norm(x[:, i])) for i, (norm, proj) in enumerate(zip(self.norms, self.projs))], dim=-1)
        logits = self.fuse(z).permute(0, 3, 1, 2)
        if size is not None:
            logits = F.interpolate(logits, size=size, mode="bilinear", align_corners=False)
        return logits
