import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .config import load_config
from .data import find_voc_root, load_image, load_masks, load_splits, rows_of
from .tracking import append_csv

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
PATCH = 16


class ImageSet(Dataset):
    def __init__(self, root, ids, size):
        self.root, self.ids, self.size = root, ids, size

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        img = load_image(self.root, self.ids[i]).resize((self.size, self.size), resample=2)
        x = torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1).float() / 255
        return (x - MEAN) / STD


class Encoder(nn.Module):
    """Frozen ViT returning patch tokens of selected blocks (1-based) as B x L x C x h x w."""

    def __init__(self, spec):
        super().__init__()
        self.source, self.layers = spec["source"], spec["layers"]
        if self.source == "hf":
            from transformers import AutoModel
            self.model = AutoModel.from_pretrained(spec["name"], token=os.environ.get("HF_TOKEN"))
            self.n_prefix = 1 + self.model.config.num_register_tokens
        else:
            import timm
            self.model = timm.create_model(spec["name"], pretrained=True, dynamic_img_size=True)
        self.model.eval().requires_grad_(False)

    @torch.no_grad()
    def forward(self, x):
        b, _, h, w = x.shape
        if self.source == "hf":
            hs = self.model(pixel_values=x, output_hidden_states=True).hidden_states
            outs = [hs[l][:, self.n_prefix:].transpose(1, 2).reshape(b, -1, h // PATCH, w // PATCH) for l in self.layers]
        else:
            outs = self.model.forward_intermediates(x, indices=[l - 1 for l in self.layers], intermediates_only=True)
        return torch.stack(outs, 1)


def extract_split(encoder, root, ids, path, cfg, device):
    e = cfg["extract"]
    loader = DataLoader(ImageSet(root, ids, cfg["img_size"]), batch_size=e["batch_size"], num_workers=e["num_workers"])
    grid = cfg["img_size"] // PATCH
    out, start = None, 0
    for x in loader:
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            f = encoder(x.to(device)).half().cpu().numpy()
        if out is None:
            out = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=(len(ids), *f.shape[1:]))
        out[start:start + len(f)] = f
        start += len(f)
    assert out.shape[-1] == grid
    out.flush()
    return out


def image_vectors(feats, rows, layer=-1):
    v = np.stack([feats[r, layer].astype(np.float32).mean(axis=(1, 2)) for r in rows])
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def extract(cfg, root, splits):
    enc_name = cfg["encoder"]
    cache = Path(cfg["cache_dir"])
    cache.mkdir(parents=True, exist_ok=True)
    done = cache / f"{enc_name}.done"
    if done.exists():
        print(f"[extract] {enc_name}: cached")
        return None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = Encoder(cfg["encoders"][enc_name]).to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    train = extract_split(encoder, root, splits["train"], cache / f"feats_{enc_name}_train.npy", cfg, device)
    extract_split(encoder, root, splits["val"], cache / f"feats_{enc_name}_val.npy", cfg, device)
    seconds = time.time() - t0
    np.save(cache / f"vec_{enc_name}.npy", image_vectors(train, rows_of(splits["pool"], splits["train"])))
    done.touch()
    n_images = len(splits["train"]) + len(splits["val"])
    return {
        "encoder": enc_name,
        "model": cfg["encoders"][enc_name]["name"],
        "params_m": sum(p.numel() for p in encoder.parameters()) / 1e6,
        "extract_time_s": round(seconds, 1),
        "images_per_s": round(n_images / seconds, 2),
        "peak_mem_mb": torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else 0.0,
        "gpu": torch.cuda.get_device_name() if device.type == "cuda" else "cpu",
    }


def main():
    cfg = load_config()
    root = find_voc_root(cfg)
    splits = load_splits(cfg, root)
    print(f"[extract] voc_root={root} train={len(splits['train'])} pool={len(splits['pool'])} "
          f"dev={len(splits['dev'])} val={len(splits['val'])}")
    masks_path = Path(cfg["cache_dir"]) / "masks448_train.npy"
    if not masks_path.exists():
        masks_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(masks_path, load_masks(root, splits["train"], size=cfg["img_size"]))
    stats = extract(cfg, root, splits)
    if stats:
        print(f"[extract] {stats}")
        append_csv(Path(cfg["out_dir"]) / "encoder_stats.csv", stats)


if __name__ == "__main__":
    main()
