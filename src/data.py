import json
from pathlib import Path

import numpy as np
from PIL import Image

VOC_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]
IGNORE = 255


def is_voc_root(path):
    return (path / "ImageSets" / "Segmentation" / "train.txt").exists() and (path / "SegmentationClass").is_dir()


def find_voc_root(cfg, bases=(Path("/kaggle/input"), Path("data")), max_depth=6):
    """Finds the train/val VOC folder, e.g. /kaggle/input/<dataset>/VOC2012_train_val/VOC2012_train_val."""
    if cfg.get("voc_root"):
        return Path(cfg["voc_root"])
    for base in map(Path, bases):
        for depth in range(max_depth + 1):
            for path in sorted(base.glob("/".join(["*"] * depth))) if depth else [base]:
                if path.is_dir() and is_voc_root(path):
                    return path
    raise FileNotFoundError("VOC2012 train/val with SegmentationClass not found; set voc_root=...")


def read_ids(root, split):
    return (Path(root) / "ImageSets" / "Segmentation" / f"{split}.txt").read_text().split()


def load_image(root, image_id):
    return Image.open(Path(root) / "JPEGImages" / f"{image_id}.jpg").convert("RGB")


def load_mask(root, image_id, kind="SegmentationClass", size=None):
    mask = Image.open(Path(root) / kind / f"{image_id}.png")
    if size is not None:
        mask = mask.resize((size, size), Image.NEAREST)
    return np.array(mask, dtype=np.uint8)


def load_masks(root, ids, size=None):
    masks = [load_mask(root, i, size=size) for i in ids]
    return np.stack(masks) if size is not None else masks


def class_presence(masks, num_classes=21):
    return np.stack([np.bincount(m[m != IGNORE].ravel(), minlength=num_classes)[:num_classes] > 0 for m in masks])


def draw_dev(presence, size, min_count, seed):
    rng = np.random.default_rng(seed)
    n = len(presence)
    for _ in range(1000):
        pick = np.sort(rng.choice(n, size, replace=False))
        if presence[pick, 1:].sum(0).min() >= min_count:
            return pick
    return pick


def load_splits(cfg, root=None):
    path = Path(cfg["out_dir"]) / "splits.json"
    if path.exists():
        return json.loads(path.read_text())
    root = root or find_voc_root(cfg)
    train = read_ids(root, "train")[: cfg["limit_train"]]
    val = read_ids(root, "val")[: cfg["limit_val"]]
    presence = class_presence(load_masks(root, train))
    dev_pos = set(draw_dev(presence, cfg["dev_size"], cfg["dev_min_images_per_class"], cfg["seed"]).tolist())
    splits = {
        "train": train,
        "val": val,
        "dev": [t for j, t in enumerate(train) if j in dev_pos],
        "pool": [t for j, t in enumerate(train) if j not in dev_pos],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(splits))
    return splits


def rows_of(ids, reference):
    index = {k: i for i, k in enumerate(reference)}
    return np.array([index[k] for k in ids], dtype=np.int64)
