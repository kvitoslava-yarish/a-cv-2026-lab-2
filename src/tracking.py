import csv
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
import wandb
from PIL import Image

from .data import IGNORE, VOC_CLASSES, class_presence, load_image

COLUMNS = [
    "encoder", "method", "scorer", "round", "n_labeled", "miou", "dev_miou", "ece", "per_class_iou",
    "n_objects_selected", "boundary_len_selected", "select_time_s", "train_time_s", "sigma", "tau", "seed",
    "gpu", "peak_mem_mb", "best_dev_step", "best_dev_miou", "val_miou_at_best_dev",
]


def to_cell(v):
    if isinstance(v, list):
        return json.dumps([None if isinstance(x, float) and math.isnan(x) else x for x in v])
    return v


def append_csv(path, row, columns=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = columns or list(row)
    new = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        if new:
            writer.writeheader()
        writer.writerow({k: to_cell(v) for k, v in row.items()})


def append_result(cfg, row):
    append_csv(Path(cfg["out_dir"]) / "results.csv", row, COLUMNS)



def init_run(cfg, name, group):
    w = cfg["wandb"]
    mode = os.environ.get("WANDB_MODE", w["mode"])  # the Kaggle kernel sets offline when it has no W&B key
    run = wandb.init(project=w["project"], entity=w["entity"], mode=mode, name=name, group=group,
                     config=cfg, reinit="finish_previous")
    run.define_metric("trainer/global_step")
    run.define_metric("al/n_labeled")
    run.define_metric("al/*", step_metric="al/n_labeled")
    return run


def define_prefix(run, prefix):
    run.define_metric(f"{prefix}*", step_metric="trainer/global_step")


def save_head(cfg, run, head, name):
    path = Path(cfg["out_dir"]) / "heads" / f"{name}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(head.state_dict(), path)
    if cfg["wandb"]["log_artifacts"] and run is not None and not run.disabled:
        art = wandb.Artifact(name, type="model")
        art.add_file(str(path))
        run.log_artifact(art)
    return path


def per_class(prefix, ious):
    return {f"{prefix}/{c}": v for c, v in zip(VOC_CLASSES, ious) if not math.isnan(v)}


def iou_bar(ious, title):
    rows = [[c, v] for c, v in zip(VOC_CLASSES, ious) if not math.isnan(v)]
    return wandb.plot.bar(wandb.Table(data=rows, columns=["class", "iou"]), "class", "iou", title=title)


def class_names(mask):
    return [VOC_CLASSES[c] for c in np.unique(mask) if c not in (0, IGNORE)]


def selection_table(root, ids, masks, scores, size=224):
    """Selected images with their (now labeled) masks as an overlay, classes and selection score."""
    labels = {**dict(enumerate(VOC_CLASSES)), IGNORE: "void"}
    table = wandb.Table(columns=["rank", "id", "image", "classes", "score"])
    for rank, (image_id, mask, score) in enumerate(zip(ids, masks, scores)):
        img = load_image(root, image_id).resize((size, size))
        small = np.array(Image.fromarray(mask).resize((size, size), Image.NEAREST))
        overlay = wandb.Image(img, masks={"ground_truth": {"mask_data": small, "class_labels": labels}})
        table.add_data(rank, image_id, overlay, ", ".join(class_names(mask)), score)
    return table


def class_counts(prefix, masks):
    counts = class_presence(masks).sum(0)
    return {f"{prefix}/{c}": int(n) for c, n in zip(VOC_CLASSES, counts)}
