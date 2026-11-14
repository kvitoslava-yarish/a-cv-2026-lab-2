import csv
import json
import math
from pathlib import Path

import torch
import wandb

COLUMNS = [
    "encoder", "method", "scorer", "round", "n_labeled", "miou", "dev_miou", "ece", "per_class_iou",
    "n_objects_selected", "boundary_len_selected", "select_time_s", "train_time_s", "sigma", "tau", "seed",
    "gpu", "peak_mem_mb",
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
    run = wandb.init(project=w["project"], entity=w["entity"], mode=w["mode"], name=name, group=group,
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
