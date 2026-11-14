import json
import time
from pathlib import Path

import numpy as np
import wandb

from .analysis import class_distribution, thumbnail_grid
from .config import load_config
from .data import find_voc_root, load_masks, load_splits, rows_of
from .oracle import Oracle
from .selection import SELECTORS, SelectionState, check_batch, maxherding
from .selection.herding import median_sigma
from .tracking import append_result, define_prefix, init_run, iou_bar, per_class, save_head
from .train import evaluate, gpu_info, train_head


def load_cached(cfg):
    cache, enc = Path(cfg["cache_dir"]), cfg["encoder"]
    return {
        "train": np.load(cache / f"feats_{enc}_train.npy"),
        "val": np.load(cache / f"feats_{enc}_val.npy", mmap_mode="r"),
        "masks448": np.load(cache / "masks448_train.npy"),
        "vec": np.load(cache / f"vec_{enc}.npy"),
    }


def run_al(cfg):
    al, enc, seed = cfg["al"], cfg["encoder"], cfg["seed"]
    method = al["method"]
    tag = f"{method}-{al['scorer']}" if method == "uncertainty" else method
    name = f"{tag}-{enc}-s{seed}"
    out = Path(cfg["out_dir"])
    root = find_voc_root(cfg)
    splits = load_splits(cfg, root)
    data = load_cached(cfg)
    pool_ids = splits["pool"]
    pool_rows = rows_of(pool_ids, splits["train"])
    dev_rows = rows_of(splits["dev"], splits["train"])
    val_rows = np.arange(len(splits["val"]))
    dev_masks = load_masks(root, splits["dev"])
    val_masks = load_masks(root, splits["val"])
    oracle = Oracle(pool_ids, data["masks448"][pool_rows], root)

    sigma = median_sigma(data["vec"])
    print(f"[al] {name}: pool={len(pool_ids)} sigma={sigma:.4f}")
    run = init_run(cfg, name, group=tag)
    run.summary["sigma"] = sigma
    state = SelectionState(cfg=cfg, seed=seed, x=data["vec"], sigma=sigma, feats=data["train"], pool_rows=pool_rows)

    labeled, masks, selections = [], [], []
    for r, B in enumerate(al["budgets"]):
        candidates = np.setdiff1d(np.arange(len(pool_ids)), labeled)
        t0 = time.time()
        selector = maxherding if r == 0 else SELECTORS[method]
        batch, info = selector(state, labeled, candidates, B)
        select_time = time.time() - t0
        check_batch(batch, labeled, B)
        batch_ids = [pool_ids[i] for i in batch]
        masks.append(oracle.query(batch_ids))
        labeled += batch
        selections.append(batch_ids)
        state.labeled_masks = np.concatenate(masks)
        if r == 0:
            (out / f"cold_start_{enc}.json").write_text(json.dumps(batch_ids))

        prefix = f"r{r}/"
        define_prefix(run, prefix)
        curves = {"dev": (data["train"], dev_rows, dev_masks)}
        head, train_time = train_head(data["train"], state.labeled_masks, pool_rows[labeled], cfg, seed, run, prefix, curves)
        state.model = head
        val = evaluate(head, data["val"], val_rows, val_masks, cfg)
        dev = evaluate(head, data["train"], dev_rows, dev_masks, cfg)
        n_obj, boundary = oracle.cost_proxy([pool_ids[i] for i in labeled])
        gpu, peak = gpu_info()
        save_head(cfg, run, head, f"head-{name}-r{r}")

        row = {
            "encoder": enc, "method": method, "scorer": al["scorer"] if method == "uncertainty" else "",
            "round": r, "n_labeled": len(labeled), "miou": val["miou"], "dev_miou": dev["miou"], "ece": val["ece"],
            "per_class_iou": val["per_class_iou"], "n_objects_selected": n_obj, "boundary_len_selected": boundary,
            "select_time_s": round(select_time, 2), "train_time_s": round(train_time, 1), "sigma": sigma,
            "tau": "", "seed": seed, "gpu": gpu, "peak_mem_mb": round(peak, 1),
        }
        append_result(cfg, row)
        log = {
            "al/n_labeled": len(labeled), "al/round": r, "al/val_miou": val["miou"], "al/val_ece": val["ece"],
            "al/dev_miou": dev["miou"], "al/dev_ece": dev["ece"], "al/n_objects": n_obj, "al/boundary_len": boundary,
            "al/select_time_s": select_time, "al/train_time_s": train_time,
        }
        log.update(per_class("al/val_iou", val["per_class_iou"]))
        log[f"plots/val_iou_r{r}"] = iou_bar(val["per_class_iou"], f"{name} r{r} (n={len(labeled)}): val IoU per class")
        if "gains" in info:
            log["al/total_gain"] = sum(info["gains"])
            log["al/gain_per_pick"] = wandb.Histogram(info["gains"])
        if "scores" in info:
            log["al/candidate_scores"] = wandb.Histogram(info["scores"])
        run.log(log)
        print(f"[al] {name} round {r}: n={len(labeled)} val_miou={val['miou']:.4f} dev_miou={dev['miou']:.4f} "
              f"select={select_time:.1f}s train={train_time:.1f}s")

    (out / "selections").mkdir(parents=True, exist_ok=True)
    (out / "selections" / f"{name}.json").write_text(json.dumps(selections))
    plots = out / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    pool_masks = data["masks448"][pool_rows]
    grid = thumbnail_grid(root, selections[0], plots / f"first_batch_{name}.png")
    first = class_distribution(masks[0], pool_masks, plots / f"classes_first_{name}.png", f"{name}: first batch vs pool")
    final = class_distribution(state.labeled_masks, pool_masks, plots / f"classes_final_{name}.png", f"{name}: L vs pool")
    run.log({"plots/first_batch": wandb.Image(str(grid)), "plots/classes_first": wandb.Image(str(first)),
             "plots/classes_final": wandb.Image(str(final))})
    assert sum(len(q) for q in oracle.log) == len(labeled)
    run.finish()


if __name__ == "__main__":
    run_al(load_config())
