import json
import time
from pathlib import Path

import numpy as np
import wandb

from .analysis import class_distribution, thumbnail_grid
from .config import load_config
from .data import find_voc_root, load_masks, load_splits, rows_of
from .oracle import Oracle
from .selection import MODEL_FREE, SELECTORS, SelectionState, check_batch, device, maxherding
from .selection.herding import kernel_coverage, median_sigma
from .selection.probcover import choose_delta, coverage
from .tracking import (append_csv, append_result, class_counts, define_prefix, init_run, iou_bar, per_class,
                       save_head, selection_table)
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
    scale = al.get("sigma_scale", 1.0)
    variant = method if scale == 1 else f"{method}-sx{scale:g}"  # sigma ablation: median sigma times a factor
    tag = f"{method}-{al['scorer']}" if method == "uncertainty" else variant
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

    base_sigma = median_sigma(data["vec"])
    sigma = base_sigma * scale
    # label-free reference radius: ProbCover selects with it, every method reports P(C(L, delta)) with it
    delta, _ = choose_delta(data["vec"], al["probcover_k"], al["probcover_alpha"], seed)
    print(f"[al] {name}: pool={len(pool_ids)} sigma={sigma:.4f} delta={delta:.4f}")
    run = init_run(cfg, name, group=tag)
    run.summary.update({"sigma": sigma, "delta": delta})
    state = SelectionState(cfg=cfg, seed=seed, x=data["vec"], sigma=sigma, feats=data["train"], pool_rows=pool_rows,
                           delta=delta)

    plots = out / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    pool_masks = data["masks448"][pool_rows]
    labeled, masks, selections = [], [], []
    for r, B in enumerate(al["budgets"]):
        candidates = np.setdiff1d(np.arange(len(pool_ids)), labeled)
        t0 = time.time()
        selector = SELECTORS[method] if r > 0 or method in MODEL_FREE else maxherding
        batch, info = selector(state, labeled, candidates, B)
        select_time = time.time() - t0
        check_batch(batch, labeled, B)
        batch_ids = [pool_ids[i] for i in batch]
        masks.append(oracle.query(batch_ids))
        labeled += batch
        selections.append(batch_ids)
        state.labeled_masks = np.concatenate(masks)
        if r == 0 and selector is maxherding and scale == 1:  # the shared fallback batch of model-based methods
            (out / f"cold_start_{enc}.json").write_text(json.dumps(batch_ids))

        prefix = f"r{r}/"
        define_prefix(run, prefix)
        curves = {"dev": (data["train"], dev_rows, dev_masks), "val": (data["val"], val_rows, val_masks)}
        head, train_time, history = train_head(data["train"], state.labeled_masks, pool_rows[labeled], cfg, seed, run,
                                               prefix, curves)
        best = max(history, key=lambda h: h["dev"])
        state.model = head
        val = evaluate(head, data["val"], val_rows, val_masks, cfg)
        dev = evaluate(head, data["train"], dev_rows, dev_masks, cfg)
        n_obj, boundary = oracle.cost_proxy([pool_ids[i] for i in labeled])
        gpu, peak = gpu_info()
        save_head(cfg, run, head, f"head-{name}-r{r}")

        row = {
            "encoder": enc, "method": variant, "scorer": al["scorer"] if method == "uncertainty" else "",
            "round": r, "n_labeled": len(labeled), "miou": val["miou"], "dev_miou": dev["miou"], "ece": val["ece"],
            "per_class_iou": val["per_class_iou"], "n_objects_selected": n_obj, "boundary_len_selected": boundary,
            "select_time_s": round(select_time, 2), "train_time_s": round(train_time, 1), "sigma": sigma,
            "tau": "", "seed": seed, "gpu": gpu, "peak_mem_mb": round(peak, 1),
            "best_dev_step": best["step"], "best_dev_miou": best["dev"], "val_miou_at_best_dev": best["val"],
        }
        append_result(cfg, row)
        log = {
            "al/n_labeled": len(labeled), "al/round": r, "al/val_miou": val["miou"], "al/val_ece": val["ece"],
            "al/dev_miou": dev["miou"], "al/dev_ece": dev["ece"], "al/n_objects": n_obj, "al/boundary_len": boundary,
            "al/select_time_s": select_time, "al/train_time_s": train_time,
            "al/best_dev_step": best["step"], "al/best_dev_miou": best["dev"],
            "al/val_miou_at_best_dev": best["val"],
        }
        log.update(per_class("al/val_iou", val["per_class_iou"]))
        log[f"plots/val_iou_r{r}"] = iou_bar(val["per_class_iou"], f"{name} r{r} (n={len(labeled)}): val IoU per class")
        log.update(class_counts("al/L_images", state.labeled_masks))
        log.update(class_counts("al/batch_images", masks[-1]))
        log[f"selected/r{r}"] = selection_table(root, batch_ids, masks[-1], info["batch_scores"])
        dist = class_distribution(masks[-1], pool_masks, plots / f"classes_{name}_r{r}.png",
                                  f"{name} r{r}: batch of {B} vs pool")
        log[f"plots/classes_r{r}"] = wandb.Image(str(dist))
        if info.get("gains"):  # ProbCover may return no greedy picks once the pool is fully covered
            log["al/total_gain"] = sum(info["gains"])
            log["al/gain_per_pick"] = wandb.Histogram(info["gains"])
        # coverage of the pool by L, label-free, same sigma / delta for every method (kept out of results.csv so its
        # columns stay fixed); n_greedy < B means ProbCover ran out of uncovered points and filled the batch randomly
        cov = {"kernel_coverage": kernel_coverage(state.x, labeled, base_sigma, device()),
               "ball_coverage": coverage(state.x, labeled, delta, device())}
        log.update({f"al/{k}": v for k, v in cov.items()})
        append_csv(out / "selector_stats.csv", {
            "encoder": enc, "method": variant, "scorer": row["scorer"], "seed": seed, "round": r,
            "n_labeled": len(labeled), **cov, "sigma": base_sigma, "delta": delta,
            "n_greedy": len(info["gains"]) if "gains" in info else "", "total_gain": sum(info.get("gains", [])),
        })
        if "scores" in info:
            log["al/candidate_scores"] = wandb.Histogram(info["scores"])
        run.log(log)
        print(f"[al] {name} round {r}: n={len(labeled)} val_miou={val['miou']:.4f} dev_miou={dev['miou']:.4f} "
              f"select={select_time:.1f}s train={train_time:.1f}s")

    (out / "selections").mkdir(parents=True, exist_ok=True)
    (out / "selections" / f"{name}.json").write_text(json.dumps(selections))
    grid = thumbnail_grid(root, selections[0], plots / f"first_batch_{name}.png")
    final = class_distribution(state.labeled_masks, pool_masks, plots / f"classes_final_{name}.png", f"{name}: L vs pool")
    run.log({"plots/first_batch": wandb.Image(str(grid)), "plots/classes_final": wandb.Image(str(final))})
    assert sum(len(q) for q in oracle.log) == len(labeled)
    run.finish()


if __name__ == "__main__":
    run_al(load_config())
