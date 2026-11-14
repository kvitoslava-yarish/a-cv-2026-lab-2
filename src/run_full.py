import numpy as np

from .al_loop import load_cached
from .config import load_config
from .data import find_voc_root, load_masks, load_splits
from .tracking import append_result, define_prefix, init_run, save_head
from .train import evaluate, gpu_info, train_head


def main():
    cfg = load_config()
    enc, seed = cfg["encoder"], cfg["seed"]
    name = f"full-{enc}-s{seed}"
    root = find_voc_root(cfg)
    splits = load_splits(cfg, root)
    data = load_cached(cfg)
    val_rows = np.arange(len(splits["val"]))
    val_masks = load_masks(root, splits["val"])
    train_rows = np.arange(len(splits["train"]))

    run = init_run(cfg, name, group="full")
    define_prefix(run, "full/")
    curves = {"val": (data["val"], val_rows, val_masks)}
    head, train_time = train_head(data["train"], data["masks448"], train_rows, cfg, seed, run, "full/", curves)
    val = evaluate(head, data["val"], val_rows, val_masks, cfg)
    gpu, peak = gpu_info()
    save_head(cfg, run, head, f"head-{name}")
    append_result(cfg, {
        "encoder": enc, "method": "full", "scorer": "", "round": 0, "n_labeled": len(train_rows),
        "miou": val["miou"], "dev_miou": "", "ece": val["ece"], "per_class_iou": val["per_class_iou"],
        "select_time_s": 0, "train_time_s": round(train_time, 1), "sigma": "", "tau": "", "seed": seed,
        "gpu": gpu, "peak_mem_mb": round(peak, 1),
    })
    run.summary.update({"val_miou": val["miou"], "val_ece": val["ece"], "train_time_s": train_time,
                        "gpu_hours": train_time / 3600, "gpu": gpu, "peak_mem_mb": peak})
    print(f"[full] {name}: val_miou={val['miou']:.4f} ece={val['ece']:.4f} time={train_time:.1f}s gpu={gpu}")
    run.finish()


if __name__ == "__main__":
    main()
