import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .data import VOC_CLASSES, class_presence, load_image


def thumbnail_grid(root, ids, path, cols=7):
    rows = int(np.ceil(len(ids) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2 * cols, 2 * rows))
    for ax in np.atleast_1d(axes).ravel():
        ax.axis("off")
    for ax, i in zip(np.atleast_1d(axes).ravel(), ids):
        ax.imshow(load_image(root, i).resize((160, 160)))
        ax.set_title(i, fontsize=6)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def class_distribution(selected_masks, pool_masks, path, title=""):
    sel = class_presence(selected_masks).mean(0)
    pool = class_presence(pool_masks).mean(0)
    x = np.arange(len(VOC_CLASSES))
    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.bar(x - 0.2, pool, 0.4, label="pool")
    ax.bar(x + 0.2, sel, 0.4, label="selected")
    ax.set_xticks(x, VOC_CLASSES, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("fraction of images containing class")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
