"""Benchmark report from outputs/results.csv (+ selector_stats.csv): learning curves, summary table with AUBC and gain
over Random, cost-proxy curves, coverage curves, MaxHerding sigma ablation. Works with partial results.

    python -m src.report [out_dir=outputs]
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import load_config

KEYS = ["encoder", "label", "n_labeled"]


def label_of(df):
    scorer = df["scorer"].astype("string").fillna("")
    return np.where(scorer != "", df["method"].astype(str) + "-" + scorer, df["method"].astype(str))


def load(out):
    res = pd.read_csv(out / "results.csv")
    res["label"] = label_of(res)
    stats_path = out / "selector_stats.csv"
    stats = pd.read_csv(stats_path) if stats_path.exists() else None
    if stats is not None:
        stats["label"] = label_of(stats)
    return res, stats


def aggregate(res):
    """Mean (and std over seeds) per encoder x method x budget."""
    al = res[res["method"] != "full"]
    cols = ["miou", "ece", "n_objects_selected", "boundary_len_selected", "select_time_s", "train_time_s"]
    g = al.groupby(KEYS)[cols]
    mean, std = g.mean(), g.std().fillna(0.0)
    mean["miou_std"], mean["n_seeds"] = std["miou"], g.size()
    full = res[res["method"] == "full"].groupby("encoder")["miou"].agg(["mean", "std"]).fillna(0.0)
    return mean.reset_index(), full


def aubc(n, miou, log_x=True):
    """Normalized area under the mIoU-vs-budget curve (trapezoid). With log_x every doubling of the budget weighs
    the same, so the cold-start rounds are not drowned by the last, largest round. Result lies in [0, 1]."""
    x = np.log2(n) if log_x else np.asarray(n, dtype=float)
    if len(x) < 2:
        return float("nan")
    y = np.asarray(miou, dtype=float)
    return float(((x[1:] - x[:-1]) * (y[1:] + y[:-1]) / 2).sum() / (x[-1] - x[0]))


def summary_table(mean, full, stats):
    rows = []
    budgets = sorted(mean["n_labeled"].unique())
    for (enc, label), d in mean.groupby(["encoder", "label"]):
        d = d.sort_values("n_labeled")
        rnd = mean[(mean["encoder"] == enc) & (mean["label"] == "random")].set_index("n_labeled")["miou"]
        row = {"encoder": enc, "method": label, "seeds": int(d["n_seeds"].min())}
        for _, r in d.iterrows():
            n = int(r["n_labeled"])
            row[f"mIoU@{n}"] = r["miou"]
            if label != "random" and n in rnd.index:
                row[f"gain@{n}"] = r["miou"] - rnd[n]
        row["AUBC"] = aubc(d["n_labeled"].to_numpy(), d["miou"].to_numpy())
        row["AUBC_lin"] = aubc(d["n_labeled"].to_numpy(), d["miou"].to_numpy(), log_x=False)
        last = d.iloc[-1]
        row["ECE@last"] = last["ece"]
        row["objects@last"] = last["n_objects_selected"]
        row["boundary@last"] = last["boundary_len_selected"]
        row["select_s_total"] = d["select_time_s"].sum()
        if enc in full.index:
            row["full_mIoU"] = full.loc[enc, "mean"]
            row["% of full @last"] = 100 * last["miou"] / full.loc[enc, "mean"]
        if stats is not None:
            s = stats[(stats["encoder"] == enc) & (stats["label"] == label)]
            if len(s):
                s = s.groupby("n_labeled")[["kernel_coverage", "ball_coverage"]].mean().sort_index()
                row["kernel_cov@first"], row["kernel_cov@last"] = s["kernel_coverage"].iloc[0], s["kernel_coverage"].iloc[-1]
                row["ball_cov@first"], row["ball_cov@last"] = s["ball_coverage"].iloc[0], s["ball_coverage"].iloc[-1]
        rows.append(row)
    table = pd.DataFrame(rows)
    order = ["encoder", "method", "seeds"] + [f"mIoU@{n}" for n in budgets] + [f"gain@{n}" for n in budgets]
    order += [c for c in table.columns if c not in order]
    return table[[c for c in order if c in table.columns]]


def to_markdown(df):
    def fmt(v):
        if isinstance(v, float):
            return "" if np.isnan(v) else f"{v:.4g}"
        return str(v)
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(fmt(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(lines)


def is_main(label):
    return "-sx" not in label  # sigma-ablation variants get their own figure


def panels(encoders, title, w=6.5, h=4.5):
    fig, axes = plt.subplots(1, len(encoders), figsize=(w * len(encoders), h), squeeze=False)
    fig.suptitle(title)
    return fig, dict(zip(encoders, axes[0]))


def learning_curves(mean, full, path, labels=None, title="val mIoU vs labeled images"):
    encoders = sorted(mean["encoder"].unique())
    fig, axes = panels(encoders, title)
    for enc, ax in axes.items():
        d = mean[mean["encoder"] == enc]
        for label, c in d.groupby("label"):
            if labels is not None and label not in labels:
                continue
            c = c.sort_values("n_labeled")
            ax.plot(c["n_labeled"], c["miou"], marker="o", label=label)
            ax.fill_between(c["n_labeled"], c["miou"] - c["miou_std"], c["miou"] + c["miou_std"], alpha=0.15)
        if enc in full.index:
            ax.axhline(full.loc[enc, "mean"], ls="--", c="k", lw=1, label="full data")
        ticks = sorted(d["n_labeled"].unique())
        ax.set_xscale("log", base=2)
        ax.set_xticks(ticks, [str(int(t)) for t in ticks])
        ax.set(title=enc, xlabel="labeled images (log scale)", ylabel="val mIoU")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def cost_curves(mean, path):
    """Same curves with the post-hoc annotation-cost proxies on x: does the ranking survive another cost unit?"""
    encoders = sorted(mean["encoder"].unique())
    fig, axes = plt.subplots(2, len(encoders), figsize=(6.5 * len(encoders), 8), squeeze=False)
    fig.suptitle("val mIoU vs annotation-cost proxies")
    for j, enc in enumerate(encoders):
        d = mean[(mean["encoder"] == enc) & mean["label"].map(is_main)]
        for i, (col, name) in enumerate([("n_objects_selected", "object instances"),
                                         ("boundary_len_selected", "mask-boundary pixels")]):
            ax = axes[i, j]
            for label, c in d.groupby("label"):
                c = c.sort_values("n_labeled")
                ax.plot(c[col], c["miou"], marker="o", label=label)
            ax.set_xscale("log")
            ax.set(title=enc, xlabel=f"cumulative {name} (log scale)", ylabel="val mIoU")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def coverage_curves(stats, path):
    s = stats.groupby(KEYS)[["kernel_coverage", "ball_coverage"]].mean().reset_index()
    encoders = sorted(s["encoder"].unique())
    fig, axes = plt.subplots(2, len(encoders), figsize=(6.5 * len(encoders), 8), squeeze=False)
    fig.suptitle("coverage of the pool by L (label-free, same sigma / delta for every method)")
    for j, enc in enumerate(encoders):
        d = s[(s["encoder"] == enc) & s["label"].map(is_main)]
        for i, (col, name) in enumerate([("kernel_coverage", "generalized coverage (Gaussian, median sigma)"),
                                         ("ball_coverage", "P(C(L, delta)) (ProbCover delta)")]):
            ax = axes[i, j]
            for label, c in d.groupby("label"):
                c = c.sort_values("n_labeled")
                ax.plot(c["n_labeled"], c[col], marker="o", label=label)
            ticks = sorted(d["n_labeled"].unique())
            ax.set_xscale("log", base=2)
            ax.set_xticks(ticks, [str(int(t)) for t in ticks])
            ax.set(title=enc, xlabel="labeled images (log scale)", ylabel=name)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    cfg = load_config()
    out = Path(cfg["out_dir"])
    rep = out / "report"
    rep.mkdir(parents=True, exist_ok=True)
    res, stats = load(out)
    mean, full = aggregate(res)
    table = summary_table(mean, full, stats)
    table.to_csv(rep / "summary.csv", index=False)
    (rep / "summary.md").write_text(to_markdown(table) + "\n")
    main_labels = [l for l in mean["label"].unique() if is_main(l)]
    learning_curves(mean, full, rep / "learning_curves.png", main_labels)
    cost_curves(mean, rep / "cost_curves.png")
    if stats is not None:
        coverage_curves(stats, rep / "coverage_curves.png")
    herding = [l for l in mean["label"].unique() if l.startswith("maxherding")]
    if len(herding) > 1:
        learning_curves(mean, full, rep / "sigma_ablation.png", herding, "MaxHerding: sigma = median x factor")
    (rep / "full_data.json").write_text(json.dumps(full.to_dict(orient="index"), indent=2))
    print(f"[report] {len(table)} encoder x method rows -> {rep}")
    print(to_markdown(table[[c for c in table.columns if c.startswith(("encoder", "method", "mIoU@", "AUBC"))]]))


if __name__ == "__main__":
    main()
