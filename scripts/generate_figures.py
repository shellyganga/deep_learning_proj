"""Generate the two figures used in Section VI of the paper.

Reads ``results/llm_judge_aggregate.csv`` and ``results/retrieval_aggregate.csv``
and emits:

  - results/figures/llm_judge_bars.png
        Grouped bar chart of macro GPT-4o-mini scores by (dataset, method).
  - results/figures/intrinsic_vs_extrinsic.png
        Scatter of intrinsic macro score vs nDCG@10 (averaged across the
        four MTEB embedding models), with an OLS fit and Pearson r in the
        title. TRECCOVID is excluded because no MTEB nDCG@10 was computed
        for it in this study.

Usage:
    python scripts/generate_figures.py
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"

DATASET_ORDER = ["SciFact", "NFCorpus", "FiQA2018", "TRECCOVID"]
METHOD_ORDER = ["fixed", "langchain", "chonkie", "recur"]
METHOD_LABELS = {
    "fixed": "Fixed-size",
    "langchain": "LangChain",
    "chonkie": "Chonkie",
    "recur": "Recursive (ours)",
}
METHOD_COLORS = {
    "fixed": "#bdbdbd",
    "langchain": "#6baed6",
    "chonkie": "#fdae61",
    "recur": "#1b9e77",
}


def _load_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _llm_judge_macros(rows: List[Dict[str, str]]) -> Dict[Tuple[str, str], float]:
    out: Dict[Tuple[str, str], float] = {}
    for r in rows:
        out[(r["dataset"], r["chunking_strategy"])] = float(r["macro_avg"])
    return out


def _retrieval_ndcg_avg(rows: List[Dict[str, str]]) -> Dict[Tuple[str, str], float]:
    """Average nDCG@10 across embedding models for each (dataset, strategy)."""
    buckets: Dict[Tuple[str, str], List[float]] = {}
    for r in rows:
        key = (r["dataset"], r["chunking_strategy"])
        buckets.setdefault(key, []).append(float(r["ndcg@10"]))
    return {k: float(np.mean(v)) for k, v in buckets.items()}


def _plot_bars(macros: Dict[Tuple[str, str], float], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    n_methods = len(METHOD_ORDER)
    width = 0.18
    x = np.arange(len(DATASET_ORDER))

    for i, method in enumerate(METHOD_ORDER):
        heights = [macros.get((ds, method), float("nan")) for ds in DATASET_ORDER]
        offset = (i - (n_methods - 1) / 2) * width
        ax.bar(
            x + offset,
            heights,
            width=width,
            label=METHOD_LABELS[method],
            color=METHOD_COLORS[method],
            edgecolor="black",
            linewidth=0.4,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(DATASET_ORDER)
    ax.set_ylim(1.6, 2.55)
    ax.set_ylabel("GPT-4o-mini macro score")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.28), ncol=4, frameon=False, fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_correlation(
    macros: Dict[Tuple[str, str], float],
    ndcgs: Dict[Tuple[str, str], float],
    out_path: Path,
) -> None:
    paired: List[Tuple[float, float, str, str]] = []
    for (ds, method), macro in macros.items():
        if (ds, method) not in ndcgs:
            continue
        if method == "fixed":
            # Fixed-size baseline isn't in the MTEB extrinsic table.
            continue
        paired.append((macro, ndcgs[(ds, method)], method, ds))

    if not paired:
        raise RuntimeError("No paired (intrinsic, extrinsic) cells found.")

    xs = np.array([p[0] for p in paired])
    ys = np.array([p[1] for p in paired])
    pearson = float(np.corrcoef(xs, ys)[0, 1])

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    method_marker = {"recur": "o", "langchain": "s", "chonkie": "^"}
    for macro, ndcg, method, ds in paired:
        ax.scatter(
            macro,
            ndcg,
            s=70,
            marker=method_marker.get(method, "o"),
            color=METHOD_COLORS.get(method, "#444"),
            edgecolor="black",
            linewidth=0.5,
            label=METHOD_LABELS.get(method, method),
        )

    # OLS fit line
    slope, intercept = np.polyfit(xs, ys, 1)
    grid = np.linspace(xs.min() - 0.05, xs.max() + 0.05, 50)
    ax.plot(grid, slope * grid + intercept, linestyle="--", color="black", alpha=0.5, label="OLS fit")

    # Deduplicate legend
    handles, labels = ax.get_legend_handles_labels()
    seen: Dict[str, object] = {}
    for h, l in zip(handles, labels):
        seen.setdefault(l, h)
    ax.legend(seen.values(), seen.keys(), loc="upper left", fontsize=9, frameon=True)

    ax.set_xlabel("GPT-4o-mini macro score (intrinsic)")
    ax.set_ylabel("nDCG@10 (avg. across embeddings)")
    ax.set_title(
        f"Intrinsic vs. extrinsic agreement (Pearson r = {pearson:.2f}, n = {len(paired)})"
    )
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm-judge", default=str(RESULTS_DIR / "llm_judge_aggregate.csv"))
    parser.add_argument("--retrieval", default=str(RESULTS_DIR / "retrieval_aggregate.csv"))
    parser.add_argument("--out-dir", default=str(FIG_DIR))
    args = parser.parse_args()

    macros = _llm_judge_macros(_load_csv(Path(args.llm_judge)))
    ndcgs = _retrieval_ndcg_avg(_load_csv(Path(args.retrieval)))

    out_dir = Path(args.out_dir)
    _plot_bars(macros, out_dir / "llm_judge_bars.png")
    _plot_correlation(macros, ndcgs, out_dir / "intrinsic_vs_extrinsic.png")
    print(f"Wrote figures to {out_dir}/")


if __name__ == "__main__":
    main()
