"""Paired statistical tests over per-query retrieval scores.

For each (dataset, embedding) cell we compute:
    - paired t-test (recur vs each baseline) on Recall@10, MRR@10, NDCG@10
    - Wilcoxon signed-rank test as a robustness check

We also compute an "any baseline" recur-vs-best-baseline test per dataset
(pooled across embeddings) so the paper can cite a single per-dataset p-value.

Usage:
    python scripts/run_significance_tests.py \
        --per-query-dir results/per_query \
        --out results/significance_tests.csv
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy import stats


METRICS = ["recall@10", "mrr@10", "ndcg@10"]
BASELINES = ["langchain", "chonkie", "primer"]
PIVOT = "recur"


PATTERN = re.compile(r"^(?P<dataset>[^_]+)__(?P<embedding>[^_]+)__(?P<strategy>[^_]+)\.csv$")


def load_per_query(per_query_dir: Path) -> Dict[Tuple[str, str, str], Dict[str, np.ndarray]]:
    out: Dict[Tuple[str, str, str], Dict[str, np.ndarray]] = {}
    for path in per_query_dir.glob("*.csv"):
        m = PATTERN.match(path.name)
        if not m:
            continue
        key = (m["dataset"], m["embedding"], m["strategy"])
        cols: Dict[str, List[float]] = {metric: [] for metric in METRICS}
        with path.open(encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                for metric in METRICS:
                    cols[metric].append(float(row[metric]))
        out[key] = {metric: np.asarray(vals) for metric, vals in cols.items()}
    return out


def paired_tests(
    pivot_scores: np.ndarray,
    baseline_scores: np.ndarray,
) -> Dict[str, float]:
    diff = pivot_scores - baseline_scores
    t_stat, t_p = stats.ttest_rel(pivot_scores, baseline_scores)
    try:
        w_stat, w_p = stats.wilcoxon(pivot_scores, baseline_scores, zero_method="wilcox")
    except ValueError:
        w_stat, w_p = float("nan"), float("nan")
    sd = float(diff.std(ddof=1))
    cohen_d = float(diff.mean()) / sd if sd > 0 else float("nan")
    return {
        "mean_diff": float(diff.mean()),
        "t_stat": float(t_stat),
        "t_p_value": float(t_p),
        "wilcoxon_stat": float(w_stat),
        "wilcoxon_p_value": float(w_p),
        "cohen_d": cohen_d,
    }


def write_results(rows: List[Dict[str, float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-query-dir", default="results/per_query")
    parser.add_argument("--out", default="results/significance_tests.csv")
    args = parser.parse_args()

    data = load_per_query(Path(args.per_query_dir))

    cells: Dict[Tuple[str, str], Dict[str, Dict[str, np.ndarray]]] = {}
    for (dataset, embedding, strategy), metrics in data.items():
        cells.setdefault((dataset, embedding), {})[strategy] = metrics

    rows: List[Dict[str, float]] = []
    for (dataset, embedding), strategies in sorted(cells.items()):
        if PIVOT not in strategies:
            continue
        pivot_metrics = strategies[PIVOT]
        for baseline in BASELINES:
            if baseline not in strategies:
                continue
            baseline_metrics = strategies[baseline]
            for metric in METRICS:
                test = paired_tests(pivot_metrics[metric], baseline_metrics[metric])
                rows.append(
                    {
                        "dataset": dataset,
                        "embedding_model": embedding,
                        "comparison": f"{PIVOT}_vs_{baseline}",
                        "metric": metric,
                        **{k: round(v, 6) if isinstance(v, float) else v for k, v in test.items()},
                    }
                )

    # Per-dataset pooled tests: recur vs best non-recur baseline (langchain)
    # and recur vs primer.
    for dataset in {d for (d, _) in cells.keys()}:
        for baseline in BASELINES:
            pivot_pool: List[float] = []
            base_pool: List[float] = []
            for embedding in {e for (d, e) in cells if d == dataset}:
                strategies = cells.get((dataset, embedding), {})
                if PIVOT in strategies and baseline in strategies:
                    pivot_pool.extend(strategies[PIVOT]["ndcg@10"].tolist())
                    base_pool.extend(strategies[baseline]["ndcg@10"].tolist())
            if not pivot_pool:
                continue
            test = paired_tests(np.asarray(pivot_pool), np.asarray(base_pool))
            rows.append(
                {
                    "dataset": dataset,
                    "embedding_model": "POOLED",
                    "comparison": f"{PIVOT}_vs_{baseline}",
                    "metric": "ndcg@10",
                    **{k: round(v, 6) if isinstance(v, float) else v for k, v in test.items()},
                }
            )

    write_results(rows, Path(args.out))


if __name__ == "__main__":
    main()
