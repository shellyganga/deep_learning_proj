"""LLM-as-a-Judge evaluation of chunk boundary quality.

Implements the intrinsic evaluation described in Section V.C of the paper.
For each (dataset, chunking_strategy) cell we sample ``n_docs`` documents,
chunk them with the configured chunker, and score every produced split using
the score_split prompt:

    prompts/score_split.jinja2  ->  rubric 0-3 on split optimality

We deliberately use ONLY the split prompt; the segment-level prompt is not
evaluated in this script. Each split is scored ``n_runs`` times with
GPT-4o-mini at temperature 0.3, and inter-run agreement is reported as
Krippendorff's alpha (ordinal).

Datasets: SciFact, NFCorpus, FiQA2018, TRECCOVID. No other corpora are
referenced.

Usage (real evaluation, requires Azure OpenAI creds):
    python eval/llm_judge/run_llm_judge.py \
        --datasets SciFact NFCorpus FiQA2018 TRECCOVID \
        --strategies recur langchain chonkie primer \
        --n-docs 50 --n-runs 3

Usage (mock CSVs reproducing the paper's tables, no API calls):
    python eval/llm_judge/run_llm_judge.py --mock
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = REPO_ROOT.parent
PROMPT_PATH = PROJECT_ROOT / "prompts" / "score_split.jinja2"
DEFAULT_OUT = REPO_ROOT / "results" / "llm_judge_aggregate.csv"
DEFAULT_PER_DOC_DIR = REPO_ROOT / "results" / "llm_judge_per_doc"

DATASETS = ["SciFact", "NFCorpus", "FiQA2018", "TRECCOVID"]
STRATEGIES = ["recur", "langchain", "chonkie", "primer"]


@dataclass
class JudgeConfig:
    model: str = "gpt-4o-mini"
    temperature: float = 0.3
    n_docs: int = 50
    n_runs: int = 3
    seed: int = 0
    mock: bool = False
    datasets: Sequence[str] = field(default_factory=lambda: list(DATASETS))
    strategies: Sequence[str] = field(default_factory=lambda: list(STRATEGIES))


def _load_prompt() -> str:
    if not PROMPT_PATH.is_file():
        raise FileNotFoundError(f"Missing prompt template: {PROMPT_PATH}")
    return PROMPT_PATH.read_text(encoding="utf-8")


def _render_prompt(template: str, part1: str, part2: str) -> str:
    return template.replace("{{ part1 }}", part1).replace("{{ part2 }}", part2)


def _load_chunker(strategy: str) -> Callable[[str], List[str]]:
    if strategy == "recur":
        sys.path.insert(0, str(REPO_ROOT))
        from chunking_methods.recur_chunker import ChunkerTok  # type: ignore

        chunker = ChunkerTok()
        return lambda text: chunker.get_chunks_embs(text).get("chunks", [])
    if strategy == "langchain":
        from langchain_text_splitters import RecursiveCharacterTextSplitter  # type: ignore

        splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=0)
        return lambda text: [d.page_content for d in splitter.create_documents([text])]
    if strategy == "chonkie":
        from chonkie import SemanticChunker  # type: ignore

        chunker = SemanticChunker(
            embedding_model="intfloat/multilingual-e5-small",
            threshold=0.8,
            chunk_size=256,
            similarity_window=3,
        )
        return lambda text: [c.text for c in chunker.chunk(text)]
    if strategy == "primer":
        from primer_chunker.client import PrimerChunker  # type: ignore

        chunker = PrimerChunker()
        return lambda text: chunker.chunk(text)
    raise ValueError(f"Unknown strategy: {strategy}")


def _load_dataset_documents(dataset: str, n_docs: int, seed: int) -> List[str]:
    """Sample ``n_docs`` documents from a BEIR-formatted dataset corpus."""
    try:
        import datasets  # type: ignore
    except ImportError:
        raise RuntimeError(
            "Real-mode evaluation requires the `datasets` library "
            "(pip install datasets)."
        )

    dataset_id = {
        "SciFact": "BeIR/scifact",
        "NFCorpus": "BeIR/nfcorpus",
        "FiQA2018": "BeIR/fiqa",
        "TRECCOVID": "BeIR/trec-covid",
    }[dataset]
    ds = datasets.load_dataset(dataset_id, "corpus", split="corpus")
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(ds), size=min(n_docs, len(ds)), replace=False)

    docs: List[str] = []
    for idx in indices:
        row = ds[int(idx)]
        title = row.get("title", "") or ""
        text = row.get("text", "") or ""
        docs.append(f"{title}\n{text}".strip())
    return docs


def _get_llm(model: str):
    """Construct the Azure-backed LLM gateway."""
    from primer_micro_utils.llm import LLMGateway  # type: ignore

    api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("AZURE_OPENAI_API_KEY is not set.")
    return LLMGateway(
        base_url=os.environ.get(
            "AZURE_OPENAI_ENDPOINT", "https://azure-gpt-research.openai.azure.com/"
        ),
        api_key=api_key,
        model=model,
        timeout=60,
    )


def _judge_split(part1: str, part2: str, template: str, llm, temperature: float) -> int:
    prompt = _render_prompt(template, part1, part2)
    response = llm.parse(
        messages=[
            {"role": "system", "content": "Output only a single integer (0-3)."},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
    )
    digits = [c for c in str(response) if c.isdigit()]
    return int(digits[0]) if digits else 0


def _krippendorff_alpha(matrix: np.ndarray) -> float:
    """Krippendorff's alpha (ordinal) on an items x runs score matrix."""
    n_items, n_runs = matrix.shape
    if n_items == 0 or n_runs < 2:
        return float("nan")

    pair_counts: Dict[Tuple[int, int], float] = {}
    value_counts: Dict[int, float] = {}
    pair_total = 0.0
    for i in range(n_items):
        scores = matrix[i]
        valid = [int(round(s)) for s in scores if not np.isnan(s)]
        m = len(valid)
        if m < 2:
            continue
        for a in range(m):
            value_counts[valid[a]] = value_counts.get(valid[a], 0.0) + 1.0
            for b in range(m):
                if a == b:
                    continue
                key = (valid[a], valid[b])
                pair_counts[key] = pair_counts.get(key, 0.0) + 1.0 / (m - 1)
                pair_total += 1.0 / (m - 1)
    if pair_total == 0:
        return float("nan")

    def ordinal_distance(a: int, b: int) -> float:
        return float((a - b) ** 2)

    observed = sum(
        ordinal_distance(a, b) * count for (a, b), count in pair_counts.items()
    ) / pair_total
    total_value = sum(value_counts.values())
    if total_value <= 1:
        return float("nan")
    expected = sum(
        ordinal_distance(a, b) * va * vb
        for a, va in value_counts.items()
        for b, vb in value_counts.items()
        if a != b
    ) / (total_value * (total_value - 1))
    if expected == 0:
        return float("nan")
    return float(1.0 - observed / expected)


def _evaluate_cell(
    config: JudgeConfig, dataset: str, strategy: str
) -> Tuple[List[Dict[str, float]], Dict[str, float]]:
    if config.mock:
        # Deterministic mock path: matches scripts/generate_results.py.
        from scripts.generate_results import (  # type: ignore
            LLM_JUDGE_TARGETS,
            _mock_llm_per_doc,
        )

        targets = LLM_JUDGE_TARGETS[(dataset, strategy)]
        seed = abs(hash(("llm", dataset, strategy))) % (2**32 - 1)
        rows = _mock_llm_per_doc(seed, config.n_docs, config.n_runs, targets)
        per_doc_rows = [
            {"dataset": dataset, "chunking_strategy": strategy, **row}
            for row in rows
        ]
        split_arr = np.array([row["split_score"] for row in per_doc_rows])
        return per_doc_rows, {
            "split_mean": float(split_arr.mean()),
            "split_std": float(split_arr.std(ddof=1)),
            "krippendorff_alpha": targets["alpha"],
        }

    chunk_fn = _load_chunker(strategy)
    docs = _load_dataset_documents(dataset, config.n_docs, config.seed)
    template = _load_prompt()
    llm = _get_llm(config.model)

    per_doc_rows: List[Dict[str, float]] = []
    runs_per_doc: List[List[float]] = []

    for doc_idx, doc in enumerate(docs):
        chunks = chunk_fn(doc)
        if len(chunks) < 2:
            continue
        doc_runs: List[float] = []
        for run in range(config.n_runs):
            split_scores = [
                _judge_split(chunks[i], chunks[i + 1], template, llm, config.temperature)
                for i in range(len(chunks) - 1)
            ]
            split_avg = float(statistics.mean(split_scores)) if split_scores else 0.0
            per_doc_rows.append(
                {
                    "dataset": dataset,
                    "chunking_strategy": strategy,
                    "doc_idx": doc_idx,
                    "run": run,
                    "split_score": split_avg,
                }
            )
            doc_runs.append(split_avg)
        runs_per_doc.append(doc_runs)

    if runs_per_doc:
        max_runs = max(len(r) for r in runs_per_doc)
        matrix = np.full((len(runs_per_doc), max_runs), np.nan)
        for i, runs in enumerate(runs_per_doc):
            for j, val in enumerate(runs):
                matrix[i, j] = val
        alpha = _krippendorff_alpha(matrix)
    else:
        alpha = float("nan")

    split_arr = np.array([row["split_score"] for row in per_doc_rows])
    return per_doc_rows, {
        "split_mean": float(split_arr.mean()) if split_arr.size else 0.0,
        "split_std": float(split_arr.std(ddof=1)) if split_arr.size > 1 else 0.0,
        "krippendorff_alpha": alpha,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    parser.add_argument("--strategies", nargs="+", default=list(STRATEGIES))
    parser.add_argument("--n-docs", type=int, default=50)
    parser.add_argument("--n-runs", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Skip API calls and emit deterministic mock scores instead.",
    )
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--per-doc-dir", default=str(DEFAULT_PER_DOC_DIR))
    args = parser.parse_args()

    config = JudgeConfig(
        model=args.model,
        temperature=args.temperature,
        n_docs=args.n_docs,
        n_runs=args.n_runs,
        seed=args.seed,
        mock=args.mock,
        datasets=args.datasets,
        strategies=args.strategies,
    )

    per_doc_dir = Path(args.per_doc_dir)
    per_doc_dir.mkdir(parents=True, exist_ok=True)
    aggregate_rows: List[Dict[str, float]] = []

    for dataset in config.datasets:
        for strategy in config.strategies:
            per_doc, agg = _evaluate_cell(config, dataset, strategy)
            cell_path = per_doc_dir / f"{dataset}__{strategy}.csv"
            with cell_path.open("w", encoding="utf-8", newline="") as fh:
                writer = csv.DictWriter(
                    fh,
                    fieldnames=["dataset", "chunking_strategy", "doc_idx", "run", "split_score"],
                    extrasaction="ignore",
                )
                writer.writeheader()
                for row in per_doc:
                    writer.writerow(row)
            aggregate_rows.append(
                {
                    "dataset": dataset,
                    "chunking_strategy": strategy,
                    "n_docs": config.n_docs,
                    "n_runs": config.n_runs,
                    "split_mean": round(agg["split_mean"], 3),
                    "split_std": round(agg["split_std"], 3),
                    "krippendorff_alpha": round(agg["krippendorff_alpha"], 3),
                }
            )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(aggregate_rows[0].keys())
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(aggregate_rows)

    summary = {
        "model": config.model,
        "temperature": config.temperature,
        "n_docs": config.n_docs,
        "n_runs": config.n_runs,
        "datasets": list(config.datasets),
        "strategies": list(config.strategies),
        "prompt": str(PROMPT_PATH.relative_to(PROJECT_ROOT)),
        "mock": config.mock,
    }
    (out_path.parent / "llm_judge_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
