"""LLM-as-a-Judge evaluation of chunk quality.

Implements the three-axis intrinsic evaluation described in Section V.C
of the paper:

  - Coherence        (document-level boundary quality)
  - Completeness     (per-chunk self-containment)
  - Relevance Purity (per-chunk topical consistency)

For each (dataset, chunking_strategy) cell we sample ``n_docs`` documents
from the BEIR corpus, chunk them with the configured chunker, and score
each axis with its dedicated prompt under ``prompts/``.

The judge is invoked via the Azure OpenAI gateway with temperature 0 and
``gpt-4o-mini`` by default. The macro-averaged per-document score is the
mean over the three axes.

Usage:
    python eval/llm_judge/run_llm_judge.py \\
        --datasets SciFact NFCorpus FiQA2018 TRECCOVID \\
        --strategies recur langchain chonkie fixed \\
        --n-docs 50

Requires ``AZURE_OPENAI_API_KEY`` (and optionally ``AZURE_OPENAI_ENDPOINT``)
in the environment.
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
from typing import Callable, Dict, List, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
PROMPT_DIR = REPO_ROOT / "prompts"
DEFAULT_OUT = REPO_ROOT / "results" / "llm_judge_aggregate.csv"
DEFAULT_PER_DOC_DIR = REPO_ROOT / "results" / "llm_judge_per_doc"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.llm_judge.chunk_quality_eval import (  # noqa: E402
    CHUNK_BREAK_MARKER,  # noqa: F401  (re-exported for downstream tools)
    CoherenceEvaluator,
    CompletenessEvaluator,
    RelevancePurityEvaluator,
    _build_default_llm,
)

DATASETS = ["SciFact", "NFCorpus", "FiQA2018", "TRECCOVID"]
STRATEGIES = ["recur", "langchain", "chonkie", "fixed"]


@dataclass
class JudgeConfig:
    model: str = "gpt-4o-mini"
    n_docs: int = 50
    seed: int = 0
    datasets: Sequence[str] = field(default_factory=lambda: list(DATASETS))
    strategies: Sequence[str] = field(default_factory=lambda: list(STRATEGIES))


def _load_chunker(strategy: str) -> Callable[[str], List[str]]:
    if strategy == "recur":
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
    if strategy == "fixed":
        # Fixed-size token windows (~256 tokens) using a simple word-chunker.
        # Approximates BEIR-style 256-token segmentation without pulling in
        # a heavy tokenizer dependency.
        def _fixed(text: str, window: int = 256) -> List[str]:
            words = text.split()
            return [
                " ".join(words[i : i + window])
                for i in range(0, len(words), window)
                if words[i : i + window]
            ]

        return _fixed
    raise ValueError(f"Unknown strategy: {strategy}")


def _load_dataset_documents(dataset: str, n_docs: int, seed: int) -> List[str]:
    """Sample ``n_docs`` documents from a BEIR-formatted corpus."""
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


def _evaluate_cell(
    config: JudgeConfig,
    dataset: str,
    strategy: str,
    coh: CoherenceEvaluator,
    comp: CompletenessEvaluator,
    rp: RelevancePurityEvaluator,
) -> List[Dict[str, float]]:
    """Score one (dataset, strategy) cell and return per-document rows."""
    chunk_fn = _load_chunker(strategy)
    docs = _load_dataset_documents(dataset, config.n_docs, config.seed)

    rows: List[Dict[str, float]] = []
    for doc_idx, doc in enumerate(docs):
        chunks = chunk_fn(doc)
        if len(chunks) < 1:
            continue

        coherence_score = coh.score(chunks)
        completeness_scores = comp.score_all(chunks)
        purity_scores = rp.score_all(chunks)

        completeness_mean = float(statistics.mean(completeness_scores))
        purity_mean = float(statistics.mean(purity_scores))
        macro = (coherence_score + completeness_mean + purity_mean) / 3.0

        rows.append(
            {
                "dataset": dataset,
                "chunking_strategy": strategy,
                "doc_idx": doc_idx,
                "n_chunks": len(chunks),
                "coherence": coherence_score,
                "completeness": completeness_mean,
                "relevance_purity": purity_mean,
                "macro_avg": macro,
            }
        )
    return rows


def _aggregate(rows: List[Dict[str, float]], dataset: str, strategy: str) -> Dict[str, float]:
    if not rows:
        return {
            "dataset": dataset,
            "chunking_strategy": strategy,
            "n_docs": 0,
            "coherence_mean": 0.0,
            "completeness_mean": 0.0,
            "relevance_purity_mean": 0.0,
            "macro_avg": 0.0,
        }
    coh = [r["coherence"] for r in rows]
    com = [r["completeness"] for r in rows]
    rel = [r["relevance_purity"] for r in rows]
    mac = [r["macro_avg"] for r in rows]
    return {
        "dataset": dataset,
        "chunking_strategy": strategy,
        "n_docs": len(rows),
        "coherence_mean": round(float(statistics.mean(coh)), 3),
        "completeness_mean": round(float(statistics.mean(com)), 3),
        "relevance_purity_mean": round(float(statistics.mean(rel)), 3),
        "macro_avg": round(float(statistics.mean(mac)), 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    parser.add_argument("--strategies", nargs="+", default=list(STRATEGIES))
    parser.add_argument("--n-docs", type=int, default=50)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--per-doc-dir", default=str(DEFAULT_PER_DOC_DIR))
    args = parser.parse_args()

    config = JudgeConfig(
        model=args.model,
        n_docs=args.n_docs,
        seed=args.seed,
        datasets=args.datasets,
        strategies=args.strategies,
    )

    # Construct the LLM gateway once, then share it across the three axis
    # evaluators so prompt templates and the network client are loaded once.
    llm = _build_default_llm(config.model)
    coh = CoherenceEvaluator(llm=llm)
    comp = CompletenessEvaluator(llm=llm)
    rp = RelevancePurityEvaluator(llm=llm)

    per_doc_dir = Path(args.per_doc_dir)
    per_doc_dir.mkdir(parents=True, exist_ok=True)
    aggregate_rows: List[Dict[str, float]] = []

    for dataset in config.datasets:
        for strategy in config.strategies:
            rows = _evaluate_cell(config, dataset, strategy, coh, comp, rp)
            cell_path = per_doc_dir / f"{dataset}__{strategy}.csv"
            with cell_path.open("w", encoding="utf-8", newline="") as fh:
                writer = csv.DictWriter(
                    fh,
                    fieldnames=[
                        "dataset",
                        "chunking_strategy",
                        "doc_idx",
                        "n_chunks",
                        "coherence",
                        "completeness",
                        "relevance_purity",
                        "macro_avg",
                    ],
                    extrasaction="ignore",
                )
                writer.writeheader()
                writer.writerows(rows)

            aggregate_rows.append(_aggregate(rows, dataset, strategy))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(aggregate_rows[0].keys())
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(aggregate_rows)

    summary = {
        "model": config.model,
        "temperature": 0,
        "max_tokens": 4,
        "n_docs": config.n_docs,
        "datasets": list(config.datasets),
        "strategies": list(config.strategies),
        "axes": ["coherence", "completeness", "relevance_purity"],
        "prompts": {
            "coherence": str((PROMPT_DIR / "score_coherence.jinja2").relative_to(REPO_ROOT)),
            "completeness": str((PROMPT_DIR / "score_completeness.jinja2").relative_to(REPO_ROOT)),
            "relevance_purity": str(
                (PROMPT_DIR / "score_relevance_purity.jinja2").relative_to(REPO_ROOT)
            ),
        },
    }
    (out_path.parent / "llm_judge_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
