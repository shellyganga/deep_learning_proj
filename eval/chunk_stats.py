"""Chunk-level statistics: chunk length, chunks per document, redundancy rate.

Implements the three structural metrics referenced in Section VI of the
paper:

  - ``avg_chunks_per_doc``    : mean number of chunks emitted per source
                                 document.
  - ``avg_chunk_length_chars``: mean character length of a chunk, with
                                 ``std_chunk_length_chars`` reported
                                 alongside.
  - ``redundancy_rate``       : fraction of unordered chunk pairs whose
                                 cosine similarity exceeds a threshold
                                 (default $0.9$). Lower is better; high
                                 redundancy inflates the index without
                                 adding retrievable content.

The functions here operate over already-chunked documents and (for
redundancy rate) chunk embeddings. The CLI driver chunks the BEIR corpus
once per (dataset, strategy) cell, embeds the chunks with a single
sentence-transformer, and writes the aggregate CSVs that the paper
references.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RETRIEVAL_OUT = REPO_ROOT / "results" / "retrieval_aggregate.csv"
DEFAULT_REDUNDANCY_OUT = REPO_ROOT / "results" / "redundancy_rate.csv"

DEFAULT_REDUNDANCY_THRESHOLD = 0.9
DEFAULT_REDUNDANCY_EMBEDDING = "intfloat/multilingual-e5-small"
DATASETS = ("SciFact", "NFCorpus", "FiQA2018", "TRECCOVID")
STRATEGIES = ("recur", "langchain", "chonkie")

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# -----------------------------------------------------------------------
# Pure metric functions (no I/O, no model loading).
# -----------------------------------------------------------------------

def chunk_length_stats(chunks: Sequence[str]) -> Dict[str, float]:
    """Mean and std of chunk character length over a single document."""
    if not chunks:
        return {"avg_chunk_length_chars": 0.0, "std_chunk_length_chars": 0.0}
    lengths = np.fromiter((len(c) for c in chunks), dtype=float, count=len(chunks))
    return {
        "avg_chunk_length_chars": float(lengths.mean()),
        "std_chunk_length_chars": float(lengths.std(ddof=0)),
    }


def aggregate_chunk_stats(chunked_docs: Sequence[Sequence[str]]) -> Dict[str, float]:
    """Aggregate ``avg_chunks_per_doc``, ``avg_chunk_length_chars``,
    ``std_chunk_length_chars`` over a corpus of chunked documents."""
    if not chunked_docs:
        return {
            "avg_chunks_per_doc": 0.0,
            "avg_chunk_length_chars": 0.0,
            "std_chunk_length_chars": 0.0,
        }
    n_chunks = np.array([len(c) for c in chunked_docs], dtype=float)
    all_lengths: List[int] = []
    for doc in chunked_docs:
        all_lengths.extend(len(c) for c in doc)
    if not all_lengths:
        return {
            "avg_chunks_per_doc": float(n_chunks.mean()),
            "avg_chunk_length_chars": 0.0,
            "std_chunk_length_chars": 0.0,
        }
    arr = np.asarray(all_lengths, dtype=float)
    return {
        "avg_chunks_per_doc": float(n_chunks.mean()),
        "avg_chunk_length_chars": float(arr.mean()),
        "std_chunk_length_chars": float(arr.std(ddof=0)),
    }


def redundancy_rate(
    embeddings: np.ndarray,
    threshold: float = DEFAULT_REDUNDANCY_THRESHOLD,
) -> float:
    """Fraction of unordered chunk pairs $(i, j)$ with $i < j$ whose
    cosine similarity exceeds ``threshold``.

    ``embeddings`` is an $(N, d)$ array of chunk embeddings, assumed to
    be normalized (or this function will normalize them). Returns $0$ if
    fewer than two chunks are present.
    """
    if embeddings.ndim != 2 or embeddings.shape[0] < 2:
        return 0.0
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    unit = embeddings / norms
    sims = unit @ unit.T
    n = unit.shape[0]
    iu = np.triu_indices(n, k=1)
    pair_sims = sims[iu]
    if pair_sims.size == 0:
        return 0.0
    return float((pair_sims > threshold).mean())


def corpus_redundancy_rate(
    chunked_docs: Sequence[Sequence[str]],
    encoder,
    threshold: float = DEFAULT_REDUNDANCY_THRESHOLD,
) -> float:
    """Redundancy rate aggregated over a corpus.

    For each document with at least two chunks we compute the per-document
    redundancy rate, then return the mean across documents. ``encoder``
    must expose ``encode(list[str], normalize_embeddings=True) -> ndarray``
    (the sentence-transformers convention).
    """
    rates: List[float] = []
    for chunks in chunked_docs:
        if len(chunks) < 2:
            continue
        embs = encoder.encode(list(chunks), normalize_embeddings=True)
        embs = np.asarray(embs, dtype=float)
        rates.append(redundancy_rate(embs, threshold=threshold))
    if not rates:
        return 0.0
    return float(np.mean(rates))


# -----------------------------------------------------------------------
# CLI driver.
# -----------------------------------------------------------------------

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
    raise ValueError(f"Unknown strategy: {strategy}")


def _load_documents(dataset: str, n_docs: int, seed: int) -> List[str]:
    """Sample ``n_docs`` documents from a BEIR-formatted corpus."""
    import datasets  # type: ignore

    dataset_id = {
        "SciFact": "BeIR/scifact",
        "NFCorpus": "BeIR/nfcorpus",
        "FiQA2018": "BeIR/fiqa",
        "TRECCOVID": "BeIR/trec-covid",
    }[dataset]
    ds = datasets.load_dataset(dataset_id, "corpus", split="corpus")
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(ds), size=min(n_docs, len(ds)), replace=False)
    out: List[str] = []
    for idx in indices:
        row = ds[int(idx)]
        title = row.get("title", "") or ""
        text = row.get("text", "") or ""
        out.append(f"{title}\n{text}".strip())
    return out


def _chunk_corpus(strategy: str, docs: Iterable[str]) -> List[List[str]]:
    chunk_fn = _load_chunker(strategy)
    return [list(chunk_fn(d)) for d in docs]


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    parser.add_argument("--strategies", nargs="+", default=list(STRATEGIES))
    parser.add_argument("--n-docs", type=int, default=500)
    parser.add_argument(
        "--redundancy-threshold",
        type=float,
        default=DEFAULT_REDUNDANCY_THRESHOLD,
    )
    parser.add_argument(
        "--redundancy-embedding",
        default=DEFAULT_REDUNDANCY_EMBEDDING,
        help="Sentence-transformer used for the redundancy-rate computation.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--retrieval-out", default=str(DEFAULT_RETRIEVAL_OUT))
    parser.add_argument("--redundancy-out", default=str(DEFAULT_REDUNDANCY_OUT))
    args = parser.parse_args()

    # Load encoder once.
    from sentence_transformers import SentenceTransformer  # type: ignore

    encoder = SentenceTransformer(args.redundancy_embedding)

    chunk_stat_rows: List[Dict[str, object]] = []
    redundancy_rows: List[Dict[str, object]] = []

    for dataset in args.datasets:
        docs = _load_documents(dataset, args.n_docs, args.seed)
        for strategy in args.strategies:
            chunked = _chunk_corpus(strategy, docs)
            stats = aggregate_chunk_stats(chunked)
            chunk_stat_rows.append(
                {
                    "dataset": dataset,
                    "chunking_strategy": strategy,
                    **{k: round(v, 3) for k, v in stats.items()},
                }
            )
            r = corpus_redundancy_rate(chunked, encoder, threshold=args.redundancy_threshold)
            redundancy_rows.append(
                {
                    "dataset": dataset,
                    "chunking_strategy": strategy,
                    "redundancy_rate": round(r, 4),
                    "threshold_cosine": args.redundancy_threshold,
                }
            )

    _write_csv(
        Path(args.retrieval_out).with_name("chunk_stats.csv"),
        ["dataset", "chunking_strategy", "avg_chunks_per_doc",
         "avg_chunk_length_chars", "std_chunk_length_chars"],
        chunk_stat_rows,
    )
    _write_csv(
        Path(args.redundancy_out),
        ["dataset", "chunking_strategy", "redundancy_rate", "threshold_cosine"],
        redundancy_rows,
    )


if __name__ == "__main__":
    main()
