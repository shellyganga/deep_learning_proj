import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from chunk_test.chunking_mech.eval.load_mteb_and_eval import (
    build_document,
    load_mteb_dataset,
)

from chunk_eval_llm.chunking_methods.recur_chunker import ChunkerTok
from langchain_text_splitters import RecursiveCharacterTextSplitter
from chonkie import SemanticChunker


DATASETS_DEFAULT = ["SciFact", "NFCorpus", "FiQA2018", "TRECCOVID"]


def iter_docs(dataset_name: str) -> Iterable[Tuple[int, str, str]]:
    data = load_mteb_dataset(dataset_name)
    doc_ids = list(data.corpus.keys())
    for idx, doc_id in enumerate(doc_ids):
        text = build_document(data.corpus[doc_id])
        yield idx, doc_id, text


def chunk_recur(chunker: ChunkerTok, text: str) -> List[str]:
    try:
        res = chunker.get_chunks_embs(text)
    except Exception:
        return []
    chunks = res.get("chunks", [])
    if not isinstance(chunks, list):
        return []
    return [c for c in chunks if isinstance(c, str) and c.strip()]


def chunk_langchain(splitter: RecursiveCharacterTextSplitter, text: str) -> List[str]:
    docs = splitter.create_documents([text])
    return [d.page_content for d in docs if d.page_content.strip()]


def chunk_chonkie(chunker: SemanticChunker, text: str) -> List[str]:
    sem_objs = chunker.chunk(text)
    return [c.text for c in sem_objs if c.text.strip()]


def ensure_nonempty(chunks: List[str], text: str) -> List[str]:
    if chunks:
        return chunks
    if text.strip():
        return [text]
    return []


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute MTEB chunks for eval.")
    parser.add_argument(
        "--dataset",
        action="append",
        dest="datasets",
        help="Dataset name to process (repeatable).",
    )
    parser.add_argument(
        "--method",
        action="append",
        dest="methods",
        help="Chunking method (repeatable): recur | langchain | chonkie",
    )
    parser.add_argument(
        "--out-dir",
        default="/home/shellyschwartz/chunk_eval_llm/mteb_chunks",
        help="Output directory for JSONL chunk files.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional limit of docs per dataset (0 = all).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the last written doc_id in each output file.",
    )
    args = parser.parse_args()

    datasets = args.datasets or DATASETS_DEFAULT
    methods = args.methods or ["recur", "langchain", "chonkie"]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    recur_chunker = ChunkerTok() if "recur" in methods else None
    lc_splitter = None
    if "langchain" in methods:
        lc_splitter = RecursiveCharacterTextSplitter(
            chunk_size=256 * 4,
            chunk_overlap=0,
            add_start_index=True,
        )
    chonkie_chunker = None
    if "chonkie" in methods:
        chonkie_chunker = SemanticChunker(
            embedding_model="intfloat/multilingual-e5-small",
            threshold=0.8,
            chunk_size=256,
            similarity_window=3,
        )

    for dataset_name in datasets:
        for method in methods:
            out_path = out_dir / f"{dataset_name}_{method}.jsonl"
            start_doc_id = 0
            if out_path.exists() and args.resume:
                with out_path.open("rb") as fh:
                    try:
                        fh.seek(-2, os.SEEK_END)
                        while fh.read(1) != b"\n":
                            fh.seek(-2, os.SEEK_CUR)
                    except OSError:
                        fh.seek(0)
                    last_line = fh.readline().decode("utf-8").strip()
                if last_line:
                    try:
                        start_doc_id = json.loads(last_line).get("doc_id", 0) + 1
                    except Exception:
                        start_doc_id = 0
            elif out_path.exists():
                out_path.unlink()

            if method == "recur":
                assert recur_chunker is not None
                chunk_fn = lambda text: chunk_recur(recur_chunker, text)
                algo_name = "chunk_tok"
                config = {"model_name": "intfloat/multilingual-e5-small"}
            elif method == "langchain":
                assert lc_splitter is not None
                chunk_fn = lambda text: chunk_langchain(lc_splitter, text)
                algo_name = "langchain_recursive_char"
                config = {"chunk_size_chars": 1024, "chunk_overlap_chars": 0, "add_start_index": True}
            elif method == "chonkie":
                assert chonkie_chunker is not None
                chunk_fn = lambda text: chunk_chonkie(chonkie_chunker, text)
                algo_name = "chonkie_semantic"
                config = {"embedding_model": "intfloat/multilingual-e5-small", "threshold": 0.8, "chunk_size": 256, "similarity_window": 3}
            else:
                raise ValueError(f"Unsupported method: {method}")

            mode = "a" if out_path.exists() and args.resume else "w"
            with out_path.open(mode, encoding="utf-8") as fh:
                doc_iter = iter_docs(dataset_name)
                for i, (idx, doc_id, text) in enumerate(tqdm(doc_iter, desc=f"{dataset_name} {method}")):
                    if args.limit and args.limit > 0 and i >= args.limit:
                        break
                    if idx < start_doc_id:
                        continue
                    chunks = ensure_nonempty(chunk_fn(text), text)
                    if not chunks:
                        continue
                    rec = {
                        "dataset": dataset_name,
                        "doc_id": idx,
                        "orig_doc_id": doc_id,
                        "algo_name": algo_name,
                        "config": config,
                        "chunks": chunks,
                    }
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
