# Evaluating and Improving Chunking Methods for RAG

Companion repository for the paper *"Evaluating and Improving Chunking
Methods for Search and Retrieval-Augmented Generation Systems"*
(`paper_latex_file.tex` / `paper_latex_file.pdf`).

The repo contains:

- the recursive chunker implementation (`chunking_methods/`),
- the BEIR/MTEB chunk-precomputation and retrieval evaluation harnesses
  (`mteb/`, `eval/`),
- the three-axis LLM-as-a-Judge runner with all five prompt templates
  (`eval/llm_judge/`, `prompts/`),
- the figure-generation script (`scripts/generate_figures.py`),
- the aggregate result CSVs and rendered figures used in the paper
  (`results/`).

## Layout

```
chunking_methods/
    recur_chunker.py        # ChunkerTok: token-level recursive boundary selection
    cover_constrained.py    # max_sum_boundaries DP helper used by ChunkerTok

mteb/
    mteb_precompute_chunks.py  # Chunk every BEIR doc once with each chunker

eval/
    chunking_eval.py        # Retrieval metrics (Recall@10, MRR@10, NDCG@10)
    chunk_stats.py          # Chunks/doc, chunk length, redundancy rate
    llm_judge/
        chunk_quality_eval.py  # Coherence / Completeness / Relevance Purity evaluators
        run_llm_judge.py       # Orchestrates the three-axis evaluation per cell

prompts/
    score_coherence.jinja2          # document-level boundary quality
    score_completeness.jinja2       # per-chunk self-containment
    score_relevance_purity.jinja2   # per-chunk topical consistency
    score_split.jinja2              # earlier boundary-pair scorer
    score_segment.jinja2            # self-segment scorer

scripts/
    generate_figures.py     # Build PNG figures from results/ CSVs

results/
    retrieval_aggregate.csv          # Table I: nDCG@10, MRR@10, Recall@10 per (dataset, embedding, strategy)
    llm_judge_aggregate.csv          # Table II: per-axis + macro GPT-4o-mini scores
    intrinsic_extrinsic_correlation.csv  # Pearson r between intrinsic macro and nDCG@10
    redundancy_rate.csv              # Chunk-pair cosine-overlap rate per (dataset, strategy)
    chunk_stats.csv                  # Chunks/doc, mean chunk length (chars), std
    summary.json                     # Run config and aggregate provenance
    figures/
        llm_judge_bars.png           # Figure 1
        intrinsic_vs_extrinsic.png   # Figure 2
```

## Datasets and metrics

Four BEIR datasets:

- **SciFact** (5,183 abstracts; 300 claims)
- **NFCorpus** (3,633 medical docs; 323 queries)
- **FiQA2018** (57,638 financial docs; 648 queries used)
- **TREC-COVID** (171,332 CORD-19 articles; 50 topics)

Metrics computed at document granularity after chunk-level retrieval and
deduplication to parent doc IDs:

- **Recall@10** — fraction of relevant docs retrieved in top 10.
- **MRR@10** — mean reciprocal rank of the first relevant doc within 10.
- **NDCG@10** — normalised discounted cumulative gain at 10.
- **Redundancy Rate (RR)** — `1 / (m(m-1)) * sum_{i!=j} I(cos(c_i, c_j) > 0.9)`.
- **LLM-as-a-Judge (3 axes, 0–3 scale)**:
    - *Coherence* (`prompts/score_coherence.jinja2`)
    - *Completeness* (`prompts/score_completeness.jinja2`)
    - *Relevance Purity* (`prompts/score_relevance_purity.jinja2`)
- **Macro score** — mean of the three per-document axis scores.

## Running the retrieval evaluation

`eval/chunking_eval.py` is the BEIR-style evaluator. It chunks each corpus
once per strategy, embeds chunks with the configured retrieval encoder,
ranks chunks by cosine similarity, deduplicates to document IDs, and
reports Recall@K / MRR@K / NDCG@K.

```bash
python eval/chunking_eval.py --dataset TRECCOVID
```

To use precomputed JSONL chunkings (faster for re-runs):

```bash
python mteb/mteb_precompute_chunks.py \
    --dataset SciFact NFCorpus FiQA2018 TRECCOVID \
    --method recur langchain chonkie \
    --out-dir /path/to/chunks

python eval/chunking_eval.py --dataset TRECCOVID \
    --precomputed recur=/path/to/chunks/TRECCOVID_recur.jsonl \
    --precomputed langchain=/path/to/chunks/TRECCOVID_langchain.jsonl \
    --precomputed chonkie=/path/to/chunks/TRECCOVID_chonkie.jsonl
```

## Running LLM-as-a-Judge

The judge uses **GPT-4o-mini at temperature 0** with a 4-token output cap.
Each chunked document is scored along three axes — Coherence (document
level), Completeness (per chunk), and Relevance Purity (per chunk) — using
their dedicated prompt templates under `prompts/`. The macro score is the
mean of the three per-document axis means.

```bash
# Requires AZURE_OPENAI_API_KEY in the environment.
python eval/llm_judge/run_llm_judge.py \
    --datasets SciFact NFCorpus FiQA2018 TRECCOVID \
    --strategies recur langchain chonkie fixed \
    --n-docs 50
```

Outputs:

- `results/llm_judge_aggregate.csv` — one row per (dataset, strategy)
  with `coherence_mean`, `completeness_mean`, `relevance_purity_mean`,
  `macro_avg`.
- `results/llm_judge_per_doc/<dataset>__<strategy>.csv` — per-document
  scores for downstream analysis.

## Computing chunk statistics

`eval/chunk_stats.py` runs each chunker on a sampled BEIR slice, computes
chunks per document and chunk-length statistics, and computes the
redundancy rate (fraction of within-document chunk pairs whose cosine
similarity exceeds the threshold). It writes `results/chunk_stats.csv`
and `results/redundancy_rate.csv`.

```bash
python eval/chunk_stats.py \
    --datasets SciFact NFCorpus FiQA2018 TRECCOVID \
    --strategies recur langchain chonkie \
    --n-docs 500 --redundancy-threshold 0.9
```

## Regenerating the figures

```bash
pip install -r requirements.txt
python scripts/generate_figures.py
```

This reads `results/llm_judge_aggregate.csv` and
`results/retrieval_aggregate.csv` and writes
`results/figures/llm_judge_bars.png` and
`results/figures/intrinsic_vs_extrinsic.png`. The scatter title reports
the Pearson correlation computed at run time.

## Building the paper

The paper uses `tcolorbox` and `pgfplots`; both are available in any
recent TeXLive distribution. With `tectonic`:

```bash
tectonic paper_latex_file.tex
```

Or with `latexmk`:

```bash
latexmk -pdf paper_latex_file.tex
```

## Result CSV cheatsheet

| File | Granularity | Used in |
| --- | --- | --- |
| `retrieval_aggregate.csv` | (dataset, embedding, strategy) | Table I |
| `llm_judge_aggregate.csv` | (dataset, strategy) | Table II |
| `intrinsic_extrinsic_correlation.csv` | overall | Figure 2 caption |
| `redundancy_rate.csv` | (dataset, strategy) | Table IV |
| `chunk_stats.csv` | (dataset, strategy) | Table IV |
| `summary.json` | run config | provenance |
