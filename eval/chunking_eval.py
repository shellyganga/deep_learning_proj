import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from tqdm import tqdm
import pandas as pd
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import datasets
from mteb.tasks import Retrieval

# Import the chunking strategies
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))  # /home/shellyschwartz/chunk_test/chunking_mech/eval
parent_dir = os.path.dirname(current_dir)  # /home/shellyschwartz/chunk_test/chunking_mech
grandparent_dir = os.path.dirname(parent_dir)  # /home/shellyschwartz/chunk_test
great_grandparent_dir = os.path.dirname(grandparent_dir)  # /home/shellyschwartz

sys.path.insert(0, great_grandparent_dir)

from chunk_test.chunking_mech.chunking_strategies.dp_chunking import SimpleDPChunker
from chunk_test.chunking_mech.chunking_strategies.kmod_chunking import KMeansSemanticChunker
from chunk_test.chunking_mech.chunking_strategies.tok_dp_chunking import TokenDPChunker
from chunk_test.chunking_mech.chunking_strategies.par_chunker import ParagraphBasedChunker
from chunk_test.chunking_mech.chunking_strategies.recursive_chunker import ChunkerTok
# from ..chunking_strategies.dp_chunking import SimpleDPChunker
# from ..chunking_strategies.kmod_chunking import KMeansSemanticChunker
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

@dataclass
class ChunkResult:
    """Container for chunk with metadata"""
    text: str
    doc_id: str
    chunk_idx: int
    embedding: np.ndarray = None
    metadata: Dict[str, Any] = None


class ChunkingEvaluator:
    """Evaluation framework for semantic chunking strategies across multiple embedding models and datasets"""

    def __init__(
        self,
        batch_size: int = 32,
        precomputed_chunks: Optional[Dict[str, str]] = None,
        precomputed_doc_id_mode: str = "index",
        ignore_precomputed_dataset: bool = True,
    ):
        self.batch_size = batch_size
        self.precomputed_chunks = precomputed_chunks or {}
        self.precomputed_doc_id_mode = precomputed_doc_id_mode
        self.ignore_precomputed_dataset = ignore_precomputed_dataset
        self._precomputed_cache: Dict[str, Dict[Any, List[str]]] = {}

        # Define embedding models
        self.embedding_models = {
            'AML12v2': 'sentence-transformers/all-MiniLM-L12-v2',
            'ME5S': 'intfloat/multilingual-e5-small',
            'MQML6Cv1': 'sentence-transformers/all-MiniLM-L6-v2',
            'E5Sv2': 'intfloat/e5-small-v2',
           #'J3': 'jinaai/jina-embeddings-v3'
        }

        # Initialize chunking strategies with their embedding models
        self.chunking_strategies = {}

        # Define dataset configurations
        self.dataset_configs = {
            # MTEB datasets
            'SciFact': {
                'type': 'mteb',
                'path': 'mteb/scifact',
                'revision': '0228b52cf27578f30900b9e5271d331663a030d7',
                'split': 'test'
            },
            'NFCorpus': {
                'type': 'mteb',
                'path': 'mteb/nfcorpus',
                'revision': 'ec0fa4fe99da2ff19ca1214b7966684033a58814',
                'split': 'test'
            },
            'QuoraRetrieval': {
                'type': 'mteb',
                'path': 'mteb/quora',
                'revision': 'e4e08e0b7dbe3c8700f0daef558ff32256715259',
                'split': 'test'
            },
            'FiQA2018': {
                'type': 'mteb',
                'path': 'mteb/fiqa',
                'revision': '27a168819829fe9bcd655c2df245fb19452e8e06',
                'split': 'test'
            },
            'TRECCOVID': {
                'type': 'mteb',
                'path': 'mteb/trec-covid',
                'revision': 'bb9466bac8153a0349341eb1b22e06409e78ef4e',
                'split': 'test'
            },
            # HuggingFace datasets (LEMB)
            'LEMBWikimQARetrieval': {
                'type': 'hf',
                'path': 'dwzhu/LongEmbed',
                'revision': '10039a580487dacecf79db69166e17ace3ede392',
                'name': '2wikimqa',
                'split': 'test'
            },
            'LEMBSummScreenFDRetrieval': {
                'type': 'hf',
                'path': 'dwzhu/LongEmbed',
                'revision': '10039a580487dacecf79db69166e17ace3ede392',
                'name': 'summ_screen_fd',
                'split': 'test'
            },
            'LEMBQMSumRetrieval': {
                'type': 'hf',
                'path': 'dwzhu/LongEmbed',
                'revision': '10039a580487dacecf79db69166e17ace3ede392',
                'name': 'qmsum',
                'split': 'test'
            },
            'LEMBNeedleRetrieval': {
                'type': 'hf',
                'path': 'dwzhu/LongEmbed',
                'revision': '6e346642246bfb4928c560ee08640dc84d074e8c',
                'name': 'needle',
                'split': 'test_1024'
            },
            'LEMBPasskeyRetrieval': {
                'type': 'hf',
                'path': 'dwzhu/LongEmbed',
                'revision': '6e346642246bfb4928c560ee08640dc84d074e8c',
                'name': 'passkey',
                'split': 'test_1024'
            },
            'NarrativeQARetrieval': {
                'type': 'hf',
                'path': 'narrativeqa',
                'revision': '2e643e7363944af1c33a652d1c87320d0871c4e4',
                'split': 'test'
            }
        }

        # Metrics to track
        self.metrics = ['recall@1', 'recall@5', 'recall@10', 'mrr@10', 'ndcg@10']

    def initialize_chunking_strategies(self, model_name: str):
        """Initialize chunking strategies with the specified embedding model"""
        # Use the same embedding model for chunking that we'll use for retrieval
        # This ensures consistency in the semantic space

        # Precomputed chunking strategies (skip chunking, use JSONL)
        for strategy_name in self.precomputed_chunks:
            self.chunking_strategies[strategy_name] = "precomputed"

        if self.precomputed_chunks:
            return

        # DP Chunking with separation (SimpleDPChunker API)
        # self.chunking_strategies['dp_chunking'] = SimpleDPChunker(
        #     model_name=model_name,
        #     max_tokens=256,
        #     min_tokens=50,
        #     cohesiveness_weight=0.6,
        #     separation_weight=0.4,
        #     embedding_context=1
        # )
        # #
        # # # KMeans Chunking with positional encoding
        # self.chunking_strategies['kmod_chunking'] = KMeansSemanticChunker(
        #     model_name=model_name,
        #     max_tokens=256,
        #     embedding_context=1,
        #     max_k=6,
        #     position_weight=0.4,
        #     position_method="encoding"  # Only use encoding method as specified
        # )
        #
        # self.chunking_strategies['paragraph_chunking'] = ParagraphBasedChunker(
        #     model_name=model_name,
        #     max_token_size=256,  # As specified in original code
        #     min_line_break_for_paragraph=4,  # Original default
        #     fallback_line_break=2  # Original fallback
        # )

        self.chunking_strategies['recur_chunk'] = ChunkerTok()

        # self.chunking_strategies['tok_dp_chunking'] = TokenDPChunker(
        #     model_name=model_name,
        #     w_c=.4,
        #     w_s=.6,
        #     block_size=10,
        #     min_tokens_per_chunk=50,
        #     max_tokens_per_chunk=512,
        #     length_weighted=False,
        #     algo_name="token_dp_block",
        #     boundary_bonus=0.2,
        #     boundary_penalty=0.1
        # )



    def load_dataset(self, dataset_name: str) -> Tuple[Dict, Dict, Dict]:
        """Load a dataset based on its configuration"""
        config = self.dataset_configs.get(dataset_name)
        if not config:
            print(f"Dataset {dataset_name} not found in configurations")
            return None, None, None

        if config['type'] == 'mteb':
            return self._load_mteb_dataset(dataset_name, config)
        elif config['type'] == 'hf':
            return self._load_hf_dataset(dataset_name, config)
        else:
            print(f"Unknown dataset type: {config['type']}")
            return None, None, None

    def _load_mteb_dataset(self, dataset_name: str, config: Dict) -> Tuple[Dict, Dict, Dict]:
        """Load MTEB dataset"""
        try:
            # Try to load from MTEB tasks
            task = getattr(Retrieval, dataset_name)()
            task.load_data()

            split = config['split']
            corpus = task.corpus[split]
            queries = task.queries[split]
            relevant_docs = task.relevant_docs[split]

            return corpus, queries, relevant_docs
        except Exception as e:
            print(f"Error loading MTEB dataset {dataset_name}: {e}")
            # Fallback: load cached HF dataset configs directly.
            try:
                path = config["path"]
                revision = config.get("revision")

                corpus_ds = datasets.load_dataset(path, "corpus", revision=revision, split="corpus")
                queries_ds = datasets.load_dataset(path, "queries", revision=revision, split="queries")
                qrels_ds = datasets.load_dataset(path, "default", revision=revision, split=config["split"])

                corpus = {
                    row["_id"]: {
                        "title": row.get("title", ""),
                        "text": row.get("text", ""),
                    }
                    for row in corpus_ds
                }
                queries = {row["_id"]: row["text"] for row in queries_ds}

                relevant_docs: Dict[str, Dict[str, int]] = {}
                for row in qrels_ds:
                    qid = row.get("query-id") or row.get("query_id") or row.get("qid")
                    did = row.get("corpus-id") or row.get("corpus_id") or row.get("doc_id")
                    if qid is None or did is None:
                        continue
                    score = row.get("score", 1)
                    relevant_docs.setdefault(str(qid), {})[str(did)] = int(score)

                return corpus, queries, relevant_docs
            except Exception as e2:
                print(f"Fallback HF load failed for {dataset_name}: {e2}")
                return None, None, None

    def _load_hf_dataset(self, dataset_name: str, config: Dict) -> Tuple[Dict, Dict, Dict]:
        """Load HuggingFace dataset"""
        try:
            dataset_dict = {
                'path': config['path'],
                'revision': config.get('revision'),
            }
            if 'name' in config:
                dataset_dict['name'] = config['name']

            # Load different components
            if 'LEMB' in dataset_name:
                query_list = datasets.load_dataset(**dataset_dict)["queries"]
                corpus_list = datasets.load_dataset(**dataset_dict)["corpus"]
                qrels_list = datasets.load_dataset(**dataset_dict)["qrels"]

                # Special handling for datasets with context_length splits
                if dataset_name in ['LEMBNeedleRetrieval', 'LEMBPasskeyRetrieval']:
                    context_length = int(config['split'].split('_')[1])
                    query_list = query_list.filter(lambda x: x["context_length"] == context_length)
                    corpus_list = corpus_list.filter(lambda x: x["context_length"] == context_length)
                    qrels_list = qrels_list.filter(lambda x: x["context_length"] == context_length)

                queries = {row["qid"]: row["text"] for row in query_list}
                corpus = {row["doc_id"]: {"text": row["text"]} for row in corpus_list}
                relevant_docs = {row["qid"]: {row["doc_id"]: 1} for row in qrels_list}
            else:
                # Handle other HF datasets
                dataset = datasets.load_dataset(**dataset_dict)
                split = config['split']

                # Extract corpus, queries, and relevant docs based on dataset structure
                corpus = {}
                queries = {}
                relevant_docs = {}

            return corpus, queries, relevant_docs

        except Exception as e:
            print(f"Error loading HuggingFace dataset {dataset_name}: {e}")
            return None, None, None

    def load_embedding_model(self, model_name: str):
        """Load a specific embedding model"""
        model_ref = self.embedding_models[model_name]
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Jina/Nomic models → load with Hugging Face (requires trust_remote_code for v3)
        if "jinaai/" in model_ref or "nomic-ai/" in model_ref:
            tokenizer = AutoTokenizer.from_pretrained(model_ref, trust_remote_code=True)
            model = AutoModel.from_pretrained(model_ref, trust_remote_code=True).to(device)
            model.eval()

            class HFEncoder:
                def __init__(self, model, tokenizer):
                    self.model = model
                    self.tokenizer = tokenizer
                    self.device = model.device

                def encode(self, sentences, batch_size=32, convert_to_numpy=True,
                           normalize_embeddings=True, show_progress_bar=False):
                    if isinstance(sentences, str):
                        sentences = [sentences]
                    all_embeddings = []
                    for i in range(0, len(sentences), batch_size):
                        batch = sentences[i:i + batch_size]
                        inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(
                            self.device)
                        with torch.no_grad():
                            outputs = self.model(**inputs)
                            # mean pooling with attention mask
                            token_embs = outputs.last_hidden_state
                            mask = inputs["attention_mask"].unsqueeze(-1).expand(token_embs.size()).float()
                            emb = (token_embs * mask).sum(1) / mask.sum(1)
                            if normalize_embeddings:
                                emb = F.normalize(emb, p=2, dim=1)
                            if convert_to_numpy:
                                emb = emb.cpu().numpy()
                        all_embeddings.append(emb)
                    return np.vstack(all_embeddings)

            return HFEncoder(model, tokenizer)

        # Fallback: classic SBERT
        else:
            from sentence_transformers import SentenceTransformer
            return SentenceTransformer(model_ref)

    def apply_chunking_strategy(
        self,
        document: str,
        doc_id: str,
        strategy_name: str,
        doc_index: Optional[int] = None,
    ) -> List[ChunkResult]:
        """Apply the specified chunking strategy to a document"""
        if strategy_name in self.precomputed_chunks:
            chunks = self._get_precomputed_chunks(strategy_name, doc_id, doc_index)
            results = []
            for idx, chunk_text in enumerate(chunks):
                results.append(
                    ChunkResult(
                        text=chunk_text,
                        doc_id=doc_id,
                        chunk_idx=idx,
                        embedding=None,
                        metadata={"source": "precomputed"}
                    )
                )
            return results

        chunker = self.chunking_strategies[strategy_name]

        if strategy_name == 'dp_chunking':
            chunks = chunker.chunk_text(document)
            results = []
            for idx, ch in enumerate(chunks):
                chunk_text = ch.get('text', '')
                results.append(ChunkResult(
                    text=chunk_text,
                    doc_id=doc_id,
                    chunk_idx=idx,
                    embedding=None,
                    metadata={
                        'cohesiveness': ch.get('cohesiveness', 0.0),
                        'separation_score': ch.get('separation', 0.0),
                        'combined_score': ch.get('score', 0.0),
                        'token_count': ch.get('token_count', 0),
                        'n_sentences': ch.get('num_sentences', 0)
                    }
                ))
            return results

        elif strategy_name == 'kmod_chunking':
            chunks = chunker.chunk_text(document)
            results = []
            for idx, chunk in enumerate(chunks):
                # Concatenate all sentence texts in the chunk
                chunk_text = ' '.join([sent.text for sent in chunk.sentences])
                # Average the embeddings of sentences in the chunk
                chunk_embedding = np.mean([sent.embedding for sent in chunk.sentences], axis=0)
                results.append(ChunkResult(
                    text=chunk_text,
                    doc_id=doc_id,
                    chunk_idx=idx,
                    embedding=chunk_embedding,
                    metadata={
                        'token_count': chunk.token_count,
                        'n_sentences': len(chunk.sentences)
                    }
                ))
            return results
        elif strategy_name == 'tok_dp_chunking':

            chunks, chunk_embs, avg_score, chunk_ends = chunker.segment(
                document
            )

            results = []
            for idx, chunk in enumerate(chunks):
                # Extract the text directly from chunk dictionary
                results.append(ChunkResult(
                    text=chunk,
                    doc_id=doc_id,
                    chunk_idx=idx,
                    embedding=None,
                    metadata={
                        'score': avg_score
                    }
                ))
            return results
        elif strategy_name == 'paragraph_chunking':
            chunks = chunker.chunk_text(document)
            results = []

            for idx, chunk in enumerate(chunks):
                # Extract data from chunk dictionary
                chunk_text = chunk['text']
                chunk_embedding = chunk.get('embedding')

                # If embedding not computed, compute it now
                if chunk_embedding is None:
                    chunk_embedding = chunker.model.encode([chunk_text])[0]

                results.append(ChunkResult(
                    text=chunk_text,
                    doc_id=doc_id,
                    chunk_idx=idx,
                    embedding=chunk_embedding,
                    metadata={
                        'token_count': chunk['token_count'],
                        'n_sentences': chunk['num_sentences'],
                        'start_idx': chunk['start_idx'],
                        'end_idx': chunk['end_idx'],
                        'sentence_spans': chunk['sentence_spans']
                    }
                ))
            return results
        elif strategy_name == 'recur_chunk':

            result = chunker.get_chunks_embs(document)


            chunks = result.get('chunks', [])
            chunk_ends = result.get('chunk_ends', [])
            algo_name = result.get('algo_name', 'recur_chunk')

            results = []
            for idx, chunk_text in enumerate(chunks):
                try:
                    num_tokens = len(chunker.tokenizer(chunk_text)["input_ids"])
                except Exception as e:
                    print("\n[CRASH DETECTED DURING TOKENIZATION]")
                    print(f"doc_id={doc_id} | chunk_idx={idx}")
                    print(f"Full chunk_ends array: {chunk_ends}")
                    print(f"Error message: {e}")  # this contains "(673 > 512)"
                    raise  # let it crash afterward so you see the full traceback

                metadata = {
                    'chunk_end': chunk_ends[idx] if idx < len(chunk_ends) else None,
                    'chunk_length': len(chunk_text),
                    'num_chunks': len(chunks),
                    'algo_name': algo_name
                }
                results.append(
                    ChunkResult(
                        text=chunk_text,
                        doc_id=doc_id,
                        chunk_idx=idx,
                        embedding=None,  # embeddings computed later
                        metadata=metadata
                    )
                )

            return results

            # result = chunker.get_chunks_embs(document)
            # # print(len(document))
            # # print("num_tokens:", len(chunker.tokenizer(document)["input_ids"]))
            # chunks = result.get('chunks', [])
            # result = chunker.get_chunks_embs(document)
            # chunks = result['chunks']
            # print("len(chunk_texts):", len(chunks))
            # if len(chunks) == 0:
            #     "NOPISHFOHDFOPIDSHFHH____________________________________________"
            # chunk_embs = result.get('embs_chunks', [])
            # chunk_ends = result.get('chunk_ends', [])
            #
            # results = []
            # for idx, chunk_text in enumerate(chunks):
            #     embedding = None
            #     if idx < len(chunk_embs):
            #         embedding = chunk_embs[idx]
            #     metadata = {
            #         'chunk_end': chunk_ends[idx] if idx < len(chunk_ends) else None,
            #         'chunk_length': len(chunk_text),
            #         'num_chunks': len(chunks),
            #         'algo_name': result.get('algo_name', 'recur_chunk')
            #     }
            #     results.append(ChunkResult(
            #         text=chunk_text,
            #         doc_id=doc_id,
            #         chunk_idx=idx,
            #         embedding=embedding,
            #         metadata=metadata
            #     ))
            # return results


    def construct_document(self, doc_dict: Dict) -> str:
        """Construct document text from dictionary, handling title if present"""
        if isinstance(doc_dict, str):
            return doc_dict
        elif 'title' in doc_dict and doc_dict['title']:
            return f"{doc_dict['title']} {doc_dict['text']}"
        else:
            return doc_dict['text']

    def embed_chunks_from_results(self, chunks: List[ChunkResult], model) -> np.ndarray:
        """Extract embeddings from ChunkResult objects or compute if needed"""
        if chunks and chunks[0].embedding is not None:
            # Use precomputed embeddings from chunking
            embeddings = np.vstack([chunk.embedding for chunk in chunks])
        else:
            # Compute embeddings if not available
            texts = [chunk.text for chunk in chunks]
            embeddings = model.encode(texts, batch_size=self.batch_size, show_progress_bar=False)
        return embeddings

    def embed_queries(self, queries: List[str], model) -> np.ndarray:
        """Embed queries using the given model"""
        return model.encode(queries, batch_size=self.batch_size, show_progress_bar=False)

    def compute_retrieval_metrics(self,
                                  query_embeddings: np.ndarray,
                                  chunk_embeddings: np.ndarray,
                                  chunk_doc_ids: List[str],
                                  relevant_docs: Dict[int, List[str]],
                                  k_values: List[int] = [1, 5, 10]) -> Dict[str, float]:
        """Compute retrieval metrics for chunked documents"""
        # Compute similarity scores
        similarities = cosine_similarity(query_embeddings, chunk_embeddings)

        metrics = {}

        for k in k_values:
            recalls = []
            mrrs = []
            ndcgs = []

            for q_idx in range(len(query_embeddings)):
                # Get top-k chunks for this query
                chunk_scores = similarities[q_idx]
                top_k_indices = np.argsort(chunk_scores)[::-1][:k]

                # Get unique documents from top-k chunks
                retrieved_docs = []
                seen_docs = set()
                for idx in top_k_indices:
                    doc_id = chunk_doc_ids[idx]
                    if doc_id not in seen_docs:
                        retrieved_docs.append(doc_id)
                        seen_docs.add(doc_id)

                # Get relevant documents for this query
                relevant = relevant_docs.get(q_idx, [])
                if not relevant:
                    continue

                # Compute recall
                hits = len(set(retrieved_docs) & set(relevant))
                recall = hits / len(relevant)
                recalls.append(recall)

                # Compute MRR
                mrr = 0.0
                for i, doc in enumerate(retrieved_docs):
                    if doc in relevant:
                        mrr = 1.0 / (i + 1)
                        break
                mrrs.append(mrr)

                # Compute NDCG (simplified)
                dcg = 0.0
                for i, doc in enumerate(retrieved_docs):
                    if doc in relevant:
                        dcg += 1.0 / np.log2(i + 2)

                idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
                ndcg = dcg / idcg if idcg > 0 else 0.0
                ndcgs.append(ndcg)

            metrics[f'recall@{k}'] = np.mean(recalls) if recalls else 0.0
            metrics[f'mrr@{k}'] = np.mean(mrrs) if mrrs else 0.0
            metrics[f'ndcg@{k}'] = np.mean(ndcgs) if ndcgs else 0.0

        return metrics

    def evaluate_dataset(self,
                         dataset_name: str,
                         corpus: Dict[str, Dict[str, str]],
                         queries: Dict[str, str],
                         relevant_docs: Dict[str, Dict[str, int]]) -> pd.DataFrame:
        """Evaluate chunking strategies on a single dataset across all embedding models"""
        results = []

        # Convert corpus and queries to lists for easier processing
        doc_ids = list(corpus.keys())
        documents = [self.construct_document(corpus[doc_id]) for doc_id in doc_ids]
        doc_id_to_index = {doc_id: idx for idx, doc_id in enumerate(doc_ids)}
        query_ids = list(queries.keys())
        query_texts = [queries[qid] for qid in query_ids]

        # Convert relevant docs to query index mapping
        relevant_docs_by_idx = {}
        for idx, qid in enumerate(query_ids):
            if qid in relevant_docs:
                relevant_docs_by_idx[idx] = list(relevant_docs[qid].keys())

        # Evaluate each embedding model
        for model_name in self.embedding_models:
            print(f"\nEvaluating {model_name} on {dataset_name}")

            # Initialize chunking strategies with this embedding model
            self.initialize_chunking_strategies(self.embedding_models[model_name])

            # Load the retrieval model
            retrieval_model = self.load_embedding_model(model_name)

            # Embed queries once per model
            query_embeddings = self.embed_queries(query_texts, retrieval_model)

            # Evaluate each chunking strategy
            for strategy_name in self.chunking_strategies:
                print(f"  - {strategy_name}")

                # Chunk all documents
                all_chunks = []
                chunk_doc_ids = []

                for doc_id, doc_text in tqdm(zip(doc_ids, documents),
                                             total=len(documents),
                                             desc=f"Chunking with {strategy_name}",
                                             leave=False):
                    try:
                        chunks = self.apply_chunking_strategy(
                            doc_text,
                            doc_id,
                            strategy_name,
                            doc_index=doc_id_to_index.get(doc_id),
                        )
                        all_chunks.extend(chunks)
                        chunk_doc_ids.extend([chunk.doc_id for chunk in chunks])
                    except Exception as e:
                        print("\n" + "=" * 80)
                        print(f"[ERROR] while chunking doc_id={doc_id} (strategy={strategy_name})")
                        print(f"Error: {e}")
                        print(f"Doc length (chars): {len(doc_text)}")
                        # show a sample of the text so logs aren't enormous
                        print("TEXT START >>>")
                        print(doc_text[:1000])
                        print("<<< TEXT END")
                        print("=" * 80 + "\n")
                        # optionally continue instead of halting
                        continue

                # Get chunk embeddings
                # For consistency, re-embed chunks with the retrieval model
                # This ensures embeddings are in the same space as queries
                chunk_texts = [chunk.text for chunk in all_chunks]
                chunk_embeddings = retrieval_model.encode(chunk_texts,
                                                          batch_size=self.batch_size,
                                                          show_progress_bar=False)

                # Compute metrics
                metrics = self.compute_retrieval_metrics(
                    query_embeddings,
                    chunk_embeddings,
                    chunk_doc_ids,
                    relevant_docs_by_idx
                )

                # Compute chunking statistics
                chunk_stats = self._compute_chunk_statistics(all_chunks, documents)

                # Store results
                result = {
                    'dataset': dataset_name,
                    'embedding_model': model_name,
                    'chunking_strategy': strategy_name,
                    'n_chunks': len(all_chunks),
                    'avg_chunks_per_doc': len(all_chunks) / len(documents),
                    **chunk_stats,
                    **metrics
                }
                results.append(result)

        return pd.DataFrame(results)

    def _compute_chunk_statistics(self, chunks: List[ChunkResult], documents: List[str]) -> Dict[str, float]:
        """Compute statistics about the chunking"""
        stats = {}

        # Basic statistics
        chunk_lengths = [len(chunk.text) for chunk in chunks]
        stats['avg_chunk_length'] = np.mean(chunk_lengths)
        stats['std_chunk_length'] = np.std(chunk_lengths)

        # Strategy-specific statistics
        if chunks and 'cohesiveness' in chunks[0].metadata:
            # DP chunking statistics
            cohesiveness_scores = [chunk.metadata['cohesiveness'] for chunk in chunks]
            separation_scores = [chunk.metadata['separation_score'] for chunk in chunks]
            combined_scores = [chunk.metadata['combined_score'] for chunk in chunks]

            stats['avg_cohesiveness'] = np.mean(cohesiveness_scores)
            stats['avg_separation'] = np.mean(separation_scores)
            stats['avg_combined_score'] = np.mean(combined_scores)

        # Token statistics
        if chunks and 'token_count' in chunks[0].metadata:
            token_counts = [chunk.metadata['token_count'] for chunk in chunks]
            stats['avg_tokens_per_chunk'] = np.mean(token_counts)
            stats['std_tokens_per_chunk'] = np.std(token_counts)

        return stats

    def _load_precomputed_chunks(self, path: str) -> Dict[Any, List[str]]:
        """Load precomputed chunks from JSONL into a doc_id -> chunks map."""
        out: Dict[Any, List[str]] = {}
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if not self.ignore_precomputed_dataset and "dataset" in obj:
                    continue
                doc_key = obj.get("doc_id")
                chunks = obj.get("chunks") or obj.get("chunks_stripped") or []
                if doc_key is None:
                    continue
                out[doc_key] = chunks
        return out

    def _get_precomputed_chunks(
        self,
        strategy_name: str,
        doc_id: str,
        doc_index: Optional[int],
    ) -> List[str]:
        path = self.precomputed_chunks[strategy_name]
        if path not in self._precomputed_cache:
            self._precomputed_cache[path] = self._load_precomputed_chunks(path)

        mapping = self._precomputed_cache[path]
        if self.precomputed_doc_id_mode == "index":
            if doc_index is None:
                return []
            return mapping.get(doc_index, [])
        return mapping.get(doc_id, [])

    def run_evaluation(self, dataset_names: List[str] = None, save_detailed: bool = True) -> pd.DataFrame:
        """
        Run evaluation across all datasets and models

        Args:
            dataset_names: List of dataset names to evaluate. If None, evaluates all.
            save_detailed: Whether to save detailed results for each dataset
        """
        if dataset_names is None:
            dataset_names = list(self.dataset_configs.keys())

        all_results = []

        for dataset_name in dataset_names:
            print(f"\n{'=' * 50}")
            print(f"Loading and evaluating {dataset_name}")
            print(f"{'=' * 50}")

            # Load dataset
            corpus, queries, relevant_docs = self.load_dataset(dataset_name)

            if corpus is None:
                print(f"Failed to load {dataset_name}, skipping...")
                continue

            print(f"Loaded {len(corpus)} documents and {len(queries)} queries")

            # Evaluate dataset
            results_df = self.evaluate_dataset(dataset_name, corpus, queries, relevant_docs)
            all_results.append(results_df)

            # Save detailed results for this dataset if requested
            if save_detailed:
                dataset_file = f'chunking_eval_{dataset_name}.csv'
                results_df.to_csv(dataset_file, index=False)
                print(f"Saved detailed results to {dataset_file}")

        # Combine all results
        final_results = pd.concat(all_results, ignore_index=True)

        # Add summary statistics
        print("\n\nSummary Results:")
        print("=" * 80)

        # Overall performance by model and strategy
        summary = final_results.groupby(['embedding_model', 'chunking_strategy'])[
            ['recall@10', 'mrr@10', 'ndcg@10']
        ].mean()
        print("\nOverall Performance:")
        print(summary)

        # Best performing combination for each dataset
        print("\n\nBest Performing Combinations by Dataset:")
        best_by_dataset = final_results.loc[
            final_results.groupby('dataset')['ndcg@10'].idxmax()
        ][['dataset', 'embedding_model', 'chunking_strategy', 'ndcg@10']]
        print(best_by_dataset)

        return final_results


# Example usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run chunking evaluation on MTEB datasets.")
    parser.add_argument(
        "--dataset",
        action="append",
        dest="datasets",
        help="Dataset name to evaluate (repeatable).",
    )
    parser.add_argument(
        "--precomputed",
        action="append",
        default=[],
        help="Precomputed strategy mapping, e.g. name=/path/to/file.jsonl",
    )
    parser.add_argument(
        "--doc-id-mode",
        choices=["index", "id"],
        default="index",
        help="How to map precomputed doc_id values to corpus (index or id).",
    )
    parser.add_argument(
        "--respect-dataset-field",
        action="store_true",
        help="Only use JSONL entries that match dataset field (if present).",
    )
    args = parser.parse_args()

    precomputed = {}
    for item in args.precomputed:
        if "=" not in item:
            raise ValueError(f"Invalid --precomputed entry: {item}")
        name, path = item.split("=", 1)
        precomputed[name.strip()] = path.strip()

    evaluator = ChunkingEvaluator(
        batch_size=32,
        precomputed_chunks=precomputed,
        precomputed_doc_id_mode=args.doc_id_mode,
        ignore_precomputed_dataset=not args.respect_dataset_field,
    )

    # Run evaluation on specific datasets for testing
    # Use smaller datasets first for quick testing
    #test_datasets = ['SciFact','NFCorpus','TRECCOVID']
    test_datasets = ['TRECCOVID']
    if args.datasets:
        test_datasets = args.datasets

    print("Running evaluation on test datasets...")
    results = evaluator.run_evaluation(test_datasets, save_detailed=True)
    #corpus, queries, relevant_docs = evaluator.load_dataset("TRECCOVID")
    #
    # doc = corpus["pd1g119c"]  # {'title': ..., 'text': ...}
    # text = evaluator.construct_document(doc)  # combines title + text like your code
    # # print(len(text))
    # print(text)
    # tokenizer = AutoTokenizer.from_pretrained("intfloat/e5-small-v2")
    # tokens = tokenizer.encode(text, add_special_tokens=False)
    # print("Number of tokens:", len(tokens))


    # # Save overall results
    # results.to_csv('chunking_evaluation_results.csv', index=False)
    # print("\nOverall results saved to chunking_evaluation_results.csv")
    #
    # # Create visualizations
    # import matplotlib.pyplot as plt
    # import seaborn as sns
    #
    # # Set style
    # plt.style.use('seaborn-v0_8-darkgrid')
    # sns.set_palette("husl")
    #
    # # 1. Performance comparison by strategy and model
    # fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    #
    # metrics = ['recall@10', 'mrr@10', 'ndcg@10']
    # for idx, metric in enumerate(metrics):
    #     pivot_data = results.pivot_table(
    #         values=metric,
    #         index='embedding_model',
    #         columns='chunking_strategy',
    #         aggfunc='mean'
    #     )
    #
    #     pivot_data.plot(kind='bar', ax=axes[idx])
    #     axes[idx].set_title(f'Average {metric.upper()} by Model and Strategy')
    #     axes[idx].set_xlabel('Embedding Model')
    #     axes[idx].set_ylabel(metric.upper())
    #     axes[idx].legend(title='Chunking Strategy')
    #     axes[idx].tick_params(axis='x', rotation=45)
    #
    # plt.tight_layout()
    # plt.savefig('chunking_performance_comparison.png', dpi=300, bbox_inches='tight')
    # print("Performance comparison plot saved to chunking_performance_comparison.png")
    #
    # # 2. Heatmap of performance across datasets
    # fig, ax = plt.subplots(figsize=(12, 8))
    #
    # # Create a pivot table for the heatmap
    # heatmap_data = results.pivot_table(
    #     values='ndcg@10',
    #     index=['dataset'],
    #     columns=['embedding_model', 'chunking_strategy'],
    #     aggfunc='mean'
    # )
    #
    # sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax)
    # ax.set_title('NDCG@10 Performance Heatmap')
    # plt.xticks(rotation=45, ha='right')
    # plt.tight_layout()
    # plt.savefig('chunking_performance_heatmap.png', dpi=300, bbox_inches='tight')
    # print("Performance heatmap saved to chunking_performance_heatmap.png")
    #
    # # 3. Chunking statistics visualization
    # if 'avg_cohesiveness' in results.columns:
    #     fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    #
    #     # Cohesiveness vs Performance (for DP chunking)
    #     dp_results = results[results['chunking_strategy'] == 'dp_chunking']
    #     if not dp_results.empty:
    #         axes[0, 0].scatter(dp_results['avg_cohesiveness'], dp_results['ndcg@10'])
    #         axes[0, 0].set_xlabel('Average Cohesiveness')
    #         axes[0, 0].set_ylabel('NDCG@10')
    #         axes[0, 0].set_title('Cohesiveness vs Performance (DP Chunking)')
    #
    #         axes[0, 1].scatter(dp_results['avg_separation'], dp_results['ndcg@10'])
    #         axes[0, 1].set_xlabel('Average Separation')
    #         axes[0, 1].set_ylabel('NDCG@10')
    #         axes[0, 1].set_title('Separation vs Performance (DP Chunking)')
    #
    #     # Chunks per document vs Performance
    #     axes[1, 0].scatter(results['avg_chunks_per_doc'], results['ndcg@10'])
    #     axes[1, 0].set_xlabel('Average Chunks per Document')
    #     axes[1, 0].set_ylabel('NDCG@10')
    #     axes[1, 0].set_title('Chunking Granularity vs Performance')
    #
    #     # Token distribution
    #     if 'avg_tokens_per_chunk' in results.columns:
    #         axes[1, 1].scatter(results['avg_tokens_per_chunk'], results['ndcg@10'])
    #         axes[1, 1].set_xlabel('Average Tokens per Chunk')
    #         axes[1, 1].set_ylabel('NDCG@10')
    #         axes[1, 1].set_title('Chunk Size vs Performance')
    #
    #     plt.tight_layout()
    #     plt.savefig('chunking_statistics_analysis.png', dpi=300, bbox_inches='tight')
    #     print("Chunking statistics analysis saved to chunking_statistics_analysis.png")
    #
    # print("\nEvaluation complete!")
