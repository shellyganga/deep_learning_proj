"""Three-axis intrinsic chunk evaluation using an LLM-as-a-Judge.

For each chunked document we score three axes on a 0-3 ordinal scale:

- Coherence (document level): whether the split points divide the text
  at semantically independent transitions.
- Completeness (per chunk): whether a chunk can be read in isolation,
  with the entities and references it introduces resolved internally.
- Relevance Purity (per chunk): whether a chunk maintains a single,
  consistent topic without mixing unrelated information.

Each axis is scored with its own dedicated Jinja2 prompt template
(``prompts/score_coherence.jinja2`` etc.). The judge is invoked with
temperature 0 and a hard 4-token output cap so off-format responses
fail closed rather than silently miscount.

This module is imported by ``run_llm_judge.py``; there is no CLI entry
point here.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, List, Sequence

from jinja2 import Template
from tenacity import retry, retry_if_exception_type, stop_after_attempt

REPO_ROOT = Path(__file__).resolve().parents[2]
PROMPT_DIR = REPO_ROOT / "prompts"

COHERENCE_PROMPT_PATH = PROMPT_DIR / "score_coherence.jinja2"
COMPLETENESS_PROMPT_PATH = PROMPT_DIR / "score_completeness.jinja2"
RELEVANCE_PURITY_PROMPT_PATH = PROMPT_DIR / "score_relevance_purity.jinja2"

CHUNK_BREAK_MARKER = "\n[CHUNK BREAK]\n"

_DIGIT_RE = re.compile(r"[0-3]")


class LLMError(Exception):
    """Retryable failure from the LLM gateway."""


def _load_template(path: Path) -> Template:
    if not path.is_file():
        raise FileNotFoundError(f"Missing prompt template: {path}")
    return Template(path.read_text(encoding="utf-8"), autoescape=False)


def _parse_score(raw: str) -> int:
    """Extract the first 0-3 digit from a model response."""
    match = _DIGIT_RE.search(raw)
    if not match:
        raise LLMError(f"No 0-3 digit in response: {raw!r}")
    return int(match.group(0))


def _build_default_llm(model: str = "gpt-4o-mini"):
    """Construct the Azure-backed gateway used by the rest of the repo."""
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


class _AxisEvaluator:
    """Shared plumbing for the three axis evaluators."""

    PROMPT_PATH: Path  # set on subclasses
    RENDER_KEY: str    # template variable name expected by the prompt

    def __init__(self, llm=None, model: str = "gpt-4o-mini"):
        self.template = _load_template(self.PROMPT_PATH)
        self.llm = llm if llm is not None else _build_default_llm(model)
        # Determinism + tight token cap so off-format output fails closed.
        self.llm_kwargs: Dict[str, object] = {"temperature": 0, "max_tokens": 4}

    def _render(self, value: str) -> str:
        return self.template.render(**{self.RENDER_KEY: value})

    @retry(
        retry=retry_if_exception_type(LLMError),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    def _score(self, value: str) -> int:
        prompt = self._render(value)
        try:
            response = self.llm.parse(
                messages=[
                    {"role": "system", "content": "Output only a single integer (0-3)."},
                    {"role": "user", "content": prompt},
                ],
                **self.llm_kwargs,
            )
            return _parse_score(str(response))
        except LLMError:
            raise
        except Exception as exc:
            raise LLMError(str(exc))


class CoherenceEvaluator(_AxisEvaluator):
    """Document-level: do boundaries fall at semantic transitions?"""

    PROMPT_PATH = COHERENCE_PROMPT_PATH
    RENDER_KEY = "chunked_text"

    def score(self, chunks: Sequence[str]) -> int:
        if len(chunks) < 2:
            # No boundaries to mis-place.
            return 3
        chunked_text = CHUNK_BREAK_MARKER.join(c.strip() for c in chunks)
        return self._score(chunked_text)


class CompletenessEvaluator(_AxisEvaluator):
    """Per-chunk: can the chunk stand alone?"""

    PROMPT_PATH = COMPLETENESS_PROMPT_PATH
    RENDER_KEY = "chunk"

    def score(self, chunk: str) -> int:
        return self._score(chunk)

    def score_all(self, chunks: Sequence[str]) -> List[int]:
        return [self.score(c) for c in chunks]


class RelevancePurityEvaluator(_AxisEvaluator):
    """Per-chunk: single consistent topic?"""

    PROMPT_PATH = RELEVANCE_PURITY_PROMPT_PATH
    RENDER_KEY = "chunk"

    def score(self, chunk: str) -> int:
        return self._score(chunk)

    def score_all(self, chunks: Sequence[str]) -> List[int]:
        return [self.score(c) for c in chunks]


def evaluate_chunks(
    chunks: Sequence[str],
    *,
    model: str = "gpt-4o-mini",
    coherence_eval: "CoherenceEvaluator | None" = None,
    completeness_eval: "CompletenessEvaluator | None" = None,
    relevance_purity_eval: "RelevancePurityEvaluator | None" = None,
) -> Dict[str, object]:
    """Score a single chunked document along all three axes.

    Returns a dict:
      - coherence        : int (0-3), document-level
      - completeness     : list[int], per-chunk
      - relevance_purity : list[int], per-chunk
      - macro_avg        : float, mean of the three per-document means

    The evaluators may be passed in to amortize prompt-template loading
    and gateway construction across many documents; otherwise fresh
    evaluators are constructed for this call.
    """
    if len(chunks) == 0:
        raise ValueError("Cannot evaluate an empty chunk list.")

    coh = coherence_eval or CoherenceEvaluator(model=model)
    comp = completeness_eval or CompletenessEvaluator(model=model)
    rp = relevance_purity_eval or RelevancePurityEvaluator(model=model)

    coherence = coh.score(chunks)
    completeness = comp.score_all(chunks)
    relevance_purity = rp.score_all(chunks)

    macro_avg = (
        coherence
        + (sum(completeness) / len(completeness))
        + (sum(relevance_purity) / len(relevance_purity))
    ) / 3.0

    return {
        "coherence": coherence,
        "completeness": completeness,
        "relevance_purity": relevance_purity,
        "macro_avg": macro_avg,
    }
