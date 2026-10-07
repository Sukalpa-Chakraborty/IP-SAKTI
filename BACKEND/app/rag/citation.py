"""
Citation validation.

Every answer produced by the RAG pipeline must cite only chunks that were
actually retrieved.  This module validates that:

  1. Each citation references a chunk_id in the retrieved set.
  2. The title and source_type in the citation match the real chunk.

If validation fails, the answer is replaced with the insufficient-evidence
controlled response rather than letting a hallucinated citation reach the user.
"""
from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Set, Tuple

from app.llm.schemas import CitationObject
from app.rag.schemas import RetrievedChunk

# Render jurisdiction codes ("IN", "INT") as human names ("India",
# "International") so a citation never shows an ambiguous code (spec item 7).
from app.analysis.ip_schemas import human_jurisdiction

logger = logging.getLogger(__name__)

INSUFFICIENT_EVIDENCE_MESSAGE = (
    "Insufficient evidence in the current verified corpus to answer this question. "
    "The available documents do not contain enough relevant information. "
    "Please consult qualified legal, patent, or regulatory professionals for advice specific to your situation.\n\n"
    "This platform provides preliminary, source-backed information and decision support. "
    "It does not constitute legal, patent, regulatory, medical, or government advice or approval."
)


def validate_citations(
    answer_citations: List[CitationObject],
    retrieved_chunks: List[RetrievedChunk],
) -> Tuple[bool, List[CitationObject]]:
    """
    Validate that all citations reference actually retrieved chunks.

    Parameters
    ----------
    answer_citations:
        Citations the LLM included in its structured response.
    retrieved_chunks:
        Chunks returned by hybrid_retrieve() and passed to the LLM.

    Returns
    -------
    (is_valid, valid_citations)
        ``is_valid`` is True only when ALL citations are valid.
        ``valid_citations`` contains only the citations that passed validation.
    """
    if not retrieved_chunks:
        logger.warning("Citation validation: no retrieved chunks — treating as insufficient evidence")
        return False, []

    # Build lookup by chunk_id
    chunk_by_id: Dict[int, RetrievedChunk] = {c.chunk_id: c for c in retrieved_chunks}
    retrieved_ids: Set[int] = set(chunk_by_id.keys())

    valid: List[CitationObject] = []
    all_valid = True

    for citation in answer_citations:
        if citation.chunk_id not in retrieved_ids:
            logger.warning(
                "Citation validation FAILED: chunk_id=%d not in retrieved set %s",
                citation.chunk_id,
                sorted(retrieved_ids),
            )
            all_valid = False
            continue

        real_chunk = chunk_by_id[citation.chunk_id]

        # Verify title matches (case-insensitive prefix match)
        if not _titles_match(citation.title, real_chunk.title):
            logger.warning(
                "Citation title mismatch: cited='%s', real='%s' (chunk_id=%d)",
                citation.title,
                real_chunk.title,
                citation.chunk_id,
            )
            all_valid = False
            continue

        # Fill in real metadata from the retrieved chunk
        citation.document_id = real_chunk.document_id
        citation.source_type = real_chunk.source_type
        citation.jurisdiction = human_jurisdiction(real_chunk.jurisdiction)
        citation.publication_date = real_chunk.publication_date
        citation.url = real_chunk.source_url
        citation.page_number = real_chunk.page_number

        # Confidence indicator (SIH requirement: "mandatory source citations
        # with a confidence indicator"). Every path that returns citations to
        # a caller runs through here, so this is the one place that has to
        # populate it.
        score = confidence_score_for(real_chunk)
        citation.confidence_score = score
        citation.confidence_label = confidence_label_for(score)

        valid.append(citation)

    if not valid:
        all_valid = False

    logger.debug(
        "Citation validation: %d/%d citations valid",
        len(valid),
        len(answer_citations),
    )
    return all_valid, valid


def _titles_match(cited_title: str, real_title: str) -> bool:
    """Fuzzy title match — allow the LLM to shorten the title slightly."""
    cited = cited_title.lower().strip()
    real = real_title.lower().strip()
    # Exact match, or one is a prefix/substring of the other
    return cited == real or cited in real or real[:40] in cited


# -------------------------------------------------------
# Confidence indicator
# -------------------------------------------------------

# ms-marco-MiniLM-L-6-v2 emits raw logits (unbounded). A sigmoid squashes
# them into a 0-1 similarity the UI can show: logit 0.0 -> 0.5, so passages
# the cross-encoder considers net-relevant land above 0.5.
def _sigmoid(value: float) -> float:
    """Numerically-stable logistic function."""
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    z = math.exp(value)
    return z / (1.0 + z)


# Tier boundaries for the human-readable label shown next to a citation.
CONFIDENCE_HIGH = 0.70
CONFIDENCE_MEDIUM = 0.45


def confidence_score_for(chunk: RetrievedChunk) -> Optional[float]:
    """
    Normalised 0-1 confidence for one retrieved chunk.

    Preference order, chosen so the number means the same thing whichever
    path produced it:

    1. cross-encoder reranker logit -> sigmoid (the normal path: the
       cross-encoder reads query and passage jointly and is the strongest
       relevance signal we have).
    2. pgvector cosine similarity (already 0-1) when the reranker was
       unavailable and retrieval fell back to bi-encoder ordering.
    3. ``None`` when neither exists — RRF/keyword scores are rank-fusion
       artefacts, not relevance probabilities, so reporting them as a
       confidence would be inventing precision we do not have.

    Returns ``None`` only when no meaningful score exists; the label helper
    then returns ``None`` too, and the UI hides the badge rather than
    showing a fabricated one.
    """
    if chunk.rerank_score is not None:
        return round(_sigmoid(float(chunk.rerank_score)), 3)
    if chunk.vector_score:
        # Cosine similarity from BGE-M3 is already on 0-1 for these vectors.
        vector = max(0.0, min(1.0, float(chunk.vector_score)))
        if vector > 0.0:
            return round(vector, 3)
    return None


def confidence_label_for(score: Optional[float]) -> Optional[str]:
    """Map a 0-1 confidence score to HIGH / MEDIUM / LOW (or None)."""
    if score is None:
        return None
    if score >= CONFIDENCE_HIGH:
        return "HIGH"
    if score >= CONFIDENCE_MEDIUM:
        return "MEDIUM"
    return "LOW"
