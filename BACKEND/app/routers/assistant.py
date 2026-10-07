"""
IP-SAKTI Assistant (RAG chatbot) router.

Endpoints:
  POST   /api/assistant/chat                   — send a message, get RAG answer
  POST   /api/assistant/attachments            — upload a PDF to attach to a chat
  GET    /api/assistant/conversations          — list user's sessions
  GET    /api/assistant/conversations/{id}     — get session + messages
  DELETE /api/assistant/conversations/{id}     — delete session
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from typing import Dict, List, Literal, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel, Field
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.config import get_settings
from app.utils import get_current_user_model
from app.bhashini.translator import from_canonical_answer, to_canonical_query
from app.database import get_db
from app.llm.provider import get_llm_provider, get_llm_provider_for
from app.llm.schemas import CitationObject, StructuredRAGResponse
from app.models.models import User
from app.models.rag_models import ChatAttachment, ChatMessage, ChatSession, SourceChunk, SourceDocument
from app.rag.generation import generate_rag_answer
from app.rag.ingestion import store_parsed_chunks
from app.rag.parser import parse_document
from app.services.audit_service import AuditService
from app.rag.partial_answer import (
    ANSWERED_STATUSES,
    STATUS_EXPERT_REVIEW,
    STATUS_INSUFFICIENT,
    STATUS_PARTIAL,
    STATUS_PROCESSING_ERROR,
    STATUS_SUPPORTED,
    TOPIC_GENERAL,
    PartialAnswerResult,
    SectionContext,
    SectionResult,
    SubQuestion,
    answer_partial,
    decompose_query,
    processing_error_result,
    render_answer,
    should_decompose,
    wrap_as_sections,
)
from app.rag.jurisdiction_scope import (
    JurisdictionScopeTracker,
    allowed_jurisdictions,
    allowed_for_market,
    claim_standing,
    compute_evidence_status,
    gap_warnings_for,
    has_verified_evidence,
    is_launch_question,
    market_display_label,
    markets_from_product_context,
    missing_information_for,
    no_source_sentence,
    snapshot_content,
    strip_unselected_us_references,
)
from app.rag.retrieval import hybrid_retrieve
from app.rag.regime import (
    JURISDICTION_MODES,
    REGIME_LABELS,
    available_regimes,
    describe_regime,
    exclusion_notice,
    regimes_for_mode,
)
from app.rag.schemas import MetadataFilter

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/assistant", tags=["Assistant"])

GENERAL_DISCLAIMER = (
    "This platform provides preliminary, source-backed information and decision support. "
    "It does not constitute legal, patent, regulatory, medical, or government advice or approval."
)

# Chat PDF attachments: hard cap on the stored extracted text, and the
# slice injected into the prompt (keeps both LLM providers in context).
CHAT_ATTACHMENT_MAX_CHARS = 300_000
CHAT_DOC_CONTEXT_CHARS = 100_000


# -------------------------------------------------------
# Request / Response schemas
# -------------------------------------------------------

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    session_id: Optional[int] = None
    product_id: Optional[int] = None
    product_version_id: Optional[int] = None
    input_language: str = "en"
    output_language: str = "en"
    filter_source_types: Optional[List[str]] = None
    filter_jurisdictions: Optional[List[str]] = None
    include_my_documents: bool = True
    # PDFs uploaded for this message (bound to the session server-side).
    attachment_ids: Optional[List[int]] = None
    # Which LLM answers: groq (default) | sarvam.
    provider: Optional[Literal["groq", "sarvam"]] = None
    # ---------------------------------------------------------------
    # Explicit jurisdiction toggle (SIH requirement A1):
    #   "an explicit jurisdiction switch so that answers are never
    #    conflated ... with the two answer-sets kept visibly separate"
    # ---------------------------------------------------------------
    #   "india"         -> one answer-set, national regime sources only
    #   "international" -> one answer-set, foreign/treaty sources only
    #   "both"          -> two separately generated answer-sets
    #   omitted / null  -> legacy combined pass (no regime split); kept so
    #                      existing clients and tests behave exactly as before
    jurisdiction_mode: Optional[Literal["india", "international", "both"]] = None


class JurisdictionAnswerSet(BaseModel):
    """
    One regime's answer, kept structurally separate from the others.

    The two layers are produced by *separate retrievals and separate
    generations* - not by splitting one text - so neither can bleed into
    the other.
    """

    regime: str                      # india | international | all
    label: str                       # "India — national regime"
    heading: str                     # "INDIA (NATIONAL REGIME)"
    description: str = ""
    answer: str
    sections: List[SectionResult] = []
    citations: List[CitationOut] = []
    overall_status: str = STATUS_SUPPORTED
    insufficient_evidence: bool = False
    warnings: List[str] = []
    # Evidence accounting for THIS regime only.
    source_count: int = 0
    jurisdictions_used: List[str] = []
    # Out-of-regime sources the toggle withheld - proves the split happened.
    excluded_source_count: int = 0
    private_source_count: int = 0
    # Standing sentence naming what was excluded.
    scope_notice: str = ""


class CitationOut(BaseModel):
    chunk_id: int
    document_id: int
    title: str
    source_type: str
    jurisdiction: Optional[str] = None
    publication_date: Optional[str] = None
    page_number: Optional[int] = None
    relevant_text: str
    url: Optional[str] = None
    # Confidence indicator (SIH requirement A3) — populated from reranker score.
    confidence_score: Optional[float] = None
    confidence_label: Optional[str] = None  # HIGH | MEDIUM | LOW


class DebugInfo(BaseModel):
    """Retrieval/answering diagnostics — exposed only when DEBUG=true."""

    pdf_processed: bool = False
    pages_extracted: int = 0
    chunks_created: int = 0
    upload_chunks_retrieved: int = 0
    corpus_chunks_retrieved: int = 0
    filters_applied: List[str] = []
    sections_supported: int = 0
    sections_abstained: int = 0


class ChatResponse(BaseModel):
    session_id: int
    message_id: int
    answer: str
    citations: List[CitationOut] = []
    insufficient_evidence: bool = False
    warnings: List[str] = []
    language: str = "en"
    # Product Passport context this answer was grounded in, when supplied.
    product_id: Optional[int] = None
    product_version_id: Optional[int] = None
    # Provenance of the sources used: VERIFIED_PUBLIC_SOURCE and/or USER_PROVIDED.
    provenance: List[str] = []
    # Provider selected for this answer (echo of the request choice).
    provider: str = "groq"
    # Selective answering: response-level status and one result per
    # sub-question (SUPPORTED / PARTIALLY_SUPPORTED / INSUFFICIENT_EVIDENCE /
    # PROCESSING_ERROR / EXPERT_REVIEW_REQUIRED and the structured statuses
    # ADDITIONAL_INFORMATION_NEEDED / USER_PROVIDED_ONLY /
    # REVIEW_RECOMMENDED / SUPPORTED_AS_WORKFLOW_ANALYSIS). Every chat
    # response is structured this way; unsupported parts abstain individually.
    overall_status: str = STATUS_SUPPORTED
    sections: List[SectionResult] = []
    # ---------------------------------------------------------------
    # Jurisdiction toggle output (A1). One entry per selected regime;
    # two entries when the toggle is on "both". Each entry is a complete,
    # independently generated answer-set with its own citations.
    # ---------------------------------------------------------------
    jurisdiction_mode: str = "all"
    answer_sets: List[JurisdictionAnswerSet] = []
    # True only when the toggle produced two separately generated sets.
    regimes_separated: bool = False
    # Follow-up output structure: the user's own facts, verified external
    # facts (empty when no citation survived) and bounded system inferences.
    user_provided_facts: List[str] = []
    verified_external_facts: List[str] = []
    system_inferences: List[str] = []
    # The platform disclaimer (identical text everywhere, never reworded).
    disclaimer: str = GENERAL_DISCLAIMER
    # ---------------------------------------------------------------
    # Market-entry context (populated when a Product Passport version
    # with target markets frames the question; all spec fields).
    # ---------------------------------------------------------------
    # Display labels of the selected target markets, e.g. ["India", "Germany/EU"].
    market_context: List[str] = []
    # Allowed source jurisdictions for this answer (market scope).
    jurisdiction_filter: List[str] = []
    # Jurisdictions whose sources were dropped because they are not selected.
    unselected_jurisdiction_sources_excluded: List[str] = []
    # Private-document search status - honest flags, never aspirational.
    private_search_requested: Optional[bool] = None
    private_search_performed: Optional[bool] = None
    private_sources_found: int = 0
    public_sources_found: int = 0
    # The standing jurisdiction warning shown in the UI.
    jurisdiction_warning: str = ""
    # {overall_status, launch_decision, reason} - evidence sufficiency.
    evidence_status: Optional[Dict] = None
    # Explicit market-specific evidence-gap sentences (rule 6).
    evidence_gap_warnings: List[str] = []
    # {selected_markets, sources_used, sources_excluded, missing_information}.
    why_this_answer: Optional[Dict] = None
    # Spec claim JSON per recorded Product Passport claim.
    claim_reviews: List[Dict] = []
    # Developer-only diagnostics; populated only when DEBUG=true.
    debug: Optional[DebugInfo] = None


class AttachmentOut(BaseModel):
    id: int
    filename: str
    char_count: int
    truncated: bool = False
    session_id: Optional[int] = None
    # Page-aware indexing info (spec diagnostics): the corpus document
    # holding the attachment's chunks, its pages/chunks and file hash.
    document_id: Optional[int] = None
    pages_extracted: int = 0
    chunks_created: int = 0
    content_hash: Optional[str] = None


class MessageOut(BaseModel):
    id: int
    role: str
    content: str
    citations: List[CitationOut] = []
    insufficient_evidence: bool = False
    created_at: str


class SessionOut(BaseModel):
    id: int
    title: Optional[str]
    product_id: Optional[int]
    product_version_id: Optional[int]
    created_at: str
    message_count: int


class SessionDetailOut(BaseModel):
    id: int
    title: Optional[str]
    product_id: Optional[int]
    product_version_id: Optional[int]
    created_at: str
    messages: List[MessageOut]


# -------------------------------------------------------
# Routes
# -------------------------------------------------------

@router.post("/chat", response_model=ChatResponse)
def chat(
    req: ChatRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user_model),
):
    """
    Send a message to the RAG chatbot and get a citation-backed answer.

    - Creates a new session if session_id is not provided.
    - Uses hybrid retrieval (pgvector + FTS) + reranking.
    - Validates citations — returns insufficient-evidence if evidence is lacking.
    """
    # -------------------------------------------------------
    # 1. Resolve or create session
    # -------------------------------------------------------
    session = _get_or_create_session(
        db,
        current_user,
        session_id=req.session_id,
        product_id=req.product_id,
        product_version_id=req.product_version_id,
        first_message=req.message,
    )

    # -------------------------------------------------------
    # 1b. Attachments: bind to this session, build document context
    # -------------------------------------------------------
    attachments = _resolve_chat_attachments(
        db, current_user, session, req.attachment_ids or []
    )
    document_context, doc_warning = _build_document_context(attachments)

    # -------------------------------------------------------
    # 2. Persist user message
    # -------------------------------------------------------
    user_msg = ChatMessage(
        session_id=session.id,
        role="user",
        content=req.message,
        language=req.input_language,
    )
    db.add(user_msg)
    db.commit()
    db.refresh(user_msg)

    # -------------------------------------------------------
    # 2b. Language layer: map to the canonical English query.
    # RAG always retrieves and reasons on the canonical form; when
    # translation is unavailable the original is used and flagged.
    # -------------------------------------------------------
    canonical_query, lang_warnings = to_canonical_query(req.message, req.input_language)

    # -------------------------------------------------------
    # 3. Optional product passport context + jurisdiction scope
    #    The selected target markets are loaded BEFORE the filters:
    #    they define which jurisdictions may be retrieved, prompted
    #    and cited for this answer.
    # -------------------------------------------------------
    product_context = None
    if req.product_version_id:
        product_context = _get_product_context(db, req.product_id, req.product_version_id)

    selected_markets = markets_from_product_context(product_context)
    # Display label (Germany/EU) -> its allowed jurisdictions.
    market_scopes: Dict[str, List[str]] = {}
    for _market in selected_markets:
        market_scopes[market_display_label(_market)] = allowed_for_market(_market)
    # Allowed list = market-derived + any explicit request filter.
    allowed: List[str] = allowed_jurisdictions(selected_markets)
    for _explicit in (req.filter_jurisdictions or []):
        if _explicit and _explicit.casefold() not in {a.casefold() for a in allowed}:
            allowed.append(_explicit)

    filters = MetadataFilter(
        source_types=req.filter_source_types,
        jurisdictions=req.filter_jurisdictions,
    )

    # -------------------------------------------------------
    # 3b. Private document ids (shared by every regime pass), and the
    #     selection of which regime(s) this request must answer.
    #     Each pass builds its own JurisdictionScopeTracker so its
    #     source counts and exclusions stay regime-specific.
    # -------------------------------------------------------
    private_doc_ids = {
        row[0]
        for row in db.query(SourceDocument.id)
        .filter(
            SourceDocument.uploader_id == current_user.id,
            SourceDocument.is_public.is_(False),
        )
        .all()
    }

    # None -> legacy combined pass (no toggle supplied).
    # "both" -> two separate passes, India then International.
    selected_regimes: List[Optional[str]] = (
        list(available_regimes(regimes_for_mode(req.jurisdiction_mode)))
        if req.jurisdiction_mode
        else [None]
    )

    launch_question = is_launch_question(req.message)
    content_doc = snapshot_content(product_context)
    has_classification = (
        _has_resolved_classification(db, req.product_version_id)
        if product_context
        else False
    )
    missing_info = (
        missing_information_for(content_doc, has_classification)
        if product_context
        else []
    )
    # What both prompt builders receive. ``counts_provider`` is deliberately
    # absent here: each regime pass substitutes its own tracker's counter so
    # one layer can never report the other layer's source counts.
    jurisdiction_scope: Optional[dict] = None
    if allowed:
        jurisdiction_scope = {
            "selected_markets": list(selected_markets),
            "market_labels": list(market_scopes.keys()),
            "allowed_jurisdictions": list(allowed),
            "launch_question": launch_question,
            "missing_information": list(missing_info),
        }

    # -------------------------------------------------------
    # 5. Retrieval + generation
    #    Multi-part questions take the selective (per-section) path:
    #    decompose -> retrieve per sub-question -> structured answer with
    #    per-section statuses. Single questions keep the original flow and
    #    are wrapped into the same structured shape afterwards.
    # -------------------------------------------------------
    user_id_for_retrieval = current_user.id if req.include_my_documents else None
    attachment_doc_ids = [a.document_id for a in attachments if a.document_id]
    attachment_texts = [a.extracted_text for a in attachments]

    def _resolve_llm():
        if req.provider == "sarvam":
            return get_llm_provider_for("sarvam")
        return get_llm_provider()

    # -------------------------------------------------------
    # 5. Retrieval + generation — ONE PASS PER SELECTED REGIME
    #    Multi-part questions take the selective (per-section) path:
    #    decompose -> retrieve per sub-question -> structured answer with
    #    per-section statuses. Single questions keep the original flow and
    #    are wrapped into the same structured shape afterwards.
    #
    #    When the jurisdiction toggle is on "both" this whole block runs
    #    twice - once restricted to Indian national sources, once to the
    #    international layer - so the two answer-sets are independently
    #    retrieved and independently generated rather than split from a
    #    single blended text (SIH requirement A1).
    # -------------------------------------------------------
    user_id_for_retrieval = current_user.id if req.include_my_documents else None
    attachment_doc_ids = [a.document_id for a in attachments if a.document_id]
    attachment_texts = [a.extracted_text for a in attachments]

    def _resolve_llm():
        if req.provider == "sarvam":
            return get_llm_provider_for("sarvam")
        return get_llm_provider()

    def _run_pass(regime: Optional[str]) -> Dict:
        """Retrieve, generate, enforce scope and translate for ONE regime.

        ``regime`` is ``"india"``, ``"international"``, or ``None`` for the
        legacy combined pass used when no toggle was supplied. Everything
        the pass produces - sections, citations, warnings, status - is
        returned together so nothing can leak between the two layers.
        """
        pass_tracker = JurisdictionScopeTracker(
            allowed_jurisdictions=allowed,
            market_scopes=market_scopes,
            private_doc_ids=private_doc_ids,
            # getattr: some tests stub get_settings() with a partial object
            # (e.g. SimpleNamespace(debug=True)); the default mirrors
            # app.config rag_rerank_top_k = 6.
            final_k=getattr(get_settings(), "rag_rerank_top_k", 6),
            regime=regime,
        )
        pass_retrieve = pass_tracker.wrap(hybrid_retrieve)

        pass_scope: Optional[dict] = None
        if allowed or regime:
            pass_scope = dict(jurisdiction_scope or {})
            pass_scope["counts_provider"] = pass_tracker.public_counts_by_market
            if regime:
                pass_scope["regime"] = regime

        pass_partial: Optional[PartialAnswerResult] = None
        pass_rag: Optional[StructuredRAGResponse] = None
        pass_debug: Dict[str, int] = {}

        if should_decompose(canonical_query):
            try:
                llm = _resolve_llm()
                subquestions = decompose_query(llm, canonical_query)
                pass_partial = answer_partial(
                    db=db,
                    query=canonical_query,
                    subquestions=subquestions,
                    llm=llm,
                    retrieve_fn=pass_retrieve,
                    filters=filters,
                    user_id=user_id_for_retrieval,
                    attachment_doc_ids=attachment_doc_ids,
                    document_context=document_context,
                    product_context=product_context,
                    attachment_texts=attachment_texts,
                    jurisdiction_scope=pass_scope,
                )
                pass_debug = dict(pass_partial.debug)
            except Exception as exc:
                logger.error("Selective RAG generation error: %s", exc, exc_info=True)
                pass_partial = processing_error_result(
                    [
                        SectionContext(
                            subquestion=SubQuestion(
                                topic=TOPIC_GENERAL, question=canonical_query
                            )
                        )
                    ],
                    reason=(str(exc)[:200] or exc.__class__.__name__),
                )
                pass_debug = dict(pass_partial.debug)

        pass_chunks: List = []
        if pass_partial is None:
            pass_chunks = pass_retrieve(
                db=db,
                query=canonical_query,
                filters=filters,
                user_id=user_id_for_retrieval,
            )
            try:
                llm = _resolve_llm()
                pass_rag = generate_rag_answer(
                    query=canonical_query,
                    chunks=pass_chunks,
                    llm=llm,
                    product_context=product_context,
                    document_context=document_context,
                    jurisdiction_scope=pass_scope,
                    launch_question=launch_question,
                )
            except Exception as exc:
                logger.error("RAG generation error: %s", exc, exc_info=True)
                from app.rag.citation import INSUFFICIENT_EVIDENCE_MESSAGE
                pass_rag = StructuredRAGResponse(
                    answer=INSUFFICIENT_EVIDENCE_MESSAGE,
                    citations=[],
                    insufficient_evidence=True,
                    warnings=["The AI system encountered an error. Please try again."],
                )
            pass_debug = {
                "upload_chunks_retrieved": sum(
                    1 for c in pass_chunks if c.document_id in attachment_doc_ids
                ),
                "corpus_chunks_retrieved": sum(
                    1 for c in pass_chunks if c.document_id not in attachment_doc_ids
                ),
            }

        # ---- assemble this pass's sections / citations / statuses ----
        if pass_partial is not None:
            pass_warnings = list(pass_partial.warnings)
            pass_sections = list(pass_partial.sections)
            pass_overall = pass_partial.overall_status
            pass_insufficient = pass_overall in (
                STATUS_INSUFFICIENT,
                STATUS_PROCESSING_ERROR,
            )
            pass_citations = [
                c.model_dump() for s in pass_sections for c in s.citations
            ]
            pass_user_facts = list(pass_partial.user_provided_facts)
            pass_verified_facts = list(pass_partial.verified_external_facts)
            pass_inferences = list(pass_partial.system_inferences)
        else:
            pass_warnings = list(pass_rag.warnings)
            pass_overall = STATUS_SUPPORTED
            pass_insufficient = pass_rag.insufficient_evidence
            pass_citations = [c.model_dump() for c in pass_rag.citations]
            # Legacy single-answer path: no decomposition happened, so the
            # structured fact arrays stay empty (structure still present).
            pass_user_facts = []
            pass_verified_facts = []
            pass_inferences = []

        pass_provenance = _with_attachment_provenance(
            _provenance_for_citations(db, pass_citations), document_context
        )

        if pass_partial is None:
            pass_sections = wrap_as_sections(
                answer=pass_rag.answer,
                insufficient_evidence=pass_rag.insufficient_evidence,
                citations=pass_rag.citations,
                provenance=pass_provenance,
            )
            pass_overall = (
                pass_sections[0].status if pass_sections else STATUS_INSUFFICIENT
            )

        if doc_warning:
            pass_warnings.append(doc_warning)

        # ---- market-scope enforcement on this pass's answer ----
        pass_market_gap_warnings: List[str] = []
        if allowed:
            removed_total = 0
            for section in pass_sections:
                cleaned, removed = strip_unselected_us_references(
                    section.answer, allowed
                )
                if removed:
                    removed_total += removed
                    section.answer = cleaned
                    if not section.answer.strip():
                        section.answer = no_source_sentence(selected_markets)
            for _facts in (pass_user_facts, pass_verified_facts, pass_inferences):
                for _i, _fact in enumerate(_facts):
                    _cleaned, _removed = strip_unselected_us_references(_fact, allowed)
                    if _removed:
                        removed_total += _removed
                        _facts[_i] = _cleaned
            if removed_total:
                _notice = (
                    f"{removed_total} U.S. regulatory reference(s) were removed "
                    "from this answer because the United States is not a "
                    "selected target market."
                )
                pass_warnings.append(_notice)
                logger.info("Chat: %s (user=%s)", _notice, current_user.id)

            if selected_markets:
                _per_market = pass_tracker.public_counts_by_market()
                _market_names = {
                    market_display_label(m): m for m in selected_markets
                }
                pass_market_gap_warnings = gap_warnings_for(
                    per_market_counts={
                        **_per_market,
                        "Private documents": len(pass_tracker.private_docs),
                    },
                    market_labels=list(market_scopes.keys()),
                    market_names=_market_names,
                    launch_question=launch_question,
                )
                pass_warnings.extend(pass_market_gap_warnings)

        # Always add the general disclaimer
        pass_warnings.append(GENERAL_DISCLAIMER)
        pass_warnings.extend(lang_warnings)

        # Map this pass's sections to the user's language (honest fallback
        # keeps English and warns when translation is unavailable), then
        # render the pass's chat-facing answer from the translated sections.
        translation_warnings: List[str] = []
        for section in pass_sections:
            translated, section_warnings = from_canonical_answer(
                section.answer, req.output_language
            )
            section.answer = translated
            translation_warnings.extend(section_warnings)
        pass_warnings.extend(_dedupe(translation_warnings))

        jurisdictions_used = sorted(
            {
                str(c.get("jurisdiction"))
                for c in pass_citations
                if c.get("jurisdiction")
            }
        )

        return {
            "regime": regime,
            "tracker": pass_tracker,
            "scope": pass_scope,
            "sections": pass_sections,
            "citations": pass_citations,
            "provenance": pass_provenance,
            "warnings": pass_warnings,
            "overall_status": pass_overall,
            "insufficient": pass_insufficient,
            "user_facts": pass_user_facts,
            "verified_facts": pass_verified_facts,
            "inferences": pass_inferences,
            "market_gap_warnings": pass_market_gap_warnings,
            "retrieval_debug": pass_debug,
            "was_selective": pass_partial is not None,
            "jurisdictions_used": jurisdictions_used,
            "answer": render_answer(pass_sections),
        }

    passes = [_run_pass(regime) for regime in selected_regimes]

    # -------------------------------------------------------
    # 6. Assemble warnings, sections, provenance, statuses across passes
    # -------------------------------------------------------
    scope_tracker = passes[0]["tracker"]
    partial_result = passes[0] if passes[0]["was_selective"] else None

    warnings = _dedupe([w for p in passes for w in p["warnings"]])
    sections = [s for p in passes for s in p["sections"]]
    citations_list = [c for p in passes for c in p["citations"]]
    user_provided_facts = [f for p in passes for f in p["user_facts"]]
    verified_external_facts = [f for p in passes for f in p["verified_facts"]]
    system_inferences = [f for p in passes for f in p["inferences"]]
    market_gap_warnings = [w for p in passes for w in p["market_gap_warnings"]]
    # Provenance is a set over the whole response: the label says whether a
    # claim rested on verified public sources or the user's own documents,
    # so a two-regime answer reports the union of both layers.
    provenance_labels = sorted(
        {label for p in passes for label in p["provenance"]}
    )

    if len(passes) == 1:
        overall_status = passes[0]["overall_status"]
        insufficient_flag = passes[0]["insufficient"]
    else:
        overall_status = _combine_statuses(
            [p["overall_status"] for p in passes]
        )
        # A two-regime answer is "insufficient" only when BOTH layers
        # abstained; one supported layer is reported as such rather than
        # hiding it behind the other.
        insufficient_flag = all(p["insufficient"] for p in passes)

    retrieval_debug: Dict[str, int] = {}
    for _pass in passes:
        for _key, _value in _pass["retrieval_debug"].items():
            retrieval_debug[_key] = retrieval_debug.get(_key, 0) + _value

    # -------------------------------------------------------
    # 6d. Per-regime answer-sets (SIH requirement A1)
    # -------------------------------------------------------
    answer_sets: List[JurisdictionAnswerSet] = []
    for _pass in passes:
        _regime = _pass["regime"]
        if _regime is None:
            _meta = {
                "regime": "all",
                "label": "All regimes (no toggle applied)",
                "heading": "",
                "description": "",
            }
            _notice = ""
        else:
            _meta = describe_regime(_regime)
            _notice = exclusion_notice(
                _regime, _pass["tracker"].regime_excluded_count
            )
        _answer = _pass["answer"]
        if _regime is None or not _meta["heading"]:
            _rendered = _answer
        elif _meta["heading"] in _answer[:400]:
            _rendered = _answer
        else:
            _rendered = f"{_meta['heading']}\n\n{_answer}"
        answer_sets.append(
            JurisdictionAnswerSet(
                regime=_meta["regime"],
                label=_meta["label"],
                heading=_meta["heading"],
                description=_meta["description"],
                answer=_rendered,
                sections=_pass["sections"],
                citations=[CitationOut(**c) for c in _pass["citations"]],
                overall_status=_pass["overall_status"],
                insufficient_evidence=_pass["insufficient"],
                warnings=_dedupe(_pass["warnings"]),
                source_count=len(
                    {c.get("document_id") for c in _pass["citations"]}
                ),
                jurisdictions_used=_pass["jurisdictions_used"],
                excluded_source_count=_pass["tracker"].regime_excluded_count,
                private_source_count=len(_pass["tracker"].private_docs),
                scope_notice=_notice,
            )
        )

    if len(passes) == 1:
        final_answer = passes[0]["answer"]
    else:
        final_answer = "\n\n".join(
            item.answer for item in answer_sets
        )


    # -------------------------------------------------------
    # 6c. Market-entry context metadata (jurisdiction scope)
    #     All spec fields: market context, source scope, evidence
    #     status, jurisdiction warning, why-panel and claim reviews.
    # -------------------------------------------------------
    market_context = [market_display_label(m) for m in selected_markets]
    per_market_counts = scope_tracker.public_counts_by_market()
    private_found = len(scope_tracker.private_docs)
    public_found = len(scope_tracker.public_docs)
    excluded_labels = scope_tracker.excluded_labels()
    jurisdiction_warning_text = (
        "Only sources relevant to the selected markets are used for "
        "market-entry guidance."
    ) if allowed else ""

    evidence_status: Optional[Dict] = None
    why_this_answer: Optional[Dict] = None
    claim_reviews: List[Dict] = []
    if selected_markets:
        _claims_need_review = False
        if content_doc:
            _verified = has_verified_evidence(content_doc)
            _claims_need_review = any(
                claim_standing(c, _verified).get("review_required")
                for c in content_doc.get("claims") or []
            )
        evidence_status = compute_evidence_status(
            launch_question=launch_question,
            per_market_counts={
                **per_market_counts,
                "Private documents": private_found,
            },
            missing_information=missing_info,
            claims_need_review=_claims_need_review,
        )
        why_this_answer = {
            "selected_markets": list(selected_markets),
            "sources_used": {
                **per_market_counts,
                "Private documents": private_found,
            },
            "sources_excluded": [
                f"{label} — not a selected target market"
                for label in excluded_labels
            ],
            "missing_information": list(missing_info),
        }
    if content_doc:
        _verified = has_verified_evidence(content_doc)
        claim_reviews = [
            claim_standing(c, _verified)
            for c in content_doc.get("claims") or []
        ]

    # -------------------------------------------------------
    # 6b. Diagnostics (always logged; response field only when DEBUG=true)
    # -------------------------------------------------------
    filters_applied = [f"source_type:{v}" for v in (req.filter_source_types or [])]
    filters_applied += [f"jurisdiction:{v}" for v in (req.filter_jurisdictions or [])]
    for _jur in allowed:
        _entry = f"jurisdiction:{_jur}"
        if _entry not in filters_applied:
            filters_applied.append(_entry)

    debug_info: Optional[DebugInfo] = None
    if get_settings().debug:
        pages_extracted = 0
        chunks_created = 0
        if attachment_doc_ids:
            attachment_rows = (
                db.query(SourceChunk.page_number, SourceChunk.id)
                .filter(SourceChunk.document_id.in_(attachment_doc_ids))
                .all()
            )
            pages_extracted = len({row.page_number for row in attachment_rows})
            chunks_created = len(attachment_rows)
        debug_info = DebugInfo(
            pdf_processed=bool(attachments),
            pages_extracted=pages_extracted,
            chunks_created=chunks_created,
            upload_chunks_retrieved=retrieval_debug.get("upload_chunks_retrieved", 0),
            corpus_chunks_retrieved=retrieval_debug.get("corpus_chunks_retrieved", 0),
            filters_applied=filters_applied,
            sections_supported=sum(
                1
                for s in sections
                if s.status in ANSWERED_STATUSES
            ),
            sections_abstained=sum(
                1
                for s in sections
                if s.status not in ANSWERED_STATUSES
            ),
        )

    logger.info(
        "Chat: user=%s mode=%s provider=%s attachments=%d doc_chars=%d "
        "filters=%s overall=%s | %s",
        current_user.id,
        "selective" if partial_result is not None else "single",
        req.provider or "groq",
        len(attachments),
        len(document_context or ""),
        filters_applied or "none",
        overall_status,
        "; ".join(
            f"{s.topic}={s.status} chunks={s.evidence.retrieved_chunk_count} "
            f"top={s.evidence.top_similarity:.3f}"
            + (f" reason={s.abstention_reason}" if s.abstention_reason else "")
            for s in sections
        ),
    )

    # -------------------------------------------------------
    # 7. Persist assistant message
    # -------------------------------------------------------
    assistant_msg = ChatMessage(
        session_id=session.id,
        role="assistant",
        content=final_answer,
        citations_json=json.dumps(citations_list),
        sources_json=json.dumps([
            {"chunk_id": c["chunk_id"], "title": c["title"], "source_type": c["source_type"]}
            for c in citations_list
        ]),
        insufficient_evidence=insufficient_flag,
        language=req.output_language,
    )
    db.add(assistant_msg)
    db.commit()
    db.refresh(assistant_msg)

    # Audit: every chat query is logged for DPDP / audit-trail compliance.
    AuditService.log_action(
        db,
        user_id=current_user.id,
        action="chat_query",
        resource="chat_session",
        resource_id=session.id,
        details={
            "session_id": session.id,
            "message_id": assistant_msg.id,
            "overall_status": overall_status,
            "citation_count": len(citations_list),
            "insufficient_evidence": insufficient_flag,
            "provider": req.provider or "groq",
            "input_language": req.input_language,
            "output_language": req.output_language,
            "product_id": req.product_id,
            "product_version_id": req.product_version_id,
            "jurisdiction_mode": req.jurisdiction_mode or "all",
            "regimes_separated": len(passes) > 1,
        },
    )

    # -------------------------------------------------------
    # 8. Return response
    # -------------------------------------------------------
    return ChatResponse(
        session_id=session.id,
        message_id=assistant_msg.id,
        answer=final_answer,
        citations=[CitationOut(**c) for c in citations_list],
        insufficient_evidence=insufficient_flag,
        warnings=warnings,
        language=req.output_language,
        product_id=req.product_id,
        product_version_id=req.product_version_id,
        provenance=provenance_labels,
        provider=req.provider or "groq",
        overall_status=overall_status,
        sections=sections,
        # --- jurisdiction toggle (SIH requirement A1) ---
        jurisdiction_mode=req.jurisdiction_mode or "all",
        answer_sets=answer_sets,
        regimes_separated=len(passes) > 1,
        user_provided_facts=user_provided_facts,
        verified_external_facts=verified_external_facts,
        system_inferences=system_inferences,
        disclaimer=GENERAL_DISCLAIMER,
        # --- market-entry / jurisdiction context (spec fields) ---
        market_context=market_context,
        jurisdiction_filter=list(allowed),
        unselected_jurisdiction_sources_excluded=excluded_labels,
        private_search_requested=bool(req.include_my_documents),
        private_search_performed=bool(req.include_my_documents)
        and scope_tracker.search_executed,
        private_sources_found=private_found,
        public_sources_found=public_found,
        jurisdiction_warning=jurisdiction_warning_text,
        evidence_status=evidence_status,
        evidence_gap_warnings=market_gap_warnings,
        why_this_answer=why_this_answer,
        claim_reviews=claim_reviews,
        debug=debug_info,
    )


@router.post(
    "/attachments",
    response_model=AttachmentOut,
    status_code=status.HTTP_201_CREATED,
)
async def upload_chat_attachment(
    file: UploadFile = File(...),
    session_id: Optional[int] = Form(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user_model),
):
    """Upload a PDF to attach to the chat.

    The text is extracted immediately and stored for the caller (and
    bound to the given session, when provided). The chat then answers
    questions using the document; follow-up messages in the same
    conversation reuse it without re-uploading.
    """
    settings = get_settings()
    filename = (file.filename or "document.pdf").strip() or "document.pdf"

    if not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files can be attached to a chat message.",
        )

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if len(data) > settings.max_upload_size_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size: {settings.max_upload_size_mb} MB",
        )
    if not data.startswith(b"%PDF"):
        raise HTTPException(
            status_code=400,
            detail="This does not look like a valid PDF file.",
        )

    # Optional session binding (must belong to the caller)
    session = _require_own_session(session_id, current_user, db) if session_id else None

    # Extract with the same parser the knowledge base uses (PyMuPDF).
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        parsed = parse_document(tmp_path)
    except Exception as exc:
        logger.warning("Chat attachment parse failed for '%s': %s", filename, exc)
        raise HTTPException(
            status_code=400,
            detail="The PDF could not be read. It may be corrupted or password-protected.",
        )
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    text = (parsed.text or "").strip()
    if not text:
        raise HTTPException(
            status_code=400,
            detail=(
                "This PDF has no selectable text (it may be a scan). "
                "OCR is not available yet - upload a text-based PDF."
            ),
        )

    truncated = len(text) > CHAT_ATTACHMENT_MAX_CHARS
    if truncated:
        text = text[:CHAT_ATTACHMENT_MAX_CHARS]

    # Index the document so its content is retrievable page by page with
    # provenance (USER_PROVIDED) and deduplicated by content hash. The
    # full parsed text is indexed even when the prompt-injected copy is
    # capped. Failures never break the upload - injection still works.
    content_hash = hashlib.sha256(data).hexdigest()
    document_id: Optional[int] = None
    chunks_created = 0
    try:
        document_id, chunks_created = _index_chat_attachment(
            db, current_user, filename, content_hash, parsed
        )
    except Exception as exc:
        logger.error(
            "Attachment indexing failed for '%s': %s", filename, exc, exc_info=True
        )

    attachment = ChatAttachment(
        user_id=current_user.id,
        session_id=session.id if session else None,
        filename=filename,
        extracted_text=text,
        char_count=len(text),
        truncated=truncated,
        document_id=document_id,
    )
    db.add(attachment)
    db.commit()
    db.refresh(attachment)
    logger.info(
        "Chat attachment stored: id=%s user=%s chars=%s truncated=%s "
        "pages=%s document=%s chunks=%s",
        attachment.id,
        current_user.id,
        attachment.char_count,
        truncated,
        parsed.num_pages,
        document_id,
        chunks_created,
    )
    return AttachmentOut(
        id=attachment.id,
        filename=attachment.filename,
        char_count=attachment.char_count,
        truncated=truncated,
        session_id=attachment.session_id,
        document_id=document_id,
        pages_extracted=parsed.num_pages,
        chunks_created=chunks_created,
        content_hash=content_hash,
    )


def _resolve_chat_attachments(
    db: Session,
    user: User,
    session: ChatSession,
    requested_ids: List[int],
) -> List[ChatAttachment]:
    """Validate caller-owned attachments, bind them to the session, and
    return every attachment belonging to this conversation (including
    ones bound by earlier messages, so follow-ups reuse the document)."""
    ids = sorted({i for i in requested_ids if i})
    if ids:
        rows = (
            db.query(ChatAttachment)
            .filter(
                ChatAttachment.user_id == user.id,
                ChatAttachment.id.in_(ids),
            )
            .all()
        )
        found = {row.id for row in rows}
        missing = [i for i in ids if i not in found]
        if missing:
            raise HTTPException(status_code=404, detail="Attachment not found.")
        for row in rows:
            if row.session_id is None:
                row.session_id = session.id
            elif row.session_id != session.id:
                raise HTTPException(
                    status_code=400,
                    detail="This attachment belongs to a different conversation.",
                )
        db.commit()

    return (
        db.query(ChatAttachment)
        .filter(
            ChatAttachment.user_id == user.id,
            ChatAttachment.session_id == session.id,
        )
        .order_by(ChatAttachment.id)
        .all()
    )


def _build_document_context(
    attachments: List[ChatAttachment],
):
    """Concatenate attached documents into a prompt block (capped).

    Returns (context, warning): context is None when nothing is
    attached; the warning explains truncation honestly when the cap
    was hit.
    """
    if not attachments:
        return None, None

    parts: List[str] = []
    used = 0
    truncated = False
    for attachment in attachments:
        block = (
            f"--- Attached document: {attachment.filename} ---\n"
            f"{attachment.extracted_text}"
        )
        if used + len(block) > CHAT_DOC_CONTEXT_CHARS:
            block = block[: max(0, CHAT_DOC_CONTEXT_CHARS - used)]
            truncated = True
        parts.append(block)
        used += len(block)
        if truncated:
            break

    context = "\n\n".join(parts)
    warning = None
    if truncated:
        warning = (
            "Only the first "
            f"{CHAT_DOC_CONTEXT_CHARS:,} characters of your attached documents "
            "were used."
        )
    return (context or None), warning


def _with_attachment_provenance(
    provenance: List[str], document_context: Optional[str]
) -> List[str]:
    """Attached PDFs are user-provided evidence - surface that in provenance."""
    if document_context and "USER_PROVIDED" not in provenance:
        provenance.append("USER_PROVIDED")
    return provenance


def _dedupe(items: List[str]) -> List[str]:
    """Remove duplicates from a warning list, keeping the first occurrence."""
    seen = set()
    out: List[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _combine_statuses(statuses: List[str]) -> str:
    """
    Roll the per-regime statuses of a two-answer-set response into one.

    The combined value never flatters the result: two supported layers stay
    ``SUPPORTED``, any mix of supported and abstaining layers becomes
    ``PARTIALLY_SUPPORTED``, and a layer that errored is not masked by the
    other layer succeeding.
    """
    unique = {s for s in statuses if s}
    if not unique:
        return STATUS_INSUFFICIENT
    if len(unique) == 1:
        return next(iter(unique))
    if unique <= ANSWERED_STATUSES:
        # Every layer answered, but they do not agree on how much - that is
        # exactly what PARTIALLY_SUPPORTED means.
        return STATUS_PARTIAL
    answered = unique & ANSWERED_STATUSES
    if answered:
        return STATUS_PARTIAL
    if unique == {STATUS_PROCESSING_ERROR}:
        return STATUS_PROCESSING_ERROR
    return sorted(unique)[0]


def _index_chat_attachment(
    db: Session,
    user: User,
    filename: str,
    content_hash: str,
    parsed,
) -> tuple:
    """Store the attachment's text as a private, page-aware corpus document.

    This is what makes an uploaded PDF *searchable* (per page, with
    provenance and a per-file content hash) instead of only prompt-injected.

    Returns ``(document_id, chunks_created)``. Re-uploading the same file
    deduplicates onto the existing document (content hash). Indexing
    problems never fail the upload: the attachment still works through
    prompt injection and the document is recorded as FAILED for a reindex.
    """
    existing = (
        db.query(SourceDocument)
        .filter(
            SourceDocument.document_hash == content_hash,
            SourceDocument.uploader_id == user.id,
        )
        .first()
    )
    if existing is not None:
        if existing.status == "INDEXED" and (existing.chunk_count or 0) > 0:
            logger.info(
                "Attachment '%s' deduplicated onto document %s (%d chunks)",
                filename,
                existing.id,
                existing.chunk_count,
            )
            return existing.id, existing.chunk_count or 0
        try:
            return existing.id, store_parsed_chunks(db, existing, parsed)
        except Exception as exc:
            logger.error("Re-index of document %s failed: %s", existing.id, exc)
            return existing.id, 0

    document_hash = content_hash
    if (
        db.query(SourceDocument.id)
        .filter(SourceDocument.document_hash == content_hash)
        .first()
    ):
        # The same file already exists for another owner and the column is
        # unique - store a derived hash for this user's copy (chunk
        # metadata keeps the true content hash).
        document_hash = hashlib.sha256(
            f"{content_hash}:{user.id}".encode("utf-8")
        ).hexdigest()
        logger.info(
            "Attachment '%s' already indexed by another owner; storing a "
            "per-user copy under a derived hash",
            filename,
        )

    doc = SourceDocument(
        title=filename,
        source_type="USER_UPLOAD",
        language="en",
        document_hash=document_hash,
        file_path=None,
        uploader_id=user.id,
        is_public=False,
        status="PENDING",
    )
    db.add(doc)
    try:
        db.commit()
    except IntegrityError:
        # Concurrent duplicate upload - reuse the row that won.
        db.rollback()
        again = (
            db.query(SourceDocument)
            .filter(
                SourceDocument.document_hash == document_hash,
                SourceDocument.uploader_id == user.id,
            )
            .first()
        )
        if again is None:
            raise
        return again.id, again.chunk_count or 0
    db.refresh(doc)

    try:
        return doc.id, store_parsed_chunks(db, doc, parsed)
    except Exception as exc:
        logger.error(
            "Attachment indexing failed for document %s: %s", doc.id, exc, exc_info=True
        )
        try:
            db.rollback()
        except Exception:
            pass
        doc = db.get(SourceDocument, doc.id)
        if doc is not None:
            doc.status = "FAILED"
            doc.error_message = str(exc)[:500]
            db.commit()
        return doc.id if doc else None, 0


@router.get("/conversations", response_model=List[SessionOut])
def list_sessions(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user_model),
):
    """List all chat sessions for the current user."""
    sessions = (
        db.query(ChatSession)
        .filter(ChatSession.user_id == current_user.id)
        .order_by(ChatSession.created_at.desc())
        .all()
    )
    return [
        SessionOut(
            id=s.id,
            title=s.title,
            product_id=s.product_id,
            product_version_id=s.product_version_id,
            created_at=s.created_at.isoformat(),
            message_count=len(s.messages),
        )
        for s in sessions
    ]


@router.get("/conversations/{session_id}", response_model=SessionDetailOut)
def get_session(
    session_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user_model),
):
    """Get a session with all its messages."""
    session = _require_own_session(session_id, current_user, db)
    return SessionDetailOut(
        id=session.id,
        title=session.title,
        product_id=session.product_id,
        product_version_id=session.product_version_id,
        created_at=session.created_at.isoformat(),
        messages=[_msg_to_out(m) for m in session.messages],
    )


@router.delete("/conversations/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_session(
    session_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user_model),
):
    """Delete a chat session and all its messages."""
    session = _require_own_session(session_id, current_user, db)
    db.delete(session)
    db.commit()


# -------------------------------------------------------
# Escalation endpoint (SIH requirement A4)
# "a path to escalate to a human IP facilitator"
# -------------------------------------------------------

class EscalateRequest(BaseModel):
    session_id: int
    message_id: Optional[int] = None
    reason: Optional[str] = Field(None, max_length=1000, description="Why you are requesting human review.")
    contact_preference: Optional[str] = Field(None, description="email | phone | either")


class EscalateOut(BaseModel):
    escalation_id: str
    status: str
    message: str
    referral_instructions: List[str]
    facilitator_channels: List[dict]
    disclaimer: str


@router.post("/escalate", response_model=EscalateOut, status_code=status.HTTP_201_CREATED)
def escalate_to_facilitator(
    req: EscalateRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user_model),
):
    """
    Request handoff to a human IP facilitator.

    This endpoint is the "path to escalate to a human IP facilitator" required
    by the SIH problem statement. It logs the escalation request in the audit
    trail and returns structured referral information pointing the user to the
    authorised channels for professional Ayurveda-IP guidance.

    The platform does NOT connect to a live facilitator queue in the MVP.
    It generates a logged escalation record and directs the user to the
    appropriate official bodies (IP India Cell for AYUSH, DPIIT Fast-Track
    Cell, CSIR-TKDL access channel, NBA for ABS, AYUSH IP helpdesk).
    """
    # Validate session ownership
    session = _require_own_session(req.session_id, current_user, db)

    import uuid
    escalation_id = f"ESC-{uuid.uuid4().hex[:12].upper()}"

    # Audit-log the escalation for DPDP compliance
    AuditService.log_action(
        db,
        user_id=current_user.id,
        action="escalate_to_facilitator",
        resource="chat_session",
        resource_id=session.id,
        details={
            "escalation_id": escalation_id,
            "session_id": session.id,
            "message_id": req.message_id,
            "contact_preference": req.contact_preference,
            "reason_provided": bool(req.reason),
        },
    )

    facilitator_channels = [
        {
            "name": "IP India – AYUSH Cell / DPIIT Fast-Track IP Cell",
            "url": "https://ipindia.gov.in",
            "description": "Official Indian Patent Office helpdesk for Ayurveda and AYUSH IP queries.",
            "type": "free",
            "jurisdiction": "India",
        },
        {
            "name": "AYUSH IP Helpdesk (Ministry of AYUSH)",
            "url": "https://ayush.gov.in",
            "description": "Ministry of AYUSH helpline for regulatory and IP guidance for Ayurveda practitioners.",
            "type": "free",
            "jurisdiction": "India",
        },
        {
            "name": "National Biodiversity Authority (NBA) – ABS Guidance",
            "url": "https://nbaindia.org",
            "description": "NBA helpdesk for Access-and-Benefit-Sharing approvals and biodiversity compliance.",
            "type": "free",
            "jurisdiction": "India",
        },
        {
            "name": "TKDL Access Channel (CSIR-NML)",
            "url": "https://tkdl.res.in",
            "description": "Authorised channel to consult TKDL for prior-art screening in Ayurveda patent matters.",
            "type": "restricted_access",
            "jurisdiction": "India",
        },
        {
            "name": "WIPO GRATK Treaty helpdesk",
            "url": "https://www.wipo.int/tk/en/",
            "description": "WIPO guidance on the 2024 Treaty on IP, Genetic Resources and Associated TK.",
            "type": "free",
            "jurisdiction": "International",
        },
    ]

    referral_instructions = [
        f"Your escalation reference is: {escalation_id}",
        "This reference is logged in your audit trail (GET /api/audit).",
        "Contact one or more of the facilitator channels listed below.",
        "Provide your escalation reference when contacting an official helpdesk.",
        "IP-SAKTI Sahayak provides information only — the facilitator provides legal advice.",
        "For confidential matter, do NOT include proprietary details in open helpdesk emails.",
    ]

    return EscalateOut(
        escalation_id=escalation_id,
        status="LOGGED",
        message=(
            "Your request for human IP facilitator review has been logged. "
            "Please contact one of the official channels listed. "
            "Quote your escalation reference when doing so."
        ),
        referral_instructions=referral_instructions,
        facilitator_channels=facilitator_channels,
        disclaimer=GENERAL_DISCLAIMER,
    )



def _get_or_create_session(
    db: Session,
    user: User,
    session_id: Optional[int],
    product_id: Optional[int],
    product_version_id: Optional[int],
    first_message: str,
) -> ChatSession:
    if session_id:
        session = db.get(ChatSession, session_id)
        if session is None or session.user_id != user.id:
            raise HTTPException(status_code=404, detail="Session not found.")
        return session

    # Auto-title from first message (truncated)
    title = first_message[:80].strip()
    session = ChatSession(
        user_id=user.id,
        title=title,
        product_id=product_id,
        product_version_id=product_version_id,
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


def _require_own_session(session_id: int, user: User, db: Session) -> ChatSession:
    session = db.get(ChatSession, session_id)
    if session is None or session.user_id != user.id:
        raise HTTPException(status_code=404, detail="Session not found.")
    return session


def _msg_to_out(msg: ChatMessage) -> MessageOut:
    try:
        citations = [CitationOut(**c) for c in json.loads(msg.citations_json or "[]")]
    except Exception:
        citations = []
    return MessageOut(
        id=msg.id,
        role=msg.role,
        content=msg.content,
        citations=citations,
        insufficient_evidence=msg.insufficient_evidence,
        created_at=msg.created_at.isoformat() if msg.created_at else "",
    )


def _provenance_for_citations(db: Session, citations: List[dict]) -> List[str]:
    """
    Label where each cited source came from.

    The assistant must be able to say whether an answer rested on the shared
    public corpus or on the user's own uploaded documents. A citation from a
    private/user-uploaded document is ``USER_PROVIDED``; anything from the public
    corpus is ``VERIFIED_PUBLIC_SOURCE`` (spec provenance vocabulary).
    """
    from app.models.rag_models import SourceDocument

    if not citations:
        return []

    document_ids = {c.get("document_id") for c in citations if c.get("document_id")}
    if not document_ids:
        return []

    rows = (
        db.query(SourceDocument.id, SourceDocument.is_public, SourceDocument.uploader_id)
        .filter(SourceDocument.id.in_(document_ids))
        .all()
    )

    labels = set()
    for _doc_id, is_public, uploader_id in rows:
        labels.add("VERIFIED_PUBLIC_SOURCE" if is_public else "USER_PROVIDED")
        if not is_public and uploader_id is None:
            labels.add("VERIFIED_PUBLIC_SOURCE")
    return sorted(labels)


def _get_product_context(db: Session, product_id: Optional[int], version_id: Optional[int]) -> Optional[dict]:
    """Build a minimal product passport context dict to inject into the RAG prompt."""
    if not version_id:
        return None
    try:
        from app.models.product_models import Product, ProductVersion
        version = db.get(ProductVersion, version_id)
        if version is None:
            return None
        product = db.get(Product, version.product_id)
        context = {
            "product_name": product.name if product else "Unknown",
            "product_description": product.description if product else "",
            "version_number": version.version_number,
            "snapshot": json.loads(version.snapshot_data) if version.snapshot_data else {},
        }
        # Spec item 17: the assistant receives the same Overall Product View
        # data the page shows (version identity, analysis hash, classification,
        # IP / biodiversity / traditional-knowledge status, disclosure history),
        # so the two can never present conflicting interpretations. Patent
        # record details stay out of the prompt on purpose: stored screening
        # records are not verified live results and must never be quoted.
        if product is not None:
            try:
                from app.services.overview_service import overview_context_for_chat

                overview = overview_context_for_chat(db, product, version)
                if overview:
                    context["overview"] = overview
            except Exception as exc:  # pragma: no cover - chat must not fail
                logger.warning("Could not attach overview context: %s", exc)
        return context
    except Exception as exc:
        logger.warning("Could not load product context: %s", exc)
        return None


def _has_resolved_classification(db: Session, version_id: Optional[int]) -> bool:
    """True when a completed analysis recorded a RESOLVED classification.

    An UNRESOLVED (incomplete) classification still means the product
    category and dosage form are missing, so the chatbot's missing-
    information list keeps reporting them honestly.
    """
    if not version_id:
        return False
    try:
        from app.models.analysis_models import (
            Analysis,
            AnalysisStatus,
            AnalysisType,
        )

        rows = (
            db.query(Analysis.results)
            .filter(
                Analysis.product_version_id == version_id,
                Analysis.analysis_type.in_(
                    [
                        AnalysisType.PRODUCT_CLASSIFICATION,
                        AnalysisType.COMPREHENSIVE,
                    ]
                ),
                Analysis.status == AnalysisStatus.COMPLETED,
            )
            .all()
        )
        for (raw,) in rows:
            if not raw:
                continue
            try:
                data = json.loads(raw)
            except (TypeError, ValueError):
                continue
            classification = data.get("product_classification") or {}
            if classification.get("category") and classification.get(
                "status"
            ) != "UNRESOLVED":
                return True
        return False
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Could not check classification state: %s", exc)
        return False
