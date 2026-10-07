"""
Jurisdiction toggle (SIH requirement A1) + citation confidence (A3).

The problem statement asks for:

    "an explicit jurisdiction switch so that answers are never conflated ...
     a jurisdiction toggle (India vs international) with the two answer-sets
     kept visibly separate"

and

    "mandatory source citations with a confidence indicator"

What is asserted here:

  * ``jurisdiction_mode="india"``           -> ONE answer-set, Indian sources only
  * ``jurisdiction_mode="international"``   -> ONE answer-set, non-Indian sources only
  * ``jurisdiction_mode="both"``            -> TWO independently generated answer-sets
                                               with distinct headings, citations that
                                               never cross regimes, and an explicit
                                               record of what was withheld
  * omitted                                  -> legacy combined pass, unchanged
  * every citation carries a confidence score/label
  * the regime helpers classify jurisdictions the way the API claims

No network: the LLM is scripted through ``get_llm_provider`` at the router
boundary. Retrieval is the real (SQLite keyword arm) hybrid pipeline.
"""
import json
import uuid

from app.llm.base import LLMProvider
from app.models.rag_models import SourceChunk, SourceDocument
from app.rag.regime import (
    REGIME_INDIA,
    REGIME_INTERNATIONAL,
    chunk_in_regime,
    regimes_for_mode,
    regime_for,
)


QUESTION = "Which licence do I need for this product?"

# All three seeded documents deliberately contain the words the test query
# asks about ("licence", "product"). The keyword arm scores on terms, so if
# a document did not match it would simply never be retrieved - and a
# regime that never saw a source cannot be shown to have excluded it.
INDIA_TEXT = (
    "An Ayurvedic medicine manufactured in India requires a manufacturing "
    "licence granted by the State Licensing Authority under the Drugs and "
    "Cosmetics Act, 1940 before the product may be sold."
)
INTL_TEXT = (
    "Under TRIPS Article 27 patents shall be available for inventions in all "
    "fields of technology, and a licence granted under foreign law or the "
    "Patent Cooperation Treaty governs the international filing of the same "
    "product."
)
UNTAGGED_TEXT = (
    "General background notes about the licence a product needs before it "
    "may be placed on any market."
)


# -------------------------------------------------------
# Helpers (same conventions as the other chat suites)
# -------------------------------------------------------

def seed_corpus(db_session, title, text, *, jurisdiction=None,
                source_type="official_guidance"):
    """Insert an indexed public document + one chunk directly."""
    doc = SourceDocument(
        title=title,
        source_type=source_type,
        language="en",
        is_public=True,
        status="INDEXED",
        jurisdiction=jurisdiction,
        document_hash=uuid.uuid4().hex,
        chunk_count=1,
    )
    db_session.add(doc)
    db_session.flush()
    chunk = SourceChunk(
        document_id=doc.id,
        chunk_index=0,
        text=text,
        page_number=1,
        metadata_json=json.dumps({"page": 1}),
    )
    db_session.add(chunk)
    db_session.commit()
    return doc, chunk


def seed_both_regimes(db_session):
    """One clearly Indian source and one clearly international source."""
    india_doc, india_chunk = seed_corpus(
        db_session,
        "India Drug Licensing Guide",
        INDIA_TEXT,
        jurisdiction="India",
    )
    intl_doc, intl_chunk = seed_corpus(
        db_session,
        "TRIPS and PCT Handbook",
        INTL_TEXT,
        jurisdiction="International",
    )
    return india_doc, india_chunk, intl_doc, intl_chunk


class ScriptedLLM(LLMProvider):
    """Returns queued JSON responses in order (the last one repeats)."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = 0
        self.system_prompts = []
        self.user_prompts = []

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        self.calls += 1
        self.system_prompts.append(system_prompt)
        self.user_prompts.append(user_prompt)
        if len(self.responses) > 1:
            return self.responses.pop(0)
        if self.responses:
            return self.responses[0]
        return json.dumps({"sections": []})


def cite(chunk, doc, source_type="official_guidance"):
    return {
        "chunk_id": chunk.id,
        "document_id": doc.id,
        "title": doc.title,
        "source_type": source_type,
        "relevant_text": chunk.text[:120],
    }


def answer_json(citations, text):
    return json.dumps({
        "answer": text,
        "insufficient_evidence": False,
        "citations": citations,
        "warnings": [],
    })


def chat(client, auth_headers, payload):
    return client.post("/api/assistant/chat", json=payload, headers=auth_headers)


def payload(**extra):
    body = {"message": QUESTION, "include_my_documents": True}
    body.update(extra)
    return body


# -------------------------------------------------------
# Regime classification helpers
# -------------------------------------------------------

class TestRegimeClassification:
    def test_indian_values_map_to_the_national_regime(self):
        assert regime_for("India") == REGIME_INDIA
        assert regime_for("india") == REGIME_INDIA
        assert regime_for("  INDIA ") == REGIME_INDIA

    def test_foreign_and_treaty_values_map_to_international(self):
        assert regime_for("United States") == REGIME_INTERNATIONAL
        assert regime_for("China") == REGIME_INTERNATIONAL
        assert regime_for("International") == REGIME_INTERNATIONAL

    def test_untagged_is_undetermined_not_silently_assigned(self):
        # A document with no jurisdiction must not be counted as either
        # layer - guessing here is exactly how the two sets get conflated.
        assert regime_for(None) is None
        assert regime_for("") is None

    def test_private_documents_stay_available_to_both_regimes(self):
        assert chunk_in_regime(None, REGIME_INDIA, is_private=True) is True
        assert chunk_in_regime(None, REGIME_INTERNATIONAL, is_private=True) is True

    def test_untagged_public_documents_join_neither_regime(self):
        assert chunk_in_regime(None, REGIME_INDIA, is_private=False) is False
        assert chunk_in_regime(None, REGIME_INTERNATIONAL, is_private=False) is False

    def test_mode_expansion_is_ordered_and_defaults_to_both(self):
        assert regimes_for_mode("india") == [REGIME_INDIA]
        assert regimes_for_mode("international") == [REGIME_INTERNATIONAL]
        assert regimes_for_mode("both") == [REGIME_INDIA, REGIME_INTERNATIONAL]
        assert regimes_for_mode(None) == [REGIME_INDIA, REGIME_INTERNATIONAL]
        assert regimes_for_mode("nonsense") == [REGIME_INDIA, REGIME_INTERNATIONAL]


# -------------------------------------------------------
# A1 - the toggle itself
# -------------------------------------------------------

class TestIndiaMode:
    def test_returns_one_national_answer_set(self, client, auth_headers,
                                             db_session, monkeypatch):
        india_doc, india_chunk, intl_doc, intl_chunk = seed_both_regimes(db_session)
        fake = ScriptedLLM(
            answer_json(
                [cite(india_chunk, india_doc)],
                "A manufacturing licence under the Drugs and Cosmetics Act is required.",
            )
        )
        monkeypatch.setattr("app.routers.assistant.get_llm_provider", lambda: fake)

        resp = chat(client, auth_headers, payload(jurisdiction_mode="india"))
        assert resp.status_code == 200, resp.text
        body = resp.json()

        assert body["jurisdiction_mode"] == "india"
        assert body["regimes_separated"] is False
        assert len(body["answer_sets"]) == 1

        only = body["answer_sets"][0]
        assert only["regime"] == REGIME_INDIA
        assert "national regime" in only["label"]
        assert only["heading"]
        # The answer-set must say what it withheld, not just omit it silently.
        assert "International" in only["scope_notice"]
        assert only["excluded_source_count"] >= 1

        assert body["citations"], "the Indian source must still be cited"
        for citation in body["citations"]:
            assert citation["jurisdiction"] == "India"

    def test_international_sources_are_not_retrieved(self, client, auth_headers,
                                                     db_session, monkeypatch):
        seed_both_regimes(db_session)
        fake = ScriptedLLM(
            answer_json([], "Insufficient evidence for this regime.")
        )
        monkeypatch.setattr("app.routers.assistant.get_llm_provider", lambda: fake)

        resp = chat(client, auth_headers, payload(jurisdiction_mode="india"))
        body = resp.json()
        for citation in body["citations"]:
            assert citation["jurisdiction"] != "International"
        # ...and the model never even saw the international chunk.
        for prompt in getattr(fake, "user_prompts", []):
            assert "TRIPS and PCT Handbook" not in prompt


class TestInternationalMode:
    def test_returns_one_international_answer_set(self, client, auth_headers,
                                                  db_session, monkeypatch):
        india_doc, india_chunk, intl_doc, intl_chunk = seed_both_regimes(db_session)
        fake = ScriptedLLM(
            answer_json(
                [cite(intl_chunk, intl_doc)],
                "TRIPS Article 27 and the PCT route govern the international layer.",
            )
        )
        monkeypatch.setattr("app.routers.assistant.get_llm_provider", lambda: fake)

        resp = chat(client, auth_headers, payload(jurisdiction_mode="international"))
        assert resp.status_code == 200, resp.text
        body = resp.json()

        assert body["jurisdiction_mode"] == "international"
        assert len(body["answer_sets"]) == 1

        only = body["answer_sets"][0]
        assert only["regime"] == REGIME_INTERNATIONAL
        assert "national regime" not in only["label"]
        assert "India" in only["scope_notice"]
        assert only["excluded_source_count"] >= 1

        for citation in body["citations"]:
            assert citation["jurisdiction"] != "India"

        # The Indian source must not leak into this pass's prompt either.
        for prompt in getattr(fake, "user_prompts", []):
            assert "India Drug Licensing Guide" not in prompt


class TestBothMode:
    def test_produces_two_visibly_separate_answer_sets(
        self, client, auth_headers, db_session, monkeypatch
    ):
        india_doc, india_chunk, intl_doc, intl_chunk = seed_both_regimes(db_session)
        fake = ScriptedLLM(
            answer_json(
                [cite(india_chunk, india_doc)],
                "INDIA (NATIONAL REGIME)\n\nA manufacturing licence is required.",
            ),
            answer_json(
                [cite(intl_chunk, intl_doc)],
                "INTERNATIONAL REGIME\n\nTRIPS and the PCT govern this layer.",
            ),
        )
        monkeypatch.setattr("app.routers.assistant.get_llm_provider", lambda: fake)

        resp = chat(client, auth_headers, payload(jurisdiction_mode="both"))
        assert resp.status_code == 200, resp.text
        body = resp.json()

        assert body["jurisdiction_mode"] == "both"
        assert body["regimes_separated"] is True
        assert len(body["answer_sets"]) == 2, "the two layers must be separate objects"

        india_set, intl_set = body["answer_sets"]
        assert india_set["regime"] == REGIME_INDIA
        assert intl_set["regime"] == REGIME_INTERNATIONAL
        assert india_set["heading"] != intl_set["heading"]
        assert india_set["label"] != intl_set["label"]
        assert india_set["answer"] and intl_set["answer"]
        assert india_set["answer"] != intl_set["answer"]

        # Each layer is generated independently: two LLM calls, each seeing
        # only its own regime's sources.
        assert fake.calls == 2

        # Citations never cross the line between the two sets.
        assert india_set["citations"] and intl_set["citations"]
        for citation in india_set["citations"]:
            assert citation["jurisdiction"] == "India"
        for citation in intl_set["citations"]:
            assert citation["jurisdiction"] == "International"

        # The combined payload carries both layers too, so a reader who only
        # looks at `answer` still cannot mistake one for the other.
        assert india_set["heading"] in body["answer"]
        assert intl_set["heading"] in body["answer"]

        # Both sets keep the standing disclaimer.
        assert any("does not constitute legal" in w for w in body["warnings"])
        assert any("does not constitute legal" in w for w in india_set["warnings"])
        assert any("does not constitute legal" in w for w in intl_set["warnings"])

    def test_the_two_passes_do_not_share_sources(self, client, auth_headers,
                                                 db_session, monkeypatch):
        india_doc, india_chunk, intl_doc, intl_chunk = seed_both_regimes(db_session)
        fake = ScriptedLLM(
            answer_json([cite(india_chunk, india_doc)], "National answer."),
            answer_json([cite(intl_chunk, intl_doc)], "International answer."),
        )
        monkeypatch.setattr("app.routers.assistant.get_llm_provider", lambda: fake)

        body = chat(client, auth_headers, payload(jurisdiction_mode="both")).json()
        india_ids = {c["chunk_id"] for c in body["answer_sets"][0]["citations"]}
        intl_ids = {c["chunk_id"] for c in body["answer_sets"][1]["citations"]}
        assert india_ids.isdisjoint(intl_ids)

    def test_untagged_public_sources_appear_in_neither_set(
        self, client, auth_headers, db_session, monkeypatch
    ):
        india_doc, india_chunk, intl_doc, intl_chunk = seed_both_regimes(db_session)
        seed_corpus(db_session, "Untagged General Notes", UNTAGGED_TEXT)
        fake = ScriptedLLM(
            answer_json([cite(india_chunk, india_doc)], "National answer."),
            answer_json([cite(intl_chunk, intl_doc)], "International answer."),
        )
        monkeypatch.setattr("app.routers.assistant.get_llm_provider", lambda: fake)

        body = chat(client, auth_headers, payload(jurisdiction_mode="both")).json()
        for answer_set in body["answer_sets"]:
            titles = [c["title"] for c in answer_set["citations"]]
            assert "Untagged General Notes" not in titles
        # ...and it was not silently absorbed into the combined answer either.
        combined_titles = [c["title"] for c in body["citations"]]
        assert "Untagged General Notes" not in combined_titles


# -------------------------------------------------------
# Backwards compatibility - no toggle means no behaviour change
# -------------------------------------------------------

class TestLegacyCombinedPass:
    def test_omitting_the_field_keeps_one_combined_answer(
        self, client, auth_headers, db_session, monkeypatch
    ):
        india_doc, india_chunk, intl_doc, intl_chunk = seed_both_regimes(db_session)
        fake = ScriptedLLM(
            answer_json(
                [cite(india_chunk, india_doc), cite(intl_chunk, intl_doc)],
                "Both regimes are discussed together.",
            )
        )
        monkeypatch.setattr("app.routers.assistant.get_llm_provider", lambda: fake)

        body = chat(client, auth_headers, payload()).json()
        assert body["jurisdiction_mode"] == "all"
        assert body["regimes_separated"] is False
        assert len(body["answer_sets"]) == 1
        assert body["answer_sets"][0]["regime"] == "all"
        assert fake.calls == 1, "no toggle must not double the LLM work"


# -------------------------------------------------------
# A3 - confidence indicator on every citation
# -------------------------------------------------------

class TestCitationConfidence:
    def test_confidence_is_populated_on_returned_citations(
        self, client, auth_headers, db_session, monkeypatch
    ):
        india_doc, india_chunk, _, _ = seed_both_regimes(db_session)
        fake = ScriptedLLM(
            answer_json(
                [cite(india_chunk, india_doc)],
                "A manufacturing licence is required.",
            )
        )
        monkeypatch.setattr("app.routers.assistant.get_llm_provider", lambda: fake)

        body = chat(client, auth_headers, payload()).json()
        assert body["citations"], "the assertion needs a citation to inspect"
        for citation in body["citations"]:
            score = citation["confidence_score"]
            label = citation["confidence_label"]
            if score is None:
                # Honest absence beats a made-up number, but the label must
                # agree with the score either way.
                assert label is None
                continue
            assert 0.0 <= score <= 1.0
            assert label in {"HIGH", "MEDIUM", "LOW"}

    def test_label_tiers_follow_the_documented_boundaries(self):
        from app.rag.citation import (
            CONFIDENCE_HIGH,
            CONFIDENCE_MEDIUM,
            confidence_label_for,
        )

        assert confidence_label_for(CONFIDENCE_HIGH) == "HIGH"
        assert confidence_label_for(CONFIDENCE_HIGH - 0.01) == "MEDIUM"
        assert confidence_label_for(CONFIDENCE_MEDIUM) == "MEDIUM"
        assert confidence_label_for(CONFIDENCE_MEDIUM - 0.01) == "LOW"
        assert confidence_label_for(0.0) == "LOW"
        assert confidence_label_for(1.0) == "HIGH"
        assert confidence_label_for(None) is None

    def test_reranker_logits_are_squashed_into_zero_one(self):
        from app.rag.citation import confidence_score_for, confidence_label_for
        from app.rag.schemas import RetrievedChunk

        def chunk_with(logit):
            return RetrievedChunk(
                chunk_id=1,
                document_id=1,
                title="t",
                source_type="official_guidance",
                text="text",
                rerank_score=logit,
            )

        # The cross-encoder emits raw logits, not probabilities.
        assert confidence_score_for(chunk_with(0.0)) == 0.5
        assert confidence_score_for(chunk_with(100.0)) > 0.99
        assert confidence_score_for(chunk_with(-100.0)) < 0.01
        assert confidence_label_for(confidence_score_for(chunk_with(6.0))) == "HIGH"
        assert confidence_label_for(confidence_score_for(chunk_with(-6.0))) == "LOW"

    def test_no_score_reports_none_instead_of_a_fake_number(self):
        from app.rag.citation import confidence_label_for, confidence_score_for
        from app.rag.schemas import RetrievedChunk

        # RRF rank-fusion values are not relevance probabilities; reporting
        # them as a confidence would invent precision we do not have.
        chunk = RetrievedChunk(
            chunk_id=1,
            document_id=1,
            title="t",
            source_type="official_guidance",
            text="text",
            rrf_score=1 / 61,
        )
        assert confidence_score_for(chunk) is None
        assert confidence_label_for(confidence_score_for(chunk)) is None

    def test_vector_cosine_is_used_when_the_reranker_is_unavailable(self):
        from app.rag.citation import confidence_score_for, confidence_label_for
        from app.rag.schemas import RetrievedChunk

        chunk = RetrievedChunk(
            chunk_id=1,
            document_id=1,
            title="t",
            source_type="official_guidance",
            text="text",
            vector_score=0.82,
        )
        assert confidence_score_for(chunk) == 0.82
        assert confidence_label_for(0.82) == "HIGH"
