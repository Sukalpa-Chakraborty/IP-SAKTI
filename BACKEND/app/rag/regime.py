"""
The explicit India vs International jurisdiction switch (SIH requirement A1).

The problem statement asks for::

    an explicit jurisdiction switch so that answers are never conflated ...
    a jurisdiction toggle (India vs international) with the two answer-sets
    kept visibly separate

This module owns that concept. It is deliberately separate from
``app.rag.jurisdiction_scope``: that module scopes an answer to the *target
markets* recorded on a Product Passport (a market-entry question about
Germany must not cite Chinese law), whereas this module implements the
*regime switch* itself - the national (India) layer versus the international
layer. The two compose: a request can be scoped to Germany/EU *and* answered
in the international regime.

Why post-filtering and not a SQL predicate
-------------------------------------------
``source_documents.jurisdiction`` is nullable - every privately uploaded
document has ``NULL`` because it carries no regime. A SQL
``jurisdiction = ANY(...)`` therefore silently drops the user's own
documents whenever a regime filter is active. Regime membership is applied
after retrieval instead (see ``JurisdictionScopeTracker``), which keeps the
user's own evidence in both answer-sets while separating the two *corpus*
layers.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

# -------------------------------------------------------
# Vocabulary
# -------------------------------------------------------

REGIME_INDIA = "india"
REGIME_INTERNATIONAL = "international"

#: Toggle positions accepted by the chat API.
JURISDICTION_MODES = ("india", "international", "both")

#: How each regime is headed in the UI. The labels are deliberately explicit
#: about "national" / "international" so a reader can never mistake one set
#: for the other.
REGIME_LABELS: Dict[str, str] = {
    REGIME_INDIA: "India — national regime",
    REGIME_INTERNATIONAL: "International — foreign & treaty regimes",
}

#: Short heading used inside the rendered answer text.
REGIME_HEADINGS: Dict[str, str] = {
    REGIME_INDIA: "INDIA (NATIONAL REGIME)",
    REGIME_INTERNATIONAL: "INTERNATIONAL REGIME",
}

#: One-line explanation rendered under each answer-set so the separation is
#: self-evident even to someone who never touched the toggle.
REGIME_DESCRIPTIONS: Dict[str, str] = {
    REGIME_INDIA: (
        "Answers drawn only from Indian law: the Patents Act 1970 (as amended, "
        "with the 2024 Rules), Geographical Indications, Trade Marks, Designs, "
        "Copyright, Plant Varieties, the Biological Diversity Act 2002 (as "
        "amended in 2023) with its 2024 Rules, the Drugs and Cosmetics Act, "
        "the Drugs and Magic Remedies (Objectionable Advertisements) Act, "
        "FSSAI Ayurveda-Aahar rules, the TKDL and Indian case law."
    ),
    REGIME_INTERNATIONAL: (
        "Answers drawn only from outside India: TRIPS, the Convention on "
        "Biological Diversity and the Nagoya Protocol, the WIPO Treaty on "
        "Genetic Resources and Associated Traditional Knowledge (2024), the "
        "PCT, the Madrid and Hague systems, the Budapest Treaty, and the "
        "herbal-product market-access regimes of key export markets."
    ),
}

#: Jurisdiction values (``source_documents.jurisdiction``) that belong to the
#: Indian national regime. Compared case-insensitively after normalisation.
_INDIA_VALUES = {
    "india",
    "ind",
    "in",
    "भारत",
}

#: Jurisdictions that are explicitly *not* India but are still recognisably
#: single-country rather than treaty-level. Listed so the international set
#: is described accurately rather than guessed at.
_KNOWN_FOREIGN_VALUES = {
    "united states", "us", "u.s.", "usa", "united states of america",
    "china", "germany", "deutschland", "de", "eu", "european union",
    "japan", "uk", "united kingdom", "france", "wipo", "pdt",
}


def normalise_jurisdiction(value: Optional[str]) -> str:
    """Lower-cased, whitespace-trimmed jurisdiction string ('' when unset)."""
    if not value:
        return ""
    return str(value).strip().casefold()


def regime_for(jurisdiction: Optional[str]) -> Optional[str]:
    """
    Which regime a ``source_documents.jurisdiction`` value belongs to.

    Returns ``REGIME_INDIA``, ``REGIME_INTERNATIONAL``, or ``None`` when the
    document carries no jurisdiction tag at all (private uploads, and the few
    corpus rows ``scripts/fix_corpus_jurisdictions.py`` could not classify).
    ``None`` means "undetermined" - it is never silently counted as either
    regime, which is what keeps the two answer-sets honest.
    """
    value = normalise_jurisdiction(jurisdiction)
    if not value:
        return None
    if value in _INDIA_VALUES:
        return REGIME_INDIA
    return REGIME_INTERNATIONAL


def regimes_for_mode(mode: Optional[str]) -> List[str]:
    """
    Ordered list of regimes the request must answer.

    ``"both"`` (the default the UI sends) yields both, in the fixed order
    India -> International, so the two answer-sets always render in the same
    order regardless of how the payload was built. An unknown value falls
    back to both rather than answering nothing.
    """
    cleaned = (mode or "both").strip().casefold()
    if cleaned == REGIME_INDIA:
        return [REGIME_INDIA]
    if cleaned == REGIME_INTERNATIONAL:
        return [REGIME_INTERNATIONAL]
    return [REGIME_INDIA, REGIME_INTERNATIONAL]


def chunk_in_regime(
    jurisdiction: Optional[str],
    regime: str,
    *,
    is_private: bool = False,
) -> bool:
    """
    Whether a retrieved chunk may appear in ``regime``'s answer-set.

    * Private (user-uploaded) documents are always admitted: they are the
      user's own evidence, not part of either public regime, and withholding
      them because a toggle was flipped would break the
      "search my private documents" contract.
    * Untagged public documents are admitted to **neither** set. Admitting
      them to both would duplicate them; admitting them to one would
      mis-assign them. They surface instead through the answer-set's
      ``untagged_source_count`` so the gap is visible rather than hidden.
    """
    if is_private:
        return True
    return regime_for(jurisdiction) == regime


def describe_regime(regime: str) -> Dict[str, str]:
    """``{regime, label, heading, description}`` for one answer-set."""
    return {
        "regime": regime,
        "label": REGIME_LABELS.get(regime, regime),
        "heading": REGIME_HEADINGS.get(regime, regime.upper()),
        "description": REGIME_DESCRIPTIONS.get(regime, ""),
    }


def split_by_regime(
    jurisdictions: Iterable[Optional[str]],
) -> Dict[str, int]:
    """Tally how many tagged sources fall in each regime (diagnostics)."""
    tally = {REGIME_INDIA: 0, REGIME_INTERNATIONAL: 0, "untagged": 0}
    for value in jurisdictions:
        regime = regime_for(value)
        if regime is None:
            tally["untagged"] += 1
        else:
            tally[regime] += 1
    return tally


def opposite_regime(regime: str) -> str:
    """The regime *not* selected - used for the exclusion notice."""
    if regime == REGIME_INDIA:
        return REGIME_INTERNATIONAL
    return REGIME_INDIA


def exclusion_notice(regime: str, excluded_count: int = 0) -> str:
    """
    The standing sentence that proves the two sets were kept apart.

    Rendered under every single-regime answer so a reader can see, without
    having to trust us, that the other layer was withheld rather than merged.
    """
    other = REGIME_LABELS.get(opposite_regime(regime), "the other regime")
    notice = (
        f"Scope of this answer-set: {REGIME_LABELS.get(regime, regime)}. "
        f"Sources belonging to {other} were excluded and are not blended in. "
        "Switch the jurisdiction toggle to read that layer separately."
    )
    if excluded_count:
        notice += f" ({excluded_count} out-of-regime source(s) filtered out.)"
    return notice


def available_regimes(regimes: Sequence[str]) -> List[str]:
    """Deduplicate while preserving the order the caller asked for."""
    seen: List[str] = []
    for regime in regimes:
        if regime in REGIME_LABELS and regime not in seen:
            seen.append(regime)
    return seen or [REGIME_INDIA, REGIME_INTERNATIONAL]
