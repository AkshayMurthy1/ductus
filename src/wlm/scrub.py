"""PII + topic-entity scrubbing of training targets.

Two jobs at once (plan §4.3):
  1. Privacy: names, emails, phones, addresses, secrets never reach the trainer.
  2. Style hygiene: replacing topic nouns with typed placeholders is the main defense against
     the MLP learning *what* the person writes about instead of *how*. The MLP holds the most
     parameters, so it is exactly where topic leak shows up.

Deliberate design choice: replace with typed placeholders (<PERSON>, <ORG>) rather than delete.
Deleting changes sentence length and syntax -- the very things we're trying to learn. A
placeholder holds the slot so cadence survives.

Every removal is logged to data/interim/scrub_log.jsonl so you can audit what left the corpus.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]{2,}\b")
PHONE = re.compile(r"(?:\+?\d{1,2}[\s.-])?\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}\b")
SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
CARD = re.compile(r"\b(?:\d[ -]?){13,16}\b")
IPV4 = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
URLISH = re.compile(r"\b(?:https?://|www\.)\S+")
HANDLE = re.compile(r"(?<![\w])@[A-Za-z_][A-Za-z0-9_]{2,}")
SECRET = re.compile(
    r"\b(?:sk-[A-Za-z0-9_-]{16,}|gh[pousr]_[A-Za-z0-9]{16,}|AKIA[0-9A-Z]{16}|"
    r"AIza[0-9A-Za-z_-]{20,}|xox[baprs]-[0-9A-Za-z-]{10,})"
)
STREET = re.compile(
    r"\b\d{1,5}\s+(?:[A-Z][a-z]+\s){1,3}"
    r"(?:St|Street|Ave|Avenue|Rd|Road|Blvd|Boulevard|Ln|Lane|Dr|Drive|Ct|Court|Way|Pl|Place)\b\.?",
)
ZIP = re.compile(r"\b\d{5}(?:-\d{4})?\b")

REGEX_RULES: list[tuple[str, re.Pattern[str]]] = [
    ("SECRET", SECRET),
    ("EMAIL", EMAIL),
    ("SSN", SSN),
    ("CARD", CARD),
    ("PHONE", PHONE),
    ("IP", IPV4),
    ("URL", URLISH),
    ("ADDRESS", STREET),
    ("HANDLE", HANDLE),
]

# spaCy labels we replace when NER is available. MONEY/DATE/CARDINAL are intentionally kept:
# how someone renders numbers and dates ("the 3rd" vs "March 3rd") is style, not content.
NER_LABELS = {
    "PERSON": "PERSON",
    "ORG": "ORG",
    "GPE": "PLACE",
    "LOC": "PLACE",
    "FAC": "PLACE",
    "PRODUCT": "THING",
    "WORK_OF_ART": "TITLE",
    "EVENT": "EVENT",
    "NORP": "GROUP",
    "LAW": "THING",
    "LANGUAGE": "THING",
}

# Terms that trip the NER but are load-bearing vocabulary rather than identity. Technical
# eponyms dominate academic prose, and en_core_web_sm reads them as people: scrubbing "Euler" out
# of "Euler-Lagrange equations" corrupts a sentence the author wrote and teaches the model that
# its own subject matter is unspeakable. Extend via `data.never_scrub` in YAML.
DEFAULT_NEVER_SCRUB = frozenset({
    "euler", "lagrange", "lagrangian", "fourier", "gauss", "gaussian", "newton", "newtonian",
    "bayes", "bayesian", "hamiltonian", "laplace", "taylor", "riemann", "hilbert", "markov",
    "poisson", "boltzmann", "maxwell", "hessian", "jacobian", "navier", "stokes", "dirac",
    "fermi", "noether", "hooke", "bernoulli", "kalman", "monte carlo", "black-scholes",
    "python", "numpy", "scipy", "pytorch", "tensorflow", "matlab", "ansys", "julia", "linux",
})

_WORDCHAR = re.compile(r"[0-9A-Za-z]")
# Only true compound joiners. En/em dashes are punctuation in this kind of prose -- treating
# them as joiners silently suppresses real scrubs ("Kevin Esvelt—the leader of..." kept a name).
_HYPHENS = "-‐‑"


def _span_is_scrubbable(text: str, ent: Any, never: frozenset[str] | set[str]) -> bool:
    """Reject entity spans that would damage a word the author actually wrote.

    Three failure modes, all observed on real text with en_core_web_sm:

    1. A span starting mid-word. The tokenizer splits "wasn't" into "was" + "n't" and the model
       tags "n't" as a GPE, so a faithful replacement yields "was<PLACE>" -- a destroyed
       contraction, and contraction rate is a feature the stylometry and verifier measure.
    2. A span covering only part of a hyphenated compound: "Euler-Lagrange" -> "<PERSON>-Lagrange".
       When the entity spans the whole compound ("UW-Milwaukee") it is a real name and still goes.
    3. An all-caps PERSON, which is an acronym or a piece of software ("ANSYS"), never a name.
    """
    if ent.text.strip().lower() in never:
        return False
    if ent.label_ == "PERSON" and ent.text.isupper() and len(ent.text.strip()) > 1:
        return False

    before = text[ent.start_char - 1] if ent.start_char > 0 else ""
    after = text[ent.end_char] if ent.end_char < len(text) else ""

    # Left edge: inside a word, or trailing a hyphen/apostrophe that follows one.
    if before and (
        _WORDCHAR.match(before)
        or (
            before in _HYPHENS + "'’"
            and ent.start_char >= 2
            and _WORDCHAR.match(text[ent.start_char - 2])
        )
    ):
        return False
    # Right edge: inside a word, or leading a hyphen that precedes one. Apostrophes are allowed
    # here so possessives ("Akshay's") still scrub.
    if after and (
        _WORDCHAR.match(after)
        or (
            after in _HYPHENS
            and ent.end_char + 1 < len(text)
            and _WORDCHAR.match(text[ent.end_char + 1])
        )
    ):
        return False
    return True


_NLP = None
_NLP_TRIED = False


def _nlp():
    """Lazy spaCy load. Absent model is a warning, not an error -- regex rules still run."""
    global _NLP, _NLP_TRIED
    if _NLP_TRIED:
        return _NLP
    _NLP_TRIED = True
    try:
        import spacy

        _NLP = spacy.load("en_core_web_sm", disable=["lemmatizer", "textcat"])
    except Exception:
        print(
            "[scrub] spaCy en_core_web_sm unavailable -- entity scrubbing degraded to regex "
            "only. Run: python -m spacy download en_core_web_sm"
        )
        _NLP = None
    return _NLP


def scrub_text(
    text: str,
    *,
    entities: bool = True,
    extra_terms: list[str] | None = None,
    never_scrub: list[str] | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """Return (scrubbed_text, removals). Placeholders preserve token slots and cadence."""
    never = DEFAULT_NEVER_SCRUB | {t.strip().lower() for t in (never_scrub or []) if t.strip()}
    removals: list[dict[str, Any]] = []
    out = text

    # User-supplied always-scrub list (own name, employer, partner's name, project codenames).
    for term in extra_terms or []:
        term = term.strip()
        if len(term) < 3:
            continue
        pat = re.compile(rf"\b{re.escape(term)}\b", re.I)
        if pat.search(out):
            removals.append({"kind": "CUSTOM", "text": term, "n": len(pat.findall(out))})
            out = pat.sub("<NAME>", out)

    for kind, pat in REGEX_RULES:
        found = pat.findall(out)
        if found:
            removals.append({"kind": kind, "n": len(found)})
            out = pat.sub(f"<{kind}>", out)

    if entities:
        nlp = _nlp()
        if nlp is not None:
            doc = nlp(out)
            # Right-to-left so earlier spans keep their offsets.
            spans = [e for e in doc.ents if e.label_ in NER_LABELS]
            for ent in sorted(spans, key=lambda e: e.start_char, reverse=True):
                if not _span_is_scrubbable(out, ent, never):
                    removals.append({"kind": ent.label_, "text": ent.text, "n": 0, "kept": True})
                    continue
                ph = f"<{NER_LABELS[ent.label_]}>"
                removals.append({"kind": ent.label_, "text": ent.text, "n": 1})
                out = out[: ent.start_char] + ph + out[ent.end_char :]

    # ZIPs last: they overlap with harmless numbers, so only strip when near an address slot.
    if "<ADDRESS>" in out or "<PLACE>" in out:
        n = len(ZIP.findall(out))
        if n:
            removals.append({"kind": "ZIP", "n": n})
            out = ZIP.sub("<ZIP>", out)

    out = re.sub(r"(<(?:[A-Z]+)>)(\s+\1)+", r"\1", out)  # collapse placeholder runs
    out = re.sub(r"[ \t]{2,}", " ", out)
    return out, removals


PLACEHOLDER = re.compile(r"<[A-Z]+>")


def placeholder_density(text: str) -> float:
    words = max(1, len(text.split()))
    return len(PLACEHOLDER.findall(text)) / words


def scrub_records(
    rows: list[dict[str, Any]],
    *,
    entities: bool = True,
    extra_terms: list[str] | None = None,
    never_scrub: list[str] | None = None,
    max_placeholder_density: float = 0.12,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Scrub every chunk. Returns (kept, log, summary).

    Chunks that end up mostly placeholders are dropped: after heavy scrubbing they are no longer
    natural prose, and training on them teaches the model to emit angle brackets.
    """
    kept: list[dict[str, Any]] = []
    log: list[dict[str, Any]] = []
    dropped = 0
    counts: Counter[str] = Counter()

    for r in rows:
        scrubbed, removals = scrub_text(
            r["text"], entities=entities, extra_terms=extra_terms, never_scrub=never_scrub
        )
        for rem in removals:
            counts[rem["kind"]] += rem.get("n", 1)
        density = placeholder_density(scrubbed)
        log.append(
            {
                "chunk_id": r["chunk_id"],
                "placeholder_density": round(density, 4),
                # Log kinds and counts, not the removed strings -- the log itself would
                # otherwise become the PII file we were trying to avoid.
                "removed": [{"kind": x["kind"], "n": x.get("n", 1)} for x in removals],
            }
        )
        if density > max_placeholder_density:
            dropped += 1
            continue
        kept.append({**r, "text": scrubbed, "text_len": len(scrubbed.split())})

    summary = {
        "in": len(rows),
        "kept": len(kept),
        "dropped_over_dense": dropped,
        "removals_by_kind": dict(counts.most_common()),
        "spacy_ner": _nlp() is not None,
    }
    return kept, log, summary
