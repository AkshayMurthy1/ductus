"""Normalization: strip everything the author did not type.

Boilerplate is the single biggest source of fake style signal. If half the corpus ends with
"Best, Akshay" the adapter will learn to sign off, score well on the verifier for the wrong
reason, and teach you nothing about voice.
"""

from __future__ import annotations

import re

# --- quoted replies / forwarded mail ------------------------------------------------------
QUOTE_MARKERS = [
    # Drop the whole quoted line, not just its ">" prefix -- the quoted text is someone
    # else's voice, which is exactly the contamination we're removing.
    re.compile(r"^[ \t]*>+.*(?:\n|$)", re.M),
    re.compile(r"^-+\s*Original Message\s*-+.*", re.S | re.I),
    re.compile(r"^-+\s*Forwarded message\s*-+.*", re.S | re.I),
    re.compile(r"^On .{5,80}\bwrote:\s*$.*", re.S | re.M),
    re.compile(r"^From:\s.*?^Subject:\s.*?$", re.S | re.M),
]

# --- sign-offs ---------------------------------------------------------------------------
SIGNOFF = re.compile(
    r"\n+\s*(best|best regards|regards|thanks|thank you|cheers|sincerely|warmly|yours|"
    r"talk soon|all the best|kind regards|respectfully)\s*[,!.]?\s*\n.{0,80}$",
    re.I | re.S,
)
SENT_FROM = re.compile(r"\n+\s*(sent from my .{0,40}|get outlook for .{0,20})\s*$", re.I)
UNSUBSCRIBE = re.compile(r"\n.*\b(unsubscribe|view this email in your browser)\b.*", re.I)
CONFIDENTIALITY = re.compile(
    r"\n+.*\b(this (e-?mail|message) (and any attachments )?is (intended|confidential)|"
    r"privileged and confidential)\b.*", re.I | re.S,
)

# --- document furniture ------------------------------------------------------------------
PAGE_NUM = re.compile(r"^\s*(page\s+)?\d+\s*(of\s+\d+)?\s*$", re.I | re.M)
HEADING_ONLY = re.compile(r"^\s*[A-Z][A-Z0-9 \-&/]{3,60}\s*$", re.M)
BULLET_RUN = re.compile(r"(?:^\s*(?:[-*•●]|\d+[.)])\s+.*\n?){3,}", re.M)
TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$", re.M)
URL = re.compile(r"https?://\S+|www\.\S+")
CODE_FENCE = re.compile(r"```.*?```", re.S)

WHITESPACE_RUN = re.compile(r"[ \t]{2,}")
NEWLINE_RUN = re.compile(r"\n{3,}")

# Template phrases that mark a document as not-first-party prose.
TEMPLATE_HINTS = (
    "lorem ipsum",
    "click here to",
    "[insert",
    "tbd",
    "xxx",
    "copyright ©",
    "all rights reserved",
)


def normalize_document(text: str, *, drop_bullet_runs: bool = True) -> str:
    """Return only author-typed running prose. Idempotent."""
    if not text:
        return ""

    t = text.replace("\r\n", "\n").replace("\r", "\n")
    t = CODE_FENCE.sub(" ", t)

    for pat in QUOTE_MARKERS:
        t = pat.sub("", t)

    t = CONFIDENTIALITY.sub("", t)
    t = UNSUBSCRIBE.sub("", t)
    t = SENT_FROM.sub("", t)
    # Apply twice: a sign-off often sits above a "Sent from my iPhone".
    t = SIGNOFF.sub("", t)
    t = SIGNOFF.sub("", t)

    t = TABLE_ROW.sub("", t)
    t = PAGE_NUM.sub("", t)
    if drop_bullet_runs:
        # Bullet lists are outline-shaped, not voice-shaped. They teach the model list syntax.
        t = BULLET_RUN.sub("\n", t)

    # Keep the fact that a link was there, drop the URL string itself (pure content noise).
    t = URL.sub("<link>", t)

    # Smart quotes / dashes are style-bearing; keep them, only unify the exotic ones.
    t = t.replace(" ", " ").replace("​", "")

    t = WHITESPACE_RUN.sub(" ", t)
    t = NEWLINE_RUN.sub("\n\n", t)
    return t.strip()


def looks_like_template(text: str, *, min_words: int = 80) -> bool:
    """Cheap first-party filter: templates, forms, and stubs are not this person's voice."""
    low = text.lower()
    if any(h in low for h in TEMPLATE_HINTS):
        return True
    words = text.split()
    if len(words) < min_words:
        return True
    # A wall of very short lines is a form or an outline, not prose.
    lines = [ln for ln in text.split("\n") if ln.strip()]
    if lines and sum(1 for ln in lines if len(ln.split()) <= 4) / len(lines) > 0.6:
        return True
    # Prose has sentence-ending punctuation. Slides and forms mostly don't.
    if text.count(".") + text.count("?") + text.count("!") < len(words) / 60:
        return True
    return False


def prose_ratio(text: str) -> float:
    """Fraction of non-empty lines that look like running prose. Useful as an ingest filter."""
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    if not lines:
        return 0.0
    prose = sum(1 for ln in lines if len(ln.split()) >= 8 and not HEADING_ONLY.fullmatch(ln))
    return prose / len(lines)
