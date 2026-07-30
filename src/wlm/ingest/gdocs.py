"""Google Docs read-only ingest with a paste-detection pass.

Why revision history matters: the corpus is only useful if the person actually *typed* it.
A doc full of pasted research is someone else's voice. Drive exposes revision metadata; the
Docs API exposes `suggestionsViewMode` and structural content but not keystroke provenance,
so `--typed-only` uses a heuristic: reconstruct each revision's text and treat any single
revision that adds a large contiguous block as a paste.

Setup:
  1. console.cloud.google.com -> new project -> enable "Google Drive API" and "Google Docs API"
  2. Create an OAuth client ID of type "Desktop app", download the JSON
  3. Put its path in GOOGLE_CLIENT_SECRETS (see .env.example)
  4. First run opens a browser once and caches a token at GOOGLE_TOKEN_PATH

Scope is read-only. Nothing is ever written back to Drive.
"""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path
from typing import Any

from wlm.ingest.local import doc_id
from wlm.ingest.normalize import looks_like_template, normalize_document, prose_ratio

SCOPES = [
    "https://www.googleapis.com/auth/drive.readonly",
    "https://www.googleapis.com/auth/documents.readonly",
]

# A single revision adding more than this many chars at once *faster than a person can type*
# is almost certainly a paste. Both conditions are required: Docs coalesces minutes of typing into
# one revision, so size alone would flag ordinary drafting.
PASTE_BLOCK_CHARS = 600
# Sustained typing tops out around 400 characters per minute. Above this, the characters did not
# arrive from a keyboard.
PASTE_CHARS_PER_MINUTE = 400
# How many consecutive revisions to sample. Must be consecutive: deltas between non-adjacent
# revisions sum up ordinary typing into fake "blocks" and would reject first-party prose.
MAX_REVISIONS_SAMPLED = 13


def _credentials():
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow

    token_path = Path(os.environ.get("GOOGLE_TOKEN_PATH", "token.json"))
    secrets_path = Path(os.environ.get("GOOGLE_CLIENT_SECRETS", "client_secret.json"))

    creds = None
    if token_path.exists():
        creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
        # Persist the refreshed token, or every run pays for a refresh round-trip.
        token_path.write_text(creds.to_json(), encoding="utf-8")
    if not creds or not creds.valid:
        if not secrets_path.exists():
            raise FileNotFoundError(
                f"OAuth client secrets not found at {secrets_path}. "
                "See the docstring in wlm/ingest/gdocs.py for the 4-step setup."
            )
        flow = InstalledAppFlow.from_client_secrets_file(str(secrets_path), SCOPES)
        creds = flow.run_local_server(port=0)
        token_path.write_text(creds.to_json(), encoding="utf-8")
    return creds


def _services():
    from googleapiclient.discovery import build

    creds = _credentials()
    return (
        build("drive", "v3", credentials=creds, cache_discovery=False),
        build("docs", "v1", credentials=creds, cache_discovery=False),
    )


def list_docs(drive, *, folder_id: str | None = None, owned_by_me: bool = True) -> list[dict]:
    q = ["mimeType='application/vnd.google-apps.document'", "trashed=false"]
    if owned_by_me:
        q.append("'me' in owners")
    if folder_id:
        q.append(f"'{folder_id}' in parents")
    out, token = [], None
    while True:
        resp = (
            drive.files()
            .list(
                q=" and ".join(q),
                fields="nextPageToken, files(id,name,createdTime,modifiedTime,owners(emailAddress))",
                pageSize=100,
                pageToken=token,
            )
            .execute()
        )
        out.extend(resp.get("files", []))
        token = resp.get("nextPageToken")
        if not token:
            return out


def doc_to_text(docs, file_id: str) -> str:
    """Flatten a Docs API document to plain text, preserving paragraph breaks."""
    doc = docs.documents().get(documentId=file_id).execute()
    parts: list[str] = []
    for el in doc.get("body", {}).get("content", []):
        para = el.get("paragraph")
        if not para:
            continue
        line = "".join(
            run.get("textRun", {}).get("content", "") for run in para.get("elements", [])
        )
        parts.append(line)
    return "".join(parts)


def paste_score(drive, file_id: str) -> float:
    """Fraction of the document's growth that arrived in large contiguous jumps.

    0.0 = typed incrementally. Near 1.0 = pasted in. Returns -1.0 when the provenance signal is
    unavailable, which the caller must treat as "unknown", never as "typed".

    Note on the size shortcut: Drive v3 only populates `revisions.size` for files with binary
    content stored in Drive. Native Google Docs do not have it, so we export each retained
    revision's plain text and diff successive lengths instead. Drive prunes most Docs revisions
    (only "keepForever"/named ones survive long), so for many documents this legitimately returns
    -1.0 -- the filter can only ever be a partial defense.
    """
    try:
        revs = (
            drive.revisions()
            .list(fileId=file_id, fields="revisions(id,modifiedTime,size)")
            .execute()
            .get("revisions", [])
        )
    except Exception as e:
        print(f"    [paste_score] revision list unavailable: {type(e).__name__}")
        return -1.0
    if len(revs) < 2:
        return -1.0

    # CONSECUTIVE revisions only, and the most recent window (that's where the drafting is).
    window = revs[-MAX_REVISIONS_SAMPLED:]
    creds = _credentials()  # hoisted: one call, never inside the per-revision loop

    samples: list[tuple[int, float]] = []  # (chars, epoch_minutes)
    for r in window:
        when = _epoch_minutes(r.get("modifiedTime"))
        n = r.get("size")
        if n is None:
            n = _revision_char_count(drive, file_id, r["id"], creds)
        if n is None or when is None:
            continue
        samples.append((int(n), when))

    if len(samples) < 2:
        return -1.0

    total, pasted = 0, 0
    for (n0, t0), (n1, t1) in zip(samples, samples[1:], strict=False):
        added = max(0, n1 - n0)
        if added == 0:
            continue
        total += added
        minutes = max(t1 - t0, 1e-6)
        if added >= PASTE_BLOCK_CHARS and (added / minutes) > PASTE_CHARS_PER_MINUTE:
            pasted += added
    if total <= 0:
        return -1.0
    return pasted / total


def _epoch_minutes(rfc3339: str | None) -> float | None:
    if not rfc3339:
        return None
    import datetime as dt

    try:
        return dt.datetime.fromisoformat(rfc3339.replace("Z", "+00:00")).timestamp() / 60.0
    except ValueError:
        return None


def _revision_char_count(drive, file_id: str, revision_id: str, creds) -> int | None:
    """Export one revision as plain text and return its length. None on any failure."""
    try:
        blob = (
            drive.revisions()
            .get(fileId=file_id, revisionId=revision_id, fields="exportLinks")
            .execute()
        )
        link = (blob.get("exportLinks") or {}).get("text/plain")
        if not link:
            return None
        req = urllib.request.Request(link)
        req.add_header("Authorization", f"Bearer {creds.token}")
        with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
            return len(resp.read().decode("utf-8", errors="replace"))
    except Exception:
        return None


def ingest_gdocs(
    *,
    folder_id: str | None = None,
    typed_only: bool = True,
    max_paste_score: float = 0.5,
    min_prose_ratio: float = 0.35,
    limit: int | None = None,
    register: str = "unknown",
) -> list[dict[str, Any]]:
    drive, docs_api = _services()
    files = list_docs(drive, folder_id=folder_id)
    if limit:
        files = files[:limit]
    print(f"[gdocs] {len(files)} candidate document(s)")

    out: list[dict[str, Any]] = []
    n_unknown_provenance = 0
    for f in files:
        raw = doc_to_text(docs_api, f["id"])
        text = normalize_document(raw)
        if not text or looks_like_template(text) or prose_ratio(text) < min_prose_ratio:
            print(f"  - skip {f['name']!r}: not first-party prose")
            continue
        ps = paste_score(drive, f["id"]) if typed_only else -1.0
        if typed_only and ps >= 0 and ps > max_paste_score:
            print(f"  - skip {f['name']!r}: paste_score={ps:.2f}")
            continue
        if typed_only and ps < 0:
            n_unknown_provenance += 1
        out.append(
            {
                "doc_id": doc_id(f["id"], text),
                "source": f["name"],
                "origin": "gdocs",
                "gdoc_id": f["id"],
                "register": register,
                "paste_score": ps,
                "created": f.get("createdTime"),
                "text": text,
            }
        )
    print(f"[gdocs] kept {len(out)} document(s)")
    if n_unknown_provenance:
        print(
            f"[gdocs] WARNING provenance unknown for {n_unknown_provenance} of {len(out)} kept "
            "documents — Drive had no usable revision history for them, so `--typed-only` could "
            "not verify they were typed rather than pasted. Pasted text is someone else's voice, "
            "which is the one contamination this project cannot recover from. Spot-check those "
            "documents by hand, or restrict ingest to a folder you know you drafted in."
        )
    return out
