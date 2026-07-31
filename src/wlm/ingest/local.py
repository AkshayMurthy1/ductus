"""Local-file ingest: .txt / .md / .docx. Use this to develop the pipeline before wiring OAuth."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from wlm.ingest.normalize import looks_like_template, normalize_document, prose_ratio

SUPPORTED = {".txt", ".md", ".markdown", ".docx"}


def _read_docx(path: Path) -> str:
    try:
        import docx  # python-docx
    except ImportError as e:
        raise ImportError("pip install python-docx to ingest .docx files") from e
    d = docx.Document(str(path))
    return "\n\n".join(p.text for p in d.paragraphs)


def _read(path: Path) -> str:
    if path.suffix.lower() == ".docx":
        return _read_docx(path)
    return path.read_text(encoding="utf-8", errors="replace")


def doc_id(source: str, text: str) -> str:
    return hashlib.sha1(f"{source}|{len(text)}".encode()).hexdigest()[:12]


def split_front_matter(raw: str) -> tuple[dict[str, str], str]:
    """Pull a leading `---` block off a document.

    Used to carry the *real* prompt a passage was written to answer. A generated question is a
    guess; when the true prompt exists, it is ground truth and should be preferred. Parsed here,
    before normalize_document, because normalization would eat the delimiters.
    """
    if not raw.lstrip().startswith("---"):
        return {}, raw
    body = raw.lstrip()
    end = body.find("\n---", 3)
    if end == -1:
        return {}, raw
    meta: dict[str, str] = {}
    for line in body[3:end].strip().split("\n"):
        key, sep, val = line.partition(":")
        if sep and key.strip():
            meta[key.strip()] = val.strip()
    return meta, body[end + 4:].lstrip("\n")


def ingest_local(
    in_dir: str | Path,
    *,
    min_prose_ratio: float = 0.35,
    keep_templates: bool = False,
    register: str | None = None,
) -> list[dict[str, Any]]:
    """Walk a directory and return normalized document records.

    `register` tags the whole batch as e.g. "formal" or "informal". Tag your corpus -- the plan
    expects informal text to be the hard case, and you cannot see that in the eval unless the
    tag exists.
    """
    in_dir = Path(in_dir)
    if not in_dir.exists():
        raise FileNotFoundError(f"{in_dir} does not exist")

    docs: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for path in sorted(in_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED:
            continue
        raw = _read(path)
        meta, raw_body = split_front_matter(raw)
        text = normalize_document(raw_body)
        if not text:
            skipped.append({"source": str(path), "reason": "empty after normalize"})
            continue
        pr = prose_ratio(text)
        if pr < min_prose_ratio:
            skipped.append({"source": str(path), "reason": f"prose_ratio={pr:.2f}"})
            continue
        if not keep_templates and looks_like_template(text):
            skipped.append({"source": str(path), "reason": "template/too short"})
            continue
        rec = {
            "doc_id": doc_id(str(path), text),
            "source": str(path.relative_to(in_dir)),
            "origin": "local",
            "register": register or _guess_register(path),
            "chars_raw": len(raw),
            "text": text,
        }
        if meta.get("prompt"):
            rec["prompt"] = meta["prompt"]
        docs.append(rec)

    if skipped:
        print(f"[ingest] skipped {len(skipped)} file(s):")
        for s in skipped[:10]:
            print(f"  - {s['source']}: {s['reason']}")
        if len(skipped) > 10:
            print(f"  ... and {len(skipped) - 10} more")
    return docs


def _guess_register(path: Path) -> str:
    """Directory names are the cheapest register label you'll ever get. Use them."""
    parts = {p.lower() for p in path.parts}
    if parts & {"informal", "notes", "journal", "chat", "texts", "dms", "slack"}:
        return "informal"
    if parts & {"formal", "essays", "reports", "papers", "work"}:
        return "formal"
    return "unknown"
