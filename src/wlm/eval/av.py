"""Authorship verification — the primary metric (plan §6, method after arXiv 2509.14543).

Question the AV asks: given this text, would a style-sensitive model attribute it to the target
author rather than to anyone else? Two components:

  1. A **style embedder**. Default is `AnnaWegmann/Style-Embedding`, trained contrastively to
     represent writing style while being deliberately *invariant to topic* ("Same Author or Just
     Same Topic?"). That invariance is the whole reason to prefer it over a generic sentence
     encoder here: a generic encoder would let topic overlap masquerade as style match.
  2. A **verifier head** fit on author-vs-distractor embeddings (logistic regression by default,
     or a centroid/cosine scorer when data is very small).

You need a distractor corpus: text by other people, ideally in the same registers. Without it
the classifier has nothing to discriminate against and its scores mean nothing. Put plain-text
files in data/raw/distractor/.

Guardrails built in:
  - Fit only on TRAIN documents. Fitting on blind text inflates every subsequent number.
  - Report a held-out AUC for the verifier itself. If the verifier can't separate real author
    text from distractor text, it cannot possibly evaluate your generations -- and a high
    "attribution rate" from a broken verifier is the most likely way this project fools you.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_EMBEDDER = "AnnaWegmann/Style-Embedding"


class StyleEmbedder:
    """Sentence-transformers wrapper with chunk-mean pooling for long texts."""

    def __init__(self, model_name: str = DEFAULT_EMBEDDER, device: str | None = None) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)

    def encode(self, texts: list[str], *, batch_size: int = 16) -> np.ndarray:
        vecs = self.model.encode(
            texts, batch_size=batch_size, convert_to_numpy=True, normalize_embeddings=True,
            show_progress_bar=len(texts) > 64,
        )
        return np.asarray(vecs, dtype=np.float32)


@dataclass
class AvMetrics:
    auc: float
    accuracy: float
    threshold: float
    n_author: int
    n_distractor: int
    embedder: str
    head: str


class AuthorshipVerifier:
    def __init__(self, embedder: StyleEmbedder, head: str = "logreg") -> None:
        self.embedder = embedder
        self.head_kind = head
        self.head: Any = None
        self.centroid: np.ndarray | None = None
        self.threshold: float = 0.5
        self.metrics: AvMetrics | None = None

    # ------------------------------------------------------------------ fitting
    def fit(
        self,
        author_texts: list[str],
        distractor_texts: list[str],
        *,
        seed: int = 17,
        test_frac: float = 0.25,
        author_groups: list[str] | None = None,
    ) -> AvMetrics:
        """Fit the verifier head and report a genuinely held-out AUC.

        Two guards that decide whether this AUC means anything:

        - **Dedupe.** `build_pairs` emits several questions per chunk, so the same passage appears
          `questions_per_chunk` times in train.jsonl. A row-level split then puts identical text on
          both sides and the AUC is memorization.
        - **Group by document.** Sibling chunks from one document share topic and phrasing. Pass
          `author_groups` (doc ids) so they can't straddle the split -- the same invariant the
          train/val/blind split enforces.
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import balanced_accuracy_score, roc_auc_score
        from sklearn.model_selection import GroupShuffleSplit, train_test_split

        author_texts, author_groups = _dedupe(author_texts, author_groups)
        distractor_texts, _ = _dedupe(distractor_texts, None)

        if len(author_texts) < 6 or len(distractor_texts) < 6:
            raise ValueError(
                f"need >=6 distinct texts per side to fit a verifier (got {len(author_texts)} "
                f"author, {len(distractor_texts)} distractor). Add more distractor text; a "
                "verifier fit on a handful of samples reports noise."
            )

        X = self.embedder.encode(author_texts + distractor_texts)
        y = np.array([1] * len(author_texts) + [0] * len(distractor_texts))

        if author_groups:
            # Distractors get one synthetic group each; only the author side can leak by document.
            groups = np.array(
                author_groups + [f"_d{i}" for i in range(len(distractor_texts))], dtype=object
            )
            splitter = GroupShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
            tr_idx, te_idx = next(splitter.split(X, y, groups=groups))
            Xtr, Xte, ytr, yte = X[tr_idx], X[te_idx], y[tr_idx], y[te_idx]
            if len(set(yte)) < 2:
                print("[av] group split left one class empty in the test half; falling back")
                Xtr, Xte, ytr, yte = train_test_split(
                    X, y, test_size=test_frac, random_state=seed, stratify=y
                )
        else:
            print(
                "[av] no document groups supplied — sibling chunks from one document may straddle "
                "the verifier's own train/test split, which inflates the AUC below."
            )
            Xtr, Xte, ytr, yte = train_test_split(
                X, y, test_size=test_frac, random_state=seed, stratify=y
            )

        if self.head_kind == "logreg":
            clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
            clf.fit(Xtr, ytr)
            self.head = clf
            scores_te = clf.predict_proba(Xte)[:, 1]
        elif self.head_kind == "centroid":
            self.centroid = Xtr[ytr == 1].mean(axis=0)
            self.centroid /= np.linalg.norm(self.centroid) + 1e-9
            scores_te = Xte @ self.centroid
        else:
            raise ValueError(f"unknown head {self.head_kind!r}")

        auc = float(roc_auc_score(yte, scores_te))
        # Balanced accuracy, not plain accuracy: on an imbalanced held-out split plain accuracy
        # drags the threshold toward the majority class, and the attribution rate every downstream
        # number is read off would move with class sizes rather than with style.
        best_t, best_acc = 0.5, 0.0
        for t in np.unique(scores_te):
            acc = balanced_accuracy_score(yte, (scores_te >= t).astype(int))
            if acc > best_acc:
                best_t, best_acc = float(t), float(acc)
        self.threshold = best_t

        self.metrics = AvMetrics(
            auc=round(auc, 4),
            accuracy=round(best_acc, 4),
            threshold=round(best_t, 4),
            n_author=len(author_texts),
            n_distractor=len(distractor_texts),
            embedder=self.embedder.model_name,
            head=self.head_kind,
        )
        if auc < 0.75:
            print(
                f"[av] WARNING held-out AUC={auc:.3f}. The verifier can barely tell the author "
                "from the distractors, so downstream attribution numbers are not trustworthy. "
                "Fix this before reading any Stage-A result: add more author text, add "
                "register-matched distractors, or try a different embedder."
            )
        return self.metrics

    # ------------------------------------------------------------------ scoring
    def score(self, texts: list[str]) -> np.ndarray:
        if self.head is None and self.centroid is None:
            raise RuntimeError("verifier is not fit; call fit() or load()")
        X = self.embedder.encode(texts)
        if self.head is not None:
            return self.head.predict_proba(X)[:, 1]
        return X @ self.centroid

    def attribution_rate(self, texts: list[str]) -> float:
        """Fraction of generations the verifier attributes to the true author. Primary number."""
        s = self.score(texts)
        return float((s >= self.threshold).mean())

    def evaluate(self, texts: list[str]) -> dict[str, Any]:
        s = self.score(texts)
        return {
            "attribution_rate": round(float((s >= self.threshold).mean()), 4),
            "score_mean": round(float(s.mean()), 4),
            "score_median": round(float(np.median(s)), 4),
            "score_sd": round(float(s.std()), 4),
            "n": len(texts),
            "threshold": round(self.threshold, 4),
        }

    # ------------------------------------------------------------------ io
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        with (path / "head.pkl").open("wb") as f:
            pickle.dump({"head": self.head, "centroid": self.centroid}, f)
        (path / "av_meta.json").write_text(
            json.dumps(
                {
                    "embedder": self.embedder.model_name,
                    "head_kind": self.head_kind,
                    "threshold": self.threshold,
                    "metrics": asdict(self.metrics) if self.metrics else None,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path, *, device: str | None = None) -> AuthorshipVerifier:
        path = Path(path)
        meta = json.loads((path / "av_meta.json").read_text(encoding="utf-8"))
        obj = cls(StyleEmbedder(meta["embedder"], device=device), head=meta["head_kind"])
        with (path / "head.pkl").open("rb") as f:
            blob = pickle.load(f)  # noqa: S301 -- our own artifact
        obj.head, obj.centroid = blob["head"], blob["centroid"]
        obj.threshold = meta["threshold"]
        if meta.get("metrics"):
            obj.metrics = AvMetrics(**meta["metrics"])
        return obj


def _dedupe(
    texts: list[str], groups: list[str] | None
) -> tuple[list[str], list[str] | None]:
    seen: set[str] = set()
    out_t: list[str] = []
    out_g: list[str] = []
    for i, t in enumerate(texts):
        key = t.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out_t.append(t)
        if groups:
            out_g.append(groups[i])
    return out_t, (out_g if groups else None)


def load_distractor_texts(
    dir_path: str | Path, *, min_words: int = 60, max_words: int = 320, limit: int | None = None
) -> list[str]:
    """Read data/raw/distractor/*.txt|md and window it to match author-chunk length.

    Length matching matters: the style embedder is length-sensitive, and if your author chunks
    are 200 words while distractors are 2000, the classifier learns length, not style.
    """
    from wlm.chunk import chunk_document

    dir_path = Path(dir_path)
    out: list[str] = []
    for p in sorted(dir_path.rglob("*")):
        if p.suffix.lower() not in {".txt", ".md", ".markdown"} or not p.is_file():
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        out.extend(chunk_document(text, min_words=min_words, max_words=max_words))
        if limit and len(out) >= limit:
            return out[:limit]
    if not out:
        raise FileNotFoundError(
            f"no distractor text found in {dir_path}. The AV needs text by other people to "
            "discriminate against. Public-domain prose (Gutenberg), your own scraped-and-cleared "
            "blog corpus, or any register-matched third-party writing works."
        )
    return out
