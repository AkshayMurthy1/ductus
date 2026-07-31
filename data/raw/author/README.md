# Author corpus — YOUR writing goes here

This is the one directory whose contents are the point of the project. Everything downstream
treats what lives here as ground truth: Stage-A targets, Stage-B `chosen`, the blind split every
number is measured against.

Drop `.txt` / `.md` / `.docx` files in, or use `wlm ingest gdocs` to pull from Google Docs.
Subdirectory names set the register — `formal/`, `essays/`, `papers/` tag as formal;
`informal/`, `notes/`, `journal/`, `chat/` tag as informal. Tag both: `docs/PLAN.md` treats
informal as the hard case, and you cannot measure it if it isn't there.

A document may carry the real prompt it was written to answer, as front matter:

```markdown
---
prompt: Analyse the role of the unreliable narrator in one novel we studied.
---

The trouble with trusting a narrator is that…
```

`data.use_supplied_prompts` (default true) then trains against that instead of a generated
question. A real prompt beats an invented one, and some prompts — "List five things that are
important to you" — cannot be recovered from the answer at all.

Nothing here is ever committed: `.gitignore` excludes `data/**` apart from `.gitkeep` and these
README files.

---

## ⚠️ What is in here right now is NOT a real corpus

The current contents are a **development fixture**: public-domain essays by **G.K. Chesterton**,
pulled from Project Gutenberg by `scripts/build_dev_corpus.py`. 148 documents, ~256k words.

It exists to exercise and validate the pipeline at realistic scale — a corpus large enough that
the authorship verifier is stable (AUC ≈ 0.92, and the balanced and unbalanced fits agree to
within 0.003, which a small corpus will not do). It is **not** anyone's private writing, it
proves nothing about any real user, and no adapter trained on it is a product.

**Before any real run:** delete these files and replace them with the writing you actually want
modelled. A mixed corpus — one real person plus Chesterton — would be worse than either alone,
because the adapter would learn the average of two voices and the verifier would have no
coherent positive class.

```bash
rm -rf data/raw/author/formal          # drop the fixture
python scripts/build_dev_corpus.py     # or rebuild it, if you want it back
```
