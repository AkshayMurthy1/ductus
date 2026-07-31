# Distractor corpus — writing by OTHER people

Put plain-text (`.txt` / `.md`) writing by **other people** here. The authorship verifier needs
a negative class; without it, `wlm eval fit-av` has nothing to discriminate against and every
attribution number downstream is meaningless.

What works:

- Public-domain prose from Project Gutenberg (aim for prose, not verse or drama)
- Blog posts or essays by other people, in the same registers as your own corpus
- Any third-party writing you have the rights to use locally

Target: at least a few hundred chunks, ideally 2-5x your author-chunk count (the CLI balances
classes at fit time, so a set far larger than your author corpus is partly unused — check
`n_distractor` in the metrics).

## Register match matters more than volume

This is the easy thing to get wrong, and it fails in the direction that looks like success: a
mismatched negative class produces a **high** AUC that means nothing.

If your corpus is casual notes and your distractors are 19th-century novels, the verifier learns
"modern casual vs Victorian formal," reports something near 0.99, and will then rubber-stamp any
generic modern-casual text as yours. `scripts/00_phase0_cpu.sh` treats an AUC above ~0.97 as a
signal to inspect your distractors, not to celebrate.

Two things that do **not** fix a mismatch, both established on this project by experiment:

- **Proficiency matching.** Swapping in higher-scoring, same-grade student essays moved a
  saturated 1.0 by nothing at all.
- **More data.** Adding easy negatives makes the task easier, not harder.

What fixes it is **genre and register match** — other people doing the same *kind* of writing.
Not the same topic: entity scrubbing and the style embedder already suppress subject matter, so
hunting for topic-matched negatives is wasted effort.

## What is in here right now is a development fixture

420 windows from **21 different essayists** — Addison, Arnold, Belloc, Benson, Birrell, Burroughs,
Chapman, Emerson, Gissing, Hazlitt, Lamb, Le Gallienne, Mencken, Meynell, More, Pater,
Quiller-Couch, Repplier, Stevenson, Thoreau, Twain — all public domain, all built by
`scripts/build_dev_corpus.py`.

It is period- and genre-matched to the **Chesterton** development fixture in `data/raw/author/`
and is useful for nothing else. Chesterton is deliberately absent; the build script asserts it,
because an author appearing in their own negative class destroys the metric.

**Replace this when you replace the author corpus.** Negatives matched to a 1910s essayist tell
you nothing about a modern writer. Pick prose in the same genre and register as whatever ends up
in `data/raw/author/`, from as many distinct authors as you can find — author diversity matters
more than word count, and no single author should exceed ~15% of the files.
