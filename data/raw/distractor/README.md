# Distractor corpus

Put plain-text (`.txt` / `.md`) writing by **other people** here. The authorship verifier needs
a negative class; without it, `wlm eval fit-av` has nothing to discriminate against and every
attribution number downstream is meaningless.

What works:

- Public-domain prose from Project Gutenberg (aim for prose, not verse or drama)
- Blog posts or essays by other people, in the same registers as your own corpus
- Any third-party writing you have the rights to use locally

What matters more than volume: **register match**. If your corpus is casual notes and your
distractors are 19th-century novels, the verifier learns "modern casual vs Victorian formal" and
reports a near-perfect AUC that tells you nothing about whether the model captured *your* voice
rather than *a* modern casual voice. Mix in contemporary informal prose from other people.

Target: at least a few hundred chunks, ideally 2-5x your author-chunk count (the CLI balances
classes at fit time).
