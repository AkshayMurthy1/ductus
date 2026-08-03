# Three proposals — style acquisition vs. content memorization

Ranked by novelty × likelihood of working. All three are the *same intervention at three
different levels*: restrict the **hypothesis class** (§1), restrict the **data** (§2), or
penalize the **objective** (§3). That framing is worth stating explicitly in a pitch — it turns
three ideas into one research program with a control at each level.

Shared premise: this repo already measures both axes (authorship attribution on a held-out blind
split; verbatim + entity + semantic-echo leakage; fluency veto; contamination probe). The
contribution is not any single architecture — it is that these become **measurable privacy dials
with an exchange rate**.

---

## 1. Spectrally band-limited adapters

**Proposal.** Take the FFT of hidden states along the *sequence* axis, apply the low-rank update
only inside a chosen frequency band, then inverse-FFT back. The cutoff frequency is a continuous
dial. Hypothesis: content is low-frequency (topic and entities persist over hundreds of tokens),
style is high-frequency (function-word alternation, clause rhythm, punctuation cadence at 5–50
token scales). Keep the high band, discard the low band, and content loses its channel.

**Literature / intuition.** LoRA (Hu et al. 2021) applied to `nn.Linear` is **position-wise**:
`h_i → h_i + BA h_i` independently per position. It structurally cannot represent any
cross-position statistic — yet burstiness, sentence-length variance, and clause rhythm, which
classical stylometry treats as the core of authorial identity, are *exactly* cross-position
statistics. The adapter can only reach them indirectly through attention. In the Fourier basis
over positions, convolution becomes per-mode multiplication, so a band-limited spectral operator
is a cheap cross-position mixer. FNet (Lee-Thorp et al. 2021) and Hyena (Poli et al. 2023)
establish that frequency-domain sequence mixing works at scale; FNO (Li et al. 2020) shows that
**mode truncation is itself the regularizer**. Spectral bias (Rahaman et al. 2019) — networks fit
low frequencies first — predicts that if content really is low-frequency, memorization should
arrive *early* in training, which is independently checkable against the checkpoint-trajectory run.

**What it actually optimizes.** Nothing changes in the loss. It constrains the *hypothesis space*
of `ΔW` to a frequency band, so style-per-unit-leakage improves by construction of the function
class rather than by an added penalty term.

**Key concept.** Band-limiting as capacity control — spectral truncation as an inductive prior.

**Falsifier.** Sweep the cutoff. If attribution and leakage fall together at every band, content
and style are spectrally entangled and the dial does not exist. Publishable either way.

---

## 2. Skeleton / lexicon two-channel training (the α-dial)

**Proposal.** Train on two channels: `(question → content-masked skeleton)` at weight α, and
`(question → full text)` at weight 1−α. The skeleton keeps function words, punctuation, and
POS structure; every content word is replaced by a typed placeholder. α is the dial.

**Literature / intuition.** Authorship attribution has rested on *function words* since Mosteller
& Wallace's Federalist study (1963) and Burrows's Delta (2002) — the discriminative signal lives
in the closed-class vocabulary and syntax, not in topic words. If that holds, the skeleton retains
a near-sufficient statistic for style while removing the only carrier of verbatim and entity
leakage. The α=1 endpoint has **provably zero content leakage**: there is no content in the
targets to leak. This is the cheapest of the three and the most likely to work, because it depends
on a claim with sixty years of evidence behind it.

**What it actually optimizes.** It interpolates the *training distribution* between pure form and
form-plus-lexicon. The dial measures how much of author identity is unrecoverable from syntax
alone — a number nobody has, and one that bounds how much lexical exposure a style adapter needs.

**Key concept.** Nuisance removal at the data level via sufficient statistics, rather than at the
model level. More auditable than architectural privacy: you can *read* the training targets and
confirm the content is gone.

**Build cost.** No architecture change. Reuses `scrub.py` and `dataset.py`. Two days.

---

## 3. Adversarial content-invariance

**Proposal.** Attach a discriminator that predicts *which training document* a hidden state or
generation came from, and train the adapter against it via gradient reversal. Push toward
representations that are maximally author-identifiable and minimally document-identifiable. λ is
the dial.

**Literature / intuition.** Gradient reversal is standard in domain-adversarial training (Ganin &
Lempitsky 2015) and fair representation learning (Louizos et al. 2016), where the adversary
variationally approximates a mutual-information term. The objective here is

    max_θ  I(output ; author)  −  λ · I(output ; document_id)

which is the information bottleneck (Tishby et al. 1999) with document identity as the nuisance.
Memorization in LMs is well characterized empirically (Carlini et al. 2021, 2022) but is almost
always *measured* post hoc; making it a differentiable training signal is the move.

**What it actually optimizes.** It turns the leakage axis from an evaluation metric into part of
the objective — the frontier stops being something you observe and becomes something you descend.

**Key concept.** Minimax nuisance-invariance / variational MI minimization. Established machinery;
the novelty is the choice of nuisance variable (the specific training passage) and the fact that
there is a real leakage instrument to validate the adversary against.

**Risk.** Adversarial training is unstable at small scale, and the discriminator may simply learn
topic. Mitigate by giving it only the scrubbed text, so it must key on something other than
entities.

---

## Ranking rationale

| | Novelty | Likely to work | Build cost | Best for |
|---|---|---|---|---|
| §1 Spectral | **highest** | medium-high | medium | strongest research pitch; FNO lineage |
| §2 Skeleton | medium | **highest** | low | build first — produces the frontier plot that motivates the rest |
| §3 Adversarial | medium-high | medium | medium | cleanest objective; appeals to theory-leaning readers |

**Build §2 first.** It needs no new architecture, and if skeleton-only training yields non-trivial
attribution, that single α-sweep plot is direct evidence for the style/content separability premise
that §1 and §3 both assume. It de-risks the other two.

## Honest positioning

Name the prior art yourself in any outreach — FNet, Hyena, FNO, Text-to-LoRA, representation
engineering, Shumailov et al. (2024) on recursive collapse. A reader will recognize all of it
within seconds; preempting it signals literacy rather than naivety. Lead with the negative result
you are willing to accept: *if style and content are spectrally entangled, the dial does not
exist, and that is the finding.*
