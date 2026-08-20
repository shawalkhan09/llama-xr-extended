# Findings & Progress Log

A running record of what was built, what broke, what was learned, and why — for the
project itself and for anyone (including future-you) reading this repo later.

## Project goal

Reproduce LLaMA-XR (Jahangir et al., 2026, [arXiv:2506.03178](https://arxiv.org/abs/2506.03178)) —
a framework that fine-tunes LLaMA 3.1 8B with QLoRA to generate chest X-ray radiology reports,
conditioned on DenseNet-121 classifier scores — then extend it in ways the paper's own
Limitations section calls for but doesn't do:

1. **Clinical-accuracy evaluation** (the paper only measures BLEU/ROUGE-L/METEOR, which
   don't verify whether the correct diagnosis was captured)
2. **Hallucination flagging** (the paper discusses this risk but never builds a check for it)
3. **Dense visual embeddings** (replacing the paper's 36-number classifier-score bottleneck
   with real image features) — planned, not yet built

The paper's own code repository contains no released implementation — only a citation and
README — so everything here is a fresh build from the paper's Methods section, not a fork.

## Status at a glance

| Component | Status |
|---|---|
| Data pipeline (Phase 0) | Working, reproducible on both Colab and Kaggle |
| Baseline reproduction, paper's exact hyperparameters | Complete — model failed to learn (see Key Finding) |
| Corrected training run (adjusted LR) | Complete — fixed the degenerate output, but surfaced a second issue (see Second Finding) |
| Clinical-accuracy eval (`eval_clinical.py`) | Built, negation-detection bug found and fixed, ready |
| Hallucination check (`hallucination_check.py`) | Built, tested on sample data, ready |
| Inference script (`generate_reports.py`) | Built, decoding fixed (repetition penalty), verified |
| Dense visual embeddings (Phase 4) | Data extraction built and verified (`data_prep_dense.py`); training integration not built |

## Key finding: the paper's stated learning rate does not reproduce

The paper (Section 3.5) states a learning rate of 2×10⁻⁶, batch size 8, gradient
accumulation 4, and 3 epochs. We matched these settings as closely as our hardware allowed
(same effective batch size of 32, same 3 epochs, same ~195 total optimizer steps).

**Result: the fine-tuned model failed to learn the task.** Every generated report, regardless
of the actual input classifier scores, simply recited the full list of 18 condition names
verbatim — the same list that appears in the fixed instruction text of every training prompt.
The model was not grounding its output in the input data at all; it was reproducing the most
salient vocabulary sitting in its own prompt.

**Why:** 2×10⁻⁶ is roughly 100x lower than typical QLoRA learning rates (commonly 1e-4 to
3e-4). Combined with only ~195 gradient updates total, this was very likely insufficient for
the LoRA adapter to move the base model's behavior away from that degenerate pattern.

**What we changed:** learning rate raised to 2×10⁻⁴, with the paper's periodic in-training
evaluation and `load_best_model_at_end=True` restored (these were present in the paper's
described method but had been removed from our script in an earlier optimization pass for
speed — restoring them both gives loss-curve visibility and matches the paper's actual
procedure more closely).

This is a documented, deliberate deviation, not a hidden one — see `train_baseline_v2.py`,
which is kept as a separate file from the original paper-faithful `train_baseline.py` so both
attempts remain visible in the repo's history.

The corrected run finished: final eval_loss 0.6966, train_loss 0.841, all 195 steps completed
cleanly. See the next finding for what the generated text actually looked like.

## Second finding: the model doesn't condition on its input (output collapse)

With the LR fix, the model stopped reciting the instruction's condition list verbatim — a
real improvement. But a second, distinct problem surfaced once we looked at actual generated
text closely: **the model produces near-identical output regardless of what's actually in the
input.**

Evidence, built up in stages to rule out simpler explanations first:

1. Across a 20-example preview, 15/20 generations were exact or near-exact duplicates of one
   template (opening: *"The heart size appears normal in contour. There has been interval
   development of left-sided pleuro-pulmonary scarring compatible with prior granulomatous
   disease or tuberculosis..."*).
2. Checking whether this was just coincidentally similar inputs: the underlying classifier
   scores for the duplicated cases were *not* identical, but were very close (differences in
   the 4th decimal place) — plausible for genuinely normal X-rays, since a pathology
   classifier would output uniformly low confidence across the board for both.
3. To rule this out properly, we specifically selected test cases with **genuine, different,
   described abnormalities** (lung opacity; emphysema + hernia; atelectasis; atelectasis +
   infiltration; emphysema + infiltration + opacity) — clinically distinct pictures, verified
   via `eval_clinical.py`'s labeler (after fixing a negation-detection bug in that labeler
   itself — see below). The model produced the same generic template for all five, ignoring
   the actual differences between cases.

This rules out both "coincidence" and "labeling error" as explanations. The model has learned
what a radiology report *sounds like*, but not to ground its content in the actual input
scores.

**Confirmed at full scale.** Running `generate_reports.py` and all three eval scripts on the
complete 590-example test set (not just the spot-checked cases above) makes this quantitative:

| Metric | This model | Paper's reported value |
|---|---|---|
| BLEU-4 | 0.0073 | — |
| ROUGE-L | 0.1182 | 0.433 |
| METEOR | 0.1904 | 0.336 |
| Clinical Macro-F1 | 0.018 | not measured by the paper |
| Clinical Micro-F1 | 0.068 | not measured by the paper |
| Hallucination flags | 0/590 | not measured by the paper |

Lexical scores land well below the paper's claimed numbers (ROUGE-L at roughly a quarter of
theirs). The clinical F1 numbers are the more telling result: a macro-F1 of 0.018 across 590
cases means the model essentially isn't reliably producing the correct diagnosis, consistent
with what the spot-checked examples showed. The hallucination check coming back clean (0/590
flagged) shouldn't be read as reassuring on its own -- it more likely reflects the model
defaulting to generic, low-commitment phrasing rather than making confident, checkable claims,
which is the same underlying grounding problem showing up from a different angle.

**Likely cause**: the 36 raw classifier-confidence floats are serialized as a JSON array
string in the prompt (e.g. `"[0.599, 0.612, ...]"`). Floating-point numbers get fragmented
into arbitrary subword tokens during tokenization, which is a well-known weak point for LLM
numeracy — especially with only ~195 fine-tuning steps to learn a mapping from that fragile
representation to report content.

**What this means for next steps**: this is now real, direct evidence (not just a
theoretical concern from reading the paper) that Phase 4 — replacing the 36-number bottleneck
with dense image embeddings — addresses a genuine, measured problem, not a hypothetical one.
The data-extraction half of that fix is built and verified (`data_prep_dense.py` — real
1024-dim DenseNet-121 features per image, cached to disk); the harder half — modifying the
training loop to actually inject those features as soft tokens into the model, bypassing the
standard tokenized-text path — was scoped but not built, given the time and compute already
invested in this reproduction. See Next Steps for the honest state of this.

### Bug found and fixed along the way: negation detection in `eval_clinical.py`

While diagnosing the above, `label_report()`'s negation check (a fixed 60-character lookback
window) was found to miss common radiology phrasing like "No focal consolidation,
pneumothorax, or pleural effusion" — by the time the window reaches "effusion" at the end of
a negated list, it's often more than 60 characters past the "No" that negates the whole list.
This produced false positives: negated findings were being labeled as present. Fixed by
scoping the negation check to the current sentence (bounded by the nearest preceding period)
instead of a fixed character count, which correctly handles negated lists of any length. This
dropped the count of test examples flagged as having a genuine positive finding from 182/590
to 131/590 — the difference is exactly the false positives this bug was producing.

## Engineering notes (useful if reproducing this yourself)

A number of library/environment issues surfaced during this reproduction, worth knowing about
if you're doing something similar in 2026:

- **`trl`'s `SFTTrainer` API changed significantly** across recent releases: `tokenizer=`
  became `processing_class=`, `max_seq_length` moved off `SFTTrainer` onto a new `SFTConfig`
  object as `max_length`, and `TrainingArguments`' `evaluation_strategy` was renamed to
  `eval_strategy`.
- **Unsloth must be imported before `trl`/`transformers`/`peft`**, not after — importing it
  later causes a `ValueError` about the EOS token not being found in the tokenizer's
  vocabulary. This is a known Unsloth issue (unslothai/unsloth#2797), not obvious from the
  error message itself.
- **`torchxrayvision`'s DenseNet-121 classifier defaults to CPU** unless the model and input
  tensors are explicitly moved to the GPU device — silently falling back to CPU rather than
  erroring, which turned an expected few-minute step into a 45-60 minute one until caught.
- **A100 vs. T4 memory**: the paper's batch size of 8 (A100) OOMs immediately on a free-tier
  T4 (14.5GB). Batch size 2 with gradient accumulation 16 preserves the same effective batch
  size of 32 while fitting in T4 memory.

## Infrastructure notes

Reproducing an 8B-parameter fine-tune on free-tier compute (Google Colab, then Kaggle after
hitting Colab's usage limits) surfaces a set of practical constraints that don't show up when
reading the paper alone:

- Free-tier GPU sessions can disconnect from usage limits or inactivity with no fixed,
  published threshold — training checkpoints need to be written somewhere that survives a
  session dying mid-run (Google Drive on Colab; Kaggle's committed Dataset output on Kaggle),
  not just local/ephemeral session storage.
- Kaggle's interactive editor sessions have a 1-hour inactivity limit; a proper unattended
  multi-hour job needs to run as a "Save and Run All (Commit)" batch job instead, which is
  immune to that limit and persists its output automatically.
- Standard benchmark splits (here, R2Gen's preprocessed IU X-ray train/val/test split) are
  worth using over re-deriving your own from the raw dataset archive — the raw XML reports
  from the original source don't reliably distinguish frontal vs. lateral X-ray views, and
  using the same split as prior published methods keeps results comparable.

## Project status: complete for now

This reproduction is being closed out at this point, deliberately — not because there's
nothing left to do (there's a clear next step, below), but because the project already
stands as complete, honest, evidenced work without it:

- A working, tested, end-to-end reproduction pipeline (data prep → training → inference →
  three evaluation methods), runnable on either Colab or Kaggle.
- Two real findings, not assumptions: the paper's stated learning rate doesn't reproduce
  (diagnosed and fixed), and the corrected model has a measured grounding problem — confirmed
  quantitatively across the full 590-example test set (ROUGE-L 0.118 vs. the paper's claimed
  0.433; clinical macro-F1 0.018), not just anecdotal spot-checks.
- Two genuine contributions beyond the paper: a clinical-accuracy evaluator and a
  hallucination-flagging check, filling gaps the paper's own Limitations section names but
  doesn't fill.
- A scoped, partially-built path to the architectural fix (Phase 4), with the safer,
  lower-risk half (real image feature extraction) already built and verified.

## If picking this back up later

1. Build the training-loop integration for Phase 4: wire `VisualProjector` (in
   `visual_embeddings.py`) and the cached features from `data_prep_dense.py` into
   `train_baseline_v2.py`, replacing the tokenized 36-number input with injected soft
   tokens. The open technical question is whether this pattern works cleanly with
   Unsloth's patched model internals, or whether it needs standard HF Transformers + PEFT
   instead (slower, but more predictable).
2. This findings log plus the working, tested Phase 2/3 eval scripts, backed now by full
   590-example numbers, are real, presentable material for a portfolio write-up as they
   stand today.
