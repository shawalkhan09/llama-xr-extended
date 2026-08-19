# LLaMA-XR Extended — Chest X-ray Report Generation

**See [FINDINGS.md](FINDINGS.md) for the full progress log, technical issues encountered and
fixed, and the key reproducibility finding on the paper's stated learning rate.**

Reproduction and extension of:

> Jahangir et al., "LLaMA-XR: A Novel Framework for Radiology Report Generation using LLaMA
> and QLoRA Fine-Tuning," 2026. https://arxiv.org/abs/2506.03178

**Important note on scope:** the original authors' GitHub repo (`zihadbinjahangir/LLaMA-XR`)
currently contains only a README and citation — no training/eval code has been released.
Everything in `src/` here is a fresh implementation built directly from the paper's Methods
section (architecture, prompt format, hyperparameters), not a fork of existing code. That
actually makes this more valuable as a portfolio project: it's the first working
implementation of the described method, plus three extensions the paper's own Limitations
section calls for but doesn't do.

The original paper's repo is licensed CC BY-NC-SA 4.0 (non-commercial, research use, share-alike).
Treat this project under the same terms — fine for a portfolio/research write-up, not for a
commercial product.

## Results summary

Both training runs completed. The paper's exact stated hyperparameters produced a model that
failed to learn the task; a corrected run (higher learning rate) fixed that but surfaced a
second, deeper issue — the model doesn't ground its output in the actual input scores. Both
are real, evidenced findings, documented in full in **[FINDINGS.md](FINDINGS.md)**, along with
a scaffolded (data extraction built and verified, training integration not yet built) path to
the architectural fix.

## The phases

| Phase | What | Needs GPU? | Status |
|---|---|---|---|
| 0 | Data prep — download IU X-ray, run frozen DenseNet-121, build Alpaca prompts | No | Done — runs on Colab or Kaggle |
| 1 | Baseline reproduction — QLoRA fine-tune LLaMA 3.1 8B matching the paper's exact setup | Yes | Done — paper's stated LR failed to learn the task (see FINDINGS.md) |
| 1b | Corrected training run (`train_baseline_v2.py`) — learning rate fix, restored eval | Yes | Done — fixed the degenerate output, surfaced the grounding issue |
| 2 | **New: clinical-accuracy eval** — the metric the paper's Limitations section says is missing | No | Done, tested, negation-detection bug found and fixed |
| 3 | **New: hallucination flagging** — cross-check generated claims against classifier confidence | No | Done, tested |
| 4 | **New: dense visual embeddings** — replace the 36-number bottleneck with real image features | Yes | Data extraction done and verified (`data_prep_dense.py`); training-loop integration not built |

Phases 2 and 3 don't need the trained model or a GPU to build and test — they're pure text/logic,
runnable immediately with the sample data included here.

## Why this ordering

1. **Phase 0–1 first**: get the exact paper setup running so you have a real baseline to compare
   against — this is also what "I reproduced a 2026 paper" means concretely in a CV bullet.
2. **Phase 2 next**: this is the paper's own acknowledged gap (they measure ROUGE-L/METEOR but
   never check whether the *right diagnosis* was actually generated). Cheapest, highest-signal
   contribution — no retraining needed, just scoring text.
3. **Phase 3**: layers a safety/reliability check on top — doesn't touch the model, just adds a
   verification pass.
4. **Phase 4 last, and optional**: the deeper architectural change (real image embeddings instead
   of 18 classifier scores). Higher effort, higher payoff — the data-extraction half is done;
   see FINDINGS.md for why this became a well-evidenced priority rather than a guess.

## Setup

1. Push this project (code only — not the dataset) to your own GitHub repo.
2. Upload the R2Gen IU X-ray zip (https://github.com/zhjohnchan/R2Gen#datasets) to Google
   Drive (for Colab) or as a Kaggle Dataset (for Kaggle) — either platform works; see
   FINDINGS.md's Infrastructure Notes for the tradeoffs we hit with each.
3. On Colab: open `notebooks/phase1_baseline_colab.ipynb`, set Runtime → GPU, run top to
   bottom — it clones your repo, mounts Drive, and unzips the dataset automatically.
   On Kaggle: create a notebook, attach your dataset via "Add Input", enable a GPU
   accelerator, and adapt the same cell sequence (clone repo, install deps, run data prep,
   run training) — Kaggle's paths differ slightly (`/kaggle/input/...`), see FINDINGS.md.

## Local setup (for Phase 2/3, which don't need a GPU)

```bash
pip install -r requirements.txt
```

## Directory layout

```
llama-xr-extended/
├── src/
│   ├── data_prep.py           # Phase 0: image → DenseNet-121 scores → Alpaca prompt
│   ├── data_prep_dense.py     # Phase 4: image → dense 1024-dim features (data extraction only)
│   ├── train_baseline.py      # Phase 1: paper's exact hyperparameters (documented failure)
│   ├── train_baseline_v2.py   # Phase 1b: corrected learning rate + restored eval
│   ├── generate_reports.py    # Inference: runs the trained adapter on the test set
│   ├── eval_lexical.py        # BLEU / ROUGE-L / METEOR (paper's own metrics)
│   ├── eval_clinical.py       # Phase 2: NEW — per-condition clinical F1
│   ├── hallucination_check.py # Phase 3: NEW — flags unsupported claims
│   └── visual_embeddings.py   # Phase 4: NEW — projector architecture (not yet wired into training)
├── notebooks/
│   └── phase1_baseline_colab.ipynb   # Colab notebook for Phase 0+1 (also adaptable for Kaggle)
├── data/
│   └── sample_reports.json    # small hand-made sample so Phase 2/3 run without the real dataset
└── requirements.txt
```

## Next step for you

Run `python src/eval_clinical.py` right now — it works on the bundled sample data with no
setup. That's the fastest way to see the new contribution in action before touching Colab or
Kaggle at all.
