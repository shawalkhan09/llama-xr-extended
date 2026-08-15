# LLaMA-XR Extended — Chest X-ray Report Generation

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

## The four phases

| Phase | What | Needs GPU? | Status |
|---|---|---|---|
| 0 | Data prep — download IU X-ray, run frozen DenseNet-121, build Alpaca prompts | No | Code ready |
| 1 | Baseline reproduction — QLoRA fine-tune LLaMA 3.1 8B (matches paper's exact setup) | Yes (Colab T4/A100) | Code ready, run on Colab |
| 2 | **New: clinical-accuracy eval** — the metric the paper's Limitations section says is missing | No | Code ready, testable now |
| 3 | **New: hallucination flagging** — cross-check generated claims against classifier confidence | No | Code ready, testable now |
| 4 | **New: dense visual embeddings** — replace the 36-number bottleneck with real image features | Yes | Scaffold ready, wire in after Phase 1 works |

Phases 2 and 3 don't need the trained model or a GPU to build and test — they're pure text/logic,
so they're already runnable with the sample data included here. That's deliberate: you can
verify those work correctly before ever touching Colab.

## Why this ordering

1. **Phase 0–1 first**: get the exact paper setup running so you have a real baseline to compare
   against — this is also what "I reproduced a 2026 paper" means concretely in a CV bullet.
2. **Phase 2 next**: this is the paper's own acknowledged gap (they measure ROUGE-L/METEOR but
   never check whether the *right diagnosis* was actually generated). Cheapest, highest-signal
   contribution — no retraining needed, just scoring text.
3. **Phase 3**: layers a safety/reliability check on top — doesn't touch the model, just adds a
   verification pass.
4. **Phase 4 last, and optional**: the deeper architectural change (real image embeddings instead
   of 18 classifier scores). Higher effort, higher payoff — do this once 1–3 are solid and you
   want to push further.

## Setup (Colab)

1. Push this project (code only — not the dataset) to your own GitHub repo.
2. Upload the R2Gen IU X-ray zip (https://github.com/zhjohnchan/R2Gen#datasets) to your
   Google Drive — the single `.zip`, not the unzipped folder.
3. Open `notebooks/phase1_baseline_colab.ipynb` in Colab, set Runtime → GPU, and run the
   cells top to bottom. It clones your repo, mounts Drive, and unzips the dataset into
   `data/iu_xray/` automatically — no manual path wrangling needed.

## Local setup (for Phase 2/3, which don't need a GPU)

```bash
pip install -r requirements.txt
```

## Directory layout

```
llama-xr-extended/
├── src/
│   ├── data_prep.py          # Phase 0: image → DenseNet-121 scores → Alpaca prompt
│   ├── train_baseline.py     # Phase 1: Unsloth + QLoRA fine-tuning (Colab)
│   ├── eval_lexical.py       # Phase 1: BLEU / ROUGE-L / METEOR (paper's own metrics)
│   ├── eval_clinical.py      # Phase 2: NEW — per-condition clinical F1
│   ├── hallucination_check.py# Phase 3: NEW — flags unsupported claims
│   └── visual_embeddings.py  # Phase 4: NEW — dense embeddings instead of 18-d scores
├── notebooks/
│   └── phase1_baseline_colab.ipynb   # one-click Colab notebook for Phase 0+1
├── data/
│   └── sample_reports.json   # small hand-made sample so Phase 2/3 run without the real dataset
└── requirements.txt
```

## Next step for you

Run `python src/eval_clinical.py` right now — it works on the bundled sample data with no
setup. That's the fastest way to see the new contribution in action before touching Colab at all.
