"""
Phase 1 evaluation -- the paper's own metrics (BLEU-4, ROUGE-L, METEOR), so you can confirm
your reproduction lands in the same range they report: ROUGE-L 0.433, METEOR 0.336.

These measure text overlap only -- see eval_clinical.py for the clinical-accuracy check
these metrics can't provide.
"""

import json
import sys
from pathlib import Path

import nltk
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer

for pkg in ("wordnet", "punkt", "omw-1.4"):
    try:
        nltk.data.find(f"corpora/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)

rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
smoothing = SmoothingFunction().method1


def score_pair(reference: str, generated: str) -> dict:
    ref_tokens = reference.lower().split()
    gen_tokens = generated.lower().split()

    bleu4 = sentence_bleu([ref_tokens], gen_tokens, weights=(0.25, 0.25, 0.25, 0.25),
                           smoothing_function=smoothing)
    rouge_l = rouge.score(reference, generated)["rougeL"].fmeasure
    meteor = meteor_score([ref_tokens], gen_tokens)

    return {"bleu4": round(bleu4, 4), "rouge_l": round(rouge_l, 4), "meteor": round(meteor, 4)}


def evaluate_dataset(records: list) -> dict:
    scores = [score_pair(r["reference_report"], r["generated_report"]) for r in records]
    n = len(scores)
    return {
        "bleu4": round(sum(s["bleu4"] for s in scores) / n, 4),
        "rouge_l": round(sum(s["rouge_l"] for s in scores) / n, 4),
        "meteor": round(sum(s["meteor"] for s in scores) / n, 4),
        "per_sample": scores,
    }


def main():
    data_path = Path(__file__).resolve().parent.parent / "data" / "sample_reports.json"
    if len(sys.argv) > 1:
        data_path = Path(sys.argv[1])

    with open(data_path) as f:
        records = json.load(f)

    results = evaluate_dataset(records)
    print(f"Lexical evaluation on {len(records)} report pairs\n")
    print(f"BLEU-4:  {results['bleu4']}")
    print(f"ROUGE-L: {results['rouge_l']}   (paper reports 0.433)")
    print(f"METEOR:  {results['meteor']}   (paper reports 0.336)")


if __name__ == "__main__":
    main()
