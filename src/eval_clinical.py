"""
Phase 2 — Clinical-accuracy evaluation.

The LLaMA-XR paper (arXiv:2506.03178) evaluates only with BLEU-4 / ROUGE-L / METEOR,
which measure text overlap, not whether the correct diagnosis was produced. The authors
explicitly flag this as a gap in their Limitations section and call for future work to
check whether key clinical findings are correctly identified. This script fills that gap.

Approach (a lightweight, from-scratch stand-in for the CheXpert/CheXbert labeler used in
the wider radiology-report-generation literature):
  1. For each of the paper's 18 conditions, search the report text for that condition's
     synonyms.
  2. For each mention, look at a small window of preceding words for negation cues
     ("no", "without", "free of", "negative for", ...) to decide POSITIVE vs NEGATIVE.
  3. Conditions never mentioned default to NEGATIVE (absent) — the standard convention
     used by CheXpert-style labelers.
  4. Compare the reference report's label vector against the generated report's label
     vector, per condition, across the whole test set -> precision / recall / F1.

This is a simplified proxy, not a clinical-grade labeler. The natural upgrade path is to
swap in the real CheXbert labeler (a fine-tuned BERT model, ~400MB, from the Stanford ML
Group) once you want lab-grade rigor -- the interface below (`label_report`) is designed
so that swap only touches one function.
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

CONDITIONS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion",
    "Emphysema", "Enlarged Cardiomediastinum", "Fibrosis", "Fracture", "Hernia",
    "Infiltration", "Lung Lesion", "Lung Opacity", "Mass", "Nodule",
    "Pleural Thickening", "Pneumonia", "Pneumothorax",
]

# Synonym / surface-form dictionary. Extend this as you see more report phrasing.
SYNONYMS = {
    "Atelectasis": [
        "atelectasis", "atelectatic", "subsegmental atelectasis", "plate atelectasis",
        "linear atelectasis", "volume loss",
    ],
    "Cardiomegaly": [
        "cardiomegaly", "enlarged heart", "heart is enlarged", "cardiac enlargement",
        "enlarged cardiac silhouette", "heart size is enlarged", "borderline cardiomegaly",
        "cardiac silhouette is enlarged",
    ],
    "Consolidation": [
        "consolidation", "consolidative", "airspace consolidation", "lobar consolidation",
        "focal consolidation",
    ],
    "Edema": [
        "edema", "oedema", "pulmonary edema", "vascular congestion", "interstitial edema",
        "pulmonary vascular congestion", "fluid overload",
    ],
    "Effusion": [
        "effusion", "pleural effusion", "pleural fluid", "fluid in the pleural space",
        "layering effusion",
    ],
    "Emphysema": [
        "emphysema", "hyperexpanded lungs", "hyperinflation", "hyperinflated lungs",
        "emphysematous changes", "bullous changes", " bullae",
    ],
    "Enlarged Cardiomediastinum": [
        "enlarged cardiomediastinum", "mediastinal widening", "widened mediastinum",
        "mediastinal enlargement", "wide mediastinum",
    ],
    "Fibrosis": [
        "fibrosis", "fibrotic", "fibrotic changes", "pulmonary fibrosis", "fibrotic scarring",
    ],
    "Fracture": ["fracture", "fractures", "fractured", "rib fracture", "acute fracture"],
    "Hernia": ["hernia", "hiatal hernia", "diaphragmatic hernia"],
    "Infiltration": [
        "infiltrate", "infiltration", "infiltrates", "infiltrative", "pulmonary infiltrate",
    ],
    "Lung Lesion": ["lesion", "cavitary lesion", "pulmonary lesion", "lung lesion"],
    "Lung Opacity": [
        "opacity", "opacities", "airspace disease", "focal airspace", "airspace opacity",
        "opacification", "hazy opacity",
    ],
    "Mass": [" mass ", "masses", "pulmonary mass", "lung mass"],  # spaces around "mass" avoid
                                                                    # matching inside "massive"
    "Nodule": ["nodule", "nodular", "nodules", "pulmonary nodule", "lung nodule"],
    "Pleural Thickening": ["pleural thickening", "pleural scarring", "thickened pleura"],
    "Pneumonia": ["pneumonia", "pneumonic", "pneumonic infiltrate"],
    "Pneumothorax": ["pneumothorax", "collapsed lung", "air in the pleural space"],
}

NEGATION_CUES = [
    "no ", "no evidence of", "without", "free of", "negative for",
    "clear of", "absence of", "not seen", "is not", "are not", "rule out",
    "unremarkable", "within normal limits", "no focal", "no acute",
    "no significant", "no definite", "no evidence for", "resolution of", "resolved",
]

def label_report(text: str) -> dict:
    """Return {condition: 0 or 1} for a single report's text. 1 = positive finding.

    Negation scope is bounded by sentence (the last '.' before the mention), not a fixed
    character count. A fixed window (e.g. 60 chars) misses common radiology phrasing like
    "No focal consolidation, pneumothorax, or pleural effusion" -- by the time you reach
    "effusion" at the end of a negated list, you can easily be more than 60 characters past
    the "No" that negates the whole list. Scoping to the sentence handles lists of any length
    correctly, while still not reaching back into an unrelated earlier sentence.
    """
    text_lower = f" {text.lower()} "
    labels = {}
    for condition in CONDITIONS:
        found_positive = False
        for syn in SYNONYMS[condition]:
            for m in re.finditer(re.escape(syn.lower()), text_lower):
                sentence_start = text_lower.rfind(".", 0, m.start())
                sentence_start = sentence_start + 1 if sentence_start != -1 else 0
                window = text_lower[sentence_start:m.start()]
                negated = any(cue in window for cue in NEGATION_CUES)
                if not negated:
                    found_positive = True
                    break
            if found_positive:
                break
        labels[condition] = 1 if found_positive else 0
    return labels


def score_pair(ref_labels: dict, gen_labels: dict, tally: dict) -> None:
    """Accumulate TP/FP/FN/TN per condition into `tally`."""
    for condition in CONDITIONS:
        ref, gen = ref_labels[condition], gen_labels[condition]
        if ref == 1 and gen == 1:
            tally[condition]["tp"] += 1
        elif ref == 0 and gen == 1:
            tally[condition]["fp"] += 1
        elif ref == 1 and gen == 0:
            tally[condition]["fn"] += 1
        else:
            tally[condition]["tn"] += 1


def compute_metrics(tally: dict) -> dict:
    per_condition = {}
    macro_f1_values = []
    total_tp = total_fp = total_fn = 0

    for condition in CONDITIONS:
        c = tally[condition]
        tp, fp, fn = c["tp"], c["fp"], c["fn"]
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        per_condition[condition] = {
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
            "support": tp + fn,  # how many times this condition actually appeared
        }
        # only count conditions that appear at least once in the reference set for macro-F1,
        # otherwise conditions never seen would drag the average down meaninglessly
        if (tp + fn) > 0:
            macro_f1_values.append(f1)
        total_tp += tp
        total_fp += fp
        total_fn += fn

    macro_f1 = sum(macro_f1_values) / len(macro_f1_values) if macro_f1_values else 0.0
    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if (micro_precision + micro_recall) else 0.0
    )

    return {
        "per_condition": per_condition,
        "macro_f1": round(macro_f1, 3),
        "micro_f1": round(micro_f1, 3),
    }


def evaluate_dataset(records: list) -> dict:
    tally = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "tn": 0})
    per_sample = []

    for record in records:
        ref_labels = label_report(record["reference_report"])
        gen_labels = label_report(record["generated_report"])
        score_pair(ref_labels, gen_labels, tally)
        per_sample.append({
            "id": record["id"],
            "reference_positive": [c for c, v in ref_labels.items() if v == 1],
            "generated_positive": [c for c, v in gen_labels.items() if v == 1],
            "missed_findings": [c for c in CONDITIONS if ref_labels[c] == 1 and gen_labels[c] == 0],
            "extra_findings": [c for c in CONDITIONS if ref_labels[c] == 0 and gen_labels[c] == 1],
        })

    metrics = compute_metrics(tally)
    metrics["per_sample"] = per_sample
    return metrics


def main():
    data_path = Path(__file__).resolve().parent.parent / "data" / "sample_reports.json"
    if len(sys.argv) > 1:
        data_path = Path(sys.argv[1])

    with open(data_path) as f:
        records = json.load(f)

    results = evaluate_dataset(records)

    print(f"Clinical-accuracy evaluation on {len(records)} report pairs\n")
    print(f"{'Condition':<28} {'Precision':>10} {'Recall':>8} {'F1':>6} {'Support':>8}")
    print("-" * 64)
    for condition, m in results["per_condition"].items():
        if m["support"] > 0:
            print(f"{condition:<28} {m['precision']:>10} {m['recall']:>8} {m['f1']:>6} {m['support']:>8}")

    print("-" * 64)
    print(f"Macro-F1 (over conditions present in reference set): {results['macro_f1']}")
    print(f"Micro-F1 (pooled TP/FP/FN across all conditions):   {results['micro_f1']}\n")

    print("Per-sample errors:")
    for s in results["per_sample"]:
        if s["missed_findings"] or s["extra_findings"]:
            print(f"  {s['id']}:")
            if s["missed_findings"]:
                print(f"    MISSED (in reference, not generated): {s['missed_findings']}")
            if s["extra_findings"]:
                print(f"    EXTRA  (generated, not in reference): {s['extra_findings']}")


if __name__ == "__main__":
    main()
