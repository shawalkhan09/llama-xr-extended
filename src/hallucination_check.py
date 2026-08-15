"""
Phase 3 -- Hallucination flagging.

The LLaMA-XR paper discusses hallucination risk at length in its Discussion section but
never builds a check for it -- it's left as a qualitative observation. This script adds a
concrete, cheap guardrail: since the pipeline already computes an 18-condition confidence
score from the frozen DenseNet-121 classifier *before* the LLM ever writes a word, we can
cross-check the LLM's generated claims against those scores after the fact.

Logic: if the generated report positively asserts a condition (via eval_clinical's labeler)
but the classifier's own confidence for that condition is below a threshold, flag it as
"unsupported by the visual evidence the model was given" -- a likely hallucination rather
than a genuine finding the classifier under-weighted.

This is a guardrail layer, not a fix to the model itself -- exactly the "disclaimer wrapper"
idea from the system design: it sits after generation, doesn't touch model weights, and can
be applied to any report the pipeline produces.
"""

import json
import sys
from pathlib import Path

from eval_clinical import label_report, CONDITIONS

UNSUPPORTED_THRESHOLD = 0.30  # classifier confidence below this = "not supported"


def check_report(generated_report: str, classifier_scores: dict, threshold: float = UNSUPPORTED_THRESHOLD) -> dict:
    """
    Returns which positively-claimed findings in the generated report are and aren't
    backed by the classifier's own confidence scores.
    """
    claimed_labels = label_report(generated_report)
    claimed_positive = [c for c in CONDITIONS if claimed_labels[c] == 1]

    supported, unsupported = [], []
    for condition in claimed_positive:
        score = classifier_scores.get(condition, 0.0)
        if score >= threshold:
            supported.append({"condition": condition, "classifier_confidence": score})
        else:
            unsupported.append({"condition": condition, "classifier_confidence": score})

    return {
        "claimed_findings": claimed_positive,
        "supported": supported,
        "unsupported": unsupported,
        "flag_for_review": len(unsupported) > 0,
    }


def main():
    data_path = Path(__file__).resolve().parent.parent / "data" / "sample_reports.json"
    if len(sys.argv) > 1:
        data_path = Path(sys.argv[1])

    with open(data_path) as f:
        records = json.load(f)

    def fmt(items):
        return ", ".join(f"{it['condition']} ({it['classifier_confidence']:.2f})" for it in items)

    flagged_count = 0
    for record in records:
        result = check_report(record["generated_report"], record["classifier_scores"])
        status = "FLAGGED" if result["flag_for_review"] else "clean"
        print(f"{record['id']} [{status}]")
        if result["supported"]:
            print(f"  Supported:   {fmt(result['supported'])}")
        if result["unsupported"]:
            print(f"  Unsupported: {fmt(result['unsupported'])}")
            flagged_count += 1
        if not result["claimed_findings"]:
            print("  No positive findings claimed.")
        print()

    print(f"{flagged_count}/{len(records)} generated reports flagged for review "
          f"(claimed a finding the classifier itself wasn't confident about, threshold={UNSUPPORTED_THRESHOLD})")


if __name__ == "__main__":
    main()
