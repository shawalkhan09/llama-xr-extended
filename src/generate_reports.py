"""
Phase 1 inference -- generates reports on the held-out test set with the fine-tuned adapter,
producing generated_reports.json: the input eval_lexical.py, eval_clinical.py, and
hallucination_check.py all expect (id, reference_report, generated_report, classifier_scores).

Run from the repo root: python src/generate_reports.py [optional: limit N examples]
"""

import json
import sys
from pathlib import Path

from unsloth import FastLanguageModel

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_prep import CONDITIONS  # noqa: E402

ADAPTER_DIR = "/content/drive/MyDrive/llama-xr-extended/outputs/llama-xr-baseline"
TEST_PATH = "data/test.jsonl"
OUTPUT_PATH = "generated_reports.json"
MAX_SEQ_LENGTH = 1024

ALPACA_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
"""


def load_test_examples(path: str, limit: int = None) -> list:
    examples = []
    with open(path) as f:
        for line in f:
            examples.append(json.loads(line))
    return examples[:limit] if limit else examples


def merge_classifier_scores(input_json_str: str) -> dict:
    """The 36-dim vector data_prep.py wrote is [frontal_18, lateral_18]. hallucination_check.py
    and eval_clinical.py expect a single 18-condition dict, so merge views by taking the max
    confidence across frontal/lateral per condition -- either view showing a strong signal
    is meaningful."""
    scores = json.loads(input_json_str)
    frontal, lateral = scores[:18], scores[18:]
    merged = [max(f, l) for f, l in zip(frontal, lateral)]
    return {condition: round(score, 4) for condition, score in zip(CONDITIONS, merged)}


def extract_response(full_text: str) -> str:
    """The model echoes the whole prompt back before its answer -- keep only what follows
    the last '### Response:' marker."""
    marker = "### Response:"
    idx = full_text.rfind(marker)
    return full_text[idx + len(marker):].strip() if idx != -1 else full_text.strip()


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=ADAPTER_DIR,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)  # switches on Unsloth's faster inference path

    examples = load_test_examples(TEST_PATH, limit=limit)
    results = []

    for i, example in enumerate(examples):
        prompt = ALPACA_PROMPT.format(instruction=example["instruction"], input=example["input"])
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        output_ids = model.generate(
            **inputs, max_new_tokens=200, use_cache=True, do_sample=False,
        )
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        results.append({
            "id": f"test_{i:04d}",
            "reference_report": example["output"],
            "generated_report": extract_response(generated_text),
            "classifier_scores": merge_classifier_scores(example["input"]),
        })

        if (i + 1) % 25 == 0:
            print(f"Generated {i + 1}/{len(examples)}")

    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {len(results)} generated reports to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()