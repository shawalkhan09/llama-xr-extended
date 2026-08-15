"""
Phase 0 -- Data preparation, matching the paper's Section 3.1 / 3.4 exactly.

Pipeline:
  1. Load a frontal + lateral X-ray pair (single-view cases: duplicate the one view's
     18 scores to fill the 36-dim vector, per the paper's handling of the ~12.4% of
     IU X-ray cases with only one view).
  2. Resize to 224x224, single grayscale channel, normalize.
  3. Run the frozen "densenet121-res224-all" model (torchxrayvision) to get an 18-condition
     confidence vector per view.
  4. Concatenate frontal + lateral -> 36-dim vector.
  5. Format as an Alpaca-style instruction/input/output triple, ready for SFT with the
     `trl` library's SFTTrainer (see train_baseline.py).

Expected input layout: R2Gen's standard preprocessed IU X-ray split (images + annotation.json),
downloaded from https://github.com/zhjohnchan/R2Gen#datasets -- this is the same split used by
every method the paper compares against in Table 2, so results are directly comparable.

    data/iu_xray/images/*.png
    data/iu_xray/annotation.json   # {"train": [...], "val": [...], "test": [...]}
                                    # each entry: {"id", "report", "image_path": [img1, img2?]}
"""

import json
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image

CONDITIONS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion",
    "Emphysema", "Enlarged Cardiomediastinum", "Fibrosis", "Fracture", "Hernia",
    "Infiltration", "Lung Lesion", "Lung Opacity", "Mass", "Nodule",
    "Pleural Thickening", "Pneumonia", "Pneumothorax",
]

IMG_SIZE = 224

TRANSFORM = T.Compose([
    T.Grayscale(num_output_channels=1),
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.5], std=[0.5]),
])

ALPACA_INSTRUCTION = (
    "Generate a medical report based on the given classification scores from a patient's "
    "X-ray, which includes two views: Frontal and Lateral. The classification scores "
    "consist of 18 values for each view, representing the confidence scores for 18 "
    "different medical conditions. For both the Frontal and Lateral views, these "
    "conditions are as follows: " + ", ".join(CONDITIONS) + ". The first 18 values "
    "correspond to the Frontal view, while the remaining 18 values correspond to the "
    "Lateral view."
)


def load_classifier():
    """Loads the frozen DenseNet-121 classifier used in the paper. Uses GPU if available --
    on CPU, running this across the full ~7,470-image dataset one image at a time can take
    45-60+ minutes; on GPU it's a few minutes."""
    import torchxrayvision as xrv
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = xrv.models.DenseNet(weights="densenet121-res224-all")
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model.to(device)


@torch.no_grad()
def extract_scores(model, image_path: str) -> list:
    """Returns an 18-dim list of condition confidence scores for one X-ray image."""
    device = next(model.parameters()).device
    img = Image.open(image_path).convert("L")
    tensor = TRANSFORM(img).unsqueeze(0).to(device)
    raw_output = model(tensor)[0]

    model_pathologies = model.pathologies
    scores = []
    for condition in CONDITIONS:
        if condition in model_pathologies:
            idx = model_pathologies.index(condition)
            scores.append(round(float(raw_output[idx]), 8))
        else:
            scores.append(0.0)
    return scores


def build_alpaca_record(frontal_scores: list, lateral_scores: list, report_text: str) -> dict:
    combined = frontal_scores + lateral_scores
    return {
        "instruction": ALPACA_INSTRUCTION,
        "input": json.dumps(combined),
        "output": report_text.strip(),
    }


def prepare_split(model, examples: list, images_dir: Path) -> list:
    records = []
    for example in examples:
        image_paths = example["image_path"]

        frontal_scores = extract_scores(model, str(images_dir / image_paths[0]))
        if len(image_paths) > 1:
            lateral_scores = extract_scores(model, str(images_dir / image_paths[1]))
        else:
            lateral_scores = frontal_scores

        records.append(build_alpaca_record(frontal_scores, lateral_scores, example["report"]))
    return records


def prepare_dataset(annotation_json: str, images_dir: str, output_dir: str) -> None:
    model = load_classifier()
    images_dir = Path(images_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(annotation_json) as f:
        annotations = json.load(f)

    for split in ("train", "val", "test"):
        if split not in annotations:
            continue
        records = prepare_split(model, annotations[split], images_dir)
        out_path = output_dir / f"{split}.jsonl"
        with open(out_path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        print(f"Wrote {len(records)} records to {out_path}")


if __name__ == "__main__":
    prepare_dataset(
        annotation_json="data/iu_xray/annotation.json",
        images_dir="data/iu_xray/images",
        output_dir="data",
    )