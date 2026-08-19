"""
Phase 4 data prep -- extracts dense 1024-dim penultimate-layer DenseNet-121 features per
image, instead of the 18-dim classifier head used in data_prep.py. This is the input Phase 4
training needs: real visual features instead of the 36-number bottleneck.

Precomputes and caches these once (DenseNet-121 stays frozen, same checkpoint as Phase 0) --
much cheaper than re-running the CNN forward pass every training epoch.

Output layout, per split (train/val/test):
    data/dense/{split}_features.pt   -- tensor [N, 2, 1024] (N examples, frontal+lateral, 1024-dim)
    data/dense/{split}_meta.jsonl    -- {id, instruction, output} per example, same order as
                                         the tensor rows, so row i in the .pt file corresponds
                                         to line i in this file

Run from the repo root: python src/data_prep_dense.py
"""

import json
from pathlib import Path

import torch

from data_prep import CONDITIONS, IMG_SIZE, TRANSFORM, ALPACA_INSTRUCTION  # reuse, don't duplicate


def load_classifier():
    """Same frozen DenseNet-121 as Phase 0, kept frozen -- inference only."""
    import torchxrayvision as xrv
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = xrv.models.DenseNet(weights="densenet121-res224-all")
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model.to(device)


@torch.no_grad()
def extract_dense_features(model, image_path: str) -> torch.Tensor:
    """Returns the 1024-dim penultimate-layer feature vector for one image (not the
    18-class classifier head)."""
    from PIL import Image
    device = next(model.parameters()).device
    img = Image.open(image_path).convert("L")
    tensor = TRANSFORM(img).unsqueeze(0).to(device)
    features = model.features(tensor)                          # DenseNet feature maps
    pooled = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))
    return pooled.flatten(1).squeeze(0).cpu()                   # [1024]


def prepare_dense_split(model, examples: list, images_dir: Path) -> tuple:
    """Returns (feature_tensor [N, 2, 1024], meta_records list)."""
    all_features = []
    meta_records = []

    for example in examples:
        image_paths = example["image_path"]

        frontal_feat = extract_dense_features(model, str(images_dir / image_paths[0]))
        if len(image_paths) > 1:
            lateral_feat = extract_dense_features(model, str(images_dir / image_paths[1]))
        else:
            lateral_feat = frontal_feat  # single-view case, same convention as Phase 0

        all_features.append(torch.stack([frontal_feat, lateral_feat]))  # [2, 1024]
        meta_records.append({
            "id": example.get("id", ""),
            "instruction": ALPACA_INSTRUCTION,
            "output": example["report"].strip(),
        })

    feature_tensor = torch.stack(all_features)  # [N, 2, 1024]
    return feature_tensor, meta_records


def prepare_dense_dataset(annotation_json: str, images_dir: str, output_dir: str) -> None:
    model = load_classifier()
    images_dir = Path(images_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(annotation_json) as f:
        annotations = json.load(f)

    for split in ("train", "val", "test"):
        if split not in annotations:
            continue
        feature_tensor, meta_records = prepare_dense_split(model, annotations[split], images_dir)

        torch.save(feature_tensor, output_dir / f"{split}_features.pt")
        with open(output_dir / f"{split}_meta.jsonl", "w") as f:
            for r in meta_records:
                f.write(json.dumps(r) + "\n")

        print(f"{split}: saved {feature_tensor.shape} feature tensor and "
              f"{len(meta_records)} meta records to {output_dir}")


if __name__ == "__main__":
    prepare_dense_dataset(
        annotation_json="data/iu_xray/annotation.json",
        images_dir="data/iu_xray/images",
        output_dir="data/dense",
    )
