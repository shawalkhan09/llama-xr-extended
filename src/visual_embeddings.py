"""
Phase 4 (optional, higher-effort extension) -- dense visual embeddings instead of the
paper's 36-number bottleneck.

The paper's LLM never sees the X-ray -- it sees 36 numbers (18 pathology confidence scores
per view). That's a severe information bottleneck: anything not captured by those 18 fixed
labels (devices, incidental findings, laterality, severity nuance -- their own dataset
examples mention things like "cholecystectomy clips" that no label covers) is structurally
invisible to the LLM.

This module is a scaffold for the standard fix, used in real vision-language models
(LLaVA, RaDialog, etc.): take dense penultimate-layer image features instead of the
18-class classifier head, project them into the LLM's embedding space with a small
trainable adapter, and inject them as extra "soft tokens" at the start of the prompt --
instead of serializing 36 numbers as text.

This is a bigger lift than Phases 2-3: it requires hooking into the LLM's embedding layer
(not just prompting it with text), and the projector itself needs a short training pass.
Do this only after Phase 1 (baseline) is working and Phases 2-3 (evaluation) confirm where
the bottleneck is actually costing you accuracy -- e.g. use eval_clinical.py's per-sample
"missed_findings" list to check whether misses cluster around conditions/details that the
18-label taxonomy can't represent in the first place. If they do, that's your evidence this
extension is worth the effort.

Architecture sketch:

    X-ray image (224x224)
        -> DenseNet-121 (frozen, same checkpoint as data_prep.py)
        -> penultimate-layer features (1024-dim, instead of the 18-dim classifier head)
        -> VisualProjector (trainable MLP: 1024 -> LLM hidden size)
        -> N visual "soft tokens" prepended to the text prompt's embeddings
        -> LLaMA 3.1 8B (QLoRA-tuned, as in Phase 1) generates the report conditioned on
           both the visual tokens and the text instruction
"""

import torch
import torch.nn as nn


class VisualProjector(nn.Module):
    """
    Maps dense DenseNet-121 features into the LLM's embedding space, LLaVA-style.
    Only this module is trained in Phase 4 (in addition to the existing LoRA adapters);
    the DenseNet-121 backbone stays frozen, same as Phase 0/1.
    """

    def __init__(self, image_feature_dim: int = 1024, llm_hidden_dim: int = 4096,
                 num_visual_tokens: int = 8):
        super().__init__()
        self.num_visual_tokens = num_visual_tokens
        self.proj = nn.Sequential(
            nn.Linear(image_feature_dim, llm_hidden_dim),
            nn.GELU(),
            nn.Linear(llm_hidden_dim, llm_hidden_dim * num_visual_tokens),
        )
        self.llm_hidden_dim = llm_hidden_dim

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        """
        image_features: [batch, image_feature_dim]  (one frozen DenseNet-121 feature
                         vector per view -- call this twice for frontal + lateral and
                         concatenate along the token dimension, mirroring the paper's
                         frontal+lateral concatenation)
        returns:         [batch, num_visual_tokens, llm_hidden_dim] soft token embeddings
        """
        batch = image_features.shape[0]
        out = self.proj(image_features)
        return out.view(batch, self.num_visual_tokens, self.llm_hidden_dim)


def extract_dense_features(model, image_tensor: torch.Tensor) -> torch.Tensor:
    """
    Pulls the 1024-dim penultimate-layer feature instead of the 18-dim classification
    head used in data_prep.py. Requires the same frozen DenseNet-121 checkpoint.
    """
    with torch.no_grad():
        features = model.features(image_tensor)          # DenseNet feature maps
        pooled = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))
        return pooled.flatten(1)                          # [batch, 1024]


# --- Wiring into training (sketch, not wired up automatically) ---------------------
#
# 1. In data_prep.py, additionally save each image's 1024-dim dense feature vector
#    (via extract_dense_features) alongside the existing 18-dim scores.
# 2. In train_baseline.py, before calling the LLM's forward pass, run the frontal and
#    lateral dense features through a shared VisualProjector to get soft token
#    embeddings, and concatenate them with the tokenizer's text embeddings for the
#    prompt (replacing the "### Input:\n[36 numbers]" section, or keeping both as an
#    ablation).
# 3. Add VisualProjector's parameters to the optimizer alongside the LoRA adapters --
#    everything else in train_baseline.py's TrainingArguments can stay the same.
# 4. Compare against the Phase 1 baseline using both eval_lexical.py and
#    eval_clinical.py -- the clinical F1 on conditions/details NOT in the 18-label
#    taxonomy is the metric that should move if this extension is working.
