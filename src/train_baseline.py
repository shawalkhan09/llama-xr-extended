"""
Phase 1 -- Baseline reproduction.

Matches the paper's Section 3.5 training setup as closely as the paper specifies:
  - Base model: LLaMA 3.1 8B, 4-bit quantized, loaded via Unsloth
  - Method: QLoRA + SFT via trl's SFTTrainer
  - Batch size 8, gradient accumulation 4 (effective batch size 32)
  - 3 epochs
  - Learning rate 2e-6, AdamW 8-bit optimizer
  - Linear LR schedule, 30 warmup steps
  - Weight decay 0.01
  - fp16 mixed precision

NOTE ON LoRA RANK: the paper reports 41,943,040 trainable parameters but does not state
the LoRA rank (r) or alpha used to get there. r=16 (below) is Unsloth's common default for
an 8B model and lands in the same order of magnitude.

NOTE ON TRL VERSION: trl's SFTTrainer API changed significantly across 2025-2026 releases --
`tokenizer=` was replaced by `processing_class=`, `max_seq_length` moved off SFTTrainer and
onto `SFTConfig` as `max_length`, and `TrainingArguments`' `evaluation_strategy` was renamed
to `eval_strategy`.

Run this on Colab (T4 for a slower run, A100 to match the paper's setup) -- not in a plain
CPU environment.
"""

from unsloth import FastLanguageModel  # must be imported first -- before trl/transformers/peft,
                                        # or Unsloth's tokenizer patches don't apply correctly
                                        # and eos_token gets corrupted (unslothai/unsloth#2797)
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer

MODEL_NAME = "unsloth/Meta-Llama-3.1-8B-bnb-4bit"
MAX_SEQ_LENGTH = 1024  # the paper's radiology reports are short (a few sentences) --
                        # 2048 was wasted headroom on a memory-constrained GPU
TRAIN_PATH = "data/train.jsonl"  # produced by data_prep.py, from R2Gen's existing train split
VAL_PATH = "data/val.jsonl"      # from R2Gen's existing val split
NUM_EPOCHS = 3  # matches the paper. For a fast first end-to-end smoke test before committing
                # hours of Colab time, temporarily set this to 1.

# Checkpoints go to Drive, not local Colab storage -- /content is wiped on disconnect, and a
# multi-hour T4 run is a real disconnect risk on free tier. Requires Drive already mounted.
# Adjust the path if your Drive folder is named differently.
OUTPUT_DIR = "/content/drive/MyDrive/llama-xr-extended/outputs/llama-xr-baseline"

# NOTE ON GPU: the paper trains on an A100 with per_device_train_batch_size=8. On a free-tier
# T4 (14.5GB) that OOMs on the very first step. batch_size=2 x grad_accum=16 (below, in
# SFTConfig) keeps the same *effective* batch size of 32 as the paper, just spread over more,
# smaller steps. If you have Colab Pro with an A100, feel free to set these back to 8 / 4.

ALPACA_TEMPLATE = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}"""


def add_text_field(example):
    """Pre-bakes the Alpaca-formatted string into a `text` column -- the most stable,
    version-independent way to hand SFTTrainer pre-formatted examples (dataset_text_field),
    rather than relying on formatting_func, whose exact call signature has shifted across
    trl versions."""
    example["text"] = ALPACA_TEMPLATE.format(
        instruction=example["instruction"],
        input=example["input"],
        output=example["output"],
    )
    return example


def main():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,             # see note above on trainable-parameter count
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing=True,
    )

    train_dataset = load_dataset("json", data_files=TRAIN_PATH, split="train").map(add_text_field)

    sft_config = SFTConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2,       # was 8 in the paper -- reduced for T4 memory
        gradient_accumulation_steps=16,      # was 4 -- 2 x 16 = 32, same effective batch
                                              # size as the paper's 8 x 4
        num_train_epochs=NUM_EPOCHS,
        learning_rate=2e-6,
        optim="adamw_8bit",
        lr_scheduler_type="linear",
        warmup_steps=30,
        weight_decay=0.01,
        fp16=True,
        logging_steps=10,
        eval_strategy="no",                  # in-training loss eval is redundant here -- we
                                              # evaluate the saved model afterward with
                                              # eval_clinical.py, which is the metric that
                                              # actually matters for this project
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,                  # cap checkpoints kept on Drive
        max_length=MAX_SEQ_LENGTH,           # was max_seq_length on SFTTrainer in older trl
        dataset_text_field="text",
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,          # was tokenizer= in older trl
        train_dataset=train_dataset,
        args=sft_config,
    )

    trainer.train()
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Saved fine-tuned adapter to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()