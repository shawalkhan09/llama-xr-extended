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
  - Early stopping, best checkpoint restored at the end

NOTE ON LoRA RANK: the paper reports 41,943,040 trainable parameters but does not state
the LoRA rank (r) or alpha used to get there. r=16 (below) is Unsloth's common default for
an 8B model and lands in the same order of magnitude -- treat this as the one hyperparameter
you may need to tune to exactly match their trainable-parameter count, everything else here
is taken directly from the paper's stated numbers.

Run this on Colab (T4 for a slower run, A100 to match the paper's setup) -- not in a plain
CPU environment.
"""

from datasets import load_dataset
from transformers import EarlyStoppingCallback, TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel

MODEL_NAME = "unsloth/Meta-Llama-3.1-8B-bnb-4bit"
MAX_SEQ_LENGTH = 2048
TRAIN_PATH = "data/train.jsonl"  # produced by data_prep.py, from R2Gen's existing train split
VAL_PATH = "data/val.jsonl"      # from R2Gen's existing val split -- no re-splitting needed,
                                  # this keeps you on the same patient-level split the paper uses
OUTPUT_DIR = "outputs/llama-xr-baseline"

ALPACA_TEMPLATE = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}"""


def formatting_func(example):
    return ALPACA_TEMPLATE.format(
        instruction=example["instruction"],
        input=example["input"],
        output=example["output"],
    )


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

    train_dataset = load_dataset("json", data_files=TRAIN_PATH, split="train")
    val_dataset = load_dataset("json", data_files=VAL_PATH, split="train")

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,       # effective batch size 32
        num_train_epochs=3,
        learning_rate=2e-6,
        optim="adamw_8bit",
        lr_scheduler_type="linear",
        warmup_steps=30,
        weight_decay=0.01,
        fp16=True,
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        formatting_func=formatting_func,
        max_seq_length=MAX_SEQ_LENGTH,
        args=training_args,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    trainer.train()
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Saved fine-tuned adapter to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
