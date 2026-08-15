"""
Phase 1 -- Baseline reproduction.

Matches the paper's Section 3.5 training setup as closely as the paper specifies, adapted
for T4 memory constraints (see note below).
"""

from unsloth import FastLanguageModel  # must be imported first -- before trl/transformers/peft,
                                        # or Unsloth's tokenizer patches don't apply correctly
                                        # and eos_token gets corrupted (unslothai/unsloth#2797)
from datasets import load_dataset
from transformers import EarlyStoppingCallback
from trl import SFTConfig, SFTTrainer

MODEL_NAME = "unsloth/Meta-Llama-3.1-8B-bnb-4bit"
MAX_SEQ_LENGTH = 1024  # the paper's radiology reports are short (a few sentences) --
                        # 2048 was wasted headroom on a memory-constrained GPU
TRAIN_PATH = "data/train.jsonl"
VAL_PATH = "data/val.jsonl"
OUTPUT_DIR = "outputs/llama-xr-baseline"

# NOTE ON GPU: the paper trains on an A100 with per_device_train_batch_size=8. On a free-tier
# T4 (14.5GB) that OOMs on the very first step. batch_size=2 x grad_accum=16 (below) keeps the
# same *effective* batch size of 32 as the paper, just spread over more, smaller steps.

ALPACA_TEMPLATE = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}"""


def add_text_field(example):
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
        r=16,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing=True,
    )

    train_dataset = load_dataset("json", data_files=TRAIN_PATH, split="train").map(add_text_field)
    val_dataset = load_dataset("json", data_files=VAL_PATH, split="train").map(add_text_field)

    sft_config = SFTConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2,       # was 8 in the paper -- reduced for T4 memory
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=16,      # was 4 -- 2 x 16 = 32, same effective batch
                                              # size as the paper's 8 x 4
        num_train_epochs=3,
        learning_rate=2e-6,
        optim="adamw_8bit",
        lr_scheduler_type="linear",
        warmup_steps=30,
        weight_decay=0.01,
        fp16=True,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        max_length=MAX_SEQ_LENGTH,
        dataset_text_field="text",
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=sft_config,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    trainer.train()
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Saved fine-tuned adapter to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()