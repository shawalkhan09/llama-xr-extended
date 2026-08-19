"""
Phase 1 -- Baseline reproduction, with one deliberate deviation from the paper.

The paper (arXiv:2506.03178) states a learning rate of 2e-6. Reproducing that exactly
produced a model that had clearly not learned the task -- generated reports just recited
the instruction's own list of 18 condition names verbatim, regardless of the actual input
scores, rather than grounding in them. 2e-6 is roughly 100x lower than typical QLoRA
learning rates (commonly 1e-4 to 3e-4), and combined with only ~195 gradient updates over
3 epochs, this reproduction found it insufficient for the LoRA adapter to move the model
away from that degenerate behavior. LEARNING_RATE below is raised to 2e-4 as a documented
deviation -- worth reporting explicitly as a reproducibility finding, not hidden.

We've also restored the paper's periodic in-training evaluation and
load_best_model_at_end=True (removed in an earlier pass for speed) -- both to get loss-curve
visibility this time, and because "keep whatever checkpoint landed at the final step" isn't
actually what the paper describes doing.
"""

from unsloth import FastLanguageModel  # must be imported first -- before trl/transformers/peft,
                                        # or Unsloth's tokenizer patches don't apply correctly
                                        # and eos_token gets corrupted (unslothai/unsloth#2797)
from datasets import load_dataset
from transformers import EarlyStoppingCallback
from trl import SFTConfig, SFTTrainer

MODEL_NAME = "unsloth/Meta-Llama-3.1-8B-bnb-4bit"
MAX_SEQ_LENGTH = 1024
TRAIN_PATH = "data/train.jsonl"
VAL_PATH = "data/val.jsonl"
NUM_EPOCHS = 3
LEARNING_RATE = 2e-4  # was 2e-6 (the paper's stated value) -- see module docstring

# Adjust this to wherever you want checkpoints saved on this platform.
OUTPUT_DIR = "outputs/llama-xr-baseline-v2"

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
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=16,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        optim="adamw_8bit",
        lr_scheduler_type="linear",
        warmup_steps=30,
        weight_decay=0.01,
        fp16=True,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=25,                       # tighter than before (was 50) -- with only ~195
                                              # steps total, we want several data points on
                                              # the loss curve, not just three or four
        save_strategy="steps",
        save_steps=25,
        save_total_limit=3,
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
