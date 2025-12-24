from mydatasets.Gen_dataset_Qwen import Gen_dataset_Qwen
import numpy as np
from datasets import load_dataset
import datasets
from pathlib import Path
import os
from peft import LoraConfig, get_peft_model, TaskType
import torch
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from transformers import (
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    EvalPrediction,
)
from sklearn.metrics import accuracy_score
from mydatasets.collator import DataCollatorForCompletionOnlyLM
import argparse
import re

os.environ["WANDB_DISABLED"] = "true"
os.environ["HF_DATASETS_CACHE"] = "./data_cache"

parser = argparse.ArgumentParser()
# Experiment Params
parser.add_argument(
    "--save_path", type=str, default="models/qwen-large", help="Path to save model"
)

args = parser.parse_args()

model_id = "Qwen/Qwen2.5-7B-Instruct"

### loading the tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_id, cache_dir="./model_cache", trust_remote_code=True
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="cuda" if torch.cuda.is_available() else "cpu",
    torch_dtype=torch.bfloat16,
    cache_dir="./model_cache",
    trust_remote_code=True,
)

print(f"model {model_id} loaded!")

### change the dataset path based on your file systems
dataset_names = [
    "Llamapie_dataset/perl/",
    "Llamapie_dataset/soda/",
    "Llamapie_dataset/synthetic/",
]
# Removed "Llamapie_dataset/Generation/" as it wasn't clear if it existed in extraction

dataset_prob = [
    1,
    1,
    1,
]

# aug_config = {
#     "adapt_to_ASR": 1,
#     "drop_word": 0.75,
#     "swap_silence_speaker": 0.3
# }
aug_config = None
neg_prob = 0.25

# Filter existing only
dataset_names = [d for d in dataset_names if os.path.exists(d)]

dataset = Gen_dataset_Qwen(
    tokenizer,
    dataset_names=dataset_names,
    split_set="Train",
    mem_drop_rate=0.15,
    neg_prob=neg_prob,
    history_aware=True,
)
dataset_val = Gen_dataset_Qwen(
    tokenizer,
    dataset_names=dataset_names,
    split_set="Val",
    mem_drop_rate=0,
    neg_prob=neg_prob,
    history_aware=True,
)
print(f"Trainig dataset = {len(dataset)} and val set = {len(dataset_val)}")

config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    bias="none",
    lora_dropout=0.05,
    task_type=TaskType.CAUSAL_LM,
)
model.gradient_checkpointing_enable()
model = get_peft_model(model, config)
model.print_trainable_parameters()

training_args = TrainingArguments(
    output_dir=args.save_path,
    eval_strategy="steps",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    learning_rate=2e-5,
    weight_decay=0.01,
    push_to_hub=False,
    report_to="none",
    num_train_epochs=2,  # Reduced for demo
    save_steps=100,
    logging_steps=50,
    batch_eval_metrics=True,
    gradient_accumulation_steps=2,  # To help with memory
)


## define eval metric
# Define compute_metrics
def compute_metrics(pred: EvalPrediction, compute_result: bool):
    predictions, labels = pred.predictions, pred.label_ids
    # predictions, labels = predictions.cpu(), labels.cpu() # EvalPrediction typically numpy/torch. Check trainer.
    # If they are numpy:
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    predicted_class_indices = np.argmax(predictions, axis=-1)

    flattened_preds = predicted_class_indices.flatten()
    flattened_labels = labels.flatten()
    valid_indices = flattened_labels != -100
    flattened_preds = flattened_preds[valid_indices]
    flattened_labels = flattened_labels[valid_indices]
    accuracy = accuracy_score(flattened_labels, flattened_preds)
    return {"accuracy": accuracy}


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=dataset_val,
    data_collator=DataCollatorForCompletionOnlyLM(
        instruction_template=None,
        response_template="<|im_start|>assistant\n",
        tokenizer=dataset.tokenizer,
        mlm=False,
    ),
    compute_metrics=compute_metrics,
)

trainer.train(resume_from_checkpoint=False)
model.save_pretrained(args.save_path)
tokenizer.save_pretrained(args.save_path)
