from datasets import load_dataset
import datasets
from pathlib import Path
import os
import string
import numpy as np
from peft import LoraConfig, get_peft_model, TaskType
import argparse

os.environ["WANDB_DISABLED"] = "true"
os.environ["HF_DATASETS_CACHE"] = "./data_cache"
import torch
from transformers import AutoTokenizer, DataCollatorForTokenClassification
from transformers import (
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    EvalPrediction,
)
from model.CasualTokenClassificationQwen import Qwen2ForCausalLM_TokenClassification
from sklearn.metrics import accuracy_score
from mydatasets.Active_dataset import New_WhisperAware_dataset

# Qwen Model
model_id = "Qwen/Qwen2.5-1.5B-Instruct"

### loading the tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_id, cache_dir="./model_cache", trust_remote_code=True
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = Qwen2ForCausalLM_TokenClassification.from_pretrained(
    model_id,
    device_map="cuda" if torch.cuda.is_available() else "cpu",
    torch_dtype=torch.bfloat16,
    num_labels=2,
    cache_dir="./model_cache",
    trust_remote_code=True,
)

### change the dataset path based on your file systems
# Assuming local extraction to Llamapie_dataset
positive_base_train = [
    "Llamapie_dataset/synthetic/Train/claude",
    "Llamapie_dataset/perl/Train/claude",
    "Llamapie_dataset/soda/Train/claude",
]
positive_base_dev = [
    "Llamapie_dataset/synthetic/Val/claude",
    "Llamapie_dataset/perl/Val/claude",
    "Llamapie_dataset/soda/Val/claude",
]

parser = argparse.ArgumentParser()
# Experiment Params
parser.add_argument(
    "--save_path", type=str, help="Path to save model", default="models/qwen-small"
)

args = parser.parse_args()


# aug_config = {
#     "adapt_to_ASR": 1,
#     "drop_word": 0.7,
#     "swap_silence_speaker": 0.3
# }
aug_config = None
save_dir = args.save_path
negative_base_train = None
negative_base_dev = None

# Filter out paths that don't exist to avoid crashing
positive_base_train = [p for p in positive_base_train if os.path.exists(p)]
positive_base_dev = [p for p in positive_base_dev if os.path.exists(p)]

print(f"Loading training data from: {positive_base_train}")
dataset = New_WhisperAware_dataset(
    tokenizer,
    input_dirs=positive_base_train,
    split_set="Train",
    negative_base=negative_base_train,
    aug_config=aug_config,
)
dataset_val = New_WhisperAware_dataset(
    tokenizer,
    input_dirs=positive_base_dev,
    split_set="Val",
    negative_base=negative_base_dev,
    aug_config=aug_config,
)

print(f"Train {len(dataset)} and val {len(dataset_val)}")

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
    ],  # Qwen standard modules
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    modules_to_save=["classifier"],
)
model = get_peft_model(model, config)
model.print_trainable_parameters()

training_args = TrainingArguments(
    output_dir=save_dir,
    eval_strategy="steps",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=2e-5,
    weight_decay=0.01,
    push_to_hub=False,
    report_to="none",
    num_train_epochs=2,  # Reduced for demo/speed
    save_steps=100,
    logging_steps=50,
)


## define eval metric
def compute_metrics(pred: EvalPrediction):
    predictions, labels, inputs = pred.predictions, pred.label_ids, pred.inputs
    predicted_class_indices = np.argmax(predictions, axis=-1)
    flattened_preds = predicted_class_indices.flatten()
    flattened_labels = labels.flatten()
    valid_indices = flattened_labels != -100
    flattened_preds = flattened_preds[valid_indices]
    flattened_labels = flattened_labels[valid_indices]
    accuracy = accuracy_score(flattened_labels, flattened_preds)
    ### iterrate with each batch

    diag_correct = 0
    # Resizing predictions might depend on batching, simplistic loop here
    # For safety in metric computation when batch sizes vary or flattening happens:
    # We stick to the global accuracy for now.

    return {"accuracy": accuracy}


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=dataset_val,
    data_collator=DataCollatorForTokenClassification(tokenizer=dataset.tokenizer),
    compute_metrics=compute_metrics,
)

trainer.train(resume_from_checkpoint=False)
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)
