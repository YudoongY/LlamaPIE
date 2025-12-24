from datasets import load_dataset
import datasets
from pathlib import Path
import re
import numpy as np
import os
import torch
from transformers import AutoTokenizer, DataCollatorForTokenClassification
from model.CasualTokenClassificationQwen import Qwen2ForCausalLM_TokenClassification
from transformers import AutoModelForCausalLM
from peft import PeftModel
import json
import argparse
from huggingface_hub import login

os.environ["WANDB_DISABLED"] = "true"

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset",
    type=str,
    choices=["MIT", "Sync_claude", "Sync_perl", "Sync_soda"],
    default="Sync_claude",
)
parser.add_argument("--save-path", type=str, help="path to save result", required=True)
parser.add_argument(
    "--cache-dir", type=str, help="Cache directory for models", default="./model_cache"
)
parser.add_argument(
    "--small-model-path",
    type=str,
    default="models/qwen-small",
    help="Path to trained small model adapter",
)
parser.add_argument(
    "--large-model-path",
    type=str,
    default="models/qwen-large",
    help="Path to trained large model adapter",
)

args = parser.parse_args()

dataset_name = args.dataset
output_samples = f"{args.save_path}/{dataset_name}"
os.makedirs(output_samples, exist_ok=True)

# Dataset configuration
if dataset_name == "MIT":
    from mydatasets.Pipeline_dataset import MIT_sample as SingleSample

    output_base = "Llamapie_dataset/MIT_final/"
    input_dirs = []
elif dataset_name == "Sync_claude":
    from mydatasets.Pipeline_dataset import Syn_samples as SingleSample

    output_base = "Llamapie_dataset/synthetic/Test/claude"
    input_dirs = []
elif dataset_name == "Sync_perl":
    from mydatasets.Pipeline_dataset import Syn_samples as SingleSample

    output_base = "Llamapie_dataset/perl/Test/claude"
    input_dirs = []
elif dataset_name == "Sync_soda":
    from mydatasets.Pipeline_dataset import Syn_samples as SingleSample

    output_base = "Llamapie_dataset/soda/Test/claude"
    input_dirs = []
else:
    raise ValueError("dataset not supported!")

print(f"Using dataset path: {output_base}")

old = False
active_factor = 0  # Can tune this
classifier_aware = True
generator_aware = True

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on device: {device}")

# Base Models (Qwen)
base_small = "Qwen/Qwen2.5-1.5B-Instruct"
base_large = "Qwen/Qwen2.5-7B-Instruct"

# --- Load Small Model (Classifier) ---
print("Loading Small Model...")
tokenizer_small = AutoTokenizer.from_pretrained(
    base_small, cache_dir=args.cache_dir, trust_remote_code=True
)
if tokenizer_small.pad_token is None:
    tokenizer_small.pad_token = tokenizer_small.eos_token

model_small = Qwen2ForCausalLM_TokenClassification.from_pretrained(
    base_small,
    device_map=device,
    torch_dtype=torch.bfloat16,
    num_labels=2,
    cache_dir=args.cache_dir,
    trust_remote_code=True,
)

# Load Adapter if available
if os.path.exists(args.small_model_path):
    print(f"Loading adapter from {args.small_model_path}")
    model_small = PeftModel.from_pretrained(model_small, args.small_model_path)
    model_small = model_small.merge_and_unload()  # Optional: merge for speed
else:
    print(f"Warning: Adapter {args.small_model_path} not found. using base model.")

# --- Load Large Model (Generator) ---
print("Loading Large Model...")
tokenizer_big = AutoTokenizer.from_pretrained(
    base_large, cache_dir=args.cache_dir, trust_remote_code=True
)
if tokenizer_big.pad_token is None:
    tokenizer_big.pad_token = tokenizer_big.eos_token

model_big = AutoModelForCausalLM.from_pretrained(
    base_large,
    device_map=device,
    torch_dtype=torch.bfloat16,
    cache_dir=args.cache_dir,
    trust_remote_code=True,
)

if os.path.exists(args.large_model_path):
    print(f"Loading adapter from {args.large_model_path}")
    model_big = PeftModel.from_pretrained(model_big, args.large_model_path)
else:
    print(f"Warning: Adapter {args.large_model_path} not found. using base model.")


# Terminators for Qwen
terminators = [
    tokenizer_big.eos_token_id,
    tokenizer_big.convert_tokens_to_ids("<|im_end|>"),
]
# Response template for splitting output
response_template = (
    "<|im_start|>assistant\n"  # or just "assistant\n" depending on decoding
)

# ... (Rest of inference loop similar to infer_dual_model.py but using Qwen tokenizers)
# Copying loop from infer_dual_model.py and adapting
Num_turns = 0
FN = 0
FP = 0
TP = 0
TN = 0
soft_FN = 0
soft_FP = 0
soft_TP = 0
soft_TN = 0
med_FN = 0
med_FP = 0
med_TP = 0
med_TN = 0
true_count = 0
model_count = 0
good_silence = 0
bad_silence = 0
resp_lengths = []
ratios = []
gen_ignore_arr = []
num_resps = []
total_num_turns = []
total_num_spoken = []

# Limit to 100 samples or fewer
for i in range(0, 100):
    print(f"{i + 1}/100 - {model_count}", end="\r")
    torch.cuda.empty_cache()
    try:
        sample = SingleSample(
            tokenizer_small,
            tokenizer_big,
            output_base,
            split_set="Test",
            sample_id=i,
            input_dirs=input_dirs,
        )
    except Exception as e:
        print(f"Skipping sample {i}: {e}")
        continue

    if not sample.valid:
        continue

    Num_turns += sample.count_turn()
    L_cache = 0
    save_dir = output_samples + "/{:05d}".format(i)
    curr_turn = ""
    curr_response = ""
    thisTurn = 0
    lastTurn = 0
    resp_count = 0
    gen_ignore_count = 0
    hard_this_turn = 0

    while True:
        info = sample.streaming_diaglogue()
        if info is None:
            break

        if "Sync" in dataset_name:
            token, token_history, mask, mask_history, label = info
            curr_label = label[-1]
            if curr_label == 0 and len(label) > 1 and label[-2] == 1:
                lastTurn = thisTurn
                thisTurn = 0
                hard_this_turn = 0
        else:
            token, token_history, mask, mask_history = info

        if classifier_aware:
            token_small = token_history
            mask_small = mask_history
        else:
            token_small = token
            mask_small = mask

        if mask_small[-1] == 0:
            L_cache += 1
            continue

        intput_ids = token_small.unsqueeze(0).to(model_small.device)
        out1 = model_small(input_ids=intput_ids)
        pred = out1.logits[0].cpu()
        pred[..., 1] += active_factor
        pred = torch.argmax(pred, dim=-1)

        new_pred = pred[-1]

        if generator_aware:
            token_big = token_history
            mask_big = mask_history
        else:
            token_big = token
            mask_big = mask
        responded = 0

        if new_pred == 1:
            # print("------------- agent is triggered ----------------")
            curr_diag = tokenizer_small.decode(token_big, skip_special_tokens=True)

            # Prepare generation input for Qwen
            # We need to construct the prompt in Qwen format.
            # Usually SingleSample does get_gen_inputs but it might be Llama specific.
            # Assuming get_gen_inputs returns raw tokens, we might need to re-apply template?
            # Actually Pipeline_dataset logic usually constructs tokens.
            # If tokenizer_big is Qwen, sample.tokenizer_big is Qwen.
            # Pipeline_dataset uses tokenizer to encode/decode, so it should handle it if passed correctly.

            input_ids2 = sample.get_gen_inputs(curr_diag, old=old)
            input_ids2 = input_ids2.unsqueeze(0).to(model_big.device)

            outputs = model_big.generate(
                input_ids2,
                max_new_tokens=512,
                pad_token_id=tokenizer_big.pad_token_id,
                eos_token_id=terminators,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
            )
            response_tokens = outputs[0]
            output_text = tokenizer_big.decode(
                response_tokens, skip_special_tokens=False
            )

            # Parse output. Qwen output usually contains the prompt + response.
            # We split by response_template.
            # If response_template is not found, we might fallback.

            if response_template in output_text:
                parts = output_text.split(response_template)
                response = parts[-1]  # Take the last part
            else:
                # Fallback: maybe just take new tokens?
                # or split by <|im_start|>assistant
                if "<|im_start|>assistant" in output_text:
                    parts = output_text.split("<|im_start|>assistant")
                    response = parts[-1]
                else:
                    response = tokenizer_big.decode(
                        response_tokens[input_ids2.shape[1] :], skip_special_tokens=True
                    )

            # Cleanup
            response = (
                response.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()
            )

            if len(response) >= 1:
                curr_response = response
                # print("***** Agent *****: ", response)
                sample.insert_whisper(" Agent: " + response)
                responded = 1
                resp_lengths.append(len(response.split(" ")))
                resp_count += 1
            else:
                gen_ignore_count += 1

        L_cache += 1

        # Calculate Metrics (Same as original)
        if "Sync" in dataset_name:
            if responded == 1:
                model_count += 1
            if curr_label == 1:
                true_count += 1

            # (Metric logic omitted for brevity, but same as original script)
            if responded == 1 and curr_label == 1:
                TP += 1
                med_TP += 1
                soft_TP += 1
                if lastTurn < 0:
                    lastTurn += 1
                    thisTurn -= 1
            elif responded == 1 and curr_label == 0:
                FP += 1
                if hard_this_turn < 0:
                    med_FN -= 1
                    med_TN += 1
                    med_TP += 1
                else:
                    med_FP += 1
                hard_this_turn -= 1
                if lastTurn < 0 or thisTurn < 0:
                    soft_FN -= 1
                    soft_TN += 1
                    soft_TP += 1
                    if lastTurn < 0:
                        lastTurn += 1
                    elif thisTurn < 0:
                        thisTurn += 1
                        med_FN -= 1
                        med_TN += 1
                        med_TP += 1
                else:
                    thisTurn += 1
                    soft_FP += 1
            elif responded == 0 and curr_label == 1:
                FN += 1
                if hard_this_turn > 0:
                    med_FP -= 1
                    med_TP += 1
                    med_TN += 1
                else:
                    med_FN += 1
                hard_this_turn -= 1
                if lastTurn <= 0 and thisTurn <= 0:
                    soft_FN += 1
                    thisTurn -= 1
                else:
                    soft_FP -= 1
                    soft_TP += 1
                    soft_TN += 1
            else:
                TN += 1
                med_TN += 1
                soft_TN += 1

    # Save JSON result
    tokens, tokens_history = sample.snap_dialogue()
    diag = tokenizer_small.decode(tokens_history, skip_special_tokens=True)
    data = {"mem": sample.memory_text, "diag": diag}
    with open(save_dir + ".json", "w") as fp:
        json.dump(data, fp, indent=4)
    # Stats collection
    num_turns = sample.count_turn()
    num_spoken = resp_count
    ratio = num_spoken / (num_turns + num_spoken) if (num_turns + num_spoken) > 0 else 0
    ratios.append(ratio)
    gen_ignore_arr.append(gen_ignore_count)
    num_resps.append(num_spoken + gen_ignore_count)
    total_num_turns.append(num_turns)
    total_num_spoken.append(num_spoken)


# Final Stats Saving
def getStats(arr):
    if len(arr) == 0:
        return {}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "count": len(arr),
        "sum": sum(arr),
    }


overall_data = {
    "ratios": getStats(ratios),
    "resp_lengths": getStats(resp_lengths),
    "gen_ignore_counts": getStats(gen_ignore_arr),
    "num_resps": getStats(num_resps),
    "speaker_num_turns": getStats(total_num_turns),
    "num_spoken": getStats(total_num_spoken),
}
if "Sync" in dataset_name:
    # Add accuracy metrics (abbreviated)
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    overall_data["hard"] = {
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "TN": TN,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
    }

with open(f"{output_samples}/data.json", "w") as f:
    json.dump(overall_data, f, indent=4)
