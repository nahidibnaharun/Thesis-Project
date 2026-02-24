import os
from datasets import load_dataset

# We use download_mode="reuse_dataset_if_exists" to check the cache
datasets_to_check = {
    "GSM8K": ("openai/gsm8k", "main", "train"),
    "HaluEval": ("flowaicom/HaluEval", None, "test"),
    "BBH": ("lukaemon/bbh", "logical_deduction_five_objects", "test"),
    "StrategyQA": ("wics/strategy-qa", None, "train")
}

print("--- Verifying Local Cache ---")
for name, (path, config, split) in datasets_to_check.items():
    try:
        # trust_remote_code=True is needed for StrategyQA
        data = load_dataset(path, config, split=split, trust_remote_code=True)
        print(f"✅ {name}: Found in Cache ({len(data)} rows)")
    except Exception as e:
        print(f"❌ {name}: NOT FOUND. Error: {e}")