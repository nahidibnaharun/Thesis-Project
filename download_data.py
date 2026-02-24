import os
import pandas as pd
from datasets import load_dataset

# 1. Setup the folder
output_dir = "D:/download/Thesis/actual_files"
os.makedirs(output_dir, exist_ok=True)

def save_csv(name, df):
    df.to_csv(os.path.join(output_dir, f"{name}.csv"), index=False)
    print(f"✅ Saved: {name}.csv ({len(df)} rows)")

# --- 1. HaluEval (Pick 'qa' config) ---
print("Downloading HaluEval (QA)...")
# We specify 'qa' to avoid the 'Config name is missing' error
halu = load_dataset("pminervini/HaluEval", "qa", split="data")
save_csv("halueval_qa", pd.DataFrame(halu))

# --- 2. StrategyQA ---
print("Downloading StrategyQA...")
sqa = load_dataset("ChilleD/StrategyQA", split="train")
save_csv("strategy_qa", pd.DataFrame(sqa))

# --- 3. WikiBio Hallucination (SelfCheckGPT Data) ---
print("Downloading WikiBio Hallucination...")
# Note: This is the specific dataset used in the SelfCheckGPT paper
sc = load_dataset("potsawee/wiki_bio_gpt3_hallucination", split="evaluation")
save_csv("wikibio_hallucination", pd.DataFrame(sc))

# --- 4. QAGS (Faithfulness) ---
print("Downloading QAGS...")
qags = load_dataset("artidoro/qags", "cnndm", split="test")
save_csv("qags_cnndm", pd.DataFrame(qags))

print(f"\n🔥 SUCCESS! Check your folder: {output_dir}")