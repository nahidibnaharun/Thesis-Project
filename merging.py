import pandas as pd
import os

data_dir = "D:/download/Thesis/actual_files"
output_file = "D:/download/Thesis/merged_training_data.csv"

# Load the files
gsm = pd.read_csv(os.path.join(data_dir, "gsm8k_train.csv"))
halu = pd.read_csv(os.path.join(data_dir, "halueval_test.csv"))
bbh = pd.read_csv(os.path.join(data_dir, "bbh_logic.csv"))
sqa = pd.read_csv(os.path.join(data_dir, "strategy_qa.csv"))

print("HaluEval columns found:", halu.columns.tolist())

def prepare_data():
    final_rows = []

    # 1. GSM8K
    for _, row in gsm.iterrows():
        final_rows.append({'text': f"Q: {row['question']} A: {row['answer']}", 'label': 0})

    # 2. HaluEval (FIXED: Handling dynamic column names)
    for _, row in halu.iterrows():
        # HaluEval usually uses 'knowledge', 'question', 'right_answer', and 'hallucinated_answer'
        # We try to find the right keys:
        q = row.get('question', row.get('input', ''))
        correct = row.get('right_answer', row.get('reference', ''))
        hallucinated = row.get('hallucinated_answer', row.get('output', ''))
        
        if correct:
            final_rows.append({'text': f"Q: {q} R: {correct}", 'label': 0})
        if hallucinated:
            final_rows.append({'text': f"Q: {q} R: {hallucinated}", 'label': 1})

    # 3. BBH
    for _, row in bbh.iterrows():
        final_rows.append({'text': f"Q: {row['input']} R: {row['target']}", 'label': 0})

    # 4. StrategyQA
    for _, row in sqa.iterrows():
        final_rows.append({'text': f"Q: {row['question']} A: {row['answer']}", 'label': 0})

    return pd.DataFrame(final_rows)

# Create, Shuffle, and Save
df_final = prepare_data()
df_final = df_final.sample(frac=1).reset_index(drop=True)
df_final.to_csv(output_file, index=False)

print(f"✅ Success! Created merged dataset with {len(df_final)} rows.")
print(f"Sample Text: {df_final['text'].iloc[0][:100]}...")