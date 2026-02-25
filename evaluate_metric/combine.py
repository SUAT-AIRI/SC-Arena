import os
import json
import pandas as pd
from statistics import mean

# Folder containing the result JSON files
folder_path = "./evaluate_metric/perturbation"  # Update this with the actual folder path

# Initialize list to collect data rows
data_rows = []

# Iterate through all files in the folder
for file_name in os.listdir(folder_path):
    if file_name.endswith("_results.json"):
        model_name = file_name.replace("_results.json", "")
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, "r") as f:
            data = json.load(f)

        # Extract metrics safely with defaults
        bert_f1 = mean(data.get("bertscore", {}).get("f1", [])) * 100
        bleu_precisions = data.get("bleu", {}).get("precisions", [0, 0])
        bleu_1 = bleu_precisions[0] * 100 if len(bleu_precisions) > 0 else 0
        bleu_2 = bleu_precisions[1] * 100 if len(bleu_precisions) > 1 else 0

        rouge_data = data.get("rouge", {})
        rouge_1 = rouge_data.get("rouge1", 0)
        rouge_2 = rouge_data.get("rouge2", 0)
        rouge_L = rouge_data.get("rougeL", 0)

        meteor = data.get("meteor", 0)

        data_rows.append({
            "model": model_name,
            "bert_f1_avg": bert_f1,
            "bleu_1": bleu_1,
            "bleu_2": bleu_2,
            "rouge_1": rouge_1,
            "rouge_2": rouge_2,
            "rouge_L": rouge_L,
            "meteor": meteor
        })

# Create DataFrame and save as CSV
results_df = pd.DataFrame(data_rows)
csv_output_path = "./evaluate_metric/perturbation/combined_results.csv"
results_df.to_csv(csv_output_path, index=False)


