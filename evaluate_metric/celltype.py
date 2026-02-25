import json
from pathlib import Path
import re
import evaluate
from tqdm import tqdm  # Optional: show a progress bar.


# Current file path
current_file = Path(__file__)

# Data paths
datasetpath = current_file.parent.parent / "data" / "demo.jsonl"
output_dir = current_file.parent.parent / "outputs" / "celltype"

# Metrics to evaluate
metric_list = ["bertscore", "bleu", "exact_match", "rouge", "meteor"]


# Load reference answers
with open(datasetpath, "r", encoding="utf-8") as infile:
    ref_data = [json.loads(line.strip())["cell_type"].lower() for line in infile]




# Iterate over all *.jsonl files in output_dir (skip demo files).
for output_file in output_dir.glob("*.jsonl"):

    model_name = output_file.stem  # File name without ".jsonl" suffix.
    
    
    if model_name == "demo":
        continue  # Skip demo.jsonl.

    predict_list = []

    # Read prediction file line by line.
    with open(output_file, "r", encoding="utf-8") as pred_file:
        for line in pred_file:
            out_data = json.loads(line.strip())
            response = out_data.get("response", "")

            # Extract predicted cell type using regex.
            pattern = r"\[Predicted_Cell_Type:\s*([^\]]+?)\]"
            match = re.search(pattern, response)

            if match:
                predicted_cell_type = match.group(1).strip().lower()
            else:
                predicted_cell_type = response.strip().lower()

            predict_list.append(predicted_cell_type)

    # Check whether prediction and reference counts match.
    if len(predict_list) != len(ref_data):
        print(f"[Warning] {model_name}: Prediction count ({len(predict_list)}) doesn't match reference count ({len(ref_data)}), skipping.")
        continue
    print(len(predict_list),len(ref_data))

    # Evaluation results
    result_dict = {}

    for metric in tqdm(metric_list, desc=f"Evaluating {model_name}"):
        try:
            metric_module = evaluate.load(metric)
            if metric == "bertscore":
                result = metric_module.compute(predictions=predict_list, references=ref_data, lang="en")
            else:
                result = metric_module.compute(predictions=predict_list, references=ref_data)
            result_dict[metric] = result
        except Exception as e:
            print(f"[Error] Failed to compute {metric} for {model_name}: {e}")
            result_dict[metric] = {"error": str(e)}

    # Save as JSON
    output_result_path = current_file.parent / "celltype" / f"{model_name}.json"
    with open(output_result_path, "w", encoding="utf-8") as fout:
        json.dump(result_dict, fout, indent=2, ensure_ascii=False)

    print(f"[Saved] {model_name} results → {output_result_path.name}")

    

        
        
        
