import json
from pathlib import Path
import re
import evaluate
from tqdm import tqdm  # Optional: show a progress bar.
import pronto


# Current file path
current_file = Path(__file__)

# Data paths
datasetpath = current_file.parent.parent / "data" / "demo.jsonl"
output_dir = current_file.parent.parent / "outputs" / "captioning_new_plus"
obo_path = current_file.parent.parent / "evaluators" / "cl_nl6.obo"
ontology = pronto.Ontology(str(obo_path))

# Metrics to evaluate
metric_list = ["bertscore", "bleu", "exact_match", "rouge", "meteor"]
# metric_list = ["bertscore"]

ref_data=[]

with open(datasetpath, "r", encoding="utf-8") as infile:
    for idx, line in enumerate(infile):
        data = json.loads(line.strip())
        true_cell_type = data["cell_type"]
        matching_term = [t for t in ontology.terms() if t.name == true_cell_type][0]
        caption=matching_term.definition
        ref_data.append(caption)



# Iterate over all *.jsonl files in output_dir (skip demo files if needed).
for output_file in output_dir.glob("*.jsonl"):

    model_name = output_file.stem  # File name without ".jsonl" suffix.
    
    
    # if not model_name == "qwen3_14B_results":
    #     continue  # Skip demo.jsonl.

    predict_list = []

    # Read prediction file line by line.
    with open(output_file, "r", encoding="utf-8") as pred_file:
        for line in pred_file:
            out_data = json.loads(line.strip())
            response = out_data.get("response", "")

            pattern = r"\[Captioning:\s*([^\]]+?)\]"
            match = re.search(pattern, response)

            if match:
                predicted_caption = match.group(1).strip()
            else:
                predicted_caption = response

            predict_list.append(predicted_caption)

    # Check whether prediction and reference counts match.
    if len(predict_list) != len(ref_data):
        print(f"[Warning] {model_name}: Prediction count ({len(predict_list)}) doesn't match reference count ({len(ref_data)}), skipping.")
        continue
    with open("why_predict_list", "w", encoding="utf-8") as fout:
        json.dump(predict_list, fout, indent=2, ensure_ascii=False)
    with open("why_ref_list", "w", encoding="utf-8") as fout:
        json.dump(ref_data, fout, indent=2, ensure_ascii=False)

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
    output_result_path = current_file.parent / "captioning_new" / f"{model_name}.json"
    with open(output_result_path, "w", encoding="utf-8") as fout:
        json.dump(result_dict, fout, indent=2, ensure_ascii=False)

    print(f"[Saved] {model_name} results → {output_result_path.name}")

    

        
        
        
