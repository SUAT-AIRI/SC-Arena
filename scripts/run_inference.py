import argparse, json, yaml, pathlib, signal, sys, logging
from typing import Iterator, List, Dict, Any
from registry import PROVIDERS, Evaluators
from tqdm import tqdm   # pip install tqdm
import providers
import evaluators
import os

# ---------- Load & validate config ----------
def load_config(path: pathlib.Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) if path.suffix in {".yaml", ".yml"} else json.load(f)
    assert {"provider", "init_kwargs"} <= cfg.keys(), "config is missing required fields"
    cfg.setdefault("gen_kwargs", {})
    return cfg

# ---------- Data iterator: supports directories & batching ----------
def iter_jsonl(path: pathlib.Path, batch_size: int = 32) -> Iterator[List[str]]:
    files = [path] if path.is_file() else sorted(path.glob("**/*.jsonl"))
    buf = []
    for fp in files:
        with fp.open(encoding="utf-8") as f:
            for line in f:
                buf.append(json.loads(line)["prompt"])
                if len(buf) >= batch_size:
                    yield buf; buf = []
    if buf:
        yield buf

# ---------- Main pipeline ----------
def run(cfg, data_path: pathlib.Path, task_name:str, out_path: pathlib.Path, score_path: pathlib.Path, batch_size: int, baseurl: str, apikey: str , modelname: str,evaluated_model:str, direct_scoring:str):
    provider_cls = PROVIDERS[cfg["provider"]]
    evaluator_cls = Evaluators[task_name]
    engine = provider_cls(**cfg["init_kwargs"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    evaluator = evaluator_cls(data_path,baseurl,apikey,modelname,evaluated_model)


    # Graceful Ctrl-C shutdown
    def _sigint_handler(sig, frame):
        logging.warning("KeyboardInterrupt! Releasing resources...")
        engine.shutdown(); sys.exit(130)
    signal.signal(signal.SIGINT, _sigint_handler)
    prompts = evaluator.init_data()

    if cfg["provider"].endswith("_fm"):
        if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
            with out_path.open("w", encoding="utf-8") as fou:

                responses = engine.infer(prompts, **cfg["gen_kwargs"])
                fou.write(json.dumps({"response": responses}, ensure_ascii=False) + "\n")
                results = evaluator.evaluate(responses)

                # Print evaluation results
                accuracy = results.get("accuracy", 0.0)
                correct = results.get("correct", 0)
                total = results.get("total", 0)
                print(f"Accuracy: {accuracy:.4f} ({correct}/{total})")

                # Write score to score_path
                score_data = {
                    "task": task_name,
                    "accuracy": accuracy,
                    "correct": correct,
                    "total": total
                }

                # Ensure parent directory exists
                score_path.parent.mkdir(parents=True, exist_ok=True)

                # Write score file
                with open(score_path, "w", encoding="utf-8") as f_score:
                    json.dump(score_data, f_score, indent=2, ensure_ascii=False)
        else:
            with out_path.open("r", encoding="utf-8") as fou:
                data = json.load(fou)
                responses=data["response"]
                results = evaluator.evaluate(responses)

                # Print evaluation results
                accuracy = results.get("accuracy", 0.0)
                correct = results.get("correct", 0)
                total = results.get("total", 0)
                print(f"Accuracy: {accuracy:.4f} ({correct}/{total})")

                # Write score to score_path
                score_data = {
                    "task": task_name,
                    "accuracy": accuracy,
                    "correct": correct,
                    "total": total
                }

                # Ensure parent directory exists
                score_path.parent.mkdir(parents=True, exist_ok=True)

                # Write score file
                with open(score_path, "w", encoding="utf-8") as f_score:
                    json.dump(score_data, f_score, indent=2, ensure_ascii=False)










    else:
        if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
            with out_path.open("w", encoding="utf-8") as fou:
                # Gather all prompts (i.e., the evaluated data)
                all_responses = []
                all_prompts = []

                # Process prompts in batches
                for i in range(0, len(prompts), batch_size):
                    batch = prompts[i:i + batch_size]
                    try:
                        responses = engine.infer(batch, **cfg["gen_kwargs"])
                    except Exception as e:
                        logging.exception("Inference failed; skipped this batch")
                        continue

                    # Collect outputs
                    all_prompts.extend(batch)
                    all_responses.extend(responses)

                    # Write to output file
                    for prompt, resp in zip(batch, responses):
                        fou.write(json.dumps({"prompt": prompt, "response": resp}, ensure_ascii=False) + "\n")
                    fou.flush()

                # Evaluate after all inference is done
                print(len(all_responses))

                results = evaluator.evaluate(all_responses)

                # Print evaluation results
                accuracy = results.get("accuracy", 0.0)
                correct = results.get("correct", 0)
                total = results.get("total", 0)
                print(f"Accuracy: {accuracy:.4f} ({correct}/{total})")

                # Write score to score_path
                score_data = {
                    "task": task_name,
                    "accuracy": accuracy,
                    "correct": correct,
                    "total": total
                }

                # Ensure parent directory exists
                score_path.parent.mkdir(parents=True, exist_ok=True)

                # Write score file
                with open(score_path, "w", encoding="utf-8") as f_score:
                    json.dump(score_data, f_score, indent=2, ensure_ascii=False)
        else:
            existing_outputs = []
            with out_path.open("r", encoding="utf-8") as fou:
                for line in fou:
                    line = line.strip()
                    if line:
                        existing_outputs.append(json.loads(line))

            prompt2response = {item["prompt"]: item["response"] for item in existing_outputs}
            missing_prompts = [p for p in prompts if p not in prompt2response]
            with open("prompt2response.json", "w", encoding="utf-8") as f_score:
                json.dump(prompt2response, f_score, indent=2, ensure_ascii=False)
            with open("missing_prompts.json", "w", encoding="utf-8") as f_score:
                json.dump(missing_prompts, f_score, indent=2, ensure_ascii=False)

            print(len(prompt2response),len(missing_prompts))

            # Process prompts in batches
            for i in range(0, len(missing_prompts), batch_size):
                batch = missing_prompts[i:i + batch_size]
                try:
                    responses = engine.infer(batch, **cfg["gen_kwargs"])
                except Exception as e:
                    logging.exception("Inference failed; skipped this batch")
                    continue

                # Update prompt2response
                for p, r in zip(batch, responses):
                    prompt2response[p] = r

                # Rewrite the entire file immediately to preserve prompt order
                final_outputs = [{"prompt": p, "response": prompt2response[p]} for p in prompts if p in prompt2response]

                with out_path.open("w", encoding="utf-8") as fout:
                    for item in final_outputs:
                        fout.write(json.dumps(item, ensure_ascii=False) + "\n")

            final_outputs = [{"prompt": p, "response": prompt2response[p]} for p in prompts if p in prompt2response]

            results = evaluator.evaluate([item["response"] for item in final_outputs])
            accuracy = results.get("accuracy", 0.0)
            correct = results.get("correct", 0)
            total = results.get("total", 0)
            print(f"Accuracy: {accuracy:.4f} ({correct}/{total})")

            # Write score to score_path
            score_data = {
                "task": task_name,
                "accuracy": accuracy,
                "correct": correct,
                "total": total
            }

            # Ensure parent directory exists
            score_path.parent.mkdir(parents=True, exist_ok=True)

            # Write score file
            with open(score_path, "w", encoding="utf-8") as f_score:
                json.dump(score_data, f_score, indent=2, ensure_ascii=False)


    engine.shutdown()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data",   required=True)
    parser.add_argument("--task",   required=True)
    parser.add_argument("--out",    default="outputs/out.jsonl")
    parser.add_argument("--score",    default="scores/out.jsonl")
    parser.add_argument("--direct_scoring",    default="false")
    parser.add_argument("--baseurl",    required=True)
    parser.add_argument("--apikey",    required=True)
    parser.add_argument("--modelname",    required=True)
    parser.add_argument("--evaluated_model",    required=True)
    parser.add_argument("--batch_size", type=int, default=10)

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    cfg = load_config(pathlib.Path(args.config))
    run(cfg, pathlib.Path(args.data), args.task,pathlib.Path(args.out), pathlib.Path(args.score),args.batch_size,args.baseurl,args.apikey,args.modelname,args.evaluated_model,args.direct_scoring)

if __name__ == "__main__":
    main()
