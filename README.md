# SC-Arena: A Natural Language Benchmark for Single-Cell Reasoning with Knowledge-Augmented Evaluation

[![arXiv](https://img.shields.io/badge/arXiv-2602.23199-b31b1b.svg?style=for-the-badge)](https://arxiv.org/abs/2602.23199)

## News

- **[February 2026]:** 🎉*SC-Arena* has been accepted to **ICLR 2026**!
- **[February 2026]:** Preprint available on [arXiv](https://arxiv.org/abs/2602.23199).

---

SC-Arena is a task-oriented inference and evaluation framework for single-cell related benchmarks.
It provides a unified runtime for model inference (via pluggable providers) and task evaluation
(via pluggable evaluators), driven by YAML configs and prompt templates.

## What This Project Does

1. Load a dataset and convert each sample into task-specific prompts.
2. Run batched inference through a selected provider (for example `openai`, `vllm`, `vllm_api`).
3. Evaluate model outputs with task evaluators.
4. Save prediction outputs and final scores to JSON files.

## Repository Layout

```text
SC-Arena/
|-- base.py                       # Abstract base classes: InferenceEngine / EvaluateEngine
|-- registry.py                   # Provider and evaluator registries
|-- providers/                    # Inference backends
|   |-- openai_provider.py
|   |-- qwen3_provider.py
|   |-- vllm_provider.py
|   `-- vllm_api_provider.py
|-- evaluators/                   # Task evaluators
|   |-- cell_type_annotation.py
|   |-- perturbation_prediction.py
|   |-- captioning.py
|   |-- generation.py
|   `-- scienceqa.py
|-- prompts/                      # Prompt templates (.jsonl)
|-- data/                         # Example datasets
|-- configs/                      # Provider configs
|-- scripts/run_inference.py      # Main entry point
`-- requirements.txt
```

## Supported Tasks

| Task (`--task`) | Evaluator | Expected Answer Pattern |
|---|---|---|
| `celltype` | `CellTypeEvaluator` | `[Predicted_Cell_Type: ...]` |
| `captioning` | `CaptioningEvaluator` | `[Captioning: ...]` |
| `generation` | `GenerationEvaluator` | `[Cell_Sentence: ...]` |
| `perturbation` | `PerturbationEvaluator` | `[Up: ...] [Down: ...] [Cell_Sentence: ...]` |
| `scienceqa` | `ScienceqaEvaluator` | `[Answer: ...]` |

## Installation

```bash
git clone https://github.com/SUAT-AIRI/SC-ARENA.git
cd SC-ARENA

python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Linux/macOS:
# source .venv/bin/activate

pip install -r requirements.txt
```

Notes:
- The dependency set is large and includes GPU-related packages.
- If you only use API-based providers, you may trim dependencies for your environment.

## Configuration

Use files in `configs/` as templates:

- `configs/openai_exmaple.yaml`
- `configs/vllm_example.yaml`
- `configs/vllm_api.yaml`

Minimum schema:

```yaml
provider: openai
init_kwargs:
  model_name: "gpt-4o-mini"
  api_key: "${OPENAI_API_KEY}"

gen_kwargs:
  temperature: 0.7
  max_tokens: 1024
```

## Run Inference

Main command:

```bash
python -m scripts.run_inference \
  --config configs/openai_exmaple.yaml \
  --data data/cell_sentences_fixed.jsonl \
  --task celltype \
  --out outputs/celltype/openai_celltype.jsonl \
  --score scores/celltype/openai_celltype.json \
  --baseurl https://api.openai.com/v1 \
  --apikey YOUR_API_KEY \
  --modelname gpt-4o-mini \
  --evaluated_model openai_celltype
```

Task examples:

```bash
# 1) Cell type annotation
python -m scripts.run_inference --config configs/openai_exmaple.yaml --data data/cell_sentences_fixed.jsonl --task celltype --out outputs/celltype/result.jsonl --score scores/celltype/result.json --baseurl https://api.openai.com/v1 --apikey YOUR_API_KEY --modelname gpt-4o-mini --evaluated_model model_a

# 2) Captioning
python -m scripts.run_inference --config configs/openai_exmaple.yaml --data data/cell_sentences_fixed.jsonl --task captioning --out outputs/captioning/result.jsonl --score scores/captioning/result.json --baseurl https://api.openai.com/v1 --apikey YOUR_API_KEY --modelname gpt-4o-mini --evaluated_model model_a

# 3) Generation
python -m scripts.run_inference --config configs/openai_exmaple.yaml --data data/cell_sentences_fixed.jsonl --task generation --out outputs/generation/result.jsonl --score scores/generation/result.json --baseurl https://api.openai.com/v1 --apikey YOUR_API_KEY --modelname gpt-4o-mini --evaluated_model model_a

# 4) Perturbation
python -m scripts.run_inference --config configs/openai_exmaple.yaml --data data/test_perturbation.json --task perturbation --out outputs/perturbation/result.jsonl --score scores/perturbation/result.json --baseurl https://api.openai.com/v1 --apikey YOUR_API_KEY --modelname gpt-4o-mini --evaluated_model model_a

# 5) ScienceQA
python -m scripts.run_inference --config configs/openai_exmaple.yaml --data data/ScientificQA_final.json --task scienceqa --out outputs/scienceqa/result.jsonl --score scores/scienceqa/result.json --baseurl https://api.openai.com/v1 --apikey YOUR_API_KEY --modelname gpt-4o-mini --evaluated_model model_a
```

## Outputs

- `--out`: model prediction file (JSONL)
- `--score`: aggregated score summary (JSON), for example:

```json
{
  "task": "celltype",
  "accuracy": 0.82,
  "correct": 82,
  "total": 100
}
```

## Common Pitfalls

- `celltype` currently reads prompt templates from `prompts/test_prompt.jsonl` in code.
  If this file is missing, copy `prompts/cell_type_annotation.jsonl` to that name.
- `--baseurl`, `--apikey`, and `--modelname` are required CLI args.
- Ensure output directories are writable.

## Extending the Framework

### Add a provider

1. Create a class inheriting `InferenceEngine` in `providers/`.
2. Register it with `@register("your_provider")`.
3. Add a config file in `configs/`.

### Add an evaluator

1. Create a class inheriting `EvaluateEngine` in `evaluators/`.
2. Register it with `@register_evaluator("your_task")`.
3. Add task prompts under `prompts/`.

## License

MIT
