# CLAUDE.md

## Project Overview

This is a simulation framework for NLP research studying how LLMs behave as agents in multi-armed bandit tasks. The core research question is how **semantic properties of arm names** (helpful, misleading, neutral framing) affect LLM exploration/exploitation behavior.

## Key Principles

- **Config-driven**: All new experiment variations go in YAML configs under `base_configs/` or `experiments/`. Avoid hardcoding experiment parameters in Python code.
- **Keep it minimal**: This is a research codebase. Prefer small, focused changes. No production-style abstractions, no extra error handling for things that can't happen.
- **Never touch experiment data**: Do not modify, delete, or overwrite files under `experiments/` or `logs/`. These are results from SLURM runs.

## Architecture

```
World (world.py)          — environment: subtasks, state, action dispatch, prompt rendering
  └── Task (tasks/)       — subtask types (bandit, contextual_bandit)
GameDriver (driver.py)    — connects World + Agent, runs the game loop, handles logging
  └── Agent (agents/)     — mono_llm_agent wraps HuggingFace or Gemini backends
ExperimentManager         — runs N shuffles of a config (experiment_manager.py)
EvalManager               — computes metrics from results (eval_manager.py)
```

**LLM backends** (`src/utils/`):
- `HuggingFaceLLM` — standard HF causal LM
- `HFLLM_Thinking_Budget` — Qwen3 with thinking token budget
- `HFLLM_COT` — Llama CoT wrapper
- `GeminiLLM` — Google Gemini API (requires `GEMINI_API_KEY` env var)

**Agent dispatch**: `MonoLLMAgent` selects the backend based on `model_name` and `thinking_budget` in config. To add a new backend, register it in the `__init__` of `MonoLLMAgent`.

**Task/Agent registration**: Both use a `._register("type_name")` decorator pattern. New task types go in `src/tasks/`, new agent types in `src/agents/`.

## Experiment Folder Structure

```
experiments/
└── superfolder/
    └── semantic_variation/      ← one config per variation
        ├── config.yaml
        └── shuffles/
            ├── shuffle_0/
            │   ├── run_YYYYMMDD_HHMMSS.log
            │   └── run_YYYYMMDD_HHMMSS.jsonl
            └── shuffle_N/
```

**Naming conventions** (parsed by `eval_manager.py`):
- Nomenclatures: `alphanumeric`, `sem_rel_helpful`, `sem_rel_mislead`, `sent_helpful`, `sent_mislead`, `ordinal_helpful`, `ordinal_mislead`, `world_helpful`, `world_mislead`
- Scales: `high_scale`, `low_scale`, `low_neg_scale`, `high_neg_scale`

## Config Structure (YAML)

```yaml
subtasks:
  - name: <str>
    id: <int>
    type: bandit         # or contextual_bandit
    params:
      actions:
        - id: <int>
          mean: <float>
          std_dev: <float>

actions:
  - id: <int>
    name: "Field_1"       # or use names_path for random sampling from JSON list
    # names_path: path/to/names.json

agent:
  type: mono_llm
  model_name: "Qwen/Qwen3-8B"
  thinking_budget: 1024   # optional; enables thinking mode
  max_new_tokens: 32768
  temperature: 1.0

world:
  prompt_template_path: src/templates/v1_w_hist.jinja2
  task_prompt: "<str describing task to agent>"
  action_suffix: "Field"  # appended to action names in prompts

experiment:
  time_horizon: 10
  replicates: 10
  scalesweep: false       # special mode for exploration probability measurement
```

## Running Experiments

```bash
# Single run
python experiment_manager.py --config_path path/to/config.yaml

# N shuffles of a semantic variation folder
python experiment_manager.py --config_dir path/to/semantic_variation/ --n_runs 10

# On SLURM (single config)
sbatch run_slurm_job.sh path/to/config.yaml

# On SLURM (folder of configs)
bash run_slurm_on_folder.sh path/to/superfolder/

# Evaluate results
python eval_manager.py --superfolder_path experiments/superfolder/ --min_rows 10
```

## JSONL Output Format

Each step in a run is logged as one line:
```json
{
  "state": {},
  "full_action_content": "<full LLM output>",
  "action_selected_name": "<parsed action>",
  "action_taken_name": "<actual action (may differ if parse failed)>",
  "action_taken_id": <int>,
  "reward": <float>
}
```

## Eval Metrics

`eval_manager.py` computes per-turn and aggregate metrics across shuffles:
- Cumulative reward, step reward
- Regret (vs. optimal arm mean)
- Optimal arm pull rate
- Semantically optimal pull rate (for helpful/mislead conditions)
- Greedy rate (chose highest observed mean so far)
- Exploration count (unique arms tried)

## Active Domains

- `base_configs/farming_basic/` — farm fields, Gaussian rewards
- `base_configs/clothing/` — clothing domain
- `base_configs/abstract_bandit/` — domain-neutral arms
- `base_configs/scales/` — reward scale ablations
