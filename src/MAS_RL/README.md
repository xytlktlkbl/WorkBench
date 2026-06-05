# MAS_RL

Local RL components for generating a command-DAG multi-agent architecture.

The generator maps a text task to:

- `num_agents`
- a directed acyclic command graph where `a -> b` means agent `a` can command agent `b`
- a domain-level tool scope for each agent

The frozen LLM worker system is not trained here. This module trains only the
local architecture policy.

## Train

```powershell
D:\files\WorkBench\.venv\Scripts\python.exe src\MAS_RL\train.py --limit 64 --grpo_epochs 1 --group_size 4
```

For all WorkBench domains, `--limit` is treated as max queries per domain:

```powershell
D:\files\WorkBench\.venv\Scripts\python.exe src\MAS_RL\train.py `
  --all_domains `
  --limit 10 `
  --grpo_epochs 1 `
  --group_size 4
```

Equivalent shorthand:

```powershell
D:\files\WorkBench\.venv\Scripts\python.exe src\MAS_RL\train.py \
  --data_paths all_domains \
  --limit 10 \
  --grpo_epochs 1 \
  --group_size 8
```

By default SFT is disabled (`--sft_epochs 0`) and GRPO uses the fast local proxy
reward. To train by exploring architectures with real API rollouts:

```powershell
D:\files\WorkBench\.venv\Scripts\python.exe src\MAS_RL\train.py --data_paths all_domains --limit 5 --sft_epochs 0 --grpo_epochs 1 --group_size 2 --reward_mode api --model_name deepseek-v4-flash --output_dir data\mas_rl\checkpoints\api_grpo
```

## Eval

```powershell
D:\files\WorkBench\.venv\Scripts\python.exe src\MAS_RL\eval.py `
  --limit 64 `
  --output_jsonl data\mas_rl\eval\predictions.jsonl
```

## Single Text

```powershell
D:\files\WorkBench\.venv\Scripts\python.exe src\MAS_RL\eval.py `
  --text "If we have not contacted Cameron in a fortnight, schedule a meeting tomorrow" `
  --greedy
```

## Batch Eval

This mirrors the WorkBench inference scripts and saves per-domain CSVs plus an
optional all-domain summary.

```powershell
D:\files\WorkBench\.venv\Scripts\python.exe scripts\inference\generate_mas_rl_architecture_results.py `
  --checkpoint_dir data\mas_rl\checkpoints\dag_policy `
  --queries_path data\processed\queries_and_answers\calendar_queries_and_answers.csv `
  --max_queries 10
```

```powershell
D:\files\WorkBench\.venv\Scripts\python.exe scripts\inference\generate_mas_rl_architecture_results.py --checkpoint_dir data\mas_rl\checkpoints\dag_policy --all_domains --max_queries 10
```

By default, batch eval uses the local proxy reward and does not call the API. To
execute the generated architecture with frozen API-backed agents and calculate
WorkBench metrics:

```powershell
D:\files\WorkBench\.venv\Scripts\python.exe scripts\inference\generate_mas_rl_architecture_results.py --checkpoint_dir data\mas_rl\checkpoints\api_grpo --all_domains --eval_mode api --model_name deepseek-v4-flash --calculate_metrics
```
