"""Batch-evaluate MAS_RL architecture policies on WorkBench query files.

This mirrors the CLI shape of scripts/inference/generate_multi_agent_results.py,
but evaluates architecture outputs instead of executable function calls.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path

project_root = os.path.abspath(os.path.curdir)
sys.path.append(project_root)

import pandas as pd
import torch

from src.evals.utils import calculate_metrics, DOMAINS as STATE_DOMAINS
from src.multi_agent.multi_agent_runner import _load_api_config
from src.MAS_RL.api_executor import DAGAPIExecutor
from src.MAS_RL.data import QueryRecord, load_query_records
from src.MAS_RL.policy import ArchitecturePolicy
from src.MAS_RL.reward import coverage_metrics, proxy_architecture_reward
from src.MAS_RL.tokenizer_utils import encode_texts, load_tokenizer


ALL_DOMAINS = [
    "calendar",
    "email",
    "analytics",
    "project_management",
    "customer_relationship_manager",
    "multi_domain",
]


def _domain_from_queries_path(queries_path: str) -> str:
    return Path(queries_path).name.replace("_queries_and_answers.csv", "")


def _load_policy(checkpoint_dir: str) -> tuple[ArchitecturePolicy, object, dict]:
    with open(os.path.join(checkpoint_dir, "config.json"), "r", encoding="utf-8") as handle:
        config = json.load(handle)
    tokenizer = load_tokenizer(os.path.join(checkpoint_dir, "tokenizer.json"))
    model = ArchitecturePolicy(
        vocab_size=tokenizer.get_vocab_size(),
        max_agents=config["max_agents"],
        embedding_dim=config["embedding_dim"],
        hidden_dim=config["hidden_dim"],
    )
    model.load_state_dict(
        torch.load(
            os.path.join(checkpoint_dir, "policy.pt"),
            map_location="cpu",
            weights_only=True,
        )
    )
    model.eval()
    return model, tokenizer, config


def _predict_record(
    model: ArchitecturePolicy,
    tokenizer,
    config: dict,
    record: QueryRecord,
    greedy: bool,
) -> dict:
    input_ids, attention_mask = encode_texts(tokenizer, [record.query], config["max_length"])
    ids = torch.tensor(input_ids, dtype=torch.long)
    mask = torch.tensor(attention_mask, dtype=torch.long)

    with torch.no_grad():
        sample = model.sample_one(ids, mask, greedy=greedy)

    arch = sample.architecture
    reward = proxy_architecture_reward(record.query, record.required_domains, arch)
    metrics = coverage_metrics(record.required_domains, arch)
    return {
        "query": record.query,
        "required_domains": record.required_domains,
        "architecture": arch.to_dict(),
        "reward": reward,
        "coverage": metrics["coverage"],
        "extra_domains": metrics["extra_domains"],
        "num_agents": metrics["num_agents"],
        "num_edges": metrics["num_edges"],
        "num_tool_domains": metrics["num_tool_domains"],
        "source_path": record.source_path,
    }


def _architecture_for_record(
    model: ArchitecturePolicy,
    tokenizer,
    config: dict,
    record: QueryRecord,
    greedy: bool,
):
    input_ids, attention_mask = encode_texts(tokenizer, [record.query], config["max_length"])
    ids = torch.tensor(input_ids, dtype=torch.long)
    mask = torch.tensor(attention_mask, dtype=torch.long)

    with torch.no_grad():
        sample = model.sample_one(ids, mask, greedy=greedy)
    return sample.architecture


def _ground_truth_for_metrics(queries_path: str, max_queries: int | None) -> pd.DataFrame:
    ground_truth = pd.read_csv(queries_path)
    if max_queries:
        ground_truth = ground_truth.head(max_queries)
    ground_truth["answer"] = ground_truth["answer"].apply(ast.literal_eval)
    return ground_truth


def generate_architecture_results(
    checkpoint_dir: str,
    queries_path: str,
    output_root: str = "data/mas_rl/results",
    max_queries: int | None = None,
    sample: bool = False,
    verbose: bool = True,
    eval_mode: str = "proxy",
    model_name: str | None = None,
    calculate_workbench_metrics: bool = False,
    print_errors: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """Generate architecture predictions for one WorkBench query CSV."""
    model, tokenizer, config = _load_policy(checkpoint_dir)
    records = load_query_records([queries_path], limit=max_queries)
    if not records:
        raise RuntimeError(f"No query records found at {queries_path}")

    client = None
    available_models = []
    if eval_mode == "api":
        client, available_models = _load_api_config()
        model_name = model_name or (available_models[0] if available_models else None)
        if not model_name:
            raise RuntimeError("API eval requires --model_name or a model name in api.txt")

    rows = []
    greedy = not sample
    for index, record in enumerate(records):
        if eval_mode == "proxy":
            row = _predict_record(model, tokenizer, config, record, greedy=greedy)
        else:
            arch = _architecture_for_record(model, tokenizer, config, record, greedy=greedy)
            proxy_reward = proxy_architecture_reward(record.query, record.required_domains, arch)
            proxy_metrics = coverage_metrics(record.required_domains, arch)
            executor = DAGAPIExecutor(
                client=client,
                model_name=model_name,
            )
            api_result = executor.run(record.query, arch)
            row = {
                "query": record.query,
                "required_domains": record.required_domains,
                "architecture": arch.to_dict(),
                "function_calls": api_result.get("function_calls", []),
                "full_response": api_result.get("full_response", ""),
                "error": api_result.get("error", ""),
                "summary": api_result.get("summary", ""),
                "proxy_reward": proxy_reward,
                "coverage": proxy_metrics["coverage"],
                "extra_domains": proxy_metrics["extra_domains"],
                "num_agents": proxy_metrics["num_agents"],
                "num_edges": proxy_metrics["num_edges"],
                "num_tool_domains": proxy_metrics["num_tool_domains"],
                "source_path": record.source_path,
            }
            for state_domain in STATE_DOMAINS:
                state_domain.reset_state()
        rows.append(row)
        if verbose:
            score = row["reward"] if eval_mode == "proxy" else row["proxy_reward"]
            print(f"[{index + 1}/{len(records)}] score={score:.3f} query={record.query}")

    results = pd.DataFrame(rows)
    domain = _domain_from_queries_path(queries_path)
    checkpoint_name = Path(checkpoint_dir).name
    decode_tag = "sample" if sample else "greedy"
    mode_tag = f"{eval_mode}_{decode_tag}"
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = os.path.join(output_root, domain)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(
        save_dir,
        f"{checkpoint_name}_{mode_tag}_architecture_{timestamp}.csv",
    )
    results.to_csv(save_path, index=False, quoting=csv.QUOTE_ALL)

    actual_metrics = {}
    if eval_mode == "api" and calculate_workbench_metrics:
        ground_truth = _ground_truth_for_metrics(queries_path, max_queries)
        metrics_df = calculate_metrics(ground_truth, results, print_errors=print_errors)
        total = len(metrics_df)
        correct = int(metrics_df["correct"].sum())
        no_side = int((~metrics_df["correct"] & ~metrics_df["unwanted_side_effects"]).sum())
        with_side = int((~metrics_df["correct"] & metrics_df["unwanted_side_effects"]).sum())
        actual_metrics = {
            "accuracy": round(correct / total * 100, 2) if total else 0,
            "errors_no_side_effects": round(no_side / total * 100, 2) if total else 0,
            "errors_with_side_effects": round(with_side / total * 100, 2) if total else 0,
        }

    reward_column = "reward" if eval_mode == "proxy" else "proxy_reward"
    summary = {
        "domain": domain,
        "total_queries": len(results),
        "eval_mode": eval_mode,
        "model_name": model_name or "",
        "mean_reward": round(float(results[reward_column].mean()), 4),
        "mean_coverage": round(float(results["coverage"].mean()), 4),
        "mean_extra_domains": round(float(results["extra_domains"].mean()), 4),
        "mean_num_agents": round(float(results["num_agents"].mean()), 4),
        "mean_num_edges": round(float(results["num_edges"].mean()), 4),
        "save_path": save_path,
    }
    summary.update(actual_metrics)
    print(f"Results saved to: {save_path}")
    print(pd.DataFrame([summary]).to_string(index=False))
    return results, summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch-evaluate a MAS_RL architecture generator on WorkBench queries."
    )
    parser.add_argument(
        "--checkpoint_dir",
        default="data/mas_rl/checkpoints/dag_policy",
        help="Path to a trained MAS_RL checkpoint directory.",
    )
    parser.add_argument(
        "--queries_path",
        default=None,
        help="Path to a single WorkBench queries CSV.",
    )
    parser.add_argument(
        "--all_domains",
        action="store_true",
        default=False,
        help="Run on all WorkBench domain query files.",
    )
    parser.add_argument(
        "--max_queries",
        type=int,
        default=None,
        help="Limit to first N queries per domain.",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        default=False,
        help="Sample architectures instead of greedy decoding.",
    )
    parser.add_argument(
        "--output_root",
        default="data/mas_rl/results",
        help="Root directory for saved architecture evaluation CSVs.",
    )
    parser.add_argument(
        "--eval_mode",
        choices=["proxy", "api"],
        default="proxy",
        help="proxy scores architecture locally; api executes generated MAS with frozen LLM calls.",
    )
    parser.add_argument(
        "--model_name",
        default=None,
        help="OpenAI-compatible model name for --eval_mode api. Defaults to first model in api.txt.",
    )
    parser.add_argument(
        "--calculate_metrics",
        action="store_true",
        default=False,
        help="In API mode, compute WorkBench correctness and side-effect metrics.",
    )
    parser.add_argument(
        "--print_errors",
        action="store_true",
        default=False,
        help="Pass print_errors=True to WorkBench calculate_metrics.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        default=False,
        help="Suppress per-query logging.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.all_domains:
        summaries = []
        summary_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        for domain in ALL_DOMAINS:
            queries_path = os.path.join(
                "data",
                "processed",
                "queries_and_answers",
                f"{domain}_queries_and_answers.csv",
            )
            if not os.path.exists(queries_path):
                print(f"Skipping {domain}: file not found at {queries_path}")
                continue
            print(f"\n{'#' * 60}")
            print(f"# Running MAS_RL architecture eval on: {domain}")
            print(f"{'#' * 60}")
            _, summary = generate_architecture_results(
                checkpoint_dir=args.checkpoint_dir,
                queries_path=queries_path,
                output_root=args.output_root,
                max_queries=args.max_queries,
                sample=args.sample,
                verbose=not args.quiet,
                eval_mode=args.eval_mode,
                model_name=args.model_name,
                calculate_workbench_metrics=args.calculate_metrics,
                print_errors=args.print_errors,
            )
            summaries.append(summary)

        if summaries:
            checkpoint_name = Path(args.checkpoint_dir).name
            mode_tag = "sample" if args.sample else "greedy"
            summary_dir = os.path.join(args.output_root, "_summary")
            os.makedirs(summary_dir, exist_ok=True)
            summary_path = os.path.join(
                summary_dir,
                f"{checkpoint_name}_{mode_tag}_architecture_summary_{summary_timestamp}.csv",
            )
            summary_df = pd.DataFrame(summaries)
            summary_df.to_csv(summary_path, index=False, quoting=csv.QUOTE_ALL)
            print(f"\nSummary saved to: {summary_path}")
            print(summary_df.to_string(index=False))
        return

    if args.queries_path:
        generate_architecture_results(
            checkpoint_dir=args.checkpoint_dir,
            queries_path=args.queries_path,
            output_root=args.output_root,
            max_queries=args.max_queries,
            sample=args.sample,
            verbose=not args.quiet,
            eval_mode=args.eval_mode,
            model_name=args.model_name,
            calculate_workbench_metrics=args.calculate_metrics,
            print_errors=args.print_errors,
        )
        return

    print("Please specify either --queries_path or --all_domains.")
    print("Example:")
    print("  python scripts/inference/generate_mas_rl_architecture_results.py \\")
    print("      --checkpoint_dir data/mas_rl/checkpoints/dag_policy \\")
    print("      --queries_path data/processed/queries_and_answers/calendar_queries_and_answers.csv")
    sys.exit(1)


if __name__ == "__main__":
    main()
