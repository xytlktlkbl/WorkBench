"""
Run multi-agent inference on WorkBench queries.

Usage:
    # Single domain
    python scripts/inference/generate_multi_agent_results.py \\
        --model_name gpt-4-0125-preview \\
        --queries_path data/processed/queries_and_answers/calendar_queries_and_answers.csv

    # Multi-domain
    python scripts/inference/generate_multi_agent_results.py \\
        --model_name gpt-4-0125-preview \\
        --queries_path data/processed/queries_and_answers/multi_domain_queries_and_answers.csv

    # All domains (batch)
    python scripts/inference/generate_multi_agent_results.py \\
        --model_name gpt-4-0125-preview \\
        --all_domains

    # Quick test with a few queries
    python scripts/inference/generate_multi_agent_results.py \\
        --model_name gpt-4-0125-preview \\
        --queries_path data/processed/queries_and_answers/calendar_queries_and_answers.csv \\
        --max_queries 5
"""

import argparse
import ast
import os
import sys
import warnings

project_root = os.path.abspath(os.path.curdir)
sys.path.append(project_root)

import pandas as pd

from src.evals.utils import calculate_metrics, DOMAINS
from src.multi_agent.multi_agent_runner import generate_multi_agent_results

warnings.filterwarnings("ignore")

ALL_DOMAINS = [
    "calendar",
    "email",
    "analytics",
    "project_management",
    "customer_relationship_manager",
    "multi_domain",
]


def main():
    parser = argparse.ArgumentParser(
        description="Run multi-agent inference on WorkBench queries."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt-4-0125-preview",
        help="OpenAI model name (default: gpt-4-0125-preview)",
    )
    parser.add_argument(
        "--queries_path",
        type=str,
        default=None,
        help="Path to a single queries CSV file.",
    )
    parser.add_argument(
        "--all_domains",
        action="store_true",
        default=False,
        help="Run on all 6 domains sequentially.",
    )
    parser.add_argument(
        "--max_queries",
        type=int,
        default=None,
        help="Limit to first N queries (for quick testing).",
    )
    parser.add_argument(
        "--tool_selection",
        type=str,
        default="all",
        choices=["all", "domains"],
        help="Tool selection mode (default: all).",
    )

    args = parser.parse_args()

    if args.all_domains:
        # Run on all domains
        all_results = []
        for domain in ALL_DOMAINS:
            queries_path = os.path.join(
                "data", "processed", "queries_and_answers",
                f"{domain}_queries_and_answers.csv",
            )
            if not os.path.exists(queries_path):
                print(f"Skipping {domain}: file not found at {queries_path}")
                continue

            print(f"\n{'#'*60}")
            print(f"# Running multi-agent on: {domain}")
            print(f"{'#'*60}")

            results = generate_multi_agent_results(
                queries_path=queries_path,
                model_name=args.model_name,
                tool_selection=args.tool_selection,
                verbose=True,
            )

            # Calculate metrics for this domain
            ground_truth = pd.read_csv(queries_path)
            ground_truth["answer"] = ground_truth["answer"].apply(ast.literal_eval)
            print(f"\n--- Metrics for {domain} ---")
            calculate_metrics(ground_truth, results, print_errors=False)
            all_results.append(results)

        if all_results:
            combined = pd.concat(all_results, ignore_index=True)
            print(f"\nTotal queries processed: {len(combined)}")
            print(f"Results saved in data/results/<domain>/ directories.")

    elif args.queries_path:
        # Run on a single domain file
        if args.max_queries:
            # Load and truncate the CSV for quick testing
            original = pd.read_csv(args.queries_path)
            truncated = original.head(args.max_queries)
            temp_path = args.queries_path.replace(".csv", "_temp_truncated.csv")
            truncated.to_csv(temp_path, index=False)
            actual_path = temp_path
            print(f"Testing with first {args.max_queries} queries from {args.queries_path}")
        else:
            actual_path = args.queries_path

        results = generate_multi_agent_results(
            queries_path=actual_path,
            model_name=args.model_name,
            tool_selection=args.tool_selection,
            verbose=True,
        )

        # Calculate metrics
        ground_truth = pd.read_csv(actual_path)
        ground_truth["answer"] = ground_truth["answer"].apply(ast.literal_eval)
        print(f"\n--- Metrics ---")
        calculate_metrics(ground_truth, results, print_errors=True)

        # Clean up temp file
        if args.max_queries and os.path.exists(temp_path):
            os.remove(temp_path)

    else:
        print("Please specify either --queries_path or --all_domains.")
        print("Example:")
        print("  python scripts/inference/generate_multi_agent_results.py \\")
        print("      --model_name gpt-4-0125-preview \\")
        print("      --queries_path data/processed/queries_and_answers/calendar_queries_and_answers.csv")
        sys.exit(1)


if __name__ == "__main__":
    main()
