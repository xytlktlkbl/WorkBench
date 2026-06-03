"""
Multi-Agent Runner — bridge between the multi-agent framework and WorkBench eval.

This module provides a drop-in replacement for the original
src.evals.utils.generate_results() function. It:
  1. Creates the Orchestrator + Workers
  2. For each query, runs the multi-agent system
  3. Collects function calls and formats them for evaluation
  4. Saves results as CSV (same format as the original)

Usage:
    from src.multi_agent.multi_agent_runner import generate_multi_agent_results
    results = generate_multi_agent_results(
        queries_path="data/processed/queries_and_answers/calendar_queries_and_answers.csv",
        model_name="gpt-4",
    )
"""

import csv
import os
import sys
import traceback
from datetime import datetime
from typing import Optional

import pandas as pd
from openai import OpenAI

from src.evals.utils import DOMAINS, execute_actions_and_reset_state
from src.multi_agent.orchestrator import Orchestrator
from src.multi_agent.blackboard import Blackboard
from src.multi_agent.workers import get_worker_for_domain

def _load_api_config(config_path: str = "api.txt") -> tuple[OpenAI, list[str]]:
    """
    Load API configuration from a text file.

    File format (one value per line, blank lines ignored):
        line 1: API key
        line 2: base_url (optional — omit for vanilla OpenAI)
        line 3+: available model names (optional — defaults to GPT models)

    Returns (OpenAI client, list of available model names).
    """
    repo_root = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    paths_to_try = [
        os.path.join(repo_root, config_path),
        config_path,
    ]
    found_path = None
    for p in paths_to_try:
        if os.path.exists(p):
            found_path = p
            break

    if found_path is None:
        raise FileNotFoundError(
            f"{config_path} not found. Create it with:\n"
            f"  line 1: your API key\n"
            f"  line 2: base_url (optional)\n"
            f"  line 3+: model names (optional)\n"
        )

    with open(found_path, "r") as f:
        lines = [l.strip() for l in f if l.strip()]

    if not lines:
        raise ValueError(f"{config_path} is empty — at least an API key is required.")

    api_key = lines[0]
    base_url = lines[1] if len(lines) > 1 else None
    models = lines[2:] if len(lines) > 2 else [
        "gpt-4-0125-preview",
        "gpt-4",
        "gpt-3.5-turbo",
        "gpt-4o",
        "gpt-4o-mini",
    ]

    kwargs: dict = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url

    return OpenAI(**kwargs), models


def _create_openai_client() -> OpenAI:
    """Backward-compatible wrapper — creates client only, discards models."""
    client, _ = _load_api_config()
    return client


try:
    _, AVAILABLE_MODELS = _load_api_config()
except Exception:
    AVAILABLE_MODELS = [
        "gpt-4-0125-preview",
        "gpt-4",
        "gpt-3.5-turbo",
        "gpt-4o",
        "gpt-4o-mini",
    ]


def generate_multi_agent_results(
    queries_path: str,
    model_name: str = "gpt-4-0125-preview",
    tool_selection: str = "all",
    client: Optional[OpenAI] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Generate results using the multi-agent system for all queries in a CSV file.

    Parameters
    ----------
    queries_path : str
        Path to the queries CSV file (e.g., data/processed/queries_and_answers/...csv).
    model_name : str
        OpenAI model to use for both orchestrator and workers.
    tool_selection : str
        "all" or "domains" — passed through for filename compatibility.
    client : OpenAI, optional
        Pre-configured OpenAI client.
    verbose : bool
        Print progress for each query.

    Returns
    -------
    pd.DataFrame with columns: query, function_calls, full_response, error
    """
    if client is None:
        client = _create_openai_client()

    # Set up the orchestrator
    orchestrator = Orchestrator(model=model_name, max_iterations=10)
    orchestrator.set_client(client)

    # Read queries
    queries_df = pd.read_csv(queries_path)
    queries = queries_df["query"].tolist()

    results = pd.DataFrame(columns=["query", "function_calls", "full_response", "error"])

    for i, query in enumerate(queries):
        if verbose:
            print(f"\n{'='*60}")
            print(f"[{i+1}/{len(queries)}] Query: {query}")
            print(f"{'='*60}")

        error = ""
        function_calls = []
        full_response = ""

        try:
            orch_result = orchestrator.run(query)

            function_calls = orch_result.get("function_calls", [])
            full_response = orch_result.get("full_response", "")
            summary = orch_result.get("summary", "")

            if verbose:
                print(f"Actions ({len(function_calls)}):")
                for fc in function_calls:
                    print(f"  → {fc}")
                print(f"Summary: {summary}")

        except Exception as e:
            context_window_keywords = [
                "maximum input length",
                "maximum context length",
                "prompt is too long",
                "Request too large",
                "context_length_exceeded",
            ]
            msg = str(e)
            if any(kw in msg for kw in context_window_keywords):
                error = "Context window exceeded"
                if verbose:
                    print(f"!!! Context window exceeded")
            else:
                error = f"{type(e).__name__}: {msg}"
                if verbose:
                    print(f"!!! Error: {error}")
            traceback.print_exc()

        # Collect results
        results = pd.concat(
            [
                results,
                pd.DataFrame(
                    [[query, function_calls, full_response, error]],
                    columns=["query", "function_calls", "full_response", "error"],
                ),
            ],
            ignore_index=True,
        )

        # Reset all domain state after each query (critical for correctness!)
        for domain in DOMAINS:
            domain.reset_state()

    # Save results
    domain = queries_path.split("/")[-1].split(".")[0].replace("_queries_and_answers", "")
    save_dir = os.path.join("data", "results", domain)
    os.makedirs(save_dir, exist_ok=True)

    current_datetime = str(pd.Timestamp.now()).split(".")[0].replace(" ", "_").replace(":", "-")
    save_path = os.path.join(
        save_dir,
        f"{model_name}_multi-agent_{tool_selection}_{current_datetime}.csv",
    )
    results.to_csv(save_path, index=False, quoting=csv.QUOTE_ALL)
    print(f"\nResults saved to: {save_path}")

    return results


def run_single_query(
    query: str,
    model_name: str = "gpt-4-0125-preview",
    client: Optional[OpenAI] = None,
    verbose: bool = True,
) -> dict:
    """
    Run the multi-agent system on a single query (for debugging/testing).

    Parameters
    ----------
    query : str
        Natural language task.
    model_name : str
        OpenAI model name.
    client : OpenAI, optional
        Pre-configured client.
    verbose : bool
        Print detailed trace.

    Returns
    -------
    dict with function_calls, full_response, error, summary
    """
    if client is None:
        client = _create_openai_client()

    orchestrator = Orchestrator(model=model_name, max_iterations=10)
    orchestrator.set_client(client)

    result = orchestrator.run(query)

    if verbose:
        print(f"Query: {query}")
        print(f"Function calls: {result['function_calls']}")
        print(f"Summary: {result['summary']}")
        if result["error"]:
            print(f"Error: {result['error']}")

    # Reset state
    for domain in DOMAINS:
        domain.reset_state()

    return result
