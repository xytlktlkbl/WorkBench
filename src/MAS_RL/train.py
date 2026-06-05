"""Train the local MAS architecture generator with SFT warmup + GRPO."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

import torch
from tqdm.auto import tqdm

from src.evals.utils import DOMAINS as STATE_DOMAINS
from src.evals.utils import has_side_effects, is_correct
from src.multi_agent.multi_agent_runner import _load_api_config
from src.MAS_RL.api_executor import DAGAPIExecutor
from src.MAS_RL.data import DEFAULT_QUERY_PATHS, QueryRecord, load_query_records
from src.MAS_RL.policy import ArchitecturePolicy
from src.MAS_RL.reward import proxy_architecture_reward
from src.MAS_RL.schema import DOMAINS
from src.MAS_RL.tokenizer_utils import encode_texts, save_tokenizer, train_word_tokenizer


def _make_sft_targets(records: list[QueryRecord], max_agents: int) -> dict[str, torch.Tensor]:
    num_domains = len(DOMAINS)
    target_num_agents = []
    target_parents = []
    target_extra_edges = []
    target_tools = []

    for record in records:
        required = [domain for domain in record.required_domains if domain in DOMAINS]
        if len(required) <= 1:
            n_agents = 1
        else:
            n_agents = min(max_agents, len(required) + 1)

        target_num_agents.append(n_agents - 1)

        parents = [-1 for _ in range(max_agents)]
        for child_idx in range(1, n_agents):
            parents[child_idx] = 0
        target_parents.append(parents)

        extra = torch.zeros(max_agents, max_agents)
        target_extra_edges.append(extra)

        tools = torch.zeros(max_agents, num_domains)
        if n_agents == 1:
            for domain in required:
                tools[0, DOMAINS.index(domain)] = 1.0
        else:
            for idx, domain in enumerate(required):
                agent_idx = min(idx + 1, n_agents - 1)
                tools[agent_idx, DOMAINS.index(domain)] = 1.0
        target_tools.append(tools)

    return {
        "target_num_agents": torch.tensor(target_num_agents, dtype=torch.long),
        "target_parents": torch.tensor(target_parents, dtype=torch.long),
        "target_extra_edges": torch.stack(target_extra_edges).float(),
        "target_tools": torch.stack(target_tools).float(),
    }


def _batch_indices(size: int, batch_size: int) -> list[list[int]]:
    indices = list(range(size))
    random.shuffle(indices)
    return [indices[start : start + batch_size] for start in range(0, size, batch_size)]


def _save_checkpoint(
    output_dir: str,
    model: ArchitecturePolicy,
    tokenizer,
    config: dict,
    metrics: dict,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(output_dir, "policy.pt"))
    save_tokenizer(tokenizer, os.path.join(output_dir, "tokenizer.json"))
    with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
    with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)


def _reset_workbench_state() -> None:
    for domain in STATE_DOMAINS:
        domain.reset_state()


def _api_rollout_reward(
    record: QueryRecord,
    architecture,
    executor: DAGAPIExecutor,
) -> float:
    result = executor.run(record.query, architecture)
    function_calls = result.get("function_calls", [])
    error = result.get("error", "")
    correct = is_correct(function_calls, record.answer, error)
    side_effects = has_side_effects(function_calls, record.answer)

    reward = 3.0 if correct else -1.0
    if side_effects:
        reward -= 2.0
    if error:
        reward -= 1.0
    if not function_calls and record.answer:
        reward -= 0.5

    reward -= 0.03 * max(0, architecture.num_agents - 1)
    reward -= 0.02 * len(architecture.active_edges())
    reward -= 0.01 * sum(len(domains) for domains in architecture.tool_domains)
    _reset_workbench_state()
    return float(reward)


def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    data_paths = args.data_paths
    all_domains_requested = args.all_domains or (data_paths == ["all_domains"])
    paths = DEFAULT_QUERY_PATHS if all_domains_requested else (data_paths or DEFAULT_QUERY_PATHS)
    if all_domains_requested:
        per_path_limit = args.max_queries_per_domain or args.limit
        global_limit = None
    else:
        per_path_limit = args.max_queries_per_domain
        global_limit = args.limit
    records = load_query_records(paths, limit=global_limit, limit_per_path=per_path_limit)
    if not records:
        raise RuntimeError("No query records found.")
    total_optimizer_steps = args.grpo_epochs * len(records)
    total_rollouts = total_optimizer_steps * args.group_size
    print(
        "Training setup: "
        f"records={len(records)} "
        f"grpo_epochs={args.grpo_epochs} "
        f"group_size={args.group_size} "
        f"optimizer_steps={total_optimizer_steps} "
        f"rollouts={total_rollouts} "
        f"reward_mode={args.reward_mode}"
    )

    tokenizer = train_word_tokenizer([record.query for record in records], min_frequency=args.min_frequency)
    input_ids, attention_mask = encode_texts(tokenizer, [record.query for record in records], args.max_length)
    input_ids_tensor = torch.tensor(input_ids, dtype=torch.long)
    attention_mask_tensor = torch.tensor(attention_mask, dtype=torch.long)

    model = ArchitecturePolicy(
        vocab_size=tokenizer.get_vocab_size(),
        max_agents=args.max_agents,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    metrics: dict[str, list[float]] = {"sft_loss": [], "grpo_reward": []}

    api_executor = None
    api_model_name = args.model_name
    if args.reward_mode == "api":
        client, available_models = _load_api_config()
        api_model_name = api_model_name or (available_models[0] if available_models else None)
        if not api_model_name:
            raise RuntimeError("--reward_mode api requires --model_name or a model name in api.txt")
        api_executor = DAGAPIExecutor(
            client=client,
            model_name=api_model_name,
            max_iterations=args.api_max_iterations,
        )
        print(f"Using API reward with model={api_model_name}")

    targets = _make_sft_targets(records, args.max_agents)
    for epoch in range(args.sft_epochs):
        losses = []
        for batch in _batch_indices(len(records), args.batch_size):
            optimizer.zero_grad()
            batch_tensor = torch.tensor(batch, dtype=torch.long)
            loss = model.sft_loss(
                input_ids_tensor[batch_tensor],
                attention_mask_tensor[batch_tensor],
                targets["target_num_agents"][batch_tensor],
                targets["target_parents"][batch_tensor],
                targets["target_extra_edges"][batch_tensor],
                targets["target_tools"][batch_tensor],
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            losses.append(float(loss.detach().item()))
        mean_loss = sum(losses) / max(1, len(losses))
        metrics["sft_loss"].append(mean_loss)
        print(f"SFT epoch {epoch + 1}/{args.sft_epochs}: loss={mean_loss:.4f}")

    for epoch in range(args.grpo_epochs):
        rewards_seen = []
        epoch_indices = random.sample(range(len(records)), len(records))
        progress = tqdm(
            total=len(records) * args.group_size,
            desc=f"GRPO {epoch + 1}/{args.grpo_epochs}",
            unit="rollout",
            dynamic_ncols=True,
        )
        for idx in epoch_indices:
            record = records[idx]
            log_probs = []
            entropies = []
            rewards = []
            for rollout_idx in range(args.group_size):
                sample = model.sample_one(
                    input_ids_tensor[idx : idx + 1],
                    attention_mask_tensor[idx : idx + 1],
                    greedy=False,
                )
                if args.reward_mode == "api":
                    reward = _api_rollout_reward(record, sample.architecture, api_executor)
                else:
                    reward = proxy_architecture_reward(
                        record.query,
                        record.required_domains,
                        sample.architecture,
                    )
                log_probs.append(sample.log_prob)
                entropies.append(sample.entropy)
                rewards.append(reward)
                progress.update(1)
                progress.set_postfix(
                    {
                        "last_reward": f"{reward:.3f}",
                        "query": f"{len(rewards_seen) // args.group_size + 1}/{len(records)}",
                        "group": f"{rollout_idx + 1}/{args.group_size}",
                        "mode": args.reward_mode,
                    }
                )

            reward_tensor = torch.tensor(rewards, dtype=torch.float32)
            std = reward_tensor.std(unbiased=False)
            advantages = (reward_tensor - reward_tensor.mean()) / (std + 1e-6)

            loss = torch.zeros(())
            for log_prob, entropy, advantage in zip(log_probs, entropies, advantages):
                loss = loss - log_prob * advantage.detach()
                loss = loss - args.entropy_coef * entropy
            loss = loss / args.group_size

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            rewards_seen.extend(rewards)
            progress.set_postfix(
                {
                    "group_reward": f"{sum(rewards) / max(1, len(rewards)):.3f}",
                    "mean_reward": f"{sum(rewards_seen) / max(1, len(rewards_seen)):.3f}",
                    "rollouts": f"{len(rewards_seen)}/{len(records) * args.group_size}",
                    "mode": args.reward_mode,
                }
            )
        progress.close()

        mean_reward = sum(rewards_seen) / max(1, len(rewards_seen))
        metrics["grpo_reward"].append(mean_reward)
        print(f"GRPO epoch {epoch + 1}/{args.grpo_epochs}: reward={mean_reward:.4f}")

    config = {
        "max_agents": args.max_agents,
        "embedding_dim": args.embedding_dim,
        "hidden_dim": args.hidden_dim,
        "max_length": args.max_length,
        "domains": DOMAINS,
        "reward_mode": args.reward_mode,
        "model_name": api_model_name or "",
        "all_domains": bool(all_domains_requested),
        "limit": args.limit,
        "max_queries_per_domain": args.max_queries_per_domain,
    }
    _save_checkpoint(args.output_dir, model, tokenizer, config, metrics)
    print(f"Saved checkpoint to {args.output_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a local MAS architecture generator.")
    parser.add_argument("--data_paths", nargs="*", default=None)
    parser.add_argument("--all_domains", action="store_true", default=False)
    parser.add_argument("--output_dir", default="data/mas_rl/checkpoints/dag_policy")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Global record limit, except in --all_domains mode where it is "
            "treated as max queries per domain."
        ),
    )
    parser.add_argument("--max_queries_per_domain", type=int, default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max_agents", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=96)
    parser.add_argument("--min_frequency", type=int, default=1)
    parser.add_argument("--embedding_dim", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--sft_epochs", type=int, default=0)
    parser.add_argument("--grpo_epochs", type=int, default=3)
    parser.add_argument("--group_size", type=int, default=6)
    parser.add_argument("--reward_mode", choices=["proxy", "api"], default="proxy")
    parser.add_argument("--model_name", default=None)
    parser.add_argument("--api_max_iterations", type=int, default=12)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--entropy_coef", type=float, default=0.001)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    return parser


if __name__ == "__main__":
    train(build_parser().parse_args())
