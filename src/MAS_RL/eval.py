"""Evaluate or query a trained MAS architecture generator."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

import torch

from src.MAS_RL.data import DEFAULT_QUERY_PATHS, load_query_records
from src.MAS_RL.policy import ArchitecturePolicy
from src.MAS_RL.reward import coverage_metrics, proxy_architecture_reward
from src.MAS_RL.tokenizer_utils import encode_texts, load_tokenizer


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


def predict_text(checkpoint_dir: str, text: str, samples: int, greedy: bool) -> None:
    model, tokenizer, config = _load_policy(checkpoint_dir)
    input_ids, attention_mask = encode_texts(tokenizer, [text], config["max_length"])
    ids = torch.tensor(input_ids, dtype=torch.long)
    mask = torch.tensor(attention_mask, dtype=torch.long)

    predictions = []
    with torch.no_grad():
        for _ in range(samples):
            sample = model.sample_one(ids, mask, greedy=greedy)
            predictions.append(sample.architecture.to_dict())
    print(json.dumps(predictions[0] if samples == 1 else predictions, indent=2))


def evaluate_dataset(args: argparse.Namespace) -> None:
    model, tokenizer, config = _load_policy(args.checkpoint_dir)
    records = load_query_records(args.data_paths or DEFAULT_QUERY_PATHS, limit=args.limit)
    if not records:
        raise RuntimeError("No query records found.")

    rows = []
    rewards = []
    coverages = []
    with torch.no_grad():
        for record in records:
            input_ids, attention_mask = encode_texts(tokenizer, [record.query], config["max_length"])
            ids = torch.tensor(input_ids, dtype=torch.long)
            mask = torch.tensor(attention_mask, dtype=torch.long)
            sample = model.sample_one(ids, mask, greedy=not args.sample)
            arch = sample.architecture
            reward = proxy_architecture_reward(record.query, record.required_domains, arch)
            metrics = coverage_metrics(record.required_domains, arch)
            rewards.append(reward)
            coverages.append(metrics["coverage"])
            rows.append(
                {
                    "query": record.query,
                    "required_domains": record.required_domains,
                    "reward": reward,
                    "metrics": metrics,
                    "architecture": arch.to_dict(),
                }
            )

    if args.output_jsonl:
        os.makedirs(os.path.dirname(args.output_jsonl), exist_ok=True)
        with open(args.output_jsonl, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")

    print(
        json.dumps(
            {
                "num_records": len(records),
                "mean_reward": sum(rewards) / max(1, len(rewards)),
                "mean_coverage": sum(coverages) / max(1, len(coverages)),
                "output_jsonl": args.output_jsonl,
            },
            indent=2,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a MAS architecture generator.")
    parser.add_argument("--checkpoint_dir", default="data/mas_rl/checkpoints/dag_policy")
    parser.add_argument("--text", default=None, help="If set, output an architecture for this text.")
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--greedy", action="store_true", default=False)
    parser.add_argument("--data_paths", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sample", action="store_true", default=False)
    parser.add_argument("--output_jsonl", default=None)
    return parser


if __name__ == "__main__":
    parsed = build_parser().parse_args()
    if parsed.text:
        predict_text(parsed.checkpoint_dir, parsed.text, parsed.samples, parsed.greedy)
    else:
        evaluate_dataset(parsed)
