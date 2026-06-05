"""PyTorch policy for command-DAG architecture generation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.distributions import Bernoulli, Categorical

from src.MAS_RL.schema import Architecture, DOMAINS, architecture_from_masks


@dataclass
class SampledArchitecture:
    architecture: Architecture
    log_prob: torch.Tensor
    entropy: torch.Tensor


class ArchitecturePolicy(nn.Module):
    """A small local policy that samples command DAGs and tool scopes."""

    def __init__(
        self,
        vocab_size: int,
        max_agents: int = 4,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.max_agents = max_agents
        self.num_domains = len(DOMAINS)

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.encoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.num_agents_head = nn.Linear(hidden_dim, max_agents)
        self.parent_heads = nn.ModuleList(
            [nn.Linear(hidden_dim, child_idx) for child_idx in range(1, max_agents)]
        )
        self.extra_edge_head = nn.Linear(hidden_dim, max_agents * max_agents)
        self.tool_head = nn.Linear(hidden_dim, max_agents * self.num_domains)

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(input_ids)
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (embedded * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return self.encoder(pooled)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict[str, object]:
        hidden = self.encode(input_ids, attention_mask)
        return {
            "hidden": hidden,
            "num_agents_logits": self.num_agents_head(hidden),
            "parent_logits": [head(hidden) for head in self.parent_heads],
            "extra_edge_logits": self.extra_edge_head(hidden).view(-1, self.max_agents, self.max_agents),
            "tool_logits": self.tool_head(hidden).view(-1, self.max_agents, self.num_domains),
        }

    def sample_one(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        greedy: bool = False,
    ) -> SampledArchitecture:
        outputs = self.forward(input_ids, attention_mask)
        num_agents_logits = outputs["num_agents_logits"][0]
        parent_logits = [item[0] for item in outputs["parent_logits"]]
        extra_edge_logits = outputs["extra_edge_logits"][0]
        tool_logits = outputs["tool_logits"][0]

        log_prob = torch.zeros((), device=input_ids.device)
        entropy = torch.zeros((), device=input_ids.device)

        num_dist = Categorical(logits=num_agents_logits)
        num_idx = torch.argmax(num_dist.probs) if greedy else num_dist.sample()
        log_prob = log_prob + num_dist.log_prob(num_idx)
        entropy = entropy + num_dist.entropy()
        num_agents = int(num_idx.item()) + 1

        adjacency = [[0 for _ in range(num_agents)] for _ in range(num_agents)]

        # Each non-root node chooses one parent among earlier nodes. This gives a
        # connected command DAG rooted at agent 0.
        for child_idx in range(1, num_agents):
            dist = Categorical(logits=parent_logits[child_idx - 1][:child_idx])
            parent = torch.argmax(dist.probs) if greedy else dist.sample()
            parent_idx = int(parent.item())
            adjacency[parent_idx][child_idx] = 1
            log_prob = log_prob + dist.log_prob(parent)
            entropy = entropy + dist.entropy()

        # Optional extra DAG edges. Only i < j is allowed, so cycles are impossible.
        for src in range(num_agents):
            for dst in range(src + 1, num_agents):
                if adjacency[src][dst]:
                    continue
                dist = Bernoulli(logits=extra_edge_logits[src, dst])
                bit = (dist.probs > 0.5).float() if greedy else dist.sample()
                enabled = int(bit.item())
                adjacency[src][dst] = enabled
                log_prob = log_prob + dist.log_prob(bit)
                entropy = entropy + dist.entropy()

        tool_mask = [[0 for _ in range(self.num_domains)] for _ in range(num_agents)]
        for agent_idx in range(num_agents):
            for domain_idx in range(self.num_domains):
                dist = Bernoulli(logits=tool_logits[agent_idx, domain_idx])
                bit = (dist.probs > 0.5).float() if greedy else dist.sample()
                tool_mask[agent_idx][domain_idx] = int(bit.item())
                log_prob = log_prob + dist.log_prob(bit)
                entropy = entropy + dist.entropy()

        return SampledArchitecture(
            architecture=architecture_from_masks(num_agents, adjacency, tool_mask),
            log_prob=log_prob,
            entropy=entropy,
        )

    def sft_loss(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        target_num_agents: torch.Tensor,
        target_parents: torch.Tensor,
        target_extra_edges: torch.Tensor,
        target_tools: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.forward(input_ids, attention_mask)
        loss = nn.functional.cross_entropy(outputs["num_agents_logits"], target_num_agents)

        batch_size = input_ids.shape[0]
        for child_idx, logits in enumerate(outputs["parent_logits"], start=1):
            active = target_parents[:, child_idx] >= 0
            if active.any():
                loss = loss + nn.functional.cross_entropy(
                    logits[active, :child_idx],
                    target_parents[active, child_idx],
                )

        extra_logits = outputs["extra_edge_logits"]
        tool_logits = outputs["tool_logits"]
        loss = loss + nn.functional.binary_cross_entropy_with_logits(
            extra_logits.reshape(batch_size, -1),
            target_extra_edges.reshape(batch_size, -1),
        )
        loss = loss + nn.functional.binary_cross_entropy_with_logits(
            tool_logits.reshape(batch_size, -1),
            target_tools.reshape(batch_size, -1),
        )
        return loss

