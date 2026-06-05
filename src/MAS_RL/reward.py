"""Proxy rewards for architecture generation.

The reward here evaluates architecture quality locally using WorkBench domain
labels. It is intentionally separated from the policy so it can later be
replaced by a frozen-MAS executor reward.
"""

from __future__ import annotations

from src.MAS_RL.schema import Architecture, DOMAINS


CONDITIONAL_MARKERS = [
    "if ",
    "only if",
    "haven't",
    "hasn't",
    "check if",
    "then",
    "provided that",
]


def _is_reachable_from_root(arch: Architecture, node: int) -> bool:
    seen = {0}
    frontier = [0]
    while frontier:
        current = frontier.pop()
        for _, dst in [edge for edge in arch.active_edges() if edge[0] == current]:
            if dst not in seen:
                seen.add(dst)
                frontier.append(dst)
    return node in seen


def proxy_architecture_reward(
    query: str,
    required_domains: list[str],
    arch: Architecture,
) -> float:
    """Score an architecture without executing frozen LLM workers."""
    required = set(required_domains)
    provided = set()
    for domains in arch.tool_domains:
        provided.update(domains)

    if not required:
        required = provided

    coverage = len(required & provided) / max(1, len(required))
    extra_domains = provided - required
    missing_domains = required - provided

    reward = 0.0
    reward += 1.2 * coverage
    reward -= 0.8 * len(missing_domains)
    reward -= 0.08 * len(extra_domains)

    # Prefer compact systems unless the query really spans domains.
    multi_required = len(required) > 1
    if multi_required:
        reward += 0.2 if arch.num_agents > 1 else -0.25
    else:
        reward += 0.15 if arch.num_agents == 1 else -0.08 * (arch.num_agents - 1)

    lowered = query.lower()
    conditional = any(marker in lowered for marker in CONDITIONAL_MARKERS)
    edges = arch.active_edges()
    if conditional and len(required) > 1:
        reward += 0.12 if edges else -0.2

    # Penalize unused leaves and unreachable generated nodes.
    child_map = arch.child_map()
    for idx in range(arch.num_agents):
        has_tools = bool(arch.tool_domains[idx])
        has_children = bool(child_map[idx])
        if not has_tools and not has_children:
            reward -= 0.25
        if idx > 0 and not _is_reachable_from_root(arch, idx):
            reward -= 0.25

    # Encourage specialization: many domains on the same non-root agent is noisy.
    for domains in arch.tool_domains:
        if len(domains) > 2:
            reward -= 0.06 * (len(domains) - 2)

    # Mild complexity cost.
    reward -= 0.03 * max(0, arch.num_agents - 1)
    reward -= 0.02 * len(edges)
    reward -= 0.01 * sum(len(domains) for domains in arch.tool_domains)

    # Directory is useful as support, but usually not a primary required domain.
    if "company_directory" in provided and len(provided) == 1 and "company_directory" not in required:
        reward -= 0.2

    return float(reward)


def coverage_metrics(required_domains: list[str], arch: Architecture) -> dict[str, float]:
    required = set(required_domains)
    provided = set()
    for domains in arch.tool_domains:
        provided.update(domains)
    return {
        "coverage": len(required & provided) / max(1, len(required)),
        "extra_domains": float(len(provided - required)),
        "num_agents": float(arch.num_agents),
        "num_edges": float(len(arch.active_edges())),
        "num_tool_domains": float(sum(domain in DOMAINS for domains in arch.tool_domains for domain in domains)),
    }

