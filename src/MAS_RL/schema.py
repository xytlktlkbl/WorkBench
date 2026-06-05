"""Schema and helpers for command-DAG MAS architectures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


DOMAINS: list[str] = [
    "calendar",
    "email",
    "analytics",
    "project_management",
    "customer_relationship_manager",
    "company_directory",
]

DOMAIN_ALIASES: dict[str, str] = {
    "crm": "customer_relationship_manager",
    "pm": "project_management",
    "directory": "company_directory",
}

DOMAIN_TOOLS: dict[str, list[str]] = {
    "calendar": [
        "calendar.get_event_information_by_id",
        "calendar.search_events",
        "calendar.create_event",
        "calendar.delete_event",
        "calendar.update_event",
    ],
    "email": [
        "email.get_email_information_by_id",
        "email.search_emails",
        "email.send_email",
        "email.delete_email",
        "email.forward_email",
        "email.reply_email",
    ],
    "analytics": [
        "analytics.get_visitor_information_by_id",
        "analytics.create_plot",
        "analytics.total_visits_count",
        "analytics.engaged_users_count",
        "analytics.traffic_source_count",
        "analytics.get_average_session_duration",
    ],
    "project_management": [
        "project_management.get_task_information_by_id",
        "project_management.search_tasks",
        "project_management.create_task",
        "project_management.delete_task",
        "project_management.update_task",
    ],
    "customer_relationship_manager": [
        "customer_relationship_manager.search_customers",
        "customer_relationship_manager.update_customer",
        "customer_relationship_manager.add_customer",
        "customer_relationship_manager.delete_customer",
    ],
    "company_directory": [
        "company_directory.find_email_address",
    ],
}


def normalize_domain(domain: str) -> str:
    key = domain.strip().lower()
    return DOMAIN_ALIASES.get(key, key)


def normalize_domains(domains: list[str]) -> list[str]:
    seen = set()
    normalized = []
    for domain in domains:
        value = normalize_domain(domain)
        if value in DOMAINS and value not in seen:
            seen.add(value)
            normalized.append(value)
    return normalized


@dataclass(frozen=True)
class Architecture:
    """A command DAG plus a domain-level tool scope for each agent."""

    num_agents: int
    adjacency: list[list[int]]
    tool_domains: list[list[str]]

    def active_edges(self) -> list[tuple[int, int]]:
        edges = []
        for src in range(self.num_agents):
            for dst in range(self.num_agents):
                if self.adjacency[src][dst]:
                    edges.append((src, dst))
        return edges

    def child_map(self) -> dict[int, list[int]]:
        mapping = {i: [] for i in range(self.num_agents)}
        for src, dst in self.active_edges():
            mapping[src].append(dst)
        return mapping

    def tools_by_agent(self) -> list[list[str]]:
        tools = []
        for domains in self.tool_domains:
            agent_tools = []
            for domain in domains:
                agent_tools.extend(DOMAIN_TOOLS.get(domain, []))
            tools.append(agent_tools)
        return tools

    def to_dict(self) -> dict[str, Any]:
        agents = []
        children = self.child_map()
        tools = self.tools_by_agent()
        for idx in range(self.num_agents):
            agents.append(
                {
                    "id": idx,
                    "commands": children[idx],
                    "tool_domains": self.tool_domains[idx],
                    "tools": tools[idx],
                }
            )
        return {
            "num_agents": self.num_agents,
            "adjacency": self.adjacency,
            "edges": [{"from": src, "to": dst} for src, dst in self.active_edges()],
            "agents": agents,
        }


def architecture_from_masks(
    num_agents: int,
    adjacency: list[list[int]],
    tool_mask: list[list[int]],
) -> Architecture:
    tool_domains = []
    for agent_idx in range(num_agents):
        domains = [
            DOMAINS[domain_idx]
            for domain_idx, enabled in enumerate(tool_mask[agent_idx])
            if enabled
        ]
        tool_domains.append(domains)
    return Architecture(
        num_agents=num_agents,
        adjacency=[row[:num_agents] for row in adjacency[:num_agents]],
        tool_domains=tool_domains,
    )

