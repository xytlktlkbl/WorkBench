"""Compile generated DAG architectures into API-backed MAS executions."""

from __future__ import annotations

import json
import contextlib
import io
from collections import defaultdict
from typing import Callable

from openai import OpenAI

from src.multi_agent.base_agent import BaseAgent, tool_to_openai_schema
from src.multi_agent.workers import DATE_CONTEXT
from src.MAS_RL.schema import Architecture, DOMAIN_TOOLS
from src.tools import (
    analytics,
    calendar,
    company_directory,
    customer_relationship_manager,
    email,
    project_management,
)


DOMAIN_TOOL_OBJECTS = {
    "calendar": [
        calendar.get_event_information_by_id,
        calendar.search_events,
        calendar.create_event,
        calendar.delete_event,
        calendar.update_event,
    ],
    "email": [
        email.get_email_information_by_id,
        email.search_emails,
        email.send_email,
        email.delete_email,
        email.forward_email,
        email.reply_email,
    ],
    "analytics": [
        analytics.get_visitor_information_by_id,
        analytics.create_plot,
        analytics.total_visits_count,
        analytics.engaged_users_count,
        analytics.traffic_source_count,
        analytics.get_average_session_duration,
    ],
    "project_management": [
        project_management.get_task_information_by_id,
        project_management.search_tasks,
        project_management.create_task,
        project_management.delete_task,
        project_management.update_task,
    ],
    "customer_relationship_manager": [
        customer_relationship_manager.search_customers,
        customer_relationship_manager.update_customer,
        customer_relationship_manager.add_customer,
        customer_relationship_manager.delete_customer,
    ],
    "company_directory": [
        company_directory.find_email_address,
    ],
}


def _tool_name(tool) -> str:
    return tool.name if hasattr(tool, "name") else tool.__name__


def _make_external_tools(tool_domains: list[str]) -> list[dict]:
    tools = []
    seen = set()
    for domain in tool_domains:
        for tool in DOMAIN_TOOL_OBJECTS.get(domain, []):
            name = _tool_name(tool)
            if name in seen:
                continue
            seen.add(name)
            tools.append(
                {
                    "schema": tool_to_openai_schema(tool),
                    "callable": tool.func if hasattr(tool, "func") else tool,
                    "original_name": name,
                }
            )
    return tools


def _command_schema(child_id: int) -> dict:
    return {
        "type": "function",
        "function": {
            "name": f"command_agent_{child_id}",
            "description": (
                f"Command child agent {child_id} to complete a specific subtask. "
                "Use this when that child has the right tool scope or should handle "
                "a downstream step."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "The precise subtask for the child agent.",
                    }
                },
                "required": ["task"],
            },
        },
    }


def _command_call_prefix(child_id: int) -> str:
    return f"command_agent_{child_id}.func("


class DAGAPIExecutor:
    """Execute a generated Architecture using frozen API-backed agents."""

    def __init__(
        self,
        client: OpenAI,
        model_name: str,
        max_iterations: int = 12,
        temperature: float = 0.0,
    ):
        self.client = client
        self.model_name = model_name
        self.max_iterations = max_iterations
        self.temperature = temperature

    def run(self, query: str, architecture: Architecture) -> dict:
        result = self._run_agent(
            architecture=architecture,
            agent_id=0,
            task=query,
            parent_context="",
        )
        return {
            "function_calls": result["function_calls"],
            "full_response": json.dumps(
                {
                    "query": query,
                    "architecture": architecture.to_dict(),
                    "root_output": result["output"],
                    "agent_trace": result["trace"],
                },
                ensure_ascii=False,
            ),
            "error": result["error"],
            "summary": result["output"],
        }

    def _run_agent(
        self,
        architecture: Architecture,
        agent_id: int,
        task: str,
        parent_context: str,
    ) -> dict:
        children = [
            child
            for child, enabled in enumerate(architecture.adjacency[agent_id])
            if enabled
        ]
        child_results_by_id: dict[int, list[list[str]]] = defaultdict(list)
        child_traces = []

        tools = _make_external_tools(architecture.tool_domains[agent_id])
        for child_id in children:
            tools.append(
                {
                    "schema": _command_schema(child_id),
                    "callable": self._make_child_callable(
                        architecture=architecture,
                        child_id=child_id,
                        parent_agent_id=agent_id,
                        parent_context=parent_context,
                        child_results_by_id=child_results_by_id,
                        child_traces=child_traces,
                    ),
                    "original_name": f"command_agent_{child_id}",
                }
            )

        system_prompt = self._system_prompt(architecture, agent_id, children)
        agent = BaseAgent(
            name=f"DAGAgent{agent_id}",
            model=self.model_name,
            tools=tools,
            system_prompt=system_prompt,
            max_iterations=self.max_iterations,
            temperature=self.temperature,
        )
        agent.set_client(self.client)

        with contextlib.redirect_stderr(io.StringIO()):
            result = agent.run(task, blackboard_context=parent_context)
        ordered_calls = self._expand_child_calls(
            result.get("function_calls", []),
            child_results_by_id,
        )
        error = result.get("error", "")
        for child_trace in child_traces:
            if child_trace.get("error"):
                error = error or child_trace["error"]

        return {
            "output": result.get("output", ""),
            "function_calls": ordered_calls,
            "error": error,
            "trace": {
                "agent_id": agent_id,
                "task": task,
                "tool_domains": architecture.tool_domains[agent_id],
                "children": children,
                "output": result.get("output", ""),
                "error": result.get("error", ""),
                "child_traces": child_traces,
            },
        }

    def _make_child_callable(
        self,
        architecture: Architecture,
        child_id: int,
        parent_agent_id: int,
        parent_context: str,
        child_results_by_id: dict[int, list[list[str]]],
        child_traces: list[dict],
    ) -> Callable:
        def command_child(task: str) -> dict:
            context = (
                f"Parent agent {parent_agent_id} delegated this subtask.\n"
                f"Original parent context:\n{parent_context}"
            )
            child_result = self._run_agent(
                architecture=architecture,
                agent_id=child_id,
                task=task,
                parent_context=context,
            )
            child_results_by_id[child_id].append(child_result["function_calls"])
            child_traces.append(child_result["trace"])
            return {
                "agent_id": child_id,
                "output": child_result["output"],
                "function_calls": child_result["function_calls"],
                "error": child_result["error"],
            }

        return command_child

    def _expand_child_calls(
        self,
        calls: list[str],
        child_results_by_id: dict[int, list[list[str]]],
    ) -> list[str]:
        ordered = []
        for call in calls:
            matched_child = None
            for child_id in child_results_by_id:
                if call.startswith(_command_call_prefix(child_id)):
                    matched_child = child_id
                    break
            if matched_child is None:
                ordered.append(call)
                continue
            child_queue = child_results_by_id[matched_child]
            if child_queue:
                ordered.extend(child_queue.pop(0))
        return ordered

    def _system_prompt(self, architecture: Architecture, agent_id: int, children: list[int]) -> str:
        tool_domains = architecture.tool_domains[agent_id]
        tool_lines = []
        for domain in tool_domains:
            tool_lines.append(f"- {domain}: {', '.join(DOMAIN_TOOLS.get(domain, []))}")
        child_text = ", ".join(str(child) for child in children) if children else "none"
        tool_text = "\n".join(tool_lines) if tool_lines else "- no direct workplace tools"
        return f"""You are agent {agent_id} in a generated multi-agent command DAG.
{DATE_CONTEXT}

Command-edge semantics: if you have child agents, you may command them with the
command_agent_X tool. A child should receive a precise subtask and all details it
needs. You may also use your direct workplace tools.

Direct tool domains:
{tool_text}

Child agents you may command: {child_text}

Rules:
- Complete the task using your direct tools and/or child agents.
- Do not ask for identity confirmation; the user is Sam, the system owner.
- Search before taking side-effect actions.
- If you delegate, include all relevant names, dates, conditions, and outputs
  from previous steps in the child task.
- Return a concise summary when finished.
"""
