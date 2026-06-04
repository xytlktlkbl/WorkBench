"""
Multi-Agent System for WorkBench Evaluation
===========================================
Orchestrator + Domain Workers + Blackboard architecture.

The Orchestrator decomposes user tasks and dispatches subtasks to
specialized Domain Workers. Each Worker only has access to its own
domain's tools, reducing cognitive load. The Blackboard provides
shared memory for inter-agent communication.
"""

from src.multi_agent.blackboard import Blackboard
from src.multi_agent.base_agent import BaseAgent, tool_to_openai_schema
from src.multi_agent.workers import (
    CalendarWorker,
    EmailWorker,
    AnalyticsWorker,
    ProjectManagementWorker,
    CRMWorker,
    DirectoryWorker,
    FullAgent,
    ALL_WORKERS,
)
from src.multi_agent.orchestrator import Orchestrator

__all__ = [
    "Blackboard",
    "BaseAgent",
    "tool_to_openai_schema",
    "CalendarWorker",
    "EmailWorker",
    "AnalyticsWorker",
    "ProjectManagementWorker",
    "CRMWorker",
    "DirectoryWorker",
    "FullAgent",
    "ALL_WORKERS",
    "Orchestrator",
]
