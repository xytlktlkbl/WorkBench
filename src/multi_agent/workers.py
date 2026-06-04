"""
Domain Workers — specialized agents that each control a single domain's tools.

Each worker is a BaseAgent pre-configured with:
  - A domain-specific system prompt
  - Only the tools for that domain (3–7 tools each)
  - The company_directory tool (shared by all workers for email lookup)

Workers are stateless — they receive a task, execute it, and return results.
All inter-worker communication goes through the Blackboard (managed by the
Orchestrator).

Domain → Worker mapping:
  calendar                       → CalendarWorker
  email                          → EmailWorker
  analytics                      → AnalyticsWorker
  project_management             → ProjectManagementWorker
  customer_relationship_manager  → CRMWorker
  company_directory              → DirectoryWorker (always available)
"""

from datetime import datetime, timedelta

from src.multi_agent.base_agent import BaseAgent, tool_to_openai_schema
from src.tools import (
    calendar,
    email,
    analytics,
    project_management,
    customer_relationship_manager,
    company_directory,
)

# Hard-coded "current time" used by WorkBench. All workers need this for
# date/time reasoning (e.g., "tomorrow", "last fortnight").
HARDCODED_CURRENT_TIME = datetime(2023, 11, 30, 0, 0, 0)
CURRENT_DATE_STR = HARDCODED_CURRENT_TIME.strftime("%Y-%m-%d")
CURRENT_WEEKDAY = HARDCODED_CURRENT_TIME.strftime("%A")
CURRENT_DATETIME_STR = HARDCODED_CURRENT_TIME.strftime("%Y-%m-%d %H:%M:%S")

# Date context injected into every worker's system prompt so they can
# resolve relative time expressions correctly.
DATE_CONTEXT = (
    f"Today's date is {CURRENT_WEEKDAY}, {CURRENT_DATE_STR} "
    f"and the current time is {CURRENT_DATETIME_STR}. "
    f"Remember the current date and time when answering queries. "
    f"All dates must be in the year 2023. "
    f"Meetings must not start before 9am or end after 6pm."
)

# Base system prompt shared by all workers
_BASE_SYSTEM_PROMPT = f"""You are a specialized workplace assistant agent. {DATE_CONTEXT}

You are Sam, the owner of this workplace system. All calendar events, emails,
tasks, customers, and analytics data belong to you. When a task says "my"
meetings/emails/tasks, it refers to YOUR data. The entire system is yours.

You have access to tools for ONE specific domain. Use them to accomplish the
task you are given. Follow this process:
  1. Understand what information you need
  2. Use search/query tools to gather that information
  3. Use action tools (create/update/delete/send) to make changes
  4. Report back clearly what you did and what you found

IMPORTANT:
- Never ask "who are you?" or request identity confirmation. Always proceed with
  the task directly. You already know who you are — Sam, the system owner.
- Always search before acting — don't guess IDs or email addresses.
- Use the company_directory.find_email_address tool to look up email addresses by name.
- For dates, use the format "YYYY-MM-DD HH:MM:SS" for timestamps and "YYYY-MM-DD" for dates.
- When searching for events/emails/tasks, use the search tools first to find the right IDs.
- After making changes, briefly summarize what you did.
- If a task requires no action (e.g., condition not met), say so clearly and do NOT make any changes."""


def _make_worker_tools(*tool_lists):
    """Combine multiple tool lists and convert each to {{schema, callable}}."""
    tools = []
    seen = set()
    for tool_list in tool_lists:
        for t in tool_list:
            name = t.name if hasattr(t, "name") else t.__name__
            if name not in seen:
                seen.add(name)
                tools.append({
                    "schema": tool_to_openai_schema(t),
                    "callable": t.func if hasattr(t, "func") else t,
                    "original_name": name,
                })
    return tools


# Tool lists (matching the original toolkits.py groupings)
_CALENDAR_TOOLS = [
    calendar.get_event_information_by_id,
    calendar.search_events,
    calendar.create_event,
    calendar.delete_event,
    calendar.update_event,
]

_EMAIL_TOOLS = [
    email.get_email_information_by_id,
    email.search_emails,
    email.send_email,
    email.delete_email,
    email.forward_email,
    email.reply_email,
]

_ANALYTICS_TOOLS = [
    analytics.get_visitor_information_by_id,
    analytics.create_plot,
    analytics.total_visits_count,
    analytics.engaged_users_count,
    analytics.traffic_source_count,
    analytics.get_average_session_duration,
]

_PROJECT_MANAGEMENT_TOOLS = [
    project_management.get_task_information_by_id,
    project_management.search_tasks,
    project_management.create_task,
    project_management.delete_task,
    project_management.update_task,
]

_CRM_TOOLS = [
    customer_relationship_manager.search_customers,
    customer_relationship_manager.update_customer,
    customer_relationship_manager.add_customer,
    customer_relationship_manager.delete_customer,
]

_DIRECTORY_TOOLS = [
    company_directory.find_email_address,
]

# Company directory is always available to all workers
_COMMON_TOOLS = _DIRECTORY_TOOLS

# Combined: all tools across all domains (for single-agent / shared-tools experiments)
_ALL_TOOLS = (
    _CALENDAR_TOOLS + _EMAIL_TOOLS + _ANALYTICS_TOOLS +
    _PROJECT_MANAGEMENT_TOOLS + _CRM_TOOLS + _DIRECTORY_TOOLS
)


def _create_worker(name, model, domain_tools, domain_name):
    """Factory for creating a domain worker."""
    tools = _make_worker_tools(domain_tools, _COMMON_TOOLS)
    system_prompt = (
        _BASE_SYSTEM_PROMPT
        + f"\n\nYour domain is **{domain_name}**. You ONLY have tools for this domain "
        f"and the company directory. Focus exclusively on {domain_name} tasks."
    )
    return BaseAgent(
        name=name,
        model=model,
        tools=tools,
        system_prompt=system_prompt,
        max_iterations=12,
        temperature=0.0,
    )


class CalendarWorker(BaseAgent):
    """Worker for calendar operations (search, create, update, delete events)."""

    def __init__(self, model: str = "gpt-4-0125-preview"):
        tools = _make_worker_tools(_CALENDAR_TOOLS, _COMMON_TOOLS)
        system_prompt = (
            _BASE_SYSTEM_PROMPT
            + "\n\nYour domain is **Calendar Management**. You ONLY have tools for "
            "searching, creating, updating, and deleting calendar events. "
            "Always search for events by date/time first to find their IDs before "
            "modifying them. When scheduling, check for conflicts and respect the "
            "9am-6pm workday constraint."
        )
        super().__init__(
            name="CalendarWorker",
            model=model,
            tools=tools,
            system_prompt=system_prompt,
            max_iterations=12,
            temperature=0.0,
        )


class EmailWorker(BaseAgent):
    """Worker for email operations (search, send, delete, forward, reply)."""

    def __init__(self, model: str = "gpt-4-0125-preview"):
        tools = _make_worker_tools(_EMAIL_TOOLS, _COMMON_TOOLS)
        system_prompt = (
            _BASE_SYSTEM_PROMPT
            + "\n\nYour domain is **Email**. You ONLY have tools for searching, sending, "
            "deleting, forwarding, and replying to emails. Always search for emails "
            "by sender/recipient, subject, or date range to find the correct IDs "
            "before taking action. Use company_directory to resolve names to email "
            "addresses before sending."
        )
        super().__init__(
            name="EmailWorker",
            model=model,
            tools=tools,
            system_prompt=system_prompt,
            max_iterations=12,
            temperature=0.0,
        )


class AnalyticsWorker(BaseAgent):
    """Worker for analytics operations (queries, plots)."""

    def __init__(self, model: str = "gpt-4-0125-preview"):
        tools = _make_worker_tools(_ANALYTICS_TOOLS, _COMMON_TOOLS)
        system_prompt = (
            _BASE_SYSTEM_PROMPT
            + "\n\nYour domain is **Analytics**. You ONLY have tools for querying visitor "
            "data, counting visits/engagements/traffic sources, computing average "
            "session durations, and creating plots. Use date ranges in YYYY-MM-DD format."
        )
        super().__init__(
            name="AnalyticsWorker",
            model=model,
            tools=tools,
            system_prompt=system_prompt,
            max_iterations=12,
            temperature=0.0,
        )


class ProjectManagementWorker(BaseAgent):
    """Worker for project management (search, create, update, delete tasks)."""

    def __init__(self, model: str = "gpt-4-0125-preview"):
        tools = _make_worker_tools(_PROJECT_MANAGEMENT_TOOLS, _COMMON_TOOLS)
        system_prompt = (
            _BASE_SYSTEM_PROMPT
            + "\n\nYour domain is **Project Management**. You ONLY have tools for searching, "
            "creating, updating, and deleting project tasks. Tasks belong to boards "
            "('Back end', 'Front end', 'Design') and lists ('Backlog', 'In Progress', "
            "'In Review', 'Completed'). Always search first to find task IDs before "
            "modifying them."
        )
        super().__init__(
            name="ProjectManagementWorker",
            model=model,
            tools=tools,
            system_prompt=system_prompt,
            max_iterations=12,
            temperature=0.0,
        )


class CRMWorker(BaseAgent):
    """Worker for CRM operations (search, add, update, delete customers)."""

    def __init__(self, model: str = "gpt-4-0125-preview"):
        tools = _make_worker_tools(_CRM_TOOLS, _COMMON_TOOLS)
        system_prompt = (
            _BASE_SYSTEM_PROMPT
            + "\n\nYour domain is **Customer Relationship Management (CRM)**. You ONLY have "
            "tools for searching, adding, updating, and deleting customer records. "
            "Important fields: customer_name, assigned_to_email, status (Qualified/Won/"
            "Lost/Lead/Proposal), last_contact_date, follow_up_by, product_interest. "
            "Always search for customers first to find their IDs before modifying them. "
            "When checking if a customer has been contacted recently, compare "
            "last_contact_date to the current date."
        )
        super().__init__(
            name="CRMWorker",
            model=model,
            tools=tools,
            system_prompt=system_prompt,
            max_iterations=12,
            temperature=0.0,
        )


class DirectoryWorker(BaseAgent):
    """Worker for company directory lookups only."""

    def __init__(self, model: str = "gpt-4-0125-preview"):
        tools = _make_worker_tools(_DIRECTORY_TOOLS)
        system_prompt = (
            _BASE_SYSTEM_PROMPT
            + "\n\nYour domain is **Company Directory**. You can look up email addresses "
            "by employee name. Simply search and return the matching email(s)."
        )
        super().__init__(
            name="DirectoryWorker",
            model=model,
            tools=tools,
            system_prompt=system_prompt,
            max_iterations=5,
            temperature=0.0,
        )


class FullAgent(BaseAgent):
    """Single agent with ALL domain tools — no orchestrator, no domain workers."""

    def __init__(self, model: str = "gpt-4-0125-preview"):
        tools = _make_worker_tools(_ALL_TOOLS)
        system_prompt = (
            _BASE_SYSTEM_PROMPT
            + "\n\nYou have access to ALL tools across ALL domains: calendar, email, "
            "analytics, project_management, customer_relationship_manager, and "
            "company_directory. You must handle the complete task yourself — "
            "search across multiple domains, reason about conditions, and take "
            "actions in any domain as needed."
        )
        super().__init__(
            name="FullAgent",
            model=model,
            tools=tools,
            system_prompt=system_prompt,
            max_iterations=25,
            temperature=0.0,
        )


# Registry: domain name → Worker class
WORKER_REGISTRY: dict[str, type[BaseAgent]] = {
    "calendar": CalendarWorker,
    "email": EmailWorker,
    "analytics": AnalyticsWorker,
    "project_management": ProjectManagementWorker,
    "customer_relationship_manager": CRMWorker,
    "company_directory": DirectoryWorker,
}

# Shorthand aliases used in the dataset
_DOMAIN_ALIASES = {
    "crm": "customer_relationship_manager",
}

ALL_WORKERS = WORKER_REGISTRY


def get_worker_for_domain(domain: str, model: str = "gpt-4-0125-preview", shared_tools: bool = False) -> BaseAgent:
    """
    Create a worker instance for a given domain name.

    Parameters
    ----------
    domain : str
        Domain name. Supports aliases: 'crm' → 'customer_relationship_manager'.
    model : str
        OpenAI model name.
    shared_tools : bool
        If True, the worker gets ALL tools instead of just its domain tools.
        Used for ablation experiments.

    Returns
    -------
    BaseAgent instance (a domain worker).

    Raises
    ------
    ValueError if the domain is unknown.
    """
    domain = _DOMAIN_ALIASES.get(domain, domain)
    worker_cls = WORKER_REGISTRY.get(domain)
    if worker_cls is None:
        raise ValueError(
            f"Unknown domain: '{domain}'. Available: {list(WORKER_REGISTRY.keys())}"
        )
    if shared_tools:
        tools = _make_worker_tools(_ALL_TOOLS)
        system_prompt = (
            _BASE_SYSTEM_PROMPT
            + f"\n\nYour domain is **{domain}**. However, you have been given access "
            "to ALL tools across all domains for this experiment. You may use tools "
            "from any domain to accomplish the task."
        )
        return BaseAgent(
            name=f"Shared{worker_cls.__name__}",
            model=model,
            tools=tools,
            system_prompt=system_prompt,
            max_iterations=15,
            temperature=0.0,
        )
    return worker_cls(model=model)
