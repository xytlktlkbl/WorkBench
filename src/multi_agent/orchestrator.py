"""
Orchestrator — task decomposition, routing, and result aggregation.

The Orchestrator is the "brain" of the multi-agent system. It:
  1. Receives a natural-language task from the user
  2. Decomposes it into subtasks (one per domain)
  3. Dispatches subtasks to the appropriate Domain Workers
  4. Manages the Blackboard to pass context between workers
  5. Handles conditional logic (e.g., "if X, then Y, else do nothing")
  6. Collects all function calls and returns them for evaluation

The Orchestrator uses two "meta-tools" via OpenAI function calling:
  - dispatch_worker: route a subtask to a specific domain worker
  - finalize: declare the task complete and return results

When the LLM "calls" dispatch_worker, we actually execute the worker and
feed the result back as a tool response. When it calls finalize, we stop.
"""

import json
import traceback
from datetime import datetime
from typing import Optional

from openai import OpenAI

from src.multi_agent.blackboard import Blackboard
from src.multi_agent.workers import get_worker_for_domain, WORKER_REGISTRY

# Hard-coded date context (same as workers)
HARDCODED_CURRENT_TIME = datetime(2023, 11, 30, 0, 0, 0)
CURRENT_DATE_STR = HARDCODED_CURRENT_TIME.strftime("%Y-%m-%d")
CURRENT_WEEKDAY = HARDCODED_CURRENT_TIME.strftime("%A")
CURRENT_DATETIME_STR = HARDCODED_CURRENT_TIME.strftime("%Y-%m-%d %H:%M:%S")

DATE_CONTEXT = (
    f"Today's date is {CURRENT_WEEKDAY}, {CURRENT_DATE_STR} "
    f"and the current time is {CURRENT_DATETIME_STR}. "
    f"Remember the current date and time when reasoning about dates. "
    f"All dates are in the year 2023. "
    f"Meetings must not start before 9am or end after 6pm."
)

# Domain descriptions for the orchestrator's system prompt
_DOMAIN_DESCRIPTIONS = """
Available domain workers:

1. **calendar** — Manage calendar events. Can search events by date/query,
   get event details, create/update/delete events.
   Use for: scheduling meetings, rescheduling, cancelling, finding free slots.

2. **email** — Manage emails. Can search emails by sender/subject/date,
   get email details, send/delete/forward/reply to emails.
   Use for: finding emails by subject/sender, sending emails, cleaning inbox.

3. **analytics** — Query analytics data. Can get visitor info, count visits/
   engagements/traffic sources, compute session durations, create plots.
   Use for: any data analysis or reporting task.

4. **project_management** — Manage project tasks. Can search tasks by name/
   assignee/list/board/due date, get task details, create/update/delete tasks.
   Use for: task assignment, status updates, finding who has fewest/most tasks.

5. **customer_relationship_manager** — Manage CRM records. Can search customers
   by name/email/status/assigned rep/contact date, add/update/delete customers.
   Use for: checking customer status, adding new leads, assigning customers.

6. **company_directory** — Look up email addresses by employee name.
   Use when you need to resolve a person's name to their email address.

You also have two special commands:
- **dispatch_worker**: Send a subtask to a domain worker and get results back.
- **finalize**: End the task and return all collected function calls.
"""

ORCHESTRATOR_SYSTEM_PROMPT = f"""You are an intelligent task orchestrator for a multi-agent workplace system.
{DATE_CONTEXT}

{_DOMAIN_DESCRIPTIONS}

## Your Job

You receive a user's task and coordinate domain workers to complete it.
Follow this process:

### Step 1: Analyze the task
Read the task carefully. Identify:
- Which domains are involved? (calendar, email, analytics, project_management, crm)
- What is the logical order? (e.g., first search CRM, then conditionally create a calendar event)
- Are there conditions? (e.g., "if X, then Y", "only if not contacted recently")

### Step 2: Dispatch workers in order
For each step, use dispatch_worker with:
- **worker**: one of the 6 domain names
- **task**: a clear, specific description of what the worker should do.
  Include ALL necessary details: names, dates, email subjects, meeting durations, etc.
  If the worker needs info from a previous step, include it in the task.

### Step 3: Evaluate results
After each worker returns:
- Read the worker's findings carefully
- If a condition is not met, you may skip remaining steps and finalize
- If you have all the info to proceed, dispatch the next worker

### Step 4: Finalize
When the task is complete, call finalize with a short summary.
The system will collect all function calls from all workers automatically.

## CRITICAL RULES
- **Always include all details in the worker task** — dates, names, times, durations, subjects.
  Don't make the worker guess or search for info you already have.
- **Resolve names to email addresses** — use the company_directory worker or ask
  the domain worker to do it. Email addresses use @atlas.com domain.
- **For conditional tasks**: dispatch the first worker, check its findings, then
  decide whether to dispatch the next one or finalize.
- **If no action is needed** (condition not met), finalize immediately WITHOUT
  dispatching more workers. Say "No action needed because [reason]".
- The current year is 2023. All dates should be in 2023.
- "fortnight" = 14 days. "tomorrow" = the next calendar day.
- Meeting times use 24-hour format: "13:00:00" for 1pm.
- Work hours are 9am-6pm. Meetings start at or after 9am and end by 6pm.
"""

# Meta-tool schemas for the orchestrator
DISPATCH_WORKER_SCHEMA = {
    "type": "function",
    "function": {
        "name": "dispatch_worker",
        "description": "Dispatch a subtask to a domain-specific worker agent. The worker will execute the task using its domain tools and return findings + actions taken.",
        "parameters": {
            "type": "object",
            "properties": {
                "worker": {
                    "type": "string",
                    "enum": [
                        "calendar",
                        "email",
                        "analytics",
                        "project_management",
                        "customer_relationship_manager",
                        "company_directory",
                    ],
                    "description": "The domain worker to dispatch.",
                },
                "task": {
                    "type": "string",
                    "description": "The specific task for the worker. Include ALL details: names, dates, times, durations, addresses, etc. Be precise.",
                },
            },
            "required": ["worker", "task"],
        },
    },
}

FINALIZE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "finalize",
        "description": "Signal that the task is complete. Call this when all necessary worker dispatches are done or when you determine no action is needed.",
        "parameters": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "Brief summary of what was done and why. If no action was needed, explain the condition that was not met.",
                },
            },
            "required": ["summary"],
        },
    },
}

ORCHESTRATOR_TOOLS = [DISPATCH_WORKER_SCHEMA, FINALIZE_SCHEMA]


class Orchestrator:
    """
    Task orchestrator for the multi-agent system.

    Parameters
    ----------
    model : str
        OpenAI model name for the orchestrator's reasoning (default: gpt-4).
    max_iterations : int
        Maximum number of orchestrator steps (default: 10).
    """

    def __init__(
        self,
        model: str = "gpt-4-0125-preview",
        max_iterations: int = 10,
    ):
        self.model = model
        self.max_iterations = max_iterations
        self._client: Optional[OpenAI] = None

    def set_client(self, client: OpenAI) -> None:
        """Set the OpenAI client."""
        self._client = client

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            raise RuntimeError("OpenAI client not set. Call set_client() first.")
        return self._client

    def run(self, query: str) -> dict:
        """
        Orchestrate the completion of a user task.

        Parameters
        ----------
        query : str
            The natural-language task from the user.

        Returns
        -------
        dict with:
            - "function_calls": list[str] — all tool calls from all workers
            - "full_response": str — full conversation trace
            - "error": str — error message if any
            - "summary": str — orchestrator's final summary
        """
        blackboard = Blackboard()
        all_function_calls: list[str] = []
        all_responses: list[str] = []
        error = ""
        final_summary = ""

        # Single-domain quick path: if the query clearly only needs one domain,
        # we can skip the orchestrator LLM call and dispatch directly.
        domain_hint = self._guess_domain(query)
        if domain_hint:
            # Quick dispatch for single-domain tasks
            result = self._dispatch_worker(domain_hint, query, blackboard)
            all_function_calls.extend(result.get("function_calls", []))
            all_responses.append(f"[{domain_hint}] {result.get('output', '')}")
            final_summary = result.get("output", "Done.")
        else:
            # Multi-domain: run the orchestrator's ReAct loop
            result = self._run_orchestrator_loop(
                query, blackboard, all_function_calls, all_responses
            )
            final_summary = result.get("summary", "")
            error = result.get("error", "")

        return {
            "function_calls": all_function_calls,
            "full_response": json.dumps({
                "query": query,
                "worker_responses": all_responses,
                "final_summary": final_summary,
            }, ensure_ascii=False),
            "error": error,
            "summary": final_summary,
        }

    # ------------------------------------------------------------------
    # Single-domain fast path
    # ------------------------------------------------------------------

    def _guess_domain(self, query: str) -> Optional[str]:
        """
        Heuristic to detect single-domain queries. Returns the domain name
        if the query clearly only involves one domain, otherwise None
        (falls back to full orchestrator loop).

        This is an optimization: it avoids an LLM call for ~70% of queries
        (the single-domain ones) and only uses the orchestrator for the
        genuinely multi-domain 210 queries.
        """
        q = query.lower()

        # Strong domain signals (words that almost always indicate a domain)
        calendar_signals = [
            "meeting", "event", "schedule", "calendar", "appointment",
            "cancel", "delete my first meeting", "rename", "move my first",
            "push back", "delay", "reschedule", "book a",
        ]
        email_signals = [
            "email", "send an email", "forward", "reply",
            "my last email from", "my emails from",
        ]
        analytics_signals = [
            "plot", "analytics", "visits", "traffic", "session duration",
            "engaged users", "visitor",
        ]
        pm_signals = [
            "task", "backlog", "in progress", "in review", "board",
            "assigned to", "due on", "fewest", "overdue",
        ]
        crm_signals = [
            "customer", "crm", "lead", "client", "spoke to",
            "contacted", "fortnight", "assigned to them",
        ]

        # Count signals for each domain
        scores = {
            "calendar": 0,
            "email": 0,
            "analytics": 0,
            "project_management": 0,
            "customer_relationship_manager": 0,
        }

        for signal in calendar_signals:
            if signal in q:
                scores["calendar"] += 1
        for signal in email_signals:
            if signal in q:
                scores["email"] += 1
        for signal in analytics_signals:
            if signal in q:
                scores["analytics"] += 1
        for signal in pm_signals:
            if signal in q:
                scores["project_management"] += 1
        for signal in crm_signals:
            if signal in q:
                scores["customer_relationship_manager"] += 1

        # Count how many domains have signals
        active_domains = [d for d, s in scores.items() if s > 0]

        if len(active_domains) == 1:
            return active_domains[0]

        # Special cases: "fewest overdue tasks" + "new lead" = multi (pm + crm)
        if scores["project_management"] > 0 and scores["customer_relationship_manager"] > 0:
            if "fewest" in q and ("lead" in q or "assign" in q):
                return None  # multi-domain

        # "email" + "schedule/meeting/calendar" = multi
        if scores["email"] > 0 and scores["calendar"] > 0:
            return None

        # "crm" + "calendar" = multi
        if scores["customer_relationship_manager"] > 0 and scores["calendar"] > 0:
            if "meeting" in q or "schedule" in q or "book" in q:
                return None

        return None  # ambiguous → full orchestrator

    # ------------------------------------------------------------------
    # Worker dispatch
    # ------------------------------------------------------------------

    def _dispatch_worker(
        self,
        worker_name: str,
        task: str,
        blackboard: Blackboard,
    ) -> dict:
        """
        Create a worker, run it with the given task + blackboard context,
        and return its result.
        """
        try:
            worker = get_worker_for_domain(worker_name, model=self.model)
            worker.set_client(self.client)
            blackboard_ctx = blackboard.get_context_string()
            result = worker.run(task, blackboard_context=blackboard_ctx)
            return result
        except ValueError as e:
            return {
                "output": f"Error: {e}",
                "function_calls": [],
                "error": str(e),
            }

    # ------------------------------------------------------------------
    # Orchestrator ReAct loop
    # ------------------------------------------------------------------

    def _run_orchestrator_loop(
        self,
        query: str,
        blackboard: Blackboard,
        all_function_calls: list[str],
        all_responses: list[str],
    ) -> dict:
        """
        Run the orchestrator's own ReAct loop, where "tool calls" are
        worker dispatches.
        """
        messages: list[dict] = [
            {"role": "system", "content": ORCHESTRATOR_SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ]

        summary = ""
        error = ""

        try:
            for iteration in range(self.max_iterations):
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=ORCHESTRATOR_TOOLS,
                    tool_choice="auto",
                    temperature=0.0,
                )

                choice = response.choices[0]
                msg = choice.message

                if msg.tool_calls:
                    # Append assistant message
                    messages.append({
                        "role": "assistant",
                        "content": msg.content or "",
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                },
                            }
                            for tc in msg.tool_calls
                        ],
                    })

                    for tc in msg.tool_calls:
                        func_name = tc.function.name
                        args = json.loads(tc.function.arguments)

                        if func_name == "dispatch_worker":
                            worker_name = args["worker"]
                            task = args["task"]

                            # Actually run the worker
                            worker_result = self._dispatch_worker(
                                worker_name, task, blackboard
                            )

                            # Collect function calls
                            w_fc = worker_result.get("function_calls", [])
                            all_function_calls.extend(w_fc)
                            all_responses.append(
                                f"[{worker_name}] {worker_result.get('output', '')}"
                            )

                            # Write key findings to blackboard
                            output = worker_result.get("output", "")
                            blackboard.write(f"last_worker", worker_name)
                            blackboard.write(f"{worker_name}_result", output)

                            # Feed result back to orchestrator
                            result_str = json.dumps({
                                "worker": worker_name,
                                "findings": output,
                                "actions_taken": w_fc,
                            }, ensure_ascii=False)

                            messages.append({
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "content": result_str,
                            })

                        elif func_name == "finalize":
                            summary = args.get("summary", "Done.")
                            # Append the finalize call to messages and stop
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "content": "Task finalized. Function calls collected.",
                            })
                            break  # exit the tool_calls loop

                    else:
                        continue  # more iterations

                    break  # finalized → exit the for loop

                else:
                    # No tool call — orchestrator responded with text
                    # This shouldn't normally happen, but handle gracefully
                    summary = msg.content or "Task completed."
                    break

            else:
                error = "Orchestrator stopped due to iteration limit."
                summary = error

        except Exception as e:
            error_msg = str(e)
            context_window_keywords = [
                "maximum input length",
                "maximum context length",
                "prompt is too long",
                "Request too large",
                "context_length_exceeded",
            ]
            if any(kw in error_msg for kw in context_window_keywords):
                error = "Context window exceeded"
            else:
                error = f"{type(e).__name__}: {error_msg}"
            summary = error
            traceback.print_exc()

        return {"summary": summary, "error": error}
