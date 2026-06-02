"""
Blackboard — shared memory for inter-agent communication.

The Orchestrator writes intermediate results here, and downstream
Workers read them to get context from previous steps. This is the
primary mechanism for handling multi-domain task dependencies.

Example workflow for "Find email from X and schedule meeting with sender":
  1. EmailWorker searches → writes {sender_email: "foo@atlas.com"}
  2. CalendarWorker reads sender_email → creates meeting
"""

from typing import Any, Optional


class Blackboard:
    """
    A simple key-value store for sharing data between agents.

    Used by the Orchestrator to pass context between Workers:
    - Worker A discovers some information → Orchestrator writes to Blackboard
    - Worker B needs that information → Orchestrator passes it in the task
    """

    def __init__(self):
        self._store: dict[str, Any] = {}

    def write(self, key: str, value: Any) -> None:
        """Write a value to the blackboard."""
        self._store[key] = value

    def read(self, key: str, default: Any = None) -> Any:
        """Read a value from the blackboard. Returns default if key not found."""
        return self._store.get(key, default)

    def get_all(self) -> dict[str, Any]:
        """Return all entries on the blackboard."""
        return dict(self._store)

    def get_context_string(self) -> str:
        """
        Build a human-readable summary of the blackboard contents.
        Returns an empty string if the blackboard is empty.
        """
        if not self._store:
            return ""
        lines = ["[Context from previous steps]"]
        for key, value in self._store.items():
            lines.append(f"  {key}: {value}")
        return "\n".join(lines)

    def clear(self) -> None:
        """Clear all entries from the blackboard."""
        self._store.clear()

    def __repr__(self) -> str:
        return f"Blackboard({self._store})"
