"""
Base Agent — generic ReAct loop with OpenAI function calling.

Provides the core building block for both Domain Workers and the
Orchestrator. Each BaseAgent instance:
  1. Takes a system prompt + tools (OpenAI function schemas + callables)
  2. Runs a ReAct loop: think → act → observe → repeat
  3. Returns structured results including function calls and output
"""

import json
import inspect
import re
import traceback
from typing import Any, Callable, Optional

from openai import OpenAI


# ---------------------------------------------------------------------------
# Tool → OpenAI schema conversion
# ---------------------------------------------------------------------------

def _parse_numpy_docstring(doc: str) -> tuple[str, dict[str, str]]:
    """
    Parse a NumPy-style docstring into a short description and a map of
    parameter name → description.
    """
    if not doc:
        return "", {}

    lines = doc.strip().split("\n")
    short_desc = lines[0].strip()

    param_descs: dict[str, str] = {}
    current_param: Optional[str] = None
    current_desc: list[str] = []

    for line in lines[1:]:
        stripped = line.strip()
        # Match "name : type" or "name: type" parameter lines
        param_match = re.match(r"^(\w+)\s*:\s*\w+", stripped)
        if param_match:
            if current_param:
                param_descs[current_param] = " ".join(current_desc).strip()
            current_param = param_match.group(1)
            # Description may follow the type on the same line
            rest = re.sub(r"^\w+\s*:\s*\w+\s*,?\s*(optional)?\s*", "", stripped)
            current_desc = [rest] if rest else []
        elif stripped.startswith("Returns") or stripped.startswith("Examples"):
            if current_param:
                param_descs[current_param] = " ".join(current_desc).strip()
            break
        elif current_param and stripped:
            current_desc.append(stripped)

    if current_param:
        param_descs[current_param] = " ".join(current_desc).strip()

    return short_desc, param_descs


def tool_to_openai_schema(tool) -> dict:
    """
    Convert a LangChain @tool decorated function into an OpenAI function
    calling schema (tools array entry).

    Parameters
    ----------
    tool : LangChain tool
        A @tool-decorated function with .name, .func, and .description attributes.

    Returns
    -------
    dict
        OpenAI tool schema: {"type": "function", "function": {...}}
    """
    func = tool.func if hasattr(tool, "func") else tool
    sig = inspect.signature(func)
    doc = func.__doc__ or ""

    short_desc, param_descs = _parse_numpy_docstring(doc)

    # If the docstring parser didn't catch params, try a simpler heuristic
    if not param_descs:
        for param_name in sig.parameters:
            # Try to find a line like "param_name : type\n        Description..."
            pattern = rf"{param_name}\s*:.*?\n\s*(.*?)(?:\n|$)"
            match = re.search(pattern, doc)
            if match:
                param_descs[param_name] = match.group(1).strip()

    properties: dict[str, dict] = {}
    required: list[str] = []

    for param_name, param in sig.parameters.items():
        # Determine JSON Schema type
        annotation = param.annotation
        json_type = "string"
        if annotation != inspect.Parameter.empty:
            origin = getattr(annotation, "__origin__", None)
            if origin is not None:
                # Handle Optional[X] etc
                args = getattr(annotation, "__args__", ())
                non_none = [a for a in args if a is not type(None)]  # noqa: E721
                if non_none:
                    json_type = _python_type_to_json(non_none[0])
            else:
                json_type = _python_type_to_json(annotation)

        desc = param_descs.get(param_name, f"Parameter: {param_name}")
        properties[param_name] = {
            "type": json_type,
            "description": desc,
        }

        # Parameters without defaults are required
        if param.default is inspect.Parameter.empty:
            required.append(param_name)

    # Use the tool's own name (e.g. "calendar.create_event")
    name = tool.name if hasattr(tool, "name") else func.__name__

    return {
        "type": "function",
        "function": {
            "name": name,
            "description": short_desc or f"Call {name}",
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


def _python_type_to_json(py_type: type) -> str:
    """Map a Python type to a JSON Schema type string."""
    if py_type in (int,):
        return "integer"
    elif py_type in (float,):
        return "number"
    elif py_type in (bool,):
        return "boolean"
    elif py_type in (list, dict):
        return "object"
    return "string"


# ---------------------------------------------------------------------------
# Base Agent (ReAct loop)
# ---------------------------------------------------------------------------

class BaseAgent:
    """
    A generic ReAct agent powered by OpenAI function calling.

    The agent receives a task and iteratively:
      1. Asks the LLM what to do (may include tool calls)
      2. Executes any tool calls against real Python functions
      3. Feeds results back to the LLM
      4. Repeats until the LLM responds with a text message (no tool calls)

    Parameters
    ----------
    name : str
        Human-readable name for logging / debugging.
    model : str
        OpenAI model name (e.g. "gpt-4-0125-preview", "gpt-3.5-turbo-instruct").
    tools : list[dict]
        Each dict: {"schema": <openai_tool_schema>, "callable": <python_function>}
    system_prompt : str
        System prompt that defines the agent's role and behavior.
    max_iterations : int
        Maximum ReAct loop iterations (default 15).
    temperature : float
        LLM temperature (default 0).
    """

    def __init__(
        self,
        name: str,
        model: str,
        tools: list[dict],
        system_prompt: str,
        max_iterations: int = 15,
        temperature: float = 0.0,
    ):
        self.name = name
        self.model = model
        self.tools = tools
        self.system_prompt = system_prompt
        self.max_iterations = max_iterations
        self.temperature = temperature

        # Build lookup: function_name → callable
        self._tool_map: dict[str, Callable] = {}
        for t in tools:
            func_name = t["schema"]["function"]["name"]
            self._tool_map[func_name] = t["callable"]

        # Build OpenAI tool schemas list
        self._openai_tools = [t["schema"] for t in tools]

        # Will be set by runner
        self._client: Optional[OpenAI] = None

    def set_client(self, client: OpenAI) -> None:
        """Set the OpenAI client (set by the runner before use)."""
        self._client = client

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            raise RuntimeError("OpenAI client not set. Call set_client() first.")
        return self._client

    def run(
        self,
        user_message: str,
        blackboard_context: str = "",
    ) -> dict:
        """
        Execute the ReAct loop.

        Parameters
        ----------
        user_message : str
            The task / subtask for this agent to perform.
        blackboard_context : str
            Optional context from the Blackboard (previous worker results).

        Returns
        -------
        dict with:
            - "output" : str          — final text response from the agent
            - "function_calls" : list[str] — collected tool calls as "domain.func(args)" strings
            - "error" : str           — empty if successful, error message otherwise
        """
        messages: list[dict] = [
            {"role": "system", "content": self.system_prompt},
        ]

        # Inject blackboard context if provided
        user_content = user_message
        if blackboard_context:
            user_content = f"{blackboard_context}\n\n---\n\n{user_message}"

        messages.append({"role": "user", "content": user_content})

        function_calls: list[str] = []
        final_output = ""
        error = ""

        openai_tools = self._openai_tools if self._openai_tools else None

        try:
            for iteration in range(self.max_iterations):
                kwargs: dict = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": self.temperature,
                }
                if openai_tools:
                    kwargs["tools"] = openai_tools
                    kwargs["tool_choice"] = "auto"

                response = self.client.chat.completions.create(**kwargs)
                choice = response.choices[0]
                msg = choice.message

                # If the model makes tool calls
                if msg.tool_calls:
                    # Append assistant message with tool calls
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
                        func_args_str = tc.function.arguments

                        # Parse arguments
                        try:
                            func_args = json.loads(func_args_str)
                        except json.JSONDecodeError:
                            func_args = {}

                        # Call the tool
                        callable_fn = self._tool_map.get(func_name)
                        if callable_fn:
                            try:
                                result = callable_fn(**func_args)
                                result_str = json.dumps(result, ensure_ascii=False, default=str)
                            except Exception as e:
                                result_str = f"Error executing {func_name}: {e}"
                        else:
                            result_str = f"Unknown function: {func_name}"

                        # Record the function call for evaluation
                        call_str = _format_function_call(func_name, func_args)
                        function_calls.append(call_str)

                        # Feed tool result back
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": result_str,
                        })

                    continue  # back to the LLM for the next turn

                # No tool calls → final response
                final_output = msg.content or ""
                break  # ReAct loop done

            else:
                # Max iterations exceeded
                final_output = "Agent stopped due to iteration limit."
                error = "Agent stopped due to iteration limit or time limit."

        except Exception as e:
            error_msg = str(e)
            # Detect context-window errors across different providers
            context_window_keywords = [
                "maximum input length",
                "maximum context length",
                "prompt is too long",
                "Request too large",
                "context_length_exceeded",
                "reduce the length",
            ]
            if any(kw in error_msg for kw in context_window_keywords):
                error = "Context window exceeded"
            else:
                error = f"{type(e).__name__}: {error_msg}"
            final_output = error
            traceback.print_exc()

        return {
            "output": final_output,
            "function_calls": function_calls,
            "error": error,
        }


def _format_function_call(func_name: str, args: dict) -> str:
    """
    Format a function call as a string matching the WorkBench convention.

    Example: calendar.create_event(event_name="X", event_start="2023-12-01 13:00:00", ...)
    """
    arg_parts = []
    for k, v in args.items():
        # Escape internal double quotes in values
        v_str = str(v).replace('"', '\\"')
        arg_parts.append(f'{k}="{v_str}"')
    return f"{func_name}.func(" + ", ".join(arg_parts) + ")"
