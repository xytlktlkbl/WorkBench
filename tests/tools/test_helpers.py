from typing import Any, Callable


def get_function_from_tool(tool: Any) -> Callable[..., Any]:
    return getattr(tool, "func")
