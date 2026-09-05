"""Tool definitions shared by both harnesses.

Only the *schemas* (MCP ``inputSchema``), risk tiers, and network flags are
shared. The *handlers* differ:

* ``make_handlers(sandbox)`` — full-harness handlers that execute every side
  effect *inside the sandbox* (``DockerSandbox.execute_tool``), returning
  structured ``ToolCallError`` for timeout / network-deny / execution failures.
* ``make_bare_handlers(env)`` — baseline handlers that mutate an ``InMemoryEnv``
  directly, with no sandbox, no schema gating, and no structured errors.
"""

from __future__ import annotations

import shlex
from typing import Any, Callable

from mcp.types import Tool

from forgeharness.core.permission_engine import RiskTier
from forgeharness.core.tool_registry import ToolCallError, ToolRegistry
from forgeharness.sandbox.docker_executor import NetworkDeniedError, ToolTimeoutError

# -- schemas / metadata ------------------------------------------------------


def _obj(props: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": props,
        "required": required,
        "additionalProperties": False,
    }


def _schema(tool_name: str, props: dict[str, Any], required: list[str]) -> Tool:
    return Tool(name=tool_name, inputSchema=_obj(props, required))


TOOLS: dict[str, Tool] = {
    "write_file": _schema(
        "write_file",
        {"path": {"type": "string"}, "content": {"type": "string"}},
        ["path", "content"],
    ),
    "read_file": _schema("read_file", {"path": {"type": "string"}}, ["path"]),
    "append_file": _schema(
        "append_file",
        {"path": {"type": "string"}, "content": {"type": "string"}},
        ["path", "content"],
    ),
    "list_dir": _schema("list_dir", {"path": {"type": "string"}}, ["path"]),
    "make_dir": _schema("make_dir", {"path": {"type": "string"}}, ["path"]),
    "delete_file": _schema("delete_file", {"path": {"type": "string"}}, ["path"]),
    "compute": _schema("compute", {"expression": {"type": "string"}}, ["expression"]),
    "http_get": _schema("http_get", {"url": {"type": "string"}}, ["url"]),
    "run_shell": _schema("run_shell", {"command": {"type": "string"}}, ["command"]),
}

RISK_TIERS: dict[str, RiskTier] = {
    "write_file": RiskTier.WRITE,
    "read_file": RiskTier.READ_ONLY,
    "append_file": RiskTier.WRITE,
    "list_dir": RiskTier.READ_ONLY,
    "make_dir": RiskTier.WRITE,
    "delete_file": RiskTier.DESTRUCTIVE,
    "compute": RiskTier.READ_ONLY,
    "http_get": RiskTier.EXTERNAL_SIDE_EFFECT,
    "run_shell": RiskTier.WRITE,
}

REQUIRES_NETWORK: set[str] = {"http_get"}


# -- full-harness handlers (sandboxed) ---------------------------------------


def _parse_number(text: str) -> Any:
    text = text.strip()
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def make_handlers(sandbox: Any) -> dict[str, Callable[..., Any]]:
    def write_file(path: str, content: str) -> Any:
        sandbox.execute_tool("write_file", ["sh", "-c", f"cat > {shlex.quote(path)}"], stdin=content)
        return content

    def read_file(path: str) -> Any:
        result = sandbox.execute_tool("read_file", ["cat", path])
        if not result.ok:
            return ToolCallError(name="read_file", error_type="execution_error", message=result.stderr)
        return result.stdout

    def append_file(path: str, content: str) -> Any:
        sandbox.execute_tool("append_file", ["sh", "-c", f"cat >> {shlex.quote(path)}"], stdin=content)
        return content

    def list_dir(path: str) -> Any:
        result = sandbox.execute_tool("list_dir", ["ls", "-1", path])
        if not result.ok:
            return ToolCallError(name="list_dir", error_type="execution_error", message=result.stderr)
        return result.stdout

    def make_dir(path: str) -> Any:
        sandbox.execute_tool("make_dir", ["mkdir", "-p", path])
        return path

    def delete_file(path: str) -> Any:
        sandbox.execute_tool("delete_file", ["rm", "-rf", path])
        return path

    def compute(expression: str) -> Any:
        result = sandbox.execute_tool("compute", ["python3", "-c", f"print({expression})"])
        if not result.ok:
            return ToolCallError(name="compute", error_type="execution_error", message=result.stderr)
        return _parse_number(result.stdout)

    def http_get(url: str) -> Any:
        try:
            result = sandbox.execute_tool("http_get", ["curl", "-s", url], requires_network=True)
        except NetworkDeniedError as exc:
            return ToolCallError(name="http_get", error_type="network_denied", message=str(exc))
        return result.stdout

    def run_shell(command: str) -> Any:
        try:
            result = sandbox.execute_tool("run_shell", ["sh", "-c", command])
        except ToolTimeoutError as exc:
            return ToolCallError(name="run_shell", error_type="timeout", message=str(exc))
        except NetworkDeniedError as exc:
            return ToolCallError(name="run_shell", error_type="network_denied", message=str(exc))
        if not result.ok:
            error_type = "network_denied" if "network" in result.stderr else "execution_error"
            return ToolCallError(name="run_shell", error_type=error_type, message=result.stderr.strip())
        return result.stdout

    return {
        "write_file": write_file,
        "read_file": read_file,
        "append_file": append_file,
        "list_dir": list_dir,
        "make_dir": make_dir,
        "delete_file": delete_file,
        "compute": compute,
        "http_get": http_get,
        "run_shell": run_shell,
    }


def build_registry(sandbox: Any) -> ToolRegistry:
    registry = ToolRegistry()
    for name, handler in make_handlers(sandbox).items():
        registry.register(name, TOOLS[name], handler, risk_tier=RISK_TIERS[name])
    return registry


# -- baseline handlers (bare, unsandboxed) -----------------------------------


def _safe_eval(expression: str) -> Any:
    safe = {
        "abs": abs,
        "len": len,
        "sum": sum,
        "min": min,
        "max": max,
        "range": range,
        "int": int,
        "float": float,
        "str": str,
        "list": list,
        "dict": dict,
        "sorted": sorted,
        "round": round,
    }
    return eval(expression, {"__builtins__": safe}, {})  # noqa: S307 — baseline


def make_bare_handlers(env: Any) -> dict[str, Callable[..., Any]]:
    def write_file(path: str, content: str) -> Any:
        env.write(path, content)
        return content

    def read_file(path: str) -> Any:
        return env.read(path)

    def append_file(path: str, content: str) -> Any:
        env.write(path, content, append=True)
        return content

    def list_dir(path: str) -> Any:
        return "\n".join(env.list(path))

    def make_dir(path: str) -> Any:
        env.mkdir(path)
        return path

    def delete_file(path: str) -> Any:
        env.delete(path)
        return path

    def compute(expression: str) -> Any:
        return _safe_eval(expression)

    def http_get(url: str) -> Any:
        return f"<fake body for {url}>"

    def run_shell(command: str) -> Any:
        # The bare loop has no sandbox; simulate un-isolated execution.
        if command.startswith("sleep") or command == "hang":
            return ""
        if command.startswith("curl"):
            return "<fake network data>"
        return ""

    return {
        "write_file": write_file,
        "read_file": read_file,
        "append_file": append_file,
        "list_dir": list_dir,
        "make_dir": make_dir,
        "delete_file": delete_file,
        "compute": compute,
        "http_get": http_get,
        "run_shell": run_shell,
    }


def schema_valid(tool_name: str, args: dict[str, Any]) -> bool:
    """Schema oracle used by the baseline's metering (not for gating)."""
    import jsonschema

    tool = TOOLS.get(tool_name)
    if tool is None:
        return False
    validator = jsonschema.validators.validator_for(tool.input_schema)(tool.input_schema)
    return not list(validator.iter_errors(args))


__all__ = [
    "TOOLS",
    "RISK_TIERS",
    "REQUIRES_NETWORK",
    "make_handlers",
    "make_bare_handlers",
    "build_registry",
    "schema_valid",
]
