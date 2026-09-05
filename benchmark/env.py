"""Shared execution environment + terminal-condition evaluation for the benchmark.

Both harnesses need somewhere to observe the side effects of tool calls:

* the baseline harness uses ``InMemoryEnv`` (a plain dict filesystem) — no
  sandbox, no isolation;
* the full harness uses ``SandboxEnv`` (which wraps ``DockerSandbox``).

``evaluate_terminal`` is the single oracle that decides whether a run satisfied
its task's expected terminal condition, so both harnesses produce comparable
success metrics.
"""

from __future__ import annotations

from typing import Any, Protocol


class Env(Protocol):
    def read_file(self, path: str) -> str: ...

    def list_dir(self, path: str) -> list[str]: ...

    def exists(self, path: str) -> bool: ...


class InMemoryEnv:
    """A minimal in-process filesystem for the baseline harness."""

    def __init__(self) -> None:
        self.files: dict[str, str] = {}
        self.dirs: set[str] = {"/"}

    @staticmethod
    def _norm(path: str) -> str:
        if not path.startswith("/"):
            path = "/" + path
        return "/" + path.strip("/")

    def write(self, path: str, content: str, append: bool = False) -> None:
        key = self._norm(str(path))
        if append and key in self.files:
            self.files[key] += content
        else:
            self.files[key] = content

    def read(self, path: str) -> str:
        key = self._norm(str(path))
        if key not in self.files:
            raise KeyError(key)
        return self.files[key]

    def list(self, path: str) -> list[str]:
        key = self._norm(str(path))
        prefix = "" if key == "/" else key + "/"
        entries = set()
        for f in self.files:
            if f.startswith(prefix):
                rest = f[len(prefix):]
                if rest:
                    entries.add(rest.split("/")[0])
        for d in self.dirs:
            if d.startswith(prefix) and d != key:
                rest = d[len(prefix):]
                if rest:
                    entries.add(rest.split("/")[0])
        return sorted(entries)

    def delete(self, path: str) -> None:
        key = self._norm(str(path))
        for f in [f for f in self.files if f == key or f.startswith(key + "/")]:
            del self.files[f]
        self.dirs = {d for d in self.dirs if d != key and not d.startswith(key + "/")}
        self.dirs.add("/")

    def mkdir(self, path: str) -> None:
        self.dirs.add(self._norm(str(path)))

    # -- Env protocol --------------------------------------------------------

    def read_file(self, path: str) -> str:
        try:
            return self.read(path)
        except KeyError:
            return ""

    def list_dir(self, path: str) -> list[str]:
        return self.list(path)

    def exists(self, path: str) -> bool:
        key = self._norm(str(path))
        return key in self.files or key in self.dirs


class SandboxEnv:
    """Wraps a ``DockerSandbox`` as an ``Env`` (filesystem inspection only)."""

    def __init__(self, sandbox: Any) -> None:
        self.sandbox = sandbox

    def read_file(self, path: str) -> str:
        try:
            return self.sandbox.read_file(path)
        except Exception:  # noqa: BLE001 — missing file is "empty" for the oracle
            return ""

    def list_dir(self, path: str) -> list[str]:
        try:
            return self.sandbox.list_dir(path)
        except Exception:  # noqa: BLE001
            return []

    def exists(self, path: str) -> bool:
        return bool(self.sandbox.exists(path))


def _eq(a: Any, b: Any) -> bool:
    try:
        return float(a) == float(b)
    except (TypeError, ValueError):
        return a == b


def evaluate_terminal(
    terminal: dict[str, Any],
    env: Env,
    results: list[dict[str, Any]],
    errors: list[dict[str, Any]],
) -> bool:
    """Evaluate a task's expected terminal condition."""
    ttype = terminal["type"]
    if ttype == "file_exists":
        return env.exists(terminal["path"])
    if ttype == "file_absent":
        return not env.exists(terminal["path"])
    if ttype == "file_contains":
        return terminal["content"] in env.read_file(terminal["path"])
    if ttype == "result_equals":
        compute_results = [r for r in results if r.get("name") == "compute"]
        if not compute_results:
            return False
        return _eq(compute_results[-1].get("output"), terminal["value"])
    if ttype == "tool_succeeded":
        return any(r.get("name") == terminal["tool"] for r in results)
    if ttype == "tool_denied":
        denied = any(e.get("name") == terminal["tool"] for e in errors)
        succeeded = any(r.get("name") == terminal["tool"] for r in results)
        return denied and not succeeded
    raise ValueError(f"unknown terminal type {ttype!r}")
