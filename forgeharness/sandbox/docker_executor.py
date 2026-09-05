"""Docker-sandboxed tool execution, behind a swappable backend interface.

Tool calls with side effects (filesystem, subprocess, network) execute inside a
sandbox rather than the host process. Because Docker is *not* available in every
environment (including CI and the local test suite), the executor is split into
two layers:

* ``SandboxBackend`` — the abstract contract (``start`` / ``stop`` / ``run``).
* ``FakeBackend`` — an in-process implementation with a simulated filesystem,
  a small POSIX-ish command interpreter, simulated resource limits, and
  deterministic timeout behaviour. Used by the unit tests and ``--smoke``.
* ``DockerBackend`` — a real implementation that drives ``docker`` (via
  ``subprocess``, no SDK dependency) to run one container per session.
* ``DockerSandbox`` — the facade the harness talks to. It owns the session
  lifecycle, enforces the network default-deny/allowlist policy, applies the
  hard *per tool call* timeout, and converts timeout into the distinct
  ``ToolTimeoutError`` (never conflated with tool-logic errors).

Container lifecycle decision: **per-session**. One container (plus its scratch
volume) is started at the beginning of a session and torn down — with the scratch
volume wiped — at the end. Justification:

* A multi-step task needs intermediate artifacts to persist *across* tool calls
  (write, then read, then grep). Per-call isolation would wipe the filesystem
  between every step and make those tasks impossible without a separate volume
  dance.
* Per-call startup would pay container cold-start latency on every single tool
  call, which dominates runtime in a 150-run eval.
* The safety guarantees do not depend on re-establishing isolation per call:
  network is default-deny for the container's entire lifetime, resource limits
  (CPU/memory/pids) are enforced at the container level, and every tool call is
  still hard-time-boxed independently. Per-call isolation buys nothing here that
  these three controls don't already provide.
"""

from __future__ import annotations

import shlex
import shutil
import subprocess
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #


class SandboxError(Exception):
    """Base class for all sandbox failures."""


class ToolTimeoutError(SandboxError):
    """A *distinct* error for the hard per-tool-call timeout.

    Deliberately not a subclass of ``ToolCallError`` (that lives in ``core`` and
    means "bad arguments / unknown tool"). A timeout is a resource/execution
    failure, never a tool-logic failure, so it gets its own type so callers can
    tell the two apart unambiguously.
    """

    def __init__(self, tool_name: str, timeout: float) -> None:
        self.tool_name = tool_name
        self.timeout = timeout
        super().__init__(
            f"tool {tool_name!r} exceeded its hard per-call timeout of {timeout}s"
        )


class SandboxExecutionError(SandboxError):
    """A non-timeout failure of the sandbox itself (container errors, etc.)."""


class NetworkDeniedError(SandboxError):
    """Raised when a tool attempts network access that is not allowlisted."""

    def __init__(self, tool_name: str) -> None:
        self.tool_name = tool_name
        super().__init__(
            f"network access denied for tool {tool_name!r}: default-deny, "
            f"no allowlist entry"
        )


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


class ResourceLimits(BaseModel):
    """Per-session CPU/memory/pid limits."""

    model_config = ConfigDict(extra="forbid")

    cpu: float = 1.0
    memory: str = "512m"
    pids: int = 128


class SandboxConfig(BaseModel):
    """Everything needed to run one sandboxed session."""

    model_config = ConfigDict(extra="forbid")

    image: str = "python:3.11-slim"
    workdir: str = "/workspace"
    resource_limits: ResourceLimits = Field(default_factory=ResourceLimits)
    default_timeout: float = 30.0
    network_allowed: bool = False
    network_allowlist: dict[str, list[str]] = Field(default_factory=dict)
    scratch_dir: Optional[str] = None
    read_only_rootfs: bool = True
    session_id: str = "default"


class ExecutionResult(BaseModel):
    """Outcome of a single command inside the sandbox."""

    model_config = ConfigDict(extra="forbid")

    exit_code: int = 0
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    duration_ms: float = 0.0

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out


# --------------------------------------------------------------------------- #
# Backend interface
# --------------------------------------------------------------------------- #


class SandboxBackend(ABC):
    """Abstract backend contract.

    Implementations must provide a session-scoped sandbox: ``start`` provisions
    it (container + scratch volume), ``run`` executes one command with a hard
    timeout, and ``stop`` tears it down and wipes the scratch volume.
    """

    name: str = "abstract"

    @abstractmethod
    def start(self) -> None:
        """Provision the sandbox (container + scratch volume)."""

    @abstractmethod
    def stop(self) -> None:
        """Tear the sandbox down and wipe the scratch volume."""

    @abstractmethod
    def run(
        self,
        argv: list[str],
        *,
        timeout: float,
        workdir: Optional[str] = None,
        env: Optional[dict[str, str]] = None,
        stdin: Optional[str] = None,
        allow_network: bool = False,
    ) -> ExecutionResult:
        """Execute ``argv`` inside the sandbox with a hard timeout."""

    @property
    @abstractmethod
    def available(self) -> bool:
        """Whether this backend can actually run in the current environment."""

    def __enter__(self) -> "SandboxBackend":
        self.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.stop()


# --------------------------------------------------------------------------- #
# Fake (in-process) backend
# --------------------------------------------------------------------------- #

_NETWORK_COMMANDS = ("curl", "wget", "http", "https", "nc", "ftp")


class FakeBackend(SandboxBackend):
    """In-process sandbox used by tests and ``--smoke`` (no Docker required).

    It simulates:

    * a filesystem (``dict[str, str]`` of path -> content, plus a directory set),
    * a small command interpreter for the exact command vocabulary the benchmark
      tools emit (``cat``, ``echo``, ``ls``, ``rm``, ``mkdir``, ``python3 -c``,
      ``sh -c``, ``sleep``),
    * network default-deny (``curl``/``wget``/etc. raise unless allowlisted),
    * deterministic hard timeouts (``sleep N`` with ``N > timeout`` and the
      special ``hang`` command always time out).

    ``snapshot`` / ``load_snapshot`` let the harness persist the simulated
    scratch volume so an interrupted run can be restored (mirroring the real
    Docker scratch volume surviving a process kill).
    """

    name = "fake"

    def __init__(self, config: Optional[SandboxConfig] = None) -> None:
        self.config = config or SandboxConfig()
        self._files: dict[str, str] = {}
        self._dirs: set[str] = {"/"}
        self._started = False
        self._started_with: Optional[SandboxConfig] = None
        self._simulated: float = 0.0

    # -- lifecycle ----------------------------------------------------------

    @property
    def available(self) -> bool:
        return True

    def start(self) -> None:
        self._started = True
        self._started_with = self.config

    def stop(self) -> None:
        self._files.clear()
        self._dirs = {"/"}
        self._started = False

    # -- state persistence (simulated scratch volume) ------------------------

    def snapshot(self) -> dict[str, Any]:
        return {"files": dict(self._files), "dirs": sorted(self._dirs)}

    def load_snapshot(self, state: dict[str, Any]) -> None:
        self._files = dict(state.get("files", {}))
        self._dirs = set(state.get("dirs", ["/"]))

    # -- filesystem helpers --------------------------------------------------

    @staticmethod
    def _norm(path: str) -> str:
        path = path or "/"
        if not path.startswith("/"):
            path = "/" + path
        parts = []
        for part in path.split("/"):
            if part in ("", "."):
                continue
            if part == "..":
                if parts:
                    parts.pop()
                continue
            parts.append(part)
        return "/" + "/".join(parts) if parts else "/"

    def _parent(self, path: str) -> str:
        norm = self._norm(path)
        parent = norm.rsplit("/", 1)[0]
        return parent or "/"

    def _mkdir_p(self, path: str) -> None:
        norm = self._norm(path)
        if norm == "/":
            return
        self._dirs.add(norm)
        self._mkdir_p(self._parent(norm))

    def _exists(self, path: str) -> bool:
        norm = self._norm(path)
        return norm in self._files or norm in self._dirs

    def _read(self, path: str) -> str:
        norm = self._norm(path)
        if norm in self._files:
            return self._files[norm]
        if norm in self._dirs:
            return ""  # `cat` on a directory yields nothing in this simulator
        raise SandboxExecutionError(f"cat: {norm}: No such file or directory")

    def _write(self, path: str, content: str, append: bool = False) -> None:
        norm = self._norm(path)
        self._mkdir_p(self._parent(norm))
        if append and norm in self._files:
            self._files[norm] += content
        else:
            self._files[norm] = content

    def _delete(self, path: str) -> None:
        norm = self._norm(path)
        for f in [f for f in self._files if f == norm or f.startswith(norm + "/")]:
            del self._files[f]
        self._dirs = {d for d in self._dirs if d != norm and not d.startswith(norm + "/")}
        self._dirs.add("/")

    def _list(self, path: str) -> list[str]:
        norm = self._norm(path)
        if norm not in self._dirs:
            raise SandboxExecutionError(f"ls: {norm}: No such file or directory")
        entries: set[str] = set()
        prefix = "" if norm == "/" else norm + "/"
        for f in self._files:
            if f.startswith(prefix):
                rest = f[len(prefix):]
                entries.add(rest.split("/")[0])
        for d in self._dirs:
            if d == norm or not d.startswith(prefix):
                continue
            rest = d[len(prefix):]
            if rest:
                entries.add(rest.split("/")[0])
        return sorted(entries)

    # -- command interpreter --------------------------------------------------

    def _run_python(self, code: str) -> tuple[int, str, str]:
        import contextlib
        import io

        buf = io.StringIO()
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
            "print": print,
        }
        try:
            with contextlib.redirect_stdout(buf):
                exec(code, {"__builtins__": safe}, {})
        except Exception as exc:  # noqa: BLE001 — simulator, surface as stderr
            return 1, "", f"{type(exc).__name__}: {exc}"
        return 0, buf.getvalue(), ""

    def _dispatch(
        self,
        argv: list[str],
        *,
        stdin: Optional[str],
        allow_network: bool,
    ) -> tuple[int, str, str]:
        if not argv:
            return 0, "", ""
        cmd, args = argv[0], argv[1:]

        if cmd == "cat":
            if not args:
                return 0, stdin or "", ""
            try:
                return 0, "".join(self._read(a) for a in args), ""
            except SandboxExecutionError as exc:
                return 1, "", str(exc)
        if cmd == "echo":
            return 0, " ".join(args) + "\n", ""
        if cmd == "printf":
            return 0, " ".join(args), ""
        if cmd == "ls":
            args = [a for a in args if not a.startswith("-")]
            path = args[0] if args else "/"
            try:
                return 0, "\n".join(self._list(path)) + "\n", ""
            except SandboxExecutionError as exc:
                return 1, "", str(exc)
        if cmd == "mkdir":
            for a in args:
                if not a.startswith("-"):
                    self._mkdir_p(a)
            return 0, "", ""
        if cmd == "rm":
            for a in args:
                if not a.startswith("-"):
                    self._delete(a)
            return 0, "", ""
        if cmd == "touch":
            for a in args:
                if not a.startswith("-"):
                    self._write(a, "")
            return 0, "", ""
        if cmd == "python3":
            for i, a in enumerate(args):
                if a == "-c" and i + 1 < len(args):
                    code, out, err = self._run_python(args[i + 1])
                    return code, out, err
            return 1, "", "python3: missing -c"
        if cmd == "sh":
            if args and args[0] == "-c":
                return self._run_script(args[1] if len(args) > 1 else "", stdin=stdin, allow_network=allow_network)
            return 1, "", "sh: only -c is supported"
        if cmd == "sleep":
            try:
                self._simulated += float(args[0]) if args else 0.0
            except ValueError:
                pass
            return 0, "", ""
        if cmd == "hang":
            self._simulated = float("inf")
            return 0, "", ""
        if cmd in _NETWORK_COMMANDS:
            if allow_network:
                return 0, "network-ok\n", ""
            return 1, "", f"{cmd}: network access denied (default-deny)"
        return 127, "", f"{cmd}: command not found"

    def _run_script(self, script: str, *, stdin: Optional[str], allow_network: bool) -> tuple[int, str, str]:
        """Mini interpreter for ``sh -c`` supporting ``>``/``>>`` and ``&&``/``;``."""
        out_acc = ""
        # split on top-level `&&` and `;`
        segments: list[str] = []
        current = ""
        i = 0
        while i < len(script):
            ch = script[i]
            if ch == "&" and i + 1 < len(script) and script[i + 1] == "&":
                segments.append(current)
                current = ""
                i += 2
                continue
            if ch == ";":
                segments.append(current)
                current = ""
                i += 1
                continue
            current += ch
            i += 1
        segments.append(current)

        last_code = 0
        for seg in segments:
            seg = seg.strip()
            if not seg:
                continue
            redirect = None
            redirect_append = False
            for op in (">>", ">"):
                idx = seg.find(op)
                if idx != -1:
                    redirect = seg[idx + len(op):].strip()
                    redirect_append = op == ">>"
                    seg = seg[:idx].strip()
                    break
            cmd_args = shlex.split(seg)
            if not cmd_args:
                continue
            code, out, err = self._dispatch(cmd_args, stdin=stdin, allow_network=allow_network)
            if code != 0:
                return code, out_acc + out, err
            if redirect is not None:
                self._write(redirect, out, append=redirect_append)
                out = ""
            out_acc += out
            last_code = code
        return last_code, out_acc, ""

    # -- backend.run ----------------------------------------------------------

    def run(
        self,
        argv: list[str],
        *,
        timeout: float,
        workdir: Optional[str] = None,
        env: Optional[dict[str, str]] = None,
        stdin: Optional[str] = None,
        allow_network: bool = False,
    ) -> ExecutionResult:
        start = time.monotonic()
        self._simulated = 0.0

        code, out, err = self._dispatch(argv, stdin=stdin, allow_network=allow_network)
        elapsed = time.monotonic() - start
        if self._simulated > timeout or elapsed > timeout:
            return ExecutionResult(
                exit_code=124, timed_out=True, duration_ms=timeout * 1000,
                stderr=err or "timed out",
            )
        return ExecutionResult(
            exit_code=code, stdout=out, stderr=err, duration_ms=elapsed * 1000
        )


# --------------------------------------------------------------------------- #
# Real Docker backend
# --------------------------------------------------------------------------- #


class DockerBackend(SandboxBackend):
    """Real Docker backend: one container per session, driven via ``docker`` CLI.

    Lifecycle (per-session):

    * ``start`` creates a host scratch directory, then ``docker run -d`` an
      idle container with ``--network none``, CPU/memory/pids limits, a
      read-only rootfs, and the scratch directory mounted at ``workdir``.
    * ``run`` executes one command via ``docker exec`` with a hard timeout
      enforced by ``subprocess.run(..., timeout=...)``.
    * ``stop`` removes the container and wipes the scratch directory.

    Network allowlisting is enforced at the ``DockerSandbox`` policy layer
    (default-deny); the container itself runs with ``--network none`` as
    defense-in-depth.
    """

    name = "docker"

    def __init__(
        self,
        config: Optional[SandboxConfig] = None,
        *,
        docker_cmd: str = "docker",
        runner: Optional[Any] = None,
    ) -> None:
        self.config = config or SandboxConfig()
        self.docker_cmd = docker_cmd
        self._runner = runner or subprocess.run
        self._container_name = f"forgeharness-{self.config.session_id}"
        self._scratch: Optional[Path] = None

    # -- helpers --------------------------------------------------------------

    @property
    def available(self) -> bool:
        try:
            result = self._runner(
                [self.docker_cmd, "info"], capture_output=True, timeout=5
            )
            return result.returncode == 0
        except Exception:  # noqa: BLE001 — availability check must never raise
            return False

    def _scratch_path(self) -> Path:
        if self.config.scratch_dir:
            return Path(self.config.scratch_dir)
        return Path(tempfile_mkdtemp())

    # -- lifecycle ------------------------------------------------------------

    def start(self) -> None:
        limits = self.config.resource_limits
        self._scratch = self._scratch_path()
        self._scratch.mkdir(parents=True, exist_ok=True)

        argv = [
            self.docker_cmd, "run", "-d",
            "--name", self._container_name,
            "--network", "none",
            "--cpus", str(limits.cpu),
            "--memory", limits.memory,
            "--pids-limit", str(limits.pids),
            "-v", f"{self._scratch}:{self.config.workdir}",
            "-w", self.config.workdir,
        ]
        if self.config.read_only_rootfs:
            argv.append("--read-only")
        argv += [self.config.image, "sleep", "infinity"]

        result = self._runner(argv, capture_output=True, timeout=30)
        if result.returncode != 0:
            raise SandboxExecutionError(
                f"docker run failed: {(result.stderr or b'').decode(errors='replace')}"
            )

    def stop(self) -> None:
        try:
            self._runner(
                [self.docker_cmd, "rm", "-f", self._container_name],
                capture_output=True,
                timeout=30,
            )
        finally:
            if self._scratch is not None and self._scratch.exists():
                shutil.rmtree(self._scratch, ignore_errors=True)
            self._scratch = None

    def run(
        self,
        argv: list[str],
        *,
        timeout: float,
        workdir: Optional[str] = None,
        env: Optional[dict[str, str]] = None,
        stdin: Optional[str] = None,
        allow_network: bool = False,
    ) -> ExecutionResult:
        start = time.monotonic()
        cmd = [self.docker_cmd, "exec", "-i"]
        if workdir is not None:
            cmd += ["-w", workdir]
        for key, value in (env or {}).items():
            cmd += ["-e", f"{key}={value}"]
        cmd += [self._container_name, *argv]

        try:
            result = self._runner(
                cmd,
                input=stdin.encode() if stdin is not None else None,
                capture_output=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return ExecutionResult(
                exit_code=124, timed_out=True, duration_ms=timeout * 1000,
                stderr="timeout",
            )
        except Exception as exc:  # noqa: BLE001 — docker failures -> sandbox error
            raise SandboxExecutionError(f"docker exec failed: {exc}") from exc

        elapsed = time.monotonic() - start
        return ExecutionResult(
            exit_code=result.returncode,
            stdout=(result.stdout or b"").decode(errors="replace"),
            stderr=(result.stderr or b"").decode(errors="replace"),
            timed_out=False,
            duration_ms=elapsed * 1000,
        )


def tempfile_mkdtemp() -> str:
    import tempfile

    return tempfile.mkdtemp(prefix="forgeharness-")


# --------------------------------------------------------------------------- #
# Facade
# --------------------------------------------------------------------------- #


class DockerSandbox:
    """The facade the harness talks to: lifecycle + policy + per-call timeout.

    ``execute_tool`` is the single entry point for a *tool call* executed inside
    the sandbox. It:

    1. resolves the per-call timeout (config default unless overridden),
    2. enforces the network default-deny/allowlist policy,
    3. delegates to the backend,
    4. converts a timed-out result into ``ToolTimeoutError``.

    The hard timeout is therefore a *per tool call* guarantee even though the
    container is *per session*.
    """

    def __init__(
        self,
        backend: Optional[SandboxBackend] = None,
        config: Optional[SandboxConfig] = None,
    ) -> None:
        self.config = config or SandboxConfig()
        self.backend = backend or FakeBackend(self.config)

    # -- lifecycle ------------------------------------------------------------

    def start(self) -> None:
        self.backend.start()

    def stop(self) -> None:
        self.backend.stop()

    def __enter__(self) -> "DockerSandbox":
        self.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.stop()

    # -- policy ---------------------------------------------------------------

    def _network_allowed(self, tool_name: str) -> bool:
        return tool_name in self.config.network_allowlist

    # -- execution ------------------------------------------------------------

    def execute_tool(
        self,
        tool_name: str,
        argv: list[str],
        *,
        timeout: Optional[float] = None,
        stdin: Optional[str] = None,
        env: Optional[dict[str, str]] = None,
        requires_network: bool = False,
        workdir: Optional[str] = None,
    ) -> ExecutionResult:
        timeout = self.config.default_timeout if timeout is None else timeout

        if requires_network and not self._network_allowed(tool_name):
            raise NetworkDeniedError(tool_name)

        result = self.backend.run(
            argv,
            timeout=timeout,
            workdir=workdir,
            env=env,
            stdin=stdin,
            allow_network=self._network_allowed(tool_name),
        )

        if result.timed_out:
            raise ToolTimeoutError(tool_name, timeout)
        return result

    # -- convenience filesystem inspection (harness/environment reads) ----------

    def read_file(self, path: str) -> str:
        result = self.execute_tool("_read", ["cat", path])
        if not result.ok:
            raise SandboxExecutionError(result.stderr or f"read failed: {path}")
        return result.stdout

    def list_dir(self, path: str) -> list[str]:
        result = self.execute_tool("_ls", ["ls", "-1", path])
        if not result.ok:
            raise SandboxExecutionError(result.stderr or f"list failed: {path}")
        return [line for line in result.stdout.splitlines() if line]

    def exists(self, path: str) -> bool:
        try:
            self.read_file(path)
            return True
        except SandboxExecutionError:
            return False

    def snapshot(self) -> dict[str, Any]:
        """Persist the sandbox's scratch state (simulated volume)."""
        if hasattr(self.backend, "snapshot"):
            return self.backend.snapshot()  # type: ignore[union-attr]
        return {}

    def load_snapshot(self, state: dict[str, Any]) -> None:
        if hasattr(self.backend, "load_snapshot"):
            self.backend.load_snapshot(state)  # type: ignore[union-attr]


def make_backend(
    config: Optional[SandboxConfig] = None, kind: str = "auto"
) -> SandboxBackend:
    """Create a backend; ``auto`` prefers Docker and falls back to the fake."""
    config = config or SandboxConfig()
    if kind == "fake":
        return FakeBackend(config)
    if kind == "docker":
        return DockerBackend(config)
    docker = DockerBackend(config)
    if docker.available:
        return docker
    return FakeBackend(config)


__all__ = [
    "SandboxError",
    "ToolTimeoutError",
    "SandboxExecutionError",
    "NetworkDeniedError",
    "ResourceLimits",
    "SandboxConfig",
    "ExecutionResult",
    "SandboxBackend",
    "FakeBackend",
    "DockerBackend",
    "DockerSandbox",
    "make_backend",
]
