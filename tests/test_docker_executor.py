"""Tests for the Docker-sandboxed executor (backends + facade)."""

import pytest

from forgeharness.sandbox.docker_executor import (
    DockerBackend,
    DockerSandbox,
    ExecutionResult,
    FakeBackend,
    NetworkDeniedError,
    ResourceLimits,
    SandboxConfig,
    SandboxError,
    SandboxExecutionError,
    ToolTimeoutError,
    make_backend,
)
from forgeharness.core.tool_registry import ToolCallError


def _fake_backend(**kwargs) -> FakeBackend:
    config = SandboxConfig(default_timeout=5.0, **kwargs)
    return FakeBackend(config)


def test_fake_backend_filesystem_roundtrip():
    b = _fake_backend()
    b.start()
    b.run(["sh", "-c", "cat > /workspace/a.txt"], timeout=5, stdin="hello")
    result = b.run(["cat", "/workspace/a.txt"], timeout=5)
    assert result.stdout == "hello"
    b.run(["sh", "-c", "cat >> /workspace/a.txt"], timeout=5, stdin=" world")
    assert b.run(["cat", "/workspace/a.txt"], timeout=5).stdout == "hello world"
    b.stop()


def test_fake_backend_list_and_mkdir():
    b = _fake_backend()
    b.start()
    b.run(["mkdir", "-p", "/workspace/sub"], timeout=5)
    b.run(["sh", "-c", "cat > /workspace/sub/x.txt"], timeout=5, stdin="x")
    out = b.run(["ls", "-1", "/workspace/sub"], timeout=5).stdout
    assert "x.txt" in out
    b.stop()


def test_fake_backend_python_compute():
    b = _fake_backend()
    b.start()
    result = b.run(["python3", "-c", "print(2+3)"], timeout=5)
    assert result.stdout.strip() == "5"
    b.stop()


def test_timeout_is_distinct_error_type():
    sandbox = DockerSandbox(_fake_backend(), SandboxConfig(default_timeout=5.0))
    sandbox.start()
    try:
        with pytest.raises(ToolTimeoutError):
            sandbox.execute_tool("run_shell", ["sh", "-c", "sleep 100"])
    finally:
        sandbox.stop()
    # distinct: never conflated with tool-logic / validation errors
    assert not issubclass(ToolTimeoutError, ToolCallError)
    assert issubclass(ToolTimeoutError, SandboxError)


def test_hang_command_times_out():
    b = _fake_backend()
    b.start()
    result = b.run(["hang"], timeout=5)
    assert result.timed_out is True
    b.stop()


def test_network_default_deny_and_allowlist():
    config = SandboxConfig(default_timeout=5.0, network_allowlist={"http_get": ["example.com"]})
    sandbox = DockerSandbox(FakeBackend(config), config)
    sandbox.start()
    try:
        # not allowlisted -> policy gate raises NetworkDeniedError
        with pytest.raises(NetworkDeniedError):
            sandbox.execute_tool("other", ["curl", "-s", "http://example.com"], requires_network=True)
        # allowlisted -> passes the policy gate
        result = sandbox.execute_tool(
            "http_get", ["curl", "-s", "http://example.com"], requires_network=True
        )
        assert result.ok
        # default-deny at the backend: a raw curl without allow_network fails
        denied = sandbox.backend.run(["curl", "-s", "http://example.com"], timeout=5)
        assert not denied.ok
    finally:
        sandbox.stop()


def test_snapshot_and_load_snapshot():
    b = _fake_backend()
    b.start()
    b.run(["sh", "-c", "cat > /workspace/data.txt"], timeout=5, stdin="payload")
    snap = b.snapshot()

    b2 = _fake_backend()
    b2.start()
    b2.load_snapshot(snap)
    assert b2.run(["cat", "/workspace/data.txt"], timeout=5).stdout == "payload"
    b.stop()
    b2.stop()


def test_stop_wipes_scratch():
    b = _fake_backend()
    b.start()
    b.run(["sh", "-c", "cat > /workspace/x.txt"], timeout=5, stdin="x")
    b.stop()
    assert b.run(["cat", "/workspace/x.txt"], timeout=5).exit_code != 0


class _FakeProc:
    def __init__(self, returncode=0, stdout=b"", stderr=b""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class _RecordingRunner:
    def __init__(self, returncode=0):
        self.returncode = returncode
        self.calls = []

    def __call__(self, argv, **kwargs):
        self.calls.append((argv, kwargs))
        return _FakeProc(self.returncode)


def test_docker_backend_command_construction():
    config = SandboxConfig(
        session_id="s1",
        resource_limits=ResourceLimits(cpu=0.5, memory="256m", pids=64),
        read_only_rootfs=True,
    )
    runner = _RecordingRunner()
    backend = DockerBackend(config, runner=runner)
    backend._scratch_path = lambda: __import__("pathlib").Path("/tmp/scratch")
    backend.start()

    run_argv = runner.calls[0][0]
    joined = " ".join(run_argv)
    assert run_argv[0] == "docker"
    assert "--network" in run_argv and "none" in run_argv
    assert "--cpus" in run_argv and "0.5" in run_argv
    assert "--memory" in run_argv and "256m" in run_argv
    assert "--pids-limit" in run_argv and "64" in run_argv
    assert "--read-only" in run_argv
    assert "-v" in run_argv and "/tmp/scratch:/workspace" in run_argv
    assert "python:3.11-slim" in run_argv

    backend.run(["cat", "/workspace/f"], timeout=5)
    exec_argv = runner.calls[-1][0]
    assert exec_argv[0] == "docker" and "exec" in exec_argv
    assert backend._container_name in exec_argv
    backend.stop()


def test_docker_backend_unavailable_when_runner_fails():
    def fail_runner(argv, **kwargs):
        raise OSError("no docker")

    backend = DockerBackend(SandboxConfig(), runner=fail_runner)
    assert backend.available is False


def test_make_backend_auto_prefers_fake_without_docker():
    backend = make_backend(SandboxConfig(), kind="auto")
    assert backend.name == "fake"
    assert backend.available is True


def test_execution_result_ok_property():
    assert ExecutionResult(exit_code=0).ok is True
    assert ExecutionResult(exit_code=1).ok is False
    assert ExecutionResult(exit_code=0, timed_out=True).ok is False
