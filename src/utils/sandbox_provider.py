"""Sandbox provider abstraction layer.

Defines a provider-agnostic interface for code execution sandboxes and ships
a ``LocalSandbox`` implementation that runs code in an isolated local
directory using the system Python and shell.

Future providers (Google Cloud Run, Azure Container Instances, etc.) only
need to implement the same interface: ``files.write``, ``files.read``,
``commands.run``, and ``close``.
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import tempfile
from typing import Any, BinaryIO, Optional, Protocol, Union, runtime_checkable


# ---------------------------------------------------------------------------
# Protocol (the interface every sandbox provider must implement)
# ---------------------------------------------------------------------------

class SandboxFiles(Protocol):
    """File I/O on the sandbox filesystem."""

    def write(self, path: str, content: Union[str, bytes, BinaryIO]) -> None: ...
    def read(self, path: str) -> bytes: ...


class SandboxCommands(Protocol):
    """Command execution on the sandbox."""

    def run(self, cmd: str, timeout: Optional[int] = None) -> Any: ...


@runtime_checkable
class SandboxProvider(Protocol):
    """Minimal contract that every sandbox backend must satisfy."""

    files: SandboxFiles
    commands: SandboxCommands

    def close(self) -> None: ...


# ---------------------------------------------------------------------------
# Command result (matches the shape expected by sandbox_resilience.py)
# ---------------------------------------------------------------------------

class CommandResult:
    """Result of a sandbox command execution."""

    __slots__ = ("stdout", "stderr", "exit_code")

    def __init__(self, stdout: str, stderr: str, exit_code: int) -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.exit_code = exit_code


# ---------------------------------------------------------------------------
# LocalSandbox implementation
# ---------------------------------------------------------------------------

class _LocalFiles:
    """File operations backed by the local filesystem."""

    def __init__(self, root_dir: str) -> None:
        self._root = root_dir

    def _resolve(self, path: str) -> str:
        """Map sandbox-absolute paths into the local root directory."""
        if os.path.isabs(path):
            # /home/user/run/abc/data/raw.csv → {root}/home/user/run/abc/data/raw.csv
            rel = path.lstrip("/").lstrip("\\")
            return os.path.join(self._root, rel)
        return os.path.join(self._root, path)

    def write(self, path: str, content: Union[str, bytes, BinaryIO]) -> None:
        local_path = self._resolve(path)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        if isinstance(content, str):
            with open(local_path, "w", encoding="utf-8") as f:
                f.write(content)
        elif hasattr(content, "read"):
            data = content.read()
            with open(local_path, "wb") as f:
                f.write(data)
        else:
            with open(local_path, "wb") as f:
                f.write(content)

    def read(self, path: str) -> bytes:
        local_path = self._resolve(path)
        with open(local_path, "rb") as f:
            return f.read()


class _LocalCommands:
    """Command execution backed by local subprocess."""

    def __init__(self, root_dir: str) -> None:
        self._root = root_dir
        self._root_unix = root_dir.replace("\\", "/")
        # Detect available shell
        self._bash = shutil.which("bash") or shutil.which("sh")

    def run(self, cmd: str, timeout: Optional[int] = None) -> CommandResult:
        remapped = self._remap_paths(cmd)
        try:
            if self._bash:
                proc = subprocess.run(
                    [self._bash, "-c", remapped],
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    cwd=self._root,
                )
            else:
                proc = subprocess.run(
                    remapped,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    cwd=self._root,
                )
            return CommandResult(proc.stdout or "", proc.stderr or "", proc.returncode)
        except subprocess.TimeoutExpired:
            return CommandResult("", f"Command timed out after {timeout}s", 1)
        except Exception as exc:
            return CommandResult("", str(exc), 1)

    def _remap_paths(self, cmd: str) -> str:
        """Remap sandbox absolute paths to local root directory."""
        # Replace /home/user/ prefix with local root equivalent
        return cmd.replace("/home/user/", self._root_unix + "/home/user/")


class LocalSandbox:
    """Sandbox that executes code locally in an isolated temporary directory.

    Compatible with the ``SandboxProvider`` protocol so it can be used as a
    drop-in replacement for any remote sandbox backend.

    The temporary directory is created on construction and optionally cleaned
    up on ``close()``.
    """

    def __init__(self, *, work_dir: Optional[str] = None, cleanup_on_close: bool = False) -> None:
        self._work_dir = work_dir or tempfile.mkdtemp(prefix="sandbox_local_")
        self._cleanup = cleanup_on_close
        self.files = _LocalFiles(self._work_dir)
        self.commands = _LocalCommands(self._work_dir)

    @property
    def work_dir(self) -> str:
        return self._work_dir

    def close(self) -> None:
        if self._cleanup and os.path.isdir(self._work_dir):
            shutil.rmtree(self._work_dir, ignore_errors=True)

    # Context-manager support
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

# Registry of known providers. External modules can register new providers by
# calling ``register_sandbox_provider(name, factory_fn)``.
_PROVIDER_REGISTRY: dict[str, Any] = {
    "local": LocalSandbox,
}

_PROVIDER_ALIASES: dict[str, str] = {
    "local": "local",
    "default": "local",
    # Future providers
    "gcp": "gcp",
    "google_cloud": "gcp",
    "azure": "azure",
    "azure_container": "azure",
    "aws": "aws",
}


def register_sandbox_provider(name: str, factory: Any) -> None:
    """Register a sandbox provider class/factory.

    Args:
        name: Provider identifier (e.g., ``"gcp"``, ``"azure"``).
        factory: A callable that returns a ``SandboxProvider`` instance.
    """
    _PROVIDER_REGISTRY[name.lower()] = factory


def get_sandbox_class(provider: Optional[str] = None) -> Any:
    """Return the sandbox class for the requested provider.

    Resolution order:
    1. Explicit ``provider`` argument
    2. ``SANDBOX_PROVIDER`` environment variable
    3. Falls back to ``"local"``

    Returns:
        A class/callable that constructs a ``SandboxProvider``-compatible object.

    Raises:
        ValueError: If the provider is not registered.
    """
    key = (provider or os.getenv("SANDBOX_PROVIDER", "local")).strip().lower()
    resolved = _PROVIDER_ALIASES.get(key, key)
    cls = _PROVIDER_REGISTRY.get(resolved)
    if cls is None:
        available = sorted(_PROVIDER_REGISTRY.keys())
        raise ValueError(
            f"Unknown sandbox provider '{key}' (resolved to '{resolved}'). "
            f"Available: {available}"
        )
    return cls
