"""Sandbox provider abstraction layer.

The product exposes two execution modes:

- ``local``: run code on the same machine as the app
- ``remote``: call a company-hosted sandbox gateway over HTTP

This keeps the UI and the graph generic. Each enterprise can point the app to
its own gateway endpoint regardless of whether that gateway runs on Google
Cloud, Azure, AWS, Kubernetes, or any internal platform.
"""

from __future__ import annotations

import base64
import importlib
import json
import os
import shutil
import ssl
import subprocess
import tempfile
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, BinaryIO, Callable, Dict, Iterable, Optional, Protocol, Sequence, Union, runtime_checkable


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


class CommandResult:
    """Result of a sandbox command execution."""

    __slots__ = ("stdout", "stderr", "exit_code")

    def __init__(self, stdout: str, stderr: str, exit_code: int) -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.exit_code = exit_code


class _LocalFiles:
    def __init__(self, root_dir: str) -> None:
        self._root = root_dir

    def _resolve(self, path: str) -> str:
        if os.path.isabs(path):
            rel = path.lstrip("/").lstrip("\\")
            return os.path.join(self._root, rel)
        return os.path.join(self._root, path)

    def write(self, path: str, content: Union[str, bytes, BinaryIO]) -> None:
        local_path = self._resolve(path)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        if isinstance(content, str):
            with open(local_path, "w", encoding="utf-8") as handle:
                handle.write(content)
        elif hasattr(content, "read"):
            with open(local_path, "wb") as handle:
                handle.write(content.read())
        else:
            with open(local_path, "wb") as handle:
                handle.write(content)

    def read(self, path: str) -> bytes:
        local_path = self._resolve(path)
        with open(local_path, "rb") as handle:
            return handle.read()


class _LocalCommands:
    def __init__(self, root_dir: str) -> None:
        self._root = root_dir
        self._root_unix = root_dir.replace("\\", "/")
        self._bash = shutil.which("bash") or shutil.which("sh")

    def run(self, cmd: str, timeout: Optional[int] = None) -> CommandResult:
        remapped = cmd.replace("/home/user/", self._root_unix + "/home/user/")
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


class LocalSandbox:
    """Sandbox that executes code locally in an isolated temporary directory."""

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

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class RemoteSandboxHTTPError(RuntimeError):
    """Raised when the remote sandbox gateway returns an invalid response."""


class _RemoteSandboxClient:
    def __init__(
        self,
        *,
        endpoint: str,
        api_key: str = "",
        auth_scheme: str = "Bearer",
        auth_header: str = "Authorization",
        workspace_id: str = "",
        project: str = "",
        provider_hint: str = "",
        verify_tls: Union[str, bool] = True,
        request_timeout_s: Union[str, int, float] = 30,
        session_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        endpoint = str(endpoint or "").strip().rstrip("/")
        if not endpoint:
            raise ValueError("Remote sandbox endpoint is required")
        self._endpoint = endpoint
        self._timeout = self._coerce_timeout(request_timeout_s)
        self._context = self._build_ssl_context(verify_tls)
        self._headers = self._build_headers(
            api_key=str(api_key or "").strip(),
            auth_scheme=str(auth_scheme or "Bearer").strip() or "Bearer",
            auth_header=str(auth_header or "Authorization").strip() or "Authorization",
        )
        self._session_id: Optional[str] = None
        self._session_metadata = dict(session_metadata or {})
        self._create_session(
            workspace_id=str(workspace_id or "").strip(),
            project=str(project or "").strip(),
            provider_hint=str(provider_hint or "").strip(),
        )

    @staticmethod
    def _coerce_timeout(value: Union[str, int, float]) -> float:
        try:
            timeout = float(value)
        except Exception:
            timeout = 30.0
        return max(1.0, timeout)

    @staticmethod
    def _build_ssl_context(verify_tls: Union[str, bool]) -> Optional[ssl.SSLContext]:
        if isinstance(verify_tls, str):
            lowered = verify_tls.strip().lower()
            verify = lowered not in {"0", "false", "no", "off"}
        else:
            verify = bool(verify_tls)
        if verify:
            return None
        return ssl._create_unverified_context()

    @staticmethod
    def _build_headers(*, api_key: str, auth_scheme: str, auth_header: str) -> Dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if api_key:
            if auth_scheme.lower() == "none":
                headers[auth_header] = api_key
            else:
                headers[auth_header] = f"{auth_scheme} {api_key}".strip()
        return headers

    def _request_json(
        self,
        method: str,
        path: str,
        *,
        payload: Optional[Dict[str, Any]] = None,
        query: Optional[Dict[str, Any]] = None,
        timeout_override: Optional[float] = None,
    ) -> Dict[str, Any]:
        if query:
            query_string = urllib.parse.urlencode({k: v for k, v in query.items() if v is not None})
        else:
            query_string = ""
        url = f"{self._endpoint}{path}"
        if query_string:
            url = f"{url}?{query_string}"
        body = None
        if payload is not None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(url, data=body, method=method.upper(), headers=self._headers)
        effective_timeout = timeout_override if timeout_override is not None else self._timeout
        try:
            with urllib.request.urlopen(request, timeout=effective_timeout, context=self._context) as response:
                raw = response.read()
        except Exception as exc:
            raise RemoteSandboxHTTPError(f"Remote sandbox request failed: {exc}") from exc
        if not raw:
            return {}
        try:
            data = json.loads(raw.decode("utf-8"))
        except Exception as exc:
            raise RemoteSandboxHTTPError(f"Remote sandbox returned non-JSON response for {path}") from exc
        if not isinstance(data, dict):
            raise RemoteSandboxHTTPError(f"Remote sandbox returned invalid JSON object for {path}")
        return data

    def _create_session(self, *, workspace_id: str, project: str, provider_hint: str) -> None:
        payload = {
            "workspace_id": workspace_id or None,
            "project": project or None,
            "provider_hint": provider_hint or None,
            "metadata": self._session_metadata or None,
        }
        data = self._request_json("POST", "/sessions", payload=payload)
        session_id = str(
            data.get("session_id")
            or data.get("id")
            or ""
        ).strip()
        if not session_id:
            raise RemoteSandboxHTTPError("Remote sandbox did not return a session_id")
        self._session_id = session_id

    def write_file(self, path: str, content: Union[str, bytes, BinaryIO]) -> None:
        if hasattr(content, "read"):
            payload_bytes = content.read()
        elif isinstance(content, str):
            payload_bytes = content.encode("utf-8")
        else:
            payload_bytes = content
        data = self._request_json(
            "POST",
            f"/sessions/{self.session_id}/files/write",
            payload={
                "path": path,
                "content_base64": base64.b64encode(payload_bytes).decode("ascii"),
            },
        )
        if data.get("ok") is False:
            raise RemoteSandboxHTTPError(str(data.get("error") or "Remote sandbox rejected file write"))

    def read_file(self, path: str) -> bytes:
        data = self._request_json(
            "GET",
            f"/sessions/{self.session_id}/files/read",
            query={"path": path},
        )
        encoded = str(data.get("content_base64") or "").strip()
        if not encoded:
            raise RemoteSandboxHTTPError("Remote sandbox returned empty file payload")
        try:
            return base64.b64decode(encoded)
        except Exception as exc:
            raise RemoteSandboxHTTPError("Remote sandbox returned invalid base64 content") from exc

    def run_command(self, cmd: str, timeout: Optional[int] = None) -> CommandResult:
        # HTTP timeout must exceed the command timeout so the connection stays
        # alive while the gateway executes the script.  Add a 60-second buffer
        # for network / serialization overhead.
        http_timeout: Optional[float] = None
        if timeout is not None:
            http_timeout = float(timeout) + 60.0
        data = self._request_json(
            "POST",
            f"/sessions/{self.session_id}/commands/run",
            payload={
                "cmd": cmd,
                "timeout": timeout,
            },
            timeout_override=http_timeout,
        )
        exit_code = int(data.get("exit_code", 1) or 0)
        stdout = str(data.get("stdout") or "")
        stderr = str(data.get("stderr") or "")
        return CommandResult(stdout, stderr, exit_code)

    def close(self) -> None:
        if not self._session_id:
            return
        try:
            self._request_json("POST", f"/sessions/{self.session_id}/close", payload={})
        except Exception:
            pass
        finally:
            self._session_id = None

    @property
    def session_id(self) -> str:
        if not self._session_id:
            raise RemoteSandboxHTTPError("Remote sandbox session is not initialized")
        return self._session_id


class _RemoteFiles:
    def __init__(self, client: _RemoteSandboxClient) -> None:
        self._client = client

    def write(self, path: str, content: Union[str, bytes, BinaryIO]) -> None:
        self._client.write_file(path, content)

    def read(self, path: str) -> bytes:
        return self._client.read_file(path)


class _RemoteCommands:
    def __init__(self, client: _RemoteSandboxClient) -> None:
        self._client = client

    def run(self, cmd: str, timeout: Optional[int] = None) -> CommandResult:
        return self._client.run_command(cmd, timeout=timeout)


class RemoteSandbox:
    """Generic sandbox client backed by a company-hosted HTTP gateway."""

    def __init__(self, **kwargs: Any) -> None:
        self._client = _RemoteSandboxClient(**kwargs)
        self.files = _RemoteFiles(self._client)
        self.commands = _RemoteCommands(self._client)

    def close(self) -> None:
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


@dataclass(frozen=True)
class SandboxConfigField:
    key: str
    label: str
    description: str = ""
    placeholder: str = ""
    secret: bool = False
    required: bool = False


@dataclass(frozen=True)
class SandboxProviderSpec:
    name: str
    label: str
    description: str
    implemented: bool
    config_fields: tuple[SandboxConfigField, ...] = ()


def _default_local_tester(_: Optional[Dict[str, Any]] = None) -> tuple[bool, str]:
    return True, "Sandbox local disponible"


def _remote_gateway_tester(settings: Optional[Dict[str, Any]] = None) -> tuple[bool, str]:
    payload = settings if isinstance(settings, dict) else {}
    endpoint = str(payload.get("endpoint") or "").strip().rstrip("/")
    if not endpoint:
        return False, "Falta el endpoint del gateway remoto"

    auth_header = str(payload.get("auth_header") or "Authorization").strip() or "Authorization"
    auth_scheme = str(payload.get("auth_scheme") or "Bearer").strip() or "Bearer"
    api_key = str(payload.get("api_key") or "").strip()
    verify_tls = payload.get("verify_tls", True)
    if isinstance(verify_tls, str):
        lowered = verify_tls.strip().lower()
        verify = lowered not in {"0", "false", "no", "off"}
    else:
        verify = bool(verify_tls)
    headers = {"Accept": "application/json"}
    if api_key:
        if auth_scheme.lower() == "none":
            headers[auth_header] = api_key
        else:
            headers[auth_header] = f"{auth_scheme} {api_key}".strip()
    context = None if verify else ssl._create_unverified_context()
    request = urllib.request.Request(f"{endpoint}/health", headers=headers, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=10, context=context) as response:
            if 200 <= getattr(response, "status", 200) < 300:
                return True, "Gateway remoto disponible"
    except Exception as exc:
        return False, f"Gateway remoto no disponible: {exc}"
    return False, "Gateway remoto devolvio una respuesta no valida"


_PROVIDER_REGISTRY: dict[str, Any] = {
    "local": LocalSandbox,
    "remote": RemoteSandbox,
}

_PROVIDER_TESTERS: dict[str, Callable[[Optional[Dict[str, Any]]], tuple[bool, str]]] = {
    "local": _default_local_tester,
    "remote": _remote_gateway_tester,
}

_PROVIDER_ALIASES: dict[str, str] = {
    "local": "local",
    "default": "local",
    "remote": "remote",
    "remote_gateway": "remote",
    "gateway": "remote",
    "gcp": "remote",
    "google_cloud": "remote",
    "azure": "remote",
    "aws": "remote",
}

_REMOTE_FIELDS = (
    SandboxConfigField("endpoint", "Endpoint del gateway", "URL base del sandbox remoto expuesto por la empresa", "https://sandbox.miempresa.com", required=True),
    SandboxConfigField("api_key", "API key", "Credencial del gateway remoto", "secret-...", secret=True),
    SandboxConfigField("auth_scheme", "Auth scheme", "Por defecto Bearer; usa 'none' si el header ya contiene la credencial", "Bearer"),
    SandboxConfigField("auth_header", "Auth header", "Cabecera HTTP usada para autenticacion", "Authorization"),
    SandboxConfigField("workspace_id", "Workspace", "Identificador logico del tenant o workspace", "analytics-prod"),
    SandboxConfigField("project", "Project", "Proyecto o cuenta interna donde corre el gateway", "mi-proyecto"),
    SandboxConfigField("provider_hint", "Hint de infraestructura", "Opcional: gcp, azure, aws, k8s, onprem", "gcp"),
    SandboxConfigField("request_timeout_s", "Timeout HTTP", "Timeout del cliente hacia el gateway en segundos", "30"),
)

_PROVIDER_SPECS: dict[str, SandboxProviderSpec] = {
    "local": SandboxProviderSpec(
        name="local",
        label="Local",
        description="Ejecuta los agentes en esta misma maquina.",
        implemented=True,
        config_fields=(),
    ),
    "remote": SandboxProviderSpec(
        name="remote",
        label="Gateway remoto",
        description="Ejecuta las runs en un sandbox HTTP desplegado por la empresa en su propia nube o infraestructura.",
        implemented=True,
        config_fields=_REMOTE_FIELDS,
    ),
}

_LOADED_PROVIDER_MODULES: set[str] = set()


def load_sandbox_provider_modules(modules: Optional[Sequence[str]] = None) -> None:
    """Import optional provider modules declared by the environment."""

    if modules is None:
        raw = str(os.getenv("SANDBOX_PROVIDER_MODULES", "") or "")
        modules = [item.strip() for item in raw.replace(";", ",").split(",") if item.strip()]
    for module_name in modules:
        if module_name in _LOADED_PROVIDER_MODULES:
            continue
        importlib.import_module(module_name)
        _LOADED_PROVIDER_MODULES.add(module_name)


def register_sandbox_provider(
    name: str,
    factory: Any,
    *,
    label: Optional[str] = None,
    description: Optional[str] = None,
    config_fields: Optional[Iterable[SandboxConfigField]] = None,
    aliases: Optional[Iterable[str]] = None,
    tester: Optional[Callable[[Optional[Dict[str, Any]]], tuple[bool, str]]] = None,
) -> None:
    """Register a sandbox provider class/factory plus optional UI metadata."""

    key = str(name or "").strip().lower()
    if not key:
        raise ValueError("Sandbox provider name cannot be empty")

    _PROVIDER_REGISTRY[key] = factory
    _PROVIDER_SPECS[key] = SandboxProviderSpec(
        name=key,
        label=label or key.replace("_", " ").title(),
        description=description or "Sandbox provider registrado por la empresa.",
        implemented=True,
        config_fields=tuple(config_fields or ()),
    )
    _PROVIDER_TESTERS[key] = tester or (
        lambda _settings=None, _label=label or key: (True, f"Proveedor {_label} registrado")
    )
    if aliases:
        for alias in aliases:
            alias_key = str(alias or "").strip().lower()
            if alias_key:
                _PROVIDER_ALIASES[alias_key] = key


def resolve_sandbox_provider_name(provider: Optional[str] = None) -> str:
    load_sandbox_provider_modules()
    key = (provider or os.getenv("SANDBOX_PROVIDER", "local")).strip().lower()
    return _PROVIDER_ALIASES.get(key, key)


def list_sandbox_providers() -> list[SandboxProviderSpec]:
    load_sandbox_provider_modules()
    specs = list(_PROVIDER_SPECS.values())
    specs.sort(key=lambda spec: spec.label.lower())
    return specs


def get_sandbox_provider_spec(provider: Optional[str] = None) -> SandboxProviderSpec:
    resolved = resolve_sandbox_provider_name(provider)
    spec = _PROVIDER_SPECS.get(resolved)
    if spec is not None:
        return spec
    return SandboxProviderSpec(
        name=resolved,
        label=resolved.replace("_", " ").title(),
        description="Proveedor desconocido o no registrado.",
        implemented=False,
        config_fields=(),
    )


def is_sandbox_provider_available(provider: Optional[str] = None) -> bool:
    resolved = resolve_sandbox_provider_name(provider)
    return resolved in _PROVIDER_REGISTRY


def test_sandbox_provider_connectivity(
    provider: Optional[str] = None,
    settings: Optional[Dict[str, Any]] = None,
) -> tuple[bool, str]:
    resolved = resolve_sandbox_provider_name(provider)
    if resolved not in _PROVIDER_REGISTRY:
        spec = get_sandbox_provider_spec(resolved)
        return False, f"{spec.label}: backend no registrado en este despliegue"
    tester = _PROVIDER_TESTERS.get(resolved)
    if not callable(tester):
        spec = get_sandbox_provider_spec(resolved)
        return True, f"{spec.label}: provider registrado"
    return tester(settings or {})


def get_sandbox_class(provider: Optional[str] = None) -> Any:
    resolved = resolve_sandbox_provider_name(provider)
    cls = _PROVIDER_REGISTRY.get(resolved)
    if cls is None:
        available = sorted(_PROVIDER_REGISTRY.keys())
        raise ValueError(
            f"Unknown sandbox provider '{provider or resolved}' (resolved to '{resolved}'). "
            f"Available: {available}"
        )
    return cls
