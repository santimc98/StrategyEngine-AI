from unittest.mock import MagicMock, patch

from src.utils.sandbox_provider import (
    RemoteSandbox,
    get_sandbox_provider_spec,
    is_sandbox_provider_available,
    list_sandbox_providers,
    register_sandbox_provider,
    test_sandbox_provider_connectivity as run_provider_connectivity_test,
)


def test_sandbox_provider_catalog_includes_local_and_remote_gateway():
    specs = {spec.name: spec for spec in list_sandbox_providers()}

    assert "local" in specs
    assert specs["local"].implemented is True
    assert "remote" in specs
    assert specs["remote"].implemented is True


def test_register_sandbox_provider_exposes_provider_to_ui():
    class DummySandbox:
        def close(self):
            return None

    register_sandbox_provider(
        "test_remote",
        DummySandbox,
        label="Test Remote",
        description="Remote sandbox for tests.",
    )

    spec = get_sandbox_provider_spec("test_remote")
    ok, message = run_provider_connectivity_test("test_remote", {"endpoint": "https://example.com"})

    assert spec.label == "Test Remote"
    assert is_sandbox_provider_available("test_remote") is True
    assert ok is True
    assert "Test Remote" in message


@patch("urllib.request.urlopen")
def test_remote_gateway_connectivity_uses_health_endpoint(mock_urlopen):
    response = MagicMock()
    response.__enter__.return_value = response
    response.__exit__.return_value = False
    response.status = 200
    mock_urlopen.return_value = response

    ok, message = run_provider_connectivity_test(
        "remote",
        {"endpoint": "https://sandbox.example.com", "api_key": "secret"},
    )

    assert ok is True
    assert "Gateway remoto disponible" in message
    request = mock_urlopen.call_args.args[0]
    assert request.full_url == "https://sandbox.example.com/health"


@patch("urllib.request.urlopen")
def test_remote_sandbox_gateway_protocol_lifecycle(mock_urlopen):
    def _response(payload):
        response = MagicMock()
        response.__enter__.return_value = response
        response.__exit__.return_value = False
        response.read.return_value = payload
        response.status = 200
        return response

    mock_urlopen.side_effect = [
        _response(b'{"session_id":"sess-123"}'),
        _response(b'{"ok":true}'),
        _response(b'{"stdout":"done","stderr":"","exit_code":0}'),
        _response(b'{"content_base64":"aGVsbG8="}'),
        _response(b'{"ok":true}'),
    ]

    sandbox = RemoteSandbox(endpoint="https://sandbox.example.com", api_key="secret")
    sandbox.files.write("/tmp/file.txt", "hello")
    result = sandbox.commands.run("echo hello", timeout=5)
    content = sandbox.files.read("/tmp/file.txt")
    sandbox.close()

    assert result.exit_code == 0
    assert result.stdout == "done"
    assert content == b"hello"
