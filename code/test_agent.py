import os
import sys
import pathlib
import types as _types

# ---------------------------------------------------------------------------
# 1. Fix import path — must happen before any local imports.
# ---------------------------------------------------------------------------
_CODE_DIR = str(pathlib.Path(__file__).resolve().parent)
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

# ---------------------------------------------------------------------------
# 2. Inject dummy environment variables 
# ---------------------------------------------------------------------------
_ENV_DEFAULTS = {
    "LLM_TEMPERATURE": "0.7",
    "LLM_MAX_TOKENS": "1024",
    "TEMPERATURE": "0.7",
    "MAX_TOKENS": "1024",
    "MODEL_TEMPERATURE": "0.7",
    "MODEL_MAX_TOKENS": "1024",
    "MODEL_PROVIDER": "openai",
    "LLM_MODEL": "gpt-4.1",
    "LLM_PROVIDER": "openai",
    "TOP_P": "1.0",
    "FREQUENCY_PENALTY": "0.0",
    "PRESENCE_PENALTY": "0.0",
    "TIMEOUT": "30",
    "REQUEST_TIMEOUT": "30",
    "MAX_RETRIES": "3",
    "USE_KEY_VAULT": "false",
    "KEY_VAULT_URI": "",
    "AZURE_USE_DEFAULT_CREDENTIAL": "false",
    "OPENAI_API_KEY": "test-key-not-real",
    "ANTHROPIC_API_KEY": "test-key-not-real",
    "AZURE_OPENAI_API_KEY": "test-key-not-real",
    "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com",
    "AZURE_OPENAI_EMBEDDING_DEPLOYMENT": "text-embedding-ada-002",
    "AZURE_SEARCH_API_KEY": "test-key-not-real",
    "AZURE_SEARCH_ENDPOINT": "https://test.search.windows.net",
    "AZURE_SEARCH_INDEX_NAME": "test-index",
    "AZURE_SEARCH_SERVICE_ENDPOINT": "https://test.search.windows.net",
    "AZURE_CONTENT_SAFETY_ENDPOINT": "https://test.contentsafety.azure.com",
    "AZURE_CONTENT_SAFETY_KEY": "test-key-not-real",
    "API_KEY": "test-key-not-real",
    "API_URL": "https://test.example.com/api",
    "BASE_URL": "https://test.example.com",
    "DATABASE_URL": "sqlite:///test.db",
    "REDIS_URL": "redis://localhost:6379/0",
    "AGENT_NAME": "test-agent",
    "AGENT_ID": "test-agent-id",
    "PROJECT_NAME": "test-project",
    "PROJECT_ID": "test-project-id",
    "SERVICE_NAME": "test-agent",
    "SERVICE_VERSION": "1.0.0",
    "DEBUG": "false",
    "LOG_LEVEL": "WARNING",
    "ENVIRONMENT": "test",
    "PORT": "8000",
    "HOST": "0.0.0.0",
    "CHUNK_SIZE": "512",
    "EMBEDDING_DIMENSION": "1536",
    "BATCH_SIZE": "32",
}
for _k, _v in _ENV_DEFAULTS.items():
    os.environ.setdefault(_k, _v)

# ---------------------------------------------------------------------------
# 3. No real Azure Content Safety credentials exist in the test environment.
# ---------------------------------------------------------------------------
_cs_mod_name = "modules.guardrails.content_safety_decorator"
if _cs_mod_name not in sys.modules:
    for _parent in ("modules", "modules.guardrails"):
        if _parent not in sys.modules:
            try:
                __import__(_parent)
            except ImportError:
                _stub = _types.ModuleType(_parent)
                _stub.__path__ = [str(pathlib.Path(_CODE_DIR) / _parent.replace(".", os.sep))]
                _stub.__package__ = _parent
                sys.modules[_parent] = _stub
    _cs_stub = _types.ModuleType(_cs_mod_name)
    sys.modules[_cs_mod_name] = _cs_stub
else:
    _cs_stub = sys.modules[_cs_mod_name]

def _noop_content_safety(func=None, *, config=None):
    return func if func is not None else (lambda fn: fn)
_cs_stub.with_content_safety = _noop_content_safety

# ---------------------------------------------------------------------------
# 4. Neutralise tenacity retry loops —
# ---------------------------------------------------------------------------
try:
    import tenacity as _tenacity
    def _passthrough_retry(*_a, **_kw):
        if len(_a) == 1 and callable(_a[0]) and not _kw:
            return _a[0]
        return lambda fn: fn
    _tenacity.retry = _passthrough_retry
    if hasattr(_tenacity, "asyncio"):
        _tenacity.asyncio.retry = _passthrough_retry
except ImportError:
    pass

# ---------------------------------------------------------------------------
# 5. Inject a safe `logger` builtin 
# ---------------------------------------------------------------------------
import builtins as _builtins
import logging as _logging
if not hasattr(_builtins, "logger"):
    _shim_logger = _logging.getLogger("agent_test_shim")
    if not _shim_logger.handlers:
        _shim_logger.addHandler(_logging.NullHandler())
    _builtins.logger = _shim_logger

import pytest
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient

from agent import app


@pytest.fixture
def client():
    return TestClient(app)


def test_agent_app_is_importable(client):
    """The agent module imports successfully and the FastAPI app is reachable."""
    assert client is not None


def test_health_endpoint_returns_200(client):
    """GET /health should return HTTP 200."""
    response = client.get("/health")
    assert response.status_code == 200


def test_main_endpoint_returns_200(client):
    """POST /query returns 200 when the agent process method is mocked."""
    payload = {'requirement_description': 'test input'}
    with patch(
        "agent._agent_instance.process",
        new=AsyncMock(return_value={'success': True, 'content': 'ok'}),
    ):
        response = client.post("/query", json=payload)
    assert response.status_code == 200
