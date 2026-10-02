from __future__ import annotations

from types import SimpleNamespace

import pytest

from local_ai_hub import mcp_server
from local_ai_hub.config import load_config
from local_ai_hub.repo_tools import RepositoryTools
from local_ai_hub.services import LocalAIServices


@pytest.fixture
def captured_backend(monkeypatch):
    calls = []
    monkeypatch.setattr(mcp_server.FEATURES, "tasks", True)
    monkeypatch.setattr(mcp_server.FEATURES, "repo", True)
    monkeypatch.setattr(mcp_server.FEATURES, "has_any_model", lambda: True)

    def post(endpoint, payload, **kwargs):
        calls.append((endpoint, payload))
        return {"success": True, "text": "Explanation supplied.", "results": []}

    monkeypatch.setattr(mcp_server.CLIENT, "post", post)
    return calls


@pytest.mark.parametrize("task,prompt,expected", [
    ("", "PROMPT_SENTINEL with facts", "PROMPT_SENTINEL with facts"),
    ("   ", "PROMPT_SENTINEL", "PROMPT_SENTINEL"),
    ("TASK_SENTINEL", "PROMPT_SENTINEL", "TASK_SENTINEL"),
])
def test_reason_preserves_assignment(captured_backend, task, prompt, expected):
    mcp_server.local_ai_task(action="reason", task=task, prompt=prompt, context="FACT_SENTINEL")
    endpoint, payload = captured_backend[-1]
    assert endpoint == "/api/reason"
    assert payload["problem"] == expected
    assert payload["context"] == "FACT_SENTINEL"


def test_empty_reason_input_is_terminal(captured_backend):
    result = mcp_server.local_ai_task(action="reason", task=" ", prompt=" ")
    assert result["success"] is False
    assert result["terminal"] is True
    assert result["retryable"] is False
    assert captured_backend == []


def test_search_scope_reaches_backend(captured_backend, tmp_path):
    mcp_server.local_ai_repo(action="search", root=str(tmp_path), path="assets/scss", query="border-top")
    endpoint, payload = captured_backend[-1]
    assert endpoint == "/api/search"
    assert payload["path"] == "assets/scss"


def test_reason_quality_checked_without_paths(monkeypatch, captured_backend):
    monkeypatch.setattr(mcp_server.CLIENT, "post", lambda *a, **kw: {"success": True, "text": "pending"})
    result = mcp_server.local_ai_task(
        action="reason", task="Explain the supplied facts", response_profile="compact", max_response_tokens=1000,
    )
    assert result["advisory_only"] is True
    assert result["semantic_quality"]["usable"] is False
    assert result["bypass_reason"] == "queued"
    assert result["quality_warning"]


def _services(tmp_path, candidates=()):
    service = LocalAIServices.__new__(LocalAIServices)
    service.config = {}
    service.repo_tools = RepositoryTools(load_config())
    service.learner = None
    service.deterministic = SimpleNamespace(related_paths=lambda *args: list(candidates))
    service.code_index = None
    service.preprocessor = None
    service.evidence_store = None
    service.keys = []

    def cached(action, root, key, compute):
        service.keys.append(key)
        return compute()

    service._repo_cached = cached
    return service


def test_scoped_search_falls_back_from_irrelevant_candidates(tmp_path):
    (tmp_path / "assets" / "scss").mkdir(parents=True)
    (tmp_path / "assets" / "scss" / "site.scss").write_text("border-top: 1px solid;\n")
    (tmp_path / "rabbitmq-topology.php").write_text("topic topology\n")
    service = _services(tmp_path, ["rabbitmq-topology.php"])
    result = service.repo_search(str(tmp_path), "border-top", path="assets/scss")
    assert result["success"] is True
    assert {hit["path"] for hit in result["results"]} == {"assets/scss/site.scss"}
    assert result["preprocessed_hit"] is False


def test_search_cache_keys_include_scope(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    service = _services(tmp_path)
    service.repo_search(str(tmp_path), "border-top", path="a")
    service.repo_search(str(tmp_path), "border-top", path="b")
    assert service.keys[0] != service.keys[1]


@pytest.mark.parametrize("path", ["missing", "../outside"])
def test_service_invalid_scope_is_terminal(tmp_path, path):
    result = _services(tmp_path).repo_search(str(tmp_path), "border-top", path=path)
    assert result["success"] is False
    assert result["terminal"] is True
    assert result["retryable"] is False


def test_service_quality_checked_without_paths():
    result = LocalAIServices._apply_semantic_quality(
        {"success": True, "text": "pending"}, task="Explain facts", evidence_paths=(),
    )
    assert result["semantic_quality"]["usable"] is False
    assert result["advisory_only"] is True


@pytest.mark.parametrize("arguments,expected", [
    ({"prompt": "PROMPT_SENTINEL"}, "PROMPT_SENTINEL"),
    ({"problem": " ", "task": "TASK_SENTINEL", "prompt": "PROMPT_SENTINEL"}, "TASK_SENTINEL"),
    ({"problem": "PROBLEM_SENTINEL", "task": "TASK_SENTINEL"}, "PROBLEM_SENTINEL"),
])
def test_http_reason_assignment_normalization(arguments, expected):
    service = LocalAIServices.__new__(LocalAIServices)
    service.delegate = lambda payload, tenant: payload
    assert service.reason(arguments, "test")["task"] == expected


def test_http_reason_empty_input_never_runs_model():
    service = LocalAIServices.__new__(LocalAIServices)
    service.delegate = lambda *args: pytest.fail("empty reasoning must not run inference")
    result = service.reason({"problem": " ", "prompt": " "}, "test")
    assert result["success"] is False
    assert result["terminal"] is True
    assert result["retryable"] is False


def test_quality_warning_survives_small_response_budget(monkeypatch, captured_backend):
    monkeypatch.setattr(mcp_server.CLIENT, "post", lambda *a, **kw: {"success": True, "text": "pending"})
    result = mcp_server.local_ai_task(
        action="reason", task="Explain facts", response_profile="minimal", max_response_tokens=128,
    )
    assert result["advisory_only"] is True
    assert result["semantic_quality"]["usable"] is False
    assert result["bypass_reason"] == "queued"


def test_quality_guard_uses_paths_from_context():
    result = LocalAIServices._apply_semantic_quality(
        {"success": True, "text": "src/auth.py validates the supplied token."},
        task="Explain the supplied source", evidence_paths=(), context="src/auth.py: token validation source",
    )
    assert result["semantic_quality"]["usable"] is True
    assert "bypass_reason" not in result


def test_service_quality_reassessment_clears_obsolete_warnings():
    result = LocalAIServices._apply_semantic_quality(
        {"success": True, "text": "src/auth.py validates the supplied token.",
         "quality_warning": "old warning", "bypass_reason": "unrelated_output"},
        task="Explain source", evidence_paths=(), context="src/auth.py: token validation source",
    )
    assert result["semantic_quality"]["usable"] is True
    assert "quality_warning" not in result
    assert "bypass_reason" not in result


def test_quality_guard_reassessment_clears_obsolete_warnings():
    result = mcp_server._quality_check_semantic_result(
        {"success": True, "text": "src/auth.py validates the supplied token.",
         "quality_warning": "old warning", "bypass_reason": "unrelated_output"},
        task="Explain source", context="src/auth.py: token validation source",
    )
    assert result["semantic_quality"]["usable"] is True
    assert "quality_warning" not in result
    assert "bypass_reason" not in result
