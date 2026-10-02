from __future__ import annotations

import importlib.util
import io
import json
import socket
import sqlite3
import subprocess
import threading
import time
from contextlib import closing
from pathlib import Path
from urllib.error import HTTPError

import pytest

from local_ai_hub.cache import SQLiteCache
from local_ai_hub.client import HubClient
from local_ai_hub.generator import generate_skill_markdown
from local_ai_hub.http_server import Handler, LocalAIHTTPServer
from local_ai_hub.repo_state import RepoStateTracker
from local_ai_hub.resilience import RecoveryJournal
from local_ai_hub.scheduler import AffinityScheduler, ModelUnavailableError

ROOT = Path(__file__).resolve().parents[1]


def _config_path(tmp_path: Path, extra: str = "") -> Path:
    path = tmp_path / "config.toml"
    path.write_text(
        f'''[server]\nbind="127.0.0.1"\nport=11435\nstate_dir="{(tmp_path / 'state').as_posix()}"\nauto_start_ollama=false\n\n[hardware]\nprofile="cpu"\nauto_tune=false\n\n[prewarm]\nenabled=false\n\n[preprocessing]\nenabled=false\n\n{extra}\n''',
        encoding="utf-8",
    )
    return path


def test_sqlite_busy_is_not_treated_as_corruption(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cache = SQLiteCache(tmp_path / "cache.sqlite3", "sample", busy_timeout_seconds=0.01, busy_retries=1)
    cache.set("kept", {"value": 1})
    recovery_calls = {"n": 0}

    def forbidden_recovery() -> None:
        recovery_calls["n"] += 1
        raise AssertionError("transient SQLite busy must not invoke corruption recovery")

    monkeypatch.setattr(cache, "_recover", forbidden_recovery)
    blocker = sqlite3.connect(cache.path, timeout=0.01)
    try:
        blocker.execute("BEGIN IMMEDIATE")
        cache.set("dropped-while-busy", {"value": 2})
        assert recovery_calls["n"] == 0
        assert cache.busy_fallbacks >= 1
    finally:
        blocker.rollback()
        blocker.close()
    assert cache.get("kept") == {"value": 1}


def test_sqlite_l2_recency_write_is_sampled_once_per_process(tmp_path: Path):
    cache = SQLiteCache(tmp_path / "cache.sqlite3", "sample")
    cache.set("a", {"value": 1})
    for _ in range(5):
        assert cache.get("a") == {"value": 1}
    with closing(sqlite3.connect(cache.path)) as con:
        hits = con.execute("SELECT hits FROM cache_entries WHERE namespace=? AND cache_key='a'", (cache.namespace,)).fetchone()[0]
    assert hits == 1


class _RejectSocket:
    def __init__(self):
        self.payload = bytearray()
        self.closed = False

    def sendall(self, data: bytes) -> None:
        self.payload.extend(data)

    def shutdown(self, _how: int) -> None:
        pass

    def close(self) -> None:
        self.closed = True


def test_http_admission_gate_rejects_without_spawning_handler_thread():
    server = LocalAIHTTPServer(("127.0.0.1", 0), Handler, max_handlers=4, overload_wait_seconds=0)
    try:
        for _ in range(4):
            assert server._handler_slots.acquire(blocking=False)
        request = _RejectSocket()
        server.process_request(request, ("127.0.0.1", 1))
        assert b"503 Service Unavailable" in bytes(request.payload)
        assert b"Retry-After: 1" in bytes(request.payload)
        assert request.closed is True
        assert server.concurrency_stats()["rejected"] == 1
    finally:
        for _ in range(4):
            try:
                server._handler_slots.release()
            except ValueError:
                break
        server.server_close()


class _FailingRuntime:
    def loaded_model_details(self):
        return []

    def prepare_model(self, *_args, **_kwargs):
        raise RuntimeError("synthetic load failure")


def test_model_prepare_failure_fails_queue_and_opens_fast_circuit(tmp_path: Path):
    config = {
        "scheduler": {
            "max_parallel": 1,
            "max_queue": 8,
            "max_queued_per_tenant": 8,
            "max_inflight_per_tenant": 1,
            "model_switch_failure_cooldown_seconds": 5,
            "max_loaded_models": 1,
        },
        "preprocessing": {"idle_grace_seconds": 0, "model_switch_idle_seconds": 0},
        "models": {},
        "model_execution": {},
    }
    scheduler = AffinityScheduler(_FailingRuntime(), config)
    try:
        result = scheduler.submit("broken", "tenant", "test", lambda: {"success": True}, wait_timeout=2)
        assert result["success"] is False
        assert "model preparation failed" in result["error"]
        with pytest.raises(ModelUnavailableError):
            scheduler.submit("broken", "tenant", "test", lambda: {"success": True}, wait_timeout=1)
        assert scheduler.status()["stats"]["model_switch_failures"] >= 1
        assert scheduler.status()["stats"]["model_fast_rejections"] >= 1
    finally:
        scheduler.close()


def test_repo_fingerprint_singleflight_is_per_root_not_global(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    config = {
        "workspace_cache": {"fingerprint_ttl_seconds": 0.1, "git_probe_timeout_seconds": 1, "git_status_timeout_seconds": 1},
        "search": {"max_files": 100},
        "rag": {"ignore_dirs": [".git"]},
    }
    roots = [tmp_path / "a", tmp_path / "b"]
    for root in roots:
        root.mkdir()
    tracker = RepoStateTracker(config)
    barrier = threading.Barrier(2)

    def fake_git(root: Path, *args: str, timeout: float = 8.0):
        del root, timeout
        if args == ("rev-parse", "--is-inside-work-tree"):
            barrier.wait(timeout=1.5)
            return subprocess.CompletedProcess([], 0, b"true\n", b"")
        if args == ("rev-parse", "HEAD"):
            return subprocess.CompletedProcess([], 0, b"abc123\n", b"")
        if args and args[0] == "status":
            return subprocess.CompletedProcess([], 0, b"", b"")
        raise AssertionError(args)

    monkeypatch.setattr(tracker, "_run_git", fake_git)
    results: list[dict] = []
    errors: list[BaseException] = []

    def run(root: Path) -> None:
        try:
            results.append(tracker.fingerprint(str(root), force=True))
        except BaseException as exc:  # assertion aid for thread failures
            errors.append(exc)

    threads = [threading.Thread(target=run, args=(root,)) for root in roots]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(3)
    assert not errors
    assert len(results) == 2
    assert all(result["kind"] == "git" for result in results)


def test_client_caps_get_timeout_and_treats_overload_as_live(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cfg = _config_path(tmp_path, "[client]\nmax_request_timeout_seconds=5\nhealth_timeout_seconds=0.2\nstartup_wait_seconds=1\n")
    client = HubClient(config_path=str(cfg), auto_start=False)

    def overloaded(*_args, **_kwargs):
        raise HTTPError(client.base_url + "/health", 503, "busy", {}, io.BytesIO(b"{}"))

    import local_ai_hub.client as client_module
    monkeypatch.setattr(client_module, "urlopen", overloaded)
    assert client._online() is True

    seen: list[float] = []

    def ok(_method, _path, _body, _headers, timeout):
        seen.append(float(timeout))
        return b'{"success":true}'

    monkeypatch.setattr(client, "_pooled_open", ok)
    assert client.get("/health", timeout=999)["success"] is True
    assert seen == [5.0]


def test_setup_supports_explicit_generic_mcp_config_paths(tmp_path: Path):
    spec = importlib.util.spec_from_file_location("local_ai_hub_setup_v15", ROOT / "tools" / "setup.py")
    assert spec and spec.loader
    setup = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(setup)
    generic = tmp_path / "custom" / "mcp.json"
    vscode = tmp_path / "custom" / "vscode.json"
    cfg = {
        "setup": {"backup_existing_configs": False},
        "tool_policy": {"install_global_instructions": False},
        "code_intelligence": {"direct_agent_mcp": False},
        "agents": {
            "agent_skills_standard": False, "codex": False, "claude": False, "gemini": False,
            "cursor": False, "windsurf": False, "copilot": False,
            "extra_mcp_json_paths": [str(generic)], "extra_vscode_mcp_paths": [str(vscode)],
        },
    }
    install = tmp_path / "install"
    install.mkdir()
    setup.configure_agents(install, tmp_path / "python", None, None, cfg)
    assert json.loads(generic.read_text(encoding="utf-8"))["mcpServers"]["local-ai"]["env"]["LOCAL_AI_AGENT"] == "generic"
    assert json.loads(vscode.read_text(encoding="utf-8"))["servers"]["local-ai"]["type"] == "stdio"
    assert (install / "generated" / "agent-policy.md").read_text(encoding="utf-8") == setup.GLOBAL_POLICY + "\n"
    generated = json.loads((install / "generated" / "mcp-servers.json").read_text(encoding="utf-8"))
    assert generated["mcpServers"]["local-ai"]["env"]["LOCAL_AI_AGENT_PROFILE"] == "generic"
    assert generated["mcpServers"]["local-ai"]["args"] == ["-m", "local_ai_hub.mcp_server"]


def test_sqlite_shared_cache_initializes_same_path_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import local_ai_hub.cache as cache_module
    db = tmp_path / "shared.sqlite3"
    calls = {"n": 0}
    original = cache_module.SQLiteCache._create_schema

    def counted(self):
        calls["n"] += 1
        return original(self)

    monkeypatch.setattr(cache_module.SQLiteCache, "_create_schema", counted)
    SQLiteCache(db, "one")
    SQLiteCache(db, "two")
    assert calls["n"] == 1


class _HealthyRuntime:
    def loaded_model_details(self):
        return []

    def prepare_model(self, *_args, **_kwargs):
        return None


def test_scheduler_omitted_wait_timeout_is_still_hard_bounded():
    import time
    config = {
        "scheduler": {
            "max_parallel": 1,
            "max_queue": 8,
            "max_queued_per_tenant": 8,
            "max_inflight_per_tenant": 1,
            "max_caller_wait_timeout_seconds": 0.08,
            "max_loaded_models": 1,
        },
        "resilience": {"scheduler_wait_timeout_seconds": 0.06},
        "preprocessing": {"idle_grace_seconds": 0, "model_switch_idle_seconds": 0},
        "models": {},
        "model_execution": {},
    }
    scheduler = AffinityScheduler(_HealthyRuntime(), config)
    started = time.monotonic()
    try:
        with pytest.raises(TimeoutError) as exc_info:
            scheduler.submit("slow", "tenant", "test", lambda: (time.sleep(0.3) or {"success": True}))
        assert time.monotonic() - started < 0.2
        assert exc_info.value.error_code == "scheduler_caller_timeout"
        assert exc_info.value.state in {"running", "dispatching"}
        assert exc_info.value.retryable is True
        assert exc_info.value.job_id > 0
        assert scheduler.status()["stats"]["caller_timeouts"] >= 1
    finally:
        scheduler.close()


def test_client_preserves_retryable_http_error_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cfg = _config_path(tmp_path, "[client]\nmax_request_timeout_seconds=5\n")
    client = HubClient(config_path=str(cfg), auto_start=False)

    import local_ai_hub.client as client_module

    def overloaded(*_args, **_kwargs):
        raise HTTPError(
            client.base_url + "/api/status",
            503,
            "busy",
            {},
            io.BytesIO(b'{"success":false,"error":"busy"}'),
        )

    monkeypatch.setattr(client, "_pooled_open", overloaded)
    result = client.get("/api/status")
    assert result["status_code"] == 503
    assert result["retryable"] is True
    assert result["request_id"].startswith("req_")


def test_mcp_stdio_discards_stale_response_instead_of_requeueing(monkeypatch: pytest.MonkeyPatch):
    from local_ai_hub.external_tools import MCPStdioClient

    class Proc:
        def poll(self): return None

    client = MCPStdioClient(["unused"], call_timeout=1)
    client._proc = Proc()  # type: ignore[assignment]
    monkeypatch.setattr(client, "_send", lambda _message: None)
    client._responses.put_nowait({"jsonrpc": "2.0", "id": 999, "result": {"stale": True}})
    client._responses.put_nowait({"jsonrpc": "2.0", "id": 1, "result": {"ok": True}})
    assert client._request("x", {}, ensure_started=False) == {"ok": True}
    assert client._stale_responses == 1
    assert client._responses.empty()


def test_request_body_timeout_is_total_deadline_not_idle_timeout(monkeypatch: pytest.MonkeyPatch):
    import time
    import local_ai_hub.http_server as http_module
    from local_ai_hub.http_server import RequestBodyError

    class Connection:
        def __init__(self): self.timeout = None
        def gettimeout(self): return self.timeout
        def settimeout(self, value): self.timeout = value

    class SlowReader:
        def read1(self, _n):
            time.sleep(0.035)
            return b"x"

    handler = object.__new__(Handler)
    handler.connection = Connection()
    handler.rfile = SlowReader()
    handler._content_length = lambda *, limit: 10
    fake_app = type("App", (), {"config": {"server": {"request_body_timeout_seconds": 0.08}}})()
    monkeypatch.setattr(http_module, "APP", fake_app)
    started = time.monotonic()
    with pytest.raises(RequestBodyError) as exc:
        handler._read_body(limit=1024)
    assert exc.value.status == 408
    assert time.monotonic() - started < 0.2


def test_singleflight_waiter_returns_retryable_in_progress_quickly():
    import time
    from local_ai_hub.cache import MemoryLRUCache, SingleFlightCache

    cache = SingleFlightCache(MemoryLRUCache(8, 60), wait_timeout_seconds=0.04)
    owner_started = threading.Event()

    def compute():
        owner_started.set()
        time.sleep(0.15)
        return {"success": True, "value": 1}

    owner_result: list[dict] = []
    owner = threading.Thread(target=lambda: owner_result.append(cache.get_or_compute("same", compute)[0]))
    owner.start()
    assert owner_started.wait(0.5)
    started = time.monotonic()
    waiter, hit, coalesced = cache.get_or_compute("same", compute)
    elapsed = time.monotonic() - started
    owner.join(1)

    assert elapsed < 0.12
    assert hit is False and coalesced is True
    assert waiter["in_progress"] is True and waiter["retryable"] is True
    assert "do not start a duplicate" in waiter["error"]
    assert cache.stats()["coalesced_timeouts"] == 1
    assert owner_result == [{"success": True, "value": 1}]


def test_recovery_journal_waiter_replays_completed_owner_response(tmp_path: Path):
    journal = RecoveryJournal(tmp_path / "state")
    journal.begin("request-1", "tenant", "/api/search")

    def finish() -> None:
        time.sleep(0.02)
        journal.finish("request-1", True, status_code=200, response={"success": True, "value": "reused"})

    worker = threading.Thread(target=finish)
    worker.start()
    result = journal.wait_for("request-1", "tenant", "/api/search", timeout_seconds=0.5)
    worker.join(1)

    assert result == {"state": "done", "status_code": 200, "response": {"success": True, "value": "reused"}}


def test_recovery_journal_repairs_running_row_with_persisted_response(tmp_path: Path):
    journal = RecoveryJournal(tmp_path / "state")
    journal.begin("request-repair", "tenant", "/api/search")
    with closing(journal._connect()) as con:
        con.execute(
            "UPDATE requests SET status_code=200,response_json=? WHERE request_id=?",
            ('{"success":true,"value":"persisted"}', "request-repair"),
        )
        con.commit()

    result = journal.lookup("request-repair", "tenant", "/api/search")

    assert result is not None
    assert result["state"] == "done"
    assert result["status_code"] == 200
    assert result["response"] == {"success": True, "value": "persisted"}
    with closing(journal._connect()) as con:
        state = con.execute("SELECT state FROM requests WHERE request_id=?", ("request-repair",)).fetchone()[0]
    assert state == "done"

    journal.begin("request-startup-repair", "tenant", "/api/search")
    with closing(journal._connect()) as con:
        con.execute(
            "UPDATE requests SET status_code=200,response_json=? WHERE request_id=?",
            ('{"success":true,"value":"startup-persisted"}', "request-startup-repair"),
        )
        con.commit()

    restarted = RecoveryJournal(tmp_path / "state")
    recovered = restarted.lookup("request-startup-repair", "tenant", "/api/search")
    assert recovered is not None
    assert recovered["state"] == "done"
    assert recovered["response"] == {"success": True, "value": "startup-persisted"}


def test_memory_lru_preserves_subsecond_ttl():
    import time
    from local_ai_hub.cache import MemoryLRUCache

    cache = MemoryLRUCache(4, 0.03)
    cache.set("x", 1)
    assert cache.get("x") == 1
    time.sleep(0.06)
    assert cache.get("x") is None


def test_repo_fingerprint_flight_locks_are_released(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    config = {
        "workspace_cache": {"fingerprint_ttl_seconds": 0.1, "git_probe_timeout_seconds": 1, "git_status_timeout_seconds": 1},
        "search": {"max_files": 100},
        "rag": {"ignore_dirs": [".git"]},
    }
    tracker = RepoStateTracker(config)
    monkeypatch.setattr(
        tracker,
        "_run_git",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], 128, b"", b""),
    )
    for idx in range(20):
        root = tmp_path / f"repo-{idx}"
        root.mkdir()
        (root / "a.txt").write_text(str(idx), encoding="utf-8")
        assert tracker.fingerprint(str(root), force=True)["success"] is True
    assert tracker.stats()["fingerprint_flights"] == 0


def test_pid_alive_treats_permission_denied_as_alive(monkeypatch: pytest.MonkeyPatch):
    import local_ai_hub.process_utils as process_utils

    if process_utils.os.name == "nt":
        pytest.skip("POSIX-specific permission semantics")

    def denied(_pid: int, _sig: int):
        raise PermissionError("synthetic")

    monkeypatch.setattr(process_utils.os, "kill", denied)
    assert process_utils.pid_alive(12345) is True


def _ollama_config(tmp_path: Path) -> dict:
    return {
        "server": {"ollama_url": "http://127.0.0.1:11434", "request_timeout_seconds": 1, "state_dir": str(tmp_path)},
        "headless": {"autostart_ollama": True},
        "ollama": {"enabled": True, "request_attempts": 3, "retry_delay_seconds": 0.02, "startup_timeout_seconds": 1},
        "scheduler": {"max_parallel": 1, "max_loaded_models": 1},
        "models": {}, "model_execution": {},
    }


def test_ollama_interruptible_stream_exists_and_aggregates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import local_ai_hub.ollama as ollama_module
    from local_ai_hub.ollama import OllamaRuntime

    class Response:
        def __enter__(self): return self
        def __exit__(self, *_args): return False
        def __iter__(self):
            return iter([
                b'{"response":"hel","done":false}\n',
                b'{"response":"lo","done":true,"eval_count":2}\n',
            ])

    monkeypatch.setattr(ollama_module, "urlopen", lambda *_args, **_kwargs: Response())
    runtime = OllamaRuntime(_ollama_config(tmp_path))
    result = runtime.request_interruptible("/api/generate", {"model": "x"}, lambda: False, timeout=1)
    assert result["response"] == "hello"
    assert result["eval_count"] == 2
    assert result["_lah_retry_count"] == 0


def test_ollama_request_retries_share_one_total_deadline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import time
    from urllib.error import URLError
    import local_ai_hub.ollama as ollama_module
    from local_ai_hub.ollama import OllamaRuntime

    timeouts: list[float] = []

    def unavailable(_req, timeout):
        timeouts.append(float(timeout))
        raise URLError("synthetic")

    monkeypatch.setattr(ollama_module, "urlopen", unavailable)
    runtime = OllamaRuntime(_ollama_config(tmp_path))
    started = time.monotonic()
    result = runtime.request("/api/version", timeout=0.06)
    elapsed = time.monotonic() - started
    assert "error" in result
    assert elapsed < 0.15
    assert len(timeouts) >= 2
    assert all(b <= a + 1e-6 for a, b in zip(timeouts, timeouts[1:]))


def test_ollama_autostart_is_singleflight_and_posix_grouped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import time
    import local_ai_hub.ollama as ollama_module
    from local_ai_hub.ollama import OllamaRuntime

    runtime = OllamaRuntime(_ollama_config(tmp_path))
    state = {"online": False, "spawns": 0, "kwargs": None}

    monkeypatch.setattr(runtime, "is_online", lambda: bool(state["online"]))

    class Proc:
        pid = 424242
        def poll(self): return None

    def popen(*_args, **kwargs):
        state["spawns"] += 1
        state["kwargs"] = kwargs
        time.sleep(0.04)
        state["online"] = True
        return Proc()

    monkeypatch.setattr(ollama_module.subprocess, "Popen", popen)
    results: list[bool] = []
    threads = [threading.Thread(target=lambda: results.append(runtime.ensure_running())) for _ in range(2)]
    for thread in threads: thread.start()
    for thread in threads: thread.join(1)
    assert results == [True, True]
    assert state["spawns"] == 1
    if ollama_module.os.name != "nt":
        assert state["kwargs"]["start_new_session"] is True


def test_real_ollama_runtime_exposes_all_scheduler_and_background_methods(tmp_path: Path):
    import ast
    from local_ai_hub.ollama import OllamaRuntime

    methods = {name for name in dir(OllamaRuntime) if callable(getattr(OllamaRuntime, name, None))}
    refs: set[str] = set()
    for filename in ("scheduler.py", "background_gpu.py", "preprocess.py"):
        tree = ast.parse((ROOT / "src" / "local_ai_hub" / filename).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            value = node.func.value
            if isinstance(value, ast.Attribute) and isinstance(value.value, ast.Name) and value.value.id == "self" and value.attr == "runtime":
                refs.add(node.func.attr)
    assert refs <= methods, f"OllamaRuntime missing methods used by callers: {sorted(refs - methods)}"


def test_prepare_model_reloads_small_context_and_evicts_others(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from local_ai_hub.ollama import OllamaRuntime

    cfg = _ollama_config(tmp_path)
    cfg["models"] = {"fast_code": "fast", "general": "fast"}
    cfg["model_execution"] = {"fast": {"context_tokens": 8192, "large_context_tokens": 8192, "max_context_tokens": 8192}}
    runtime = OllamaRuntime(cfg)
    monkeypatch.setattr(runtime, "ensure_running", lambda: True)
    monkeypatch.setattr(runtime, "loaded_model_details", lambda: [
        {"name": "fast", "context_length": 4096},
        {"name": "other", "context_length": 4096},
    ])
    unloaded: list[str] = []
    monkeypatch.setattr(runtime, "unload_model", lambda model: (unloaded.append(model) or True))
    payloads: list[dict] = []
    monkeypatch.setattr(runtime, "request", lambda _endpoint, payload, timeout=None: (payloads.append(payload) or {"done": True}))

    evicted = runtime.prepare_model("fast", unload_others=True)
    assert evicted == ["other"]
    assert unloaded == ["other", "fast"]
    assert payloads[0]["options"]["num_ctx"] == 8192


def test_embedding_model_residency_uses_embed_api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from local_ai_hub.ollama import OllamaRuntime

    cfg = _ollama_config(tmp_path)
    cfg["models"] = {"embedding": "qwen3-embedding:0.6b"}
    runtime = OllamaRuntime(cfg)
    monkeypatch.setattr(runtime, "ensure_running", lambda: True)
    monkeypatch.setattr(runtime, "loaded_model_details", lambda: [])
    requests: list[tuple[str, dict]] = []

    def request(endpoint: str, payload: dict, timeout=None):
        requests.append((endpoint, payload))
        return {"embeddings": [[0.0]]} if endpoint == "/api/embed" else {"done": True}

    monkeypatch.setattr(runtime, "request", request)
    runtime.prepare_model("qwen3-embedding:0.6b")

    assert requests == [
        ("/api/embed", {"model": "qwen3-embedding:0.6b", "input": [""], "keep_alive": "24h"})
    ]


def test_code_index_production_contract_methods_use_persistent_index(tmp_path: Path):
    from local_ai_hub.code_index import CodeIndex
    from local_ai_hub.config import load_config
    from local_ai_hub.repo_tools import RepositoryTools

    cfg = load_config(str(_config_path(tmp_path)))
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "tests").mkdir()
    (repo / "src" / "service.py").write_text(
        "def calculate_total(value):\n    return value + 1\n", encoding="utf-8"
    )
    (repo / "src" / "consumer.py").write_text(
        "from src.service import calculate_total\n\ndef render():\n    return calculate_total(4)\n", encoding="utf-8"
    )
    (repo / "tests" / "test_service.py").write_text(
        "from src.service import calculate_total\n\ndef test_total():\n    assert calculate_total(1) == 2\n", encoding="utf-8"
    )
    tools = RepositoryTools(cfg)
    index = CodeIndex(cfg, tools)
    paths = ["src/service.py", "src/consumer.py", "tests/test_service.py"]
    built = index.update_files_batch(str(repo), paths)
    assert built["success"] is True

    summary = index.file_summary(str(repo), "src/service.py")
    assert summary["success"] is True
    assert any(item["name"] == "calculate_total" for item in summary["symbols"])

    related = index.related_paths(str(repo), "calculate_total", 10)
    assert "src/service.py" in related
    assert any(path in related for path in ("src/consumer.py", "tests/test_service.py"))

    impact = index.impact(str(repo), ["src/service.py"], 20, 20)
    assert impact["success"] is True
    assert impact["confidence"] >= 0.7
    assert any(item["name"] == "calculate_total" for item in impact["changed_symbols"])
    assert any(item["path"] == "src/consumer.py" for item in impact["likely_dependents"])
    assert any(item["path"] == "tests/test_service.py" for item in impact["suggested_tests"])

    pruned = index.prune(str(repo), ["src/service.py"])
    assert pruned["success"] is True
    assert pruned["pruned"] == 2
    assert index.file_summary(str(repo), "src/consumer.py")["indexed"] is False


def test_internal_runtime_and_code_index_call_sites_match_real_contracts():
    import re
    from local_ai_hub.code_index import CodeIndex
    from local_ai_hub.ollama import OllamaRuntime

    checks = {
        "runtime": (OllamaRuntime, ["scheduler.py", "background_gpu.py", "preprocess.py"]),
        "code_index": (CodeIndex, ["services.py", "preprocess.py", "deterministic.py"]),
    }
    src = ROOT / "src" / "local_ai_hub"
    for attr, (cls, files) in checks.items():
        methods = {name for name in dir(cls) if callable(getattr(cls, name, None))}
        missing: list[str] = []
        pattern = re.compile(rf"self\.{attr}\.([A-Za-z_][A-Za-z0-9_]*)\s*\(")
        for filename in files:
            text = (src / filename).read_text(encoding="utf-8")
            for method in sorted(set(pattern.findall(text))):
                if method not in methods:
                    missing.append(f"{filename}: self.{attr}.{method}()")
        assert not missing, "component contract drift: " + ", ".join(missing)


def test_code_index_impact_batches_large_changed_file_sets(tmp_path: Path):
    from local_ai_hub.code_index import CodeIndex
    from local_ai_hub.config import load_config
    from local_ai_hub.repo_tools import RepositoryTools

    cfg = load_config(str(_config_path(tmp_path)))
    repo = tmp_path / "repo-large"
    repo.mkdir()
    index = CodeIndex(cfg, RepositoryTools(cfg))
    changed = [f"src/file_{i}.py" for i in range(1200)]
    result = index.impact(str(repo), changed, 20, 20)
    assert result["success"] is True
    assert result["changed_files_count"] == 1200
    assert result["indexed_changed_files"] == 0


def test_agent_policy_is_consistent_and_has_stop_reuse_protocol():
    spec = importlib.util.spec_from_file_location("local_ai_hub_setup_policy_v15", ROOT / "tools" / "setup.py")
    assert spec and spec.loader
    setup = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(setup)
    policy = setup.GLOBAL_POLICY
    agents = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    skill = generate_skill_markdown({})
    assert setup.GLOBAL_POLICY_BEGIN in agents
    assert setup.GLOBAL_POLICY_END in agents
    for phrase in ("Stop escalating", "in_progress=true", "Do not fan out", "one bounded health/retry attempt"):
        assert phrase in agents
    for phrase in ("Stop escalating", "in_progress=true", "Do not fan out", "one bounded health/retry attempt"):
        assert phrase in policy
    assert "Stop escalating" in skill
    assert "retryable`/429/503" in skill


def test_agent_policy_has_actionable_adoption_triggers_and_recipes():
    spec = importlib.util.spec_from_file_location("local_ai_hub_setup_adoption", ROOT / "tools" / "setup.py")
    assert spec and spec.loader
    setup = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(setup)
    policy = setup.GLOBAL_POLICY
    expected = (
        "Trigger map:",
        "repository facts/files/symbols: `local_ai_repo`",
        "tests/lint/typecheck/build: `local_ai_command`",
        "exact source/evidence text: `local_ai_artifact`",
        "shared findings or overlapping edits: `local_ai_coord`",
        "semantic retrieval after indexed paths are insufficient: `local_ai_rag`",
        'semantic generation, exploration, reasoning, review, second opinion and compression: `local_ai_task(action="delegate"|"explore"|"reason"|"review"|"second_opinion"|"compress")`',
        "Recipe — Explore:",
        "Recipe — Change:",
        "Recipe — Validate:",
        "guidance, not gates",
    )
    for phrase in expected:
        assert phrase in policy

    skill = generate_skill_markdown({})
    for phrase in expected:
        assert phrase in skill


def test_mcp_tools_have_use_when_and_skip_when_cues():
    source = (ROOT / "src" / "local_ai_hub" / "mcp_server.py").read_text(encoding="utf-8")
    for tool in (
        "local_ai_status", "local_ai_task", "local_ai_repo", "local_ai_rag",
        "local_ai_command", "local_ai_coord", "local_ai_artifact",
    ):
        start = source.index(f"def {tool}(")
        end = source.find("\n\n@mcp.tool()", start)
        block = source[start:] if end < 0 else source[start:end]
        assert "Use when:" in block, f"{tool}: missing 'Use when:' in docstring"
        assert "Skip when:" in block, f"{tool}: missing 'Skip when:' in docstring"


def test_agent_policy_has_default_delegation_triggers():
    spec = importlib.util.spec_from_file_location("local_ai_hub_setup_delegation", ROOT / "tools" / "setup.py")
    assert spec and spec.loader
    setup = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(setup)
    expected = (
        "Delegation is the default for any task with useful bounded independent work.",
        "Use `local_ai_task` for bounded semantic generation, reasoning, review, independent second opinions, and semantic compression.",
        "Use the native Codex `multi_agent_v1__spawn_agent` path only for useful independent bounded work or an explicit Codex-subagent request.",
        "Codex controls each subagent's scope, `allow_write`, workspace/worktree, timeout, cancellation, sandbox, and integration.",
        "Do not duplicate the same scope across agents.",
        "Skip delegation only for trivial tasks, pure evidence lookups, security/privacy constraints, or no useful independent scope.",
    )
    policy = setup.GLOBAL_POLICY
    skill = generate_skill_markdown({})
    for phrase in expected:
        assert phrase in policy
        assert phrase in skill


def test_hub_policy_excludes_removed_external_agent_route():
    spec = importlib.util.spec_from_file_location("local_ai_hub_setup_external_agent_boundary", ROOT / "tools" / "setup.py")
    assert spec and spec.loader
    setup = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(setup)
    policy = setup.GLOBAL_POLICY
    assert "AGY" not in policy
    assert "mcp__agy" not in policy
    assert "allow_write=false" not in policy


def test_mcp_surface_preserves_eight_tools_with_agent_state_actions():
    source = (ROOT / "src" / "local_ai_hub" / "mcp_server.py").read_text(encoding="utf-8")
    tool_defs = [line for line in source.splitlines() if line.startswith("def local_ai_")]
    assert len(tool_defs) == 8
    assert "task_create" in source
    assert "context_compile" in source
    assert "candidate_create" in source
    assert "agent_state" in source
