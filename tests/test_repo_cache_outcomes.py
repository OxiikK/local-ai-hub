from __future__ import annotations

from pathlib import Path

from local_ai_hub.services import LocalAIServices


class _Flight:
    def get_or_compute(self, _key, compute):
        return compute(), True, True


def test_repo_cache_outcome_is_promoted_to_top_level_response():
    services = LocalAIServices.__new__(LocalAIServices)
    services.config = {"features": {"enriched_search": True}}
    services.repo_flight = _Flight()
    services._touch_project = lambda _root: None
    services._repo_cache_state = lambda _root: {"fingerprint": "rev", "kind": "filesystem"}

    result = services._repo_cached("search", "C:/repo", {"query": "x"}, lambda: {"success": True})

    assert result["cache_hit"] is True
    assert result["coalesced"] is True
    assert result["cache_layer"] == "workspace"


def test_repo_search_reuses_cache_for_equivalent_query_whitespace_and_case(tmp_path: Path):
    class _Cache:
        def __init__(self):
            self.values = {}

        def get_or_compute(self, key, compute):
            if key in self.values:
                return self.values[key], True, False
            value = compute()
            self.values[key] = value
            return value, False, False

    class _Tools:
        def search(self, _root, query, top_k, context_lines=None):
            return {"success": True, "query": query, "results": [{"path": "a.py"}]}

        def search_paths(self, _root, query, _paths, top_k, context_lines=None):
            return {"success": True, "query": query, "results": []}

    services = LocalAIServices.__new__(LocalAIServices)
    services.repo_flight = _Cache()
    services._touch_project = lambda _root: None
    services._repo_cache_state = lambda _root: {"fingerprint": "rev", "kind": "filesystem"}
    services.repo_tools = _Tools()
    services.learner = None
    services.deterministic = None
    services.code_index = None
    services.preprocessor = None
    services.evidence_store = None

    first = services.repo_search(str(tmp_path), "  Cache   Route  ", top_k=12)
    second = services.repo_search(str(tmp_path), "cache route", top_k=12)

    assert first["cache_hit"] is False
    assert second["cache_hit"] is True


def test_repo_search_reports_bounded_progress_stages(tmp_path: Path):
    class _Tools:
        def search(self, _root, query, top_k, context_lines=None):
            return {"success": True, "query": query, "results": []}

        def search_paths(self, *_args, **_kwargs):
            return {"success": True, "results": []}

    services = LocalAIServices.__new__(LocalAIServices)
    services.repo_flight = _Flight()
    services._touch_project = lambda _root: None
    services._repo_cache_state = lambda _root: {"fingerprint": "rev", "kind": "filesystem"}
    services.repo_tools = _Tools()
    services.learner = services.deterministic = services.code_index = services.preprocessor = None
    services.evidence_store = None
    stages: list[str] = []

    result = services.repo_search(str(tmp_path), "target", progress=stages.append)

    assert result["success"] is True
    assert stages == [
        "repo_cache.start",
        "full_search.start", "full_search.done", "repo_cache.done",
    ]


def test_enriched_repo_search_returns_symbol_and_evidence_for_hit(tmp_path: Path, monkeypatch):
    class _Tools:
        def search(self, _root, _query, _top_k, context_lines=None):
            assert context_lines == 8
            return {"success": True, "results": [{"path": "module.py", "start_line": 2, "text": "def target():"}]}

        def search_paths(self, *_args, **_kwargs):
            return {"success": True, "results": []}

    class _Evidence:
        def put_many(self, _root, hits):
            hits[0]["evidence_id"] = "E-search"
            return hits

    (tmp_path / "module.py").write_text("def target():\n    return 1\n", encoding="utf-8")
    monkeypatch.setattr(
        "local_ai_hub.services.enclosing_symbol_at_line",
        lambda _source, _path, _line: {"name": "target", "kind": "function", "line": 1, "end_line": 2, "name_path": "target"},
    )
    services = LocalAIServices.__new__(LocalAIServices)
    services.config = {"features": {"enriched_search": True}}
    services.repo_flight = _Flight()
    services._touch_project = lambda _root: None
    services._repo_cache_state = lambda _root: {"fingerprint": "rev", "kind": "filesystem"}
    services.repo_tools = _Tools()
    services.learner = services.deterministic = services.code_index = services.preprocessor = None
    services.evidence_store = _Evidence()

    result = services.repo_search(str(tmp_path), "target", enrich=True)

    assert result["enriched"] is True
    assert result["progressive_disclosure"] is True
    assert result["results"][0]["enclosing_symbol"]["name"] == "target"
    assert result["results"][0]["evidence_id"] == "E-search"


def test_foreground_refresh_skips_active_background_preprocessing(tmp_path: Path):
    class _Index:
        def __init__(self):
            self.calls = 0

        def update_files_batch(self, *_args):
            self.calls += 1
            return {}

    services = LocalAIServices.__new__(LocalAIServices)
    services._index_refresh_lock = __import__("threading").Lock()
    services._index_refresh_state = {}
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("print('x')", encoding="utf-8")
    services._repo_cache_state = lambda _root: {
        "kind": "preprocessed-watcher",
        "status": "running",
        "phase": "lexical",
        "fingerprint": "rev",
        "changed_paths": ["src/main.py"],
    }
    services.repo_tools = type("_Tools", (), {"_hash_file_only": lambda *_args: "hash"})()
    services.code_index = _Index()
    services.deterministic = _Index()

    services._refresh_changed_intelligence(str(tmp_path))

    assert services.code_index.calls == 0
    assert services.deterministic.calls == 0


def test_foreground_refresh_never_syncs_watcher_owned_index(tmp_path: Path):
    class _Index:
        def __init__(self):
            self.calls = 0

        def update_files_batch(self, *_args):
            self.calls += 1
            return {}

    services = LocalAIServices.__new__(LocalAIServices)
    services._index_refresh_lock = __import__("threading").Lock()
    services._index_refresh_state = {}
    (tmp_path / "main.py").write_text("print('x')", encoding="utf-8")
    services._repo_cache_state = lambda _root: {
        "kind": "preprocessed-watcher",
        "status": "complete",
        "phase": "complete",
        "fingerprint": "rev",
        "changed_paths": ["main.py"],
    }
    services.repo_tools = type("_Tools", (), {"_hash_file_only": lambda *_args: "hash"})()
    services.code_index = _Index()
    services.deterministic = _Index()

    services._refresh_changed_intelligence(str(tmp_path))

    assert services.code_index.calls == 0
    assert services.deterministic.calls == 0
