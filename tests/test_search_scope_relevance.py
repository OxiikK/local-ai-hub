import shutil
from types import SimpleNamespace

import pytest

from local_ai_hub.repo_tools import RepositoryTools


@pytest.fixture
def repo(tmp_path):
    files = {
        "assets/scss/site.scss": ".card { border-top: 1px solid; }\n.edge { border-top-left-radius: 2px; }\n",
        "assets/scss/topology.scss": "topic topology top border\n.card { xborder-top: 1px; }\n",
        "assets/scss/site2.scss": ".other { border-top: 2px; }\n",
        "vendor/topology.php": "topic topology top border\n",
        "vendor/styles.scss": ".card { border-top: 3px; }\n",
    }
    for rel, text in files.items():
        target = tmp_path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
    return tmp_path


@pytest.fixture
def tools():
    return RepositoryTools({"search": {"use_ripgrep_if_available": False, "use_git_grep_if_available": False}})


@pytest.mark.parametrize("method", ["search", "search_paths"])
@pytest.mark.parametrize("scope,expected", [
    ("assets/scss", {"assets/scss/site.scss", "assets/scss/site2.scss"}),
    ("assets/scss/site.scss", {"assets/scss/site.scss"}),
])
def test_scope_filters_literal_matches(repo, tools, method, scope, expected):
    kwargs = {"paths": [str(p.relative_to(repo)) for p in repo.rglob("*") if p.is_file()]} if method == "search_paths" else {}
    result = getattr(tools, method)(str(repo), "border-top", path=scope, **kwargs)
    assert result["success"]
    assert {hit["path"] for hit in result["results"]} == expected
    assert all("xborder-top" not in hit["text"] for hit in result["results"])


@pytest.mark.parametrize("method", ["search", "search_paths"])
@pytest.mark.parametrize("scope", ["missing", "../outside", "/outside-root"])
def test_invalid_scope_is_terminal(repo, tools, method, scope):
    kwargs = {"paths": ["assets/scss/site.scss"]} if method == "search_paths" else {}
    result = getattr(tools, method)(str(repo), "border-top", path=scope, **kwargs)
    assert result["success"] is False
    assert result["terminal"] is True
    assert result["retryable"] is False
    assert result["results"] == []


def test_literal_candidate_miss_allows_service_fallback(repo, tools):
    result = tools.search_paths(str(repo), "border-top", ["vendor/topology.php"])
    assert result["results"] == []


def test_exact_path_does_not_claim_arbitrary_source_lines(repo, tools):
    result = tools.search_paths(str(repo), "vendor/topology.php", ["vendor/topology.php"])
    assert result["results"] == []


def test_accelerated_results_are_revalidated(repo, tools, monkeypatch):
    monkeypatch.setattr(tools, "_ripgrep_candidates", lambda *args, **kwargs: ([
        ("assets/scss/site.scss", 1), ("assets/scss/topology.scss", 1),
        ("assets/scss/topology.scss", 2), ("vendor/styles.scss", 1),
        ("../outside", 1),
    ], False))
    result = tools.search(str(repo), "border-top", path="assets/scss")
    assert {hit["path"] for hit in result["results"]} == {"assets/scss/site.scss"}


def test_scope_normalization_is_stable(repo, tools):
    assert tools.normalize_search_scope(str(repo), "assets\\scss/../scss") == "assets/scss"
    assert tools.normalize_search_scope(str(repo), str(repo)) == ""


def test_multiword_search_requires_content_match(repo, tools):
    result = tools.search_paths(str(repo), "topology borders", ["assets/scss/site.scss"])
    assert result["results"] == []


@pytest.mark.parametrize("scope", ["assets/scss", "assets/scss/site.scss"])
def test_actual_ripgrep_matches_python_scope(repo, tools, scope):
    rg = shutil.which("rg")
    if not rg:
        pytest.skip("ripgrep not installed")
    expected = tools.search(str(repo), "border-top", path=scope)
    tools._rg = rg
    actual = tools.search(str(repo), "border-top", path=scope)
    assert actual["engine"] == "ripgrep"
    assert actual["results"] == expected["results"]


def test_git_accelerator_sends_literal_scope(repo, tools, monkeypatch):
    (repo / ".git").mkdir()
    tools.use_git_grep = True
    captured = []
    def fake_run(cmd, **kwargs):
        captured.append(cmd)
        return SimpleNamespace(returncode=0, stdout="assets/scss/site.scss:1: border-top\n")
    monkeypatch.setattr("local_ai_hub.repo_tools.subprocess.run", fake_run)
    result, failed = tools._git_grep_candidates(repo, ["border-top"], 10, path="assets/scss")
    assert not failed
    assert result == [("assets/scss/site.scss", 1)]
    assert captured[0][-2:] == ["--", ":(literal)assets/scss"]
    assert "-F" in captured[0]
    assert "-w" in captured[0]


def test_ripgrep_literal_false_prefix_cannot_consume_candidate_limit(repo, tools):
    rg = shutil.which("rg")
    if not rg:
        pytest.skip("ripgrep not installed")
    target = repo / "assets/scss/late.scss"
    target.write_text(".bad { xborder-top: 1px; }\n" * 40 + ".good { border-top: 2px; }\n")
    tools._rg = rg
    result = tools.search(str(repo), "border-top", path="assets/scss/late.scss")
    assert len(result["results"]) == 1
    assert result["results"][0]["end_line"] == 41


def test_scope_rejects_symlink_escape(repo, tools, tmp_path):
    outside = tmp_path.parent / (tmp_path.name + "-outside")
    outside.mkdir()
    link = repo / "escape"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("symlink permission unavailable")
    result = tools.search(str(repo), "border-top", path="escape")
    assert result["success"] is False
    assert result["terminal"] is True
    assert result["retryable"] is False
