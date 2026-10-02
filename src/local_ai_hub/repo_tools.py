from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import threading
import time
from collections import Counter, OrderedDict
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping

from .budget import chars_for_tokens, estimate_tokens
from .normalizer import tokenize_query_terms
from .process_utils import canonical_root, hidden_run_kwargs


_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_./:-]{1,80}")
_SYMBOL_PATTERNS = [
    re.compile(r"^\s*(?:async\s+)?def\s+([A-Za-z_]\w*)"),
    re.compile(r"^\s*class\s+([A-Za-z_]\w*)"),
    re.compile(r"^\s*(?:public|private|protected|internal|static|final|virtual|override|abstract|sealed|partial|async|export|default|const|let|var|function|func|fn|type|interface|enum|struct|record|trait|impl|namespace|module)\s+(?:[\w<>,?\[\].]+\s+)*([A-Za-z_]\w*)"),
]


@dataclass(frozen=True)
class GitBlobRecord:
    state: str
    oid: str | None = None
    identity: str | None = None


@dataclass(frozen=True)
class GitSnapshot:
    common_dir: str | None
    worktree_root: str
    git_dir: str | None
    index_path: str | None
    index_signature: tuple[int, int, int, int, int, str] | None
    head_oid: str | None
    tree_oid: str | None
    status: Mapping[str, str]
    blobs: Mapping[str, GitBlobRecord]
    deleted: tuple[str, ...]
    renames: Mapping[str, str]
    degraded: bool = False
    error: str | None = None
    status_identity: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", MappingProxyType(dict(self.status)))
        object.__setattr__(self, "blobs", MappingProxyType(dict(self.blobs)))
        object.__setattr__(self, "renames", MappingProxyType(dict(self.renames)))

    @property
    def revision(self) -> str:
        """Stable repository revision for bounded guard provenance."""
        return self.head_oid or self.status_identity or ""

    @property
    def repository_revision(self) -> str:
        return self.revision

    @property
    def changed_paths(self) -> tuple[str, ...]:
        paths = {path for path, state in self.status.items() if state != "clean"}
        paths.update(self.deleted)
        paths.update(self.renames)
        paths.update(self.renames.values())
        return tuple(sorted(path for path in paths if path))


class RepositoryTools:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        rag = config.get("rag", {})
        search = config.get("search", {})
        self.extensions = {str(x).lower() for x in rag.get("extensions", [])}
        self.special_filenames = {str(x).lower() for x in search.get("special_filenames", [
            "dockerfile", "containerfile", "docker-compose.yml", "docker-compose.yaml", "compose.yml", "compose.yaml",
            "makefile", "gnumakefile", "justfile", "gemfile", "pipfile", ".env", ".env.example", ".env.sample",
            ".gitlab-ci.yml", ".gitlab-ci.yaml", "azure-pipelines.yml", "azure-pipelines.yaml",
            "pytest.ini", "tox.ini", "phpunit.xml", "phpunit.xml.dist", "phpstan.neon", "phpstan.neon.dist",
        ])}
        self.ignore_dirs = {str(name).lower() for name in rag.get("ignore_dirs", [])}
        self.max_file_bytes = int(rag.get("max_file_bytes", 2_000_000))
        self.max_files = int(search.get("max_files", 8000))
        self.max_hits = int(search.get("max_hits", 80))
        self.snippet_lines = int(search.get("snippet_lines", 5))
        self.max_snippets_per_file = int(search.get("max_snippets_per_file", 3))
        self.prefer_git = bool(search.get("prefer_git_files", True))
        self.use_ripgrep = bool(search.get("use_ripgrep_if_available", True))
        self.ripgrep_timeout = max(0.5, float(search.get("ripgrep_timeout_seconds", 15.0)))
        self.git_grep_timeout = max(0.5, float(search.get("git_grep_timeout_seconds", min(10.0, self.ripgrep_timeout))))
        self.git_files_timeout = max(0.5, float(search.get("git_files_timeout_seconds", 5.0)))
        self.git_files_cache_ttl = max(0.0, float(search.get("git_files_cache_ttl_seconds", 3.0)))
        self.git_files_slow_cooldown = max(1.0, float(search.get("git_files_slow_cooldown_seconds", 15.0)))
        self.use_git_grep = bool(search.get("use_git_grep_fallback", True))
        self._rg = shutil.which("rg") if self.use_ripgrep else None
        self._git_files_lock = threading.RLock()
        self._git_files_cache: dict[str, tuple[float, tuple[str, ...]]] = {}
        self._git_blob_cache: dict[str, tuple[float, dict[str, str]]] = {}
        self._git_snapshot_cache: dict[str, tuple[float, GitSnapshot]] = {}
        self._git_snapshot_slow_until: dict[str, float] = {}
        self._git_files_slow_until: dict[str, float] = {}
        self._git_files_flights: dict[str, threading.Lock] = {}
        self._git_grep_slow_until: dict[str, float] = {}
        self._git_stats = {"file_list_hits": 0, "file_list_misses": 0, "file_list_timeouts": 0, "file_list_coalesced": 0, "grep_timeouts": 0, "grep_cooldown_skips": 0, "snapshot_hits": 0, "snapshot_misses": 0, "snapshot_degraded": 0}
        snap = config.get("snapshot_cache", {})
        self.snapshot_max_entries = max(64, int(snap.get("max_entries", 4096)))
        self.snapshot_max_bytes = max(8_000_000, int(snap.get("max_bytes", 256_000_000)))
        self._snapshot_lock = threading.RLock()
        self._snapshots: OrderedDict[str, tuple[int, int, int, int, str, list[str], int]] = OrderedDict()
        self._hash_only_cache: OrderedDict[str, tuple[int, int, int, int, str]] = OrderedDict()
        self._hash_only_max_entries = 8192
        self._snapshot_bytes = 0
        self.snapshot_hits = 0
        self.snapshot_misses = 0

    @staticmethod
    def _root(root: str) -> Path:
        canon = canonical_root(root)
        path = Path(canon)
        if not path.is_dir():
            raise ValueError(f"root directory does not exist: {path}")
        return path

    @staticmethod
    def _search_scope(base: Path, path: str = "") -> Path:
        target = (base / str(path).replace("\\", "/")).resolve(strict=False)
        try:
            target.relative_to(base)
        except ValueError as exc:
            raise ValueError("search path must stay inside repository root") from exc
        if not target.exists() or not (target.is_file() or target.is_dir()):
            raise ValueError(f"search path does not exist: {path}")
        return target

    @classmethod
    def normalize_search_scope(cls, root: str, path: str = "") -> str:
        """Validate scope and return one stable root-relative cache identity."""
        base = cls._root(root)
        rel = cls._search_scope(base, path).relative_to(base).as_posix()
        return "" if rel == "." else rel

    @staticmethod
    def _in_search_scope(base: Path, target: Path, scope: Path) -> bool:
        try:
            target.relative_to(base)
            target.relative_to(scope) if scope.is_dir() else target.relative_to(scope.parent)
        except ValueError:
            return False
        return scope.is_dir() or target == scope

    @staticmethod
    def _invalid_search_scope(base: Path, exc: ValueError) -> dict[str, Any]:
        return {"success": False, "root": str(base), "results": [],
                "error": str(exc), "terminal": True, "retryable": False}

    def _git_files(self, root: Path) -> list[Path] | None:
        """Return tracked + untracked files using bounded, coalesced Git probes.

        A short cache absorbs bursts, a per-root flight lock prevents concurrent
        callers from spawning identical Git processes, and a timeout cooldown keeps
        pathological repositories off the foreground hot path. Returning ``None``
        intentionally delegates to the bounded Python inventory fallback.
        """
        if not self.prefer_git or not (root / ".git").exists():
            return None
        key = str(root)
        now = time.monotonic()
        with self._git_files_lock:
            cached = self._git_files_cache.get(key)
            if cached and (self.git_files_cache_ttl <= 0 or now - cached[0] <= self.git_files_cache_ttl):
                self._git_stats["file_list_hits"] += 1
                return [(root / rel).resolve(strict=False) for rel in cached[1]]
            if now < self._git_files_slow_until.get(key, 0.0):
                return None
            flight = self._git_files_flights.setdefault(key, threading.Lock())

        # Do not wait longer for a coalesced probe than a direct Git probe could take.
        if not flight.acquire(timeout=self.git_files_timeout + 0.25):
            with self._git_files_lock:
                self._git_stats["file_list_coalesced"] += 1
            return None
        try:
            now = time.monotonic()
            with self._git_files_lock:
                cached = self._git_files_cache.get(key)
                if cached and (self.git_files_cache_ttl <= 0 or now - cached[0] <= self.git_files_cache_ttl):
                    self._git_stats["file_list_hits"] += 1
                    self._git_stats["file_list_coalesced"] += 1
                    return [(root / rel).resolve(strict=False) for rel in cached[1]]
                if now < self._git_files_slow_until.get(key, 0.0):
                    return None
                self._git_stats["file_list_misses"] += 1
            try:
                completed = subprocess.run(
                    ["git", "-C", str(root), "ls-files", "-co", "--exclude-standard", "-z"],
                    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=self.git_files_timeout, check=False,
                    **hidden_run_kwargs(),
                )
                if completed.returncode != 0:
                    return None
                rels: list[str] = []
                for raw in completed.stdout.split(b"\0"):
                    if not raw:
                        continue
                    rel = raw.decode("utf-8", errors="surrogateescape").replace("\\", "/")
                    candidate = (root / rel).resolve(strict=False)
                    try:
                        candidate.relative_to(root)
                    except ValueError:
                        continue
                    rels.append(rel)
                    if len(rels) >= self.max_files:
                        break
                with self._git_files_lock:
                    self._git_files_cache[key] = (time.monotonic(), tuple(rels))
                    self._git_files_slow_until.pop(key, None)
                    if len(self._git_files_cache) > 64:
                        for k in list(self._git_files_cache.keys())[:-32]:
                            self._git_files_cache.pop(k, None)
                return [(root / rel).resolve(strict=False) for rel in rels]
            except subprocess.TimeoutExpired:
                with self._git_files_lock:
                    self._git_stats["file_list_timeouts"] += 1
                    self._git_files_slow_until[key] = time.monotonic() + self.git_files_slow_cooldown
                return None
            except OSError:
                with self._git_files_lock:
                    self._git_files_slow_until[key] = time.monotonic() + self.git_files_slow_cooldown
                return None
        finally:
            flight.release()

    def iter_files(self, root: str, max_depth: int = 25) -> Iterable[Path]:
        base = self._root(root)
        git_files = self._git_files(base)
        if git_files is not None:
            candidates = git_files
        else:
            candidates = []
            visited_realpaths: set[str] = set()
            try:
                visited_realpaths.add(os.path.realpath(base))
            except Exception:
                pass
            for dirpath, dirnames, filenames in os.walk(base, followlinks=False):
                dirnames[:] = [d for d in dirnames if d.lower() not in self.ignore_dirs]
                try:
                    rel_parts = Path(dirpath).relative_to(base).parts
                    if len(rel_parts) >= max_depth:
                        dirnames.clear()
                        continue
                except Exception:
                    pass

                kept_dirnames: list[str] = []
                for d in dirnames:
                    subdir = os.path.join(dirpath, d)
                    try:
                        rpath = os.path.realpath(subdir)
                        if rpath in visited_realpaths:
                            continue
                        visited_realpaths.add(rpath)
                        kept_dirnames.append(d)
                    except Exception:
                        kept_dirnames.append(d)
                dirnames[:] = kept_dirnames

                for name in filenames:
                    p = Path(dirpath) / name
                    if self.extensions and p.suffix.lower() not in self.extensions and p.name.lower() not in self.special_filenames:
                        continue
                    candidates.append(p)
                    if len(candidates) >= self.max_files:
                        break
                if len(candidates) >= self.max_files:
                    break
        count = 0
        for path in candidates:
            if count >= self.max_files:
                break
            try:
                relative = path.relative_to(base)
            except ValueError:
                continue
            # `git ls-files -co --exclude-standard` also returns untracked files.
            # External code-intelligence tools create project-local metadata that
            # must never feed back into Local AI Hub's own repository inventory.
            if any(part.lower() in self.ignore_dirs for part in relative.parts[:-1]):
                continue
            if self.extensions and path.suffix.lower() not in self.extensions and path.name.lower() not in self.special_filenames:
                continue
            try:
                if path.stat().st_size > self.max_file_bytes:
                    continue
            except OSError:
                continue
            count += 1
            yield path

    def _read_snapshot(self, path: Path) -> tuple[str, list[str]]:
        """Cache decoded file content/hash by stat identity across different repo queries."""
        stat = path.stat()
        key = str(path)
        with self._snapshot_lock:
            item = self._snapshots.get(key)
            identity = (int(stat.st_mtime_ns), int(stat.st_size), int(getattr(stat, "st_ctime_ns", 0)), int(getattr(stat, "st_ino", 0)))
            if item is not None and item[:4] == identity:
                self._snapshots.move_to_end(key)
                self.snapshot_hits += 1
                return item[4], item[5]
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        lines = raw.decode("utf-8", errors="replace").splitlines()
        size = len(raw)
        with self._snapshot_lock:
            old = self._snapshots.pop(key, None)
            if old:
                self._snapshot_bytes -= old[6]
            self._snapshots[key] = (int(stat.st_mtime_ns), int(stat.st_size), int(getattr(stat, "st_ctime_ns", 0)), int(getattr(stat, "st_ino", 0)), digest, lines, size)
            self._snapshot_bytes += size
            self._hash_only_cache[key] = (int(stat.st_mtime_ns), int(stat.st_size), int(getattr(stat, "st_ctime_ns", 0)), int(getattr(stat, "st_ino", 0)), digest)
            if len(self._hash_only_cache) > self._hash_only_max_entries:
                self._hash_only_cache.popitem(last=False)
            self.snapshot_misses += 1
            while self._snapshots and (len(self._snapshots) > self.snapshot_max_entries or self._snapshot_bytes > self.snapshot_max_bytes):
                _k, evicted = self._snapshots.popitem(last=False)
                self._snapshot_bytes -= evicted[6]
        return digest, lines

    def _hash_file_only(self, path: Path) -> str:
        """Return SHA-256 without decoding source text or populating the decoded snapshot LRU.

        Preprocessing uses this when it only needs content identity. Interactive reads
        continue to use ``_read_snapshot`` so decoded text is cached on demand.
        """
        stat = path.stat()
        key = str(path)
        identity = (int(stat.st_mtime_ns), int(stat.st_size), int(getattr(stat, "st_ctime_ns", 0)), int(getattr(stat, "st_ino", 0)))
        with self._snapshot_lock:
            item = self._snapshots.get(key)
            if item is not None and item[:4] == identity:
                self._snapshots.move_to_end(key)
                self.snapshot_hits += 1
                return item[4]
            h_item = self._hash_only_cache.get(key)
            if h_item is not None and h_item[:4] == identity:
                self._hash_only_cache.move_to_end(key)
                self.snapshot_hits += 1
                return h_item[4]
        h = hashlib.sha256()
        with path.open("rb") as fh:
            while True:
                chunk = fh.read(1024 * 1024)
                if not chunk:
                    break
                h.update(chunk)
        digest = h.hexdigest()
        with self._snapshot_lock:
            self._hash_only_cache[key] = (identity[0], identity[1], identity[2], identity[3], digest)
            if len(self._hash_only_cache) > self._hash_only_max_entries:
                self._hash_only_cache.popitem(last=False)
        return digest

    @staticmethod
    def _git_empty_snapshot(root: str, error: str) -> GitSnapshot:
        return GitSnapshot(
            common_dir=None,
            worktree_root=root,
            git_dir=None,
            index_path=None,
            index_signature=None,
            head_oid=None,
            tree_oid=None,
            status=MappingProxyType({}),
            blobs=MappingProxyType({}),
            deleted=(),
            renames=MappingProxyType({}),
            degraded=True,
            error=error,
        )

    def _git_snapshot_run(self, base: Path, *args: str) -> bytes:
        completed = subprocess.run(
            ["git", "-C", str(base), *args],
            capture_output=True,
            timeout=self.git_files_timeout,
            check=False,
            **hidden_run_kwargs(),
        )
        if not isinstance(completed.stdout, (bytes, bytearray)):
            raise ValueError("git command failed")
        if completed.returncode != 0:
            # ``HEAD`` is legitimately unresolved before the first commit. Git
            # still returns complete repository/index metadata in this case.
            unborn_head = (
                args[-1:] == ("HEAD",)
                and bytes(completed.stdout).splitlines()[-1:] == [b"HEAD"]
                and isinstance(completed.stderr, (bytes, bytearray))
                and b"ambiguous argument 'HEAD'" in bytes(completed.stderr)
            )
            if not unborn_head:
                raise ValueError("git command failed")
        return bytes(completed.stdout)

    @staticmethod
    def _git_oid(raw: bytes) -> str:
        try:
            oid = raw.decode("ascii")
        except UnicodeDecodeError as exc:
            raise ValueError("invalid git object id") from exc
        if len(oid) not in {40, 64} or re.fullmatch(r"[0-9a-fA-F]+", oid) is None:
            raise ValueError("invalid git object id")
        return oid.lower()

    @staticmethod
    def _git_path(root: Path, raw: bytes) -> str:
        path = raw.decode("utf-8", errors="surrogateescape").replace("\\", "/")
        if not path:
            raise ValueError("empty git path")
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = root / candidate
        return str(candidate.resolve(strict=False))

    @staticmethod
    def _git_relpath(raw: bytes) -> str:
        path = raw.decode("utf-8", errors="surrogateescape").replace("\\", "/")
        parts = path.split("/")
        if not path or path.startswith("/") or any(part in {"", ".", ".."} for part in parts):
            raise ValueError("invalid git relative path")
        return path

    @staticmethod
    def _git_nul_records(raw: bytes) -> list[bytes]:
        if not isinstance(raw, bytes):
            raise ValueError("git output is not bytes")
        if not raw:
            return []
        if not raw.endswith(b"\0"):
            raise ValueError("unterminated git output")
        return [record for record in raw.split(b"\0")[:-1]]

    @classmethod
    def _git_parse_index(cls, raw: bytes) -> dict[str, str | None]:
        entries: dict[str, list[tuple[int, str]]] = {}
        for record in cls._git_nul_records(raw):
            if b"\t" not in record:
                raise ValueError("malformed git index record")
            metadata, raw_path = record.split(b"\t", 1)
            fields = metadata.split(b" ")
            if len(fields) != 3 or re.fullmatch(rb"[0-7]{6}", fields[0]) is None:
                raise ValueError("malformed git index metadata")
            try:
                stage = int(fields[2])
            except ValueError as exc:
                raise ValueError("malformed git index stage") from exc
            if stage not in {0, 1, 2, 3}:
                raise ValueError("malformed git index stage")
            path = cls._git_relpath(raw_path)
            entries.setdefault(path, []).append((stage, cls._git_oid(fields[1])))
        result: dict[str, str | None] = {}
        for path, values in entries.items():
            stage_zero = [oid for stage, oid in values if stage == 0]
            result[path] = stage_zero[0] if len(values) == 1 and len(stage_zero) == 1 else None
        return result

    @classmethod
    def _git_parse_status(cls, raw: bytes) -> list[tuple[str, str, str, str, str | None]]:
        records = cls._git_nul_records(raw)
        parsed: list[tuple[str, str, str, str, str | None]] = []
        index = 0
        while index < len(records):
            record = records[index]
            if record.startswith(b"1 "):
                fields = record.split(b" ", 8)
                if len(fields) != 9 or len(fields[1]) != 2:
                    raise ValueError("malformed porcelain-v2 ordinary record")
                path = cls._git_relpath(fields[8])
                x, y = chr(fields[1][0]), chr(fields[1][1])
                parsed.append((path, x, y, cls._git_status_state(x, y), None))
            elif record.startswith(b"2 "):
                fields = record.split(b" ", 9)
                if len(fields) != 10 or len(fields[1]) != 2 or index + 1 >= len(records):
                    raise ValueError("malformed porcelain-v2 rename record")
                path = cls._git_relpath(fields[9])
                original = cls._git_relpath(records[index + 1])
                if fields[8][:1] not in {b"R", b"C"} or not fields[8][1:].isdigit():
                    raise ValueError("malformed porcelain-v2 rename score")
                x, y = chr(fields[1][0]), chr(fields[1][1])
                parsed.append((path, x, y, cls._git_status_state(x, y), original))
                index += 1
            elif record.startswith(b"u "):
                fields = record.split(b" ", 10)
                if len(fields) != 11 or len(fields[1]) != 2:
                    raise ValueError("malformed porcelain-v2 unmerged record")
                path = cls._git_relpath(fields[10])
                parsed.append((path, chr(fields[1][0]), chr(fields[1][1]), "unmerged", None))
            elif record.startswith(b"? "):
                parsed.append((cls._git_relpath(record[2:]), ".", ".", "untracked", None))
            elif record.startswith(b"! "):
                cls._git_relpath(record[2:])
            else:
                raise ValueError("malformed porcelain-v2 status record")
            index += 1
        return parsed

    @staticmethod
    def _git_status_state(x: str, y: str) -> str:
        if "U" in {x, y}:
            return "unmerged"
        if "D" in {x, y}:
            return "deleted"
        staged = x != "."
        modified = y != "."
        if staged and modified:
            return "staged_and_worktree_modified"
        if staged:
            return "staged"
        if modified:
            return "modified"
        return "clean"

    def _git_index_signature(self, base: Path, index_path: str) -> tuple[int, int, int, int, int, str] | None:
        try:
            stat = Path(index_path).stat()
        except OSError:
            return None

        completed = subprocess.run(
            ["git", "-C", str(base), "hash-object", "--", index_path],
            capture_output=True,
            timeout=self.git_files_timeout,
            check=False,
            **hidden_run_kwargs(),
        )
        if completed.returncode != 0 or not isinstance(completed.stdout, (bytes, bytearray)):
            raise ValueError("git index digest failed")
        digest = self._git_oid(bytes(completed.stdout).strip())
        return (
            int(stat.st_mtime_ns),
            int(stat.st_size),
            int(getattr(stat, "st_ctime_ns", 0)),
            int(getattr(stat, "st_dev", 0)),
            int(getattr(stat, "st_ino", 0)),
            digest,
        )

    @staticmethod
    def _git_index_cache_identity(signature: tuple[int, int, int, int, int, str] | None) -> tuple[int, str] | None:
        if signature is None:
            return None
        # `git status` may atomically replace the index file while preserving its
        # logical contents. Size + content digest stay stable across that rewrite.
        return (signature[1], signature[5])

    def git_snapshot(self, root: str) -> GitSnapshot:
        try:
            base = self._root(root)
        except ValueError:
            return self._git_empty_snapshot(str(root), "invalid repository root")
        worktree_root = str(base)
        key = worktree_root
        if shutil.which("git") is None:
            return self._git_empty_snapshot(worktree_root, "git unavailable")
        now = time.monotonic()
        with self._git_files_lock:
            if now < self._git_snapshot_slow_until.get(key, 0.0):
                self._git_stats["snapshot_degraded"] += 1
                return self._git_empty_snapshot(worktree_root, "git snapshot cooldown")
            cached_entry = self._git_snapshot_cache.get(key)
        try:
            identity_output = self._git_snapshot_run(base, "rev-parse", "--git-common-dir", "--git-dir", "--git-path", "index", "HEAD")
            identity_lines = identity_output.splitlines()
            if len(identity_lines) != 4 or any(not line for line in identity_lines[:3]):
                raise ValueError("malformed git identity output")
            common_dir = self._git_path(base, identity_lines[0])
            git_dir = self._git_path(base, identity_lines[1])
            index_path = self._git_path(base, identity_lines[2])
            unborn_head = identity_lines[3] == b"HEAD"
            head_oid = None if unborn_head else self._git_oid(identity_lines[3])

            status_output = self._git_snapshot_run(base, "status", "--porcelain=v2", "--untracked-files=all", "-z")
            status_entries = self._git_parse_status(status_output)
            staged_output = self._git_snapshot_run(base, "diff", "--cached", "--raw", "-z")
            status_identity = hashlib.sha256(status_output + b"\0" + staged_output).hexdigest()
            index_signature = self._git_index_signature(base, index_path)
            if cached_entry is not None:
                cached_at, cached = cached_entry
                same_identity = (
                    cached.common_dir == common_dir
                    and cached.worktree_root == worktree_root
                    and cached.git_dir == git_dir
                    and cached.index_path == index_path
                    and self._git_index_cache_identity(cached.index_signature) == self._git_index_cache_identity(index_signature)
                    and cached.head_oid == head_oid
                    and cached.status_identity == status_identity
                )
                if same_identity and (self.git_files_cache_ttl <= 0 or now - cached_at <= self.git_files_cache_ttl):
                    with self._git_files_lock:
                        self._git_stats["snapshot_hits"] += 1
                    return cached

            if unborn_head:
                tree_oid = None
            else:
                tree_output = self._git_snapshot_run(base, "rev-parse", "HEAD^{tree}")
                tree_lines = tree_output.splitlines()
                if len(tree_lines) != 1:
                    raise ValueError("malformed git tree output")
                tree_oid = self._git_oid(tree_lines[0])
            index_entries = self._git_parse_index(self._git_snapshot_run(base, "ls-files", "--stage", "-z"))
            status: dict[str, str] = {path: "clean" for path in index_entries}
            renames: dict[str, str] = {}
            deleted: set[str] = set()
            for path, _x, _y, state, original in status_entries:
                status[path] = state
                if state == "deleted":
                    deleted.add(path)
                if original is not None:
                    renames[path] = original
            blobs: dict[str, GitBlobRecord] = {}
            for path, oid in index_entries.items():
                state = status.get(path, "clean")
                identity = f"git:{oid}" if (state == "clean" or (unborn_head and state == "staged")) and oid is not None else None
                blobs[path] = GitBlobRecord(state=state, oid=oid, identity=identity)
            for path, state in status.items():
                if path not in blobs:
                    blobs[path] = GitBlobRecord(state=state)
            snapshot = GitSnapshot(
                common_dir=common_dir,
                worktree_root=worktree_root,
                git_dir=git_dir,
                index_path=index_path,
                index_signature=index_signature,
                head_oid=head_oid,
                tree_oid=tree_oid,
                status=MappingProxyType(dict(status)),
                blobs=MappingProxyType(dict(blobs)),
                deleted=tuple(sorted(deleted)),
                renames=MappingProxyType(dict(renames)),
                status_identity=status_identity,
            )
            with self._git_files_lock:
                self._git_snapshot_cache[key] = (now, snapshot)
                self._git_snapshot_slow_until.pop(key, None)
                self._git_stats["snapshot_misses"] += 1
                if len(self._git_snapshot_cache) > 64:
                    for old_key in list(self._git_snapshot_cache)[:-32]:
                        self._git_snapshot_cache.pop(old_key, None)
            return snapshot
        except (OSError, subprocess.TimeoutExpired, ValueError):
            with self._git_files_lock:
                self._git_stats["snapshot_degraded"] += 1
                self._git_snapshot_slow_until[key] = time.monotonic() + self.git_files_slow_cooldown
            return self._git_empty_snapshot(worktree_root, "bounded git snapshot failed")

    def git_blob_map(self, root: str) -> dict[str, str]:
        """Return ``git:<blob_oid>`` identities for clean tracked paths."""
        snapshot = self.git_snapshot(root)
        if snapshot.degraded:
            return {}
        return {
            path: record.identity
            for path, record in snapshot.blobs.items()
            if (record.state == "clean" or (snapshot.head_oid is None and record.state == "staged")) and record.identity is not None
        }

    def git_blob_hashes(self, root: str, paths: list[str]) -> dict[str, str]:
        """Return clean tracked Git blob IDs for selected paths."""
        normalized = [str(Path(path).as_posix()).lstrip("./") for path in paths]
        if not normalized:
            return {}
        result = self.git_blob_map(root)
        return {path: result[path] for path in normalized if path in result}

    def snapshot_stats(self) -> dict[str, Any]:
        with self._snapshot_lock:
            snapshot = {"entries": len(self._snapshots), "bytes": self._snapshot_bytes, "hits": self.snapshot_hits, "misses": self.snapshot_misses}
        with self._git_files_lock:
            now = time.monotonic()
            snapshot["git_files"] = {
                **self._git_stats,
                "cached_roots": len(self._git_files_cache),
                "cooldown_roots": sum(1 for until in self._git_files_slow_until.values() if until > now),
                "grep_cooldown_roots": sum(1 for until in self._git_grep_slow_until.values() if until > now),
                "snapshot_cached_roots": len(self._git_snapshot_cache),
                "snapshot_cooldown_roots": sum(1 for until in self._git_snapshot_slow_until.values() if until > now),
            }
        return snapshot

    @staticmethod
    def _terms(query: str) -> list[str]:
        return [t.lower() for t in tokenize_query_terms(query, min_len=2, max_terms=20)]

    def _search_terms(self, query: str) -> tuple[list[str], str]:
        phrase = query.strip().lower()
        literal = phrase if phrase and not re.search(r"\s", phrase) and re.search(r"[-_/.:]", phrase) else ""
        return ([literal] if literal else self._terms(query)), literal

    @staticmethod
    def _line_matches(line: str, terms: list[str], literal: str) -> list[str]:
        if literal:
            # Keep a CSS/property/path identifier intact. A hyphen suffix such as
            # border-top-left is relevant; xborder-top and topology are not.
            return [literal] if re.search(r"(?<![\w])" + re.escape(literal) + r"(?![\w])", line) else []
        return [term for term in terms if term in line]

    def _ripgrep_candidates(self, base: Path, terms: list[str], max_matches: int, path: str = "") -> tuple[list[tuple[str, int]] | None, bool]:
        if not self._rg or not terms:
            return None, False
        cmd = [self._rg, "--json", "--fixed-strings", "--ignore-case", "--max-count", str(max(2, self.max_snippets_per_file * 3))]
        if len(terms) == 1 and re.search(r"[-_/.:]", terms[0]):
            cmd.append("--word-regexp")
        for term in terms[:12]:
            cmd.extend(["-e", term])
        cmd.append(str(self._search_scope(base, path)))
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=self.ripgrep_timeout, check=False, encoding="utf-8", errors="replace", **hidden_run_kwargs())
        except Exception:
            return None, True
        # rg returns 1 when there are no matches; both 0/1 are successful searches.
        # Preserve the distinction between an empty successful result and an engine
        # failure so a miss never falls back to an O(repository) Python scan.
        if proc.returncode not in {0, 1}:
            return None, True
        out: list[tuple[str, int]] = []
        seen: set[tuple[str, int]] = set()
        for raw in proc.stdout.splitlines():
            try:
                event = json.loads(raw)
            except Exception:
                continue
            if event.get("type") != "match":
                continue
            data = event.get("data", {})
            path_data = data.get("path", {}) if isinstance(data, dict) else {}
            path_text = path_data.get("text") if isinstance(path_data, dict) else None
            line_no = int(data.get("line_number", 0) or 0) if isinstance(data, dict) else 0
            if not path_text or line_no <= 0:
                continue
            try:
                path = Path(str(path_text)).resolve(strict=False)
                rel = str(path.relative_to(base)).replace("\\", "/")
            except Exception:
                continue
            match_key = (rel, line_no)
            if match_key in seen:
                continue
            seen.add(match_key); out.append(match_key)
            if len(out) >= max_matches:
                break
        return out, False

    def _git_grep_candidates(self, base: Path, terms: list[str], max_matches: int, path: str = "") -> tuple[list[tuple[str, int]] | None, bool]:
        """Fast portable fallback when ripgrep is unavailable.

        `git grep` covers tracked files and is used only as a bounded accelerator. A
        successful zero-result probe is final for this accelerator path; preprocessed
        candidates and the normal file inventory cover untracked-file discovery.
        """
        if not self.use_git_grep or not terms or not (base / ".git").exists():
            return None, False
        key = str(base)
        with self._git_files_lock:
            if time.monotonic() < self._git_grep_slow_until.get(key, 0.0):
                self._git_stats["grep_cooldown_skips"] += 1
                return None, False
        cmd = ["git", "-C", str(base), "grep", "-n", "-I", "-i", "-F", "--no-color", "--full-name"]
        if len(terms) == 1 and re.search(r"[-_/.:]", terms[0]):
            cmd.append("-w")
        for term in terms[:12]:
            cmd.extend(["-e", term])
        cmd.append("--")
        if path:
            cmd.append(":(literal)" + path)
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=self.git_grep_timeout,
                check=False, encoding="utf-8", errors="replace",
                **hidden_run_kwargs(),
            )
        except subprocess.TimeoutExpired:
            with self._git_files_lock:
                self._git_stats["grep_timeouts"] += 1
                self._git_grep_slow_until[key] = time.monotonic() + self.git_files_slow_cooldown
            return None, True
        except OSError:
            return None, True
        if proc.returncode not in {0, 1}:
            return None, True
        with self._git_files_lock:
            self._git_grep_slow_until.pop(key, None)
        out: list[tuple[str, int]] = []
        seen: set[tuple[str, int]] = set()
        for raw in proc.stdout.splitlines():
            parts = raw.split(":", 2)
            if len(parts) < 2:
                continue
            rel = parts[0].replace("\\", "/")
            try:
                line_no = int(parts[1])
                target = (base / rel).resolve(strict=False)
                target.relative_to(base)
            except Exception:
                continue
            match_key = (rel, line_no)
            if match_key in seen:
                continue
            seen.add(match_key); out.append(match_key)
            if len(out) >= max_matches:
                break
        return out, False

    def search(self, root: str, query: str, top_k: int = 12, context_lines: int | None = None, path: str = "") -> dict[str, Any]:
        base = self._root(root)
        try:
            scope = self._search_scope(base, path)
        except ValueError as exc:
            return self._invalid_search_scope(base, exc)
        normalized_path = scope.relative_to(base).as_posix()
        scoped_args = {"path": normalized_path} if scope != base else {}
        terms, literal = self._search_terms(query)
        phrase = query.strip().lower()
        if not terms and not phrase:
            return {"success": True, "root": str(base), "results": []}
        effective_snippet_lines = max(1, min(int(context_lines), 40)) if context_lines is not None else self.snippet_lines
        hits: list[dict[str, Any]] = []
        scanned = 0
        candidate_engine = "python"
        candidates, accelerator_failed = self._ripgrep_candidates(base, terms, self.max_hits * 4, **scoped_args)
        accelerated = candidates is not None
        if accelerated:
            candidate_engine = "ripgrep"
        else:
            candidates, git_grep_failed = self._git_grep_candidates(base, terms, self.max_hits * 4, **scoped_args)
            accelerator_failed = accelerator_failed or git_grep_failed
            accelerated = candidates is not None
            if accelerated:
                candidate_engine = "git-grep"
        if candidates:
            per_path_lines: dict[str, list[int]] = {}
            for rel, line_no in candidates:
                per_path_lines.setdefault(rel, []).append(line_no)
            for rel, line_numbers in per_path_lines.items():
                path = (base / rel).resolve(strict=False)
                if not self._in_search_scope(base, path, scope):
                    continue
                try:
                    file_hash, lines = self._read_snapshot(path)
                except Exception:
                    continue
                scanned += 1
                path_lower = rel.lower()
                for line_no in line_numbers:
                    idx = line_no - 1
                    if idx < 0 or idx >= len(lines): continue
                    low = lines[idx].lower()
                    matched = self._line_matches(low, terms, literal)
                    if not matched:
                        continue
                    path_matches = sum(1 for t in terms if t in path_lower)
                    score = float(len(matched) * 3 + path_matches * 2 + sum(min(low.count(t), 3) * 0.5 for t in matched))
                    if phrase and len(phrase) <= 120 and phrase in low: score += 8
                    start = max(0, idx - effective_snippet_lines); end = min(len(lines), idx + effective_snippet_lines + 1)
                    snippet = "\n".join(f"{n + 1}: {lines[n]}" for n in range(start, end))
                    hits.append({"path": rel, "start_line": start + 1, "end_line": end, "score": round(score, 3), "text": snippet, "file_sha256": file_hash})
        # A timeout/error from both bounded accelerators must not trigger an
        # unbounded O(repository) Python scan. A valid accelerated zero result is
        # still final; Python remains the fallback only when no accelerator exists.
        if not accelerated and accelerator_failed:
            return {
                "success": False,
                "root": str(base),
                "terms": terms,
                "results": [],
                "engine": "bounded-accelerator-failure",
                "error": "bounded search accelerators timed out or failed",
                "retryable": True,
            }
        if not hits and not accelerated:
            inventory = self.iter_files(str(scope)) if scope.is_dir() else [scope]
            for path in inventory:
                path = path.resolve(strict=False)
                if not self._in_search_scope(base, path, scope):
                    continue
                scanned += 1
                rel = str(path.relative_to(base)).replace("\\", "/")
                path_lower = rel.lower()
                try:
                    file_hash, lines = self._read_snapshot(path)
                except Exception:
                    continue
                for idx, line in enumerate(lines):
                    low = line.lower()
                    matched = self._line_matches(low, terms, literal)
                    path_matches = sum(1 for t in terms if t in path_lower)
                    if not matched:
                        continue
                    score = float(len(matched) * 3 + path_matches * 2)
                    if phrase and len(phrase) <= 120 and phrase in low:
                        score += 8
                    score += sum(min(low.count(t), 3) * 0.5 for t in matched)
                    start = max(0, idx - effective_snippet_lines)
                    end = min(len(lines), idx + effective_snippet_lines + 1)
                    snippet = "\n".join(f"{n + 1}: {lines[n]}" for n in range(start, end))
                    hits.append({
                        "path": rel, "start_line": start + 1, "end_line": end,
                        "score": round(score, 3), "text": snippet, "file_sha256": file_hash,
                    })
                    if len(hits) >= self.max_hits * 4:
                        break
        hits.sort(key=lambda x: (-x["score"], x["path"], x["start_line"]))
        dedup: list[dict[str, Any]] = []
        seen: set[tuple[str, int]] = set()
        per_file: Counter[str] = Counter()
        for hit in hits:
            key = (hit["path"], hit["start_line"] // max(1, self.snippet_lines))
            if key in seen or per_file[hit["path"]] >= self.max_snippets_per_file:
                continue
            seen.add(key)
            per_file[hit["path"]] += 1
            dedup.append(hit)
            if len(dedup) >= max(1, int(top_k)):
                break
        candidate_pool = hits[: max(len(dedup), max(1, int(top_k)) * 4)]
        candidate_tokens = sum(estimate_tokens(str(item.get("text", ""))) for item in candidate_pool)
        result = {"success": True, "root": str(base), "scanned_files": scanned, "terms": terms, "results": dedup, "engine": candidate_engine}
        if candidate_tokens:
            # Diagnostic only. Local RG-like filtering is not proof that a cloud
            # agent would otherwise have paid for the candidate snippets.
            result["candidate_context_tokens_est"] = candidate_tokens
        return result

    def repo_map(self, root: str, max_symbols: int = 120, paths: Iterable[str] | None = None) -> dict[str, Any]:
        base = self._root(root)
        extensions: Counter[str] = Counter()
        top_dirs: Counter[str] = Counter()
        symbols: list[dict[str, Any]] = []
        files = 0
        if paths is None:
            candidates: Iterable[Path] = self.iter_files(str(base))
        else:
            bounded: list[Path] = []
            for raw in paths:
                candidate = (base / str(raw)).resolve(strict=False)
                try:
                    candidate.relative_to(base)
                except ValueError:
                    continue
                bounded.append(candidate)
            candidates = bounded
        for path in candidates:
            if not path.is_file():
                continue
            if self.extensions and path.suffix.lower() not in self.extensions and path.name.lower() not in self.special_filenames:
                continue
            try:
                if path.stat().st_size > self.max_file_bytes:
                    continue
            except OSError:
                continue
            files += 1
            rel = str(path.relative_to(base)).replace("\\", "/")
            extensions[path.suffix.lower() or "<none>"] += 1
            top_dirs[rel.split("/", 1)[0]] += 1
            if len(symbols) >= max_symbols:
                continue
            try:
                _hash, lines = self._read_snapshot(path)
            except Exception:
                continue
            for line_no, line in enumerate(lines, 1):
                for pattern in _SYMBOL_PATTERNS:
                    match = pattern.match(line)
                    if match:
                        symbols.append({"name": match.group(1), "path": rel, "line": line_no})
                        break
                if len(symbols) >= max_symbols:
                    break
        return {
            "success": True, "root": str(base), "files": files,
            "extensions": extensions.most_common(20), "top_paths": top_dirs.most_common(30),
            "symbols_sample": symbols,
        }

    def project_profile(self, root: str) -> dict[str, Any]:
        """Detect project languages, manifests and likely validation commands without an LLM.

        Only small manifest/config files are read. Validation commands are suggestions;
        this method never executes project code.
        """
        base = self._root(root)
        language_by_ext = {
            ".py": "Python", ".php": "PHP", ".js": "JavaScript", ".jsx": "JavaScript",
            ".ts": "TypeScript", ".tsx": "TypeScript", ".cs": "C#", ".java": "Java",
            ".go": "Go", ".rs": "Rust", ".cpp": "C++", ".cc": "C++", ".cxx": "C++",
            ".c": "C", ".rb": "Ruby", ".kt": "Kotlin", ".swift": "Swift", ".scala": "Scala",
            ".dart": "Dart", ".ex": "Elixir", ".exs": "Elixir",
        }
        manifest_names = {
            "package.json", "pyproject.toml", "requirements.txt", "pipfile", "pipfile.lock", "poetry.lock", "uv.lock",
            "composer.json", "composer.lock", "cargo.toml", "cargo.lock", "go.mod", "go.sum",
            "pom.xml", "build.gradle", "build.gradle.kts", "gradlew", "gradlew.bat", "makefile",
            "cmakelists.txt", "gemfile", "gemfile.lock", "mix.exs", "pubspec.yaml", "deno.json", "deno.jsonc",
            "package-lock.json", "pnpm-lock.yaml", "yarn.lock", "bun.lock", "bun.lockb",
        }
        project_suffixes = {".sln", ".csproj", ".fsproj", ".vbproj"}
        raw_files = self._git_files(base)
        if raw_files is None:
            raw_files = []
            for dirpath, dirnames, filenames in os.walk(base):
                dirnames[:] = [d for d in dirnames if d.lower() not in self.ignore_dirs]
                for name in filenames:
                    p = Path(dirpath) / name
                    ext = p.suffix.lower()
                    if ext not in language_by_ext and name.lower() not in manifest_names and ext not in project_suffixes:
                        continue
                    raw_files.append(p)
                    if len(raw_files) >= self.max_files:
                        break
                if len(raw_files) >= self.max_files:
                    break
        raw_files = raw_files[:self.max_files]
        languages: Counter[str] = Counter()
        manifests: list[str] = []
        by_name: dict[str, list[Path]] = {}
        basenames: set[str] = set()

        for path in raw_files:
            try:
                rel = str(path.resolve(strict=False).relative_to(base)).replace("\\", "/")
            except ValueError:
                continue
            ext = path.suffix.lower()
            language = language_by_ext.get(ext)
            if language:
                languages[language] += 1
            name = path.name.lower()
            basenames.add(name)
            by_name.setdefault(name, []).append(path)
            if name in manifest_names or ext in project_suffixes:
                manifests.append(rel)

        tools: set[str] = set()
        commands: list[dict[str, str]] = []

        def add_command(command: str, purpose: str, confidence: str = "high") -> None:
            if not any(x["command"] == command for x in commands):
                commands.append({"command": command, "purpose": purpose, "confidence": confidence})

        package_files = by_name.get("package.json", [])[:20]
        if package_files:
            tools.add("node")
            package_manager = "npm"
            if "pnpm-lock.yaml" in basenames:
                package_manager = "pnpm"
            elif "yarn.lock" in basenames:
                package_manager = "yarn"
            elif "bun.lock" in basenames or "bun.lockb" in basenames:
                package_manager = "bun"
            tools.add(package_manager)
            import json
            for package_file in package_files:
                try:
                    if package_file.stat().st_size > 1_000_000:
                        continue
                    data = json.loads(package_file.read_text(encoding="utf-8"))
                except Exception:
                    continue
                scripts = data.get("scripts", {}) if isinstance(data, dict) else {}
                if isinstance(scripts, dict):
                    for script, purpose in [("test", "tests"), ("lint", "lint"), ("typecheck", "type-check"), ("check", "checks"), ("build", "build")]:
                        if script in scripts:
                            add_command(f"{package_manager} run {script}", purpose)
                deps: dict[str, Any] = {}
                for section in ("dependencies", "devDependencies"):
                    value = data.get(section, {}) if isinstance(data, dict) else {}
                    if isinstance(value, dict):
                        deps.update(value)
                for dep, label in [("typescript", "typescript"), ("eslint", "eslint"), ("vitest", "vitest"), ("jest", "jest"), ("next", "nextjs"), ("vite", "vite")]:
                    if dep in deps:
                        tools.add(label)

        pyproject_files = by_name.get("pyproject.toml", [])[:20]
        requirements_files = by_name.get("requirements.txt", [])[:20]
        if pyproject_files or requirements_files or languages.get("Python"):
            tools.add("python")
            all_py_parts: list[str] = []
            for file in [*pyproject_files, *requirements_files]:
                try:
                    if file.stat().st_size <= 1_000_000:
                        all_py_parts.append(file.read_text(encoding="utf-8", errors="replace").lower())
                except Exception:
                    pass
            all_py = "\n".join(all_py_parts)
            if "pytest" in all_py:
                tools.add("pytest"); add_command("python -m pytest", "tests")
            if "ruff" in all_py:
                tools.add("ruff"); add_command("ruff check .", "lint")
            if "mypy" in all_py:
                tools.add("mypy"); add_command("mypy .", "type-check", "medium")
            if "pyright" in all_py:
                tools.add("pyright"); add_command("pyright", "type-check", "medium")

        composer_files = by_name.get("composer.json", [])[:20]
        if composer_files or languages.get("PHP"):
            tools.add("php")
            import json
            for composer in composer_files:
                try:
                    if composer.stat().st_size > 1_000_000:
                        continue
                    data = json.loads(composer.read_text(encoding="utf-8"))
                except Exception:
                    continue
                scripts = data.get("scripts", {}) if isinstance(data, dict) else {}
                if isinstance(scripts, dict):
                    for script, purpose in [("test", "tests"), ("phpunit", "tests"), ("phpstan", "static-analysis"), ("cs", "style/checks"), ("check", "checks")]:
                        if script in scripts:
                            add_command(f"composer {script}", purpose)
                reqs: dict[str, Any] = {}
                for section in ("require", "require-dev"):
                    value = data.get(section, {}) if isinstance(data, dict) else {}
                    if isinstance(value, dict):
                        reqs.update(value)
                for dep, label in [("phpunit/phpunit", "phpunit"), ("phpstan/phpstan", "phpstan"), ("cakephp/cakephp", "cakephp"), ("laravel/framework", "laravel"), ("symfony/framework-bundle", "symfony")]:
                    if dep in reqs:
                        tools.add(label)

        if "cargo.toml" in basenames or languages.get("Rust"):
            tools.update({"rust", "cargo"}); add_command("cargo test", "tests"); add_command("cargo check", "type/build-check")
        if "go.mod" in basenames or languages.get("Go"):
            tools.add("go"); add_command("go test ./...", "tests")
        if any(Path(x).suffix.lower() in project_suffixes for x in manifests) or languages.get("C#"):
            tools.add("dotnet"); add_command("dotnet test", "tests", "medium"); add_command("dotnet build --no-restore", "build", "medium")
        if "pom.xml" in basenames:
            tools.add("maven"); add_command("mvn test", "tests")
        if "build.gradle" in basenames or "build.gradle.kts" in basenames:
            tools.add("gradle")
            gradle_cmd = ".\\gradlew.bat test" if os.name == "nt" and "gradlew.bat" in basenames else "./gradlew test" if "gradlew" in basenames else "gradle test"
            add_command(gradle_cmd, "tests")
        if "makefile" in basenames:
            tools.add("make")

        ci: list[str] = []
        for candidate in [".github/workflows", ".gitlab-ci.yml", "azure-pipelines.yml", "Jenkinsfile"]:
            if (base / candidate).exists():
                ci.append(candidate)

        manifests = sorted(set(manifests), key=lambda x: (x.count("/"), x.lower()))[:80]
        return {
            "success": True,
            "root": str(base),
            "scanned_paths": len(raw_files),
            "languages": [{"name": name, "files": count} for name, count in languages.most_common(12)],
            "manifests": manifests,
            "tools": sorted(tools),
            "validation_commands": commands[:24],
            "ci": ci,
            "note": "Validation commands are detected/suggested only; Local AI Hub does not execute project code.",
        }


    def verify_evidence(self, root: str, evidence: list[dict[str, Any]]) -> dict[str, Any]:
        base = self._root(root)
        checked: list[dict[str, Any]] = []
        stale = 0
        for item in evidence[:200]:
            rel = str(item.get("path", ""))
            expected = str(item.get("file_sha256", ""))
            candidate = (base / rel).resolve(strict=False)
            try:
                candidate.relative_to(base)
            except ValueError:
                checked.append({"path": rel, "status": "invalid-path"}); stale += 1; continue
            if not candidate.is_file():
                checked.append({"path": rel, "status": "missing"}); stale += 1; continue
            actual = hashlib.sha256(candidate.read_bytes()).hexdigest()
            status = "current" if expected and actual == expected else "stale"
            if status != "current":
                stale += 1
            checked.append({"path": rel, "status": status, "expected_sha256": expected or None, "actual_sha256": actual})
        return {"success": True, "root": str(base), "stale": stale > 0, "stale_count": stale, "checked": checked}

    def git_diff(self, root: str, base: str = "HEAD", staged: bool = False, max_tokens: int = 10000) -> dict[str, Any]:
        repo = self._root(root)
        try:
            probe = subprocess.run(
                ["git", "-C", str(repo), "rev-parse", "--is-inside-work-tree"], capture_output=True, text=True,
                timeout=3, check=False, encoding="utf-8", errors="replace", **hidden_run_kwargs(),
            )
        except Exception as exc:
            return {"success": False, "error": str(exc), "terminal": True, "retryable": False}
        if probe.returncode != 0 or probe.stdout.strip().lower() != "true":
            return {"success": False, "error": "git diff requires a Git repository", "terminal": True, "retryable": False}
        cmd = ["git", "-C", str(repo), "diff", "--no-ext-diff", "--unified=3"]
        if staged:
            cmd.append("--cached")
        elif base:
            cmd.append(base)
        budget_chars = max(1, chars_for_tokens(max_tokens))
        def bounded_process(command: list[str], retain_limit: int, on_stdout_chunk: Any = None, digest_output: bool = False) -> dict[str, Any]:
            retained = bytearray()
            stderr_retained = bytearray()
            digest = hashlib.sha256() if digest_output else None
            total = 0
            stream_errors: list[str] = []

            def drain(stream: Any, output: bytearray, digest_value: Any = None, stream_name: str = "stdout") -> None:
                nonlocal total
                try:
                    while True:
                        chunk = stream.read(65536)
                        if not chunk:
                            return
                        if digest_value is not None:
                            digest_value.update(chunk)
                            total += len(chunk)
                        if on_stdout_chunk is not None and output is retained:
                            try:
                                on_stdout_chunk(chunk)
                            except Exception:
                                pass
                        remaining = max(0, retain_limit - len(output))
                        if remaining:
                            output.extend(chunk[:remaining])
                except Exception as exc:
                    if len(stream_errors) < 2:
                        detail = str(exc).replace("\r", " ").replace("\n", " ")[:200]
                        stream_errors.append(f"{stream_name}: {type(exc).__name__}: {detail}")
                    return

            process = None
            try:
                process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, **hidden_run_kwargs())
                stdout_thread = threading.Thread(target=drain, args=(process.stdout, retained, digest, "stdout"), daemon=True)
                stderr_thread = threading.Thread(target=drain, args=(process.stderr, stderr_retained, None, "stderr"), daemon=True)
                stdout_thread.start()
                stderr_thread.start()
                try:
                    returncode = process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
                    stdout_thread.join(timeout=5)
                    stderr_thread.join(timeout=5)
                    return {"returncode": None, "error": "git diff timed out", "retryable": True}
                stdout_thread.join(timeout=5)
                stderr_thread.join(timeout=5)
                if stdout_thread.is_alive() or stderr_thread.is_alive():
                    stream_errors.append("reader thread did not drain before timeout")
                if stream_errors:
                    return {
                        "returncode": None,
                        "error": "git diff stream read failed: " + "; ".join(stream_errors[:2]),
                        "retryable": True,
                    }
                return {
                    "returncode": returncode,
                    "stdout": bytes(retained),
                    "stderr": bytes(stderr_retained),
                    "total": total,
                    "sha256": digest.hexdigest() if digest is not None else "",
                }
            except Exception as exc:
                return {"returncode": None, "error": str(exc), "retryable": True}
            finally:
                if process is not None:
                    for stream in (process.stdout, process.stderr):
                        try:
                            stream.close()
                        except Exception:
                            pass

        content_result = bounded_process(cmd, budget_chars, digest_output=True)
        if content_result.get("returncode") is None:
            return {"success": False, "error": content_result.get("error", "git diff failed"), "retryable": content_result.get("retryable", False)}
        stderr_text = bytes(content_result.get("stderr", b"")).decode("utf-8", errors="replace").strip()
        if content_result["returncode"] != 0:
            return {"success": False, "error": stderr_text or "git diff failed"}
        stdout_retained = bytes(content_result.get("stdout", b""))
        stdout_total = int(content_result.get("total", len(stdout_retained)))
        text = stdout_retained.decode("utf-8", errors="replace")
        truncated = stdout_total > len(stdout_retained) or len(text) > budget_chars
        if len(text) > budget_chars:
            text = text[:budget_chars]
            truncated = True
        if truncated:
            text += "\n\n[... additional diff sections omitted to fit local review budget ...]\n"
        changed: list[str] = []
        retained_stdout = stdout_retained.decode("utf-8", errors="replace")
        for line in retained_stdout.splitlines():
            if not line.startswith("diff --git a/"):
                continue
            match = re.match(r"diff --git a/(.+?) b/(.+)$", line)
            if match:
                changed.append(match.group(2))
        path_names: list[str] = []
        path_pending = bytearray()
        path_capture_error = ""

        def collect_path_chunk(chunk: bytes) -> None:
            nonlocal path_capture_error
            if path_capture_error:
                return
            if len(path_names) >= 4096:
                path_capture_error = "path count cap reached"
                return
            data = bytes(path_pending) + chunk
            pieces = data.split(b"\0")
            tail = pieces.pop()
            if len(tail) > 4096:
                path_capture_error = "path byte cap reached"
                return
            path_pending.clear()
            path_pending.extend(tail)
            for piece in pieces:
                if not piece:
                    continue
                if len(piece) > 4096:
                    path_capture_error = "path byte cap reached"
                    return
                if len(path_names) >= 4096:
                    path_capture_error = "path count cap reached"
                    return
                path_names.append(piece.decode("utf-8", errors="replace"))

        path_cmd = ["git", "-C", str(repo), "diff", "--no-ext-diff", "--name-only", "-z"]
        if staged:
            path_cmd.append("--cached")
        elif base:
            path_cmd.append(base)
        path_result = bounded_process(path_cmd, 0, on_stdout_chunk=collect_path_chunk)
        if path_capture_error:
            return {
                "success": False,
                "error": f"git diff path capture failed: {path_capture_error}",
                "terminal": False,
                "retryable": True,
                "paths_complete": False,
            }
        if path_result.get("returncode") is None or path_result.get("returncode") != 0:
            path_error = bytes(path_result.get("stderr", b"")).decode("utf-8", errors="replace").strip()
            return {
                "success": False,
                "error": f"git diff path capture failed: {path_error or path_result.get('error', 'non-zero exit')}",
                "terminal": False,
                "retryable": bool(path_result.get("retryable", True)),
                "paths_complete": False,
            }
        if path_pending and len(path_names) < 4096:
            path_names.append(bytes(path_pending)[:4096].decode("utf-8", errors="replace"))
        snapshot = self.git_snapshot(str(repo))
        changed_files = list(dict.fromkeys(path_names + changed))
        changed_paths = list(dict.fromkeys(changed_files + list(snapshot.changed_paths)))
        return {
            "success": True, "root": str(repo), "diff": text, "changed_files": changed_files,
            "changed_paths": changed_paths, "revision": snapshot.revision,
            "repository_revision": snapshot.repository_revision,
            "estimated_tokens": estimate_tokens(text),
            "original_estimated_tokens": max(estimate_tokens(text), (stdout_total + 3) // 4),
            "truncated": truncated, "diff_sha256": content_result["sha256"],
            "paths_complete": True,
        }


    def impact_analysis(
        self,
        root: str,
        base: str = "HEAD",
        staged: bool = False,
        max_symbols: int = 48,
        max_dependents: int = 30,
    ) -> dict[str, Any]:
        """Cheap generic change-impact/test targeting without invoking an LLM.

        This is deliberately heuristic. Serena/CodeGraph remain the preferred source
        for exact symbol relationships; this fallback is available everywhere and is
        useful before spending cloud context.
        """
        repo = self._root(root)
        diff = self.git_diff(str(repo), base=base, staged=staged, max_tokens=2500)
        if not diff.get("success"):
            return diff
        changed_files = [str(x) for x in diff.get("changed_files", [])]
        if not changed_files:
            return {
                "success": True, "root": str(repo), "changed_files": [],
                "changed_symbols": [], "likely_dependents": [], "suggested_tests": [],
                "risk": {"score": 0, "level": "low", "reasons": ["no changed files"]},
            }

        changed_symbols: list[dict[str, Any]] = []
        changed_set = set(changed_files)
        for rel in changed_files:
            path = (repo / rel).resolve(strict=False)
            try:
                path.relative_to(repo)
            except ValueError:
                continue
            if not path.is_file() or len(changed_symbols) >= max_symbols:
                continue
            try:
                lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            except Exception:
                continue
            for line_no, line in enumerate(lines, 1):
                for pattern in _SYMBOL_PATTERNS:
                    match = pattern.match(line)
                    if match:
                        name = match.group(1)
                        if len(name) >= 3 and not any(x["name"] == name for x in changed_symbols):
                            changed_symbols.append({"name": name, "path": rel, "line": line_no})
                        break
                if len(changed_symbols) >= max_symbols:
                    break

        symbol_names = [x["name"] for x in changed_symbols]
        dependents: list[dict[str, Any]] = []
        test_candidates: list[dict[str, Any]] = []
        changed_stems = {Path(x).stem.lower().replace("test_", "").replace("_test", "") for x in changed_files}

        # Local AI deterministic fast path: let ripgrep reduce the dependent candidate
        # set before opening files. Exact counts are still verified against our own
        # decoded snapshots, so rg is only an accelerator and never a source of truth.
        dependent_candidates: set[str] | None = None
        if self._rg and symbol_names:
            cmd = [self._rg, "--files-with-matches", "--fixed-strings"]
            for name in symbol_names[:32]:
                cmd.extend(["-e", name])
            cmd.append(str(repo))
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=self.ripgrep_timeout, check=False, encoding="utf-8", errors="replace", **hidden_run_kwargs())
                if proc.returncode in {0, 1}:
                    dependent_candidates = set()
                    for raw in proc.stdout.splitlines():
                        try:
                            rel = str(Path(raw).resolve(strict=False).relative_to(repo)).replace("\\", "/")
                        except Exception:
                            continue
                        dependent_candidates.add(rel)
            except Exception:
                dependent_candidates = None

        for path in self.iter_files(str(repo)):
            rel = str(path.relative_to(repo)).replace("\\", "/")
            low_rel = rel.lower()
            is_test = (
                "/test" in f"/{low_rel}" or "/spec" in f"/{low_rel}"
                or Path(low_rel).name.startswith("test_")
                or any(marker in Path(low_rel).stem for marker in ("_test", ".test", "_spec", ".spec"))
            )
            stem = Path(low_rel).stem.lower().replace("test_", "").replace("_test", "").replace(".test", "").replace("_spec", "").replace(".spec", "")
            if is_test:
                score = sum(3 for changed in changed_stems if changed and (changed in stem or stem in changed))
                score += sum(1 for name in symbol_names[:20] if name.lower() in low_rel)
                if score:
                    test_candidates.append({"path": rel, "score": score})
            if rel in changed_set or not symbol_names:
                continue
            if dependent_candidates is not None and rel not in dependent_candidates:
                continue
            try:
                _digest, snapshot_lines = self._read_snapshot(path)
                text = "\n".join(snapshot_lines)
            except Exception:
                continue
            matched = []
            mentions = 0
            for name in symbol_names:
                count = text.count(name)
                if count:
                    matched.append(name)
                    mentions += min(count, 8)
            if matched:
                dependents.append({"path": rel, "symbols": matched[:8], "mentions": mentions})

        dependents.sort(key=lambda x: (-x["mentions"], x["path"]))
        test_candidates.sort(key=lambda x: (-x["score"], x["path"]))
        dependents = dependents[:max(1, int(max_dependents))]
        suggested_tests = test_candidates[:20]

        score = 0
        reasons: list[str] = []
        if len(changed_files) >= 6:
            score += 2; reasons.append("multi-file change")
        if len(changed_files) >= 15:
            score += 2; reasons.append("large change surface")
        risky_terms = {"auth", "security", "migration", "schema", "database", "transaction", "concurrency", "thread", "lock", "payment", "permission", "crypto"}
        if any(any(term in rel.lower() for term in risky_terms) for rel in changed_files):
            score += 2; reasons.append("sensitive subsystem path")
        if len(dependents) >= 10:
            score += 1; reasons.append("many likely dependents")
        if not suggested_tests and any(Path(x).suffix.lower() in self.extensions for x in changed_files):
            score += 1; reasons.append("no likely tests found")
        if diff.get("truncated"):
            score += 1; reasons.append("diff exceeded local impact budget")
        level = "high" if score >= 5 else "medium" if score >= 2 else "low"
        if not reasons:
            reasons.append("small localized change")

        return {
            "success": True, "root": str(repo), "changed_files": changed_files,
            "changed_symbols": changed_symbols, "likely_dependents": dependents,
            "suggested_tests": suggested_tests,
            "risk": {"score": score, "level": level, "reasons": reasons},
            "diff_sha256": diff.get("diff_sha256"),
            "note": "Heuristic fallback; use Serena/CodeGraph for exact symbol relationships when available.",
        }



    def search_paths(self, root: str, query: str, paths: list[str], top_k: int = 10, context_lines: int | None = None, path: str = "") -> dict[str, Any]:
        """Search only preselected files. Used with preprocessed semantic cards to avoid full-repo scans."""
        base = self._root(root)
        try:
            scope = self._search_scope(base, path)
        except ValueError as exc:
            return self._invalid_search_scope(base, exc)
        terms, literal = self._search_terms(query)
        phrase = query.strip().lower()
        effective_snippet_lines = max(1, min(int(context_lines), 40)) if context_lines is not None else self.snippet_lines
        hits: list[dict[str, Any]] = []
        scanned = 0
        seen_paths: set[str] = set()
        for rel in paths[: max(1, min(len(paths), 64))]:
            rel = str(rel).replace("\\", "/")
            if rel in seen_paths:
                continue
            seen_paths.add(rel)
            path = (base / rel).resolve(strict=False)
            if not self._in_search_scope(base, path, scope):
                continue
            if not path.is_file():
                continue
            scanned += 1
            try:
                file_hash, lines = self._read_snapshot(path)
            except Exception:
                continue
            path_lower = rel.lower()
            for idx, line in enumerate(lines):
                low = line.lower()
                matched = self._line_matches(low, terms, literal)
                path_matches = sum(1 for t in terms if t in path_lower)
                if not matched:
                    continue
                score = float(len(matched) * 3 + path_matches * 2)
                if phrase and len(phrase) <= 120 and phrase in low:
                    score += 8
                score += sum(min(low.count(t), 3) * 0.5 for t in matched)
                start = max(0, idx - effective_snippet_lines)
                end = min(len(lines), idx + effective_snippet_lines + 1)
                snippet = "\n".join(f"{n + 1}: {lines[n]}" for n in range(start, end))
                hits.append({
                    "path": rel, "start_line": start + 1, "end_line": end,
                    "score": round(score, 3), "text": snippet, "file_sha256": file_hash,
                })
        hits.sort(key=lambda x: (-x["score"], x["path"], x["start_line"]))
        dedup: list[dict[str, Any]] = []
        per_file: Counter[str] = Counter()
        for hit in hits:
            if per_file[hit["path"]] >= self.max_snippets_per_file:
                continue
            per_file[hit["path"]] += 1
            dedup.append(hit)
            if len(dedup) >= max(1, int(top_k)):
                break
        candidate_pool = hits[: max(len(dedup), max(1, int(top_k)) * 4)]
        candidate_tokens = sum(estimate_tokens(str(item.get("text", ""))) for item in candidate_pool)
        result = {"success": True, "root": str(base), "scanned_files": scanned, "terms": terms, "results": dedup, "targeted": True}
        if candidate_tokens:
            result["candidate_context_tokens_est"] = candidate_tokens
        return result

    def context_pack_paths(self, root: str, query: str, paths: list[str], max_tokens: int = 2600, top_k: int = 10) -> dict[str, Any]:
        search = self.search_paths(root, query, paths, top_k=top_k)
        if not search.get("success"):
            return search
        budget_chars = chars_for_tokens(max_tokens)
        pieces: list[str] = []
        evidence: list[dict[str, Any]] = []
        used = 0
        candidate_tokens = 0
        for item in search.get("results", []):
            header = f"--- {item['path']}:{item['start_line']}-{item['end_line']} score={item['score']} ---\n"
            piece = header + item["text"] + "\n"
            candidate_tokens += estimate_tokens(piece)
            if used + len(piece) > budget_chars:
                remaining = budget_chars - used
                if remaining <= 500:
                    break
                piece = piece[:remaining] + "\n[snippet truncated]\n"
            pieces.append(piece); used += len(piece)
            ev = {k: item[k] for k in ("path", "start_line", "end_line", "score", "file_sha256")}
            ev["raw"] = item.get("text", "")
            evidence.append(ev)
            if used >= budget_chars:
                break
        packed = "\n".join(pieces)
        return {
            "success": True, "root": search["root"], "query": query, "context": packed,
            "evidence": evidence, "estimated_tokens": estimate_tokens(packed),
            "original_estimated_tokens": max(candidate_tokens, estimate_tokens(packed)),
            "scanned_files": search.get("scanned_files", 0), "targeted": True,
        }

    def file_inventory(self, root: str, include_hashes: bool = True) -> dict[str, Any]:
        """Return a deterministic file inventory backed by the shared snapshot cache.

        Preprocessing uses this as its durable content-addressed source of truth.
        """
        base = self._root(root)
        items: list[dict[str, Any]] = []
        total_bytes = 0
        for path in self.iter_files(str(base)):
            try:
                stat = path.stat()
                rel = str(path.relative_to(base)).replace("\\", "/")
                item: dict[str, Any] = {
                    "path": rel, "size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns),
                    "extension": path.suffix.lower(),
                }
                if include_hashes:
                    digest, _lines = self._read_snapshot(path)
                    item["sha256"] = digest
                items.append(item)
                total_bytes += int(stat.st_size)
            except Exception:
                continue
        items.sort(key=lambda x: x["path"])
        digest = hashlib.sha256(
            "\n".join(f"{x['path']}:{x.get('sha256','')}:{x['size']}" for x in items).encode("utf-8")
        ).hexdigest()
        return {"success": True, "root": str(base), "files": items, "file_count": len(items), "total_bytes": total_bytes, "inventory_sha256": digest}

    def file_slice(self, root: str, path: str, start_line: int = 1, end_line: int = 240, max_chars: int = 16000) -> dict[str, Any]:
        """Read an exact bounded source slice through the snapshot cache."""
        base = self._root(root)
        candidate = (base / path).resolve(strict=False)
        try:
            candidate.relative_to(base)
        except ValueError:
            return {"success": False, "error": "path escapes repository root"}
        if not candidate.is_file():
            return {"success": False, "error": f"file not found: {path}"}
        try:
            digest, lines = self._read_snapshot(candidate)
        except Exception as exc:
            return {"success": False, "error": str(exc)}
        rel = str(candidate.relative_to(base)).replace("\\", "/")
        if not lines:
            return {
                "success": True, "root": str(base), "path": rel, "start_line": 1, "end_line": 0,
                "text": "", "file_sha256": digest, "truncated": False,
            }
        start = max(1, min(int(start_line), len(lines)))
        end = max(start, min(int(end_line), len(lines)))
        rendered = "\n".join(f"{n}: {lines[n-1]}" for n in range(start, end + 1))
        truncated = len(rendered) > max_chars
        if truncated:
            rendered = rendered[:max_chars] + "\n[…slice truncated…]"
        return {
            "success": True, "root": str(base), "path": rel, "start_line": start, "end_line": end,
            "text": rendered, "file_sha256": digest, "truncated": truncated,
        }

    def context_pack(self, root: str, query: str, max_tokens: int = 4500, top_k: int = 14) -> dict[str, Any]:
        search = self.search(root, query, top_k=top_k)
        if not search.get("success"):
            return search
        budget_chars = chars_for_tokens(max_tokens)
        pieces: list[str] = []
        evidence: list[dict[str, Any]] = []
        used = 0
        candidate_tokens = 0
        for item in search.get("results", []):
            header = f"--- {item['path']}:{item['start_line']}-{item['end_line']} score={item['score']} ---\n"
            piece = header + item["text"] + "\n"
            candidate_tokens += estimate_tokens(piece)
            if used + len(piece) > budget_chars:
                remaining = budget_chars - used
                if remaining > 500:
                    piece = piece[:remaining] + "\n[snippet truncated]\n"
                else:
                    break
            pieces.append(piece)
            used += len(piece)
            ev = {k: item[k] for k in ("path", "start_line", "end_line", "score", "file_sha256")}
            ev["raw"] = item.get("text", "")
            evidence.append(ev)
            if used >= budget_chars:
                break
        packed = "\n".join(pieces)
        return {
            "success": True, "root": search["root"], "query": query, "context": packed,
            "evidence": evidence, "estimated_tokens": estimate_tokens(packed),
            "original_estimated_tokens": max(candidate_tokens, estimate_tokens(packed)),
            "scanned_files": search.get("scanned_files", 0),
        }
