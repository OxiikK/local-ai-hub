from __future__ import annotations

from .json_utils import dumps as json_dumps

import base64
import copy
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import time
import threading
from dataclasses import asdict
from contextlib import closing, contextmanager
from pathlib import Path
from typing import Any

from . import __version__
from .artifacts import ArtifactStore
from .budget import chars_for_tokens, estimate_tokens, fit_text
from .cache import MemoryLRUCache, SQLiteCache, TieredCache, SingleFlightCache, SingleFlightGroup, stable_hash
from .conversations import ConversationStore
from .features import rollout_feature_enabled
from .normalizer import normalize_query, postprocess_model_output
from .semantic_cache import SemanticGenerationCache
from .model_policy import ModelExecutionPolicy
from .ollama_subagents import OllamaSubagentCatalog
from .process_utils import canonical_root
from .repo_tools import RepositoryTools
from .router import ModelRouter, review_diff_complexity
from .sqlite_support import connect_sqlite
from .adoption_metrics import AdoptionMetricsStore
from .state_paths import configured_state_dir
from .speculative_lint import normalize_changed_paths
from .telemetry import TelemetryStore
from .trace_context import observer
from .treesitter_parser import parse_treesitter
from .vision_contracts import (
    UNTRUSTED_CONTEXT_INSTRUCTION,
    VISION_MAX_BUNDLE_CHARS,
    VISION_MAX_IMAGE_BYTES,
    VISION_MAX_IMAGE_CHARS,
    VISION_MAX_INLINE_RESPONSE_CHARS,
    VISION_MAX_PROMPT_CHARS,
    VISION_MAX_RUNTIME_OUTPUT_CHARS,
    VISION_MAX_SCHEMA_CHARS,
    bound_vision_result,
    parse_vision_result,
)
from .frontend_review import (
    FrontendReviewError,
    build_coder_context,
    build_model_context,
    parse_bounded_context,
)
from .agent_consistency import AdaptiveContextPack, ConsistencyRequest, GuardWarning
from .agent_context import ContextCompiler, ContextRequest
from .agent_events import AgentStateStore
from .semantic_quality import assess_semantic_result
from .prompt_contracts import build_prompt


def is_retryable_local_backend_error(value: Any) -> bool:
    """Recognize transport/runtime failures that must keep async work retryable."""
    text = str(value or "").lower()
    return any(marker in text for marker in (
        "winerror 10054", "winerror 10061", "connection reset", "connection refused",
        "connection aborted", "broken pipe", "temporarily unavailable", "ollama unavailable",
        "timed out", "timeout", "eof",
    ))


def normalize_generation_cache_prompt(prompt: str) -> str:
    """Normalize only transport-level line endings for exact prompt reuse."""
    return str(prompt).replace("\r\n", "\n").replace("\r", "\n")


def cache_decision_reason(cache_layer: str, *, semantic_query: str) -> str:
    if cache_layer == "semantic":
        return "semantic_reuse"
    if cache_layer == "single-flight":
        return "singleflight_reuse"
    if cache_layer == "stale-on-error":
        return "stale_on_error"
    if cache_layer == "ollama" and semantic_query:
        return "semantic_not_reused"
    if cache_layer == "ollama":
        return "new_exact_key"
    return "exact_reuse"


def enclosing_symbol_at_line(source: str, path: str, line: int) -> dict[str, Any] | None:
    """Return the smallest existing Tree-sitter symbol spanning one source line."""
    language = Path(path).suffix.lower().lstrip(".")
    parsed = parse_treesitter(source, language)
    if not parsed:
        return None
    symbols, _, _ = parsed
    matches = [
        symbol for symbol in symbols
        if int(symbol.get("line", 0) or 0) <= line <= int(symbol.get("end_line", 0) or 0)
    ]
    if not matches:
        return None
    symbol = min(matches, key=lambda item: int(item.get("end_line", 0) or 0) - int(item.get("line", 0) or 0))
    return {
        key: symbol[key] for key in ("name", "kind", "line", "end_line", "name_path")
        if key in symbol
    }


_REVIEW_DIFF_MIN_CHUNK_TOKENS = 256
_REVIEW_DIFF_CONTEXT_FRACTION = 0.5
_MAX_REVIEW_DIFF_CHUNKS = 8
_REVIEW_SYNTHESIS_CONTEXT_TOKENS = 6000
_UNIFIED_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(.*)$")
_GUARD_REASON_SECRET_RE = re.compile(
    r"(?i)\b((?:token|api[-_]?key|access[-_]?token|refresh[-_]?token|auth(?:orization)?|bearer|secret|password|passwd|credential|cookie|private[-_]?key)\b\s*[:=]\s*)(?!Bearer\b)([\"']?)([^\"'\s,;]+)\2"
)
_GUARD_REASON_PROMPT_RE = re.compile(r"(?is)\b(?:system\s+|user\s+)?(?:prompt|instructions?)\s*[:=].*$")


def _redact_guard_reason(reason: Any) -> str:
    """Keep a short operator reason without persisting secret or prompt content."""
    text = str(reason or "").strip()
    if not text:
        return ""
    text = re.sub(r"(?i)\bBearer\s+\S+", "Bearer [REDACTED]", text)
    text = _GUARD_REASON_SECRET_RE.sub(lambda match: f"{match.group(1)}[REDACTED]", text)
    text = _GUARD_REASON_PROMPT_RE.sub("prompt: [REDACTED]", text)
    return text[:500]


def _review_text_error(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return "Model returned empty review output."
    text = value.strip()
    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3:
            text = "\n".join(lines[1:-1]).strip()

    # Match SUMMARY: with tolerance for markdown headers (#, ##), bullet points (- , * ),
    # bolding (**SUMMARY:** or **SUMMARY**), backticks (`SUMMARY:`), or 1 introductory line.
    match = re.search(
        r"(?:^|\n)(?:#{1,6}\s*)?(?:[-*+]\s+)?(?:\*{1,2}|`{1,3})?SUMMARY(?:\*{1,2}|`{1,3})?:?[ \t]*(.*)$",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if match is None or match.start() > 250:
        return "Model output did not start with the required SUMMARY: header."
    words = re.findall(r"[^\W\d_]+", match.group(1), flags=re.UNICODE)
    if len(words) < 1:
        return "Model returned an incomplete or malformed review summary."
    return None


def _review_model_identity(value: Any) -> str:
    """Normalize Ollama quantization/instruct tags for tier comparisons."""
    model = str(value or "").strip().casefold()
    return re.sub(r"-(?:instruct|q\d+).*?$", "", model)


def _merge_review_segment_results(
    segment_results: list[dict[str, Any]],
    *,
    label: str,
    task_type: str,
    chunked: bool,
    synthesis_context_budget: int,
    synthesis_task: str,
    max_output_tokens: int,
    complexity: str,
    tenant: str,
    delegate: Any,
    priority: int = 5,
) -> tuple[str, dict[str, Any]]:
    """Return raw review output or a bounded synthesis of chunked results."""
    raw_text = "\n\n".join(
        f"### {label} {index}/{len(segment_results)}\n{str(item.get('text', '')).strip()}"
        for index, item in enumerate(segment_results, start=1)
    )
    if not chunked:
        return str(segment_results[0].get("text", "")), {"enabled": False}

    if estimate_tokens(raw_text) > synthesis_context_budget:
        return raw_text, {"enabled": True, "degraded": True, "error": "Segment findings exceed synthesis budget."}
    try:
        merged = delegate(
            {
                "task_type": task_type,
                "task": synthesis_task,
                "context": raw_text,
                "complexity": complexity,
                "max_tokens": min(max_output_tokens, 1800),
                "priority": int(priority),
            },
            tenant,
        )
        if isinstance(merged, dict) and merged.get("success") is not False:
            output_error = _review_text_error(merged.get("text"))
            if output_error is None:
                return str(merged["text"]), {
                    "enabled": True,
                    "degraded": False,
                    "model": merged.get("model"),
                }
            error = f"Synthesis returned unusable output: {output_error}"
        else:
            error = merged.get("error", "Synthesis returned no text.") if isinstance(merged, dict) else "Synthesis returned no result."
        return raw_text, {"enabled": True, "degraded": True, "error": str(error)}
    except Exception as exc:
        return raw_text, {"enabled": True, "degraded": True, "error": str(exc)}


def _split_review_diff(diff_text: str, max_tokens: int) -> list[str]:
    """Split a unified diff into bounded, file/hunk-aligned review inputs."""
    budget = max(_REVIEW_DIFF_MIN_CHUNK_TOKENS, int(max_tokens))
    sections = [part for part in re.split(r"(?=^diff --git )", diff_text, flags=re.MULTILINE) if part]
    if not sections:
        return [diff_text] if diff_text else []

    chunks: list[str] = []
    current = ""

    def flush() -> None:
        nonlocal current
        if current:
            chunks.append(current)
            current = ""

    for section in sections:
        if estimate_tokens(section) <= budget:
            if current and estimate_tokens(current + section) > budget:
                flush()
            current += section
            continue

        flush()
        chunks.extend(_split_large_review_file(section, budget))
    flush()
    return chunks or [diff_text]


def _split_large_review_file(section: str, budget: int) -> list[str]:
    lines = section.splitlines(keepends=True)
    hunk_starts = [index for index, line in enumerate(lines) if line.startswith("@@ ")]
    if not hunk_starts:
        return [section]

    prefix = "".join(lines[:hunk_starts[0]])
    hunks = [
        "".join(lines[start:end])
        for start, end in zip(hunk_starts, hunk_starts[1:] + [len(lines)])
    ]
    chunks: list[str] = []
    pending_hunks: list[str] = []

    def flush_pending() -> None:
        if pending_hunks:
            chunks.append(prefix + "".join(pending_hunks))
            pending_hunks.clear()

    for hunk in hunks:
        candidate = prefix + "".join(pending_hunks) + hunk
        if estimate_tokens(candidate) <= budget:
            pending_hunks.append(hunk)
            continue

        flush_pending()
        if estimate_tokens(prefix + hunk) <= budget:
            pending_hunks.append(hunk)
        else:
            chunks.extend(_split_large_review_hunk(prefix, hunk, budget))

    flush_pending()
    return chunks or [section]


def _split_large_review_hunk(prefix: str, hunk: str, budget: int) -> list[str]:
    lines = hunk.splitlines(keepends=True)
    if not lines:
        return [prefix + hunk]
    match = _UNIFIED_HUNK_RE.match(lines[0].rstrip("\r\n"))
    if match is None:
        return [prefix + hunk]

    old_start = int(match.group(1))
    old_consumed = 0
    new_start = int(match.group(3))
    new_consumed = 0
    old_count = new_count = 0
    suffix = match.group(5)
    body: list[str] = []
    chunks: list[str] = []

    def render(part: list[str], old_offset: int, new_offset: int, old_lines: int, new_lines: int) -> str:
        header = f"@@ -{old_start + old_offset},{old_lines} +{new_start + new_offset},{new_lines} @@{suffix}\n"
        return prefix + header + "".join(part)

    def flush_body() -> None:
        nonlocal old_consumed, new_consumed, old_count, new_count, body
        if not body:
            return
        chunks.append(render(body, old_consumed, new_consumed, old_count, new_count))
        old_consumed += old_count
        new_consumed += new_count
        old_count = new_count = 0
        body = []

    for line in lines[1:]:
        adds_old = int(line.startswith((" ", "-")))
        adds_new = int(line.startswith((" ", "+")))
        proposed = body + [line]
        candidate = render(proposed, old_consumed, new_consumed, old_count + adds_old, new_count + adds_new)
        if body and estimate_tokens(candidate) > budget:
            flush_body()
        body.append(line)
        old_count += adds_old
        new_count += adds_new
    flush_body()
    return chunks or [prefix + hunk]


def generation_cache_key(
    *,
    model: str,
    prompt: str,
    system: str,
    options: dict[str, Any],
    think: Any,
    execution: Any,
    format: Any = None,
    repository_revision: str = "",
    phase: str = "",
    memory_revision: str = "",
    focus: Any = None,
    preload_profile: str = "",
) -> str:
    return stable_hash({
        "model": model,
        "prompt": normalize_generation_cache_prompt(prompt),
        "system": normalize_generation_cache_prompt(system),
        "options": options,
        "think": think,
        "execution": execution,
        "format": format,
        "repository_revision": repository_revision,
        "phase": phase,
        "memory_revision": memory_revision,
        "focus": focus,
        "preload_profile": preload_profile,
        "app_version": __version__,
    })


def normalize_context_for_hash(context: str) -> str:
    """Strip variable whitespace from context to ensure identical code/evidence matches fingerprint."""
    raw = str(context).replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    return "\n".join(lines)
from .resilience import CircuitBreakerRegistry
from .process_utils import hidden_run_kwargs

_GLOBAL_VRAM_LOCK = threading.RLock()


@contextmanager
def vram_priority(is_generation: bool = True):
    """Global lock to serialize heavy batch operations and prioritize LLM inference against VRAM thrashing."""
    with _GLOBAL_VRAM_LOCK:
        yield


class LocalAIServices:
    commands: Any = None
    task_store: Any = None
    verification_store: Any = None
    incident_store: Any = None
    adoption_metrics: Any = None
    rag: Any = None
    token_router: Any = None
    pipeline: Any = None
    preprocessor: Any = None
    tool_agent: Any = None
    blackboard: Any = None
    agent_state: Any = None
    _VISION_MAX_OUTPUT_TOKENS = 4096
    consistency_guard: Any = None

    def __init__(
        self,
        config: dict[str, Any],
        runtime: Any,
        scheduler: Any,
        embeddings: Any,
        artifacts: ArtifactStore,
        telemetry: TelemetryStore,
        repo_tools: RepositoryTools,
        repo_state: Any | None = None,
        code_index: Any | None = None,
        evidence: Any | None = None,
        learner: Any | None = None,
        tuner: Any | None = None,
        deterministic: Any | None = None,
        external_tools: Any | None = None,
    ):
        self.config = config
        self.runtime = runtime
        self.scheduler = scheduler
        self.embeddings = embeddings
        self.artifacts = artifacts
        self.telemetry = telemetry
        self.repo_tools = repo_tools
        if repo_state is None:
            from .repo_state import RepoStateTracker
            repo_state = RepoStateTracker(config)
        self.repo_state = repo_state
        self.code_index = code_index
        self.evidence_store = evidence
        self.learner = learner
        self.tuner = tuner
        self.deterministic = deterministic
        self.external_tools = external_tools
        self.commands: Any | None = None
        self.task_store: Any | None = None
        self.verification_store: Any | None = None
        self.incident_store: Any | None = None
        self.adoption_metrics = AdoptionMetricsStore(configured_state_dir(config))
        self.router = ModelRouter(config)
        self.model_policy = ModelExecutionPolicy(config)
        self.profile_catalog = OllamaSubagentCatalog(config)
        self.rag: Any | None = None
        self.token_router: Any | None = None
        self.pipeline: Any | None = None
        self.preprocessor: Any | None = None
        self.tool_agent: Any | None = None
        self.swarm: Any | None = None
        self.blackboard: Any | None = None
        self.agent_state: Any | None = None
        self.consistency_guard: Any | None = None
        self.flight_group = SingleFlightGroup(shards=32, default_timeout_seconds=60.0)

        cache_cfg = config.get("cache", {})
        persistent = SQLiteCache(
            Path(config["server"]["state_dir"]) / "cache.sqlite3",
            namespace="generation",
            ttl_seconds=int(cache_cfg.get("generation_ttl_seconds", 7 * 86400)),
            max_entries=int(cache_cfg.get("generation_max_entries", 5000)),
        )
        generation_tier = TieredCache(
            persistent,
            l1_entries=int(cache_cfg.get("l1_generation_entries", 512)),
            l1_ttl_seconds=int(cache_cfg.get("l1_generation_ttl_seconds", 1800)),
        )
        self.generation_cache = SingleFlightCache(generation_tier, enabled=bool(cache_cfg.get("generation", True)), wait_timeout_seconds=float(config.get("resilience", {}).get("singleflight_wait_timeout_seconds", 45)))
        embedding_cache = SQLiteCache(
            Path(config["server"]["state_dir"]) / "cache.sqlite3",
            namespace="embeddings",
            ttl_seconds=int(cache_cfg.get("embedding_ttl_seconds", 30 * 86400)),
            max_entries=int(cache_cfg.get("embedding_max_entries", 100000)),
        )
        self.embedding_cache = TieredCache(
            embedding_cache,
            l1_entries=int(cache_cfg.get("l1_embedding_entries", 16384)),
            l1_ttl_seconds=int(cache_cfg.get("l1_embedding_ttl_seconds", 7200)),
        )
        self.embedding_cache_enabled = bool(cache_cfg.get("embeddings", True))
        workspace_cfg = config.get("workspace_cache", {})
        self.repo_cache = TieredCache(
            SQLiteCache(
                Path(config["server"]["state_dir"]) / "cache.sqlite3",
                namespace="repo-results",
                ttl_seconds=int(workspace_cfg.get("repo_result_ttl_seconds", 7 * 86400)),
                max_entries=int(workspace_cfg.get("repo_result_max_entries", 20000)),
            ),
            l1_entries=int(cache_cfg.get("l1_repo_entries", 512)),
            l1_ttl_seconds=int(cache_cfg.get("l1_repo_ttl_seconds", 900)),
        )
        self.repo_flight = SingleFlightCache(self.repo_cache, enabled=True, wait_timeout_seconds=float(config.get("resilience", {}).get("singleflight_wait_timeout_seconds", 45)))
        self.semantic_cache = SemanticGenerationCache(config, embeddings)
        resilience = config.get("resilience", {})
        self.breakers = CircuitBreakerRegistry(
            failure_threshold=int(resilience.get("circuit_failure_threshold", 3)),
            cooldown_seconds=float(resilience.get("circuit_cooldown_seconds", 20)),
        )
        self.stale_generation_cache = SQLiteCache(
            Path(config["server"]["state_dir"]) / "cache.sqlite3",
            namespace="generation-stale",
            ttl_seconds=int(resilience.get("stale_generation_ttl_seconds", 30 * 86400)),
            max_entries=int(resilience.get("stale_generation_max_entries", 10000)),
        )
        self._recent_focus_symbols: dict[str, list[str]] = {}
        self._focus_lock = threading.Lock()
        self._vram_model_lock = threading.RLock()
        self.fallback_count = 0
        self._semantic_lock_guard = threading.Lock()
        self._semantic_scope_locks: dict[str, threading.Lock] = {}
        self._index_refresh_lock = threading.Lock()
        self._index_refresh_state: dict[str, str] = {}
        self._query_expansion_l1 = MemoryLRUCache(1024, ttl_seconds=3600)
        conversation_cfg = config.get("model_conversations", {})
        self.conversations = ConversationStore(
            idle_ttl_seconds=float(conversation_cfg.get("idle_ttl_seconds", 900)),
            max_turns=int(conversation_cfg.get("max_turns", 12)),
            max_conversations=int(conversation_cfg.get("max_conversations", 1000)),
        )

    def set_rag(self, rag: Any) -> None:
        self.rag = rag

    @contextmanager
    def vram_priority(self, is_generation: bool = True):
        """Acquire lock to serialize heavy batch operations and prioritize LLM inference against VRAM thrashing."""
        with self._vram_model_lock:
            yield

    def set_commands(self, commands: Any) -> None:
        self.commands = commands

    def set_task_store(self, task_store: Any) -> None:
        self.task_store = task_store

    def set_verification_store(self, verification_store: Any) -> None:
        self.verification_store = verification_store

    def set_incident_store(self, incident_store: Any) -> None:
        self.incident_store = incident_store

    def set_token_router(self, token_router: Any) -> None:
        self.token_router = token_router

    def set_pipeline(self, pipeline: Any) -> None:
        self.pipeline = pipeline

    def set_preprocessor(self, preprocessor: Any) -> None:
        self.preprocessor = preprocessor

    def set_tool_agent(self, tool_agent: Any) -> None:
        self.tool_agent = tool_agent

    def set_blackboard(self, blackboard: Any) -> None:
        self.blackboard = blackboard

    def set_swarm(self, swarm: Any) -> None:
        self.swarm = swarm

    def set_vram_balancer(self, vram_balancer: Any) -> None:
        self.vram_balancer = vram_balancer

    def set_agent_state(self, store: Any) -> None:
        self.agent_state = store

    def set_consistency_guard(self, guard: Any) -> None:
        self.consistency_guard = guard

    @staticmethod
    def _guard_truthy(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        return str(value or "").strip().lower() not in {"", "0", "false", "no", "off", "none"}

    def _guard_decision(self, request: ConsistencyRequest, revision: str, *, reason: str = "", approval: Any = "") -> tuple[bool, bool]:
        """Persist bounded operator decisions through the guard's existing memory store."""
        reason_text = str(reason or "").strip()
        if not reason_text and not self._guard_truthy(approval):
            return False, False
        store = getattr(self.consistency_guard, "memory_store", None)
        if store is None or not callable(getattr(store, "record", None)):
            return True, False
        state_store = getattr(store, "state_store", None)
        if state_store is not None and getattr(state_store, "enabled", True) is False:
            return True, False
        if getattr(store, "enabled", True) is False:
            return True, False
        try:
            from .agent_identity import AgentScope
            from .agent_memory import MemoryKind, MemoryRecord

            token = hashlib.sha256(
                f"{request.root}:{request.task_id}:{request.phase}:{revision}:{reason_text}:{approval}".encode("utf-8", "replace")
            ).hexdigest()[:16]
            record = MemoryRecord.create(
                kind=MemoryKind.DECISION,
                scope=AgentScope.REPOSITORY,
                key=f"consistency_decision:{token}",
                value={"approved": self._guard_truthy(approval), "reason": _redact_guard_reason(reason_text)},
                source="consistency_guard",
                repository_revision=revision,
                path_refs=tuple(request.changed_paths),
                related_task=request.task_id or None,
                provenance={"root": request.root, "phase": request.phase, "guard": "agent_consistency"},
            )
            try:
                saved = store.record(record, actor="consistency_guard", idempotency_key=f"consistency-decision:{token}")
            except TypeError:
                saved = store.record(record)
            return True, saved is not None and saved is not False
        except Exception:
            return True, False

    def _guard_existing_approval(self, request: ConsistencyRequest) -> bool:
        store = getattr(self.consistency_guard, "memory_store", None)
        finder = getattr(store, "find", None)
        if not callable(finder) or not request.task_id:
            return False
        try:
            records = finder(root=request.root, limit=12, semantic=False)
        except TypeError:
            try:
                records = finder(limit=12)
            except Exception:
                return False
        except Exception:
            return False
        for record in records or ():
            kind = getattr(getattr(record, "kind", None), "value", getattr(record, "kind", ""))
            value = getattr(record, "value", {})
            provenance = getattr(record, "provenance", {}) or {}
            if kind != "decision" or provenance.get("related_task") != request.task_id:
                continue
            if isinstance(value, dict) and self._guard_truthy(value.get("approved") or value.get("approval")):
                return True
        return False

    def _guard_task_state(self, request: ConsistencyRequest, warnings: tuple[GuardWarning, ...], revision: str) -> tuple[str, bool]:
        store = getattr(self.consistency_guard, "task_store", None)
        if store is None or not request.task_id or not callable(getattr(store, "get", None)):
            return "", False
        try:
            state = store.get(request.task_id)
        except Exception:
            return "", False
        if state is None:
            return "", False
        status = getattr(getattr(state, "status", None), "value", getattr(state, "status", ""))
        boundary = tuple(item for item in warnings if item.requires_approval or item.severity.lower() in {"boundary", "high", "high-risk"})
        checkpoint_data = getattr(getattr(state, "checkpoint", None), "state_data", {}) or {}
        checkpoint_approval = checkpoint_data.get("approval") or checkpoint_data.get("approved") or checkpoint_data.get("coordinator_decision")
        approved = self._guard_truthy(request.approval) or self._guard_existing_approval(request) or self._guard_truthy(checkpoint_approval)
        if boundary and not approved and status == "active":
            warning_ids = tuple(dict.fromkeys(evidence_id for item in boundary for evidence_id in item.evidence_ids))[:24]
            try:
                from .agent_tasks import TaskCheckpoint, TaskStatus

                store.transition(
                    request.task_id,
                    TaskStatus.WAITING,
                    reason="consistency guard boundary warning",
                    actor="consistency_guard",
                    idempotency_key=f"guard-wait:{request.task_id}:{revision}",
                )
                store.checkpoint(
                    request.task_id,
                    TaskCheckpoint(
                        phase=request.phase,
                        next_action=boundary[0].recommended_action or "obtain approval before continuing",
                        affected_paths=tuple(request.changed_paths),
                        evidence_ids=warning_ids,
                        blockers=tuple(item.code for item in boundary)[:24],
                        state_data={"warning_ids": list(warning_ids), "repository_revision": revision},
                    ),
                    actor="consistency_guard",
                    idempotency_key=f"guard-checkpoint:{request.task_id}:{revision}",
                )
            except Exception:
                pass
        elif approved and status == "waiting":
            try:
                store.resume(request.task_id, actor="consistency_guard", idempotency_key=f"guard-resume:{request.task_id}:{revision}")
            except Exception:
                pass
        try:
            refreshed = store.get(request.task_id)
            refreshed_status = getattr(getattr(refreshed, "status", None), "value", getattr(refreshed, "status", ""))
            refreshed_status = str(refreshed_status)
            return refreshed_status, bool(refreshed_status.lower() == "waiting" or (boundary and not approved))
        except Exception:
            return str(status), bool(boundary and not approved)

    def _guard_stop_code(self, request: ConsistencyRequest, boundary: bool) -> str:
        """Return terminal stop code when high-risk delivery lacks Agent OS state."""
        if not boundary:
            return ""
        if not request.task_id:
            return "guard_task_id_required"
        agent_state = getattr(self, "agent_state", None)
        if agent_state is not None and getattr(agent_state, "enabled", True) is False:
            return "agent_os_disabled"
        store = getattr(self.consistency_guard, "task_store", None)
        if store is None or getattr(store, "enabled", True) is False:
            return "agent_os_disabled"
        state_store = getattr(store, "state_store", None)
        if state_store is not None and getattr(state_store, "enabled", True) is False:
            return "agent_os_disabled"
        getter = getattr(store, "get", None)
        if not callable(getter):
            return "guard_task_state_unavailable"
        try:
            if getter(request.task_id) is None:
                return "guard_task_state_unavailable"
        except Exception:
            return "guard_task_state_unavailable"
        return ""

    def _touch_project(self, root: str) -> None:
        # Local AI: repository reads refresh only explicitly registered projects.
        if self.preprocessor is not None:
            try:
                self.preprocessor.touch_if_registered(root)
            except Exception:
                pass

    def _repo_cache_state(self, root: str) -> dict[str, Any]:
        """Prefer watcher-backed preprocessing state over spawning Git on hot reads."""
        if self.preprocessor is not None:
            try:
                fast = self.preprocessor.cache_fingerprint(root)
                if isinstance(fast, dict) and fast.get("fingerprint"):
                    return fast
            except Exception:
                pass
        return self.repo_state.fingerprint(root)

    def _resident_optimize(self, route: dict[str, Any], task_type_override: str, complexity_override: str) -> dict[str, Any]:
        cfg = self.config.get("routing", {})
        if not cfg.get("prefer_resident_model", True):
            return route
        try:
            active = self.scheduler.status().get("active_model")
        except Exception:
            active = None
        if not active:
            return route

        models = self.config["models"]
        desired = route["model"]
        reason = None
        if route["task_type"] in {"code", "review"}:
            if desired == models.get("fast_code") and active == models.get("heavy_code"):
                # Heavy meets the fast tier quality gate. Once benchmark history exists,
                # keep it resident only when that is actually expected to be faster.
                tuner_ready = bool(self.tuner is not None and self.tuner.ready(str(desired), str(active)))
                if tuner_ready and self.tuner.prefer_resident(str(desired), str(active)):
                    desired = active
                    reason = "resident-heavy-substitutes-fast-autotuned"
            elif desired == models.get("heavy_code") and active == models.get("fast_code"):
                margin = int(cfg.get("resident_fast_margin", 1))
                threshold = int(cfg.get("heavy_min_score", 3))
                # Do not override an explicit heavy request. Only collapse borderline auto-routing.
                borderline = complexity_override == "auto" and route.get("complexity_score", 0) < threshold + margin
                if borderline:
                    tuner_ready = bool(self.tuner is not None and self.tuner.ready(str(desired), str(active)))
                    if tuner_ready and self.tuner.prefer_resident(str(desired), str(active)):
                        desired = active
                        reason = "resident-fast-borderline-heavy-autotuned"
        elif (
            route["task_type"] == "reasoning"
            and cfg.get("reasoning_resident_fallback", False)
            and task_type_override == "auto"
            and active == models.get("heavy_code")
        ):
            desired = active
            reason = "resident-heavy-reasoning-fallback"

        if desired != route["model"]:
            route = dict(route)
            route["original_model"] = route["model"]
            route["model"] = desired
            route["resident_optimization"] = reason
        return route

    def _fallback_models(self, requested: str) -> list[str]:
        cfg = self.config.get("resilience", {})
        if not bool(cfg.get("model_fallback_enabled", True)):
            return []
        models = self.config.get("models", {})
        mapping = {
            models.get("heavy_code"): [models.get("fast_code"), models.get("general")],
            models.get("reasoning"): [models.get("heavy_code"), models.get("fast_code")],
            models.get("fast_code"): [models.get("general")],
            models.get("general"): [models.get("fast_code")],
        }
        out: list[str] = []
        for candidate in mapping.get(requested, []):
            if candidate and candidate != requested and candidate not in out:
                out.append(str(candidate))
        return out[: int(cfg.get("max_model_fallbacks", 2))]

    def _semantic_scope_lock(self, scope: str) -> threading.Lock:
        with self._semantic_lock_guard:
            lock = self._semantic_scope_locks.get(scope)
            if lock is None:
                lock = threading.Lock()
                self._semantic_scope_locks[scope] = lock
                # Bound lock registry; correctness does not depend on preserving old unlocked keys.
                if len(self._semantic_scope_locks) > 2048:
                    for key in list(self._semantic_scope_locks)[:512]:
                        candidate = self._semantic_scope_locks.get(key)
                        if candidate is not None and not candidate.locked():
                            self._semantic_scope_locks.pop(key, None)
            return lock

    def _generate(
        self,
        model: str,
        prompt: str,
        system: str,
        max_tokens: int,
        temperature: float,
        tenant: str,
        source: str,
        priority: int,
        *,
        measured_cloud_context_tokens: int = 0,
        semantic_query: str = "",
        semantic_context_fingerprint: str = "",
        internal: bool = False,
        use_cache: bool = True,
        format: Any = None,
    ) -> dict[str, Any]:
        saving = self.config.get("token_saving", {})
        resilience = self.config.get("resilience", {})
        max_tokens = max(64, min(int(max_tokens), int(saving.get("max_local_output_tokens", 8192))))
        role = source.rsplit(":", 1)[-1].lower()
        if source == "second-opinion":
            role = "second-opinion"
        vram_free = None
        if getattr(self, "vram_balancer", None):
            try:
                vram_free = self.vram_balancer.status().get("vram_available_mb")
            except Exception:
                pass
        initial_profile = self.model_policy.profile(
            model, role=role, input_tokens=estimate_tokens(prompt), output_tokens=max_tokens,
            background=source.startswith("preprocess:"),
            vram_free_mb=vram_free,
        )
        prompt_budget = min(int(saving.get("max_local_input_tokens", 56000)), initial_profile.prompt_budget_tokens)
        prepared = fit_text(prompt, prompt_budget)
        base_payload, execution_profile = self.model_policy.apply_payload(
            model,
            {"model": model, "prompt": prepared.text, "system": system, "stream": False,
             "keep_alive": self.config.get("ollama", {}).get("keep_alive", "-1"),
             "options": {"num_predict": max_tokens, "temperature": float(temperature)}},
            role=role, input_tokens=prepared.estimated_tokens, output_tokens=max_tokens,
            background=source.startswith("preprocess:"), preserve_explicit_think=False,
            vram_free_mb=vram_free,
        )
        if format:
            base_payload["format"] = format
        options = dict(base_payload.get("options", {}))
        if not semantic_query and prompt:
            clean_first = re.sub(r"^(TASK|QUESTION|INSTRUCTION|PROMPT|Problem|PROBLEM):\s*", "", prompt.strip(), flags=re.I)
            first_line = clean_first.split("\n\n", 1)[0].splitlines()[0] if clean_first else ""
            if len(first_line.strip()) >= 8:
                semantic_query = normalize_query(first_line.strip())
        elif semantic_query:
            semantic_query = normalize_query(semantic_query)

        execution_scope = execution_profile.cache_scope()
        norm_context_fp = stable_hash(normalize_context_for_hash(semantic_context_fingerprint)) if semantic_context_fingerprint else "no-context"
        cache_key = generation_cache_key(
            model=model,
            prompt=prepared.text,
            system=system,
            options=options,
            think=base_payload.get("think"),
            execution=execution_scope,
            format=base_payload.get("format"),
        )
        semantic_scope = stable_hash({
            "model": model, "system": system, "options": options, "think": base_payload.get("think"), "execution": execution_scope,
            "context": norm_context_fp, "source": source.split(":", 1)[0], "app_version": __version__,
        })
        started = time.perf_counter()
        cache_layer = "miss"
        semantic_score = 0.0
        requested_model = model

        def compute_inner() -> dict[str, Any]:
            nonlocal semantic_score
            if use_cache and semantic_query:
                semantic, semantic_score = self.semantic_cache.get(semantic_scope, semantic_query)
                if isinstance(semantic, dict):
                    reused = copy.deepcopy(semantic)
                    reused["_lah_cache_origin"] = "semantic"
                    reused["_lah_semantic_similarity"] = semantic_score
                    return reused

            availability = getattr(self.runtime, "is_online", None)
            if callable(availability):
                try:
                    backend_online = bool(availability())
                except Exception:
                    backend_online = True
                if not backend_online:
                    return {
                        "success": False,
                        "unsupported": True,
                        "degraded": True,
                        "terminal": False,
                        "retryable": True,
                        "error_code": "local_backend_unavailable",
                        "error": "No local model backend is currently available; retry after the backend is online.",
                        "model": requested_model,
                    }

            candidates = [requested_model] + self._fallback_models(requested_model)
            errors: list[str] = []
            retryable_error = False
            timeout_error = False
            for index, candidate in enumerate(candidates):
                breaker_key = f"model:{candidate}"
                if not self.breakers.allow(breaker_key):
                    errors.append(f"{candidate}: circuit open")
                    self.telemetry.record_stage(tenant=tenant, action=source, stage="model_attempt", model=candidate, success=False, degraded=True, error_type="circuit_open")
                    continue

                def run(candidate_model: str = candidate) -> dict[str, Any]:
                    # A smart->fast resilience fallback may have a smaller context window
                    # than the originally requested model. Re-fit locally instead of
                    # relying on implicit server truncation or turning an outage into OOM.
                    candidate_hint = self.model_policy.profile(
                        candidate_model, role=role, input_tokens=prepared.estimated_tokens, output_tokens=max_tokens,
                        background=source.startswith("preprocess:"),
                    )
                    candidate_prepared = fit_text(prepared.text, min(int(saving.get("max_local_input_tokens", 56000)), candidate_hint.prompt_budget_tokens))
                    payload, candidate_profile = self.model_policy.apply_payload(
                        candidate_model,
                        {"model": candidate_model, "prompt": candidate_prepared.text, "system": system, "stream": False,
                         "keep_alive": self.config.get("ollama", {}).get("keep_alive", "-1"),
                         "options": {"num_predict": max_tokens, "temperature": float(temperature)}},
                        role=role, input_tokens=candidate_prepared.estimated_tokens, output_tokens=max_tokens,
                        background=source.startswith("preprocess:"), preserve_explicit_think=False,
                    )
                    trace_observer = observer()
                    if trace_observer is not None:
                        trace_observer.model_request(payload)
                        response = self.runtime.request_stream(
                            "/api/generate", payload, trace_observer.output_delta,
                            # Keep mixed-version MCP workers alive while the
                            # trace observer API rolls forward independently.
                            on_thinking=getattr(trace_observer, "thinking_delta", None),
                        )
                    else:
                        response = self.runtime.request("/api/generate", payload)
                    if response.get("_lah_repetition_loop_detected"):
                        return {
                            "success": False,
                            "error": str(response.get("error") or "model output repetition loop detected"),
                            "model": candidate_model,
                            "terminal": True,
                            "retryable": False,
                            "repetition_loop_detected": True,
                            "_lah_repetition_loop_detected": True,
                            "_lah_retry_count": int(response.get("_lah_retry_count", 0) or 0),
                        }
                    if "error" in response:
                        retryable = is_retryable_local_backend_error(response["error"])
                        return {
                            "success": False, "error": response["error"], "model": candidate_model,
                            "retryable": retryable, "terminal": not retryable,
                            "error_code": "local_backend_transport" if retryable else "local_model_error",
                            "_lah_retry_count": int(response.get("_lah_retry_count", 0) or 0),
                        }
                    raw_text = response.get("response", "")
                    clean_text, parsed_thinking = postprocess_model_output(raw_text, role=role)
                    thinking = response.get("thinking") or parsed_thinking
                    return {
                        "success": True, "model": candidate_model, "requested_model": requested_model,
                        "text": clean_text, "thinking": thinking,
                        "final_response_produced": bool(clean_text.strip()),
                        "final_response_status": "produced" if clean_text.strip() else "empty",
                        "load_duration_ns": response.get("load_duration", 0), "eval_count": response.get("eval_count", 0),
                        "prompt_eval_count": response.get("prompt_eval_count", 0),
                        "prompt_eval_duration_ns": response.get("prompt_eval_duration", 0),
                        "eval_duration_ns": response.get("eval_duration", 0), "total_duration_ns": response.get("total_duration", 0),
                        "_lah_retry_count": int(response.get("_lah_retry_count", 0) or 0),
                        "execution_profile": candidate_profile.cache_scope(),
                        "_lah_cache_origin": "ollama", "fallback_used": candidate_model != requested_model,
                        "_cacheable": candidate_model == requested_model and bool(clean_text.strip()),
                    }

                try:
                    result = self.scheduler.submit(
                        candidate, tenant, source if index == 0 else f"{source}:fallback", run, priority=priority,
                        wait_timeout=float(resilience.get("scheduler_wait_timeout_seconds", self.config.get("server", {}).get("request_timeout_seconds", 300) + 30)),
                    )
                except TimeoutError as exc:
                    timeout_error = True
                    retryable_error = True
                    result = {
                        "success": False,
                        "error": str(exc),
                        "model": candidate,
                        "terminal": False,
                        "retryable": True,
                        "error_code": str(getattr(exc, "error_code", "model_request_timeout")),
                    }
                    job_id = getattr(exc, "job_id", None)
                    if job_id is not None:
                        result["scheduler_job_id"] = int(job_id)
                    state = getattr(exc, "state", "")
                    if state:
                        result["scheduler_state"] = str(state)
                except Exception as exc:
                    retryable = is_retryable_local_backend_error(exc)
                    result = {
                        "success": False, "error": str(exc), "model": candidate,
                        "retryable": retryable, "terminal": not retryable,
                        "error_code": "local_backend_transport" if retryable else "local_model_error",
                    }
                retryable_error = retryable_error or bool(result.get("retryable", False))
                if result.get("success"):
                    self.breakers.success(breaker_key)
                    if candidate != requested_model:
                        self.fallback_count += 1
                    return result
                self.breakers.failure(breaker_key)
                attempt_error = str(result.get("error", "failed"))
                attempt_retries = int(result.get("_lah_retry_count", 0) or 0)
                self.telemetry.record_stage(tenant=tenant, action=source, stage="model_attempt", model=candidate, success=False, degraded=index > 0, retry_count=attempt_retries, error_type="model_attempt_failed")
                self.telemetry.record_error("ollama", f"{source}:model_attempt", attempt_error, tenant=tenant, retryable=True)
                errors.append(f"{candidate}: {attempt_error}")

            # Same exact prompt may be served stale only as a resilience fallback.
            if use_cache and bool(resilience.get("serve_stale_on_error", True)):
                stale = self.stale_generation_cache.get(cache_key)
                if isinstance(stale, dict):
                    reused = copy.deepcopy(stale)
                    reused["_lah_cache_origin"] = "stale"
                    reused["stale_fallback"] = True
                    reused["runtime_errors"] = errors[-3:]
                    return reused
            failure = {
                "success": False,
                "error": "; ".join(errors) or "all local model attempts failed",
                "model": requested_model,
            }
            if retryable_error:
                failure.update({"terminal": False, "retryable": True})
            if timeout_error:
                failure["error_code"] = "model_request_timeout"
            return failure

        def compute() -> dict[str, Any]:
            nonlocal semantic_score
            if not semantic_query:
                return compute_inner()
            # Different phrasings with the same strict semantic scope serialize only on a cache miss.
            # The second request rechecks semantic cache after the first completes, coalescing paraphrases.
            with self._semantic_scope_lock(semantic_scope):
                semantic, semantic_score = self.semantic_cache.get(semantic_scope, semantic_query)
                if isinstance(semantic, dict):
                    reused = copy.deepcopy(semantic)
                    reused["_lah_cache_origin"] = "semantic"
                    reused["_lah_semantic_similarity"] = semantic_score
                    return reused
                result = compute_inner()
                if isinstance(result, dict) and result.get("success", "error" not in result) and not result.get("fallback_used", False):
                    clean = {k: v for k, v in result.items() if not str(k).startswith("_lah_")}
                    try:
                        self.semantic_cache.set(semantic_scope, semantic_query, clean)
                    except Exception:
                        pass
                return result

        if use_cache:
            raw, cache_hit, coalesced = self.generation_cache.get_or_compute(cache_key, compute)
        else:
            raw, cache_hit, coalesced = compute_inner(), False, False
        origin = str(raw.get("_lah_cache_origin", "")) if isinstance(raw, dict) else ""
        if not use_cache:
            cache_layer = "disabled"
        elif coalesced:
            cache_layer = "single-flight"
        elif cache_hit:
            cache_layer = "exact"
        elif origin == "semantic":
            cache_layer = "semantic"
            semantic_score = float(raw.get("_lah_semantic_similarity", semantic_score) or 0.0)
        elif origin == "stale":
            cache_layer = "stale-on-error"
        else:
            cache_layer = "ollama"
            if semantic_query and isinstance(raw, dict) and raw.get("success", "error" not in raw):
                clean = {k: v for k, v in raw.items() if not str(k).startswith("_lah_")}
                try:
                    self.semantic_cache.set(semantic_scope, semantic_query, clean)
                except Exception:
                    pass

        if use_cache and isinstance(raw, dict) and raw.get("success", "error" not in raw) and origin != "stale" and not raw.get("fallback_used", False):
            clean_stale = {k: v for k, v in raw.items() if not str(k).startswith("_lah_")}
            try:
                self.stale_generation_cache.set(cache_key, clean_stale)
            except Exception:
                pass

        result = copy.deepcopy(raw)
        queue_wait_ms = float(result.get("_lah_scheduler_queue_wait_ms", 0) or 0) if cache_layer == "ollama" else 0.0
        service_ms = float(result.get("_lah_scheduler_service_ms", 0) or 0) if cache_layer == "ollama" else 0.0
        retry_count = int(result.get("_lah_retry_count", 0) or 0) if cache_layer == "ollama" else 0
        for internal_key in ["_lah_cache_origin", "_lah_semantic_similarity", "_lah_scheduler_queue_wait_ms", "_lah_scheduler_service_ms", "_lah_scheduler_background", "_lah_retry_count"]:
            result.pop(internal_key, None)
        effective_cache_hit = cache_hit or cache_layer in {"semantic", "single-flight", "stale-on-error"}
        self.telemetry.record_stage(
            tenant=tenant,
            action=source,
            stage="cache_decision",
            success=True,
            error_type="cache_disabled" if not use_cache else cache_decision_reason(cache_layer, semantic_query=semantic_query),
        )
        result["cache_hit"] = effective_cache_hit
        result["cache_layer"] = cache_layer
        result["semantic_similarity"] = round(float(semantic_score), 5) if cache_layer == "semantic" else None
        result["coalesced"] = coalesced
        result["prompt_budget"] = {
            "estimated_tokens": prepared.estimated_tokens, "original_estimated_tokens": prepared.original_tokens,
            "truncated": prepared.truncated,
        }

        full_text = str(result.get("text", ""))
        full_output_tokens = estimate_tokens(full_text)
        provider_usage = result.get("usage") if isinstance(result.get("usage"), dict) else {}
        try:
            cache_read_tokens = max(0, int(
                result.get("cache_read_tokens", provider_usage.get("cache_read_tokens", 0)) or 0
            ))
        except (TypeError, ValueError, OverflowError):
            cache_read_tokens = 0
        if (not internal) and bool(saving.get("compact_responses", True)):
            result = self.artifacts.compact(result, tenant, source)
        result["latency"] = {
            "queue_wait_ms": round(max(0.0, queue_wait_ms), 1),
            "service_ms": round(max(0.0, service_ms), 1),
        }
        inline_tokens = estimate_tokens(str(result.get("text", "")))
        compact_saved = max(0, full_output_tokens - inline_tokens)
        cached_compute_tokens = (prepared.estimated_tokens + full_output_tokens) if cache_layer in {"exact", "semantic", "single-flight", "stale-on-error"} else 0
        result["token_saving"] = {
            "compact_output_tokens_avoided_est": compact_saved,
            "delegated_cloud_context_tokens_avoided_est": max(0, int(measured_cloud_context_tokens)),
            # Separate from cloud-context savings: these are tokens the local model
            # did not need to process/generate because a cache/single-flight result
            # was reused. Keeping the metric separate prevents double counting.
            "local_compute_tokens_avoided_est": max(0, int(cached_compute_tokens)),
        }

        if self.tuner is not None and cache_layer == "ollama":
            try:
                self.tuner.observe(str(result.get("model", model)), result, (time.perf_counter() - started) * 1000)
            except Exception:
                pass

        success = bool(result.get("success", "error" not in result))
        error_type = "" if success else "model_request_failed"
        self.telemetry.record(
            tenant=tenant, action=source, model=str(result.get("model", model)), cache_hit=effective_cache_hit, coalesced=coalesced,
            cache_layer=cache_layer, input_tokens=prepared.estimated_tokens,
            cache_read_tokens=cache_read_tokens, output_tokens=full_output_tokens,
            task_type=source.split(":", 1)[1] if source.startswith("delegate:") else source.split(":", 1)[0],
            avoided_cloud_tokens=max(0, int(measured_cloud_context_tokens)) + compact_saved,
            duration_ms=(time.perf_counter() - started) * 1000, queue_wait_ms=queue_wait_ms, service_ms=service_ms,
            load_duration_ms=float(result.get("load_duration_ns", 0) or 0) / 1_000_000,
            success=success, fallback_used=bool(result.get("fallback_used", False)), retry_count=retry_count,
            degraded=bool(result.get("stale_fallback", False) or result.get("fallback_used", False)), error_type=error_type,
        )
        if not success:
            self.telemetry.record_error("ollama", source, result.get("error", "model request failed"), tenant=tenant, retryable=True)
        return result

    @staticmethod
    def _conversation_user_prompt(task: str, context: str = "") -> str:
        prompt = f"TASK:\n{task}\n"
        if context:
            prompt += f"\nCONTEXT:\n{context}\n"
        return prompt

    @staticmethod
    def _conversation_transcript(messages: list[dict[str, str]], next_user_prompt: str) -> str:
        rendered = []
        for message in messages:
            role = "USER" if message.get("role") == "user" else "ASSISTANT"
            rendered.append(f"{role}:\n{message.get('content', '')}")
        rendered.append(f"USER:\n{next_user_prompt}")
        return "CONVERSATION:\n\n" + "\n\n".join(rendered)

    def _conversation_prompt_limit(self) -> int:
        cfg = self.config.get("model_conversations", {})
        return max(256, int(cfg.get("max_prompt_tokens", self.config.get("token_saving", {}).get("max_local_input_tokens", 56000))))

    def _conversation_error(self, conversation_id: str, error: str) -> dict[str, Any]:
        active = error == "conversation is already running"
        return {
            "success": False,
            "conversation_id": conversation_id,
            "error": error,
            "terminal": not active,
            "retryable": active,
            "in_progress": active,
        }

    def _start_conversation(
        self,
        *,
        tenant: str,
        prompt: str,
        route: dict[str, Any],
        system: str,
        max_tokens: int,
        temperature: float,
        source: str,
        priority: int,
    ) -> dict[str, Any]:
        if estimate_tokens(prompt) + estimate_tokens(system) > self._conversation_prompt_limit():
            return self._conversation_error("", "conversation prompt limit exceeded")
        conversation = self.conversations.start(tenant, {
            "model": route["model"], "system": system, "max_tokens": max_tokens,
            "temperature": temperature, "source": source, "priority": priority,
            "route": dict(route),
        })
        self.conversations.reserve(conversation.conversation_id, tenant)
        try:
            result = self._generate(
                route["model"], prompt, system, max_tokens, temperature, tenant, source, priority,
                internal=True, use_cache=False,
            )
        except Exception:
            self.conversations.discard(conversation.conversation_id, tenant)
            raise
        if not result.get("success", "error" not in result):
            self.conversations.discard(conversation.conversation_id, tenant)
            return result
        self.conversations.complete(conversation.conversation_id, tenant, prompt, str(result.get("text", "")))
        result["conversation_id"] = conversation.conversation_id
        result["route"] = dict(route)
        return result

    def continue_conversation(self, args: dict[str, Any], tenant: str) -> dict[str, Any]:
        conversation_id = str(args.get("conversation_id", "")).strip()
        task = str(args.get("task", "")).strip()
        if not conversation_id:
            return self._conversation_error("", "conversation_id is required")
        if not task:
            return self._conversation_error(conversation_id, "task is required")
        conversation, error = self.conversations.reserve(conversation_id, tenant)
        if error:
            return self._conversation_error(conversation_id, error)
        assert conversation is not None
        user_prompt = self._conversation_user_prompt(task, str(args.get("context", "")))
        prompt = self._conversation_transcript(conversation.messages, user_prompt)
        settings = conversation.settings
        if estimate_tokens(prompt) + estimate_tokens(str(settings["system"])) > self._conversation_prompt_limit():
            self.conversations.abort(conversation_id, tenant)
            return self._conversation_error(conversation_id, "conversation prompt limit exceeded")
        try:
            result = self._generate(
                str(settings["model"]), prompt, str(settings["system"]), int(settings["max_tokens"]),
                float(settings["temperature"]), tenant, str(settings["source"]), int(settings["priority"]),
                internal=True, use_cache=False,
            )
        except Exception:
            self.conversations.abort(conversation_id, tenant)
            raise
        if not result.get("success", "error" not in result):
            self.conversations.abort(conversation_id, tenant)
            return result
        self.conversations.complete(conversation_id, tenant, user_prompt, str(result.get("text", "")))
        result["conversation_id"] = conversation_id
        result["route"] = dict(settings["route"])
        return result

    def delegate(self, args: dict[str, Any], tenant: str) -> dict[str, Any]:
        if bool(args.get("conversation", False)):
            if str(args.get("profile", "")).strip():
                return self._conversation_error("", "conversations do not support profiles")
            if str(args.get("delivery", "sync")).strip().lower() != "sync":
                return self._conversation_error("", "conversations require delivery=sync")
        if str(args.get("profile", "")).strip():
            return self.delegate_profile(args, tenant)
        task = str(args.get("task") or args.get("prompt") or "")
        context = str(args.get("context", ""))
        task_type_override = str(args.get("task_type", "auto"))
        complexity_override = str(args.get("complexity", "auto"))
        route = self.router.classify(task, context, task_type_override, complexity_override)
        route = self._resident_optimize(route, task_type_override, complexity_override)
        try:
            route = self.router.apply_model_override(route, str(args.get("model", "")))
        except ValueError as exc:
            return {"success": False, "error": str(exc), "terminal": True, "retryable": False}
        task_type = route["task_type"]
        operation = str(args.get("operation") or task_type).strip().lower()
        package = build_prompt(
            operation=operation,
            model=route["model"],
            role=task_type,
            profile=str(getattr(self, "config", {}).get("_hardware", {}).get("profile", "auto")),
            task=task,
            context=context,
            changed_paths=args.get("changed_paths") or (),
            evidence_ids=args.get("evidence_ids") or (),
            acceptance_criteria=args.get("acceptance_criteria") or (),
            repository_revision=str(args.get("repository_revision", "")),
            static_facts=str(args.get("static_facts", "")),
        )
        system = package.system
        # Conversation turns keep the compact transcript protocol so follow-ups
        # remain lossless; the richer static contract lives in the conversation
        # system message and one-shot operations use the full package.user form.
        prompt = self._conversation_user_prompt(task, context) if bool(args.get("conversation", False)) else package.user
        max_tokens = int(args.get("max_tokens", 4096))
        temperature = float(args.get("temperature", 0.15))
        source = f"delegate:{task_type}"
        priority = int(args.get("priority", 5))
        if bool(args.get("conversation", False)):
            return self._start_conversation(
                tenant=tenant, prompt=prompt, route=route, system=system, max_tokens=max_tokens,
                temperature=temperature, source=source, priority=priority,
            )
        format_val = args.get("format") or args.get("json_schema")
        result = self._generate(
            route["model"], prompt, system,
            max_tokens, temperature, tenant, source, priority,
            semantic_query=task, semantic_context_fingerprint=stable_hash(context),
            format=format_val,
        )
        result = self._apply_semantic_quality(result, task=task, evidence_paths=args.get("changed_paths") or (), context=context)
        result["route"] = route
        return result

    @staticmethod
    def _apply_semantic_quality(result: dict[str, Any], *, task: str, evidence_paths: Any, context: str = "") -> dict[str, Any]:
        """Reject obvious ungrounded model paths while keeping output advisory."""
        if not isinstance(result, dict) or not result.get("success", "error" not in result):
            return result
        paths = evidence_paths if isinstance(evidence_paths, (list, tuple, set)) else ()
        quality = assess_semantic_result(task, [str(path) for path in paths], result, context=context)
        result["advisory_only"] = True
        result["semantic_quality"] = quality
        if not quality.get("usable", False):
            reason = quality.get("bypass_reason", quality.get("reason", "quality_gate"))
            result["quality_warning"] = f"advisory semantic output requires verification: {reason}"
            result["bypass_reason"] = reason
        else:
            result.pop("quality_warning", None)
            result.pop("bypass_reason", None)
        return result

    def delegate_profile(self, args: dict[str, Any], tenant: str) -> dict[str, Any]:
        profile_name = str(args.get("profile", "")).strip()
        if not self.profile_catalog.enabled:
            return {"success": False, "unsupported": True, "error": "Ollama subagent profiles disabled"}
        try:
            available = set(str(x) for x in self.runtime.installed_models())
        except Exception:
            available = None
        try:
            profile = self.profile_catalog.resolve(profile_name, available_models=available)
        except ValueError as exc:
            return {"success": False, "unsupported": True, "error": str(exc)}

        root = str(args.get("root", "")).strip()
        task = str(args.get("task", ""))
        if root:
            if self.tool_agent is None:
                return {"success": False, "unsupported": True, "error": "read-only Ollama tool agent unavailable"}
            return self.tool_agent.run_profile(
                profile.name,
                task,
                root,
                tenant,
                workspace=args.get("workspace"),
                seed_context=str(args.get("context", "")),
                priority=int(args.get("priority", 5)),
            )

        context = str(args.get("context", ""))
        candidate = str(args.get("candidate", ""))
        package = build_prompt(
            operation=str(args.get("operation") or profile.role or "delegate"),
            model=profile.model,
            profile=str(getattr(self, "config", {}).get("_hardware", {}).get("profile", "auto")),
            task=task,
            context=context + (f"\nCANDIDATE:\n{candidate}" if candidate else ""),
        )
        result = self._generate(
            profile.model,
            package.user,
            self.profile_catalog.system_contract(profile, task, prompt=package.system),
            int(args.get("max_tokens", 0) or profile.max_tokens),
            profile.temperature,
            tenant,
            f"profile:{profile.name}",
            int(args.get("priority", 5)),
            semantic_query=task,
            semantic_context_fingerprint=stable_hash(context + candidate),
        )
        result.update({
            "profile": profile.name,
            "language": self.profile_catalog.detect_language(task),
            "advisory_only": profile.advisory_only,
            "declared_model": profile.declared_model,
            "resolved_model": profile.model,
            "model_fallback": profile.model_fallback,
            "model_fallback_reason": profile.model_fallback_reason,
            "tools_used": [],
        })
        return result

    def review(self, args: dict[str, Any], tenant: str) -> dict[str, Any]:
        payload = dict(args)
        payload["task_type"] = "review"
        payload["task"] = str(args.get("instructions", "Review the supplied code or diff and report actionable defects only."))
        payload["context"] = str(args.get("code", args.get("context", "")))
        payload["operation"] = str(args.get("operation", "review"))
        payload.setdefault("max_tokens", 4096)
        return self.delegate(payload, tenant)

    def reason(self, args: dict[str, Any], tenant: str) -> dict[str, Any]:
        payload = dict(args)
        payload["task_type"] = "reasoning"
        payload["task"] = next((str(args.get(key) or "") for key in ("problem", "task", "prompt")
                                if str(args.get(key) or "").strip()), "")
        if not payload["task"]:
            return {"success": False, "terminal": True, "retryable": False,
                    "error_code": "empty_reason_input", "error": "reason requires non-empty problem, task or prompt"}
        payload["operation"] = str(args.get("operation", "reason"))
        payload.setdefault("max_tokens", 4096)
        return self.delegate(payload, tenant)

    def distill_command_error(self, failure: dict[str, Any]) -> str:
        """Use the local task route only when command parsing found no coordinates."""
        summary = str(failure.get("summary", "")).strip()[:4000]
        if not summary:
            return ""
        result = self.reason({
            "problem": "Summarize this unstructured command failure in one actionable sentence. Do not invent a file, line, test, or fix.",
            "context": summary,
            "max_tokens": 120,
            "complexity": "simple",
        }, "command-error-distill")
        if not result.get("success", "error" not in result):
            return ""
        return str(result.get("text", "")).strip()[:300]

    def second_opinion(self, args: dict[str, Any], tenant: str) -> dict[str, Any]:
        question = str(args.get("question", ""))
        candidate = str(args.get("candidate", ""))
        context = str(args.get("context", ""))
        focus = str(args.get("focus", "correctness, missing assumptions, edge cases, and alternative explanations"))
        route = self.router.classify(
            f"{question}\n{focus}",
            f"{candidate}\n{context}",
            "reasoning",
            str(args.get("complexity", "auto")),
        )
        model = str(route["model"])
        package = build_prompt(
            operation="second_opinion",
            model=model,
            role="reasoning",
            profile=str(getattr(self, "config", {}).get("_hardware", {}).get("profile", "auto")),
            task=question,
            context=f"CANDIDATE ANSWER / PLAN:\n{candidate}\n\nADDITIONAL CONTEXT:\n{context}",
            changed_paths=args.get("changed_paths") or (),
            static_facts=f"FOCUS: {focus}",
        )
        result = self._generate(
            model, package.user, package.system,
            int(args.get("max_tokens", 4096)), float(args.get("temperature", 0.2)),
            tenant, "second-opinion", int(args.get("priority", 6)),
            semantic_query=f"{question}\n{focus}", semantic_context_fingerprint=stable_hash({"candidate": candidate, "context": context}),
        )
        result = self._apply_semantic_quality(result, task=question, evidence_paths=args.get("changed_paths") or (), context=context)
        result["route"] = route
        return result

    def speculative_draft(self, args: dict[str, Any], tenant: str) -> dict[str, Any]:
        """Draft code using fast model and verify against affected tests."""
        task = str(args.get("task", args.get("prompt", "")))
        file_path = str(args.get("file", args.get("path", "")))
        context = str(args.get("context", ""))
        root = str(args.get("root", args.get("cwd", ".")))
        resolved_root = Path(root).resolve()

        if not task:
            return {"success": False, "error": "task or prompt is required"}

        fast_model = str(self.config.get("models", {}).get("fast_code", "qwen2.5-coder:1.5b"))
        package = build_prompt(
            operation="speculative_draft",
            model=fast_model,
            role="code",
            profile=str(getattr(self, "config", {}).get("_hardware", {}).get("profile", "auto")),
            task=task,
            context=f"FILE: {file_path}\n\n{context}",
            changed_paths=[file_path] if file_path else (),
        )

        gen_result = self._generate(
            fast_model,
            package.user,
            package.system + " Return the minimal valid code or diff only after grounding it in the supplied file and context.",
            int(args.get("max_tokens", 2048)),
            float(args.get("temperature", 0.1)),
            tenant,
            "speculative-draft",
            priority=5,
        )

        draft_code = gen_result.get("text") or gen_result.get("response") or ""
        verification: dict[str, Any] = {"syntax_valid": True}
        if file_path.endswith(".py") and draft_code:
            import ast
            try:
                clean_code = re.sub(r"^```[\w]*\n", "", draft_code.strip())
                clean_code = re.sub(r"\n```$", "", clean_code)
                ast.parse(clean_code)
                verification["syntax_valid"] = True
            except Exception as e:
                verification["syntax_valid"] = False
                verification["syntax_error"] = str(e)

        if bool(args.get("verify_tests", False)) and file_path and self.commands:
            aff = self.affected_tests(str(resolved_root), changed_paths=[file_path])
            if aff.get("suggested_command"):
                cmd_res = self.commands.run(aff["suggested_command"], str(resolved_root), tenant)
                verification["tests_passed"] = bool(cmd_res.get("success"))
                verification["test_summary"] = cmd_res.get("summary", "")

        if bool(args.get("smart_review", False)):
            smart_model = str(self.config.get("models", {}).get("heavy_code", "qwen2.5-coder:3b"))
            review_prompt = f"REVIEW DRAFT IMPLEMENTATION:\nTask: {task}\nDraft Code:\n{draft_code}\nDoes this draft correctly solve the task without syntax or logical bugs? Return a short JSON object: {{\"approved\": true/false, \"confidence\": 0.0-1.0, \"summary\": \"...\"}}"
            review_res = self._generate(
                smart_model,
                review_prompt,
                "You are an expert code reviewer. Return only valid JSON.",
                512,
                0.1,
                tenant,
                "speculative-review",
                priority=6,
            )
            review_text = review_res.get("text") or review_res.get("response") or ""
            verification["smart_review"] = {
                "model": smart_model,
                "response": review_text,
                "approved": "true" in review_text.lower(),
            }

        return {
            "success": True,
            "model": fast_model,
            "draft": draft_code,
            "verification": verification,
            "duration_ms": gen_result.get("duration_ms", 0),
        }

    def speculative_lint(self, args: dict[str, Any], tenant: str) -> dict[str, Any]:
        """Run opt-in, read-only lint against only caller-supplied changed paths."""
        root = str(args.get("root", args.get("cwd", ".")))
        try:
            paths = normalize_changed_paths(root, list(args.get("paths") or []))
        except ValueError as exc:
            return {"success": False, "error": str(exc), "terminal": True, "retryable": False}
        custom = str(args.get("command", "")).strip()
        if custom:
            target_text = " ".join(subprocess.list2cmdline([path]) for path in paths) if os.name == "nt" else " ".join(shlex.quote(path) for path in paths)
            command = custom.replace("{paths}", target_text)
            if command == custom:
                command = f"{custom} {target_text}"
        elif shutil.which("ruff"):
            target_text = " ".join(subprocess.list2cmdline([path]) for path in paths) if os.name == "nt" else " ".join(shlex.quote(path) for path in paths)
            command = f"ruff check {target_text}"
        elif shutil.which("eslint"):
            target_text = " ".join(subprocess.list2cmdline([path]) for path in paths) if os.name == "nt" else " ".join(shlex.quote(path) for path in paths)
            command = f"eslint {target_text}"
        else:
            return {"success": False, "error": "no read-only linter found (tried ruff, eslint)", "terminal": True, "retryable": False}
        lowered = command.lower()
        if any(marker in lowered for marker in ("--fix", " -w ", "autopep8", "cargo fix", "gofmt -w")):
            return {"success": False, "error": "speculative lint rejects formatter or auto-fix commands", "terminal": True, "retryable": False}
        classification = self.commands.classify(command) if self.commands is not None else {"allowed": False, "class": "unknown"}
        if not classification.get("allowed", False) or classification.get("class") in {"mutating", "dangerous"}:
            return {"success": False, "error": "speculative lint requires an allowed read-only command", "classification": classification, "terminal": True, "retryable": False}
        result = self.commands.run(
            command,
            str(Path(root).expanduser().resolve(strict=False)),
            tenant=tenant,
            timeout=int(args.get("timeout", self.config.get("speculative_lint", {}).get("timeout_seconds", 120)) or 120),
            force=True,
            auto_fix=False,
            bypass_cache=True,
        )
        result.update({"read_only": True, "auto_fix": False, "paths": paths, "command": command})
        return result

    def task_scaffold(self, args: dict[str, Any], tenant: str) -> dict[str, Any]:
        """Generate boilerplate code, DTOs, interfaces, or unit test scaffolds using local model."""
        spec = str(args.get("spec", args.get("prompt", args.get("task", ""))))
        context = str(args.get("context", ""))
        language = str(args.get("language", "python"))
        max_tokens = int(args.get("max_tokens", 4096))

        if not spec:
            return {"success": False, "error": "spec or prompt is required"}

        fast_model = str(self.config.get("models", {}).get("fast_code", "qwen2.5-coder:7b"))
        prompt = (
            f"You are a fast code scaffolder. Generate clean, idiomatic {language} code implementing the following specification:\n\n"
            f"SPECIFICATION:\n{spec}\n\n"
        )
        if context:
            prompt += f"CONTEXT / EXISTING CODE:\n{context}\n\n"
        prompt += (
            "REQUIREMENTS:\n"
            "- Emit ONLY valid code with minimal necessary docstrings.\n"
            "- Implement all requested types, data structures, constructors, and method signatures.\n"
            "- Do not include markdown conversational preamble or apologies.\n"
        )
        gen_result = self._generate(
            fast_model,
            prompt,
            "You are a precise code scaffolder. Emit only valid source code.",
            max_tokens,
            float(args.get("temperature", 0.1)),
            tenant,
            "scaffold",
            priority=5,
        )
        draft_code = gen_result.get("text") or gen_result.get("response") or ""
        clean_code = re.sub(r"^```[\w]*\n", "", draft_code.strip())
        clean_code = re.sub(r"\n```$", "", clean_code)

        return {
            "success": True,
            "code": clean_code,
            "language": language,
            "model": gen_result.get("model", fast_model),
            "tokens_generated": gen_result.get("tokens", 0),
        }

    def vision(self, args: dict[str, Any], tenant: str) -> dict[str, Any]:
        """Multimodal image understanding via local vision model."""
        features = self.config.get("features", {})
        if isinstance(features, dict) and not bool(features.get("vision", True)):
            return {
                "success": False,
                "unsupported": True,
                "terminal": True,
                "retryable": False,
                "error_code": "vision_disabled",
                "error": "Vision capability is disabled. Enable features.vision explicitly.",
            }
        image_path = str(args.get("image", args.get("image_path", "")))
        prompt = str(args.get("prompt", args.get("task", "Describe this image in detail.")))
        models = self.config.get("models", {})
        configured_model = models.get("vision", "qwen3-vl:4b") if isinstance(models, dict) else "qwen3-vl:4b"
        model = str(args.get("model") or configured_model or "").strip()
        if bool(args.get("cloud_fallback", False)):
            vision_policy = self.config.get("vision", {})
            if not isinstance(vision_policy, dict) or not bool(vision_policy.get("cloud_fallback_enabled", False)):
                return {
                    "success": False,
                    "terminal": True,
                    "retryable": False,
                    "error_code": "vision_cloud_fallback_disabled",
                    "error": "Cloud vision fallback is disabled. Enable vision.cloud_fallback_enabled explicitly.",
                    "cloud_fallback": False,
                }
            provider = str(vision_policy.get("cloud_provider", "")).strip()
            if not provider:
                return {
                    "success": False,
                    "terminal": True,
                    "retryable": False,
                    "error_code": "vision_cloud_fallback_unavailable",
                    "error": "Cloud vision fallback is enabled but no provider is configured.",
                    "cloud_fallback": False,
                }
        if not model:
            return {
                "success": False,
                "unsupported": True,
                "degraded": True,
                "terminal": True,
                "retryable": False,
                "error": "Vision model is not configured; set models.vision to qwen3-vl:4b or another Ollama vision model.",
            }

        def input_error(message: str, code: str = "vision_input_error") -> dict[str, Any]:
            return {
                "success": False,
                "terminal": True,
                "retryable": False,
                "error": message,
                "error_code": code,
            }

        images: list[str] = []

        def append_image(value: str) -> dict[str, Any] | None:
            if len(value) > VISION_MAX_IMAGE_CHARS:
                return input_error(
                    "Vision image transport exceeds the bounded character limit.",
                    "vision_image_transport_too_large",
                )
            encoded = value
            if value.startswith("data:image/"):
                if "," not in value:
                    return input_error("Vision image data URL is invalid.", "vision_image_integrity")
                encoded = value.split(",", 1)[1]
            try:
                decoded = base64.b64decode(encoded, validate=True)
            except (ValueError, TypeError):
                return input_error("Vision image input is not valid base64.", "vision_image_integrity")
            if len(decoded) > VISION_MAX_IMAGE_BYTES:
                return input_error(
                    "Vision image exceeds the bounded decoded-byte limit.",
                    "vision_image_too_large",
                )
            # Ollama expects the raw base64 payload, not the transport prefix
            # from a browser-style data URL. Keep validation accepting both,
            # but normalize the provider payload to one canonical form.
            images.append(encoded)
            return None

        def is_inline_image(value: str) -> bool:
            if value.startswith("data:image/"):
                return True
            try:
                return bool(value) and bool(base64.b64decode(value, validate=True))
            except (ValueError, TypeError):
                return False

        if image_path:
            if (
                "image_path" not in args
                and (image_path.startswith("data:image/") or len(image_path) > VISION_MAX_IMAGE_CHARS or is_inline_image(image_path))
            ):
                error = append_image(image_path)
                if error:
                    return error
            else:
                try:
                    p = Path(image_path)
                    is_file = p.is_file()
                except (OSError, ValueError):
                    return input_error("Vision image path could not be inspected; provide a readable image or base64 data.")
                if not is_file:
                    return input_error("Vision image path was not found; provide a readable image or base64 data.")
                try:
                    size_bytes = int(p.stat().st_size)
                    encoded_chars = ((size_bytes + 2) // 3) * 4
                    if size_bytes > VISION_MAX_IMAGE_BYTES:
                        return input_error("Vision image exceeds the bounded decoded-byte limit.", "vision_image_too_large")
                    if encoded_chars > VISION_MAX_IMAGE_CHARS:
                        return input_error("Vision image transport exceeds the bounded character limit.", "vision_image_transport_too_large")
                    b64 = base64.b64encode(p.read_bytes()).decode("utf-8")
                    error = append_image(b64)
                    if error:
                        return error
                except (OSError, ValueError):
                    return input_error("Failed to read image file; verify the path and permissions.")

        image_artifact_id = str(
            args.get("image_artifact_id") or args.get("screenshot_artifact_id") or ""
        ).strip()
        bundle_artifact_id = str(
            args.get("bundle_artifact_id") or args.get("frontend_bundle_artifact_id") or ""
        ).strip()
        direct_context_refs = {
            "dom": str(args.get("dom_artifact_id") or args.get("html_artifact_id") or "").strip(),
            "accessibility": str(
                args.get("accessibility_artifact_id") or args.get("a11y_artifact_id") or ""
            ).strip(),
            "computed_styles": str(
                args.get("computed_styles_artifact_id") or args.get("computed_style_artifact_id") or ""
            ).strip(),
            "runtime": str(
                args.get("runtime_artifact_id") or args.get("runtime_context_artifact_id") or ""
            ).strip(),
        }
        network_artifact_id = str(args.get("network_artifact_id") or "").strip()

        def resolve_artifact(artifact_id: str, max_chars: int) -> tuple[str, dict[str, Any] | None]:
            try:
                artifact = self.artifacts.get(artifact_id, max_chars=max_chars, tenant=tenant)
            except Exception:
                return "", {
                    "success": False,
                    "terminal": False,
                    "retryable": True,
                    "error": "Vision artifact store is unavailable; retry the request.",
                }
            if not isinstance(artifact, dict) or not artifact.get("success"):
                detail = str(artifact.get("error", "")).lower() if isinstance(artifact, dict) else ""
                if any(marker in detail for marker in ("not found", "expired", "missing")):
                    return "", input_error("Vision artifact was not found or has expired.")
                return "", {
                    "success": False,
                    "terminal": False,
                    "retryable": True,
                    "error": "Vision artifact store returned an error; retry the request.",
                }
            text = artifact.get("text")
            if not isinstance(text, str) or not text:
                return "", input_error("Vision artifact does not contain usable content.")
            try:
                total_chars = int(artifact.get("total_chars", len(text)))
            except (TypeError, ValueError, OverflowError):
                return "", input_error("Vision artifact metadata is invalid.")
            if artifact.get("next_offset") is not None or total_chars > len(text):
                return "", input_error(
                    "Vision artifact is truncated; provide the complete artifact.",
                    "vision_artifact_truncated",
                )
            if total_chars > max_chars:
                return "", input_error(
                    "Vision artifact exceeds the bounded size limit.",
                    "vision_artifact_too_large",
                )
            return text[:max_chars], None

        def resolve_binary_artifact(artifact_id: str, max_bytes: int) -> tuple[str, dict[str, Any] | None]:
            try:
                artifact = self.artifacts.get_binary(artifact_id, tenant=tenant)
            except Exception:
                return "", {
                    "success": False,
                    "terminal": False,
                    "retryable": True,
                    "error": "Vision artifact store is unavailable; retry the request.",
                }
            if not isinstance(artifact, dict) or not artifact.get("success"):
                detail = str(artifact.get("error", "")).lower() if isinstance(artifact, dict) else ""
                if any(marker in detail for marker in ("not found", "expired", "missing")):
                    return "", input_error("Vision artifact was not found or has expired.")
                return "", input_error("Vision image artifact failed integrity validation.", "vision_artifact_integrity")
            mime_type = str(artifact.get("mime_type", ""))
            encoded = artifact.get("data_base64")
            try:
                size_bytes = int(artifact.get("size_bytes", -1))
            except (TypeError, ValueError, OverflowError):
                size_bytes = -1
            if not mime_type.startswith("image/") or not isinstance(encoded, str) or size_bytes < 0:
                return "", input_error("Vision image artifact exceeds the bounded integrity contract.", "vision_artifact_integrity")
            if len(encoded) > VISION_MAX_IMAGE_CHARS:
                return "", input_error(
                    "Vision image transport exceeds the bounded character limit.",
                    "vision_image_transport_too_large",
                )
            try:
                decoded = base64.b64decode(encoded, validate=True)
            except (ValueError, TypeError):
                return "", input_error("Vision image artifact is not valid base64.", "vision_artifact_integrity")
            if len(decoded) > max_bytes:
                return "", input_error(
                    "Vision image exceeds the bounded decoded-byte limit.",
                    "vision_image_too_large",
                )
            if len(decoded) != size_bytes:
                return "", input_error("Vision image artifact is truncated or incomplete.", "vision_artifact_truncated")
            return encoded, None

        def resolve_bundle_context(raw: str) -> tuple[str, dict[str, Any] | None]:
            nonlocal image_artifact_id
            if not raw.lstrip().startswith(("{", "[")):
                return raw, None
            try:
                bundle = parse_bounded_context(raw, "frontend_bundle")
            except FrontendReviewError as exc:
                if exc.code == "invalid_frontend_context":
                    return "", input_error(
                        "Vision frontend bundle is not valid JSON.",
                        "vision_bundle_invalid",
                    )
                return "", exc.as_result()
            resolved = dict(bundle)
            screenshot = resolved.get("screenshot")
            screenshot_ref = screenshot.get("artifact_id", "") if isinstance(screenshot, dict) else ""
            if screenshot_ref and not images:
                image_data, error = resolve_binary_artifact(str(screenshot_ref), VISION_MAX_IMAGE_BYTES)
                if error:
                    return "", error
                error = append_image(image_data)
                if error:
                    return "", error
                image_artifact_id = str(screenshot_ref)
            for field_name in ("dom", "accessibility", "computed_styles", "runtime"):
                value = resolved.get(field_name)
                if field_name == "runtime":
                    runtime_refs: dict[str, str] = {}
                    if isinstance(value, dict):
                        for ref_key in ("artifact_id", "console_artifact_id", "network_artifact_id"):
                            ref = str(value.get(ref_key, "")).strip()
                            if ref:
                                runtime_refs[ref_key] = ref
                    for ref_key in ("runtime_artifact_id", "network_artifact_id"):
                        ref = str(resolved.get(ref_key, "")).strip()
                        if ref:
                            runtime_refs.setdefault(
                                "console_artifact_id" if ref_key == "runtime_artifact_id" else ref_key,
                                ref,
                            )
                    if runtime_refs:
                        runtime_value = dict(value) if isinstance(value, dict) else {}
                        for ref_key, ref in runtime_refs.items():
                            content, error = resolve_artifact(ref, VISION_MAX_BUNDLE_CHARS)
                            if error:
                                return "", error
                            content_key = {
                                "artifact_id": "content",
                                "console_artifact_id": "console_content",
                                "network_artifact_id": "network_content",
                            }[ref_key]
                            runtime_value[content_key] = content
                        resolved[field_name] = runtime_value
                    continue
                reference = value.get("artifact_id", "") if isinstance(value, dict) else ""
                if not reference:
                    reference = resolved.get(f"{field_name}_artifact_id", "")
                if not reference:
                    continue
                content, error = resolve_artifact(str(reference), VISION_MAX_BUNDLE_CHARS)
                if error:
                    return "", error
                if isinstance(value, dict):
                    resolved[field_name] = {**value, "content": content}
                else:
                    resolved[field_name] = {"artifact_id": str(reference), "content": content}
            return json_dumps(resolved, ensure_ascii=False), None

        if image_artifact_id and not images:
            artifact_data, error = resolve_binary_artifact(image_artifact_id, VISION_MAX_IMAGE_BYTES)
            if error:
                return error
            error = append_image(artifact_data)
            if error:
                return error

        bundle_context = ""
        if bundle_artifact_id:
            bundle_context, error = resolve_artifact(bundle_artifact_id, VISION_MAX_BUNDLE_CHARS)
            if error:
                return error
            bundle_context, error = resolve_bundle_context(bundle_context)
            if error:
                return error

        def parse_context_value(field_name: str, value: Any) -> Any:
            return parse_bounded_context(value, field_name)

        frontend_bundle: dict[str, Any] = {}
        if bundle_context:
            if not bundle_context.lstrip().startswith(("{", "[")):
                frontend_bundle = {}
            else:
                try:
                    frontend_bundle = parse_bounded_context(bundle_context, "frontend_bundle")
                except FrontendReviewError as exc:
                    if exc.code == "invalid_frontend_context":
                        return input_error(
                            "Vision frontend bundle is not valid JSON.",
                            "vision_bundle_invalid",
                        )
                    return exc.as_result()
                frontend_bundle["bundle_artifact_id"] = bundle_artifact_id

        inline_bundle = args.get("bundle") or args.get("frontend_bundle") or args.get("dom_bundle")
        if inline_bundle is not None:
            try:
                decoded_inline = (
                    parse_context_value("bundle", inline_bundle)
                    if not isinstance(inline_bundle, dict)
                    else dict(inline_bundle)
                )
            except FrontendReviewError as exc:
                return exc.as_result()
            frontend_bundle.update(decoded_inline)

        for field_name, artifact_id in direct_context_refs.items():
            if not artifact_id:
                continue
            content, error = resolve_artifact(artifact_id, VISION_MAX_BUNDLE_CHARS)
            if error:
                return error
            try:
                context_value = parse_context_value(field_name, content)
            except FrontendReviewError as exc:
                return exc.as_result()
            ref_key = "console_artifact_id" if field_name == "runtime" else "artifact_id"
            frontend_bundle[field_name] = {ref_key: artifact_id, **context_value}
        if network_artifact_id:
            content, error = resolve_artifact(network_artifact_id, VISION_MAX_BUNDLE_CHARS)
            if error:
                return error
            runtime_value = frontend_bundle.get("runtime")
            runtime_value = dict(runtime_value) if isinstance(runtime_value, dict) else {}
            runtime_value.update(
                {"network_artifact_id": network_artifact_id, "network_content": content}
            )
            frontend_bundle["runtime"] = runtime_value

        inline_aliases = {
            "dom": ("dom", "html"),
            "accessibility": ("accessibility", "accessibility_snapshot"),
            "computed_styles": ("computed_styles", "computed_style_data"),
            "runtime": ("runtime", "runtime_context"),
        }
        for field_name, aliases in inline_aliases.items():
            supplied = next((args[name] for name in aliases if name in args and args[name] is not None), None)
            if supplied is not None:
                try:
                    frontend_bundle[field_name] = parse_context_value(field_name, supplied)
                except FrontendReviewError as exc:
                    return exc.as_result()
        for field_name in ("viewport", "page", "source"):
            if field_name in args and args.get(field_name) is not None:
                frontend_bundle[field_name] = args[field_name]
        if image_artifact_id:
            frontend_bundle.setdefault("screenshot", {"artifact_id": image_artifact_id})

        for field_name in ("dom", "accessibility", "computed_styles", "runtime"):
            value = frontend_bundle.get(field_name)
            if isinstance(value, dict) and isinstance(value.get("content"), str):
                try:
                    decoded_value = parse_context_value(field_name, value["content"])
                except FrontendReviewError as exc:
                    return exc.as_result()
                ref = str(value.get("artifact_id", ""))
                frontend_bundle[field_name] = {"artifact_id": ref, **decoded_value} if ref else decoded_value

        model_context = None
        if any(key in frontend_bundle for key in ("dom", "accessibility", "computed_styles", "runtime")):
            try:
                dom_value = frontend_bundle.get("dom") or {}
                accessibility_value = frontend_bundle.get("accessibility") or {}
                styles_value = frontend_bundle.get("computed_styles") or {}
                runtime_value = frontend_bundle.get("runtime") or {}
                model_context = build_model_context(
                    prompt=prompt,
                    screenshot_data_url=images[0] if images else "",
                    dom=dom_value if isinstance(dom_value, dict) else parse_context_value("dom", dom_value),
                    accessibility=accessibility_value if isinstance(accessibility_value, dict) else parse_context_value("accessibility", accessibility_value),
                    computed_styles=styles_value if isinstance(styles_value, dict) else parse_context_value("computed_styles", styles_value),
                    viewport=frontend_bundle.get("viewport") or {},
                    runtime=runtime_value if isinstance(runtime_value, dict) else parse_context_value("runtime", runtime_value),
                )
                if model_context.dom.get("truncated"):
                    return FrontendReviewError(
                        "frontend_dom_context_truncated",
                        "vision review requires complete live DOM within the configured bound",
                        original_chars=model_context.dom.get("original_chars"),
                        limit_chars=model_context.dom.get("limit_chars"),
                    ).as_result()
                for field_name, value in (
                    ("dom", model_context.dom),
                    ("accessibility", model_context.accessibility),
                    ("computed_styles", model_context.computed_styles),
                    ("runtime", model_context.runtime),
                ):
                    original = frontend_bundle.get(field_name)
                    ref = original.get("artifact_id", "") if isinstance(original, dict) else ""
                    frontend_bundle[field_name] = {"artifact_id": ref, **value} if ref else value
            except FrontendReviewError as exc:
                return exc.as_result()

        if not images:
            return input_error("Vision image path, base64 data, or image artifact is required.")

        try:
            requested_max_tokens = int(args.get("max_tokens", 1200))
        except (TypeError, ValueError, OverflowError):
            requested_max_tokens = 1200
        max_tokens = max(64, min(requested_max_tokens, self._VISION_MAX_OUTPUT_TOKENS))
        schema = args.get("json_schema")
        if isinstance(schema, dict):
            try:
                schema_text = json_dumps(schema, ensure_ascii=False, separators=(",", ":"))
            except (TypeError, ValueError, OverflowError):
                return input_error("Vision JSON schema is invalid.")
            if len(schema_text) > VISION_MAX_SCHEMA_CHARS:
                return input_error("Vision JSON schema exceeds the bounded size limit.")
        prompt_suffix = (
            "\n\nReturn only a JSON object matching this contract: "
            "{\"summary\": string, \"findings\": [{\"id\": string, "
            "\"severity\": \"blocker|high|medium|low|info\", "
            "\"category\": \"layout|responsive|accessibility|interaction|visual-regression|runtime\", "
            "\"problem\": string, \"observed\": string, \"hypothesized\": string, "
            "\"uncertainty\": [string], \"confidence\": number, "
            "\"element_ids\": [string], \"bbox\": [number, number, number, number] or null, "
            "\"evidence\": [string], \"likely_cause\": string, \"fix_hint\": string, "
            "\"needs_runtime_check\": boolean}], "
            "\"unknowns\": [string], \"recommended_checks\": [string]} ."
        )
        prompt_context = prompt
        if model_context is not None:
            prompt_context += "\n\n" + UNTRUSTED_CONTEXT_INSTRUCTION + "\nFrontend evidence bundle:\n" + json_dumps(
                model_context.prompt_payload(), ensure_ascii=False
            )
        elif bundle_context:
            prompt_context += "\n\n" + UNTRUSTED_CONTEXT_INSTRUCTION + "\nFrontend evidence bundle:\n" + bundle_context
        prompt_budget = max(0, VISION_MAX_PROMPT_CHARS - len(prompt_suffix))
        if len(prompt_context) > prompt_budget:
            return input_error(
                "Vision prompt and frontend evidence exceed the bounded prompt limit.",
                "frontend_prompt_too_large",
            )
        prompt_context += prompt_suffix
        payload = {
            "model": model,
            "prompt": prompt_context,
            "images": images,
            "stream": False,
            "format": schema if isinstance(schema, dict) else "json",
            "options": {"num_predict": max_tokens},
        }
        payload, _profile = self.model_policy.apply_payload(
            model,
            payload,
            role="vision",
            output_tokens=max_tokens,
        )
        try:
            timeout = float(
                self.config.get("resilience", {}).get(
                    "vision_timeout_seconds",
                    self.config.get("server", {}).get("request_timeout_seconds", 300),
                )
            )
        except (TypeError, ValueError, OverflowError):
            timeout = 300.0
        timeout = max(0.05, min(timeout, 300.0))
        try:
            capability_response = self.runtime.request(
                "/api/show",
                {"name": model},
                timeout=min(timeout, 1.0),
            )
            if not isinstance(capability_response, dict):
                return {
                    "success": False,
                    "model": model,
                    "terminal": True,
                    "retryable": False,
                    "error": "Vision model capability preflight returned an invalid response.",
                }
            if capability_response.get("error"):
                capability_error = str(capability_response.get("error", "")).lower()
                missing_model = (
                    "model not found" in capability_error
                    or "no such model" in capability_error
                    or ("not found" in capability_error and "model" in capability_error)
                )
                transient = any(
                    marker in capability_error
                    for marker in (
                        "timeout", "timed out", "connection", "network", "handoff",
                        "temporarily", "unavailable", "try again", "http 429", "http 502", "http 503",
                    )
                )
                if missing_model:
                    return {
                        "success": False,
                        "model": model,
                        "unsupported": True,
                        "degraded": True,
                        "terminal": True,
                        "retryable": False,
                        "error": f"Vision model '{model}' is unavailable. Configure an existing vision-capable local backend and models.vision; no runtime is installed automatically.",
                        "error_code": "vision_model_missing",
                    }
                return {
                    "success": False,
                    "model": model,
                    "terminal": not transient,
                    "retryable": transient,
                    "error": "Vision service temporarily unavailable; retry the request." if transient else "Vision model capability preflight failed; inspect the configured local backend and vision model.",
                    "error_code": "vision_preflight_failed",
                }
            capability_values: list[str] = []
            for value in capability_response.get("capabilities", []):
                capability_values.append(str(value).lower())
            details = capability_response.get("details")
            if isinstance(details, dict):
                for key in ("family", "families"):
                    values = details.get(key, [])
                    if isinstance(values, list):
                        capability_values.extend(str(value).lower() for value in values)
                    elif values:
                        capability_values.append(str(values).lower())
            supports_vision = any(
                marker in value
                for value in capability_values
                for marker in ("vision", "image", "multimodal", "clip", "vl")
            )
            if not supports_vision:
                return {
                    "success": False,
                    "model": model,
                    "unsupported": True,
                    "degraded": True,
                    "terminal": True,
                    "retryable": False,
                    "error": f"Vision model '{model}' does not advertise image or multimodal capability; choose a configured vision-capable local model.",
                    "error_code": "vision_model_unsupported",
                }
            res = self.runtime.request("/api/generate", payload, timeout=timeout)
            if not isinstance(res, dict):
                return {
                    "success": False,
                    "model": model,
                    "terminal": True,
                    "retryable": False,
                    "error": "Vision runtime returned an invalid response.",
                }
            if res.get("error"):
                error_text = str(res.get("error", "")).lower()
                missing_model = (
                    "model not found" in error_text
                    or "no such model" in error_text
                    or ("not found" in error_text and "model" in error_text)
                )
                if missing_model:
                    return {
                        "success": False,
                        "model": model,
                        "unsupported": True,
                        "degraded": True,
                        "terminal": True,
                        "retryable": False,
                        "error": f"Vision model '{model}' is unavailable. Configure an existing vision-capable local backend and models.vision; no runtime is installed automatically.",
                    }
                transient = any(
                    marker in error_text
                    for marker in (
                        "timeout", "timed out", "connection", "network", "handoff",
                        "temporarily", "unavailable", "try again", "http 429", "http 502", "http 503",
                    )
                )
                return {
                    "success": False,
                    "model": model,
                    "terminal": not transient,
                    "retryable": transient,
                    "error": "Vision service temporarily unavailable; retry the request." if transient else "Vision runtime returned an error; inspect the configured local backend and vision model.",
                }
            raw_output = res.get("response", "")
            if not isinstance(raw_output, str):
                return {
                    "success": False,
                    "model": model,
                    "terminal": True,
                    "retryable": False,
                    "error": "Vision runtime returned an invalid response.",
                }
            if len(raw_output) > VISION_MAX_RUNTIME_OUTPUT_CHARS:
                raw_artifact_id = ""
                try:
                    raw_artifact_id = str(self.artifacts.put(raw_output[:VISION_MAX_RUNTIME_OUTPUT_CHARS], tenant, "vision-output"))
                except Exception:
                    pass
                return {
                    "success": False,
                    "model": model,
                    "terminal": True,
                    "retryable": False,
                    "error": "Vision runtime output exceeds the bounded response limit.",
                    "raw_output_artifact_id": raw_artifact_id,
                }
            raw_artifact_id = ""
            try:
                raw_artifact_id = str(self.artifacts.put(raw_output, tenant, "vision-output"))
            except Exception:
                pass
            parsed = parse_vision_result(
                raw_output, require_observation_fields=model_context is not None
            )
            if parsed.terminal:
                return {
                    "success": False,
                    "model": model,
                    "terminal": True,
                    "retryable": False,
                    "error": parsed.error,
                    "raw_output_artifact_id": raw_artifact_id,
                }
            parsed = bound_vision_result(parsed)
            result = {
                "success": True,
                "model": model,
                "response": raw_output[:VISION_MAX_INLINE_RESPONSE_CHARS],
                "review": asdict(parsed),
                "prompt": prompt_context,
                "image_artifact_id": image_artifact_id,
                "bundle_artifact_id": bundle_artifact_id,
                "raw_output_artifact_id": raw_artifact_id,
            }
            if frontend_bundle:
                repo_context = None
                root = str(args.get("root") or "").strip()
                fix_requested = any(
                    marker in prompt.lower()
                    for marker in ("fix", "repair", "implement", "change", "correct")
                )
                if root and fix_requested and self.repo_tools is not None:
                    findings = asdict(parsed).get("findings", [])
                    query = " ".join(
                        [
                            prompt,
                            *[str(item.get("finding_id", "")) for item in findings],
                            *[str(element_id) for item in findings for element_id in item.get("element_ids", [])],
                            *[str(item.get("fix_hint", "")) for item in findings],
                        ]
                    ).strip()
                    try:
                        context_result = self.repo_tools.context_pack(root, query, max_tokens=2600)
                        if isinstance(context_result, dict) and context_result.get("success", True):
                            repo_context = context_result
                    except Exception:
                        repo_context = None
                coder_context = build_coder_context(
                    asdict(parsed), frontend_bundle, repo_context=repo_context
                )
                if coder_context.get("terminal"):
                    return {
                        "success": False,
                        "model": model,
                        "terminal": True,
                        "retryable": False,
                        "error": coder_context.get("error", {}).get("message", "Invalid coder context"),
                        "error_code": coder_context.get("error", {}).get("code", "invalid_coder_context"),
                    }
                result["coder_context"] = coder_context
            return result
        except Exception as exc:
            detail = str(exc).lower()
            transient = isinstance(exc, (TimeoutError, ConnectionError, OSError)) or any(
                marker in detail for marker in ("timeout", "timed out", "connection", "network", "handoff", "temporarily", "unavailable")
            )
            return {
                "success": False,
                "model": model,
                "terminal": not transient,
                "retryable": transient,
                "error": "Vision service temporarily unavailable; retry the request." if transient else "Vision runtime failed; inspect the configured local backend and vision model.",
            }

    def transcribe(self, args: dict[str, Any], tenant: str) -> dict[str, Any]:
        """Transcribe audio recording to text via local Whisper / STT CLI or fallback."""
        audio_path = str(args.get("audio", args.get("audio_path", "")))
        model = str(args.get("model") or "whisper")

        p = Path(audio_path).expanduser().resolve(strict=False)
        if not p.is_file():
            return {"success": False, "error": f"audio file not found: {audio_path}"}

        whisper_cmd = shutil.which("whisper")
        if whisper_cmd:
            cmd = [whisper_cmd, str(p), "--output_format", "txt", "--model", "tiny"]
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=False, **hidden_run_kwargs())
                if proc.returncode == 0:
                    txt_path = p.with_suffix(".txt")
                    text = txt_path.read_text(encoding="utf-8", errors="replace") if txt_path.exists() else proc.stdout
                    return {
                        "success": True,
                        "text": text.strip(),
                        "audio_path": str(p),
                        "model": model,
                        "engine": "whisper_cli",
                    }
            except Exception:
                pass

        return {
            "success": True,
            "text": f"[Audio transcription registered: {p.name}]",
            "audio_path": str(p),
            "file_size_bytes": p.stat().st_size,
            "format": p.suffix.lstrip("."),
            "model": model,
            "engine": "local_stt_fallback",
            "degraded": True,
            "note": "Local whisper CLI not installed in PATH; audio metadata processed.",
        }

    def benchmark(self, tenant: str = "benchmark") -> dict[str, Any]:
        """Manual tiny benchmark used to seed the adaptive runtime cost model."""
        models=[]
        for key in ("fast_code","heavy_code","reasoning","general"):
            model=str(self.config.get("models",{}).get(key,"") or "")
            if model and model not in models: models.append(model)
        results=[]
        for model in models:
            started=time.perf_counter()
            def run(m=model):
                payload, _profile = self.model_policy.apply_payload(
                    m, {"model":m,"prompt":"Reply exactly: OK","stream":False,"keep_alive":self.config.get("ollama",{}).get("keep_alive","-1"),"options":{"num_predict":8,"temperature":0}},
                    role="benchmark", input_tokens=4, output_tokens=8, preserve_explicit_think=False,
                )
                return self.runtime.request("/api/generate", payload)
            try:
                raw=self.scheduler.submit(model, tenant, "benchmark", run, priority=2, wait_timeout=float(self.config.get("resilience",{}).get("scheduler_wait_timeout_seconds",330)))
            except Exception as exc:
                raw={"error":str(exc)}
            elapsed=(time.perf_counter()-started)*1000
            success="error" not in raw
            if success and self.tuner is not None:
                try: self.tuner.observe(model, raw, elapsed)
                except Exception: pass
            results.append({"model":model,"success":success,"elapsed_ms":round(elapsed,1),"load_ms":round(float(raw.get("load_duration",0) or 0)/1e6,1),"eval_count":int(raw.get("eval_count",0) or 0)})
        return {"success":all(x["success"] for x in results),"results":results,"autotune":self.tuner.stats() if self.tuner is not None else {"enabled":False}}

    def evaluation(self, payload: dict[str, Any]) -> dict[str, Any]:
        action = str(payload.get("action", "report")).strip().lower()
        if action == "record":
            try:
                duration_ms = float(payload.get("duration_ms", 0) or 0)
            except (TypeError, ValueError):
                return {"success": False, "error": "duration_ms must be numeric", "terminal": True}
            return self.telemetry.record_evaluation(
                task_id=str(payload.get("task_id", "")),
                cohort=str(payload.get("cohort", "")),
                quality_pass=payload.get("quality_pass"),
                test_pass=payload.get("test_pass"),
                duration_ms=duration_ms,
            )
        if action == "report":
            try:
                days = max(1, min(int(payload.get("days", 30) or 30), self.telemetry.rollup_retention_days))
            except (TypeError, ValueError):
                return {"success": False, "error": "days must be an integer", "terminal": True}
            return {"success": True, "evaluation": self.telemetry.report(days).get("evaluation", {})}
        return {"success": False, "error": "unknown evaluation action", "terminal": True}



    @staticmethod
    def _embedding_key(text: str, *, backend: str, model: str, query: bool) -> str:
        return stable_hash({
            "kind": "embedding",
            "backend": backend,
            "model": model,
            "query": bool(query),
            "text": str(text),
            "app_version": __version__,
        })

    @staticmethod
    def _json_embedding(vector: Any) -> list[float]:
        if hasattr(vector, "tolist"):
            vector = vector.tolist()
        return [float(value) for value in vector]

    def embed(self, texts: list[str], tenant: str, priority: int = 3, query: bool = False, background: bool | None = None, wait_timeout: float | None = None) -> dict[str, Any]:
        backend = self.config["models"].get("embedding_backend", "sentence-transformers")
        model = str(self.config.get("models", {}).get("embedding", "qwen3-embedding:0.6b"))
        if not texts:
            return {"success": True, "model": model, "backend": "ollama", "embeddings": []}

        def encode_uncached(batch: list[str]) -> dict[str, Any]:
            if backend in {"sentence-transformers", "openvino"}:
                return self.embeddings.encode(batch, query=query, priority=priority)
            batch_size = int(self.config.get("rag", {}).get("embedding_batch_size", 12))

            def run() -> dict[str, Any]:
                all_vectors: list[list[float]] = []
                for start in range(0, len(batch), batch_size):
                    response = self.runtime.request("/api/embed", {"model": model, "input": batch[start:start + batch_size]})
                    if "error" in response:
                        return {"success": False, "error": response["error"], "model": model}
                    all_vectors.extend(response.get("embeddings", []))
                return {"success": True, "model": model, "backend": "ollama", "embeddings": all_vectors}

            is_bg = background if background is not None else (priority <= 1)
            return self.scheduler.submit(model, tenant, "embed", run, priority=priority, background=is_bg, wait_timeout=wait_timeout)

        cache = getattr(self, "embedding_cache", None)
        cache_enabled = bool(getattr(self, "embedding_cache_enabled", self.config.get("cache", {}).get("embeddings", True)))
        if not cache_enabled or cache is None:
            return encode_uncached(list(texts))

        keys = [self._embedding_key(text, backend=backend, model=model, query=query) for text in texts]
        vectors: list[list[float] | None] = [None] * len(texts)
        missing: dict[str, list[int]] = {}
        for index, key in enumerate(keys):
            cached = cache.get(key)
            if isinstance(cached, list):
                vectors[index] = [float(value) for value in cached]
            else:
                missing.setdefault(key, []).append(index)

        if missing:
            missing_indices = [indices[0] for indices in missing.values()]
            result = encode_uncached([texts[index] for index in missing_indices])
            if not isinstance(result, dict) or not result.get("success"):
                return result if isinstance(result, dict) else {"success": False, "error": "embedding failed"}
            raw_vectors = result.get("embeddings", [])
            if len(raw_vectors) != len(missing_indices):
                return {"success": False, "error": f"embedding count mismatch: {len(raw_vectors)} != {len(missing_indices)}"}
            for vector, key, indices in zip(raw_vectors, missing, missing.values()):
                normalized = self._json_embedding(vector)
                cache.set(key, normalized)
                for index in indices:
                    vectors[index] = normalized
            result = dict(result)
            result["embeddings"] = vectors
            return result

        return {"success": True, "model": model, "backend": backend, "embeddings": vectors}

    def _repo_cached(self, operation: str, root: str, params: dict[str, Any], compute: Any) -> dict[str, Any]:
        # Worktrees/temp repositories can disappear between an agent request and a
        # coalesced background computation. Treat that as a stale input, not a hub 500.
        try:
            self._touch_project(root)
            state = self._repo_cache_state(root) or {}
            key = stable_hash({"op": operation, "state": state.get("fingerprint"), "params": params})
            raw, hit, coalesced = self.repo_flight.get_or_compute(key, compute)
        except (FileNotFoundError, ValueError) as exc:
            return {"success": False, "stale_root": True, "error": str(exc), "error_type": type(exc).__name__, "operation": operation}
        if isinstance(raw, dict):
            result = copy.deepcopy(raw)
        elif raw is not None:
            result = {"result": raw, "success": True}
        else:
            result = {"success": False, "error": f"{operation} computation failed"}
        result["workspace_cache"] = {
            "hit": bool(hit), "coalesced": bool(coalesced), "operation": operation,
            "fingerprint": state.get("fingerprint"), "kind": state.get("kind"),
            "degraded": bool(state.get("degraded", False)), "stale": bool(state.get("stale", False)),
        }
        # Promote mechanical cache reuse to the common HTTP contract. This lets
        # clients reuse repository answers and makes endpoint cache hit telemetry
        # comparable with generation cache telemetry.
        result["cache_hit"] = bool(hit)
        result["coalesced"] = bool(coalesced)
        result["cache_layer"] = "workspace" if hit else "workspace-miss"
        if isinstance(raw, dict) and "preprocessed_hit" in raw:
            result["preprocessed_hit"] = bool(raw["preprocessed_hit"])
        return result

    def repo_profile(self, root: str) -> dict[str, Any]:
        return self._repo_cached("profile", root, {}, lambda: self.repo_tools.project_profile(root))

    def repo_search(self, root: str, query: str, top_k: int = 12, context_lines: int | None = None, enrich: bool = False, progress: Any = None, path: str = "") -> dict[str, Any]:
        try:
            scope = RepositoryTools.normalize_search_scope(root, path) if path else ""
        except ValueError as exc:
            return {"success": False, "terminal": True, "retryable": False, "error": str(exc)}
        scope_args = {"path": scope} if scope else {}
        def stage(name: str) -> None:
            if progress is not None:
                try:
                    progress(name)
                except Exception:
                    pass

        if enrich and not rollout_feature_enabled(self.config, "enriched_search"):
            return {
                "success": False,
                "unsupported": True,
                "feature": "enriched_search",
                "error": "enriched search is disabled (features.enriched_search=false)",
            }
        # Search itself is case-insensitive and whitespace-tolerant. Use the same
        # canonical form for cache identity so equivalent agent queries reuse work.
        query = normalize_query(query).casefold()
        eff_ctx_lines = int(context_lines) if context_lines is not None else (8 if enrich else None)
        if self.learner is not None:
            stage("learner.start")
            try: self.learner.record(root, query)
            except Exception: pass
            stage("learner.done")
        # Local AI deterministic facts/symbols are the cheapest candidate source.
        paths: list[str] = []
        if self.deterministic is not None:
            stage("deterministic_candidates.start")
            try: paths.extend(self.deterministic.related_paths(root, query, max(16, top_k * 3)))
            except Exception: pass
            stage("deterministic_candidates.done")
        if self.code_index is not None:
            stage("code_index_candidates.start")
            try:
                for p in self.code_index.related_paths(root, query, max(16, top_k * 3)):
                    if p not in paths: paths.append(p)
            except Exception: pass
            stage("code_index_candidates.done")
        # Warm preprocessing maintains FTS + semantic cards specifically so agents do
        # not need broad scans. Treat these only as candidates; exact source search
        # remains the fallback when a card/FTS hint produces no matches.
        if self.preprocessor is not None:
            stage("preprocess_candidates.start")
            try:
                for p in self.preprocessor.candidate_paths(root, query, max(16, top_k * 3)):
                    if p not in paths:
                        paths.append(p)
            except Exception:
                pass
            stage("preprocess_candidates.done")
        def compute() -> dict[str, Any]:
            if paths:
                stage("targeted_search.start")
                targeted = (
                    self.repo_tools.search_paths(root, query, paths, top_k, context_lines=eff_ctx_lines, **scope_args)
                    if eff_ctx_lines is not None
                    else self.repo_tools.search_paths(root, query, paths, top_k, **scope_args)
                )
                stage("targeted_search.done")
            else:
                targeted = {"results": []}
            used_preprocessed = bool(paths and targeted.get("results"))
            if targeted.get("results"):
                result = targeted
            else:
                stage("full_search.start")
                result = (
                    self.repo_tools.search(root, query, top_k, context_lines=eff_ctx_lines, **scope_args)
                    if eff_ctx_lines is not None
                    else self.repo_tools.search(root, query, top_k, **scope_args)
                )
                stage("full_search.done")
            result["preprocessed_hit"] = used_preprocessed
            if enrich:
                result["enriched"] = True
                base = Path(root).expanduser().resolve()
                for hit in result.get("results", [])[:top_k]:
                    rel = str(hit.get("path") or "")
                    target = (base / rel).resolve()
                    try:
                        target.relative_to(base)
                        if target.is_file() and target.stat().st_size <= 512_000:
                            symbol = enclosing_symbol_at_line(
                                target.read_text(encoding="utf-8", errors="replace"), rel,
                                int(hit.get("start_line", 0) or 0),
                            )
                            if symbol:
                                hit["enclosing_symbol"] = symbol
                    except (OSError, ValueError):
                        continue
            # Exact search snippets become immutable evidence so agent projections can
            # send coordinates first and raw source only for the top few hits.
            if self.evidence_store is not None and isinstance(result.get("results"), list):
                stage("evidence_store.start")
                try:
                    result = dict(result)
                    result["results"] = self.evidence_store.put_many(str(result.get("root", root)), result["results"])
                    result["progressive_disclosure"] = True
                except Exception:
                    pass
                stage("evidence_store.done")
            return result
        stage("repo_cache.start")
        result = self._repo_cached("search", root, {"query": query, "path": scope, "top_k": top_k, "ci": paths[:24], "ctx": eff_ctx_lines}, compute)
        stage("repo_cache.done")
        return result

    def _refresh_changed_intelligence(self, root: str) -> None:
        """Synchronize only Git-changed files before serving a cache-miss intelligence query.

        Preprocessing remains the bulk indexer. This foreground safety net prevents a
        just-edited file from forcing a full rebuild or returning stale deterministic
        facts while the background pipeline is still catching up.
        """
        try:
            state = self._repo_cache_state(root)
            # Background preprocessing owns incremental index updates while active.
            # Do not duplicate that work synchronously inside a foreground query;
            # the durable index is safer than making the feature wait on a large batch.
            if state.get("kind") == "preprocessed-watcher":
                return
            fp = str(state.get("fingerprint") or "")
            base = Path(root).expanduser().resolve()
            root_key = str(base)
            with self._index_refresh_lock:
                if fp and self._index_refresh_state.get(root_key) == fp:
                    return
            paths = [str(x) for x in state.get("changed_paths", [])][:256]
            if not paths:
                with self._index_refresh_lock:
                    if fp:
                        self._index_refresh_state[root_key] = fp
                return
            existing: list[tuple[str, str]] = []
            missing: list[str] = []
            for rel in paths:
                target = (base / rel).resolve(strict=False)
                try:
                    target.relative_to(base)
                except ValueError:
                    continue
                if target.is_file():
                    try:
                        existing.append((rel, self.repo_tools._hash_file_only(target)))
                    except Exception:
                        pass
                else:
                    missing.append(rel)
            if self.code_index is not None:
                if existing and hasattr(self.code_index, "update_files_batch"):
                    self.code_index.update_files_batch(str(base), existing)
                for rel in missing:
                    try: self.code_index.update_file(str(base), rel)
                    except Exception: pass
            if self.deterministic is not None:
                if existing and hasattr(self.deterministic, "update_files_batch"):
                    self.deterministic.update_files_batch(str(base), existing)
                for rel in missing:
                    try: self.deterministic.update_file(str(base), rel)
                    except Exception: pass
            with self._index_refresh_lock:
                if fp:
                    self._index_refresh_state[root_key] = fp
                    if len(self._index_refresh_state) > 256:
                        self._index_refresh_state.pop(next(iter(self._index_refresh_state)), None)
        except Exception:
            # Query correctness still falls back to the last durable index plus exact
            # lexical/source tools; this accelerator must never turn a query into 500.
            return

    def deterministic_query(self, root: str, query: str, limit: int = 24) -> dict[str, Any]:
        if self.deterministic is None:
            return {"success": False, "error": "deterministic engine unavailable", "confidence": 0.0}
        if self.learner is not None:
            try: self.learner.record(root, query)
            except Exception: pass
        try:
            canonical = {
                "intent": self.deterministic.classify_intent(query).get("intent", "general"),
                "terms": sorted(set(self.deterministic._terms(query))),
                "limit": limit,
            }
        except Exception:
            canonical = {"query": " ".join(query.lower().split()), "limit": limit}
        def compute() -> dict[str, Any]:
            self._refresh_changed_intelligence(root)
            return self.deterministic.query(root, query, limit)
        return self._repo_cached("deterministic", root, canonical, compute)

    def code_query(self, root: str, query: str, limit: int = 20, include_code: bool = False) -> dict[str, Any]:
        if self.code_index is None:
            return {"success": False, "error": "code index unavailable"}
        if self.learner is not None:
            try: self.learner.record(root, query)
            except Exception: pass
        try:
            canonical = {"terms": sorted(set(self.code_index._query_terms(query))), "limit": limit, "include_code": include_code}
        except Exception:
            canonical = {"query": " ".join(query.lower().split()), "limit": limit, "include_code": include_code}
        def compute() -> dict[str, Any]:
            self._refresh_changed_intelligence(root)
            res = self.code_index.query(root, query, limit)
            if isinstance(res, dict):
                res["preprocessed_hit"] = bool(res.get("success") and res.get("symbols"))
                if include_code and res.get("symbols") and self.repo_tools is not None:
                    for sym in res["symbols"][:5]:
                        fp = sym.get("file") or sym.get("path")
                        sl = int(sym.get("line") or 1)
                        el = int(sym.get("end_line") or (sl + 30))
                        if fp:
                            slice_res = self.repo_tools.file_slice(root, fp, start_line=sl, end_line=min(el, sl + 60), max_chars=2500)
                            if slice_res.get("success"):
                                sym["code"] = slice_res.get("text")
            return res
        return self._repo_cached("code-index", root, canonical, compute)

    def repo_investigate(
        self,
        root: str,
        query: str,
        path: str = "",
        include_code: bool = True,
        limit: int = 10,
    ) -> dict[str, Any]:
        """Single-roundtrip smart investigation for cloud models."""
        cq_res = self.code_query(root, query, limit=limit, include_code=include_code)
        symbols = cq_res.get("symbols", []) if isinstance(cq_res, dict) else []

        search_hits = []
        if len(symbols) < 2:
            s_res = self.repo_search(root, query, top_k=limit)
            if isinstance(s_res, dict):
                search_hits = s_res.get("results", []) or s_res.get("hits", [])
                if include_code and self.repo_tools is not None:
                    for hit in search_hits[:3]:
                        h_path = hit.get("path") or hit.get("file")
                        h_line = int(hit.get("line") or 1)
                        if h_path:
                            sl_res = self.repo_tools.file_slice(root, h_path, start_line=max(1, h_line - 5), end_line=h_line + 30, max_chars=2000)
                            if sl_res.get("success"):
                                hit["code"] = sl_res.get("text")

        primary_sym = symbols[0].get("name", "") if symbols else (search_hits[0].get("name", "") if search_hits else query)
        callers: list[dict[str, Any]] = []
        if primary_sym and self.code_index is not None:
            try:
                ref_res = self.code_index.find_referencing_symbols(root, primary_sym)
                if isinstance(ref_res, dict) and ref_res.get("success"):
                    callers = ref_res.get("references", [])[:10]
            except Exception:
                pass

        related_tests: list[str] = []
        try:
            target_files = [s.get("file") for s in symbols if s.get("file")]
            if target_files:
                aff_res = self.affected_tests(root, changed_paths=target_files)
                if isinstance(aff_res, dict):
                    related_tests = aff_res.get("test_files", [])[:5]
        except Exception:
            pass

        return {
            "success": True,
            "query": query,
            "root": root,
            "primary_symbol": primary_sym,
            "symbols": symbols[:limit],
            "search_hits": search_hits[:limit] if not symbols else [],
            "callers": callers[:10],
            "related_tests": related_tests,
            "cascade_layers": ["code_index", "code_graph", "tests"] if symbols else ["search", "tests"],
        }

    def repo_diagnose(self, root: str, text: str) -> dict[str, Any]:
        """Parse stack traces / crash logs and fetch inline code snippets for each frame."""
        resolved_root = canonical_root(root)
        if not Path(resolved_root).is_dir():
            return {"success": False, "error": "Root directory does not exist"}

        if not text or not text.strip():
            return {"success": False, "error": "No traceback or error log text provided"}

        py_pat = re.compile(r'File "(?P<file>[^"]+)", line (?P<line>\d+)(?:, in (?P<func>\w+))?')
        js_pat = re.compile(r'at\s+(?:(?P<func>[^\s(]+)\s+\()?(?P<file>[^:()\s]+):(?P<line>\d+):(?P<col>\d+)\)?')
        cs_pat = re.compile(r'at\s+(?P<func>[^\s]+)\s+in\s+(?P<file>[^:]+):line\s+(?P<line>\d+)')
        go_pat = re.compile(r'(?P<file>[^\s:]+\.go):(?P<line>\d+)(?:\s+\+0x[0-9a-f]+)?')
        rs_pat = re.compile(r'at\s+(?P<file>[^:]+\.rs):(?P<line>\d+):(?P<col>\d+)')
        gen_pat = re.compile(r'(?P<file>[a-zA-Z0-9_./\\-]+\.[a-zA-Z0-9_]+):(?P<line>\d+)')

        raw_frames: list[dict[str, Any]] = []
        lines = text.splitlines()

        error_type = ""
        error_message = ""
        for line in reversed(lines):
            line_str = line.strip()
            if not line_str:
                continue
            err_m = re.search(r'\b(?P<type>[A-Z][A-Za-z0-9_]*(?:Error|Exception|Panic|Failure))\s*:\s*(?P<msg>.*)', line_str)
            if err_m:
                error_type = err_m.group("type")
                error_message = err_m.group("msg").strip()
                break
            if line_str.startswith("AssertionError") or line_str.startswith("FAILED") or line_str.startswith("panic:"):
                parts = line_str.split(":", 1)
                error_type = parts[0].strip()
                error_message = parts[1].strip() if len(parts) > 1 else ""
                break

        for line in lines:
            m = py_pat.search(line) or js_pat.search(line) or cs_pat.search(line) or rs_pat.search(line) or go_pat.search(line) or gen_pat.search(line)
            if m:
                d = m.groupdict()
                raw_frames.append({
                    "file": d.get("file", "").replace("\\", "/").strip(),
                    "line": int(d.get("line", 1)),
                    "function": d.get("func", ""),
                })

        base_path = Path(resolved_root)
        processed_frames: list[dict[str, Any]] = []
        seen = set()

        for f in raw_frames:
            f_str = f["file"]
            ln = f["line"]
            target = (base_path / f_str).resolve(strict=False)
            if not target.is_file() and Path(f_str).is_file():
                target = Path(f_str).resolve(strict=False)

            try:
                rel = target.relative_to(base_path).as_posix()
            except ValueError:
                rel = f_str

            key = (rel, ln)
            if key in seen:
                continue
            seen.add(key)

            snippet = ""
            if target.is_file():
                slice_res = self.repo_tools.file_slice(resolved_root, rel, start_line=max(1, ln - 4), end_line=ln + 4, max_chars=1200)
                if slice_res.get("success"):
                    snippet = slice_res.get("text", "")

            processed_frames.append({
                "file": rel,
                "line": ln,
                "function": f.get("function", ""),
                "in_project": target.is_file(),
                "snippet": snippet,
            })

        project_frames = [pf for pf in processed_frames if pf["in_project"]]
        root_cause = project_frames[-1] if project_frames else (processed_frames[-1] if processed_frames else None)

        return {
            "success": True,
            "root": resolved_root,
            "error_type": error_type or "UnknownError",
            "error_message": error_message,
            "frames_count": len(processed_frames),
            "project_frames_count": len(project_frames),
            "frames": processed_frames,
            "root_cause": root_cause,
        }

    def repo_briefing(self, root: str) -> dict[str, Any]:
        """Produce an ultra-compact (<300 token) project orientation snapshot for cloud models."""
        resolved_root = canonical_root(root)
        base = Path(resolved_root)
        if not base.is_dir():
            return {"success": False, "error": "Root directory does not exist"}

        manifests: list[str] = []
        stack_types: list[str] = []
        test_commands: list[str] = []

        if (base / "pyproject.toml").is_file() or (base / "setup.py").is_file() or (base / "requirements.txt").is_file():
            stack_types.append("Python")
            manifests.extend([m for m in ("pyproject.toml", "setup.py", "requirements.txt") if (base / m).is_file()])
            test_commands.append("pytest")
        if (base / "package.json").is_file():
            stack_types.append("TypeScript/JavaScript")
            manifests.append("package.json")
            test_commands.append("npm test")
        if any(base.glob("*.csproj")) or any(base.glob("*.sln")):
            stack_types.append("C# / .NET")
            test_commands.append("dotnet test")
        if (base / "Cargo.toml").is_file():
            stack_types.append("Rust")
            manifests.append("Cargo.toml")
            test_commands.append("cargo test")
        if (base / "go.mod").is_file():
            stack_types.append("Go")
            manifests.append("go.mod")
            test_commands.append("go test ./...")

        common_entries = [
            "src/main.py", "main.py", "app.py", "src/index.ts", "src/index.js",
            "Program.cs", "main.go", "src/main.rs"
        ]
        entry_points = [e for e in common_entries if (base / e).is_file()]

        branch = "unknown"
        dirty_files = 0
        recent_commits: list[str] = []
        try:
            from .process_utils import hidden_run_kwargs
            cp_branch = subprocess.run(
                ["git", "branch", "--show-current"],
                cwd=str(base), capture_output=True, text=True, check=False, timeout=2, **hidden_run_kwargs()
            )
            if cp_branch.returncode == 0:
                branch = cp_branch.stdout.strip()

            cp_status = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=str(base), capture_output=True, text=True, check=False, timeout=2, **hidden_run_kwargs()
            )
            if cp_status.returncode == 0:
                dirty_files = len([l for l in cp_status.stdout.splitlines() if l.strip()])

            cp_log = subprocess.run(
                ["git", "log", "-n", "3", "--oneline"],
                cwd=str(base), capture_output=True, text=True, check=False, timeout=2, **hidden_run_kwargs()
            )
            if cp_log.returncode == 0:
                recent_commits = [l.strip() for l in cp_log.stdout.splitlines() if l.strip()]
        except Exception:
            pass

        active_task = None
        task_store = getattr(self, "task_store", None)
        if task_store is not None:
            try:
                active_task = task_store.get_active_task()
            except Exception:
                active_task = None

        brief_card = (
            f"**Repo**: {base.name} | **Branch**: {branch} ({dirty_files} dirty)\n"
            f"**Stack**: {', '.join(stack_types) or 'Generic'} | **Manifests**: {', '.join(manifests) or 'None'}\n"
            f"**Test Runner**: {', '.join(test_commands) or 'None'}\n"
            f"**Entrypoints**: {', '.join(entry_points) or 'None'}\n"
            f"**Recent**: {recent_commits[0] if recent_commits else 'No git history'}"
        )

        return {
            "success": True,
            "root": resolved_root,
            "repo_name": base.name,
            "stack": stack_types,
            "manifests": manifests,
            "test_commands": test_commands,
            "entry_points": entry_points,
            "git": {
                "branch": branch,
                "dirty_files": dirty_files,
                "recent_commits": recent_commits,
            },
            "active_task": active_task,
            "markdown_card": brief_card,
        }

    def repo_map(self, root: str, max_symbols: int = 120) -> dict[str, Any]:
        def compute() -> dict[str, Any]:
            indexed_paths = None
            if self.preprocessor is not None:
                try:
                    indexed_paths = self.preprocessor.indexed_paths(root)
                except Exception:
                    indexed_paths = None
            if indexed_paths:
                res = self.repo_tools.repo_map(root, max_symbols, paths=indexed_paths)
                if isinstance(res, dict):
                    res["preprocessed_hit"] = True
                return res
            res = self.repo_tools.repo_map(root, max_symbols)
            if isinstance(res, dict):
                res["preprocessed_hit"] = False
            return res

        return self._repo_cached("map", root, {"max_symbols": max_symbols}, compute)

    def deterministic_operation(self, operation: str, root: str, params: dict[str, Any], compute: Any) -> dict[str, Any]:
        """Cache expensive deterministic analyses by repository fingerprint.

        These operations are pure derived views of repository state. Sharing the same
        cache with search/context prevents dashboard, MCP and local-model callers from
        independently repeating AST/security/test/dependency scans.
        """
        def wrapped_compute() -> dict[str, Any]:
            res = compute()
            if isinstance(res, dict):
                res["preprocessed_hit"] = bool(res.get("success", True))
            return res
        return self._repo_cached(f"det:{operation}", root, params, wrapped_compute)

    def test_matrix(self, root: str) -> dict[str, Any]:
        return self.deterministic_operation("test-matrix", root, {}, lambda: self.deterministic.test_matrix(root))

    def affected_tests(self, root: str, changed_paths: list[str] | None = None, base: str = "HEAD") -> dict[str, Any]:
        params = {"changed_paths": sorted(changed_paths or []), "base": base}
        return self.deterministic_operation("affected-tests", root, params, lambda: self.deterministic.affected_tests(root, changed_paths, base))

    def repo_topology(self, root: str) -> dict[str, Any]:
        return self.deterministic_operation("repo-topology", root, {}, lambda: self.deterministic.repo_topology(root))

    def ast_rename(self, root: str, file_path: str, old_symbol: str, new_symbol: str, apply_changes: bool = False) -> dict[str, Any]:
        return self.deterministic.ast_rename(root, file_path, old_symbol, new_symbol, apply_changes=apply_changes)

    def generate_mocks(self, root: str, file_path: str, symbol: str) -> dict[str, Any]:
        return self.deterministic.generate_mocks(root, file_path, symbol)

    def split_changes(self, root: str, changed_files: list[str] | None = None) -> dict[str, Any]:
        return self.deterministic.split_changes(root, changed_files)

    def synthesize_rules(self, root: str, limit: int = 10) -> dict[str, Any]:
        return self.deterministic.synthesize_rules(root, limit)

    def code_invariants(self, root: str, path: str | None = None) -> dict[str, Any]:
        return self.deterministic.code_invariants(root, path)

    def generate_dataset(self, root: str, schema_or_model: Any, count: int = 10, format: str = "json") -> dict[str, Any]:
        return self.deterministic.generate_dataset(root, schema_or_model, count=count, format=format)

    def profile_digest(self, profile_path: str, top_n: int = 15) -> dict[str, Any]:
        return self.deterministic.profile_digest(profile_path, top_n=top_n)

    def worktree_lease(self, root: str, branch_name: str | None = None, worktree_path: str | None = None) -> dict[str, Any]:
        from .process_utils import create_git_worktree
        return create_git_worktree(root, branch_name=branch_name, worktree_path=worktree_path)

    def worktree_release(self, root: str, worktree_path: str, delete_branch: bool = True, branch_name: str | None = None) -> dict[str, Any]:
        from .process_utils import remove_git_worktree
        return remove_git_worktree(root, worktree_path, delete_branch=delete_branch, branch_name=branch_name)

    def list_worktrees(self, root: str) -> dict[str, Any]:
        from .process_utils import list_git_worktrees
        return list_git_worktrees(root)

    def prune_worktrees(self, root: str) -> dict[str, Any]:
        from .process_utils import prune_git_worktrees
        return prune_git_worktrees(root)


    def spawn_daemon(self, command: str, cwd: str, name: str = "", env: dict[str, str] | None = None) -> dict[str, Any]:
        if not self.commands:
            return {"success": False, "error": "command broker unavailable"}
        return self.commands.spawn_daemon(command, cwd, name=name, env=env)

    def daemon_status(self, daemon_id: str | None = None) -> dict[str, Any]:
        if not self.commands:
            return {"success": False, "error": "command broker unavailable"}
        return self.commands.daemon_status(daemon_id)

    def stop_daemon(self, daemon_id: str) -> dict[str, Any]:
        if not self.commands:
            return {"success": False, "error": "command broker unavailable"}
        return self.commands.stop_daemon(daemon_id)

    def lint_fix(self, root: str, command: str | None = None, paths: list[str] | None = None, tenant: str = "agent", timeout: int | None = None) -> dict[str, Any]:
        if not self.commands:
            return {"success": False, "error": "command broker unavailable"}
        return self.commands.lint_fix(root, command=command, paths=paths, tenant=tenant, timeout=timeout)

    def http_probe(self, url: str, expected_status: int = 200, json_path: str | None = None, timeout: float = 5.0, headers: dict[str, str] | None = None) -> dict[str, Any]:
        if not self.commands:
            return {"success": False, "error": "command broker unavailable"}
        return self.commands.http_probe(url, expected_status=expected_status, json_path=json_path, timeout=timeout, headers=headers)

    def security_audit(self, root: str, limit: int = 200) -> dict[str, Any]:
        return self.deterministic_operation("security-audit", root, {"limit": int(limit)}, lambda: self.deterministic.security_audit(root, int(limit)))

    def ast_outline(self, root: str, path: str) -> dict[str, Any]:
        return self.deterministic_operation("ast-outline", root, {"path": path}, lambda: self.deterministic.ast_outline(root, path))

    def batch_replace(self, root: str, edits: list[dict[str, Any]], dry_run: bool = False) -> dict[str, Any]:
        if not rollout_feature_enabled(self.config, "batch_replacement"):
            return {
                "success": False,
                "unsupported": True,
                "feature": "batch_replacement",
                "error": "batch replacement is disabled (features.batch_replacement=false)",
            }
        if not self.deterministic:
            return {"success": False, "error": "deterministic engine disabled"}
        return self.deterministic.batch_replace(root, edits, dry_run=dry_run)

    def refactor_impact(self, root: str, file_path: str, symbol: str) -> dict[str, Any]:
        return self.deterministic_operation("refactor-impact", root, {"file": file_path, "symbol": symbol}, lambda: self.deterministic.refactor_impact(root, file_path, symbol))

    def resolve_imports(self, root: str, symbols: list[str], language: str = "auto") -> dict[str, Any]:
        params = {"symbols": sorted(str(x) for x in symbols), "language": language}
        return self.deterministic_operation("resolve-imports", root, params, lambda: self.deterministic.resolve_imports(root, symbols, language))

    def callers(self, root: str, symbol: str, limit: int = 50) -> dict[str, Any]:
        return self.deterministic_operation("callers", root, {"symbol": symbol, "limit": limit}, lambda: self.deterministic.find_callers(root, symbol, limit))

    def dead_code(self, root: str, limit: int = 200) -> dict[str, Any]:
        return self.deterministic_operation("dead-code", root, {"limit": int(limit)}, lambda: getattr(self.deterministic, "find_dead_code", getattr(self.deterministic, "detect_dead_code", None))(root, int(limit)))

    def secret_scan(self, root: str, path: str | None = None, scan_git_history: bool = False, commit_depth: int = 20) -> dict[str, Any]:
        return self.deterministic_operation("secret-scan", root, {"path": path, "git": scan_git_history, "depth": commit_depth}, lambda: self.deterministic.secret_scan(root, path, scan_git_history=scan_git_history, commit_depth=commit_depth))

    def schema_inspect(self, root: str, db_path: str | None = None) -> dict[str, Any]:
        return self.deterministic_operation("schema-inspect", root, {"db_path": db_path}, lambda: self.deterministic.schema_inspect(root, db_path))

    def explain_query(self, root: str, query: str, db_path: str | None = None) -> dict[str, Any]:
        return self.deterministic_operation("explain-query", root, {"query": query, "db_path": db_path}, lambda: self.deterministic.explain_query(root, query, db_path))

    def env_compat(self, root: str) -> dict[str, Any]:
        return self.deterministic_operation("env-compat", root, {}, lambda: self.deterministic.env_compat(root))

    def circular_dependencies(self, root: str, language: str = "python") -> dict[str, Any]:
        return self.deterministic_operation("circular-dependencies", root, {"language": language}, lambda: self.deterministic.find_circular_dependencies(root, language))

    def generate_types(self, root: str, file_path: str, write_stub: bool = False) -> dict[str, Any]:
        return self.deterministic_operation("generate-types", root, {"file": file_path, "write": write_stub}, lambda: self.deterministic.generate_types(root, file_path, write_stub))

    def code_complexity(
        self, root: str, path: str | None = None, max_results: int = 20, include_tests: bool = False
    ) -> dict[str, Any]:
        return self.deterministic_operation(
            "code-complexity", root,
            {"path": path, "max": max_results, "include_tests": include_tests},
            lambda: self.deterministic.code_complexity(root, path, max_results, include_tests),
        )

    def extract_api_spec(self, root: str, framework: str | None = None) -> dict[str, Any]:
        return self.deterministic_operation("extract-api-spec", root, {"framework": framework}, lambda: self.deterministic.extract_api_spec(root, framework))

    def slice_dependency_graph(self, root: str, symbol: str, path: str | None = None, depth: int = 2) -> dict[str, Any]:
        return self.deterministic_operation("slice-dependency-graph", root, {"symbol": symbol, "path": path, "depth": depth}, lambda: self.deterministic.slice_dependency_graph(root, symbol, path, depth))

    def migration_drift(self, root: str, db_path: str | None = None) -> dict[str, Any]:
        return self.deterministic_operation("migration-drift", root, {"db_path": db_path}, lambda: self.deterministic.migration_drift(root, db_path))

    def package_audit(self, root: str, lockfile_path: str | None = None) -> dict[str, Any]:
        return self.deterministic_operation("package-audit", root, {"lockfile": lockfile_path}, lambda: self.deterministic.package_audit(root, lockfile_path))

    def structural_search(self, root: str, pattern: str, path: str | None = None, max_results: int = 30) -> dict[str, Any]:
        return self.deterministic_operation("structural-search", root, {"pattern": pattern, "path": path, "max_results": max_results}, lambda: self.deterministic.structural_search(root, pattern, path=path, max_results=max_results))

    def context_budget(self, root: str, files: list[str] | None = None, max_tokens: int = 4000) -> dict[str, Any]:
        return self.deterministic_operation("context-budget", root, {"files": files, "max_tokens": max_tokens}, lambda: self.deterministic.context_budget(root, files=files, max_tokens=max_tokens))

    def git_diff(self, root: str, path: str | None = None, staged: bool = False, max_lines: int = 1000) -> dict[str, Any]:
        return self.deterministic_operation("git-diff", root, {"path": path, "staged": staged, "max_lines": max_lines}, lambda: self.deterministic.git_diff(root, path=path, staged=staged, max_lines=max_lines))

    def git_history_search(self, root: str, query: str, max_commits: int = 20) -> dict[str, Any]:
        return self.deterministic_operation("git-history-search", root, {"query": query, "max_commits": max_commits}, lambda: self.deterministic.git_history_search(root, query, max_commits=max_commits))

    def find_hotspots(self, root: str, days: int = 30, limit: int = 20) -> dict[str, Any]:
        return self.deterministic_operation("find-hotspots", root, {"days": days, "limit": limit}, lambda: self.deterministic.find_hotspots(root, days=days, limit=limit))

    def generate_tests_for_diff(self, root: str, diff: str | None = None, path: str | None = None) -> dict[str, Any]:
        return self.deterministic_operation("generate-tests-for-diff", root, {"diff": bool(diff), "path": path}, lambda: self.deterministic.generate_tests_for_diff(root, diff=diff, path=path))

    def cross_repo_contract(self, backend_root: str, frontend_root: str) -> dict[str, Any]:
        return self.deterministic_operation("cross-repo-contract", backend_root, {"frontend_root": frontend_root}, lambda: self.deterministic.cross_repo_contract(backend_root, frontend_root))

    def mock_server_start(self, root: str = ".", spec_path: str | None = None, port: int = 11440) -> dict[str, Any]:
        if not self.commands:
            return {"success": False, "error": "command broker unavailable"}
        return self.commands.mock_server_start(root, spec_path=spec_path, port=port)

    def mock_server_stop(self, port: int = 11440) -> dict[str, Any]:
        if not self.commands:
            return {"success": False, "error": "command broker unavailable"}
        return self.commands.mock_server_stop(port)

    def mock_server_status(self, port: int = 11440) -> dict[str, Any]:
        if not self.commands:
            return {"success": False, "error": "command broker unavailable"}
        return self.commands.mock_server_status(port)


    def diff_hunk_stage(self, root: str, patch: str) -> dict[str, Any]:
        if not self.commands:
            return {"success": False, "error": "command broker unavailable"}
        return self.commands.diff_hunk_stage(root, patch)

    def test_flaky_detect(self, root: str, command: str, runs: int = 5, timeout: int = 30) -> dict[str, Any]:
        if not self.commands:
            return {"success": False, "error": "command broker unavailable"}
        return self.commands.test_flaky_detect(root, command, runs=runs, timeout_per_run=timeout)

    def webhook_replay(self, url: str, payload: dict[str, Any] | str, secret: str = "", signature_header: str = "X-Hub-Signature-256", timeout: float = 10.0) -> dict[str, Any]:
        if not self.commands:
            return {"success": False, "error": "command broker unavailable"}
        return self.commands.webhook_replay(url, payload, secret=secret, signature_header=signature_header, timeout=timeout)

    def stash_save(self, root: str, message: str = "local_ai_hub_stash") -> dict[str, Any]:
        if not self.commands:
            return {"success": False, "error": "command broker unavailable"}
        return self.commands.stash_save(root, message=message)

    def stash_restore(self, root: str) -> dict[str, Any]:
        if not self.commands:
            return {"success": False, "error": "command broker unavailable"}
        return self.commands.stash_restore(root)

    def record_mock(self, url: str, cassette_name: str) -> dict[str, Any]:
        if not self.commands:
            return {"success": False, "error": "command broker unavailable"}
        return self.commands.record_mock(url, cassette_name)

    def replay_mock(self, cassette_name: str) -> dict[str, Any]:
        if not self.commands:
            return {"success": False, "error": "command broker unavailable"}
        return self.commands.replay_mock(cassette_name)

    def simulate_merge(self, root: str, source_branch: str, target_branch: str = "HEAD") -> dict[str, Any]:
        from .process_utils import simulate_git_merge
        return simulate_git_merge(self._root(root), source_branch, target_branch=target_branch)

    def pubsub_publish(self, topic: str, message: dict[str, Any] | str, sender: str = "agent") -> dict[str, Any]:
        from .agent_events import SwarmPubSub
        return SwarmPubSub.get_default().publish(topic, message, sender=sender)

    def pubsub_poll(self, topic: str, since_timestamp: float = 0.0, limit: int = 50) -> dict[str, Any]:
        from .agent_events import SwarmPubSub
        return SwarmPubSub.get_default().poll(topic, since_timestamp=since_timestamp, limit=limit)

    def synthesize_commit(self, root: str, hint: str = "", task_id: str = "") -> dict[str, Any]:
        tasks_data: list[dict[str, Any]] = []
        receipts_data: list[dict[str, Any]] = []

        if self.task_store is not None:
            try:
                if task_id:
                    t = self.task_store.get(task_id)
                    if t:
                        tasks_data.append(t.to_dict())
                else:
                    active = self.task_store.list_tasks(status=None, limit=5)
                    for t in active:
                        tasks_data.append(t.to_dict())
            except Exception:
                pass

        if self.verification_store is not None:
            try:
                tids = [task_id] if task_id else [str(t.get("task_id")) for t in tasks_data if t.get("task_id")]
                for tid in tids[:5]:
                    comp = self.verification_store.completion(tid)
                    for r in comp.receipts:
                        if r.passed:
                            receipts_data.append(r.to_dict())
            except Exception:
                pass

        params = {"hint": hint, "task_id": task_id, "tasks_count": len(tasks_data), "receipts_count": len(receipts_data)}
        return self.deterministic_operation(
            "commit-synthesis",
            root,
            params,
            lambda: self.deterministic.synthesize_commit(
                root, hint, task_id=task_id, tasks=tasks_data, receipts=receipts_data
            ),
        )

    def code_inspect_symbol(self, root: str, symbol: str) -> dict[str, Any]:
        if self.code_index is None:
            return {"success": False, "error": "code index disabled"}
        return self._repo_cached("ci:inspect_symbol", root, {"symbol": symbol}, lambda: self.code_index.inspect_symbol(root, symbol))

    def code_find_symbol(self, root: str, pattern: str, depth: int = 0, include_body: bool = False, include_info: bool = True, relative_path: str | None = None, limit: int = 30) -> dict[str, Any]:
        if self.code_index is None:
            return {"success": False, "error": "code index disabled"}
        params = {"pattern": pattern, "depth": depth, "include_body": include_body, "include_info": include_info, "path": relative_path, "limit": limit}
        return self._repo_cached("ci:find_symbol", root, params, lambda: self.code_index.find_symbol(root, pattern, depth=depth, include_body=include_body, include_info=include_info, relative_path=relative_path, limit=limit))

    def code_find_declaration(self, root: str, symbol: str, path: str | None = None) -> dict[str, Any]:
        if self.code_index is None:
            return {"success": False, "error": "code index disabled"}
        return self._repo_cached("ci:find_declaration", root, {"symbol": symbol, "path": path}, lambda: self.code_index.find_declaration(root, symbol, path=path))

    def code_find_implementations(self, root: str, symbol: str, path: str | None = None) -> dict[str, Any]:
        if self.code_index is None:
            return {"success": False, "error": "code index disabled"}
        return self._repo_cached("ci:find_implementations", root, {"symbol": symbol, "path": path}, lambda: self.code_index.find_implementations(root, symbol, path=path))

    def code_find_referencing_symbols(self, root: str, symbol: str, path: str | None = None) -> dict[str, Any]:
        if self.code_index is None:
            return {"success": False, "error": "code index disabled"}
        return self._repo_cached("ci:find_referencing_symbols", root, {"symbol": symbol, "path": path}, lambda: self.code_index.find_referencing_symbols(root, symbol, path=path))

    def code_symbols_overview(self, root: str, path: str, depth: int = 1) -> dict[str, Any]:
        if self.code_index is None:
            return {"success": False, "error": "code index disabled"}
        return self._repo_cached("ci:symbols_overview", root, {"path": path, "depth": depth}, lambda: self.code_index.get_symbols_overview(root, path, depth=depth))

    def code_diagnostics(self, root: str, path: str) -> dict[str, Any]:
        if self.code_index is None:
            return {"success": False, "error": "code index disabled"}
        return self._repo_cached("ci:diagnostics", root, {"path": path}, lambda: self.code_index.get_diagnostics_for_file(root, path))

    def audit_dependencies(self, root: str) -> dict[str, Any]:
        if self.deterministic is None:
            return {"success": False, "error": "deterministic engine disabled"}
        return self.deterministic_operation("audit-dependencies", root, {}, lambda: self.deterministic.audit_dependencies(root))

    def symbol_callgraph(self, root: str, symbol: str | None = None, limit: int = 50) -> dict[str, Any]:
        if self.deterministic is None:
            return {"success": False, "error": "deterministic engine disabled"}
        return self.deterministic_operation("symbol-callgraph", root, {"symbol": symbol, "limit": limit}, lambda: self.deterministic.symbol_callgraph(root, symbol, limit))

    def repo_impact(self, root: str, base: str = "HEAD", staged: bool = False, max_symbols: int = 48, max_dependents: int = 30) -> dict[str, Any]:
        def compute() -> dict[str, Any]:
            # Local AI deterministic primary path: refresh only changed files in the code
            # index, then resolve dependents/tests through SQLite refs/edges. A full
            # repository scan is only a bounded fallback for sparse indexes.
            if self.code_index is not None and hasattr(self.code_index, "impact"):
                diff = self.repo_tools.git_diff(root, base, staged, max_tokens=24000)
                if diff.get("terminal"):
                    return diff
                if diff.get("success"):
                    changed = [str(x) for x in diff.get("changed_files", [])]
                    if not changed:
                        return {
                            "success": True, "root": str(Path(root).expanduser().resolve()), "changed_files": [],
                            "changed_symbols": [], "likely_dependents": [], "suggested_tests": [],
                            "risk": {"score": 0, "level": "low", "reasons": ["no changed files"]}, "method": "code-index",
                        }
                    try:
                        indexed = self.code_index.impact(root, changed, max_symbols, max_dependents)
                    except Exception:
                        indexed = {"success": False}
                    if indexed.get("success") and (indexed.get("changed_symbols") or float(indexed.get("confidence", 0.0) or 0.0) >= 0.70):
                        det = {}
                        if self.deterministic is not None:
                            try: det = self.deterministic.diff_facts(str(diff.get("diff", "")))
                            except Exception: det = {}
                        signals = det.get("risk_signals", {}) if isinstance(det, dict) else {}
                        score = 0; reasons: list[str] = []
                        if len(changed) >= 6: score += 2; reasons.append("multi-file change")
                        if det.get("manifest_files"): score += 2; reasons.append("manifest/build metadata changed")
                        if det.get("breaking_changes"): score += 4; reasons.append(f"{len(det['breaking_changes'])} breaking change(s) detected")
                        for key, weight, label in (("security",2,"security-sensitive code"),("concurrency",2,"concurrency-sensitive code"),("database",1,"database/persistence code"),("shell-exec",2,"process/shell execution"),("dynamic-eval",3,"dynamic evaluation")):
                            if int(signals.get(key, 0) or 0): score += weight; reasons.append(label)
                        if len(indexed.get("likely_dependents", [])) >= 10: score += 1; reasons.append("many indexed dependents")
                        if not indexed.get("suggested_tests") and any(Path(x).suffix.lower() in self.repo_tools.extensions for x in changed):
                            score += 1; reasons.append("no indexed tests found")
                        level = "high" if score >= 5 else "medium" if score >= 2 else "low"
                        if not reasons: reasons.append("localized indexed change")
                        return {
                            "success": True, "root": str(Path(root).expanduser().resolve()), "changed_files": changed,
                            "changed_symbols": indexed.get("changed_symbols", []), "likely_dependents": indexed.get("likely_dependents", []),
                            "suggested_tests": indexed.get("suggested_tests", []), "risk": {"score": score, "level": level, "reasons": reasons},
                            "diff_sha256": diff.get("diff_sha256"), "method": "code-index", "index_confidence": indexed.get("confidence", 0.0),
                            "deterministic_diff": det,
                        }
            fallback = self.repo_tools.impact_analysis(root, base, staged, max_symbols, max_dependents)
            if isinstance(fallback, dict): fallback.setdefault("method", "scan-fallback")
            return fallback
        return self._repo_cached(
            "impact", root, {"base": base, "staged": staged, "max_symbols": max_symbols, "max_dependents": max_dependents}, compute,
        )

    def repo_diff(self, root: str, base: str = "HEAD", staged: bool = False, max_tokens: int = 10000) -> dict[str, Any]:
        return self._repo_cached(
            "diff", root, {"base": base, "staged": staged, "max_tokens": max_tokens},
            lambda: self.repo_tools.git_diff(root, base, staged, max_tokens),
        )

    @staticmethod
    def _adaptive_label(value: Any, limit: int = 160) -> str:
        text = str(value or "")[:limit]
        if re.search(r"(?i)(api[_ -]?key|access[_ -]?token|password|secret|authorization|bearer)\s*[:=]", text):
            return "<redacted>"
        return re.sub(r"[^A-Za-z0-9_./:@+ -]", " ", text).strip()[:limit]

    def _adaptive_memory_revision(self, request: ConsistencyRequest) -> str:
        explicit = str(getattr(request, "memory_revision", "") or "")[:200]
        if explicit:
            return explicit
        store = getattr(self.consistency_guard, "memory_store", None)
        for name in ("memory_revision", "revision"):
            value = getattr(store, name, "") if store is not None else ""
            if callable(value):
                try:
                    value = value(request.root)
                except TypeError:
                    try:
                        value = value()
                    except Exception:
                        value = ""
                except Exception:
                    value = ""
            if value:
                return str(value)[:200]
        return ""

    def _context_preload_source(self) -> ContextCompiler | None:
        compiler = getattr(self, "context_compiler", None)
        if compiler is not None:
            return compiler
        try:
            compiler = ContextCompiler(
                AgentStateStore(
                    configured_state_dir(getattr(self, "config", {}), create=False) / "agent_state.sqlite3",
                    enabled=False,
                ),
                config=getattr(self, "config", {}),
            )
        except Exception:
            return None
        self.context_compiler = compiler
        return compiler

    def _apply_context_preloads(
        self, request: ConsistencyRequest, base: dict[str, Any],
    ) -> tuple[dict[str, Any], tuple[GuardWarning, ...]]:
        compiler = self._context_preload_source()
        if compiler is None:
            return base, ()
        try:
            elements, preload_warnings = compiler.preload_elements(ContextRequest(
                task_id=request.task_id,
                token_budget=max(1, int(request.token_budget)),
                changed_paths=tuple(request.changed_paths or ()),
                root=request.root,
                tenant=request.tenant,
                phase=request.phase,
                focus=tuple(request.focus or ()),
                preload_profile=request.preload_profile,
            ))
        except Exception:
            return base, ()
        if not elements and not preload_warnings:
            return base, ()

        enriched = dict(base)
        evidence = [dict(item) for item in (base.get("evidence") or ()) if isinstance(item, dict)]
        context = str(base.get("context", "") or "").strip()
        for element in elements:
            content = str(element.content or "")
            context = "\n\n".join(item for item in (context, content) if item)
            provenance = dict(element.provenance or {})
            evidence.append({
                "evidence_id": element.element_id,
                "source_kind": element.source_kind,
                "path": provenance.get("path", ""),
                "text": content,
                "reason": element.reason,
            })
        enriched["context"] = context
        enriched["evidence"] = evidence[:64]
        enriched["preload_profile"] = request.preload_profile
        enriched["preload_evidence_ids"] = [element.element_id for element in elements[:32]]
        warning_objects = tuple(
            GuardWarning(
                "warning",
                str(item.get("code", "preload_warning")),
                str(item.get("message", "preload warning")),
                affected_paths=(str(item["path"]),) if item.get("path") else (),
                recommended_action="review configured context preload",
            )
            for item in preload_warnings
            if isinstance(item, dict)
        )
        return enriched, warning_objects

    def _adaptive_relevance(self, request: ConsistencyRequest, evidence: tuple[dict[str, Any], ...], revision: str, memory_revision: str) -> dict[str, Any]:
        guard = self.consistency_guard
        if guard is None or not evidence:
            return {"success": False, "degraded": True, "degraded_reason": "no_authoritative_evidence", "warnings": []}
        try:
            projector = getattr(guard, "structured_evidence", None)
            if callable(projector):
                structured = list(projector(evidence, limit=24))
            else:
                structured = []
                for item in evidence[:24]:
                    if isinstance(item, dict):
                        structured.append({
                            "evidence_id": str(item.get("evidence_id") or item.get("id") or "")[:200],
                            "authority": "deterministic",
                            "path": self._adaptive_label(item.get("path"), 240),
                            "start_line": max(0, int(item.get("start_line", 0) or 0)),
                            "end_line": max(0, int(item.get("end_line", 0) or 0)),
                        })
            structured = [item for item in structured if isinstance(item, dict) and item.get("evidence_id")][:24]
            if not structured:
                return {"success": False, "degraded": True, "degraded_reason": "no_authoritative_evidence", "warnings": []}
            focus = [self._adaptive_label(item, 80) for item in tuple(getattr(request, "focus", ()) or ())[:16]]
            model_focus = ["focus-" + hashlib.sha256(item.encode("utf-8", "replace")).hexdigest()[:12] for item in focus if item]
            phase = self._adaptive_label(request.phase, 80)
            if phase.casefold() not in {"plan", "edit", "implementation", "review", "test", "handoff"}:
                phase = "other"
            preload_profile = self._adaptive_label(request.preload_profile, 120)
            if preload_profile.casefold() not in {"", "default", "plan", "edit", "review", "test", "handoff"}:
                preload_profile = "profile-" + hashlib.sha256(preload_profile.encode("utf-8", "replace")).hexdigest()[:12]
            payload = {
                "operation": "context_relevance",
                "phase": phase,
                "focus": model_focus,
                "preload_profile": preload_profile,
                "repository_revision": str(revision)[:200],
                "memory_revision": str(memory_revision)[:200],
                "evidence": structured,
                "limits": {"max_selected_evidence": 12, "max_claims": 16, "max_gaps": 16, "max_wording_chars": 1200},
            }
            params = {
                "repository_revision": str(revision)[:200],
                "phase": self._adaptive_label(request.phase, 80),
                "memory_revision": str(memory_revision)[:200],
                "focus": tuple(focus),
                "preload_profile": self._adaptive_label(request.preload_profile, 120),
                "evidence_ids": tuple(str(item["evidence_id"])[:200] for item in structured),
            }

            def compute() -> dict[str, Any]:
                try:
                    models = getattr(self, "config", {}).get("models", {})
                    model = getattr(self, "config", {}).get("context_relevance_model") or models.get("general") or models.get("fast_code") or "context-relevance"
                    generated = self._generate(
                        str(model),
                        json.dumps(payload, ensure_ascii=True, separators=(",", ":")),
                        "Return JSON only. Use only supplied deterministic evidence IDs. Never invent repository facts, repeat source text, reveal secrets, or use the user prompt as evidence.",
                        min(512, max(64, int(request.token_budget) // 4)),
                        0.0,
                        request.tenant or "context",
                        "context-relevance",
                        4,
                        semantic_query="context relevance",
                        semantic_context_fingerprint=stable_hash(params),
                        internal=True,
                        format={"type": "object"},
                    )
                except Exception:
                    return {"success": False, "degraded": True, "degraded_reason": "model_error", "warnings": []}
                if not isinstance(generated, dict) or not generated.get("success"):
                    return {"success": False, "degraded": True, "degraded_reason": "model_unavailable", "warnings": []}
                raw = generated.get("text") or generated.get("response") or ""
                try:
                    if isinstance(raw, dict):
                        parsed = raw
                    else:
                        text = str(raw).strip()
                        if text.startswith("```"):
                            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I | re.S)
                        parsed = json.loads(text)
                    if not isinstance(parsed, dict):
                        raise ValueError("structured response required")
                except Exception:
                    return {"success": False, "degraded": True, "degraded_reason": "invalid_model_output", "warnings": []}
                claims = parsed.get("claims") if isinstance(parsed.get("claims"), list) else []
                postprocess = getattr(guard, "postprocess_model_claims", None)
                if callable(postprocess):
                    checked = postprocess(evidence, claims, request)
                else:
                    checked = {"claims": [], "unknowns": [], "warnings": []}
                authoritative = set(checked.get("authoritative_evidence_ids", ()))
                selected = [str(item)[:200] for item in (parsed.get("selected_evidence_ids") or ()) if str(item) in authoritative][:12]
                gaps = [self._adaptive_label(item, 240) for item in (parsed.get("gaps") or ())][:16]
                wording = self._adaptive_label(parsed.get("wording"), 1200)
                warning_dicts = []
                for warning in checked.get("warnings", ())[:24]:
                    converter = getattr(warning, "to_dict", None)
                    warning_dicts.append(converter() if callable(converter) else dict(warning) if isinstance(warning, dict) else {"code": "unsupported_model_claim", "message": self._adaptive_label(warning, 240)})
                return {
                    "success": True,
                    "selected_evidence_ids": selected,
                    "claims": list(checked.get("claims", ()))[:16],
                    "unknowns": [self._adaptive_label(item, 240) for item in checked.get("unknowns", ())][:16],
                    "gaps": [item for item in gaps if item],
                    "wording": wording,
                    "warnings": warning_dicts[:24],
                }

            return self._repo_cached("adaptive-relevance", request.root, params, compute)
        except Exception:
            return {"success": False, "degraded": True, "degraded_reason": "relevance_error", "warnings": []}

    def adaptive_context_pack(self, request: ConsistencyRequest, *, mode: str = "fast", since_hash: str = "") -> dict[str, Any]:
        """Build deterministic context plus bounded, soft consistency findings."""
        if not isinstance(request, ConsistencyRequest):
            raise TypeError("guarded context request must be ConsistencyRequest")
        if mode == "full":
            base = self._hybrid_context(request.root, request.query, request.tenant, request.workspace or None, request.token_budget)
        else:
            base = self.fast_context(request.root, request.query, request.token_budget)
        if not isinstance(base, dict):
            base = {"success": False, "context": "", "evidence": []}
        else:
            base = dict(base)
        base, preload_warnings = self._apply_context_preloads(request, base)
        guard = self.consistency_guard
        if guard is None:
            return {**base, "guarded": False, "context_pack": AdaptiveContextPack(warnings=preload_warnings).to_dict()}

        revision = ""
        try:
            # Use the bounded worktree fingerprint for guarded provenance.  The
            # GitSnapshot revision is HEAD-only, which makes dirty working-tree
            # receipts look stale even when the indexed evidence is current.
            state = self.repo_state.fingerprint(request.root, force=True)
            if isinstance(state, dict) and not state.get("degraded") and not state.get("stale"):
                revision = str(state.get("fingerprint", "") or "")[:200]
        except Exception:
            pass
        try:
            snapshot = self.repo_tools.git_snapshot(request.root)
            if not revision:
                revision = str(getattr(snapshot, "revision", "") or "")[:200]
            snapshot_paths = tuple(str(path) for path in (getattr(snapshot, "changed_paths", ()) or ()))[:64]
        except Exception:
            snapshot_paths = ()
        evidence = tuple(item for item in (base.get("evidence") or ()) if isinstance(item, dict))[:24]
        changed_paths = tuple(request.changed_paths or base.get("changed_paths") or snapshot_paths)[:64]
        if since_hash and since_hash == revision:
            changed_paths = ()
        contract = guard.build_contract(request)
        candidates = tuple(guard.find_reuse_candidates(request, contract))[:24]
        mappings, mapping_warnings = guard.build_contract_mappings(request, evidence)
        diff: Any = None
        try:
            diff = self.repo_tools.git_diff(request.root, base=request.base, staged=request.staged, max_tokens=request.token_budget)
            if since_hash and isinstance(diff, dict):
                diff = {**diff, "diff_sha256": since_hash}
        except Exception:
            pass
        drift_warnings = guard.check_drift(request, contract, changed_paths, diff)
        warnings = tuple(dict.fromkeys((*preload_warnings, *mapping_warnings, *drift_warnings)))[:24]
        boundary = any(item.requires_approval or item.severity.lower() in {"boundary", "high", "high-risk"} for item in warnings)
        stop_code = self._guard_stop_code(request, boundary)
        if stop_code:
            warning_payload = [
                warning.to_dict() if hasattr(warning, "to_dict") else dict(warning)
                for warning in warnings
            ]
            return {
                **base,
                "success": False,
                "guarded": True,
                "delivery_mode": mode,
                "terminal": True,
                "retryable": False,
                "stop_code": stop_code,
                "error": f"guarded context stopped: {stop_code}",
                "warnings": warning_payload[:24],
                "repo_revision": revision,
                "changed_paths": list(changed_paths),
                "task_status": "",
                "waiting": False,
            }
        memory_revision = self._adaptive_memory_revision(request)
        relevance = self._adaptive_relevance(request, evidence, revision, memory_revision)
        metric = getattr(guard, "record_metric", None)
        if callable(metric):
            if since_hash and since_hash == revision:
                metric("duplicate_context_reuse")
            if not isinstance(relevance, dict) or not relevance.get("success"):
                metric("degraded_local_model_fallback")
            relevance_warnings = tuple(relevance.get("warnings", ())) if isinstance(relevance, dict) else ()
            for warning in relevance_warnings:
                severity = getattr(warning, "severity", None)
                if severity is None and isinstance(warning, dict):
                    severity = warning.get("severity", "warning")
                metric("warning", severity=str(severity or "warning"))
        model_warnings = tuple(relevance.get("warnings", ()))[:24] if isinstance(relevance, dict) else ()
        pack = AdaptiveContextPack(
            contract=contract,
            reuse_candidates=candidates,
            mappings=tuple(mappings),
            warnings=warnings,
            evidence=evidence,
            repo_revision=revision,
            changed_paths=changed_paths,
            stale=False,
            context_id=hashlib.sha256(f"{request.root}:{revision}:{request.task_id}:{request.phase}:{request.query}".encode("utf-8", "replace")).hexdigest()[:24],
            phase=request.phase,
            focus=request.focus,
            preload_profile=request.preload_profile,
            memory_revision=memory_revision,
            model_warnings=model_warnings,
        )
        pack_data = pack.to_dict()
        ordinary = any(not (item.requires_approval or item.severity.lower() in {"boundary", "high", "high-risk"}) for item in warnings)
        decision_recorded, decision_persisted = self._guard_decision(request, revision, reason=request.override_reason, approval=request.approval)
        task_status, waiting = self._guard_task_state(request, warnings, revision)
        approved = self._guard_truthy(request.approval) or self._guard_existing_approval(request)
        if str(task_status).lower() == "waiting":
            approved = False
        result = dict(base)
        result.update({
            "guarded": True,
            "delivery_mode": mode,
            "adaptive_context_pack": pack_data,
            "context_pack": pack_data,
            "contract": pack_data["contract"],
            "reuse": pack_data["reuse_candidates"],
            "reuse_candidates": pack_data["reuse_candidates"],
            "mappings": pack_data["mappings"],
            "warnings": [*pack_data["warnings"], *list(model_warnings)][:24],
            "evidence_ids": list(dict.fromkeys(item.get("evidence_id", "") for item in evidence if item.get("evidence_id")))[:24],
            "warning_ids": list(dict.fromkeys(
                evidence_id
                for item in (*warnings, *model_warnings)
                for evidence_id in (item.evidence_ids if hasattr(item, "evidence_ids") else item.get("evidence_ids", ()) if isinstance(item, dict) else ())
            ))[:24],
            "model_warnings": list(model_warnings),
            "relevance": {
                key: relevance.get(key)
                for key in ("selected_evidence_ids", "claims", "unknowns", "gaps", "wording")
                if isinstance(relevance, dict) and key in relevance
            },
            "model_degraded": bool(not isinstance(relevance, dict) or not relevance.get("success")),
            "model_degraded_reason": str(relevance.get("degraded_reason", ""))[:120] if isinstance(relevance, dict) else "relevance_error",
            "repo_revision": revision,
            "memory_revision": memory_revision,
            "changed_paths": list(changed_paths),
            "since_hash": str(since_hash)[:200],
            "delta_from": str(since_hash)[:200] if since_hash else "",
            "requires_override": bool(ordinary and not str(request.override_reason or "").strip() and not approved),
            "requires_approval": bool(boundary and not approved),
            "decision_recorded": bool(decision_recorded),
            "decision_persisted": bool(decision_persisted),
            "task_status": task_status,
            "waiting": bool(waiting),
        })
        snapshotter = getattr(getattr(self, "telemetry", None), "record_snapshot", None)
        metrics_getter = getattr(guard, "metrics_snapshot", None)
        if callable(snapshotter) and callable(metrics_getter):
            try:
                snapshotter("consistency_guard", metrics_getter())
            except Exception:
                pass
        return result

    def fast_context(self, root: str, query: str, max_tokens: int) -> dict[str, Any]:
        """Return bounded deterministic context when foreground SLO excludes hybrid retrieval."""
        if self.deterministic is None:
            return {"success": False, "terminal": True, "retryable": False, "error": "deterministic context is unavailable"}
        dcfg = self.config.get("deterministic", {})
        char_budget = min(int(dcfg.get("context_max_chars", 5200)), max(900, int(max_tokens) * 4))
        packed = self._repo_cached(
            "fast-context", root,
            {"query": query, "max_chars": char_budget, "raw": int(dcfg.get("context_raw_evidence", 5))},
            lambda: self.deterministic.context_pack(
                root, query, max_chars=char_budget,
                max_raw_evidence=int(dcfg.get("context_raw_evidence", 5)),
            ),
        )
        if isinstance(packed, dict):
            packed["context_source"] = "deterministic-fast"
            packed["degraded"] = True
            packed["continuation"] = {
                "available": True, "mode": "full",
                "hint": "Request context mode=full only when deterministic-fast context is insufficient.",
            }
        return packed

    def _hybrid_context(self, root: str, query: str, tenant: str, workspace: str | None, max_tokens: int) -> dict[str, Any]:
        resolved_root = Path(root).expanduser().resolve(strict=False)
        if not resolved_root.is_dir():
            return {
                "success": False,
                "root": str(resolved_root),
                "query": query,
                "stale_root": True,
                "degraded": True,
                "error_type": "ValueError",
                "error": f"root directory does not exist: {resolved_root}",
                "context": "",
                "evidence": [],
            }
        # Local AI enforced deterministic-first fast path. This lives below the MCP/skill
        # layer so even a client that directly asks for context/delegation avoids
        # embeddings/reranking when parser/index evidence is already sufficient.
        if self.deterministic is not None:
            try:
                det = self.deterministic_query(root, query, 30)
                dcfg = self.config.get("deterministic", {})
                enough = (
                    det.get("success")
                    and float(det.get("confidence", 0.0) or 0.0) >= float(dcfg.get("context_confidence", 0.80))
                    and bool(det.get("facts") or det.get("dependencies") or det.get("scripts") or det.get("evidence") or det.get("code_index"))
                )
                if enough:
                    char_budget = min(
                        int(dcfg.get("context_max_chars", 5200)),
                        max(900, int(max_tokens) * 4),
                    )
                    packed = self._repo_cached(
                        "deterministic-context", root,
                        {"query": query, "max_chars": char_budget, "raw": int(dcfg.get("context_raw_evidence", 5))},
                        lambda: self.deterministic.context_pack(
                            root, query, max_chars=char_budget,
                            max_raw_evidence=int(dcfg.get("context_raw_evidence", 5)),
                        ),
                    )
                    if isinstance(packed, dict):
                        packed["context_source"] = "deterministic"
                    return packed
            except Exception:
                pass
        rag_revision = self.rag.revision(tenant, workspace or self.rag.workspace_id(root)) if self.rag is not None else None
        result = self._repo_cached(
            "hybrid-context", root,
            {"query": query, "workspace": workspace, "max_tokens": max_tokens, "rag_revision": rag_revision},
            lambda: self._hybrid_context_uncached(root, query, tenant, workspace, max_tokens),
        )
        if isinstance(result, dict): result.setdefault("context_source", "hybrid")
        return result

    @staticmethod
    def _rrf_fuse(lexical_ranked: list[str], vector_ranked: list[str], k: int = 60) -> list[tuple[str, float]]:
        """Reciprocal Rank Fusion: RRF(d) = sum(1 / (k + rank_i(d)))."""
        scores: dict[str, float] = {}
        for rank, item in enumerate(lexical_ranked):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank + 1)
        for rank, item in enumerate(vector_ranked):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank + 1)
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

    def _hybrid_context_uncached(self, root: str, query: str, tenant: str, workspace: str | None, max_tokens: int) -> dict[str, Any]:
        if self.learner is not None:
            try: self.learner.record(root, query)
            except Exception: pass
        target_workspace = workspace
        if self.rag is not None and self.config.get("features", {}).get("rag", True) and not target_workspace:
            candidate = self.rag.workspace_id(root)
            known = {item["workspace"] for item in self.rag.list_workspaces(tenant)}
            if candidate in known:
                target_workspace = candidate

        search_cfg = self.config.get("search", {})
        semantic_fraction = max(0.0, min(float(search_cfg.get("semantic_fraction", 0.35)), 0.7))
        lexical_tokens = max_tokens if not target_workspace else max(512, int(max_tokens * (1.0 - semantic_fraction)))
        # Local AI: preprocessed deterministic/semantic cards narrow the lexical scan to likely files.
        # This is an optimization hint only; if cards are absent or select nothing,
        # the full lexical search remains the correctness fallback.
        candidate_paths: list[str] = []
        # Prioritize files explicitly named in the query/task (e.g. AGENTS.md, src/main.py)
        root_path = Path(root)
        for token in query.split():
            clean_tok = token.strip("`'\"(),:;[]{}*")
            if clean_tok and ("." in clean_tok or "/" in clean_tok or "\\" in clean_tok):
                try:
                    rel_p = clean_tok.replace("\\", "/").lstrip("./")
                    if (root_path / rel_p).exists() and rel_p not in candidate_paths:
                        candidate_paths.append(rel_p)
                except Exception:
                    pass
        if self.deterministic is not None:
            try: candidate_paths.extend(self.deterministic.related_paths(root, query, limit=28))
            except Exception: pass
        if self.code_index is not None:
            try:
                for p in self.code_index.related_paths(root, query, limit=24):
                    if p not in candidate_paths: candidate_paths.append(p)
            except Exception: pass
        if self.preprocessor is not None:
            try:
                for p in self.preprocessor.candidate_paths(
                    root, query, limit=max(12, int(self.config.get("preprocessing", {}).get("lookup_file_cards", 5)) * 4)
                ):
                    if p not in candidate_paths: candidate_paths.append(p)
            except Exception:
                # Keep already-resolved deterministic/code-index candidates. One
                # optional preprocessing failure must not force a full repository scan.
                pass
        if candidate_paths:
            candidate_paths = list(dict.fromkeys(candidate_paths))
            lexical = self.repo_tools.context_pack_paths(
                root, query, candidate_paths, max_tokens=lexical_tokens,
                top_k=int(search_cfg.get("context_top_k", 14)),
            )
            # If the targeted cards produce no exact evidence, fall back to the
            # complete lexical scan. Cards accelerate; they never become a completeness gate.
            if lexical.get("success") and not lexical.get("evidence"):
                lexical = self.repo_tools.context_pack(
                    root, query, max_tokens=lexical_tokens,
                    top_k=int(search_cfg.get("context_top_k", 14)),
                )
        else:
            lexical = self.repo_tools.context_pack(
                root, query, max_tokens=lexical_tokens,
                top_k=int(search_cfg.get("context_top_k", 14)),
            )
        if not lexical.get("success"):
            return lexical
        evidence = list(lexical.get("evidence", []))
        if self.evidence_store is not None:
            try: evidence = self.evidence_store.put_many(str(lexical.get("root", root)), evidence)
            except Exception: pass
        context = str(lexical.get("context", ""))
        semantic_used = False
        per_path: dict[str, int] = {}
        for item in evidence:
            path = str(item.get("path", ""))
            per_path[path] = per_path.get(path, 0) + 1
        max_per_file = max(1, int(search_cfg.get("max_snippets_per_file", 3)))

        if self.rag is not None and target_workspace:
            from .token_router import strip_boilerplate
            rr = self.rag.search(query, tenant, target_workspace, top_k=10, use_reranker=True)
            if rr.get("success") and rr.get("results"):
                # RRF candidate ordering
                lex_paths = [str(e.get("path")) for e in evidence if e.get("path")]
                vec_paths = [str(r.get("path")) for r in rr["results"] if r.get("path")]
                fused_order = [p for p, _ in self._rrf_fuse(lex_paths, vec_paths)]
                fused_rank = {p: i for i, p in enumerate(fused_order)}
                sorted_results = sorted(rr["results"], key=lambda item: fused_rank.get(str(item.get("path", "")), 999))
                remaining = max(0, chars_for_tokens(max_tokens) - len(context))
                seen_chunks: set[tuple[str, int]] = set()
                for item in sorted_results:
                    path = str(item.get("path", ""))
                    chunk_no = int(item.get("chunk_no", 0))
                    key = (path, chunk_no)
                    if key in seen_chunks or per_path.get(path, 0) >= max_per_file:
                        continue
                    seen_chunks.add(key)
                    clean_text = strip_boilerplate(str(item.get("text", "")))
                    rendered = f"\n--- semantic {path} chunk={chunk_no} ---\n{clean_text}\n"
                    if len(rendered) > remaining:
                        if remaining > 400:
                            rendered = rendered[:remaining] + "\n[semantic snippet truncated]\n"
                        else:
                            break
                    context += rendered
                    remaining -= len(rendered)
                    per_path[path] = per_path.get(path, 0) + 1
                    semantic_used = True
                    evidence.append({
                        "path": path, "chunk_no": chunk_no, "semantic": True,
                        "content_hash": item.get("content_hash"),
                    })
                    if remaining <= 0:
                        break
        return {
            "success": True, "root": lexical["root"], "query": query, "context": context,
            "evidence": evidence, "estimated_tokens": estimate_tokens(context),
            "scanned_files": lexical.get("scanned_files", 0), "semantic_used": semantic_used,
            "context_budget_tokens": max_tokens,
        }

    def delegate_repo(self, args: dict[str, Any], tenant: str) -> dict[str, Any]:
        if str(args.get("profile", "")).strip():
            return self.delegate_profile(args, tenant)
        root = str(args.get("root", "."))
        task = str(args.get("task", ""))
        requested_context_tokens = int(args.get("context_tokens", 0) or 0)
        if requested_context_tokens > 0:
            max_context_tokens = requested_context_tokens
        else:
            initial_route = self.router.classify(
                task, "", str(args.get("task_type", "auto")), str(args.get("complexity", "auto"))
            )
            workflow = self.config.get("workflow", {})
            if initial_route.get("complexity") == "heavy":
                max_context_tokens = int(workflow.get("heavy_repo_context_tokens", 4200))
            else:
                max_context_tokens = int(workflow.get("fast_repo_context_tokens", 2200))
        packed = self._hybrid_context(root, task, tenant, args.get("workspace"), max_context_tokens)
        if not packed.get("success"):
            return packed
        payload = {
            "task": task,
            "context": packed.get("context", ""),
            "task_type": str(args.get("task_type", "auto")),
            "complexity": str(args.get("complexity", "auto")),
            "max_tokens": int(args.get("max_tokens", 4096)),
            "priority": int(args.get("priority", 5)),
            # Packed local context is a diagnostic, not proof of cloud-side input
            # avoided: the agent may have used RG or another narrower tool.
        }
        result = self.delegate(payload, tenant)
        result["repo_context"] = {
            "root": packed.get("root", root), "evidence": packed.get("evidence", []),
            "local_context_tokens": packed.get("estimated_tokens", 0), "semantic_used": bool(packed.get("semantic_used", False)),
            "scanned_files": packed.get("scanned_files", 0),
        }
        return result

    def solve_repo(self, args: dict[str, Any], tenant: str) -> dict[str, Any]:
        if self.pipeline is None:
            return self.delegate_repo(args, tenant)
        return self.pipeline.solve_repo(args, tenant)

    def route_context(self, args: dict[str, Any], tenant: str) -> dict[str, Any]:
        if self.token_router is None:
            return {"success": False, "error": "lossless token router unavailable"}
        query = str(args.get("query", args.get("task", "")))
        if args.get("path"):
            return self.token_router.route_file(str(args.get("root", ".")), str(args.get("path")), query, tenant)
        return self.token_router.route_text(str(args.get("text", args.get("context", ""))), query, tenant, path=args.get("label"))

    def review_diff(self, args: dict[str, Any], tenant: str) -> dict[str, Any]:
        diff = self.repo_diff(
            str(args.get("root", ".")), str(args.get("base", "HEAD")), bool(args.get("staged", False)),
            max_tokens=int(args.get("diff_tokens", self.config.get("token_saving", {}).get("max_diff_tokens", 8000))),
        )
        if not diff.get("success"):
            return diff
        if not diff.get("diff", "").strip():
            return {"success": True, "text": "No diff to review.", "changed_files": [], "diff_truncated": False}
        det_diff = {}
        if self.deterministic is not None:
            try:
                det_diff = self.deterministic.diff_facts(str(diff.get("diff", "")))
            except Exception:
                det_diff = {}
        instructions = str(args.get("instructions", "Review this git diff for actionable defects, regressions, security/concurrency issues and missing tests. Cite changed files/hunks."))
        det_hint = ""
        if det_diff:
            det_hint = "\nDETERMINISTIC DIFF METADATA (facts only, verify semantics in the diff):\n" + json_dumps(det_diff, ensure_ascii=False, separators=(",", ":"))[:1800] + "\n"
            if det_diff.get("breaking_changes"):
                det_hint += "\nPOTENTIAL BREAKING CHANGES DETECTED:\n"
                for bc in det_diff["breaking_changes"][:10]:
                    det_hint += f"- [{bc.get('type')}] {bc.get('symbol')} in {bc.get('file')}: {bc.get('description')}\n"

        regressions: list[dict[str, Any]] = []
        incident_store = getattr(self, "incident_store", None)
        if incident_store is not None and diff.get("changed_files"):
            try:
                found = incident_store.find_regressions(diff["changed_files"])
                if isinstance(found, list):
                    regressions = [r for r in found if isinstance(r, dict)]
            except Exception:
                regressions = []
        if regressions:
            det_hint += "\nAUTO-REGRESSION WATCHDOG (verified fixes from prior incidents affecting changed files):\n"
            for reg in regressions[:5]:
                aff = ", ".join(reg.get("affected_paths", []))
                det_hint += f"- [PRIOR VERIFIED FIX] {reg.get('error_class')} in {aff}: {reg.get('verified_fix')}\n"

        if args.get("mode") == "ast" or str(args.get("instructions", "")).strip().lower() == "ast":
            return {
                "success": True,
                "mode": "ast",
                "changed_files": diff.get("changed_files", []),
                "diff_facts": det_diff,
                "breaking_changes": det_diff.get("breaking_changes", []),
                "regressions": regressions,
                "summary": f"AST Diff Summary: {len(diff.get('changed_files', []))} files, {det_diff.get('added_lines_count', 0)} additions, {det_diff.get('deleted_lines_count', 0)} deletions across {det_diff.get('hunks_count', 0)} hunks. {len(det_diff.get('breaking_changes', []))} potential breaking changes.",
            }

        # Keep non-diff prompt material bounded too; metadata descriptions and custom
        # instructions can otherwise undo the per-segment input limit.
        instructions = fit_text(instructions, 600).text
        det_hint = fit_text(det_hint, 700).text if det_hint else ""

        diff_text = str(diff["diff"])
        complexity_hint = review_diff_complexity(args, det_diff, diff)
        full_route = self.router.classify(
            instructions + det_hint, diff_text, "review", complexity_hint
        )
        full_route = self._resident_optimize(full_route, "review", complexity_hint)
        review_model = str(full_route.get("model", ""))
        selected_tier = self.model_policy.tier_for(review_model)
        complexity = (
            "heavy" if selected_tier == "smart"
            else "fast" if selected_tier in {"fast", "background"}
            else str(full_route.get("complexity", "fast"))
        )
        max_output_tokens = max(64, int(args.get("max_tokens", 1800)))
        vram_free_mb = None
        try:
            if getattr(self, "vram_balancer", None):
                vram_free_mb = self.vram_balancer.status().get("vram_available_mb")
        except Exception:
            vram_free_mb = None
        profile = self.model_policy.profile(
            review_model,
            role="review",
            input_tokens=estimate_tokens(instructions + det_hint + diff_text),
            output_tokens=max_output_tokens,
            vram_free_mb=vram_free_mb,
        )
        configured_input_budget = max(
            _REVIEW_DIFF_MIN_CHUNK_TOKENS,
            int(self.config.get("token_saving", {}).get("max_local_input_tokens", 56000)),
        )
        effective_prompt_budget_tokens = min(configured_input_budget, profile.prompt_budget_tokens)
        task_budget = max(128, int(effective_prompt_budget_tokens * 0.15))
        review_task = fit_text(instructions + det_hint, task_budget).text
        prompt_overhead = estimate_tokens(f"TASK:\n{review_task}\n\nCONTEXT:\n") + 64
        synthesis_task = fit_text(
            "SYNTHESIZE REVIEW FINDINGS ONLY. Merge duplicate findings and keep distinct actionable findings. "
            "Preserve file paths, hunk/line references, severity and uncertainty. Do not re-review the source, "
            "invent new findings, or discard a finding unless supplied findings show it is a duplicate or false positive. "
            f"Return a concise final review.\n\nReview goal: {review_task}",
            task_budget,
        ).text
        synthesis_prompt_overhead = estimate_tokens(f"TASK:\n{synthesis_task}\n\nCONTEXT:\n") + 64
        synthesis_context_budget = max(
            0,
            min(
                _REVIEW_SYNTHESIS_CONTEXT_TOKENS,
                int(effective_prompt_budget_tokens * _REVIEW_DIFF_CONTEXT_FRACTION),
                effective_prompt_budget_tokens - synthesis_prompt_overhead,
            ),
        )
        review_chunk_token_budget = min(
            int(effective_prompt_budget_tokens * _REVIEW_DIFF_CONTEXT_FRACTION),
            effective_prompt_budget_tokens - prompt_overhead,
        )
        if review_chunk_token_budget < _REVIEW_DIFF_MIN_CHUNK_TOKENS:
            return {
                "success": False,
                "error": "The selected model's prompt budget is too small for a safe diff review.",
                "changed_files": diff.get("changed_files", []),
                "effective_prompt_budget_tokens": effective_prompt_budget_tokens,
            }

        review_chunks = _split_review_diff(diff_text, max_tokens=review_chunk_token_budget)
        largest_review_chunk_tokens = max(estimate_tokens(chunk) for chunk in review_chunks)
        if largest_review_chunk_tokens > review_chunk_token_budget:
            return {
                "success": False,
                "error": (
                    f"A diff fragment exceeds the {review_chunk_token_budget}-token review budget and cannot be split safely. "
                    "Narrow the diff or exclude generated/minified files."
                ),
                "changed_files": diff.get("changed_files", []),
                "diff_truncated": bool(diff.get("truncated", False)),
                "largest_review_chunk_tokens": largest_review_chunk_tokens,
                "review_chunk_token_budget": review_chunk_token_budget,
            }
        max_review_chunks = _MAX_REVIEW_DIFF_CHUNKS
        if bool(args.get("_async_job")):
            max_review_chunks = max(
                _MAX_REVIEW_DIFF_CHUNKS,
                int(self.config.get("review", {}).get("max_async_chunks", 32)),
            )
        if len(review_chunks) > max_review_chunks:
            return {
                "success": False,
                "error": (
                    f"Review requires {len(review_chunks)} segments; the safe limit is "
                    f"{max_review_chunks}. Narrow the diff or lower diff_tokens."
                ),
                "changed_files": diff.get("changed_files", []),
                "diff_truncated": bool(diff.get("truncated", False)),
                "review_chunks": len(review_chunks),
            }
        chunked = len(review_chunks) > 1
        if chunked:
            header_tokens = sum(
                estimate_tokens(f"### Counter-review segment {index}/{len(review_chunks)}\n") + 2
                for index in range(1, len(review_chunks) + 1)
            )
            per_chunk_output_tokens = min(
                max_output_tokens,
                700,
                max(64, (synthesis_context_budget - header_tokens - 64) // len(review_chunks)),
            )
        else:
            per_chunk_output_tokens = max_output_tokens
        review_payloads: list[dict[str, Any]] = []
        primary_results: list[dict[str, Any]] = []
        format_recovery_used = False
        format_recoveries: list[dict[str, Any]] = []
        for index, review_code in enumerate(review_chunks, start=1):
            task = review_task
            if chunked:
                task += (
                    f"\n\nThis is review segment {index}/{len(review_chunks)} of one diff. "
                    "Review only the changed code shown in this segment; other segments are reviewed separately. "
                    "Ground each finding in the displayed file and hunk."
                )
            review_payload = {
                "task_type": "review",
                "operation": "review_diff",
                "task": task,
                "context": review_code,
                "changed_paths": diff.get("changed_files", []),
                "static_facts": det_hint,
                "complexity": complexity,
                "max_tokens": per_chunk_output_tokens,
                "priority": int(args.get("priority", 5)),
            }
            review_payloads.append(review_payload)
            response = self.delegate(review_payload, tenant)
            if not isinstance(response, dict) or response.get("success") is False:
                if len(review_chunks) == 1 and isinstance(response, dict):
                    return response
                return {
                    "success": False,
                    "error": f"Review segment {index}/{len(review_chunks)} failed.",
                    "failed_segment": index,
                    "partial_reviews": [str(item.get("text", "")) for item in primary_results],
                }
            output_error = _review_text_error(response.get("text"))
            if output_error:
                if format_recovery_used:
                    return {
                        "success": False,
                        "error": f"Review segment {index}/{len(review_chunks)} returned unusable output after the single format-recovery attempt: {output_error}",
                        "failed_segment": index,
                        "model": response.get("model"),
                        "attempted_models": [str(response.get("model") or "")],
                        "invalid_model_output": True,
                        "partial_reviews": [str(item.get("text", "")) for item in primary_results],
                    }
                format_recovery_used = True
                primary_model = str(response.get("model") or "").strip()
                models = self.config.get("models", {})
                fast_model = str(models.get("fast_code") or "").strip() if isinstance(models, dict) else ""
                heavy_model = str(models.get("heavy_code") or "").strip() if isinstance(models, dict) else ""
                retry_model = fast_model or heavy_model or primary_model
                if (
                    fast_model
                    and heavy_model
                    and primary_model
                    and _review_model_identity(primary_model) == _review_model_identity(fast_model)
                ):
                    retry_model = heavy_model
                retry_payload = dict(review_payload)
                if retry_model:
                    retry_payload["model"] = retry_model
                retry_payload["task"] = (
                    f"{review_payload.get('task', review_task)}\n\nFORMAT RECOVERY: The previous response was unusable ({output_error}). "
                    "Review this diff again. Return plain text whose first line begins `SUMMARY:` followed by at least two words. "
                    "Then list only actionable defects, or write `- None.` when there are none. Do not emit fragments or markdown fences."
                )
                try:
                    retry_response = self.delegate(retry_payload, tenant)
                except Exception as exc:
                    return {
                        "success": False,
                        "error": f"Review segment {index}/{len(review_chunks)} format recovery failed: {exc}",
                        "failed_segment": index,
                        "model": retry_model or primary_model or None,
                        "attempted_models": [model for model in (primary_model, retry_model) if model],
                        "invalid_model_output": True,
                        "partial_reviews": [str(item.get("text", "")) for item in primary_results],
                    }
                retry_error = (
                    _review_text_error(retry_response.get("text"))
                    if isinstance(retry_response, dict) and retry_response.get("success") is not False
                    else "fallback model did not return a successful review"
                )
                if retry_error:
                    retry_model_returned = (
                        str(retry_response.get("model") or retry_model or "").strip()
                        if isinstance(retry_response, dict)
                        else retry_model
                    )
                    return {
                        "success": False,
                        "error": f"Review segment {index}/{len(review_chunks)} remained unusable after format recovery: {retry_error}",
                        "failed_segment": index,
                        "model": retry_model_returned or primary_model or None,
                        "attempted_models": [model for model in (primary_model, retry_model_returned) if model],
                        "invalid_model_output": True,
                        "partial_reviews": [str(item.get("text", "")) for item in primary_results],
                    }
                response = retry_response
                format_recoveries.append({
                    "segment": index,
                    "from_model": primary_model or None,
                    "to_model": str(response.get("model") or retry_model or "").strip() or None,
                })
            primary_results.append(response)

        result = dict(primary_results[0])
        if format_recoveries:
            result["degraded"] = True
            result["format_recoveries"] = format_recoveries

        merge_options = {
            "chunked": chunked,
            "synthesis_context_budget": synthesis_context_budget,
            "synthesis_task": synthesis_task,
            "max_output_tokens": max_output_tokens,
            "complexity": complexity,
            "tenant": tenant,
            "delegate": self.delegate,
            "priority": int(args.get("priority", 5)),
        }
        result["text"], result["review_synthesis"] = _merge_review_segment_results(
            primary_results, label="Review segment", task_type="review", **merge_options
        )
        if result["review_synthesis"].get("model"):
            result["model"] = result["review_synthesis"]["model"]

        # Multi-model consensus review for high-risk breaking changes or when explicitly requested
        consensus_requested = bool(args.get("consensus", False))
        auto_consensus = args.get("consensus") is None and bool(det_diff.get("breaking_changes"))

        if consensus_requested or auto_consensus:
            try:
                secondary_results = []
                for review_payload in review_payloads:
                    sec_payload = dict(review_payload)
                    sec_payload["task_type"] = "reasoning"
                    sec_payload["task"] = (
                        "CRITICAL COUNTER-REVIEW / CONSENSUS AUDIT:\n"
                        "Analyze the shown diff segment independently and verify potential defects or breaking changes. "
                        "Confirm genuine issues and flag false positives.\n\n"
                        + str(review_payload["task"])
                    )
                    sec_result = self.delegate(sec_payload, tenant)
                    if not isinstance(sec_result, dict) or sec_result.get("success") is False:
                        raise ValueError(f"Counter-review segment {len(secondary_results) + 1} failed.")
                    output_error = _review_text_error(sec_result.get("text"))
                    if output_error:
                        raise ValueError(
                            f"Counter-review segment {len(secondary_results) + 1} returned unusable output: {output_error}"
                        )
                    secondary_results.append(sec_result)
                if secondary_results:
                    secondary_text, secondary_synthesis = _merge_review_segment_results(
                        secondary_results,
                        label="Counter-review segment",
                        task_type="reasoning",
                        **merge_options,
                    )
                    result["consensus"] = {
                        "enabled": True,
                        "triggered_by": "explicit" if consensus_requested else "breaking_changes",
                        "primary_model": result.get("model", "primary"),
                        "secondary_model": secondary_results[0].get("model", "secondary"),
                        "secondary_review": secondary_text,
                        "synthesis": secondary_synthesis,
                    }
                    result["text"] = str(result.get("text", "")) + "\n\n### Consensus / Counter-Review Findings:\n" + secondary_text
            except Exception as exc:
                result["consensus"] = {"enabled": True, "degraded": True, "error": str(exc)}

        if regressions:
            result["regressions"] = regressions
        result["diff"] = {
            "changed_files": diff["changed_files"],
            "truncated": diff["truncated"],
            "local_diff_tokens": diff["estimated_tokens"],
            "original_diff_tokens": diff["original_estimated_tokens"],
            "deterministic": det_diff,
            "review_chunks": len(review_chunks),
            "effective_prompt_budget_tokens": effective_prompt_budget_tokens,
            "review_chunk_token_budget": review_chunk_token_budget,
            "review_chunk_context_fraction": _REVIEW_DIFF_CONTEXT_FRACTION,
            "review_segment_output_budget_tokens": per_chunk_output_tokens,
            "review_synthesis_context_budget_tokens": synthesis_context_budget,
            "review_model": review_model,
            "largest_review_chunk_tokens": largest_review_chunk_tokens,
        }
        return result

    def compress(self, args: dict[str, Any], tenant: str) -> dict[str, Any]:
        text = str(args.get("text", ""))
        instruction = str(args.get("instruction", "Compress this while preserving facts, identifiers, numbers, decisions, errors and uncertainty."))
        target_tokens = max(128, int(args.get("target_tokens", 900)))
        if not text.strip():
            return {"success": True, "text": "", "input_tokens": 0}

        # Local AI deterministic compressor handles logs/repetitive diagnostics without
        # spending any local-model inference. Narrative text falls through to Ollama.
        if self.deterministic is not None and bool(self.config.get("deterministic", {}).get("compression", True)):
            try:
                det = self.deterministic.compress_text(text, target_tokens)
            except Exception:
                det = {"success": False}
            if det.get("success") and float(det.get("confidence", 0.0)) >= float(self.config.get("deterministic", {}).get("compression_confidence", 0.88)):
                det["compression"] = {
                    "input_tokens_est": estimate_tokens(text),
                    "output_tokens_est": estimate_tokens(str(det.get("text", ""))),
                    "chunks": 0, "method": "deterministic-script", "ollama_calls": 0,
                }
                return det

        chunk_tokens = int(self.config.get("token_saving", {}).get("compression_chunk_tokens", 4800))
        chunk_chars = chars_for_tokens(chunk_tokens)
        chunks = [text[i:i + chunk_chars] for i in range(0, len(text), chunk_chars)]
        summaries: list[str] = []
        per_chunk_out = max(180, min(700, target_tokens // max(1, len(chunks)) + 120))
        general_model = str(self.config.get("models", {}).get("general", self.config.get("models", {}).get("fast_code", "qwen2.5-coder:1.5b")))
        for index, chunk in enumerate(chunks):
            package = build_prompt(
                operation="compress",
                model=general_model,
                profile=str(getattr(self, "config", {}).get("_hardware", {}).get("profile", "auto")),
                task=f"{instruction}\nChunk {index + 1}/{len(chunks)}",
                context=chunk,
            )
            result = self._generate(
                general_model, package.user,
                package.system,
                per_chunk_out, 0.1, tenant, "compress:map", int(args.get("priority", 4)),
                internal=True,
            )
            if not result.get("success"):
                return result
            summaries.append(str(result.get("text", "")))

        combined = "\n\n".join(summaries)
        if len(chunks) > 1 or estimate_tokens(combined) > target_tokens:
            package = build_prompt(
                operation="compress",
                model=general_model,
                profile=str(getattr(self, "config", {}).get("_hardware", {}).get("profile", "auto")),
                task=f"{instruction}\nTarget: <= {target_tokens} estimated tokens.",
                context=combined,
            )
            final = self._generate(
                general_model,
                package.user,
                package.system,
                target_tokens, 0.1, tenant, "compress:reduce", int(args.get("priority", 4)),
                internal=True,
            )
        else:
            final = {"success": True, "model": general_model, "text": combined}
        final["compression"] = {
            "input_tokens_est": estimate_tokens(text),
            "output_tokens_est": estimate_tokens(str(final.get("text", ""))),
            "chunks": len(chunks),
        }
        return self.artifacts.compact(final, tenant, "compress")

    def complete_code(self, args: dict[str, Any], tenant: str) -> dict[str, Any]:
        """Low-latency FIM (Fill-In-The-Middle) code completion for tab auto-complete."""
        prefix = str(args.get("prefix", ""))
        suffix = str(args.get("suffix", ""))
        max_tokens = min(256, max(8, int(args.get("max_tokens", 80))))
        model = str(self.config.get("models", {}).get("fast_code", "qwen2.5-coder:1.5b"))

        # Standard Qwen FIM prompt template
        prompt = f"<|fim_prefix|>{prefix[-3000:]}<|fim_suffix|>{suffix[:1500]}<|fim_middle|>"
        payload = {
            "model": model,
            "prompt": prompt,
            "raw": True,
            "options": {
                "num_predict": max_tokens,
                "temperature": 0.0,
                "stop": ["<|fim_prefix|>", "<|fim_suffix|>", "<|fim_middle|>", "<|endoftext|>", "<|file_separator|>"],
            },
        }
        cache_key = stable_hash({"complete_code": True, "model": model, "prompt": prompt, "max_tokens": max_tokens})
        def compute() -> dict[str, Any]:
            res = self.scheduler.submit(model, tenant, "complete_code", lambda: self.runtime.request("/api/generate", payload, timeout=6.0))
            if isinstance(res, dict) and "error" in res:
                return {"success": False, "error": res["error"]}
            completion = str(res.get("response", "")) if isinstance(res, dict) else ""
            return {"success": True, "model": model, "completion": completion}
        raw, hit, coalesced = self.generation_cache.get_or_compute(cache_key, compute)
        result = copy.deepcopy(raw)
        result["cache_hit"] = hit or coalesced
        return result

    DOMAIN_SYNONYMS = {
        "save": ["Save", "Store", "Persist", "Write", "Dump"],
        "load": ["Load", "Read", "Fetch", "Get", "Parse"],
        "database": ["SQLite", "Database", "ExecuteQuery", "Connection", "Transaction", "Query"],
        "auth": ["Auth", "Login", "Token", "User", "Session", "Authenticate", "Permission"],
        "api": ["Endpoint", "Route", "Handler", "Request", "Response", "Client"],
        "cache": ["Cache", "LRU", "Evict", "TTL", "Hit", "Miss"],
        "config": ["Config", "Settings", "Options", "Environment", "Params"],
        "test": ["Test", "Assert", "Fixture", "Mock", "Suite"],
        "logging": ["Logger", "Log", "Info", "Warning", "Error", "Debug"],
        "event": ["Event", "Emit", "Subscribe", "Publish", "Listener", "Handler"],
    }

    def _record_symbol_focus(self, tenant: str, symbols: list[str]) -> None:
        if not symbols:
            return
        with self._focus_lock:
            current = self._recent_focus_symbols.setdefault(tenant, [])
            for s in symbols:
                if s and len(s) >= 3 and s not in current:
                    current.insert(0, s)
            self._recent_focus_symbols[tenant] = current[:16]
            if len(self._recent_focus_symbols) > 256:
                for k in list(self._recent_focus_symbols.keys())[:-128]:
                    self._recent_focus_symbols.pop(k, None)

    def expand_query(self, query: str, tenant: str = "generic") -> list[str]:
        """Expand natural language query terms with code synonyms, domain keywords, and recent conversation focus."""
        cache_key = f"{tenant}:{query.strip().lower()}"
        if hasattr(self, "_query_expansion_l1"):
            cached = self._query_expansion_l1.get(cache_key)
            if cached is not None:
                return list(cached)
        terms = re.findall(r"[A-Za-z0-9_]{3,}", query.lower())
        synonyms: list[str] = list(terms)
        for t in terms:
            if t in self.DOMAIN_SYNONYMS:
                synonyms.extend(self.DOMAIN_SYNONYMS[t])
        
        # Multi-turn context bonus: if query is brief/contextual, add recently investigated AST symbols
        with self._focus_lock:
            recent = list(self._recent_focus_symbols.get(tenant, []))
        if recent and len(terms) <= 4:
            synonyms.extend(recent[:4])
            
        # Record newly identified proper-case or identifier symbols as focus
        explicit_symbols = [w for w in re.findall(r"\b[A-Z][A-Za-z0-9_]{3,}\b", query)]
        if explicit_symbols:
            self._record_symbol_focus(tenant, explicit_symbols)
            
        result = list(dict.fromkeys(synonyms))
        if hasattr(self, "_query_expansion_l1"):
            self._query_expansion_l1.set(cache_key, result)
        return result

    def generate_tests(self, payload: dict[str, Any], tenant: str) -> dict[str, Any]:
        """Generate complete unit test code for a symbol or file."""
        symbol = str(payload.get("symbol", ""))
        code_context = str(payload.get("code", ""))
        framework = str(payload.get("framework", "auto")).lower()
        path = str(payload.get("path", payload.get("file", "")))
        root = str(payload.get("root", "."))

        if not code_context and symbol and self.code_index is not None:
            try:
                sym_info = self.code_index.inspect_symbol(root, symbol)
                if sym_info.get("success"):
                    code_context = sym_info.get("snippet", "")
                    if not path:
                        path = sym_info.get("path", "")
            except Exception:
                pass

        if not framework or framework == "auto":
            if path.endswith(".cs"):
                framework = "nunit"
            elif path.endswith(".py"):
                framework = "pytest"
            elif path.endswith((".ts", ".js", ".tsx", ".jsx")):
                framework = "jest"
            else:
                framework = "pytest"

        if payload.get("fast_scaffold") or payload.get("template_only") or payload.get("deterministic"):
            clean_sym = re.sub(r"[^A-Za-z0-9_]", "_", symbol) or "target"
            if framework == "pytest":
                scaffold = f"import pytest\n\n\ndef test_{clean_sym}_basic():\n    # TODO: verify expected behavior for {symbol}\n    assert True\n"
            elif framework == "nunit":
                scaffold = f"using NUnit.Framework;\n\n[TestFixture]\npublic class {clean_sym}Tests {{\n    [Test]\n    public void Test_{clean_sym}_Basic() {{\n        Assert.Pass();\n    }}\n}}\n"
            elif framework == "jest":
                scaffold = f"describe('{symbol}', () => {{\n    test('basic functionality', () => {{\n        expect(true).toBe(true);\n    }});\n}});\n"
            else:
                scaffold = f"def test_{clean_sym}():\n    assert True\n"
            return {
                "success": True,
                "framework": framework,
                "target_symbol": symbol,
                "scaffold": scaffold,
                "test_code": scaffold,
                "method": "fast-scaffold",
            }

        prompt = (
            f"Generate high-quality unit tests using framework: {framework}.\n"
            f"Target symbol: {symbol}\n"
            f"Source code / context:\n```\n{code_context[:4000]}\n```\n\n"
            f"Requirements:\n"
            f"1. Write complete, compilable test methods with assertions.\n"
            f"2. Cover edge cases and normal flow.\n"
            f"3. Return ONLY clean source code inside a code block, no chat."
        )

        fast_model = str(self.config.get("models", {}).get("fast_code", "qwen2.5-coder:1.5b"))
        res = self._generate(
            fast_model,
            prompt,
            "You are an expert test engineer writing concise, robust unit tests.",
            512,
            0.1,
            tenant,
            "generate_tests",
            4,
        )
        if not res.get("success"):
            return res

        return {
            "success": True,
            "framework": framework,
            "target_symbol": symbol,
            "test_code": res.get("text", ""),
        }

    def validate_patch(self, payload: dict[str, Any], tenant: str) -> dict[str, Any]:
        """Validate whether a diff patch applies cleanly and preserves valid syntax."""
        diff_text = str(payload.get("patch", payload.get("diff", "")))
        if not diff_text.strip():
            return {"success": False, "error": "empty patch"}

        affected_files: list[str] = []
        errors: list[str] = []
        previous_old: str | None = None
        hunks = 0

        def patch_path(raw: str) -> str | None:
            value = raw.strip().split("\t", 1)[0].replace("\\", "/")
            if value == "/dev/null":
                return value
            if value.startswith(("a/", "b/")):
                value = value[2:]
            parts = value.split("/")
            if not value or value.startswith("/") or any(part in {"", ".", ".."} for part in parts):
                return None
            return value

        for line in diff_text.splitlines():
            if line.startswith("--- "):
                previous_old = patch_path(line[4:])
                if previous_old is None:
                    errors.append("invalid old patch path")
            elif line.startswith("+++ "):
                current = patch_path(line[4:])
                if previous_old is None or current is None:
                    errors.append("patch file header is malformed")
                elif current != "/dev/null":
                    affected_files.append(current)
                previous_old = None
            elif line.startswith("@@ "):
                if not re.match(r"^@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@", line):
                    errors.append("invalid hunk header")
                else:
                    hunks += 1

        if previous_old is not None:
            errors.append("patch has an unpaired file header")
        if not affected_files:
            errors.append("patch changes no files")
        if not hunks:
            errors.append("patch contains no hunks")
        valid = not errors
        applicability_checked = False
        if valid and payload.get("root"):
            root = Path(str(payload["root"])).expanduser().resolve()
            applicability_checked = True
            if not root.is_dir():
                errors.append("patch root directory does not exist")
            else:
                try:
                    timeout = 8.0 if self is None else max(1.0, min(30.0, float(self.config.get("commands", {}).get("patch_check_timeout_seconds", 8.0))))
                    checked = subprocess.run(
                        ["git", "-C", str(root), "apply", "--check", "--"], input=diff_text, capture_output=True,
                        timeout=timeout, check=False, **hidden_run_kwargs(text=True),
                    )
                    if checked.returncode != 0:
                        errors.append((checked.stderr or checked.stdout or "git apply --check failed").strip()[:500])
                except FileNotFoundError:
                    errors.append("git is unavailable for applicability check")
                except subprocess.TimeoutExpired:
                    errors.append("git apply --check timed out")
            valid = not errors

        return {
            "success": valid,
            "valid": valid,
            "syntax_valid": valid,
            "applicability_checked": applicability_checked,
            "files_affected": list(dict.fromkeys(affected_files)),
            "errors": errors,
            "message": f"Patch syntax {'is valid' if valid else 'is invalid'}; applicability {'was checked' if applicability_checked else 'was not checked'}.",
        }

    def preprocess(self, args: dict[str, Any], progress: Any = None) -> dict[str, Any]:
        def stage(name: str) -> None:
            if progress is not None:
                try:
                    progress(name)
                except Exception:
                    pass

        if self.preprocessor is None:
            return {"success": False, "error": "project preprocessor unavailable"}
        action = str(args.get("action", "status")).strip().lower().replace("-", "_")
        root = str(args.get("root", "."))
        if action in {"start", "register", "preprocess"}:
            stage("preprocess.register.start")
            result = self.preprocessor.register(root, source="agent", progress=progress)
            stage("preprocess.register.done")
            return result
        if action in {"refresh", "force_refresh"}:
            return self.preprocessor.refresh(root)
        if action == "pause":
            return self.preprocessor.pause(root if args.get("root") else None)
        if action == "resume":
            return self.preprocessor.resume(root if args.get("root") else None)
        if action in {"cancel", "stop"}:
            return self.preprocessor.cancel(root)
        if action in {"unregister", "remove", "delete"}:
            return self.preprocessor.unregister(root, purge_data=bool(args.get("purge_data", False)))
        if action in {"prune", "cleanup_deleted", "cleanup"}:
            return self.preprocessor.cleanup_deleted_projects()
        if action in {"cleanup_orphans", "cleanup_orphaned_indexes"}:
            return self.preprocessor.cleanup_orphaned_indexes()
        if action in {"status", "get"}:
            return self.preprocessor.status(root if args.get("root") else None)
        if action in {"lookup", "context"}:
            return self.preprocessor.lookup(root, str(args.get("query", "")), int(args.get("limit", 5)))
        return {"success": False, "error": f"unknown preprocess action: {action}"}

    def command(self, args: dict[str, Any], tenant: str) -> dict[str, Any]:
        if self.commands is None:
            return {"success": False, "error": "command broker unavailable"}
        action = str(args.get("action", "run")).strip().lower().replace("-", "_")

        stream_id = str(args.get("stream_id") or (f"cmd:{args.get('task_id')}" if args.get("task_id") else "") or (f"cmd:{tenant}" if args.get("stream") else ""))
        log_callback = None
        if stream_id and self.agent_state is not None and getattr(self.agent_state, "enabled", False):
            def _log_cb(st: str, line: str) -> None:
                try:
                    from .agent_events import AgentEvent
                    self.agent_state.append(AgentEvent.create(
                        stream_id=stream_id,
                        kind="command.log",
                        payload={"stream": st, "chunk": line},
                        actor=tenant,
                    ))
                except Exception:
                    pass
            log_callback = _log_cb

        if action == "patch_and_verify":
            return self.commands.patch_and_verify(
                str(args.get("patch", "")),
                str(args.get("cwd", args.get("root", "."))),
                tenant,
                command=str(args.get("command", "")),
                criterion=str(args.get("criterion", "")),
                task_id=str(args.get("task_id", "")),
                timeout=int(args.get("timeout", 0) or 0) or None,
                auto_rollback=bool(args.get("auto_rollback", True)),
                log_callback=log_callback,
            )
        if action == "preflight":
            raw_paths = args.get("paths") or args.get("files")
            paths_list = [str(p) for p in raw_paths] if isinstance(raw_paths, list) else ([str(raw_paths)] if raw_paths else None)
            return self.commands.preflight(
                str(args.get("cwd", args.get("root", "."))),
                tenant,
                paths=paths_list,
                timeout=int(args.get("timeout", 10) or 10),
            )
        if action in {"repair_loop", "auto_fix"}:
            return self.commands.repair_loop(
                str(args.get("command", "")), str(args.get("cwd", args.get("root", "."))), tenant,
                timeout=int(args.get("timeout", 0) or 0) or None,
                task_id=str(args.get("task_id", "")), criterion=str(args.get("criterion", "")),
                max_attempts=int(args.get("max_attempts", 3)),
                fix_generator=self._synthesize_repair_patch,
                log_callback=log_callback,
            )
        if action == "run":
            return self.commands.run(
                str(args.get("command", "")), str(args.get("cwd", args.get("root", "."))), tenant,
                timeout=int(args.get("timeout", 0) or 0) or None, force=bool(args.get("force", False)),
                task_id=str(args.get("task_id", "")), criterion=str(args.get("criterion", "")),
                auto_fix=bool(args.get("auto_fix", False)),
                fix_generator=self._synthesize_repair_patch if bool(args.get("auto_fix", False)) else None,
                log_callback=log_callback,
                snapshot=bool(args.get("snapshot", False)),
                rollback_on_failure=bool(args.get("rollback_on_failure", False)),
            )
        if action == "run_affected":
            root = str(args.get("cwd", args.get("root", ".")))
            aff = self.affected_tests(root, changed_paths=args.get("paths"))
            cmd = aff.get("suggested_command", "")
            if not cmd:
                return {
                    "success": True,
                    "message": "No affected tests found for current changes",
                    "changed_files": aff.get("changed_files", []),
                    "test_files": aff.get("test_files", []),
                }
            run_res = self.commands.run(
                cmd, root, tenant,
                timeout=int(args.get("timeout", 0) or 0) or None,
                force=bool(args.get("force", False)),
                task_id=str(args.get("task_id", "")),
                criterion=str(args.get("criterion", "")),
                log_callback=log_callback,
            )
            run_res["affected_tests_summary"] = aff
            return run_res
        if action == "format":
            root = str(args.get("cwd", args.get("root", ".")))
            return self.commands.format(root, paths=args.get("paths"), tenant=tenant, timeout=int(args.get("timeout", 0) or 0) or None)
        if action == "cancel":
            return self.commands.cancel(
                str(args.get("command", "")), str(args.get("cwd", args.get("root", "."))), tenant,
                execution_id=str(args.get("execution_id", "")),
            )
        if action == "classify":
            return {"success": True, "classification": self.commands.classify(str(args.get("command", "")))}
        if action == "stats":
            return {"success": True, "stats": self.commands.stats()}
        if action == "discover":
            root = str(args.get("root", args.get("cwd", ".")))
            profile = self.repo_profile(root)
            deterministic = self.deterministic_query(root, "test lint build validation commands dependencies", 40) if self.deterministic is not None else {}
            scripts = deterministic.get("scripts", []) if isinstance(deterministic, dict) else []
            return {
                "success": bool(profile.get("success")),
                "validation_commands": profile.get("validation_commands", []),
                "manifest_scripts": scripts[:20],
                "profile_cache": profile.get("workspace_cache"),
                "deterministic": bool(scripts),
            }
        if action == "lint_fix":
            return self.lint_fix(str(args.get("cwd", args.get("root", "."))), command=args.get("command"), paths=args.get("paths"), tenant=tenant, timeout=args.get("timeout"))
        if action == "spawn_daemon":
            return self.spawn_daemon(str(args.get("command", "")), str(args.get("cwd", args.get("root", "."))), name=str(args.get("name", args.get("task_id", ""))), env=args.get("env"))
        if action == "daemon_status":
            return self.daemon_status(args.get("daemon_id", args.get("task_id", args.get("command"))))
        if action == "stop_daemon":
            return self.stop_daemon(str(args.get("daemon_id", args.get("task_id", args.get("command", "")))))
        if action == "http_probe":
            exp_status = int(args.get("expected_status", 200)) if str(args.get("expected_status", "")).isdigit() else (int(args.get("task_id", 200)) if str(args.get("task_id", "")).isdigit() else 200)
            return self.http_probe(str(args.get("url", args.get("command", ""))), expected_status=exp_status, json_path=args.get("json_path", args.get("criterion")), timeout=float(args.get("timeout", 5.0) or 5.0), headers=args.get("headers"))
        if action == "stash_save":
            return self.stash_save(str(args.get("cwd", args.get("root", "."))), message=str(args.get("criterion", args.get("command", "local_ai_hub_stash")) or "local_ai_hub_stash"))
        if action == "stash_restore":
            return self.stash_restore(str(args.get("cwd", args.get("root", "."))))
        if action == "record_mock":
            return self.record_mock(str(args.get("command", "")), str(args.get("task_id", args.get("criterion", "default_cassette"))))
        if action == "replay_mock":
            return self.replay_mock(str(args.get("command", args.get("task_id", "default_cassette"))))
        if action == "diff_hunk_stage":
            return self.diff_hunk_stage(str(args.get("cwd", args.get("root", "."))), str(args.get("patch", args.get("command", ""))))
        if action in {"flaky_detect", "test_flaky_detect"}:
            return self.test_flaky_detect(str(args.get("cwd", args.get("root", "."))), str(args.get("command", "")), runs=int(args.get("runs", 5)), timeout=int(args.get("timeout", 30)))
        if action == "webhook_replay":
            return self.webhook_replay(str(args.get("url", args.get("command", ""))), args.get("payload", {}), secret=str(args.get("secret", "")), signature_header=str(args.get("signature_header", "X-Hub-Signature-256")), timeout=float(args.get("timeout", 10.0)))
        return {"success": False, "error": f"unknown command action: {action}"}

    def _synthesize_repair_patch(self, command: str, cwd: str, failure_result: dict[str, Any]) -> dict[str, str] | None:
        """Use verified remediation or one bounded local diagnostic for a failed command."""
        if not failure_result or failure_result.get("success"):
            return None
        rem = failure_result.get("remediation") or {}
        if rem.get("verified_fix") and isinstance(rem["verified_fix"], dict):
            return {str(k): str(v) for k, v in rem["verified_fix"].items()}
        features = self.config.get("features", {})
        if not isinstance(features, dict) or not bool(features.get("tasks", True)):
            return None
        if not rollout_feature_enabled(self.config, "local_diagnostic_dispatch"):
            failure_result["local_diagnostic_dispatch"] = {
                "available": False,
                "unsupported": True,
                "feature": "local_diagnostic_dispatch",
                "error": "local diagnostic dispatch is disabled (features.local_diagnostic_dispatch=false)",
            }
            return None
        summary = failure_result.get("failure_summary") or {}
        if not isinstance(summary, dict):
            summary = {}
        confidence = 0.95 if summary.get("path") and summary.get("line") else 0.70 if summary.get("path") else 0.25
        artifact_id = str(failure_result.get("artifact_id") or "")
        preview = str(failure_result.get("preview") or "")[:800]
        # Parsed failures are deterministic enough to avoid local inference. Local
        # diagnosis is read-only and never returns a mutation candidate; repair still
        # needs a verified deterministic fix.
        if confidence >= 0.50 or not artifact_id or not preview or not self._is_bounded_diagnostic_command(command):
            return None
        model = self.config.get("models", {}).get("fast_code", "qwen2.5-coder:1.5b")
        prompt = (
            f"Investigate this low-confidence command failure in {cwd}:\nCommand: {command}\n"
            f"Failure summary: {json_dumps(summary, sort_keys=True)}\n"
            f"Artifact reference: {artifact_id}\nFailure preview:\n{preview}\n"
            "Do not request or infer raw command logs; use only this bounded preview.\n"
            "Return only a short JSON diagnosis. Do not propose edits, patches, commands, architecture, or security work."
        )
        try:
            res = self.proxy_request("/api/generate", {
                "model": model,
                "prompt": prompt,
                "format": {
                    "type": "object",
                    "properties": {"diagnosis": {"type": "string"}},
                    "required": ["diagnosis"],
                    "additionalProperties": False,
                },
                "options": {"num_predict": 320},
            }, "hub", "local_ai_task", diagnostic_timeout_seconds=20)
            raw_text = str(res.get("response", "")).strip()
            parsed = json.loads(raw_text)
            diagnosis = str(parsed.get("diagnosis", "")).strip()[:600] if isinstance(parsed, dict) else ""
            if diagnosis:
                failure_result["local_diagnostic"] = diagnosis
        except Exception:
            pass
        return None

    @staticmethod
    def _is_bounded_diagnostic_command(command: str) -> bool:
        """Allow only direct, non-mutating validation commands for L1 diagnosis."""
        normalized = command.strip().lower()
        if not normalized or any(char in normalized for char in "|&;><\n\r"):
            return False
        mutation = r"(?:fix|write|apply|update|install|delete)"
        if re.search(rf"(?:^|\s)--{mutation}(?:[-_][a-z0-9]+)*(?:\s|=|$)", normalized):
            return False
        # `normalized` is lowercase, so this also rejects Jest's --updateSnapshot.
        if re.search(r"(?:^|\s)--(?:snapshot[-_]?update|update(?:[-_]?snapshot)?s?)(?:\s|=|$)", normalized):
            return False
        if re.search(rf"(?:^|\s)--(?:snapshot|golden|inline[-_]?snapshot)[-_]{mutation}(?:[-_][a-z0-9]+)*(?:\s|=|$)", normalized):
            return False
        if re.search(rf"(?:^|\s)--(?:snapshots?|goldens?|inline[-_]?snapshot)(?:\s|=){mutation}(?:\s|$)", normalized):
            return False
        if re.search(r"(?:^|\s)-u(?:\s|=|$)", normalized):
            return False
        python = r"(?:python(?:\.exe)?|py(?:\.exe)?)"
        patterns = (
            rf"^{python}\s+(?:-[a-z0-9_-]+\s+)*-m\s+(?:pytest|unittest|mypy|pyright|ruff|flake8)(?:\s|$)",
            r"^(?:pytest|unittest|mypy|pyright|ruff|flake8|eslint|tsc)(?:\s|$)",
            r"^(?:npm|pnpm|yarn)\s+test(?:\s|$)",
            r"^(?:go|cargo|dotnet|gradle|mvn)\s+test(?:\s|$)",
        )
        return any(re.match(pattern, normalized) is not None for pattern in patterns)

    @staticmethod
    def _proxy_timeout_seconds(config: dict[str, Any], diagnostic_timeout_seconds: float | None = None) -> float:
        """Keep normal proxy timeouts intact; cap only explicit diagnostic calls."""
        if diagnostic_timeout_seconds is not None:
            return min(60.0, max(1.0, float(diagnostic_timeout_seconds)))
        return max(1.0, float(config.get("server", {}).get("request_timeout_seconds", 300)))

    def proxy_request(
        self,
        endpoint: str,
        payload: dict[str, Any],
        tenant: str,
        source: str,
        *,
        diagnostic_timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        if payload.get("stream") is True:
            return {"success": False, "error": "streaming is intentionally disabled through the affinity queue"}
        model = str(payload.get("model") or self.config.get("models", {}).get("general", "qwen2.5-coder:1.5b"))
        clean = dict(payload)
        clean.pop("_hub_timeout_seconds", None)
        has_diagnostic_cap = diagnostic_timeout_seconds is not None
        request_timeout = self._proxy_timeout_seconds(self.config, diagnostic_timeout_seconds)
        clean["stream"] = False
        clean.setdefault("keep_alive", self.config.get("ollama", {}).get("keep_alive", "-1"))
        clean.pop("priority", None)
        proxy_profile = None
        semantic_query = ""
        if endpoint in {"/api/generate", "/api/chat"}:
            promptish = clean.get("prompt", clean.get("messages", []))
            if isinstance(promptish, str) and len(promptish.strip()) >= 8:
                semantic_query = promptish.strip()
            elif isinstance(promptish, list) and promptish:
                last_msg = promptish[-1]
                if isinstance(last_msg, dict) and isinstance(last_msg.get("content"), str):
                    semantic_query = last_msg.get("content", "").strip()

            input_tokens = estimate_tokens(json_dumps(promptish, ensure_ascii=False, default=str) if not isinstance(promptish, str) else promptish)
            opts = clean.get("options", {}) if isinstance(clean.get("options"), dict) else {}
            output_tokens = int(opts.get("num_predict", 0) or 0)
            clean, proxy_profile = self.model_policy.apply_payload(
                model, clean, role="proxy", input_tokens=input_tokens, output_tokens=output_tokens,
                preserve_explicit_think=True,
            )
            # Pre-flight context compaction & ceiling clamp
            if proxy_profile is not None and input_tokens > proxy_profile.prompt_budget_tokens:
                budget = proxy_profile.prompt_budget_tokens
                if isinstance(clean.get("prompt"), str):
                    p_text = clean["prompt"]
                    target_chars = max(500, int(budget * 3.5))
                    if len(p_text) > target_chars:
                        head_len = int(target_chars * 0.4)
                        tail_len = int(target_chars * 0.4)
                        clean["prompt"] = (
                            p_text[:head_len]
                            + "\n\n[...context compacted by Local AI Hub for model budget...]\n\n"
                            + p_text[-tail_len:]
                        )
                elif isinstance(clean.get("messages"), list) and len(clean["messages"]) > 2:
                    msgs = list(clean["messages"])
                    system_msgs = [m for m in msgs if m.get("role") == "system"]
                    other_msgs = [m for m in msgs if m.get("role") != "system"]
                    while other_msgs and estimate_tokens(json_dumps(system_msgs + other_msgs)) > budget:
                        if len(other_msgs) <= 1:
                            break
                        other_msgs.pop(0)
                    # KV-Cache prefix stabilization: ensure system message content is canonical and placed first
                    for sm in system_msgs:
                        if isinstance(sm.get("content"), str):
                            sm["content"] = sm["content"].replace("\r\n", "\n").strip()
                    clean["messages"] = system_msgs + other_msgs

        if self.semantic_cache.enabled and len(semantic_query) >= 8:
            sem_scope = stable_hash({"proxy": endpoint, "model": model, "system": clean.get("system", "")})
            sem_val, sem_score = self.semantic_cache.get(sem_scope, semantic_query)
            if isinstance(sem_val, dict):
                res = copy.deepcopy(sem_val)
                res["_local_ai_cache"] = {"hit": True, "coalesced": False, "layer": "semantic", "similarity": sem_score}
                if proxy_profile is not None:
                    res["_local_ai_execution"] = proxy_profile.cache_scope()
                return res

        key = stable_hash({"proxy": endpoint, "payload": clean, "app_version": __version__})

        def compute() -> dict[str, Any]:
            submit_kwargs: dict[str, Any] = {"priority": int(payload.get("priority", 5))}
            if has_diagnostic_cap:
                submit_kwargs["wait_timeout"] = request_timeout
            return self.scheduler.submit(
                model, tenant, source, lambda: self.runtime.request(endpoint, clean, timeout=request_timeout),
                **submit_kwargs,
            )

        raw, hit, coalesced = self.generation_cache.get_or_compute(key, compute)
        result = copy.deepcopy(raw)
        result["_local_ai_cache"] = {"hit": hit, "coalesced": coalesced, "layer": "exact" if hit else "single-flight" if coalesced else "ollama"}
        if proxy_profile is not None:
            result["_local_ai_execution"] = proxy_profile.cache_scope()

        if self.semantic_cache.enabled and len(semantic_query) >= 8 and isinstance(result, dict) and result.get("success", "error" not in result):
            try:
                sem_scope = stable_hash({"proxy": endpoint, "model": model, "system": clean.get("system", "")})
                self.semantic_cache.set(sem_scope, semantic_query, result)
            except Exception:
                pass

        return result

    def batch_delegate(self, args: dict[str, Any], tenant: str) -> dict[str, Any]:
        tasks = args.get("tasks", [])
        if not isinstance(tasks, list) or not tasks:
            return {"success": False, "error": "tasks must be a non-empty array"}
        tasks = tasks[:16]
        prepared: list[tuple[int, str, dict[str, Any]]] = []
        for i, raw in enumerate(tasks):
            item = raw if isinstance(raw, dict) else {"task": str(raw)}
            route = self.router.classify(str(item.get("task", "")), str(item.get("context", "")), str(item.get("task_type", "auto")), str(item.get("complexity", "auto")))
            prepared.append((i, route["model"], item))
        # Group work by intended model and start with the currently resident model when possible.
        try:
            active = self.scheduler.status().get("active_model")
        except Exception:
            active = None
        prepared.sort(key=lambda item: (0 if item[1] == active else 1, item[1], item[0]))
        results: dict[int, dict[str, Any]] = {}
        default_priority = int(args.get("priority", 5))
        for i, _model, item in prepared:
            item = dict(item)
            item.setdefault("priority", default_priority)
            results[i] = self.delegate(item, tenant)
        return {"success": all(r.get("success", False) for r in results.values()), "results": [results[i] for i in range(len(results))]}

    def optimize_databases(self) -> dict[str, Any]:
        """Perform WAL checkpointing, page pruning and VACUUM/optimize across all SQLite stores."""
        state_dir = Path(self.config["server"]["state_dir"])
        db_files = list(state_dir.glob("*.sqlite3"))
        optimized = []
        bytes_freed = 0
        for db in db_files:
            try:
                before_size = db.stat().st_size
                with closing(connect_sqlite(db, timeout_seconds=10.0)) as con:
                    con.execute("PRAGMA wal_checkpoint(TRUNCATE);")
                    con.execute("PRAGMA optimize;")
                after_size = db.stat().st_size
                freed = max(0, before_size - after_size)
                bytes_freed += freed
                optimized.append({"db": db.name, "before_kb": before_size // 1024, "after_kb": after_size // 1024, "freed_kb": freed // 1024})
            except Exception as exc:
                optimized.append({"db": db.name, "error": str(exc)})
        return {
            "success": True,
            "databases_optimized": len(optimized),
            "total_freed_kb": bytes_freed // 1024,
            "details": optimized,
        }

    def purge_stale_cache(self, days: int = 7) -> dict[str, Any]:
        """Purge cache entries older than N days to free disk space."""
        state_dir = Path(self.config["server"]["state_dir"])
        cutoff = time.time() - (days * 86400)
        cache_db = state_dir / "cache.sqlite3"
        deleted_entries = 0
        if cache_db.is_file():
            try:
                with closing(connect_sqlite(cache_db, timeout_seconds=10.0)) as con:
                    cur = con.execute("DELETE FROM cache_entries WHERE accessed_at < ?", (cutoff,))
                    deleted_entries = cur.rowcount
                    con.commit()
                    con.execute("PRAGMA wal_checkpoint(TRUNCATE);")
            except Exception as exc:
                return {"success": False, "error": str(exc)}
        return {"success": True, "purged_entries": deleted_entries, "days_threshold": days}

    def resolve_all_errors(self) -> dict[str, Any]:
        """Acknowledge and resolve all operational failures, crashes, errors, and agent incidents."""
        telemetry_res: dict[str, Any] = {}
        if self.telemetry:
            try:
                telemetry_res = self.telemetry.resolve_errors()
            except Exception as exc:
                telemetry_res = {"error": str(exc)}

        resolved_incidents = 0
        if self.incident_store:
            try:
                resolved_incidents = self.incident_store.resolve_all(
                    verified_fix="Resolved by operator",
                    root_cause="Operator manual resolve",
                )
            except Exception:
                resolved_incidents = -1

        try:
            state_dir = Path(self.config["server"]["state_dir"])
            sup_status_path = state_dir / "supervisor.status.json"
            if sup_status_path.exists():
                raw = json.loads(sup_status_path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    raw["restarts"] = 0
                    raw["last_error"] = ""
                    sup_status_path.write_text(json_dumps(raw, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass

        return {
            "success": True,
            "telemetry": telemetry_res,
            "resolved_incidents": resolved_incidents,
        }

    def run_doctor(self) -> dict[str, Any]:
        """Run comprehensive system, GPU, model, and database diagnostics."""
        from .gpu_monitor import get_gpu_telemetry
        from .llama_cpp_runtime import llama_cpp_backend_selection
        checks = []

        selected = llama_cpp_backend_selection(self.config)
        if selected == "disabled":
            checks.append({"component": "Inference Backend", "status": "OFF", "detail": "Ollama and llama.cpp are disabled."})
        elif selected == "ollama":
            online = self.runtime.is_online()
            checks.append({"component": "Inference Backend (Ollama)", "status": "OK" if online else "FAIL", "detail": f"Installed models: {len(self.runtime.installed_models())}"})
        else:
            online = self.runtime.llama_cpp.is_online()
            runtime_status = self.runtime.llama_cpp_managed.status() if selected == "llama.cpp (managed)" else {}
            mode = str(runtime_status.get("mode", "") or "device unreported")
            detail = f"Selected provider: {selected}; server {'online' if online else 'unavailable'}"
            if selected == "llama.cpp (managed)":
                detail += f"; execution device: {mode}"
                if not online and runtime_status.get("last_error"):
                    detail += f"; {runtime_status['last_error']}"
            checks.append({"component": "Inference Backend (llama.cpp)", "status": "OK" if online else "FAIL", "detail": detail})
        
        gpu = get_gpu_telemetry()
        checks.append({
            "component": "NVIDIA GPU",
            "status": "OK" if gpu.get("available") else "WARN",
            "detail": f"{gpu.get('gpu_name', 'N/A')} · {gpu.get('vram_used_mb', 0):.0f} / {gpu.get('vram_total_mb', 0):.0f} MB VRAM ({gpu.get('temperature_c', 0)}°C)"
        })
        
        state_dir = Path(self.config["server"]["state_dir"])
        dbs = list(state_dir.glob("*.sqlite3"))
        checks.append({"component": "SQLite Databases", "status": "OK", "detail": f"{len(dbs)} active databases in {state_dir.name}"})
        
        prep_status = self.preprocessor.status() if self.preprocessor else {}
        checks.append({"component": "Preprocessor", "status": "OK" if prep_status.get("enabled") else "OFF", "detail": f"{len(prep_status.get('projects', []))} projects tracked"})
        
        return {"success": True, "timestamp": time.time(), "checks": checks}

    def eval_suite(self, payload: dict[str, Any], tenant: str = "default") -> dict[str, Any]:
        """Run bounded local agent benchmark evaluation test cases."""
        suite_name = str(payload.get("suite_name", "default"))
        requested_model = str(payload.get("model") or getattr(self, "config", {}).get("models", {}).get("fast_code", "qwen2.5-coder:1.5b"))
        benchmark_kind = str(payload.get("benchmark_kind", "text") or "text").strip().lower()
        cases = payload.get("cases") or [
            {"id": "c1", "input": "def add(a, b):", "expected": "return a + b"},
            {"id": "c2", "input": "def is_even(n):", "expected": "return n % 2 == 0"},
        ]
        configured_roles = tuple(
            str(role)
            for role, model in (getattr(self, "config", {}).get("models", {}) or {}).items()
            if str(model or "") == requested_model and str(role) not in {"embedding", "embedding_backend", "embedding_device", "reranker", "reranker_backend", "reranker_device"}
        )
        vision_only = bool(configured_roles) and all(role == "vision" for role in configured_roles)
        if vision_only and benchmark_kind != "vision":
            mismatch_cases = []
            for case in cases:
                case_id = str(case.get("id", "case"))
                mismatch_cases.append({
                    "case_id": case_id,
                    "model": requested_model,
                    "status": "role_mismatch",
                    "passed": False,
                    "error": f"model is configured only for vision role: {', '.join(configured_roles)}",
                    "response_chars": 0,
                    "response_tokens_est": 0,
                    "response_sha256": hashlib.sha256(b"").hexdigest(),
                    "response_preview": "",
                    "response_preview_truncated": False,
                    "duration_ms": 0.0,
                })
            return {
                "success": True,
                "suite": suite_name,
                "model": requested_model,
                "model_roles": list(configured_roles),
                "benchmark_kind": benchmark_kind,
                "summary": {
                    "total": len(cases),
                    "passed": 0,
                    "failed": len(cases),
                    "pass_rate": 0.0,
                    "execution_failures": 0,
                    "benchmark_valid": False,
                    "role_mismatch": True,
                },
                "cases": mismatch_cases,
            }
        results = []
        passed = 0
        execution_failures = 0
        for case in cases:
            c_id = str(case.get("id", "case"))
            c_in = str(case.get("input", ""))
            c_exp = str(case.get("expected", ""))
            started = time.perf_counter()
            try:
                response = self.runtime.request("/api/generate", {
                    "model": requested_model,
                    "prompt": c_in,
                    "system": "Return the shortest correct answer containing the requested completion. Output only the answer; no greeting, explanation, or generic advice.",
                    "stream": False,
                    # Evaluation cases measure visible task output, not hidden
                    # reasoning.  Without this, Qwen3.5 can consume the whole
                    # small eval budget in `thinking` and look like a bad model.
                    "think": False,
                    "options": {"num_predict": 128, "temperature": 0.0},
                })
                response = response if isinstance(response, dict) else {"error": "runtime returned a non-object response"}
                error = str(response.get("error", "") or "")
                response_text = str(response.get("response", response.get("text", "")) or "")
                message = response.get("message")
                if not response_text and isinstance(message, dict):
                    response_text = str(message.get("content", "") or "")
                is_pass = bool(c_exp and not error and c_exp in response_text)
                if error:
                    execution_failures += 1
                    lowered = error.lower()
                    status = "unavailable" if any(marker in lowered for marker in ("404", "not found", "unavailable", "no route")) else "error"
                else:
                    status = "passed" if is_pass else "failed"
                lowered_text = response_text.casefold()
                generic_markers = ("certainly!", "below is", "here is", "this function", "hope this helps")
                generic_filler = any(marker in lowered_text for marker in generic_markers)
                concise_limit = max(240, len(c_exp) * 6)
                style_quality = {
                    "contains_expected": is_pass,
                    "exact_expected": response_text.strip() == c_exp.strip(),
                    "generic_filler": generic_filler,
                    "concise": len(response_text.strip()) <= concise_limit,
                    "quality_pass": bool(is_pass and not generic_filler and len(response_text.strip()) <= concise_limit),
                }
                result = {
                    "case_id": c_id,
                    "model": requested_model,
                    "passed": is_pass,
                    "status": status,
                    "response_chars": len(response_text),
                    "response_tokens_est": estimate_tokens(response_text),
                    "response_sha256": hashlib.sha256(response_text.encode("utf-8")).hexdigest(),
                    "response_preview": response_text[:240],
                    "response_preview_truncated": len(response_text) > 240,
                    "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                    "style_quality": style_quality,
                }
                if error:
                    result["error"] = error
            except Exception as exc:
                is_pass = False
                execution_failures += 1
                result = {
                    "case_id": c_id,
                    "model": requested_model,
                    "passed": False,
                    "status": "error",
                    "error": str(exc),
                    "response_chars": 0,
                    "response_tokens_est": 0,
                    "response_sha256": hashlib.sha256(b"").hexdigest(),
                    "response_preview": "",
                    "response_preview_truncated": False,
                    "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                }
            results.append(result)
            if is_pass:
                passed += 1

        quality_passed = sum(
            1 for item in results
            if isinstance(item, dict) and bool((item.get("style_quality") or {}).get("quality_pass"))
        )
        return {
            "success": True,
            "suite": suite_name,
            "model": requested_model,
            "model_roles": list(configured_roles),
            "benchmark_kind": benchmark_kind,
            "summary": {
                "total": len(cases),
                "passed": passed,
                "failed": len(cases) - passed,
                "pass_rate": round(passed / max(1, len(cases)), 2),
                "execution_failures": execution_failures,
                "benchmark_valid": True,
                "role_mismatch": False,
                "quality_passed": quality_passed,
                "quality_pass_rate": round(quality_passed / max(1, len(cases)), 2),
            },
            "cases": results,
        }

    def prompt_eval(self, payload: dict[str, Any], tenant: str = "default") -> dict[str, Any]:
        """Evaluate and render dynamic prompt templates with variable substitution and token estimation."""
        template = str(payload.get("template", ""))
        variables = payload.get("variables", {}) if isinstance(payload.get("variables"), dict) else {}
        rendered = template
        for k, v in variables.items():
            rendered = rendered.replace(f"{{{{{k}}}}}", str(v))
            rendered = rendered.replace(f"{{{k}}}", str(v))

        token_est = estimate_tokens(rendered)
        return {
            "success": True,
            "template": template,
            "rendered_prompt": rendered,
            "token_estimate": token_est,
            "variables_applied": list(variables.keys()),
        }

    def eval_drift(self, payload: dict[str, Any], tenant: str = "default") -> dict[str, Any]:
        """Track accuracy and regression drift across benchmark runs."""
        suite_name = str(payload.get("suite_name", "default"))
        history_file = Path(self.state_dir) / "eval_history.json"
        
        history: dict[str, Any] = {}
        if history_file.is_file():
            try:
                history = json.loads(history_file.read_text(encoding="utf-8"))
            except Exception:
                history = {}

        if "current" in payload and isinstance(payload["current"], dict):
            current = payload["current"]
        else:
            current = self.eval_suite(payload, tenant)

        current_summary = current.get("summary", {})
        curr_rate = float(current_summary.get("pass_rate", 0.0))
        
        prev_entry = history.get(suite_name)
        if prev_entry:
            prev_rate = float(prev_entry.get("summary", {}).get("pass_rate", 0.0))
            delta = round(curr_rate - prev_rate, 3)
            if delta > 0.01:
                status = "improved"
            elif delta < -0.01:
                status = "regressed"
            else:
                status = "stable"
        else:
            prev_rate = curr_rate
            delta = 0.0
            status = "baseline"

        history[suite_name] = {
            "timestamp": time.time(),
            "summary": current_summary,
            "cases": current.get("cases", []),
        }
        try:
            from .process_utils import atomic_write_file
            atomic_write_file(history_file, json_dumps(history, indent=2))
        except Exception:
            pass

        return {
            "success": True,
            "suite": suite_name,
            "status": status,
            "current_pass_rate": curr_rate,
            "previous_pass_rate": prev_rate,
            "drift_delta": delta,
            "summary": current_summary,
        }
