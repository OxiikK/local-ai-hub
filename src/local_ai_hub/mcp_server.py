from __future__ import annotations

import atexit
import contextvars
import functools
import hashlib
import inspect
import os
import queue
import re
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Literal, TypeAlias
from urllib.parse import quote

from local_ai_hub.client import HubClient
from local_ai_hub.compact import compact_result
from local_ai_hub.response_budget import (
    _CONTEXT_PROTECTED_KEYS,
    budget_response,
    delta_response,
    result_id,
)
from local_ai_hub.context_ledger import ContextLedger
from local_ai_hub.budget import RootFamilyBudget
from local_ai_hub.projection import AgentProjector
from local_ai_hub.config import load_config
from local_ai_hub.features import FeatureSet
from local_ai_hub.ollama_subagents import OllamaSubagentCatalog
from local_ai_hub.process_utils import canonical_root, is_rooted_path
from local_ai_hub.token_accounting import account_projection, attach_accounting, context_digest, finalize_tool_accounting, json_tokens, pop_accounting
from local_ai_hub.trace_context import efficiency_metadata, reset_metadata, set_metadata
from local_ai_hub.adoption_metrics import AdoptionMetricsStore
from local_ai_hub.routing import semantic_handoff_hint
from local_ai_hub.state_paths import configured_state_dir
from local_ai_hub.semantic_quality import assess_semantic_result
from local_ai_hub.evidence_contract import evidence_meta

try:
    from mcp.server.fastmcp import FastMCP
    import mcp.types as mcp_types
except ImportError:
    FastMCP = None  # type: ignore[assignment,misc]
    mcp_types = None  # type: ignore[assignment]

class _MissingMCP:
    """Import-safe stand-in used only when the optional MCP SDK is unavailable.

    Keeping module import side-effect free lets diagnostics, tests, and packaging
    tooling inspect the MCP surface without requiring the transport dependency.
    Actual server execution still fails fast with an actionable dependency error.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self.tools: list[Any] = []

    def tool(self):
        def decorator(fn: Any) -> Any:
            self.tools.append(fn)
            return fn

        return decorator

    def run(self) -> None:
        raise SystemExit("Missing MCP dependency. Re-run setup or install requirements-core.txt")

# Module-level initialisation: FastMCP requires tool decorators at import
# time, so HubClient, config and projectors must be created here. Any failure
# produces a clear SystemExit instead of a confusing AttributeError later.
AGENT_NAME = os.environ.get("LOCAL_AI_AGENT", "agent")
_workspace = canonical_root(Path.cwd())
_workspace_hash = hashlib.sha1(_workspace.encode("utf-8"), usedforsecurity=False).hexdigest()[:8]
TENANT = os.environ.get("LOCAL_AI_TENANT") or f"{AGENT_NAME}:{_workspace_hash}"

try:
    CLIENT = HubClient(tenant=TENANT)
    CFG = load_config(os.environ.get("LOCAL_AI_CONFIG"))
    FEATURES = FeatureSet.from_config(CFG)
    MCP_CFG = CFG.get("mcp", {})
    PROJECTOR = AgentProjector(CFG)
    PROFILE_CATALOG = OllamaSubagentCatalog(CFG)
    MAX_TEXT = int(MCP_CFG.get("compact_max_text_chars", 6000))
    MAX_EVIDENCE = int(MCP_CFG.get("compact_max_evidence", 16))
    LEAN_SCHEMAS = bool(MCP_CFG.get("lean_schemas", True))
    RESPONSE_BUDGET_CFG = MCP_CFG.get("response_budget", {}) if isinstance(MCP_CFG.get("response_budget", {}), dict) else {}
    RESPONSE_BUDGET_ENABLED = bool(RESPONSE_BUDGET_CFG.get("enabled", True))
    RESPONSE_DEFAULT_TOKENS = max(128, int(RESPONSE_BUDGET_CFG.get("default_tokens", 3200)))
    RESPONSE_TOOL_TOKENS = RESPONSE_BUDGET_CFG.get("tool_tokens", {}) if isinstance(RESPONSE_BUDGET_CFG.get("tool_tokens", {}), dict) else {}
    RESPONSE_REUSE_LIMIT = max(16, int(RESPONSE_BUDGET_CFG.get("reuse_cache_size", 256)))
    CONTEXT_LEDGER = ContextLedger(max_entries=int(RESPONSE_BUDGET_CFG.get("ledger_size", 512)))
    ROOT_FAMILY_BUDGET = RootFamilyBudget(
        default_limit=int(CFG.get("workflow", {}).get("root_family_token_budget", 0) or 0),
        max_active_scopes=int(CFG.get("workflow", {}).get("root_family_max_active_scopes", 1) or 1),
    )
    ADOPTION_METRICS = AdoptionMetricsStore(configured_state_dir(CFG))
except Exception as _init_exc:  # pragma: no cover
    import sys
    print(f"[local-ai-hub] MCP server init failed: {_init_exc}", file=sys.stderr)
    raise SystemExit(1) from _init_exc


def _timeout(kind: str) -> float:
    defaults = {"quick": 180.0, "context": 600.0, "model": 1800.0, "long": 2400.0}
    return max(1.0, float(MCP_CFG.get(f"{kind}_timeout_seconds", defaults[kind])))


def _client_root(r: str = "") -> str:
    cleaned = str(r or "").strip()
    if not cleaned or cleaned == ".":
        return _workspace
    try:
        p = Path(cleaned)
        if not is_rooted_path(cleaned):
            return canonical_root(Path(_workspace) / p)
        # Preserve foreign-platform rooted paths. Calling Path.resolve() on a
        # Windows drive path while running on POSIX would incorrectly prefix cwd.
        if not p.is_absolute():
            return cleaned.replace("\\", "/")
        return canonical_root(p)
    except Exception:
        return _workspace


_ACTION_ALIASES: dict[str, dict[str, str]] = {
    "coord": {
        "ctx": "context_compile",
        "compile_ctx": "context_compile",
        "neg_rec": "negative_knowledge_record",
        "neg_find": "negative_knowledge_find",
        "mem_put": "memory_record",
        "mem_rec": "memory_record",
        "mem_get": "memory_get",
        "mem_find": "memory_find",
        "mem_search": "memory_find",
        "task_new": "task_create",
        "task_chk": "task_checkpoint",
        "task_done": "task_complete",
    },
    "command": {
        "patch": "patch_and_verify",
        "fix": "auto_fix",
        "repair": "repair_loop",
        "fmt": "format",
    },
    "repo": {
        "idx": "code_index",
        "review": "review_diff",
        "sec": "security_audit",
        "ctx": "context",
    },
    "task": {
        "gen": "delegate",
        "rev": "review",
        "del": "delegate",
    },
}


def _resolve_action(tool_kind: str, action: str) -> str:
    cleaned = str(action or "").strip().lower().replace("-", "_")
    tool_aliases = _ACTION_ALIASES.get(tool_kind, {})
    return tool_aliases.get(cleaned, cleaned)


def _normalize_deterministic(val: Any) -> Any:
    """Sort dictionary keys and normalize microsecond timestamps to maintain stable prompt caches."""
    if isinstance(val, dict):
        out = {}
        for k in sorted(val.keys()):
            out[k] = _normalize_deterministic(val[k])
        return out
    if isinstance(val, list):
        return [_normalize_deterministic(x) for x in val]
    return val


mcp = FastMCP("Local AI Hub (compact)") if FastMCP is not None else _MissingMCP("Local AI Hub (compact)")


# ---------------------------------------------------------------------------
# Dynamic tool description builders — read FEATURES (from config) at import
# time so tool schemas reflect only the active backends.
# ---------------------------------------------------------------------------

def _actions_note(actions: list[str]) -> str:
    """Expose the action surface supported by this installation's feature gates."""
    return f" Supported actions: {', '.join(actions)}." if actions else " No actions are enabled."


def _specialized_note(tool_name: str) -> str:
    """Expose enabled feature-specific triggers alongside compact tool names."""
    notes = [line[2:] for line in FEATURES.specialized_trigger_lines() if f"`{tool_name}" in line]
    return f" Specialized triggers: {'; '.join(notes)}." if notes else ""


def _desc_status() -> str:
    if LEAN_SCHEMAS:
        return "Health, cache, telemetry, and agent_state inspector. Aggregate-bounded responses; optional max_response_tokens, response_profile, reuse_key. Actions: brief, cache, telemetry, full, agent_state."
    ollama_note = " ollama_online," if FEATURES.ollama else ""
    agent_state_note = " With Agent OS enabled, `detail=agent_state` summarizes durable task state." if FEATURES.agent_os else ""
    return (
        "Health/queue/token-saving status. detail: brief, cache, telemetry, full, agent_state. scope: process (default) or window."
        f" Telemetry is metadata-only.{ollama_note}"
        f"{agent_state_note}"
        f"{_specialized_note('local_ai_status')}"
        " Do not poll status during normal repository work or while preprocessing/model startup is in progress;"
        " one bounded health check is enough before native fallback."
        " Use when: make one bounded health, cache, or telemetry check."
        " Skip when: repository evidence or task work is needed."
    )


def _semantic_handoff_contract(enabled: bool, status_enabled: bool = True) -> str:
    if not enabled:
        if not status_enabled:
            return (
                " Semantic handoff is unavailable because local inference is disabled."
                " If a permitted boundary excludes local inference, state the permitted bypass explicitly"
                " without using unavailable status tooling."
            )
        return (
            " Semantic handoff is unavailable because local inference is disabled."
            " If a permitted boundary excludes local inference, report the bypass through"
            " `local_ai_status(adoption_signal=\"bypassed\", target_tool=\"local_ai_task\", target_action=\"reason\")`."
        )
    bypass = (
        " If local inference is unavailable or intentionally excluded by a permitted boundary, report the bypass through"
        " `local_ai_status(adoption_signal=\"bypassed\", target_tool=\"local_ai_task\", target_action=\"reason\")`."
        if status_enabled
        else
        " If local inference is unavailable or intentionally excluded by a permitted boundary, report the permitted bypass explicitly"
        " without using unavailable status tooling."
    )
    return (
        " Semantic handoff is mandatory: after deterministic/indexed evidence, planning, interpretation,"
        " synthesis, generation, review, compression, and second-opinion work must call `local_ai_task`"
        " before cloud reasoning. The cloud agent integrates the bounded local result and does not redo"
        " semantic work."
        + bypass
        + " Preserve exceptions for architecture, security, mutations, open-ended coding,"
        " exact evidence, and verification."
    )


def _desc_task() -> str:
    local_enabled = FEATURES.tasks and FEATURES.has_any_model()
    if not local_enabled:
        return (
            "Local inference is disabled on this installation for this tool."
            " Returns unsupported=true for local-model actions."
            " Semantic generation, exploration, reasoning, review, independent second opinions, and semantic compression are unavailable until a local model backend is enabled."
            " Use deterministic/indexed evidence only."
        )
    if LEAN_SCHEMAS:
        return (
            "Bounded local-model worker for semantic generation, exploration, reasoning, review, independent second opinions, and semantic compression. "
            "Use deterministic/indexed tools for exact facts, symbols, diff and tests; they are not substitutes for these semantic tasks. "
            f"{_semantic_handoff_contract(local_enabled, FEATURES.status)}"
            "Command failure diagnosis remains disabled by default; never pass raw logs or open-ended coding. "
            "Local AI Hub does not route or manage native Codex agents. "
            f"{_actions_note(FEATURES.supported_task_actions())}"
        )
    profile_note = ""
    if FEATURES.subagents and FEATURES.subagent_profiles:
        profile_names = ", ".join(f"`{p}`" for p in FEATURES.subagent_profiles)
        profile_note = (
            f" Named advisory profiles ({profile_names}) use Local AI Hub read-only tooling"
            " directly when `root` is supplied."
        )
    return (
        f"Bounded local-model worker for the main agent."
        f" Route preprocessing through `{FEATURES.background_model}`, quick tasks through `{FEATURES.fast_model}`, complex tasks through `{FEATURES.smart_model}`, and the hardest reasoning through `{FEATURES.reasoning_model}`."
        f" Explicit model overrides must match a configured model tag."
        f"{profile_note}"
        " Use deterministic/indexed tools for exact facts, symbols, diff and tests; use this worker for semantic generation, exploration, reasoning, review, independent second opinions and semantic compression after any needed evidence."
        f"{_semantic_handoff_contract(local_enabled, FEATURES.status)}"
        " Command failure diagnosis is disabled by default; enable `features.local_diagnostic_dispatch=true` only for one local diagnostic after low-confidence deterministic command parsing with an artifact reference and narrow preview, never raw logs."
        " Never automatically dispatch local inference for architecture, security, mutations, or open-ended coding."
        " Use it for bounded generation, exploration, reasoning, boilerplate, review, independent second opinions and semantic compression after any needed indexed evidence."
         " Local AI Hub does not route or manage native Codex subagents; those are managed directly by Codex outside Local AI Hub."
        f"{_actions_note(FEATURES.supported_task_actions())}"
        f"{_specialized_note('local_ai_task')}"
        " `delivery=sync` preserves the foreground contract; `async` returns a durable job now;"
        " `auto` requires a positive `latency_budget_ms` and only defers after sufficient endpoint history shows p95 above it."
        " Evaluation stores only opaque ids, booleans, and numeric metadata; it never stores prompts or source text."
        " Async wait is bounded to 90 seconds; never poll loops."
        " Start a short-lived conversation with `conversation=true` on delegate or reason, then use `continue` with its opaque `conversation_id`; conversations are sync-only and process-memory only."
        " Use when: bounded local generation, compression, review, routing, second opinion, or named advisory subagent work is needed."
        " Skip when: task is trivial, pure evidence lookup, security/privacy constrained, no useful independent scope,"
        " or Codex-owned subagent orchestration is the right owner."
    )


def _desc_repo() -> str:
    local_enabled = FEATURES.tasks and FEATURES.has_any_model()
    if not local_enabled:
        return (
            "Primary repository worker for repository navigation, symbols, and impact."
            " Local inference is disabled on this installation."
            " Semantic generation, reasoning, review, independent second opinions, and compression are unavailable until a local model backend is enabled."
            " Use deterministic/indexed evidence only for exact facts, symbols, diff, tests, and verification."
            " Native fallback requires terminal=true and retryable=false."
        )
    if LEAN_SCHEMAS:
        return "Primary repository worker for repository navigation, symbols, and impact with aggregate-bounded responses; optional max_response_tokens, response_profile, reuse_key. `context` is the default adaptive context pack before non-trivial planning, edit, review or test; it accepts guarded task/phase/focus fields, requires evidence IDs and reuse candidates first, treats deterministic/indexed evidence as authoritative, limits local models to ranking/compression of structured evidence, and requires `override_reason` plus approval when requested. Raw model/debug fields stay omitted unless requested through extra_fields. Omit guarded fields to preserve legacy fast/full behavior. Use deterministic/indexed actions for exact facts, symbols, diff and tests; use `local_ai_task` for semantic generation, reasoning, review, independent second opinions and compression. `solve` preserves one bounded local pass for explicit semantic requests even when exact evidence is strong. Native fallback requires terminal=true and retryable=false. Actions: search, code_index, context, solve, review_diff, symbols, callers, dead_code."
    semantic_hint = ""
    if FEATURES.has_semantic():
        semantic_hint = f" -> {FEATURES.semantic_hint()} for relationships"
    model_hint = ""
    if FEATURES.has_any_model():
        model_hint = f" -> {FEATURES.fast_model} -> smart model"
    rag_hint = " -> RAG" if FEATURES.rag else ""
    return (
        "Primary bounded repository worker for repository navigation, symbols, and impact."
        " CALL THIS BEFORE broad repository reads/searches for any non-trivial repo task. MANDATORY GATE."
        f" Use deterministic, code_index/search,{' ' + FEATURES.semantic_hint() + ',' if FEATURES.has_semantic() else ''}"
        " context and solve for bounded evidence and implementation support."
        " `context` is the default adaptive context pack before non-trivial planning, edit, review or test; reuse existing evidence and reuse candidates first, require evidence IDs, treat deterministic/indexed evidence as authoritative, and limit local models to ranking/compression of structured evidence. Guarded overrides require `override_reason` and approval when requested. Raw model/debug fields stay omitted unless requested through `extra_fields`; omitting guarded fields preserves legacy fast/full behavior."
        " For implementation, diagnosis, refactoring or complex review, call `solve` after evidence and before native edits."
        f"{' When generating, use `' + FEATURES.fast_model + '` for quick tasks, `' + FEATURES.smart_model + '` for complex work, and `' + FEATURES.reasoning_model + '` for hardest reasoning.' if local_enabled else ''}"
        " `review_diff` and `security_audit` are targeted local checks."
        f"{'  After any needed indexed evidence, use `local_ai_task` for semantic generation, reasoning, review, independent second opinions and compression. `solve` preserves one bounded local pass for explicit semantic requests even when exact evidence is strong.' if local_enabled else ''}"
        f"{_semantic_handoff_contract(local_enabled, FEATURES.status)}"
        " Codex separately decides whether to use native Codex subagents;"
        " Local AI Hub does not route or manage those agents."
        " On first use of a stable absolute root call action=preprocess exactly once and continue immediately;"
        " never poll/wait/force-refresh preprocessing."
        f" Cheapest sufficient path for repository evidence: deterministic -> code_index/search{semantic_hint} -> context/solve{rag_hint}; semantic generation, reasoning, review, independent second opinions and compression use `local_ai_task` after any needed evidence;"
        " STOP as soon as a cheaper layer is sufficient and never fan out overlapping retrieval layers for the same question."
        " Reuse fresh evidence/artifact slices and never repeat an identical root/query/action while repo state is unchanged."
        " `in_progress` means another owner is doing identical work; retryable/429/503 means back off;"
        " degraded/stale means verify only the affected slice."
        " Native fallback requires terminal=true and retryable=false."
        " Always pass the stable absolute project root; never rely on MCP cwd. Never loop or increase timeouts indefinitely."
        " Use when: every non-trivial repository task needs indexed evidence or a bounded Hub operation."
        " Skip when: the task is not repository-scoped or fresh evidence already answers it and no independent Hub scope exists."
        f"{_actions_note(FEATURES.supported_repo_actions())}"
        f"{_specialized_note('local_ai_repo')}"
    )



def _desc_rag() -> str:
    if LEAN_SCHEMAS:
        return "Semantic RAG & documentation search with aggregate-bounded responses; optional max_response_tokens, response_profile, reuse_key. Actions: search, index, docset_search, ingest_document."
    if not FEATURES.rag:
        return (
            "Semantic retrieval — RAG backend is disabled on this installation (`features.rag=false`)."
            " All actions return unsupported=true. Enable RAG in config.toml to activate."
            " Skip when: always (RAG disabled). Use when: never."
        )
    return (
        "Fallback semantic retrieval for the main agent only after deterministic, code_index, search"
        " and preprocessed evidence are insufficient."
        " Keep queries bounded and use it only for bounded semantic retrieval, not open-ended agent orchestration."
        f"{_actions_note(FEATURES.supported_rag_actions())}"
        f"{_specialized_note('local_ai_rag')}"
        " Index is file-incremental and query-cached."
        " Do not trigger index repeatedly for a fresh stable workspace."
        " Use when: cheaper indexed repository paths cannot answer a bounded semantic retrieval question."
        " Skip when: deterministic/indexed evidence is sufficient or Codex-owned subagent orchestration is the right owner."
    )


def _desc_command() -> str:
    if LEAN_SCHEMAS:
        return "Safe CLI command broker for test, lint, typecheck, or build commands with aggregate-bounded responses; optional max_response_tokens, response_profile, reuse_key. Native fallback requires terminal=true and retryable=false. Mutations never cache or single-flight. Speculative lint is opt-in, read-only, debounced, changed-path scoped, and cancellable. Actions: run, auto_fix, format, patch_and_verify, repair_loop, speculative_lint."
    agent_os_note = " Optional task_id and criterion link passing validation commands directly to evidence-backed VerificationReceipts." if FEATURES.agent_os else ""
    return (
        "Bounded broker for test, lint, typecheck, or build commands; also repeatable analysis/read-only commands."
        " Shared safe CLI broker."
        f"{_actions_note(FEATURES.supported_command_actions())}"
        f"{_specialized_note('local_ai_command')}"
        f"{agent_os_note}"
        " Results are keyed by command + bounded repo state and duplicate runs coalesce across agents. Reuse fresh results."
        " Mutations never cache or single-flight."
        " If run returns in_progress=true, DO NOT start the command natively or with force; continue independent work and retry later so the owner can populate the cache."
        " Native fallback requires terminal=true and retryable=false."
        " cancel only stops an active matching command. force=true is exceptional recovery/admin behavior, never a retry button."
        " Use when: a repeatable test, lint, typecheck, build, analysis, or safe read-only command is needed."
        " Skip when: no command is needed or a fresh cached result already answers it."
    )


def _desc_coord() -> str:
    if LEAN_SCHEMAS:
        return "Agent OS coordination for task contracts, ownership, checkpoints, and verification receipts with aggregate-bounded responses; optional max_response_tokens, response_profile, reuse_key. Actions: claim, release, memory_record, memory_find, context_compile, task_create, task_checkpoint."
    if FEATURES.agent_os:
        agent_os_note = (
            " For non-trivial multi-step, long-running, delegated, or acceptance-criteria work, create an Agent OS task contract first;"
            " checkpoint meaningful phases, search/record durable memory as useful, compile context when resuming,"
            " and gate task completion on verification receipts."
        )
        usage_note = (
            " Use when: any non-trivial task benefits from durable state/verification, or Hub workers share edit paths or reusable findings."
            " Skip when: task is a trivial one-step lookup with no shared state to preserve."
        )
    else:
        agent_os_note = ""
        usage_note = (
            " Use when: Hub workers share edit paths, leases, or reusable findings."
            " Skip when: work is isolated and no shared Hub state or memo is involved."
        )
    return (
        "Cross-agent coordination for ownership, checkpoints, and verification receipts."
        f"{_actions_note(FEATURES.supported_coord_actions())}"
        f"{agent_os_note}"
        " Claim overlapping edit paths before concurrent Hub work."
        " Search/get memos before repeating expensive investigation and store concise reusable findings after discovery."
        " Native peer subagents are coordinated by Codex rather than by this Hub tool."
        f"{usage_note}"
    )


def _desc_work() -> str:
    if LEAN_SCHEMAS:
        return "Delegate closed, low-risk work with a verified handoff and aggregate-bounded response; optional max_response_tokens, response_profile, reuse_key. Skip micro-edits and live discussion. Actions: submit, status, wait, get, cancel."
    return (
        "Delegate one closed, low-risk work item with a verified handoff to Local AI Hub: plan a bounded dependency DAG, execute the smallest independently verifiable steps, "
        "apply transactional leased edits, run safe validation, verify the integrated result against the original request, and return a compact handoff. "
        "Actions: submit, status, wait, get, cancel, continue. response_profile=minimal|compact|standard|debug; return_fields selects only needed top-level fields; "
        "max_output_tokens bounds the handoff while full details remain artifact-backed. Use when: the task can be delegated as a self-contained repository outcome. "
        "Skip when: the agent must make an unresolved product decision, credentials/network are required, only one tiny lookup is needed, or work is micro-edits/live discussion."
    )


def _desc_artifact() -> str:
    if LEAN_SCHEMAS:
        return "Fetch an exact source or log slice or bounded binary artifact metadata; binary retrieval never inlines payloads; optional max_response_tokens, response_profile, reuse_key. Actions: get, slice, list."
    return (
        "Fetch one exact source or log slice or bounded binary artifact metadata. Binary payloads are never inlined in MCP responses. Evidence IDs start with E."
        " Use when: exact source or evidence text is required after indexed discovery."
        " Skip when: no source slice is needed or the existing compact result is sufficient."
    )


TaskAction: TypeAlias = Literal[
    "delegate", "explore", "reason", "continue", "review", "second_opinion", "compress", "route", "batch",
    "benchmark", "hardware_benchmark", "evaluation_record", "evaluation_report", "submit", "status", "wait",
    "result", "cancel", "candidate_create", "candidate_promote", "speculative_draft", "vision", "transcribe",
    "eval_suite", "prompt_eval", "eval_drift", "complete_code", "scaffold",
]
_REPO_ACTIONS = (
    "profile", "search", "map", "code_index", "semantic", "graph", "intelligence",
    "deterministic", "context", "route", "delegate", "solve", "review_diff", "impact",
    "refactor_impact", "resolve_imports", "generate_tests", "validate_patch",
    "audit_dependencies", "ast_outline", "test_matrix", "security_audit", "git_status", "repo_state",
    "synthesize_commit", "verify", "preprocess", "preprocess_status", "preprocess_refresh",
    "preprocess_pause", "preprocess_resume", "preprocess_cancel", "preprocess_unregister",
    "context_compile", "verify_receipt", "verify_completion",
    "cross_project_graph", "cross_project_symbols", "cross_project_impact",
    "cross_repo_graph", "cross_repo_symbols", "cross_repo_impact",
    "call_graph_diff", "semantic_diff",
    "affected_tests", "topology", "ast_refactor",
    "generate_mocks", "split_changes", "synthesize_rules",
    "code_invariants", "generate_dataset", "profile_digest",
    "callers", "dead_code", "secret_scan", "schema_inspect", "explain_query", "env_compat",
    "circular_dependencies", "generate_types", "complexity", "api_spec", "dependency_slice", "migration_drift", "package_audit",
    "structural_search", "context_budget",
    "git_diff", "git_history_search", "hotspots", "generate_tests_for_diff", "cross_repo_contract",
    "reachability_dead_code", "mutation_test", "type_stubs", "skeletonize", "investigate",
    "diagnose", "briefing",
)
RepoAction: TypeAlias = Literal.__getitem__(_REPO_ACTIONS + (("batch_replace",) if FEATURES.batch_replacement else ()))
RagAction: TypeAlias = Literal["index", "search", "list", "docset_index", "docset_search", "ingest_document", "ingest_diagram"]
CommandAction: TypeAlias = Literal[
    "run", "cancel", "classify", "discover", "stats", "repair_loop", "auto_fix", "run_affected", "format",
    "lint_fix", "spawn_daemon", "daemon_status", "stop_daemon", "http_probe",
    "stash_save", "stash_restore", "record_mock", "replay_mock",
    "diff_hunk_stage", "flaky_detect", "webhook_replay",
    "mock_server", "mock_server_start", "mock_server_stop", "mock_server_status", "patch_and_verify",
    "preflight", "speculative_lint",
]
CoordAction: TypeAlias = Literal[
    "claim", "renew", "release", "leases", "memo_put", "memo_get", "memo_search", "memo_delete",
    "task_create", "task_get", "task_checkpoint", "task_rollback", "task_transition", "task_resume", "task_list", "task_complete", "task_fail", "task_heartbeat",
    "memory_record", "memory_get", "memory_find", "memory_promote", "memory_reap",
    "relation_record", "relation_find", "relation_traverse",
    "context_compile", "verify_receipt", "verify_completion",
    "negative_knowledge_record", "negative_knowledge_find", "incident_decision",
    "blackboard_update", "blackboard_get", "blackboard_list", "blackboard_merge", "blackboard_delete",
    "swarm_dispatch", "swarm_step", "swarm_status", "swarm_list", "swarm_cancel",
    "worktree_lease", "worktree_release",
    "pubsub_publish", "pubsub_poll", "simulate_merge",
    "curate_dataset", "task_sync", "task_zombie_reap", "task_cleanup_worktree",
]
StatusDetail: TypeAlias = Literal["brief", "cache", "telemetry", "full", "agent_state"]
StatusScope: TypeAlias = Literal["process", "window"]
WorkAction: TypeAlias = Literal["submit", "status", "wait", "get", "cancel", "continue"]


def _invalid_action(tool: str, action: str, valid: tuple[str, ...], guidance: str) -> dict[str, Any]:
    return {
        "success": False,
        "error": f"Unknown {tool} action '{action}'. Valid actions: {', '.join(valid)}. {guidance}",
    }


def _release_root_family_reservation() -> None:
    reservation = _CURRENT_ROOT_FAMILY_RESERVATION.get()
    if not reservation:
        return
    ROOT_FAMILY_BUDGET.release(
        str(reservation.get("family", "")),
        tokens=int(reservation.get("tokens", 0) or 0),
        scope=str(reservation.get("scope", "")),
    )
    _CURRENT_ROOT_FAMILY_RESERVATION.set(None)


def _release_root_family_on_exit(fn: Any) -> Any:
    @functools.wraps(fn)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        try:
            return fn(*args, **kwargs)
        finally:
            _release_root_family_reservation()

    return wrapped



_CURRENT_EXTRA_FIELDS: contextvars.ContextVar[list[str] | None] = contextvars.ContextVar("_CURRENT_EXTRA_FIELDS", default=None)
_CURRENT_RESPONSE_OPTIONS: contextvars.ContextVar[dict[str, Any]] = contextvars.ContextVar("_CURRENT_RESPONSE_OPTIONS", default={})
_CURRENT_ROOT_FAMILY_RESERVATION: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "_CURRENT_ROOT_FAMILY_RESERVATION", default=None
)
_STATUS_PROTECTED_KEYS = (
    "success", "status_code", "error", "warning", "version", "ollama_online",
    "active_model", "queued", "inflight", "cache_hits", "coalesced_waiters",
    "net_cloud_token_delta_est", "vram_pressure", "context_budget_factor",
    "summary", "window", "queue", "hotspots", "by_operation", "error_fingerprints",
)
_REUSE_DIGESTS: OrderedDict[str, str] = OrderedDict()
_REUSE_VALUES: OrderedDict[str, Any] = OrderedDict()
_GUARDED_CONTEXT_TOP_LEVEL = frozenset({
    "success", "status", "status_code", "terminal", "retryable", "error", "warning", "message",
    "context", "context_source", "continuation", "evidence", "evidence_ids", "warnings", "warning_ids",
    "adaptive_context_pack", "context_pack", "context_id", "repo_revision", "revision", "memory_revision",
    "changed_paths", "stale", "since_hash", "delta_from", "guarded", "delivery_mode", "degraded",
    "fallback_used", "requires_override", "requires_approval", "decision_recorded", "decision_persisted",
    "task_status", "waiting", "contract", "reuse", "reuse_candidates", "mappings", "model_warnings",
    "relevance", "model_degraded", "model_degraded_reason", "response_budget", "reuse_key", "reused",
    "unchanged", "preload_profile", "preload_evidence_ids",
    "task_context", "complete", "partial", "text", "source_layers", "agent_state", "repository",
    "provenance", "omitted_sections", "next_action", "estimated_tokens", "truncated",
})
_GUARDED_CONTEXT_PACK = frozenset({
    "contract", "reuse_candidates", "mappings", "warnings", "evidence", "repo_revision", "changed_paths",
    "stale", "context_id", "phase", "focus", "preload_profile", "memory_revision", "model_warnings",
    "delta_from", "since_hash",
})
_GUARDED_RAW_FIELDS = frozenset({
    "raw", "prompt", "model_debug", "raw_model_output", "raw_output", "model_output", "model_response",
    "debug", "debug_trace", "trace", "full_trace", "completion", "response_raw",
})


def _guarded_field_is_safe(name: str, extra_fields: set[str]) -> bool:
    normalized = re.sub(r"[-\s]+", "_", name.strip().lower())
    if normalized in extra_fields:
        return True
    if normalized in _GUARDED_RAW_FIELDS:
        return False
    return not any(marker in normalized for marker in ("raw_", "_raw", "prompt", "debug", "trace"))


def _guarded_context_input(value: Any, *, extra_fields: list[str] | None = None, depth: int = 0, parent: str = "") -> Any:
    extra = {re.sub(r"[-\s]+", "_", str(item).strip().lower()) for item in (extra_fields or ()) if str(item).strip()}
    if isinstance(value, list):
        return [_guarded_context_input(item, extra_fields=list(extra), depth=depth + 1, parent=parent) for item in value[:64]]
    if not isinstance(value, dict):
        return value
    safe: dict[str, Any] = {}
    for raw_key, item in list(value.items())[:64]:
        key = str(raw_key)
        normalized = re.sub(r"[-\s]+", "_", key.strip().lower())
        if not _guarded_field_is_safe(key, extra):
            continue
        if depth == 0 and normalized not in _GUARDED_CONTEXT_TOP_LEVEL and normalized not in extra:
            continue
        if depth == 1 and parent in {"adaptive_context_pack", "context_pack"} and normalized not in _GUARDED_CONTEXT_PACK and normalized not in extra:
            continue
        safe[key] = _guarded_context_input(item, extra_fields=list(extra), depth=depth + 1, parent=normalized)
    return safe


def _subminimum_budget_rejection(options: dict[str, Any]) -> dict[str, Any] | None:
    try:
        requested = int(options.get("max_response_tokens") or 0)
    except (TypeError, ValueError, OverflowError):
        return None
    if requested <= 0 or requested >= 128:
        return None
    return {
        "success": False,
        "status_code": 400,
        "terminal": True,
        "retryable": False,
        "error": "max_response_tokens must be at least 128",
        "response_budget": {"requested_tokens": requested, "minimum_tokens": 128, "rejected": True},
    }


def _response_budget(task_kind: str, options: dict[str, Any]) -> tuple[int, str, str, bool]:
    requested_profile = str(options.get("response_profile") or "").strip().lower()
    profile = requested_profile or "compact"
    if profile not in {"minimal", "compact", "standard", "debug", "delta"}:
        profile = "compact"
    if not requested_profile:
        profile = CONTEXT_LEDGER.adaptive_profile(profile)
    try:
        requested = int(options.get("max_response_tokens") or 0)
    except (TypeError, ValueError, OverflowError):
        requested = 0
    tool_name = str(options.get("tool") or "")
    configured = RESPONSE_TOOL_TOKENS.get(tool_name) or RESPONSE_TOOL_TOKENS.get(task_kind) or RESPONSE_DEFAULT_TOKENS
    if not requested:
        try:
            requested = int(configured)
        except (TypeError, ValueError, OverflowError):
            requested = RESPONSE_DEFAULT_TOKENS
    try:
        max_allowed = max(RESPONSE_DEFAULT_TOKENS, int(RESPONSE_BUDGET_CFG.get("max_tokens", RESPONSE_DEFAULT_TOKENS * 2)))
    except (TypeError, ValueError, OverflowError):
        max_allowed = RESPONSE_DEFAULT_TOKENS * 2
    requested = min(requested, max_allowed)
    if profile == "minimal":
        requested = min(requested, max(256, RESPONSE_DEFAULT_TOKENS // 2))
    elif profile == "standard":
        requested = max(requested, RESPONSE_DEFAULT_TOKENS)
    elif profile == "debug":
        requested = max(requested, int(RESPONSE_DEFAULT_TOKENS * 2))
    return max(128, requested), profile, str(options.get("reuse_key") or "")[:160], RESPONSE_BUDGET_ENABLED


def _reuse_only(key: str, value: Any) -> bool:
    return _reuse_state(key, value)[0]


def _reuse_state(key: str, value: Any) -> tuple[bool, Any]:
    if not key:
        return False, None
    digest = hashlib.sha256(repr(_normalize_deterministic(value)).encode("utf-8", errors="replace")).hexdigest()
    previous = _REUSE_DIGESTS.get(key)
    previous_value = _REUSE_VALUES.get(key)
    _REUSE_DIGESTS[key] = digest
    _REUSE_VALUES[key] = value
    _REUSE_DIGESTS.move_to_end(key)
    _REUSE_VALUES.move_to_end(key)
    while len(_REUSE_DIGESTS) > RESPONSE_REUSE_LIMIT:
        _REUSE_DIGESTS.popitem(last=False)
    while len(_REUSE_VALUES) > RESPONSE_REUSE_LIMIT:
        _REUSE_VALUES.popitem(last=False)
    return previous == digest, previous_value


def _compact(value: Any, task_kind: str = "general", extra_fields: list[str] | None = None) -> Any:
    # Capture measured savings before the projector intentionally removes internal
    # token_saving/runtime fields. Private accounting metadata is stripped by the
    # instrumented MCP boundary and never enters agent context.
    if extra_fields is None:
        extra_fields = _CURRENT_EXTRA_FIELDS.get()
    options = _CURRENT_RESPONSE_OPTIONS.get()
    rejection = _subminimum_budget_rejection(options)
    if rejection is not None:
        return rejection
    raw_value = value
    value = attach_accounting(value)
    projected = PROJECTOR.project(value, AGENT_NAME, task_kind, extra_fields=extra_fields)
    compacted = compact_result(projected, max_text_chars=MAX_TEXT, max_evidence=MAX_EVIDENCE, extra_fields=extra_fields)
    requested, profile, reuse_key, enabled = _response_budget(task_kind, options)
    if enabled:
        repeated, previous = _reuse_state(reuse_key, compacted)
        if profile == "delta" and previous is not None and not repeated:
            compacted = delta_response(previous, compacted, result_id=result_id(compacted))
        protected_keys = (
            _CONTEXT_PROTECTED_KEYS if task_kind in {"context", "context_compile"}
            else _STATUS_PROTECTED_KEYS if task_kind == "status"
            else ()
        )
        compacted = budget_response(
            compacted,
            max_tokens=requested,
            profile=profile,
            reuse_key=reuse_key,
            reuse_only=repeated and json_tokens(compacted) > requested // 2,
            protected_keys=protected_keys,
            token_metadata=efficiency_metadata(),
        )
    return _normalize_deterministic(account_projection(raw_value, compacted))


def _telemetry_status_projection(value: Any) -> dict[str, Any]:
    """Keep status telemetry decision-grade instead of returning the full report."""
    if not isinstance(value, dict):
        return {"success": False, "terminal": True, "retryable": True, "error": "telemetry response must be an object"}
    envelope_success = bool(value.get("success", True))
    report = value.get("report") if isinstance(value.get("report"), dict) else value
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    keep_summary = (
        "events", "cache_hits", "coalesced", "input_tokens", "output_tokens",
        "avoided_cloud_tokens", "avg_duration_ms", "failure_rate", "avg_queue_wait_ms",
        "fallback_count", "degraded_count", "retry_count", "cache_hit_rate", "net_cloud_token_delta_est",
    )
    projected: dict[str, Any] = {
        "success": envelope_success and bool(report.get("success", True)),
        "window": report.get("window") if isinstance(report.get("window"), dict) else {},
        "summary": {key: summary[key] for key in keep_summary if key in summary},
        "queue": report.get("queue") if isinstance(report.get("queue"), dict) else summary.get("queue", {}),
        "hotspots": list(report.get("hotspots") or [])[:12],
        "by_operation": list(report.get("by_operation") or [])[:20],
        "error_fingerprints": list(report.get("error_fingerprints") or [])[:12],
        "detail": report.get("detail") if isinstance(report.get("detail"), dict) else {},
        "decision_grade": report.get("decision_grade") if isinstance(report.get("decision_grade"), dict) else {},
    }
    if value.get("error") or report.get("error"):
        projected["error"] = value.get("error") or report.get("error")
    return projected


def _load_task_context(
    *,
    task_id: str,
    root: str,
    context: str,
    phase: str = "",
    focus: list[str] | None = None,
    preload_profile: str = "",
    changed_paths: list[str] | None = None,
    repository_revision: str = "",
    token_budget: int = 0,
) -> tuple[str, dict[str, Any]]:
    """Fetch one unified Agent OS + repository context for a model call."""
    if not task_id.strip():
        return context, {}
    payload = {
        "action": "compile",
        "task_id": task_id,
        "token_budget": token_budget or 4000,
        "root": _client_root(root),
        "phase": phase,
        "focus": list(focus or []),
        "preload_profile": preload_profile,
        "changed_paths": list(changed_paths or []),
        "repository_revision": repository_revision,
        "compact": True,
    }
    raw = CLIENT.post("/api/agent-state/context", payload, timeout=_timeout("context"))
    if not isinstance(raw, dict):
        return "", {
            "success": False,
            "terminal": True,
            "retryable": True,
            "error": "task context response must be a JSON object",
        }
    task_context = raw.get("task_context")
    if not isinstance(task_context, dict):
        # Keep legacy endpoint responses usable for callers that do not request
        # repository context, while making the missing unified layer explicit.
        task_context = {}
    if not raw.get("success", False) or not raw.get("complete", bool(task_context)):
        return "", {
            "success": False,
            "terminal": bool(raw.get("terminal", False)),
            "retryable": bool(raw.get("retryable", True)),
            "error": str(raw.get("error") or "task context is incomplete"),
            "task_context": task_context,
        }
    task_text = str(task_context.get("text") or raw.get("text") or "").strip()
    if not task_text:
        return "", {
            "success": False,
            "terminal": True,
            "retryable": False,
            "error": "task context is empty",
            "task_context": task_context,
        }
    context_id = str(task_context.get("context_id") or "")[:80]
    context_etag = str(task_context.get("etag") or "")[:80]
    evidence_ids = [str(item)[:120] for item in (task_context.get("evidence_ids") or [])[:64] if str(item)]
    receipt = (
        f"Task context receipt: context_id={context_id or 'unknown'} "
        f"etag={context_etag or 'unknown'} evidence_ids={','.join(evidence_ids) or 'none'}"
    )
    merged = "\n\n".join(part for part in (context.strip(), "Task-scoped context:\n" + task_text, receipt) if part)
    return merged, {
        "success": True,
        "context_id": context_id,
        "etag": context_etag,
        "evidence_ids": evidence_ids,
    }


def _attach_task_context_receipt(result: Any, metadata: dict[str, Any]) -> Any:
    """Expose the exact task-context receipt used by a semantic model call."""
    if not isinstance(result, dict) or not metadata.get("success"):
        return result
    enriched = dict(result)
    enriched["task_context_id"] = str(metadata.get("context_id") or "")[:80]
    enriched["task_context_etag"] = str(metadata.get("etag") or "")[:80]
    enriched["task_context_evidence_ids"] = [
        str(item)[:120] for item in (metadata.get("evidence_ids") or [])[:64] if str(item)
    ]
    return enriched


_SEMANTIC_PATH_RE = re.compile(r"(?<![\w./-])[A-Za-z0-9_.-]+(?:[\\/][A-Za-z0-9_.-]+)+(?:\.[A-Za-z0-9_-]+)(?::\d+)?")


def _quality_check_semantic_result(
    result: Any,
    *,
    task: str,
    context: str,
    changed_paths: list[str] | None = None,
) -> Any:
    if not isinstance(result, dict) or not result.get("success", "error" not in result):
        return result
    # A semantic success is only meaningful when it carries a usable payload.
    # Empty model output used to leak through as success=true and poisoned
    # caches/async handoffs with no evidence for the caller.
    # ``answer`` is a legacy/provider-neutral text field and is valid content;
    # normalize it so the public projection has one canonical text surface.
    if result.get("answer") not in (None, "") and result.get("text") in (None, ""):
        result = dict(result)
        result["text"] = str(result.get("answer"))
    content_keys = ("text", "summary", "artifact_id", "job_id", "result", "results", "task_context", "candidate")
    has_content = any(
        key in result and result.get(key) not in (None, "", [], {})
        for key in content_keys
    )
    if not has_content:
        result = dict(result)
        result.update({
            "success": False,
            "terminal": False,
            "retryable": True,
            "error_code": "empty_semantic_result",
            "error": "Local semantic backend returned success without text, summary, result, or artifact reference.",
        })
        return result
    evidence_paths = [str(path) for path in (changed_paths or []) if str(path).strip()]
    if not evidence_paths:
        evidence_paths = list(dict.fromkeys(_SEMANTIC_PATH_RE.findall(context)))[:64]
    quality = assess_semantic_result(task, evidence_paths, result, context=context)
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


class _ProtocolAccountingReporter:
    """Non-blocking MCP-boundary telemetry reporter.

    Tool accounting must never add foreground latency, so calls enqueue a tiny
    metadata-only event. A lazy daemon batches events to the hub. If the hub is
    unavailable, accounting is dropped rather than delaying the agent.
    """

    def __init__(self) -> None:
        self._queue: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=2048)
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._client: HubClient | None = None

    def _ensure_worker(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, name="local-ai-token-accounting", daemon=True)
            self._thread.start()

    def record(self, event: dict[str, Any]) -> None:
        try:
            self._queue.put_nowait(event)
        except queue.Full:
            return
        self._ensure_worker()

    def _send(self, events: list[dict[str, Any]]) -> None:
        if not events:
            return
        try:
            if self._client is None:
                self._client = HubClient(tenant=TENANT, auto_start=False)
            # batch_id prevents HubClient single-flight from coalescing two
            # numerically identical accounting batches from concurrent MCP clients.
            self._client.post(
                "/api/telemetry/tool-accounting",
                {"batch_id": f"{os.getpid()}-{time.time_ns()}", "events": events},
                timeout=0.25,
            )
        except Exception:
            # Observability is best effort and must never become a retry/backpressure
            # path for the MCP foreground request.
            return

    def _run(self) -> None:
        batch: list[dict[str, Any]] = []
        idle_since = time.monotonic()
        while not self._stop.is_set():
            try:
                batch.append(self._queue.get(timeout=0.35))
                idle_since = time.monotonic()
            except queue.Empty:
                pass
            while len(batch) < 32:
                try:
                    batch.append(self._queue.get_nowait())
                except queue.Empty:
                    break
            if batch:
                self._send(batch)
                batch.clear()
            if time.monotonic() - idle_since > 15.0:
                # Retire atomically with submit/start. A record that races with
                # retirement either makes the queue non-empty here or observes
                # _thread=None in _ensure_worker and starts a replacement.
                with self._lock:
                    if self._queue.empty() and self._thread is threading.current_thread():
                        self._thread = None
                        return
        if batch:
            self._send(batch)
        with self._lock:
            if self._thread is threading.current_thread():
                self._thread = None

    def close(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=0.35)
        remaining: list[dict[str, Any]] = []
        while len(remaining) < 64:
            try:
                remaining.append(self._queue.get_nowait())
            except queue.Empty:
                break
        if remaining:
            self._send(remaining)
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass


_ACCOUNTING_REPORTER = _ProtocolAccountingReporter()
atexit.register(_ACCOUNTING_REPORTER.close)


@functools.lru_cache(maxsize=32)
def _schema_descriptor(tool_name: str) -> dict[str, Any]:
    """Return one stable public-tool descriptor for token-cost estimation."""
    try:
        fn = globals().get(tool_name)
        desc_map = globals().get("_all_desc_map", {})
        builder = desc_map.get(tool_name) if isinstance(desc_map, dict) else None
        description = builder() if callable(builder) else (getattr(fn, "__doc__", "") or "")
        signature = str(inspect.signature(fn)) if callable(fn) else ""
        return {"name": tool_name, "description": description, "signature": signature}
    except Exception:
        return {"name": tool_name}


@functools.lru_cache(maxsize=1)
def _tool_catalog_schema_tokens() -> int:
    """Estimate enabled MCP schema exposure once per process.

    Hosts differ in how often the catalog is re-injected, so this value is kept as
    a separate conservative/upper-bound adjustment rather than folded into the
    default protocol cost.  Caching avoids repeated inspect/description work on
    every foreground tool call.
    """
    try:
        desc_map = globals().get("_all_desc_map", {})
        names = [name for name in desc_map if name not in set(getattr(FEATURES, "disabled_tools", []) or [])]
        return json_tokens({"tools": [_schema_descriptor(name) for name in names]})
    except Exception:
        return 0


def _instrumented_tool():
    def decorator(fn: Any) -> Any:
        signature = inspect.signature(fn)

        @functools.wraps(fn)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            started = time.monotonic()
            handoff_added = False
            try:
                bound = signature.bind_partial(*args, **kwargs)
                # Do not apply Python defaults: omitted optional arguments are not
                # emitted by the model and therefore must not inflate tool-call cost.
                arguments = dict(bound.arguments)
            except Exception:
                arguments = dict(kwargs)
            extra = arguments.get("extra_fields") or kwargs.get("extra_fields")
            token = _CURRENT_EXTRA_FIELDS.set(extra) if extra is not None else None
            response_token = _CURRENT_RESPONSE_OPTIONS.set({
                "max_response_tokens": arguments.get("max_response_tokens", 0),
                "response_profile": arguments.get("response_profile", ""),
                "reuse_key": arguments.get("reuse_key", ""),
                "tool": fn.__name__,
            })
            metadata_token = set_metadata(
                task_id=arguments.get("task_id", ""),
                parent_task=arguments.get("parent_task_id", ""),
                phase=arguments.get("phase", ""),
                activity=f"{fn.__name__}:{arguments.get('action', 'default')}",
                revision=arguments.get("repository_revision") or arguments.get("repo_revision", ""),
                context_digest=context_digest(arguments.get("context", "")) if arguments.get("context") else "",
                fanout_count=len(arguments.get("tasks") or []) if isinstance(arguments.get("tasks"), list) else 0,
            )
            metadata = {}
            try:
                result = fn(*args, **kwargs)
            except Exception:
                _record_adoption(fn.__name__, arguments, None, (time.monotonic() - started) * 1000, failed=True)
                raise
            finally:
                metadata = efficiency_metadata()
                reset_metadata(metadata_token)
                if token is not None:
                    _CURRENT_EXTRA_FIELDS.reset(token)
                _CURRENT_RESPONSE_OPTIONS.reset(response_token)
            clean, measured = pop_accounting(result)
            if isinstance(clean, dict) and clean.get("success") is True:
                hint = semantic_handoff_hint(
                    fn.__name__,
                    str(arguments.get("action") or "").lower(),
                    FEATURES.tasks and FEATURES.has_any_model(),
                    FEATURES.status,
                )
                if hint is not None:
                    candidate = dict(clean)
                    routing = candidate.get("routing")
                    routing_added = False
                    if routing is None:
                        candidate["routing"] = {"semantic_handoff": hint}
                        routing_added = True
                    elif isinstance(routing, dict) and "semantic_handoff" not in routing:
                        candidate["routing"] = {**routing, "semantic_handoff": hint}
                        routing_added = True
                    elif not isinstance(routing, dict):
                        candidate["routing"] = {"value": routing, "semantic_handoff": hint}
                        routing_added = True
                    requested_tokens = int(arguments.get("max_response_tokens") or 0)
                    if routing_added and (requested_tokens <= 0 or json_tokens(candidate) <= requested_tokens):
                        clean = candidate
                        handoff_added = True
            _record_adoption(fn.__name__, arguments, clean, (time.monotonic() - started) * 1000, recommended=handoff_added)
            try:
                event = finalize_tool_accounting(
                    tool_name=fn.__name__, arguments=arguments, response=clean, measured=measured,
                    schema_tokens_est=_tool_catalog_schema_tokens(),
                    metadata=metadata,
                )
                event.update({"tenant": TENANT, "agent": AGENT_NAME, "created_at": time.time()})
                CONTEXT_LEDGER.record(
                    tool=fn.__name__,
                    operation=str(arguments.get("action") or "default"),
                    request_tokens=event.get("agent_tool_request_tokens_est", 0),
                    response_tokens=event.get("agent_tool_response_tokens_est", 0),
                    saved_tokens=event.get("projected_response_saved_tokens_est", 0),
                    cache_outcome=event.get("cache_outcome", ""),
                    result_id=clean.get("result_id", "") if isinstance(clean, dict) else "",
                    session_id=TENANT,
                )
                _ACCOUNTING_REPORTER.record(event)
            except Exception:
                pass
            return clean

        return wrapped
    return decorator


def _adoption_reason(value: Any) -> str:
    text = str(value or "").lower()
    if "timeout" in text: return "timeout"
    if "unsupported" in text or "disabled" in text: return "unsupported"
    if "policy" in text or "forbidden" in text or "denied" in text: return "policy"
    if "validation" in text or "invalid" in text or "required" in text: return "validation"
    if "unavailable" in text or "connection" in text: return "unavailable"
    return "other"


_ADOPTION_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")
_ADOPTION_TOOLS = {"local_ai_status", "local_ai_task", "local_ai_repo", "local_ai_rag", "local_ai_command", "local_ai_coord", "local_ai_work", "local_ai_artifact"}
_ADOPTION_ACTIONS = {
    "local_ai_status": set(StatusDetail.__args__),
    "local_ai_task": set(TaskAction.__args__),
    "local_ai_repo": set(_REPO_ACTIONS) | {"batch_replace"},
    "local_ai_rag": set(RagAction.__args__),
    "local_ai_command": set(CommandAction.__args__),
    "local_ai_coord": set(CoordAction.__args__),
    "local_ai_work": set(WorkAction.__args__),
    "local_ai_artifact": {"get"},
}


def _adoption_target(arguments: dict[str, Any]) -> tuple[str, str] | None:
    if arguments.get("adoption_signal") != "bypassed":
        return None
    tool = str(arguments.get("target_tool") or "").strip().lower()
    action = str(arguments.get("target_action") or "").strip().lower()
    if tool not in _ADOPTION_TOOLS or not _ADOPTION_IDENTIFIER.fullmatch(action) or action not in _ADOPTION_ACTIONS[tool]:
        return None
    return tool, action


def _record_adoption(tool: str, arguments: dict[str, Any], result: Any, duration_ms: float, *, failed: bool = False, recommended: bool = False) -> None:
    """Best-effort aggregate telemetry; never retain request/response values."""
    try:
        action = str(arguments.get("action") or "default").lower()
        target = _adoption_target(arguments)
        if target is not None:
            tool, action = target
        intent = {"local_ai_repo": "repository", "local_ai_command": "validation", "local_ai_coord": "coordination"}.get(tool, "hub")
        payload = result if isinstance(result, dict) else {}
        if target is not None:
            outcome, reason = "bypassed", "explicit_client_signal"
        elif failed:
            outcome, reason = "failed", "other"
        elif payload.get("fallback_used") is True:
            outcome, reason = "fallback_used", _adoption_reason(payload.get("error"))
        elif payload.get("success") is True:
            outcome, reason = "used", None
        elif payload.get("blocked") is True or payload.get("unsupported") is True:
            outcome, reason = "blocked", _adoption_reason(payload.get("error"))
        else:
            outcome, reason = "failed", _adoption_reason(payload.get("error"))
        # Do not serialize payloads merely to measure them: they can contain source,
        # prompts, paths, or secrets.  A bounded structural estimate is enough for a
        # coarse output-size bucket and never reads any value.
        output_size = min(4096, len(payload) * 64)
        ADOPTION_METRICS.record(tool, action, intent, outcome, fallback_reason=reason, duration_ms=duration_ms, output_size=output_size)
        if recommended:
            ADOPTION_METRICS.record(tool, action, intent, "recommended", duration_ms=duration_ms, output_size=output_size)
    except Exception:
        pass


@mcp.tool()
@_instrumented_tool()
def local_ai_status(detail: StatusDetail = "brief", scope: str = "process", extra_fields: list[str] | None = None, adoption_signal: Literal["", "bypassed"] = "", target_tool: str = "", target_action: str = "", max_response_tokens: int = 0, response_profile: str = "", reuse_key: str = "") -> dict[str, Any]:
    """Health/queue/token-saving status. detail: brief, cache, telemetry, full, agent_state. scope: process (default) or window. To report a deliberate bypass, set adoption_signal=bypassed with validated target_tool and target_action; omitted means no bypass. Telemetry is metadata-only. Do not poll status during normal repository work or while preprocessing/model startup is in progress; one bounded health check is enough before native fallback. Use when: make one bounded health, cache, or telemetry check. Skip when: repository evidence or task work is needed."""
    if adoption_signal not in {"", "bypassed"}:
        return {"success": False, "error": "adoption_signal must be empty or bypassed"}
    if adoption_signal == "bypassed" and _adoption_target({"adoption_signal": adoption_signal, "target_tool": target_tool, "target_action": target_action}) is None:
        return {"success": False, "error": "bypass report requires a supported target_tool and safe target_action"}
    if adoption_signal == "" and (target_tool or target_action):
        return {"success": False, "error": "target_tool and target_action require adoption_signal=bypassed"}
    if not FEATURES.status:
        return {"success": False, "unsupported": True, "error": "local_ai_status is disabled in configuration"}
    scope = scope.strip().lower()
    if scope not in {"process", "window"}:
        return {"success": False, "error": "scope must be process or window"}
    # Status stays on the foreground agent path; live/light keeps this read
    # local, bounded, and independent of a slow user-managed runtime.
    status = CLIENT.get(f"/api/live/status?light=1&scope={scope}")
    runtime_stats = status.get("runtime_stats") if isinstance(status, dict) else {}
    if not isinstance(runtime_stats, dict):
        runtime_stats = {}
    if detail == "full":
        status["capabilities"] = CLIENT.get("/api/capabilities")
        return _compact(status, "status")
    if detail == "telemetry":
        report = CLIENT.get(f"/api/telemetry/report?days=30&scope={scope}")
        return _compact(_telemetry_status_projection(report), "status")
    if detail == "agent_state":
        return _compact(CLIENT.status(detail="agent_state"), "status")
    if detail == "cache":
        return _compact({
            "success": bool(status.get("hub_online", False)),
            "generation_cache": runtime_stats.get("generation_cache"),
            "semantic_cache": runtime_stats.get("semantic_cache"),
            "repo_cache": runtime_stats.get("repo_cache"),
            "repo_snapshots": runtime_stats.get("repo_snapshots"),
            "commands": runtime_stats.get("commands"),
            "lossless_router": status.get("lossless_router"),
            "pipeline": runtime_stats.get("pipeline"),
            "preprocessing": status.get("preprocessing"),
            "tool_agent": runtime_stats.get("tool_agent"),
            "circuit_breakers": status.get("circuit_breakers"),
            "fallback_count": status.get("fallback_count"),
            "embeddings": status.get("embeddings"),
            "reranker": status.get("reranker"),
            "token_saving": status.get("observability"),
            "context_ledger": CONTEXT_LEDGER.snapshot(),
        }, "status")
    sched = status.get("scheduler", {}) if isinstance(status, dict) else {}
    saving = status.get("observability", {}) if isinstance(status, dict) else {}
    vram = status.get("vram_balancer") if isinstance(status, dict) else None
    res = {
        "success": bool(status.get("hub_online", False)),
        "version": status.get("version"),
        "ollama_online": status.get("ollama_online"),
        "active_model": sched.get("active_model"),
        "queued": sched.get("queued"),
        "inflight": sched.get("inflight"),
        "net_cloud_token_delta_est": saving.get("net_cloud_token_delta_est", 0),
        "cache_hits": saving.get("cache_hits", 0),
        "coalesced_waiters": saving.get("coalesced_waiters", 0),
    }
    if vram and isinstance(vram, dict):
        res["vram_pressure"] = vram.get("pressure_level", "nominal")
        res["context_budget_factor"] = vram.get("context_budget_factor", 1.0)
    return res


@mcp.tool()
@_instrumented_tool()
@_release_root_family_on_exit
def local_ai_task(
    action: TaskAction,
    task: str = "",
    prompt: str = "",
    model: str = "",
    context: str = "",
    candidate: str = "",
    image_artifact_id: str = "",
    screenshot_artifact_id: str = "",
    bundle_artifact_id: str = "",
    dom_artifact_id: str = "",
    accessibility_artifact_id: str = "",
    computed_styles_artifact_id: str = "",
    runtime_artifact_id: str = "",
    network_artifact_id: str = "",
    source: str = "",
    cloud_fallback: bool = False,
    dom: str | dict[str, Any] | None = None,
    accessibility: str | dict[str, Any] | None = None,
    computed_styles: str | dict[str, Any] | None = None,
    runtime: str | dict[str, Any] | None = None,
    bundle: str | dict[str, Any] | None = None,
    html: str | None = None,
    accessibility_snapshot: str | dict[str, Any] | None = None,
    computed_style_data: str | dict[str, Any] | None = None,
    runtime_context: str | dict[str, Any] | None = None,
    viewport: dict[str, Any] | None = None,
    page: dict[str, Any] | None = None,
    complexity: str = "auto",
    max_tokens: int = 0,
    tasks: list[dict[str, Any]] | None = None,
    cases: list[dict[str, Any]] | None = None,
    delivery: str = "sync",
    latency_budget_ms: float = 0.0,
    job_action: str = "reason",
    job_id: str = "",
    timeout_seconds: float = 90,
    evaluation_task_id: str = "",
    evaluation_cohort: str = "",
    quality_pass: bool | None = None,
    test_pass: bool | None = None,
    evaluation_duration_ms: float = 0.0,
    evaluation_days: int = 30,
    profile: str = "",
    root: str = "",
    workspace: str = "",
    task_id: str = "",
    phase: str = "",
    focus: list[str] | None = None,
    preload_profile: str = "",
    changed_paths: list[str] | None = None,
    repository_revision: str = "",
    conversation: bool = False,
    conversation_id: str = "",
    approver: str = "",
    candidate_data: dict[str, Any] | None = None,
    format: str | dict[str, Any] | None = None,
    json_schema: dict[str, Any] | None = None,
    language: str = "auto",
    extra_fields: list[str] | None = None,
    max_response_tokens: int = 0,
    response_profile: str = "",
    reuse_key: str = "",
) -> dict[str, Any]:
    """Run one bounded semantic local-model task.

    Use deterministic/indexed tools for exact facts, symbols, diffs and tests;
    use this worker for semantic generation, exploration, reasoning, review, independent
    second opinions and semantic compression. Explicit model overrides must
    match a configured model role. Named advisory profiles use read-only
    repository tooling. Command-failure diagnosis remains opt-in and accepts
    only a narrow artifact-backed preview, never raw logs. This tool is not the
    orchestrator for native Codex subagents. The public action enum is
    feature-gated and exposed by the MCP schema, so callers must use the
    advertised actions rather than an inferred alias. Delivery and async
    constraints remain bounded; conversations are sync-only and process-memory
    only. Use when one bounded semantic task should run on a configured local
    model. Use when: one bounded semantic task should run on a configured
    local model. Skip when: local-model tasks are disabled or Codex-owned
    subagent orchestration is the right owner; Local AI Hub does not route or manage native Codex agents."""
    action = _resolve_action("task", action)
    if action == "reason":
        task = task if task.strip() else prompt
        if not task.strip():
            return {"success": False, "terminal": True, "retryable": False,
                    "error_code": "empty_reason_input", "error": "reason requires non-empty task or prompt"}
    # The static contract requires summary, evidence, analysis, limitations
    # and next step. Requests below this floor routinely truncate after a
    # generic preamble, especially on 7b/9b models. Clamp instead of rejecting
    # the task so callers keep the operation and receive a complete answer.
    if action in {"delegate", "explore", "reason", "review", "second_opinion"} and max_tokens:
        max_tokens = max(160, int(max_tokens))
    if action in {"candidate_create", "candidate_promote"}:
        if action == "candidate_create":
            cand = candidate_data or ({"name": task, "baseline_version": "baseline", "candidate_version": "candidate"} if not candidate else {"name": candidate, "baseline_version": "baseline", "candidate_version": "candidate"})
            return _compact(CLIENT.post("/api/agent-state/learning", {
                "action": "create_candidate", "candidate": cand,
            }, timeout=_timeout("quick")), "status")
        return _compact(CLIENT.post("/api/agent-state/learning", {
            "action": "promote", "candidate_id": candidate or evaluation_task_id or task, "approver": approver or "user",
        }, timeout=_timeout("quick")), "status")
    if not FEATURES.tasks or not FEATURES.has_any_model():
        return {"success": False, "unsupported": True, "error": "Local model execution is disabled (features.tasks=false or no Ollama runtime configured)"}
    context, task_context_meta = _load_task_context(
        task_id=task_id,
        root=root,
        context=context,
        phase=phase,
        focus=focus,
        preload_profile=preload_profile,
        changed_paths=changed_paths,
        repository_revision=repository_revision,
        token_budget=max_tokens,
    )
    if task_id.strip() and not task_context_meta.get("success", False):
        return task_context_meta
    context_evidence_ids = list(task_context_meta.get("evidence_ids") or [])
    if profile:
        if not PROFILE_CATALOG.enabled:
            return {"success": False, "unsupported": True, "error": "Ollama subagent profiles disabled"}
        try:
            PROFILE_CATALOG.resolve(profile)
        except ValueError as exc:
            return {"success": False, "unsupported": True, "error": str(exc)}
        if not root or not is_rooted_path(root):
            return {
                "success": False,
                "unsupported": True,
                "error": "Named Ollama profiles require an explicit absolute repository root",
            }
        if action not in {"delegate", "explore", "reason", "review", "second_opinion", "submit"}:
            return {"success": False, "unsupported": True, "error": "Ollama profiles support delegate, explore, reason, review, second_opinion and submit only"}
    if conversation and action not in {"delegate", "explore", "reason"}:
        return {"success": False, "error": "Conversations support delegate, explore, reason and continue only", "terminal": True, "retryable": False}
    if conversation and delivery.strip().lower() != "sync":
        return {"success": False, "error": "Conversations require delivery=sync", "terminal": True, "retryable": False}
    if action in {"submit", "status", "wait", "result", "cancel"}:
        return _compact(CLIENT.post("/api/async-jobs", {
            "action": action, "job_action": job_action, "job_id": job_id, "timeout_seconds": timeout_seconds,
            "task": task, "context": context, "candidate": candidate, "complexity": complexity,
            "max_tokens": max_tokens, "tasks": tasks or [], "profile": profile, "root": root, "workspace": workspace,
        }, timeout=_timeout("quick")), "status")
    budgeted_actions = {"continue", "delegate", "explore", "reason", "review", "second_opinion", "compress"}
    if task_id.strip() and action in budgeted_actions and ROOT_FAMILY_BUDGET.default_limit:
        family_key = hashlib.sha1(_client_root(root).encode("utf-8", "replace"), usedforsecurity=False).hexdigest()[:16]
        budget_tokens = max_tokens or 4096
        budget_scope = f"{task_id}:{action}"
        admission = ROOT_FAMILY_BUDGET.reserve(
            family_key,
            input_tokens=budget_tokens,
            limit=ROOT_FAMILY_BUDGET.default_limit,
            task_id=task_id,
            scope=budget_scope,
        )
        if admission.get("status") == "rejected":
            return {
                "success": False,
                "terminal": not bool(admission.get("retryable", False)),
                "retryable": bool(admission.get("retryable", False)),
                "error": str(admission.get("reason", "root family budget denied")),
                "budget": admission,
            }
        _CURRENT_ROOT_FAMILY_RESERVATION.set({"family": family_key, "tokens": budget_tokens, "scope": budget_scope})
    if action == "continue":
        if profile:
            return {"success": False, "unsupported": True, "error": "Conversations do not support profiles"}
        if not conversation_id.strip():
            return {"success": False, "error": "conversation_id is required", "terminal": True, "retryable": False}
        return _compact(_attach_task_context_receipt(_quality_check_semantic_result(CLIENT.post("/api/conversations/continue", {
            "conversation_id": conversation_id, "task": task, "context": context,
        }, timeout=_timeout("model")), task=task, context=context, changed_paths=changed_paths), task_context_meta), "delegate")
    if profile:
        payload = {
            "profile": profile, "task": task, "context": context, "candidate": candidate,
            "complexity": complexity, "max_tokens": max_tokens, "priority": 5,
            "root": root, "workspace": workspace or None,
            "operation": action,
            "changed_paths": changed_paths or [],
            "evidence_ids": context_evidence_ids,
            "repository_revision": repository_revision,
        }
        endpoint = "/api/delegate/repo" if root else "/api/delegate"
        return _compact(_attach_task_context_receipt(_quality_check_semantic_result(CLIENT.post(endpoint, payload, timeout=_timeout("model")), task=task, context=context, changed_paths=changed_paths), task_context_meta), "delegate")
    if action in {"delegate", "explore"}:
        payload = {
            "task": task, "context": context, "complexity": complexity, "max_tokens": max_tokens or 4096,
            "delivery": delivery, "latency_budget_ms": latency_budget_ms, "model": model,
            "operation": action, "changed_paths": changed_paths or [], "repository_revision": repository_revision,
            "evidence_ids": context_evidence_ids,
        }
        if format or json_schema:
            payload["format"] = format or json_schema
        if conversation:
            payload["conversation"] = True
        return _compact(_attach_task_context_receipt(_quality_check_semantic_result(CLIENT.post("/api/delegate", payload), task=task, context=context, changed_paths=changed_paths), task_context_meta), "explore" if action == "explore" else "delegate")
    if action == "reason":
        payload = {
            "problem": task, "context": context, "max_tokens": max_tokens or 4096,
            "delivery": delivery, "latency_budget_ms": latency_budget_ms, "model": model,
            "operation": "reason", "changed_paths": changed_paths or [], "repository_revision": repository_revision,
            "evidence_ids": context_evidence_ids,
        }
        if format or json_schema:
            payload["format"] = format or json_schema
        if conversation:
            payload["conversation"] = True
        return _compact(_attach_task_context_receipt(_quality_check_semantic_result(CLIENT.post("/api/reason", payload), task=task, context=context, changed_paths=changed_paths), task_context_meta), "reason")
    if action == "review":
        payload = {
            "code": context, "instructions": task or "Report actionable defects only.",
            "complexity": complexity, "max_tokens": max_tokens or 4096,
            "delivery": delivery, "latency_budget_ms": latency_budget_ms, "model": model,
            "operation": "review", "changed_paths": changed_paths or [], "repository_revision": repository_revision,
            "evidence_ids": context_evidence_ids,
        }
        if format or json_schema:
            payload["format"] = format or json_schema
        return _compact(_attach_task_context_receipt(_quality_check_semantic_result(CLIENT.post("/api/review", payload), task=task, context=context, changed_paths=changed_paths), task_context_meta), "review")
    if action == "second_opinion":
        return _compact(_attach_task_context_receipt(_quality_check_semantic_result(CLIENT.post("/api/second-opinion", {
            "question": task, "candidate": candidate, "context": context, "max_tokens": max_tokens or 4096,
            "delivery": delivery, "latency_budget_ms": latency_budget_ms, "model": model,
            "changed_paths": changed_paths or [], "repository_revision": repository_revision,
            "evidence_ids": context_evidence_ids,
        }), task=task, context=context, changed_paths=changed_paths), task_context_meta), "second_opinion")
    if action == "compress":
        return _compact(_attach_task_context_receipt(_quality_check_semantic_result(CLIENT.post("/api/compress", {
            "text": context, "instruction": task or "Compress while preserving facts, identifiers, numbers, errors, decisions and uncertainty.",
            "target_tokens": max_tokens or 650,
            "delivery": delivery, "latency_budget_ms": latency_budget_ms,
        }, timeout=_timeout("model")), task=task, context=context, changed_paths=changed_paths), task_context_meta), "compress")
    if action == "route":
        return _compact(CLIENT.post("/api/route", {"text": context, "query": task, "delivery": delivery, "latency_budget_ms": latency_budget_ms}, timeout=_timeout("model")), "route")
    if action == "batch":
        return _compact(CLIENT.post("/api/delegate/batch", {"tasks": tasks or [], "delivery": delivery, "latency_budget_ms": latency_budget_ms}, timeout=_timeout("long")), "delegate")
    if action == "benchmark":
        return _compact(CLIENT.post("/api/benchmark", {}, timeout=_timeout("long")), "status")
    if action == "hardware_benchmark":
        return _compact(CLIENT.post("/api/benchmark/run", {"model": model or "", "prompt": prompt or task, "num_tokens": max_tokens or 40}, timeout=_timeout("long")), "status")
    if action == "evaluation_record":
        return _compact(CLIENT.post("/api/evaluation", {
            "action": "record", "task_id": evaluation_task_id, "cohort": evaluation_cohort,
            "quality_pass": quality_pass, "test_pass": test_pass, "duration_ms": evaluation_duration_ms,
        }), "status")
    if action == "evaluation_report":
        return _compact(CLIENT.post("/api/evaluation", {"action": "report", "days": evaluation_days}), "status")
    if action == "speculative_draft":
        return _compact(CLIENT.post("/api/task/speculative_draft", {
            "task": task or prompt, "file": candidate or workspace, "context": context, "root": root,
        }, timeout=_timeout("model")), "delegate")
    if action == "vision":
        payload = {
            "prompt": prompt or task,
            "image": candidate or context,
            "model": model,
            "image_artifact_id": image_artifact_id,
            "screenshot_artifact_id": screenshot_artifact_id,
            "bundle_artifact_id": bundle_artifact_id,
            "dom_artifact_id": dom_artifact_id,
            "accessibility_artifact_id": accessibility_artifact_id,
            "computed_styles_artifact_id": computed_styles_artifact_id,
            "runtime_artifact_id": runtime_artifact_id,
            "network_artifact_id": network_artifact_id,
            "source": source,
            "cloud_fallback": cloud_fallback,
            "root": root,
        }
        if json_schema:
            payload["json_schema"] = json_schema
        for name, value in {
            "dom": dom,
            "accessibility": accessibility,
            "computed_styles": computed_styles,
            "runtime": runtime,
            "viewport": viewport,
            "page": page,
            "bundle": bundle,
            "html": html,
            "accessibility_snapshot": accessibility_snapshot,
            "computed_style_data": computed_style_data,
            "runtime_context": runtime_context,
        }.items():
            if value is not None:
                payload[name] = value
        return _compact(CLIENT.post("/api/task/vision", payload, timeout=_timeout("model")), "delegate")
    if action == "transcribe":
        return _compact(CLIENT.post("/api/task/transcribe", {
            "audio_path": candidate or context or task or prompt, "model": model,
        }, timeout=_timeout("long")), "delegate")
    if action == "eval_suite":
        eval_payload: dict[str, Any] = {"suite_name": task or prompt or "default"}
        if model:
            eval_payload["model"] = model[:160]
        if cases:
            bounded_cases: list[dict[str, Any]] = []
            for raw_case in cases[:32]:
                if not isinstance(raw_case, dict):
                    continue
                bounded_cases.append({
                    "id": str(raw_case.get("id", "case"))[:120],
                    "input": str(raw_case.get("input", ""))[:4000],
                    "expected": str(raw_case.get("expected", ""))[:2000],
                })
            if bounded_cases:
                eval_payload["cases"] = bounded_cases
        return _compact(CLIENT.post("/api/task/eval_suite", eval_payload, timeout=_timeout("long")), "status")
    if action == "prompt_eval":
        return _compact(CLIENT.post("/api/task/prompt_eval", {
            "template": prompt or task, "variables": candidate_data or ({"input": context} if context else {}),
        }, timeout=_timeout("model")), "delegate")
    if action == "eval_drift":
        return _compact(CLIENT.post("/api/task/eval_drift", {
            "suite_name": task or prompt or "default",
        }, timeout=_timeout("long")), "status")
    if action == "complete_code":
        return _compact(CLIENT.post("/api/complete", {
            "prefix": prompt or context or "", "suffix": task or candidate or "", "max_tokens": max_tokens or 80,
        }, timeout=_timeout("quick")), "status")
    if action == "scaffold":
        return _compact(CLIENT.post("/api/task/scaffold", {
            "spec": prompt or task, "context": context, "language": language if (language and language != "auto") else (candidate or "python"),
            "max_tokens": max_tokens or 4096, "model": model,
        }, timeout=_timeout("model")), "delegate")
    return _invalid_action("local_ai_task", action, tuple(TaskAction.__args__), "Use Local AI Hub only for bounded local-model work; use Codex-owned orchestration for peer subagents.")


def _context_input_error(message: str) -> dict[str, Any]:
    return {"success": False, "status_code": 400, "terminal": True, "retryable": False, "error": message[:400]}


def _validate_context_pack_inputs(
    *,
    task_id: Any,
    phase: Any,
    focus: Any,
    preload_profile: Any,
    changed_paths: Any,
    base: Any,
    staged: Any,
    guarded: Any,
    since_hash: Any,
    approval: Any,
    override_reason: Any,
    max_tokens: Any,
    token_budget: Any,
) -> dict[str, Any] | None:
    if not isinstance(guarded, bool):
        return _context_input_error("guarded must be boolean")
    for name, value in (
        ("task_id", task_id), ("phase", phase), ("preload_profile", preload_profile),
        ("base", base), ("since_hash", since_hash), ("override_reason", override_reason),
    ):
        if not isinstance(value, str):
            return _context_input_error(f"{name} must be string")
    for name, value, limit in (("focus", focus, 16), ("changed_paths", changed_paths, 64)):
        if value is not None:
            if not isinstance(value, list):
                return _context_input_error(f"{name} must be a list")
            if len(value) > limit:
                return _context_input_error(f"{name} exceeds maximum of {limit} items")
            if any(not isinstance(item, str) for item in value):
                return _context_input_error(f"{name} items must be strings")
    if not isinstance(staged, bool):
        return _context_input_error("staged must be boolean")
    if not isinstance(approval, (bool, str)):
        return _context_input_error("approval must be boolean or string")
    for name, value in (("max_tokens", max_tokens), ("token_budget", token_budget)):
        if not isinstance(value, int) or isinstance(value, bool):
            return _context_input_error(f"{name} must be integer")
        if value < 0:
            return _context_input_error(f"{name} must be non-negative")
    if token_budget and max_tokens and token_budget != max_tokens:
        return _context_input_error("token_budget and max_tokens must match when both are provided")
    guarded_requested = guarded or bool(task_id.strip()) or bool(phase.strip())
    guarded_only_values = (
        focus is not None or bool(preload_profile) or changed_paths is not None or base != "HEAD"
        or staged or bool(since_hash) or approval != "" or bool(override_reason) or bool(token_budget)
    )
    if guarded_only_values and not guarded_requested:
        return _context_input_error("guarded context fields require guarded=true, task_id, or phase")
    return None


def _bound_context_json(value: Any, depth: int = 0) -> Any:
    if depth > 5:
        return "[…depth…]"
    if isinstance(value, str):
        return value[:6000]
    if isinstance(value, list):
        return [_bound_context_json(item, depth + 1) for item in value[:64]]
    if isinstance(value, dict):
        return {str(key)[:160]: _bound_context_json(item, depth + 1) for key, item in list(value.items())[:64]}
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return str(value)[:600]


def _context_pack_projection(value: Any, *, extra_fields: list[str] | None = None) -> Any:
    if not isinstance(value, dict):
        return {"success": False, "status_code": 502, "terminal": True, "retryable": True, "error": "context pack response must be a JSON object"}
    options = _CURRENT_RESPONSE_OPTIONS.get()
    rejection = _subminimum_budget_rejection(options)
    if rejection is not None:
        return rejection
    safe_value = _guarded_context_input(value, extra_fields=extra_fields) if (
        bool(value.get("guarded")) or "adaptive_context_pack" in value or "context_pack" in value
    ) else value
    projected = _compact(safe_value, "context", extra_fields=extra_fields)
    if not isinstance(projected, dict):
        return projected

    # Guard output is deterministic. Reattach its decision-grade fields after the
    # generic response budget so model/postprocess fields cannot replace them.
    pack = safe_value.get("adaptive_context_pack") or safe_value.get("context_pack")
    pack = pack if isinstance(pack, dict) else {}
    authoritative: dict[str, Any] = {}
    for key in (
        "context_id", "warnings", "repo_revision", "changed_paths", "stale",
        "delta_from", "since_hash",
    ):
        if key in pack:
            authoritative[key] = pack[key]
        elif key in safe_value:
            authoritative[key] = safe_value[key]
    evidence = pack.get("evidence")
    if isinstance(evidence, list):
        ids = [item.get("evidence_id") for item in evidence if isinstance(item, dict) and item.get("evidence_id")]
        authoritative["evidence_ids"] = ids[:24]
    if "evidence_ids" not in authoritative:
        if "evidence_ids" in pack:
            authoritative["evidence_ids"] = pack["evidence_ids"]
        elif "evidence_ids" in safe_value:
            authoritative["evidence_ids"] = safe_value["evidence_ids"]
    for key in (
        "revision", "guarded", "delivery_mode", "since_hash", "delta_from", "degraded", "stale",
        "fallback_used", "requires_override", "requires_approval", "decision_recorded",
        "decision_persisted", "task_status", "waiting",
    ):
        if key in safe_value and key not in authoritative:
            authoritative[key] = safe_value[key]
    # Preserve actual deterministic context, not only its provenance pointer.
    # Without this, the final budget pass could return IDs/status while silently
    # dropping the context text needed by the next bounded agent step.
    for key in ("context", "query", "context_source", "estimated_tokens", "contract"):
        if key in safe_value and key not in authoritative:
            authoritative[key] = safe_value[key]
    for key, item in authoritative.items():
        projected[key] = _bound_context_json(item)

    # Authoritative fields were attached after _compact's response budget. Run
    # the final projection through the same bounds so they cannot bypass it.
    final = compact_result(
        projected,
        max_text_chars=MAX_TEXT,
        max_evidence=MAX_EVIDENCE,
        extra_fields=extra_fields,
    )
    if isinstance(final, dict) and ("adaptive_context_pack" in safe_value or "context_pack" in safe_value or safe_value.get("guarded")):
        final["provenance"] = evidence_meta(
            "repo.context",
            str(authoritative.get("repo_revision", safe_value.get("repo_revision", ""))),
            authoritative.get("evidence_ids", []),
            bool(authoritative.get("stale", safe_value.get("stale", False))),
            "stale" if bool(authoritative.get("stale", safe_value.get("stale", False))) else ("success" if safe_value.get("success", True) else "incomplete"),
        )
    options = _CURRENT_RESPONSE_OPTIONS.get()
    requested, profile, reuse_key, enabled = _response_budget("context", options)
    if enabled:
        final = budget_response(
            final,
            max_tokens=requested,
            profile=profile,
            reuse_key=reuse_key,
            protected_keys=tuple(authoritative),
            token_metadata=efficiency_metadata(),
        )
    return _normalize_deterministic(final)


def _local_ai_repo_impl(
    action: RepoAction,
    root: str = ".",
    query: str = "",
    diff: str = "",
    task: str = "",
    workspace: str = "",
    path: str = "",
    base: str = "HEAD",
    staged: bool = False,
    dry_run: bool = False,
    max_tokens: int = 0,
    evidence: list[dict[str, Any]] | None = None,
    mode: str = "adaptive",
    relation: str = "",
    language: str = "auto",
    profile: str = "",
    receipt: dict[str, Any] | None = None,
    task_id: str = "",
    include_code: bool = False,
    include_tests: bool = False,
    edits: list[dict[str, Any]] | None = None,
    extra_fields: list[str] | None = None,
    max_response_tokens: int = 0,
    response_profile: str = "",
    reuse_key: str = "",
    include_diagnostics: bool = False,
    clone_id: str = "",
    worktree_id: str = "",
    branch: str = "",
    repository_id: str = "",
    session_id: str = "",
    repository_revision: str = "",
    changed_paths: list[str] | None = None,
    phase: str = "",
    focus: list[str] | None = None,
    preload_profile: str = "",
    guarded: bool = False,
    since_hash: str = "",
    approval: str | bool = "",
    override_reason: str = "",
    token_budget: int = 0,
) -> dict[str, Any]:
    """Primary bounded repository worker for the main agent.

    Call this before broad repository reads/searches for any non-trivial repo
    task. Use deterministic, code-index/search, semantic/graph, context and
    solve for repository evidence and implementation support. After needed
    evidence, route semantic generation, reasoning, review, independent second
    opinions and compression to local_ai_task; use local_ai_artifact for exact
    slices and local_ai_command for repeatable validation. Use review_diff and
    security_audit for targeted local checks. Never use this
    evidence path as a substitute for semantic local-model work. Codex manages
    native Codex subagents separately. Preprocess once per stable absolute root;
    never poll or force-refresh it. Reuse fresh evidence and respect
    in_progress, retryable and degraded states. Use when a non-trivial
    repository task needs indexed evidence or bounded Hub support; skip when
    the task is not repository-scoped and no independent Hub scope exists."""
    if not FEATURES.repo:
        return {"success": False, "unsupported": True, "error": "local_ai_repo is disabled in configuration"}
    action = _resolve_action("repo", action)
    if action in {"preprocess", "preprocess_refresh", "preprocess_cancel", "preprocess_unregister"}:
        candidate = Path(root).expanduser()
        if not candidate.is_absolute():
            return {"success": False, "error": "Preprocessing requires an explicit absolute stable project root; do not use MCP process cwd"}
    else:
        root = _client_root(root)
    if action == "profile":
        return _compact(CLIENT.post("/api/repo/profile", {"root": root}, timeout=_timeout("quick")), "profile")
    if action == "investigate":
        return _compact(CLIENT.post("/api/repo/investigate", {
            "root": root, "query": query or task, "path": path, "include_code": include_code or True, "limit": 10,
        }, timeout=_timeout("quick")), "architecture")
    if action == "diagnose":
        return _compact(CLIENT.post("/api/repo/diagnose", {
            "root": root, "text": query or diff or task,
        }, timeout=_timeout("quick")), "verify")
    if action == "briefing":
        return _compact(CLIENT.post("/api/repo/briefing", {
            "root": root,
        }, timeout=_timeout("quick")), "architecture")
    if action == "batch_replace":
        if not FEATURES.batch_replacement:
            return {
                "success": False,
                "unsupported": True,
                "feature": "batch_replacement",
                "error": "batch replacement is disabled (features.batch_replacement=false)",
            }
        if not isinstance(edits, list) or not edits:
            return {"success": False, "error": "batch_replace requires a non-empty edits list"}
        return _compact(CLIENT.post("/api/code/batch_replace", {
            "root": root, "edits": edits, "dry_run": dry_run,
        }, timeout=_timeout("quick")), "verify")
    if action == "search":
        enrich = bool(include_code)
        if enrich and not FEATURES.enriched_search:
            return {
                "success": False,
                "unsupported": True,
                "feature": "enriched_search",
                "error": "enriched search is disabled (features.enriched_search=false)",
            }
        extra = list(extra_fields or [])
        if enrich:
            extra.extend(["text", "raw"])
        return _compact(CLIENT.post("/api/search", {
            "root": root, "query": query or task, "top_k": 12, "enrich": enrich,
            **({"path": path} if path else {}),
        }, timeout=_timeout("quick")), "search", extra_fields=extra if extra else None)
    if action == "map":
        return _compact(CLIENT.post("/api/repo/map", {"root": root, "max_symbols": 100}, timeout=_timeout("quick")), "architecture")
    if action == "code_index":
        return _compact(CLIENT.post("/api/repo/code-index", {"root": root, "query": query or task, "limit": 24, "include_code": include_code}, timeout=_timeout("quick")), "architecture")
    if action in {"semantic", "graph", "intelligence"}:
        backend = "serena" if action == "semantic" else "codegraph" if action == "graph" else "auto"
        operation = relation or ("find_symbol" if backend == "serena" else "relationships" if backend == "codegraph" else "search")
        return _compact(CLIENT.post("/api/code-intelligence/query", {
            "root": root, "query": query or task, "path": path, "backend": backend, "action": operation, "limit": 24,
        }, timeout=_timeout("quick")), "architecture")
    if action == "deterministic":
        return _compact(CLIENT.post("/api/repo/deterministic", {"root": root, "query": query or task, "limit": 30}, timeout=_timeout("quick")), "context")
    if action == "context":
        validation = _validate_context_pack_inputs(
            task_id=task_id, phase=phase, focus=focus, preload_profile=preload_profile,
            changed_paths=changed_paths, base=base, staged=staged, guarded=guarded,
            since_hash=since_hash, approval=approval, override_reason=override_reason,
            max_tokens=max_tokens, token_budget=token_budget,
        )
        if validation is not None:
            return validation
        guarded_requested = bool(guarded) or bool(str(task_id).strip()) or bool(str(phase).strip())
        configured_tokens = CFG.get("token_saving", {}).get("default_repo_context_tokens", 3200)
        effective_tokens = token_budget or max_tokens or configured_tokens
        payload: dict[str, Any] = {
            "root": root, "query": query or task, "workspace": workspace or None,
            "max_tokens": effective_tokens,
            "mode": "full" if mode == "full" else "fast",
        }
        if guarded_requested:
            payload.update({
                "guarded": bool(guarded),
                "task_id": task_id,
                "phase": phase,
                "focus": list(focus or []),
                "preload_profile": preload_profile,
                "changed_paths": list(changed_paths or []),
                "base": base,
                "staged": bool(staged),
                "since_hash": since_hash,
                "approval": approval,
                "override_reason": override_reason,
                "token_budget": effective_tokens,
            })
        raw = CLIENT.post("/api/context/pack", payload, timeout=_timeout("context"))
        return _context_pack_projection(raw, extra_fields=extra_fields)
    if action == "route":
        if not path:
            return {"success": False, "error": "path is required for repo route"}
        return _compact(CLIENT.post("/api/route", {"root": root, "path": path, "query": query or task}, timeout=_timeout("model")), "route")
    if action == "delegate":
        return _compact(CLIENT.post("/api/delegate/repo", {
            "root": root, "task": task or query, "workspace": workspace or None,
            "complexity": "auto", "context_tokens": max_tokens or 0, "max_tokens": max_tokens or 4096, "profile": profile,
        }, timeout=_timeout("model")), "delegate")
    if action == "solve":
        return _compact(CLIENT.post("/api/solve/repo", {
            "root": root, "task": task or query, "workspace": workspace or None, "mode": mode,
            "complexity": "auto", "context_tokens": max_tokens or 0, "base": base, "staged": staged,
        }, timeout=_timeout("long")), "solve")
    if action == "review_diff":
        return _compact(CLIENT.post("/api/review/diff", {
            "root": root, "base": base, "staged": staged,
            "instructions": task or "Report actionable defects, regressions, security/concurrency issues and missing tests only.",
            "complexity": "auto", "max_tokens": max_tokens or 4096, "mode": mode,
            # `local_ai_repo(action="review_diff")` is a targeted repository
            # check. Return its bounded result directly; callers needing
            # durable background execution can use local_ai_task submit/wait.
            "delivery": "sync", "latency_budget_ms": 30000,
        }, timeout=_timeout("long")), "review_diff")
    if action == "impact":
        return _compact(CLIENT.post("/api/repo/impact", {"root": root, "base": base, "staged": staged}, timeout=_timeout("context")), "impact")
    if action == "refactor_impact":
        return _compact(CLIENT.post("/api/refactor_impact", {"root": root, "file": path, "symbol": query or task}, timeout=_timeout("quick")), "impact")
    if action == "resolve_imports":
        syms = [s.strip() for s in (query or task).split(",") if s.strip()]
        return _compact(CLIENT.post("/api/resolve_imports", {"root": root, "symbols": syms, "language": language or "auto"}, timeout=_timeout("quick")), "context")
    if action == "generate_tests":
        return _compact(CLIENT.post("/api/generate_tests", {"root": root, "file": path, "symbol": query or task}, timeout=_timeout("context")), "delegate")
    if action == "validate_patch":
        return _compact(CLIENT.post("/api/patch/validate", {"root": root, "patch": query or task}, timeout=_timeout("quick")), "verify")
    if action == "audit_dependencies":
        return _compact(CLIENT.post("/api/audit_dependencies", {"root": root}, timeout=_timeout("quick")), "context")
    if action == "ast_outline":
        return _compact(CLIENT.post("/api/code/ast_outline", {"root": root, "path": path or query or task}, timeout=_timeout("quick")), "architecture")
    if action == "affected_tests":
        paths_arg = [path] if path else ([query] if query and not query.startswith("-") else None)
        return _compact(CLIENT.post("/api/repo/affected_tests", {"root": root, "paths": paths_arg}, timeout=_timeout("quick")), "architecture")
    if action == "topology":
        return _compact(CLIENT.post("/api/repo/topology", {"root": root}, timeout=_timeout("quick")), "architecture")
    if action == "ast_refactor":
        return _compact(CLIENT.post("/api/code/ast_rename", {
            "root": root, "file": path, "old_symbol": query or task, "new_symbol": diff, "apply": staged,
        }, timeout=_timeout("quick")), "verify")
    if action == "generate_mocks":
        return _compact(CLIENT.post("/api/code/generate_mocks", {
            "root": root, "file": path, "symbol": query or task,
        }, timeout=_timeout("quick")), "architecture")
    if action == "split_changes":
        return _compact(CLIENT.post("/api/repo/split_changes", {
            "root": root, "paths": [path] if path else ([query] if query and not query.startswith("-") else None),
        }, timeout=_timeout("quick")), "architecture")
    if action == "synthesize_rules":
        return _compact(CLIENT.post("/api/repo/synthesize_rules", {
            "root": root, "limit": 10,
        }, timeout=_timeout("quick")), "architecture")
    if action == "test_matrix":
        return _compact(CLIENT.post("/api/test_matrix", {"root": root}, timeout=_timeout("quick")), "architecture")
    if action == "security_audit":
        return _compact(CLIENT.post("/api/security_audit", {"root": root}, timeout=_timeout("quick")), "architecture")
    if action == "git_status":
        return _compact(CLIENT.get(f"/api/git/status?root={quote(root)}"), "status")
    if action == "repo_state":
        return _compact(CLIENT.post("/api/repo/state", {"root": root}, timeout=_timeout("quick")), "status")
    if action == "synthesize_commit":
        synth_payload = {"root": root, "hint": query or task}
        if task and str(task).startswith("task-"):
            synth_payload["task_id"] = str(task)
        return _compact(CLIENT.post("/api/git/synthesize_commit", synth_payload, timeout=_timeout("quick")), "status")
    if action == "verify":
        return _compact(CLIENT.post("/api/evidence/verify", {"root": root, "evidence": evidence or []}), "verify")
    if action == "preprocess":
        return _compact(CLIENT.post("/api/preprocess", {"action": "start", "root": root}), "preprocess")
    if action == "preprocess_status":
        return _compact(CLIENT.post("/api/preprocess", {"action": "status", "root": root}), "status")
    if action == "preprocess_refresh":
        return _compact(CLIENT.post("/api/preprocess", {"action": "refresh", "root": root}), "preprocess")
    if action == "preprocess_pause":
        return _compact(CLIENT.post("/api/preprocess", {"action": "pause", "root": root}), "preprocess")
    if action == "preprocess_resume":
        return _compact(CLIENT.post("/api/preprocess", {"action": "resume", "root": root}), "preprocess")
    if action == "preprocess_cancel":
        return _compact(CLIENT.post("/api/preprocess", {"action": "cancel", "root": root}), "preprocess")
    if action == "preprocess_unregister":
        return _compact(CLIENT.post("/api/preprocess", {"action": "unregister", "root": root}), "preprocess")
    if action == "context_compile":
        return _compact(CLIENT.post("/api/agent-state/context", {
            "action": "compile", "task_id": task_id or query or task,
            "token_budget": max_tokens or 4000, "root": root,
            "include_diagnostics": bool(include_diagnostics),
            "clone_id": clone_id,
            "worktree_id": worktree_id,
            "branch": branch,
            "repository_id": repository_id,
            "session_id": session_id,
            "repository_revision": repository_revision,
            "changed_paths": changed_paths or [],
        }, timeout=_timeout("context")), "context")
    if action == "verify_receipt":
        return _compact(CLIENT.post("/api/agent-state/verification", {
            "action": "receipt", "receipt": receipt or {}, "root": root,
        }, timeout=_timeout("quick")), "verify")
    if action == "verify_completion":
        return _compact(CLIENT.post("/api/agent-state/verification", {
            "action": "completion", "task_id": task_id or query or task, "root": root,
        }, timeout=_timeout("quick")), "verify")
    if action in {"cross_project_graph", "cross_repo_graph"}:
        return _compact(CLIENT.post("/api/cross_project_graph", {"roots": [root] if root else []}, timeout=_timeout("quick")), "architecture")
    if action in {"cross_project_symbols", "cross_repo_symbols"}:
        return _compact(CLIENT.post("/api/cross_project_symbols", {"roots": [root] if root else [], "query": query or task, "limit": 50}, timeout=_timeout("quick")), "architecture")
    if action in {"cross_project_impact", "cross_repo_impact"}:
        return _compact(CLIENT.post("/api/cross_project_impact", {"roots": [root] if root else [], "symbol": query or task}, timeout=_timeout("quick")), "impact")
    if action in {"call_graph_diff", "semantic_diff"}:
        return _compact(CLIENT.post("/api/repo/call_graph_diff", {"root": root or ".", "diff": diff or "" if diff else None}, timeout=_timeout("quick")), "impact")
    if action == "code_invariants":
        return _compact(CLIENT.post("/api/repo/code_invariants", {
            "root": root, "path": path or query or task or None,
        }, timeout=_timeout("quick")), "architecture")
    if action == "generate_dataset":
        return _compact(CLIENT.post("/api/repo/generate_dataset", {
            "root": root, "schema_or_model": query or task or path,
            "count": max_tokens or 10,
            "format": relation or "json",
        }, timeout=_timeout("quick")), "architecture")
    if action == "profile_digest":
        profile_path = path or query or task
        if profile_path and not is_rooted_path(profile_path):
            profile_path = str(Path(_client_root(root)) / profile_path)
        return _compact(CLIENT.post("/api/repo/profile_digest", {
            "profile_path": profile_path, "top_n": max_tokens or 15,
        }, timeout=_timeout("quick")), "architecture")
    if action == "callers":
        return _compact(CLIENT.post("/api/repo/callers", {
            "root": root, "symbol": query or task or path, "limit": max_tokens or 50,
        }, timeout=_timeout("quick")), "architecture")
    if action == "dead_code":
        return _compact(CLIENT.post("/api/repo/dead_code", {
            "root": root, "limit": max_tokens or 50,
        }, timeout=_timeout("quick")), "architecture")
    if action == "secret_scan":
        scan_git = bool(staged or (query and ("git" in query.lower() or "history" in query.lower())))
        return _compact(CLIENT.post("/api/repo/secret_scan", {
            "root": root, "path": path or query or task or None,
            "scan_git_history": scan_git,
        }, timeout=_timeout("quick")), "architecture")
    if action == "schema_inspect":
        return _compact(CLIENT.post("/api/repo/schema_inspect", {
            "root": root, "db_path": path or query or task or None,
        }, timeout=_timeout("quick")), "architecture")
    if action == "explain_query":
        return _compact(CLIENT.post("/api/repo/explain_query", {
            "root": root, "query": query or task, "db_path": path or None,
        }, timeout=_timeout("quick")), "architecture")
    if action == "env_compat":
        return _compact(CLIENT.post("/api/repo/env_compat", {
            "root": root,
        }, timeout=_timeout("quick")), "architecture")
    if action == "circular_dependencies":
        return _compact(CLIENT.post("/api/repo/circular_dependencies", {
            "root": root, "language": query or task or "python",
        }, timeout=_timeout("quick")), "architecture")
    if action == "generate_types":
        return _compact(CLIENT.post("/api/repo/generate_types", {
            "root": root, "file": path or query or task, "write_stub": bool(diff),
        }, timeout=_timeout("quick")), "architecture")
    if action == "complexity":
        return _compact(CLIENT.post("/api/repo/complexity", {
            "root": root, "path": path or query or task or None, "max_results": max_tokens or 20,
            "include_tests": bool(include_tests),
        }, timeout=_timeout("quick")), "architecture")
    if action == "api_spec":
        return _compact(CLIENT.post("/api/repo/api_spec", {
            "root": root, "framework": query or task or None,
        }, timeout=_timeout("quick")), "architecture")
    if action == "dependency_slice":
        return _compact(CLIENT.post("/api/repo/dependency_slice", {
            "root": root, "symbol": query or task or relation, "path": path or None, "depth": max(1, min(max_tokens or 2, 8)),
        }, timeout=_timeout("quick")), "architecture")
    if action == "migration_drift":
        return _compact(CLIENT.post("/api/repo/migration_drift", {
            "root": root, "db_path": path or query or task or None,
        }, timeout=_timeout("quick")), "architecture")
    if action == "package_audit":
        return _compact(CLIENT.post("/api/repo/package_audit", {
            "root": root, "lockfile_path": path or query or task or None,
        }, timeout=_timeout("quick")), "architecture")
    if action == "structural_search":
        return _compact(CLIENT.post("/api/repo/structural_search", {
            "root": root, "pattern": query or task, "path": path or None, "max_results": max_tokens or 30,
        }, timeout=_timeout("quick")), "architecture")
    if action == "context_budget":
        files_arg = [path] if path else ([query] if query and not query.startswith("-") else [])
        return _compact(CLIENT.post("/api/repo/context_budget", {
            "root": root, "files": files_arg, "max_tokens": max_tokens or 4000,
        }, timeout=_timeout("quick")), "architecture")
    if action == "git_diff":
        return _compact(CLIENT.get(f"/api/git/diff?root={quote(root)}&path={quote(path or query or '')}&staged={'true' if staged else 'false'}&max_lines={max_tokens or 500}"), "diff")
    if action == "git_history_search":
        return _compact(CLIENT.post("/api/git/history_search", {
            "root": root, "query": query or task, "max_commits": max_tokens or 50,
        }, timeout=_timeout("quick")), "architecture")
    if action == "hotspots":
        days_val = int(diff) if (diff and diff.isdigit()) else 30
        return _compact(CLIENT.post("/api/repo/hotspots", {
            "root": root, "days": days_val, "limit": max_tokens or 20,
        }, timeout=_timeout("quick")), "architecture")
    if action == "generate_tests_for_diff":
        return _compact(CLIENT.post("/api/repo/generate_tests_for_diff", {
            "root": root, "diff": diff or query or task, "path": path or None,
        }, timeout=_timeout("context")), "delegate")
    if action == "cross_repo_contract":
        return _compact(CLIENT.post("/api/repo/cross_repo_contract", {
            "backend_root": root, "frontend_root": path or workspace or query,
        }, timeout=_timeout("quick")), "architecture")
    if action == "reachability_dead_code":
        return _compact(CLIENT.post("/api/repo/reachability_dead_code", {
            "root": root, "entrypoints": [path] if path else None,
        }, timeout=_timeout("context")), "architecture")
    if action == "mutation_test":
        return _compact(CLIENT.post("/api/repo/mutation_test", {
            "root": root, "file": path or query, "diff": diff or None,
        }, timeout=_timeout("context")), "delegate")
    if action == "type_stubs":
        return _compact(CLIENT.post("/api/repo/type_stubs", {
            "root": root, "file": path or query,
        }, timeout=_timeout("quick")), "code")
    if action == "skeletonize":
        return _compact(CLIENT.post("/api/repo/skeletonize", {
            "code": query or diff, "targets": [path] if path else None,
        }, timeout=_timeout("quick")), "code")
    return _invalid_action("local_ai_repo", action, tuple(FEATURES.supported_repo_actions()), "Keep work bounded in Local AI Hub; use Codex-owned orchestration for peer subagents.")


@mcp.tool()
@_instrumented_tool()
def local_ai_repo(
    action: RepoAction,
    root: str = ".",
    query: str = "",
    diff: str = "",
    task: str = "",
    workspace: str = "",
    path: str = "",
    base: str = "HEAD",
    staged: bool = False,
    dry_run: bool = False,
    max_tokens: int = 0,
    evidence: list[dict[str, Any]] | None = None,
    mode: str = "adaptive",
    relation: str = "",
    language: str = "auto",
    profile: str = "",
    receipt: dict[str, Any] | None = None,
    task_id: str = "",
    include_code: bool = False,
    include_tests: bool = False,
    edits: list[dict[str, Any]] | None = None,
    extra_fields: list[str] | None = None,
    max_response_tokens: int = 0,
    response_profile: str = "",
    reuse_key: str = "",
    include_diagnostics: bool = False,
    clone_id: str = "",
    worktree_id: str = "",
    branch: str = "",
    repository_id: str = "",
    session_id: str = "",
    repository_revision: str = "",
    changed_paths: list[str] | None = None,
    phase: str = "",
    focus: list[str] | None = None,
    preload_profile: str = "",
    guarded: bool = False,
    since_hash: str = "",
    approval: str | bool = "",
    override_reason: str = "",
    token_budget: int = 0,
) -> dict[str, Any]:
    """Primary bounded repository worker. Use when: indexed repository evidence is needed. Skip when: fresh evidence already answers it."""
    return _local_ai_repo_impl(
        action, root, query, diff, task, workspace, path, base, staged, dry_run,
        max_tokens, evidence, mode, relation, language, profile, receipt, task_id,
        include_code, include_tests, edits, extra_fields, max_response_tokens, response_profile, reuse_key,
        include_diagnostics, clone_id, worktree_id, branch, repository_id, session_id, repository_revision,
        changed_paths, phase, focus, preload_profile, guarded, since_hash, approval, override_reason,
        token_budget,
    )


local_ai_repo.__doc__ = _local_ai_repo_impl.__doc__


def _hide_disabled_batch_schema(mcp_runtime: Any, tool: Any) -> None:
    """Project rollout-only repository fields out of optional FastMCP schemas."""
    if hasattr(mcp_runtime, "_tool_manager") and hasattr(mcp_runtime._tool_manager, "_tools"):
        repo_tool = mcp_runtime._tool_manager._tools.get("local_ai_repo")
        if repo_tool is not None:
            properties = repo_tool.parameters.get("properties", {})
            properties.pop("edits", None)
            properties.pop("dry_run", None)
    tool.__signature__ = inspect.Signature([
        parameter for parameter in inspect.signature(tool).parameters.values()
        if parameter.name not in {"edits", "dry_run"}
    ])


# Keep one public MCP tool while exposing rollout fields only in enabled schemas.
# FastMCP builds its schema at decoration time, so disabled fields need both a
# runtime-schema and introspection-signature projection after registration.
if not FEATURES.batch_replacement:
    _hide_disabled_batch_schema(mcp, local_ai_repo)


@mcp.tool()
@_instrumented_tool()
def local_ai_rag(
    action: RagAction,
    root: str = ".",
    workspace: str = "",
    query: str = "",
    top_k: int = 6,
    extra_fields: list[str] | None = None,
    max_response_tokens: int = 0,
    response_profile: str = "",
    reuse_key: str = "",
) -> dict[str, Any]:
    """Fallback semantic retrieval after deterministic and indexed repository evidence. Search defaults workspace to the explicit repository root. Use when: cheaper indexed paths are insufficient. Skip when: deterministic or indexed evidence already answers the question."""
    if not FEATURES.rag:
        return {"success": False, "unsupported": True, "error": "RAG backend is disabled (features.rag=false in config.toml)"}
    action = action.strip().lower().replace("-", "_")
    root = _client_root(root)
    effective_workspace = workspace.strip() or root
    if action == "index":
        return _compact(CLIENT.post("/api/rag/index", {"root": root, "workspace": effective_workspace}, timeout=_timeout("long")))
    if action == "search":
        return _compact(CLIENT.post("/api/rag/search", {
            "query": query, "workspace": effective_workspace, "top_k": max(1, min(top_k, 12)), "use_reranker": True,
        }))
    if action == "list":
        return _compact(CLIENT.get("/api/rag/workspaces"))
    if action == "docset_index":
        return _compact(CLIENT.post("/api/rag/docset/index", {
            "name": workspace or query or "default", "root": root,
        }, timeout=_timeout("long")))
    if action == "docset_search":
        return _compact(CLIENT.post("/api/rag/docset/search", {
            "name": workspace or "default", "query": query, "top_k": max(1, min(top_k, 12)),
        }))
    if action == "ingest_document":
        return _compact(CLIENT.post("/api/rag/ingest_document", {
            "workspace": workspace or "default", "content": query or root,
        }, timeout=_timeout("quick")), "status")
    if action == "ingest_diagram":
        return _compact(CLIENT.post("/api/rag/ingest_diagram", {
            "workspace": workspace or "default",
            "image_path": query or root,
            "caption": (extra_fields[0] if extra_fields else ""),
        }, timeout=_timeout("quick")), "status")
    return _invalid_action("local_ai_rag", action, tuple(RagAction.__args__), "Use RAG only when cheaper Local AI Hub evidence is insufficient.")



@mcp.tool()
@_instrumented_tool()
def local_ai_command(
    action: CommandAction,
    command: str = "",
    cwd: str = ".",
    timeout: int = 0,
    force: bool = False,
    task_id: str = "",
    criterion: str = "",
    auto_fix: bool = False,
    max_attempts: int = 3,
    stream: bool = False,
    stream_id: str = "",
    execution_id: str = "",
    snapshot: bool = False,
    rollback_on_failure: bool = False,
    patch: str = "",
    auto_rollback: bool = True,
    paths: list[str] | None = None,
    job_id: str = "",
    lint_action: str = "submit",
    extra_fields: list[str] | None = None,
    max_response_tokens: int = 0,
    response_profile: str = "",
    reuse_key: str = "",
) -> dict[str, Any]:
    """Bounded command broker for the main agent. MANDATORY for repeatable test/lint/typecheck/static-analysis/build/read-only commands whenever possible. Shared safe CLI broker. Actions: run, cancel, classify, discover, stats, repair_loop, auto_fix, run_affected, format, patch_and_verify, speculative_lint. Optional auto_fix=true or action=repair_loop runs autonomous self-healing test loop with safe rollback on failure. Speculative lint is disabled by default; explicit opt-in submits a read-only debounced job limited to caller-supplied changed paths and supports status/cancel through lint_action. Action patch_and_verify applies a unified diff, verifies with test command, and rolls back cleanly on error. Optional snapshot=true or rollback_on_failure=true captures git state and automatically reverts dirty changes if validation commands fail. Optional task_id and criterion link passing validation commands directly to evidence-backed VerificationReceipts. Optional stream=true or stream_id streams real-time stdout/stderr lines as command.log SSE events. Results are keyed by command + bounded repo state and duplicate runs coalesce across agents. Reuse fresh results. If run returns in_progress=true, DO NOT start the command natively or with force; continue independent work and retry later so the owner can populate the cache. Cancel a concurrent mutation only with its opaque execution_id from stats or run results. force=true is exceptional recovery/admin behavior, never a retry button. Use when: a repeatable test, lint, typecheck, build, analysis, or safe read-only command is needed. Skip when: no command is needed or a fresh cached result already answers it."""
    if not FEATURES.commands:
        return {"success": False, "unsupported": True, "error": "local_ai_command is disabled in configuration"}
    action = _resolve_action("command", action)
    eff_cwd = _client_root(cwd)
    host_timeout = _timeout("long")
    configured_command_timeout = int(CFG.get("commands", {}).get("timeout_seconds", 1800))
    requested_timeout = int(timeout or configured_command_timeout)
    # The command itself must finish before the MCP/HTTP caller's hard deadline so
    # there is always time to serialize the cached result instead of timing out at
    # exactly the same instant as the child process.
    effective_command_timeout = max(1, min(requested_timeout, max(1, int(host_timeout) - 30)))
    if action not in CommandAction.__args__:
        return _invalid_action("local_ai_command", action, tuple(CommandAction.__args__), "Use this broker for bounded commands; keep peer-agent orchestration in Codex.")
    if action == "speculative_lint":
        payload = {
            "action": str(lint_action or "submit").strip().lower().replace("-", "_"),
            "root": eff_cwd,
            "paths": paths or [],
            "command": command,
            "job_id": job_id or execution_id,
        }
        return _compact(CLIENT.post("/api/speculative-lint", payload, timeout=host_timeout), "command")
    if action.startswith("mock_server"):
        sub_act = "start" if action in {"mock_server", "mock_server_start"} else "stop" if action == "mock_server_stop" else "status"
        port_val = int(command) if (command and command.isdigit()) else 11440
        return _compact(CLIENT.post("/api/command/mock_server", {
            "action": sub_act, "root": eff_cwd, "port": port_val, "spec_path": criterion or None,
        }, timeout=host_timeout), "command")
    paths_payload = [command] if (action == "format" and command and not command.startswith("-")) else None
    return _compact(CLIENT.post("/api/command", {
        "action": action, "command": command, "cwd": eff_cwd, "root": eff_cwd,
        "timeout": effective_command_timeout, "force": force,
        "task_id": task_id, "criterion": criterion,
        "auto_fix": auto_fix, "max_attempts": max_attempts,
        "stream": stream, "stream_id": stream_id, "execution_id": execution_id,
        "snapshot": snapshot, "rollback_on_failure": rollback_on_failure,
        "paths": paths_payload,
        "patch": patch,
        "auto_rollback": auto_rollback,
    }, timeout=host_timeout), "command")


@mcp.tool()
@_instrumented_tool()
def local_ai_coord(
    action: CoordAction,
    root: str = ".",
    paths: list[str] | None = None,
    lease_id: str = "",
    key: str = "",
    value: str = "",
    query: str = "",
    command: str = "",
    ttl_seconds: int = 0,
    max_tokens: int = 0,
    task: str = "",
    task_id: str = "",
    contract: dict[str, Any] | None = None,
    checkpoint: dict[str, Any] | None = None,
    status: str = "",
    reason: str = "",
    record: dict[str, Any] | None = None,
    record_id: str = "",
    criterion: str = "",
    board_id: str = "",
    section: str = "",
    scope: str = "",
    scope_id: str = "",
    target_scope: str = "",
    approver: str = "",
    fingerprint: dict[str, Any] | None = None,
    tool_outcome: dict[str, Any] | None = None,
    extra_fields: list[str] | None = None,
    max_response_tokens: int = 0,
    response_profile: str = "",
    reuse_key: str = "",
    include_diagnostics: bool = False,
    clone_id: str = "",
    worktree_id: str = "",
    branch: str = "",
    repository_id: str = "",
    session_id: str = "",
    tenant: str = "",
    repository_revision: str = "",
    phase: str = "",
    focus: list[str] | None = None,
    preload_profile: str = "",
    since_hash: str = "",
) -> dict[str, Any]:
    """Cross-agent coordination for the main agent and bounded Hub workers. Actions: claim, release, leases, memo_put, memo_get, memo_search, memo_delete, task_create, task_get, task_checkpoint, task_rollback, task_transition, task_resume, task_list, task_complete, task_fail, task_heartbeat, memory_record, memory_get, memory_find, memory_delete, memory_promote, memory_reap, context_compile, verify_receipt, verify_completion, negative_knowledge_record, negative_knowledge_find, incident_decision, blackboard_update, blackboard_get, blackboard_list, blackboard_merge, blackboard_delete, swarm_dispatch, swarm_step, swarm_status, swarm_list, swarm_cancel. Claim overlapping edit paths before concurrent Hub work. Search/get memos before repeating expensive investigation and store concise reusable findings after discovery. Native peer subagents are coordinated by Codex rather than by this Hub tool. Use when: Hub workers share edit paths, leases, or reusable findings. Skip when: work is isolated and no shared Hub state or memo is involved."""
    if not FEATURES.coord:
        return {"success": False, "unsupported": True, "error": "local_ai_coord is disabled in configuration"}
    action = _resolve_action("coord", action)
    requested_root = str(root or "").strip()
    root = _client_root(root)
    if action == "task_sync":
        sync_act = status.lower() if status in ("export", "import") else "export"
        if sync_act == "export":
            return _compact(CLIENT.post("/api/agent-state/events/delta", {
                "action": "export", "stream_id": key or task_id or "", "after_seq": int(ttl_seconds or 0), "limit": max_tokens or 1000,
            }, timeout=_timeout("quick")), "status")
        else:
            return _compact(CLIENT.post("/api/agent-state/events/delta", {
                "action": "import", "events": record or [],
            }, timeout=_timeout("quick")), "status")
    if action == "task_zombie_reap":
        return _compact(CLIENT.post("/api/agent-state/tasks", {
            "action": "reap_expired", "auto_recover": True,
        }, timeout=_timeout("quick")), "status")
    if action == "task_cleanup_worktree":
        return _compact(CLIENT.post("/api/agent-state/tasks", {
            "action": "cleanup_worktree", "task_id": task_id or key or "",
        }, timeout=_timeout("quick")), "status")
    if action.startswith("task_"):
        task_kwargs = {
            "action": action, "task_id": task_id, "contract": contract,
            "checkpoint": checkpoint, "status": status, "reason": reason,
            "root": root, "ttl_seconds": ttl_seconds,
        }
        if action == "task_create" and task:
            task_kwargs["task"] = task
        return _compact(CLIENT.coord(
            **task_kwargs,
        ), "status")
    if action.startswith("memory_"):
        memory_kwargs = dict(
            action=action, record=record, record_id=record_id,
            target_scope=target_scope, approver=approver, key=key,
            value=value, query=query, root=(root if requested_root else ""), scope=scope, scope_id=scope_id,
            task_id=task_id, session_id=session_id, clone_id=clone_id,
            worktree_id=worktree_id, branch=branch, repository_id=repository_id, tenant=tenant,
        )
        if ttl_seconds is not None and ttl_seconds > 0:
            memory_kwargs["ttl_seconds"] = ttl_seconds
        return _compact(CLIENT.coord(**memory_kwargs), "status")
    if action.startswith("relation_"):
        return _compact(CLIENT.coord(
            action=action, source_entity=key or task_id or query or task,
            relation=reason or status or "relates_to",
            target_entity=value or target_scope or "",
            weight=float(ttl_seconds or 1.0) if ttl_seconds else 1.0,
            metadata=record,
            start_entity=key or task_id or query or task,
            max_depth=max_tokens or 2,
            limit=max_tokens or 50,
        ), "status")
    if action == "context_compile":
        since_hash = str(since_hash or fingerprint or record_id or status or "")
        return _compact(CLIENT.post("/api/agent-state/context", {
            "action": "compile", "task_id": task_id or query or value or key,
            "token_budget": max_tokens or ttl_seconds or 4000, "root": root,
            "changed_paths": paths or [],
            "since_hash": since_hash,
            "compact": True,
            "include_diagnostics": bool(include_diagnostics),
            "clone_id": clone_id,
            "worktree_id": worktree_id,
            "branch": branch,
            "repository_id": repository_id,
            "session_id": session_id,
            "repository_revision": repository_revision,
            "phase": phase,
            "focus": list(focus or []),
            "preload_profile": preload_profile,
        }, timeout=_timeout("context")), "context")
    if action == "verify_receipt":
        receipt_payload = checkpoint or {
            "task_id": task_id,
            "criterion": criterion or key or value or query,
            "passed": status.lower() in {"passed", "true", "1", "pass", "ok"} if status else True,
        }
        if checkpoint:
            receipt_payload = dict(checkpoint)
            receipt_payload.setdefault("task_id", task_id)
            receipt_payload.setdefault("criterion", criterion or key or value or query)
        if not str(receipt_payload.get("task_id", "")).strip() or not str(receipt_payload.get("criterion", "")).strip():
            return {
                "success": False,
                "terminal": True,
                "retryable": False,
                "error": "verify_receipt requires task_id and criterion",
            }
        return _compact(CLIENT.post("/api/agent-state/verification", {
            "action": "receipt", "receipt": receipt_payload, "root": root,
        }, timeout=_timeout("quick")), "verify")
    if action == "verify_completion":
        return _compact(CLIENT.post("/api/agent-state/verification", {
            "action": "completion", "task_id": task_id or query or value or key, "root": root,
            "repository_revision": repository_revision,
        }, timeout=_timeout("quick")), "verify")
    if action.startswith("negative_knowledge_"):
        sub_act = "record" if action == "negative_knowledge_record" else "find"
        return _compact(CLIENT.post("/api/agent-state/incidents", {
            "action": sub_act, "error_class": key or "AgentError", "message": value or query,
            "query": query or key or value, "root": root,
            "root_cause": reason, "verified_fix": status,
            "incident_id": record_id,
        }, timeout=_timeout("quick")), "status")
    if action == "incident_decision":
        return _compact(CLIENT.coord(
            action="incident_decision", fingerprint=fingerprint or {},
            tool_outcome=tool_outcome or {}, root=root,
        ), "status")
    if action in ("claim", "claim_batch"):
        return _compact(CLIENT.post("/api/leases/claim_batch", {
            "root": root, "paths": paths or [], "ttl_seconds": ttl_seconds or 900, "purpose": value or "agent edit",
        }))
    if action == "release":
        return _compact(CLIENT.post("/api/leases/release", {"lease_id": lease_id}))
    if action in ("renew", "lease_renew"):
        return _compact(CLIENT.post("/api/leases/renew", {"lease_id": lease_id, "ttl_seconds": ttl_seconds or 900}))
    if action == "leases":
        return _compact(CLIENT.get(f"/api/leases?root={quote(root)}"))
    if action == "memo_put":
        return _compact(CLIENT.post("/api/memory/put", {
            "root": root, "key": key, "value": value, "ttl_seconds": ttl_seconds or 604800,
        }))
    if action == "memo_get":
        return _compact(CLIENT.post("/api/memory/get", {"root": root, "key": key}))
    if action == "memo_search":
        return _compact(CLIENT.post("/api/memory/search", {"root": root, "query": query, "limit": 12}))
    if action == "memo_delete":
        return _compact(CLIENT.post("/api/memory/delete", {"root": root, "key": key}))
    if action.startswith("blackboard_"):
        sub = action[len("blackboard_"):]
        board = board_id.strip() or task_id or (key if sub in {"get", "list", "merge", "delete"} and not value else "") or "default"
        sec = section.strip() or (key if (value and sub == "delete") else (key or target_scope or ("" if sub in {"get", "delete"} else "main")))
        content = record if record is not None else (value or query)
        author = approver or "agent"
        exp_v = None
        if isinstance(checkpoint, dict) and "expected_version" in checkpoint:
            exp_v = checkpoint["expected_version"]
        elif status and status.isdigit():
            exp_v = int(status)
        return _compact(CLIENT.post("/api/agent-state/blackboard", {
            "action": sub, "board_id": board, "section": sec or None,
            "content": content, "author": author,
            "remote_sections": record or {},
            "expected_version": exp_v,
        }, timeout=_timeout("quick")), "status")
    if action == "swarm_dispatch":
        return _compact(CLIENT.post("/api/agent-state/swarm/dispatch", {
            "goal": task or value or query,
            "target_paths": paths or [],
            "test_command": command or target_scope or "",
            "author": approver or "agent",
            "root": root or ".",
        }, timeout=_timeout("quick")), "status")
    if action == "swarm_step":
        return _compact(CLIENT.post("/api/agent-state/swarm/step", {
            "swarm_id": task_id or key or "",
            "role": target_scope or "Coder",
            "action": query or "submit_patch",
            "payload": record or ({"value": value} if value else {}),
        }, timeout=_timeout("quick")), "status")
    if action == "swarm_status":
        sid = task_id or key or ""
        return _compact(CLIENT.get(f"/api/agent-state/swarm/{quote(sid)}", timeout=_timeout("quick")), "status")
    if action == "swarm_list":
        st_param = f"?state={quote(status)}" if status else ""
        return _compact(CLIENT.get(f"/api/agent-state/swarm{st_param}", timeout=_timeout("quick")), "status")
    if action == "swarm_cancel":
        sid = task_id or key or ""
        return _compact(CLIENT.post("/api/agent-state/swarm/cancel", {"swarm_id": sid, "reason": reason or value or "cancelled by agent"}, timeout=_timeout("quick")), "status")
    if action == "worktree_lease":
        return _compact(CLIENT.coord(action="worktree_lease", root=root, branch=key or task or task_id or None), "status")
    if action == "worktree_release":
        del_branch = status.lower() not in {"false", "0", "no"} if status else True
        return _compact(CLIENT.coord(action="worktree_release", root=root, worktree_path=value or lease_id or key or "", delete_branch=del_branch, branch=key or task or task_id or None), "status")
    if action == "pubsub_publish":
        return _compact(CLIENT.coord(action="pubsub_publish", topic=key or target_scope or "default", message=value or query or record or {}, publisher=approver or "agent", root=root), "status")
    if action == "pubsub_poll":
        return _compact(CLIENT.coord(action="pubsub_poll", topic=key or target_scope or "default", subscriber=approver or "agent", limit=ttl_seconds or 50, root=root), "status")
    if action == "simulate_merge":
        return _compact(CLIENT.coord(action="simulate_merge", root=root, source_branch=key or task_id or value, target_branch=target_scope or "HEAD"), "status")
    if action in {"curate_dataset", "curate_training_dataset"}:
        return _compact(CLIENT.post("/api/agent-state/tasks", {
            "action": "curate_dataset",
            "output_path": key or value or query or "training_dataset.jsonl",
            "min_receipts": int(status) if (status and status.isdigit()) else 1,
            "format": target_scope or "jsonl",
        }, timeout=_timeout("quick")), "status")
    return _invalid_action("local_ai_coord", action, tuple(CoordAction.__args__), "Use coordination for bounded shared state; the main agent remains the owner of final integration.")


@mcp.tool()
@_instrumented_tool()
def local_ai_work(
    action: WorkAction,
    root: str = ".",
    task: str = "",
    work_id: str = "",
    acceptance_criteria: list[str] | None = None,
    constraints: list[str] | None = None,
    mode: str = "execute",
    permissions: dict[str, Any] | None = None,
    budget: dict[str, Any] | None = None,
    timeout_seconds: float = 90.0,
    answer: str = "",
    response_profile: str = "compact",
    return_fields: list[str] | None = None,
    max_output_tokens: int = 0,
    keep_failed_workspace: bool = False,
    extra_fields: list[str] | None = None,
    max_response_tokens: int = 0,
    reuse_key: str = "",
) -> dict[str, Any]:
    """Whole-task local execution with durable verified handoff. See dynamic description for policy and response projection."""
    if not getattr(FEATURES, "work_orchestrator", False):
        return {"success": False, "unsupported": True, "error": "local_ai_work is disabled in configuration"}
    action = action.strip().lower().replace("-", "_")
    if action not in WorkAction.__args__:
        return _invalid_action("local_ai_work", action, tuple(WorkAction.__args__), "Use submit for a closed task and one bounded wait/get for handoff.")
    payload: dict[str, Any] = {
        "action": action, "root": _client_root(root), "task": task, "work_id": work_id,
        "acceptance_criteria": acceptance_criteria or [], "constraints": constraints or [], "mode": mode,
        "permissions": permissions or {}, "budget": budget or {}, "timeout_seconds": timeout_seconds, "answer": answer,
        "response_profile": response_profile, "return_fields": return_fields or [], "max_output_tokens": max_output_tokens,
        "keep_failed_workspace": keep_failed_workspace,
    }
    return _compact(CLIENT.post("/api/work-orders", payload, timeout=_timeout("long") if action in {"wait"} else _timeout("quick")), "status")


@mcp.tool()
@_instrumented_tool()
def local_ai_artifact(artifact_id: str, offset: int = 0, max_chars: int = 4000, section: str = "", extra_fields: list[str] | None = None, max_response_tokens: int = 0, response_profile: str = "", reuse_key: str = "", binary: bool = False) -> dict[str, Any]:
    """Fetch one needed artifact section or exact evidence slice. Evidence IDs start with E. Use when: exact source or evidence text is required after indexed discovery. Skip when: no source slice is needed or the existing compact result is sufficient."""
    if not FEATURES.artifacts:
        return {"success": False, "unsupported": True, "error": "local_ai_artifact is disabled in configuration"}
    if artifact_id.startswith("E"):
        return _compact(CLIENT.post("/api/evidence/get", {"evidence_id": artifact_id, "verify": True}), "artifact")
    result = CLIENT.post("/api/artifact/get", {
        "artifact_id": artifact_id,
        "offset": offset,
        "max_chars": max(512, min(max_chars, 12000)),
        "section": section,
        "binary": bool(binary),
    })
    if binary and isinstance(result, dict) and result.get("success"):
        result = {key: value for key, value in result.items() if key != "data_base64"}
        result["binary_payload"] = "available through /api/artifact/get with binary=true"
    return _compact(result, "artifact")


# Unregister tools that are disabled in current configuration so MCP clients do not receive them
for _disabled in FEATURES.disabled_tools:
    if hasattr(mcp, "_tool_manager") and hasattr(mcp._tool_manager, "_tools"):
        mcp._tool_manager._tools.pop(_disabled, None)
    if hasattr(mcp, "tools") and isinstance(mcp.tools, list):
        mcp.tools = [t for t in mcp.tools if getattr(t, "__name__", "") != _disabled]


# Ensure tool descriptions registered in FastMCP carry the complete dynamic guidance
_all_desc_map = {
    "local_ai_status": _desc_status,
    "local_ai_task": _desc_task,
    "local_ai_repo": _desc_repo,
    "local_ai_rag": _desc_rag,
    "local_ai_command": _desc_command,
    "local_ai_coord": _desc_coord,
    "local_ai_artifact": _desc_artifact,
    "local_ai_work": _desc_work,
}
if hasattr(mcp, "_tool_manager") and hasattr(mcp._tool_manager, "_tools"):
    for _tool_name, _desc_fn in _all_desc_map.items():
        if _tool_name in mcp._tool_manager._tools:
            mcp._tool_manager._tools[_tool_name].description = _desc_fn()

if hasattr(mcp, "_tool_manager") and hasattr(mcp, "_mcp_server") and mcp_types is not None:
    from local_ai_hub.json_utils import dumps as json_dumps

    async def _compact_mcp_call_tool(name: str, arguments: dict[str, Any]) -> Any:
        try:
            raw = await mcp._tool_manager.call_tool(name, arguments, convert_result=False)
        except Exception as exc:
            # FastMCP/Pydantic validation errors otherwise escape as a raw
            # traceback. Keep the public MCP boundary machine-readable and
            # bounded; callers can correct the advertised enum/schema.
            text = str(exc).strip().splitlines()[0][:500]
            payload = {
                "success": False,
                "status_code": 400 if type(exc).__name__ == "ValidationError" else 500,
                "terminal": True,
                "retryable": False,
                "error": f"MCP input validation failed: {text}" if type(exc).__name__ == "ValidationError" else f"MCP tool failed: {text}",
            }
            return ([mcp_types.TextContent(type="text", text=json_dumps(payload))], payload)
        if isinstance(raw, (dict, list)):
            return ([mcp_types.TextContent(type="text", text=json_dumps(raw))], raw)
        if isinstance(raw, str):
            return [mcp_types.TextContent(type="text", text=raw)]
        return await mcp._tool_manager.call_tool(name, arguments, convert_result=True)

    mcp.call_tool = _compact_mcp_call_tool
    mcp._mcp_server.call_tool(validate_input=False)(_compact_mcp_call_tool)


if __name__ == "__main__":
    mcp.run()
