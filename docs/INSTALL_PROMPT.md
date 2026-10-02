# Generic Agent Installation & Integration Prompt for Local AI Hub

Use this prompt with any AI coding agent (Claude Code, Codex, Cursor, Windsurf, Gemini, Roo, GitHub Copilot) in any repository to bootstrap Local AI Hub, configure background services and MCP, and integrate the tool and token economy policies into repository instructions.

---

```markdown
# TASK: Install, Verify, and Integrate Local AI Hub in Environment & Repository

Repository and reasoning request contracts:
- `local_ai_repo(action="search", path=...)` restricts every candidate and fallback result to the existing file or directory inside `root`; invalid scopes fail terminally. Single compound identifiers such as `border-top` are literal content queries, not separate `border`/`top` matches. Use source/artifact tools to read a file by its path.
- `local_ai_task(action="reason")` uses non-empty `task`, falling back to `prompt`; empty assignments fail before inference. `context` remains separate evidence.
- Model responses remain advisory. Inspect `semantic_quality`, `quality_warning`, and `bypass_reason`; `success=true` alone does not establish factual or logical correctness.
- After updating runtime modules, restart the HTTP service and reconnect MCP clients so their separate processes load the new adapter.

You are an autonomous AI engineering agent. Your task is to install and configure **Local AI Hub** on this system (if not already installed or running), verify all components (service, configured local backends, token economy suite, code intelligence), ensure MCP integration, and inject the canonical tool and token economy policies into this repository's agent instructions. Ollama is disabled unless explicitly opted in.

Hub adoption telemetry is aggregate-only: retain normalized tool/action/intent/outcome plus time, latency, and output-size buckets. Never send prompts, source, secrets, absolute paths, or raw request records. Mark a bypass only when the client explicitly signals one; do not infer it from missing Hub calls.

Execute the following phases deterministically:

---

### PHASE 1: ENVIRONMENT PROBE & CLONE

1. **Check Existing Installation**:
   - Standard path:
     - Windows: `%USERPROFILE%\.local-ai-hub` (or `$HOME\.local-ai-hub`)
     - Linux / macOS: `~/.local-ai-hub`
2. **Clone if Missing**:
   If the directory does not exist or lacks `tools/setup.py`, clone the repository:
   - Git command:
     ```bash
     git clone https://github.com/Harfusha/local-ai-hub.git "$HOME/.local-ai-hub"
     ```
     *(On Windows PowerShell: `git clone https://github.com/Harfusha/local-ai-hub.git "$HOME\.local-ai-hub"`)*

---

### PHASE 2: RUN BOOTSTRAP INSTALLATION

Run the platform installer from the repository root. This automatically configures Python 3.11+, virtual environment, Token Economy tools, backend-appropriate local model support, Serena/CodeGraphContext environments, global MCP configs, and background supervisor. Ollama is installed or pulled only when `[ollama].enabled = true`, `server.auto_start_ollama = true`, and `llama_cpp.fallback_to_ollama = true`; it wins provider selection whenever enabled. With Ollama disabled, `llama_cpp.mode = "on"` explicitly provisions the pinned llama.cpp runtime and default Qwen 1.5B model under `server.state_dir`, then supervises the loopback server; Intel Windows tries SYCL and falls back to CPU. `mode = "auto"` detects an existing endpoint only; `"off"` disables llama.cpp. Setup must never switch providers silently. Set `features.vision = false` to remove vision capability and its model pull. The default install deploys the `token-economizer` skill and registers its CLI directory on the user's persistent PATH; verify both after setup, then open a new terminal. Do not pass `--skip-token-economy` or `--skip-companion-skills` for the standard install.

When `[ollama].enabled = false`, setup must honor the llama.cpp mode: managed install only for `on`, external health check only for `auto`, and no runtime for `off`. Never start `ollama serve` as an implicit fallback.

- **Windows (PowerShell)**:
  ```powershell
  powershell -ExecutionPolicy Bypass -File "$HOME\.local-ai-hub\install.ps1" -Profile auto
  ```
- **macOS / Linux (Bash)**:
  ```bash
  bash "$HOME/.local-ai-hub/install.sh" --profile auto
  ```

*Note*: If an explicitly configured local backend or heavy model is unavailable, the installer still sets up deterministic tools and built-in indexers and reports the unavailable optional backend.

---

### PHASE 3: VERIFY SERVICE & HEALTH

1. **Check Service Endpoint**:
   Probe HTTP health (default port 11435):
   ```bash
   curl -s http://127.0.0.1:11435/health
   ```
   Must return `{"status":"ok", ...}` with the hub version.
2. **If Service Not Running**:
   Start it via python venv:
   - Windows:
     ```powershell
     & "$HOME\.local-ai-hub\.venv\Scripts\python.exe" "$HOME\.local-ai-hub\tools\service.py" start
     ```
   - Linux / macOS:
     ```bash
     "$HOME/.local-ai-hub/.venv/bin/python" "$HOME/.local-ai-hub/tools/service.py" start
     ```
3. **Run Doctor Diagnostic**:
   - Windows:
     ```powershell
     & "$HOME\.local-ai-hub\.venv\Scripts\python.exe" "$HOME\.local-ai-hub\tools\doctor.py"
     ```
   - Linux / macOS:
     ```bash
     "$HOME/.local-ai-hub/.venv/bin/python" "$HOME/.local-ai-hub/tools/doctor.py"
     ```
4. **Hardware Acceleration Verification (iGPU / NPU)**:
   - On Windows Intel systems, managed llama.cpp setup tries SYCL0 and retries once on CPU if initialization fails; check `hubctl status` for the active device. `mode = "auto"` uses only a pre-existing endpoint. Do not set `OLLAMA_VULKAN` for Intel llama.cpp inference.
   - Other managed supported platforms use the pinned CPU runtime; GPU builds can be configured as external loopback endpoints. Do not claim Ollama is active when llama.cpp was selected.
   - If an NPU (Intel AI Boost / AMD XDNA) or Intel iGPU is present:
     Ensure OpenVINO dependencies are installed in the venv only when active hardware/configuration selects OpenVINO for embeddings or reranking. Do not install OpenVINO on NVIDIA-only systems merely because the feature permission is true:
     ```powershell
     & "$HOME\.local-ai-hub\.venv\Scripts\pip.exe" install -r "$HOME\.local-ai-hub\requirements-openvino.txt"
     & "$HOME\.local-ai-hub\.venv\Scripts\python.exe" "$HOME\.local-ai-hub\tools\prefetch_openvino.py"
     ```

---

### PHASE 4: INJECT INSTRUCTIONS INTO CURRENT REPOSITORY

Inspect current repository root for existing agent instruction files:
- Universal: `AGENTS.md`
- Claude Code: `CLAUDE.md`
- Cursor: `.cursorrules` or `.cursor/rules/local-ai.mdc`
- Windsurf: `.windsurfrules`
- Gemini: `GEMINI.md`
- Copilot: `.github/copilot-instructions.md`

**Action**:
1. If no instruction file exists, create `AGENTS.md` at repository root.
2. In `AGENTS.md` (and any other active instruction files detected above), ensure the following two exact policy blocks are present. If older versions exist, update them between the markers; otherwise append them:

```markdown
<!-- BEGIN LOCAL AI HUB TOOL POLICY -->
Trigger map:
- repository navigation/symbols/impact: `local_ai_repo`
- test/lint/typecheck/build commands: `local_ai_command`
- exact source/log/evidence slice: `local_ai_artifact`
- task contracts, ownership leases, checkpoints, verification receipts, and receipt-gated completion: `local_ai_coord`
- semantic retrieval after indexed paths are insufficient: `local_ai_rag`
- semantic generation, exploration, reasoning, review, second opinion and compression: use `local_ai_task(action="delegate"|"explore"|"reason"|"review"|"second_opinion"|"compress")` after any needed evidence. Deterministic/indexed tools remain for exact facts, symbols, diff and tests; they do not replace semantic local-model work. `local_ai_repo(action="solve")` preserves one bounded local pass when its task text explicitly requests explore/explain/why/compare/second-opinion semantics, even when exact evidence is strong. Failure diagnosis is disabled by default; enable `features.local_diagnostic_dispatch=true` only after low-confidence deterministic command parsing with an artifact reference plus narrow preview, never raw logs. Automatic local inference never handles architecture, security, mutations, or open-ended coding.
- Semantic handoff is mandatory: after deterministic/indexed evidence, planning, interpretation, synthesis, generation, review, compression, and second-opinion work must call `local_ai_task` before cloud reasoning. The cloud agent integrates the bounded local result and does not redo semantic work. If local inference is unavailable or intentionally excluded by a permitted boundary, report the bypass through `local_ai_status(adoption_signal="bypassed", target_tool="local_ai_task", target_action="reason")`. Preserve exceptions for architecture, security, mutations, open-ended coding, exact evidence, and verification.
- closed, verified handoff work: `local_ai_work`; skip micro-edits and live discussion

Recipes (guidance, not gates):
- Recipe — Explore: preprocess once, use the cheapest repository action, fetch only required evidence slices.
- Recipe — Change: gather indexed evidence, use `local_ai_repo(action="solve")` before edits, claim `local_ai_coord` leases for overlapping paths, then run indexed impact/review before validation.
- Recipe — Validate: route repeatable commands through `local_ai_command`, reuse cached results, use `review_diff` or `security_audit` when relevant.
- Recipe — Retrieve: use `local_ai_rag` only after deterministic/indexed paths are exhausted.
- A recipe step may be skipped when irrelevant; one bounded fallback is allowed when Hub is unavailable.

Delegation is the default for any task with useful bounded independent work.

- Use `local_ai_task` for bounded semantic generation, reasoning, review, independent second opinions and compression. Use `qwen2.5-coder:1.5b` only for quick/simple tasks, `qwen2.5-coder:3b` for ordinary and more involved work, `qwen2.5-coder:7b` for hard reasoning, `qwen3.5:9b` for extreme reasoning in the balanced profile, and `qwen2.5-coder:0.5b` only for preprocessing.
- Use the native Codex `multi_agent_v1__spawn_agent` path only for useful independent bounded work or an explicit Codex-subagent request.
- Codex controls each subagent's scope, `allow_write`, workspace/worktree, timeout, cancellation, sandbox, and integration.
- Do not duplicate the same scope across agents. Keep final decisions, edits, and integration in Codex.
- Skip delegation only for trivial tasks, pure evidence lookups, security/privacy constraints, or no useful independent scope.

Routing hierarchy: the main agent is the orchestrator, planner, integrator and final owner. Use Local AI Hub first for its own precise, bounded microtasks: deterministic facts, indexed/search retrieval, preprocess artifacts, targeted impact/review/security checks, safe commands, compression, local-model synthesis and second opinions. Native `multi_agent_v1__spawn_agent` is used only for useful independent bounded work or an explicit Codex-subagent request. It is not routed or managed by Local AI Hub.

READ-ONLY AUDIT CONTRACT:
- Read-only means no Git writes, `git worktree add` or removal, dependency installation, builds/imports, generated artifacts, or other workspace side effects. Never label such work read-only when any of these occur; split validation into a separately owned, explicitly side-effecting task.
- Before native discovery or validation, retain preceding Hub result with action, absolute root, status, cache/in_progress state, and evidence IDs. Native discovery is fallback-only after one bounded terminal Hub failure.
- Native validation fallback is allowed only after `local_ai_command` returns `terminal=true` and `retryable=false`; run one bounded fallback, state side effects/owner, and never repeat identical commands.
- Codex controls subagent permissions per task. Native Codex subagents may write only when Codex explicitly enables it, and write work stays in the assigned workspace/worktree. Codex remains integrator.
- Do not run parallel duplicate commands or scopes. Tool labels are not evidence; preserve exact action, arguments, result, and ownership in the audit record.

For every non-trivial repository task, use Local AI Hub before broad native discovery or repeatable validation. Keep one stable absolute project root. On the first task for that root call `local_ai_repo(action="preprocess", root=ABS_ROOT)` exactly once, then continue immediately; preprocessing is asynchronous, so never poll/wait/force-refresh it.

Adoption gate: `local_ai_command` alone is never sufficient for a repository task. The first useful Hub operation must be `local_ai_repo` (preprocess plus the cheapest applicable deterministic/code-index/search/context action); use the command broker only for commands, after repository evidence exists. For implementation, diagnosis, refactoring or complex review, call `local_ai_repo(action="solve")` after evidence and before native edits. After edits, use the applicable indexed impact/review/security/evidence action before final validation.

Default context contract: use `local_ai_coord(action="context_compile", task_id=...)` before non-trivial resume, planning, edit, review or test, and use `local_ai_repo(action="context", task_id=...)` as the repository-facing projection. The compile endpoint returns one bounded contract joining Agent OS state with indexed repository evidence, revision, changed paths, evidence IDs, freshness, receipts, `context_id`, `etag`, omitted sections and `next_action`; `local_ai_task(task_id=...)` consumes it automatically. Never treat a partial/stale/mismatched-revision result as complete. Guarded fields include `phase`, `focus`, `preload_profile`, `changed_paths`, `since_hash`, `approval`, and `override_reason`, alongside task/budget/workspace and compact response controls. Scope or drift overrides require an explicit `override_reason` and approval when requested. Raw model/debug fields are omitted unless explicitly requested through `extra_fields`.

Cheapest path for repository evidence: deterministic -> code_index/search -> semantic/graph -> context/solve -> RAG. Semantic generation, reasoning, review, independent second opinions and compression use `local_ai_task` after any needed evidence.
 Stop escalating as soon as a cheaper layer provides enough evidence. Do not fan out overlapping retrieval layers in parallel for the same question. Before native `find`/`rg`/`grep`/recursive glob/tree or opening more than two files for discovery, use that hub path first. Reuse fresh evidence IDs, artifact slices, memos and cache hits;
 do not repeat the same hub action with the same root/query while repository state is unchanged.

Treat result state as a protocol: `cache_hit`/`coalesced` means reuse the result; `in_progress=true` means another owner is doing identical work, so never duplicate it; `retryable`/429/503 means back off and do independent work; `degraded`/`stale` means verify only the affected path/slice; native fallback requires `terminal=true` and `retryable=false`. Mutations never cache or single-flight. Never turn a transient result into larger timeouts, force refreshes, or polling loops.

Route test/lint/typecheck/build/read-only commands through `local_ai_command` before running them natively. If it returns `in_progress=true`, do not launch a duplicate command. Before an expensive `solve`/model call, search coordination memos for reusable findings. For overlapping multi-agent edits use `local_ai_coord` leases and store concise reusable discoveries as memos.
 After edits, use indexed impact/review plus targeted cached validation; do not rerun broad discovery merely because files changed. `force` and `preprocess_refresh` are recovery/admin controls, never retry buttons. If an optional backend degrades, accept the hub's deterministic/index fallback. If the hub itself is unavailable, make one bounded health/retry attempt, then fall back to native tools. Never loop on health, status, preprocessing, model startup, a failing backend, or an identical command.

Selection guide: `local_ai_repo` for bounded repository facts and checks (including `review_diff` and `security_audit`), `local_ai_command` for bounded repeatable commands, `local_ai_task` for small local-model work and second opinions, `local_ai_work` only for a complete closed task with verified handoff, `local_ai_rag` only after cheaper indexed evidence, `local_ai_artifact` for exact slices, `local_ai_coord` for task contracts, leases, checkpoints, and receipts.

For non-trivial multi-step work, create a `local_ai_coord` task contract first, claim overlapping paths, checkpoint phase changes, attach validation receipts, and complete only after receipt verification passes.

Rollout controls: `features.enriched_search`, `features.batch_replacement`, `features.diagnostic_artifacts`, and `features.local_diagnostic_dispatch` are enabled in the installed default profile. They remain independently switchable for rollback or a controlled pilot. Compare `/api/adoption` token, latency, first-pass validation, terminal failure, and native fallback metrics. Roll back immediately by setting the affected flag to `false`, restarting Hub, and running `python tools/hubctl.py generate`. Disabled flags return structured unavailable before work starts; malformed values stay disabled. `local_diagnostic_dispatch=true` may retain only its bounded failure preview for one local diagnosis; never raw output and no `diagnostic_artifacts=true` dependency.

Batch edits require `features.batch_replacement=true`. Then use `local_ai_repo(action="batch_replace", edits=[...], dry_run=true)` for preview. `staged` is not batch dry-run and is never forwarded. Each edit needs exact target text that matches once. The engine preflights all edits, rolls back write failures, and never auto-commits. Set `dry_run=false` only after review.

For durable work, create one `local_ai_coord(action="task_create")` contract, record phase changes with `local_ai_coord(action="task_checkpoint")`, then use one bounded wait on the durable job. Do not poll status loops.

Local model policy: use `qwen2.5-coder:0.5b` only for preprocessing, `qwen2.5-coder:1.5b` for bounded extraction/simple work, `qwen2.5-coder:3b` for ordinary bounded reasoning, `qwen2.5-coder:7b` for hard reasoning, and `qwen3.5:9b` for extreme reasoning in the balanced profile. The shared operation-specific prompt contract supplies capability limits, evidence IDs, acceptance criteria and no-invention rules. Every semantic result is advisory; unrelated paths, queued/empty/malformed/error output remains visible with `advisory_only=true`, `semantic_quality` and a bounded `quality_warning`/`bypass_reason`, never silently rejected. Deterministic and indexed Hub actions run first for exact facts, symbols, diff and tests; semantic generation, reasoning, review, independent second opinions and compression use `local_ai_task` only after evidence. See `docs/MODEL_CAPABILITIES.md`.
<!-- END LOCAL AI HUB TOOL POLICY -->

<!-- BEGIN TOKEN ECONOMY POLICY -->
- Before any repository task, load and follow the `token-economizer` skill when it is installed; this trigger applies even under deadline pressure.
- Zero full-file dumping: Never read files >80 lines in their entirety. Use `repo-map` for high-level structure, `grep-ast <pattern> <file>`, targeted line slices, or `local_ai_artifact(action="slice")`.
- Fast code search: Use `rg` (`ripgrep`) with `-m 5` / bounded matches and `fd` for file finding before opening files.
- AST & structural code search: Use `ast-grep` (`sg`), Serena LSP (`find_symbol`, `find_referencing_symbols`), or `local_ai_repo(action="code_index")` before opening files.
- Context compression & token measurement: Use `repomix --compress` or `files-to-prompt -c` for repo snapshots. Use `tokcount` to measure exact tokens.
- Bounded command outputs: Route tests and builds through `local_ai_command`; use `trim-run` only with bundled `tokcount`/`repo-map`, read-only `rg`/`fd`/`grep-ast`, or stdin pipelines such as `git log | trim-run`. Use `jq` for JSON.
- Surgical edits: Prefer targeted block replacements over rewriting entire files.
- Local model delegation: Use `qwen2.5-coder:1.5b` only for quick/simple microtasks, `qwen2.5-coder:3b` for ordinary and more involved work, `qwen2.5-coder:7b` for hard reasoning, and `qwen3.5:9b` for extreme reasoning in the balanced profile via `local_ai_task`; reserve `qwen2.5-coder:0.5b` for preprocessing.
<!-- END TOKEN ECONOMY POLICY -->
```

---

### PHASE 5: REPOSITORY / WORKSPACE MCP REGISTRATION (IF APPLICABLE)

If this workspace uses repository-scoped MCP (e.g., `.cursor/mcp.json` or `.vscode/mcp.json`), ensure the `local-ai` server entry is defined:

```json
{
  "mcpServers": {
    "local-ai": {
      "command": "<HOME>/.local-ai-hub/.venv/bin/python",
      "args": ["-m", "local_ai_hub.mcp_server"],
      "cwd": "<HOME>/.local-ai-hub",
      "env": {
        "LOCAL_AI_AGENT": "generic",
        "LOCAL_AI_AGENT_PROFILE": "generic",
        "LOCAL_AI_CONFIG": "<HOME>/.local-ai-hub/config.toml",
        "PYTHONPATH": "<HOME>/.local-ai-hub/src"
      }
    }
  }
}
```
*(On Windows, replace paths with Windows equivalents: `...\\Scripts\\python.exe`, backslashes, etc.)*

---

### PHASE 6: INITIAL REPO INDEXING & HANDSHAKE

Trigger one-time asynchronous preprocessing on the current repository root:
- If `local_ai_repo` MCP tool is available in your active session:
  Execute `local_ai_repo(action="preprocess", root="<ABSOLUTE_CURRENT_REPO_PATH>")`.
- Or trigger via HTTP loopback:
  ```bash
  curl -s -X POST http://127.0.0.1:11435/api/repo/preprocess \
    -H "Content-Type: application/json" \
    -d "{\"root\": \"$(pwd)\"}"
  ```

---

### PHASE 7: STATUS REPORT

Output concise confirmation with:
1. Local AI Hub version and service status (`online` / `offline`).
2. Tools verified (`rg`, `fd`, `ast-grep`, `repomix`, `tokcount`, `trim-run`).
3. Configured local backend and model status (Ollama must be reported as disabled unless explicitly opted in).
4. Instruction files created/modified in this repo.
5. Readiness confirmation.
```
### MCP response economy

Keep the generated MCP response contract enabled. Hub responses are aggregate-bounded by default; agents should prefer `response_profile="minimal"` or `"compact"`, pass a stable `reuse_key` for repeated logical queries, reuse cache-hit/pointer responses, and fetch exact detail only through `local_ai_artifact` slices. Do not weaken the budget to expose broad command output or repository text.
