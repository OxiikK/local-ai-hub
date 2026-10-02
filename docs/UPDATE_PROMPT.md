# Generic Agent Update & Upgrade Prompt for Local AI Hub

Use this prompt with any AI coding agent (Claude Code, Codex, Cursor, Windsurf, Gemini, Roo, GitHub Copilot) in any repository to upgrade an existing Local AI Hub installation from V3 to V4 from the canonical `master` branch, upgrade dependencies, restart services, and refresh the repository's instruction policies.

Preserve profile-aware model routing when refreshing repository instructions: integrated uses qwen2.5-coder:0.5b preprocessing, 1.5b quick, 3b involved, 7b hard reasoning and qwen3-vl:4b vision; balanced uses qwen2.5-coder:1.5b preprocessing, 3b fast, 7b ordinary/hard work and qwen3.5:9b only for extreme reasoning and vision; high uses 1.5b/3b/7b/9b; max uses 1.5b/3b/9b for heavy routes.

---

```markdown
# TASK: Upgrade Local AI Hub from V3 to V4 and Refresh Repository Instructions

Repository and reasoning request contracts:
- `local_ai_repo(action="search", path=...)` restricts every candidate and fallback result to the existing file or directory inside `root`; invalid scopes fail terminally. Single compound identifiers such as `border-top` are literal content queries, not separate `border`/`top` matches. Use source/artifact tools to read a file by its path.
- `local_ai_task(action="reason")` uses non-empty `task`, falling back to `prompt`; empty assignments fail before inference. `context` remains separate evidence.
- Model responses remain advisory. Inspect `semantic_quality`, `quality_warning`, and `bypass_reason`; `success=true` alone does not establish factual or logical correctness.
- After updating runtime modules, restart the HTTP service and reconnect MCP clients so their separate processes load the new adapter.

You are an autonomous AI engineering agent. Your task is to upgrade **Local AI Hub** from V3 to V4 on this system using the canonical `master` branch, update dependencies, restart the background service, re-verify health, and refresh the Local AI Hub Tool Policy and Token Economy Policy in this repository.

Preserve privacy-safe adoption telemetry: it contains only normalized aggregate tool/action/intent/outcome, time, latency, and output-size buckets. Never migrate or expose prompts, source, secrets, absolute paths, or raw request records. Bypasses require an explicit client signal.

Follow these execution phases:

---

### PHASE 1: LOCATE INSTALLATION & FAST-FORWARD TO V4 MASTER

1. **Locate Installation Directory**:
   - Windows: `$HOME\.local-ai-hub` (or `%USERPROFILE%\.local-ai-hub`)
   - Linux / macOS: `~/.local-ai-hub`
2. **Protect Local Changes**:
   - Run `git status --short` before changing branches.
   - If the installation worktree is dirty, preserve the changes and stop for operator review. Never use `git reset --hard`, `git checkout -- .`, or discard local configuration/source changes.
3. **Fetch V4 from the Canonical Master Branch**:
   - V4 is published on `origin/master`. Do not use the old `codex/v4.0-hardening` feature branch for installations.
   - Fetch `master`, switch to local `master`, and fast-forward only:
     ```bash
     git -C "$HOME/.local-ai-hub" fetch --prune origin
     git -C "$HOME/.local-ai-hub" switch master
     git -C "$HOME/.local-ai-hub" pull --ff-only origin master
     ```
     *(On Windows PowerShell: `git -C "$HOME\.local-ai-hub" fetch --prune origin`, then `git -C "$HOME\.local-ai-hub" switch master`, then `git -C "$HOME\.local-ai-hub" pull --ff-only origin master`.)*
   - If local `master` does not exist, create it only as a tracking branch from `origin/master`; never overwrite an existing local branch.

---

### PHASE 2: RUN UPDATE / RE-SETUP

Run the platform installer with the active hardware profile to apply dependency updates, regenerate MCP schemas/manifests, and update skills. Inspect provider config first: Ollama is selected when `[ollama].enabled = true` and its install/model pulls still require the existing explicit opt-ins (`server.auto_start_ollama = true`, `llama_cpp.fallback_to_ollama = true`). If Ollama is disabled, `llama_cpp.mode = "on"` provisions the pinned llama.cpp runtime and default Qwen 1.5B model under `server.state_dir`; `"auto"` uses only a configured existing endpoint; `"off"` disables it. Never switch providers silently or start `ollama serve` as a fallback. Pull `models.vision` only for an explicitly enabled Ollama backend and `features.vision = true`:

- **Windows (PowerShell)**:
  ```powershell
  powershell -ExecutionPolicy Bypass -File "$HOME\.local-ai-hub\install.ps1" -Profile auto
  ```
- **Linux / macOS (Bash)**:
  ```bash
  bash "$HOME/.local-ai-hub/install.sh" --profile auto
  ```

---

### PHASE 3: RESTART SERVICE & VERIFY HEALTH

1. **Restart Hub Service**:
   - Windows:
     ```powershell
     & "$HOME\.local-ai-hub\.venv\Scripts\python.exe" "$HOME\.local-ai-hub\tools\service.py" restart
     ```
   - Linux / macOS:
     ```bash
     "$HOME/.local-ai-hub/.venv/bin/python" "$HOME/.local-ai-hub/tools/service.py" restart
     ```
2. **Verify Health Endpoint**:
   ```bash
   curl -s http://127.0.0.1:11435/health
   ```
   Must return `{"status":"ok", ...}` with the updated package version.
3. **Run Doctor Diagnostic**:
   - Windows:
     ```powershell
     & "$HOME\.local-ai-hub\.venv\Scripts\python.exe" "$HOME\.local-ai-hub\tools\doctor.py"
     ```
   - Linux / macOS:
     ```bash
     "$HOME/.local-ai-hub/.venv/bin/python" "$HOME/.local-ai-hub/tools/doctor.py"
     ```
4. **Hardware Acceleration Check (iGPU / NPU)**:
   - On Windows Intel systems, managed llama.cpp setup tries SYCL0 and retries on CPU if GPU startup fails. `mode = "auto"` checks an existing endpoint only. Do not set `OLLAMA_VULKAN` for Intel llama.cpp inference.
   - Other supported platforms use the pinned CPU build for managed llama.cpp. NVIDIA/AMD GPU builds require an explicitly configured external endpoint.
   - If an NPU (Intel AI Boost / AMD XDNA) or Intel iGPU is present:
     Ensure OpenVINO dependencies are installed in the venv only when active hardware/configuration selects OpenVINO for embeddings or reranking. Do not install OpenVINO on NVIDIA-only systems merely because the feature permission is true:
     ```powershell
     & "$HOME\.local-ai-hub\.venv\Scripts\pip.exe" install -r "$HOME\.local-ai-hub\requirements-openvino.txt"
     & "$HOME\.local-ai-hub\.venv\Scripts\python.exe" "$HOME\.local-ai-hub\tools\prefetch_openvino.py"
     ```

---

### PHASE 4: REFRESH INSTRUCTIONS IN CURRENT REPOSITORY

Locate active agent instruction files in this repository (`AGENTS.md`, `CLAUDE.md`, `.cursorrules`, `.windsurfrules`, `GEMINI.md`, etc.):
- Replace or sync the `<!-- BEGIN LOCAL AI HUB TOOL POLICY -->` block with the latest policy from `~/.local-ai-hub/generated/agent-policy.md`.
- Replace or sync the `<!-- BEGIN TOKEN ECONOMY POLICY -->` block with the latest policy from `~/.local-ai-hub/generated/token-economy-policy.md`.
- If no instruction file exists, ensure `AGENTS.md` is created with both policy blocks.
- Verify the default setup deploys the `token-economizer` skill and registers the token-tool CLI directory on the user's persistent PATH. Keep the generated policy trigger that requires agents to load this skill before every repository task.
- Ensure the refreshed policy retains this trigger verbatim:
  - Before any repository task, load and follow the `token-economizer` skill when it is installed; this trigger applies even under deadline pressure.
- Preserve the `trim-run` safety boundary: only its bundled token tools and read-only search CLIs may be launched; use `local_ai_command` for tests/builds and arbitrary validation commands.
- For durable work, require one `local_ai_coord(action="task_create")` contract, phase changes through `local_ai_coord(action="task_checkpoint")`, and one bounded wait instead of status polling loops.
- Keep routing boundaries explicit: repository navigation/symbols/impact=`local_ai_repo`; exact source/log slices=`local_ai_artifact`; test/lint/typecheck/build=`local_ai_command`; task contracts/ownership leases/checkpoints/receipts/receipt-gated completion=`local_ai_coord`; task-wide context=`local_ai_coord(action="context_compile", task_id=...)`; semantic generation/exploration/reasoning/review/second opinion/compression=`local_ai_task(action="delegate"|"explore"|"reason"|"review"|"second_opinion"|"compress", task_id=...)`; deterministic/indexed tools remain for exact facts, symbols, diff and tests; `local_ai_repo(action="solve")` preserves one bounded local pass for explicit semantic wording; failure diagnosis disabled by default, enable `features.local_diagnostic_dispatch=true` only after low-confidence deterministic command parsing with artifact reference plus narrow preview, never raw logs, architecture, security, mutations, or open-ended coding; closed, verified handoff work=`local_ai_work`, never micro-edits or live discussion.
- Semantic handoff is mandatory: after deterministic/indexed evidence, planning, interpretation, synthesis, generation, review, compression, and second-opinion work must call `local_ai_task` after `context_compile` when a task exists. The cloud agent integrates the bounded local result and does not redo semantic work. Treat local output as advisory only; queued work, stale context, missing evidence or weak grounding remains visible with explicit metadata and must not be silently discarded. If local inference is unavailable or intentionally excluded by a permitted boundary, report the bypass through `local_ai_status(adoption_signal="bypassed", target_tool="local_ai_task", target_action="reason")`. Preserve exceptions for architecture, security, mutations, open-ended coding, exact evidence and verification. Shared specialized prompt contracts and model capability limits live in `docs/MODEL_CAPABILITIES.md`.
- Preserve native fallback gate: only after Hub returns `terminal=true` and `retryable=false`. Mutations never cache or single-flight.
- Keep `features.enriched_search`, `features.batch_replacement`, `features.diagnostic_artifacts`, and `features.local_diagnostic_dispatch` enabled in the installed default profile, while preserving independent rollback switches. Compare `/api/adoption` token, latency, first-pass validation, terminal failure, and native fallback metrics. Roll back immediately by setting the affected flag to `false`, restarting Hub, and running `python tools/hubctl.py generate`. Malformed values must remain disabled and disabled features must return structured unavailable before work starts. `local_diagnostic_dispatch=true` may retain only bounded failure-preview context for one local diagnosis; it never retains raw output and does not require `diagnostic_artifacts=true`.
- For `batch_replace`, require `features.batch_replacement=true`, then preview with explicit `dry_run=true`; `staged` is not batch dry-run and is never forwarded. Each edit must exact-match once. Keep rollback behavior and never auto-commit replacements. Set `dry_run=false` only after review.

---

### PHASE 5: RE-TRIGGER REPOSITORY PREPROCESSING

Send an asynchronous preprocess request for the current workspace:
- Via MCP tool:
  `local_ai_repo(action="preprocess", root="<CURRENT_WORKSPACE_ROOT>")`
- Or via HTTP loopback:
  ```bash
  curl -s -X POST http://127.0.0.1:11435/api/repo/preprocess \
    -H "Content-Type: application/json" \
    -d "{\"root\": \"$(pwd)\"}"
  ```

---

### PHASE 6: REPORT STATUS

Confirm:
1. Updated Local AI Hub version.
2. Service status (`running`/`healthy`).
3. Summary of instruction files refreshed.
4. Preprocessing status for current repository.
```
### MCP response economy

After updating, verify the generated schemas under the installation's `generated/` directory expose `max_response_tokens`, `response_profile`, and `reuse_key` on the existing Hub tools. These are installation artifacts, not checked-in source files. Treat generator output and configuration as trusted repository inputs; never copy tool fields from untrusted model output. Keep aggregate response budgeting enabled, preserve artifact-backed exact detail, and verify telemetry reports raw/projected/saved response estimates without prompt or source retention.
