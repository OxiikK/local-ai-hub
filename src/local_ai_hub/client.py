from __future__ import annotations

from .json_utils import dumps as json_dumps

import copy
import http.client
import io
import json
import os
import threading
import uuid
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import quote, urlsplit

from .process_utils import find_listening_pid, pid_alive
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .config import load_config


_DETACHED_CHILDREN_LOCK = threading.Lock()
_DETACHED_CHILDREN: list[subprocess.Popen[Any]] = []


def _retain_detached_child(process: subprocess.Popen[Any]) -> None:
    """Keep intentionally detached hub children alive for later polling/reaping."""
    with _DETACHED_CHILDREN_LOCK:
        _DETACHED_CHILDREN[:] = [child for child in _DETACHED_CHILDREN if child.poll() is None]
        _DETACHED_CHILDREN.append(process)


def _live_start_lock(path: Path, stale_seconds: float) -> bool:
    """Return whether a startup lock still belongs to a live, recent process."""
    try:
        parts = path.read_text(encoding="utf-8").strip().split()
        pid = int(parts[0]) if parts else 0
        age = max(0.0, time.time() - path.stat().st_mtime)
        return age <= max(1.0, stale_seconds) and pid_alive(pid)
    except (OSError, ValueError):
        return False


class HubClient:
    def __init__(self, tenant: str | None = None, config_path: str | None = None, *, auto_start: bool = True):
        self.config = load_config(config_path)
        server = self.config["server"]
        bind_host = str(server.get("bind", "127.0.0.1"))
        client_host = str(server.get("client_host", "")).strip() or ("127.0.0.1" if bind_host in {"0.0.0.0", "::", "[::]"} else bind_host)
        self.base_url = f"http://{client_host}:{int(server.get('port', 11435))}"
        endpoint = urlsplit(self.base_url)
        self._http_host = str(endpoint.hostname or "127.0.0.1")
        self._http_port = int(endpoint.port or 80)
        self.api_token = str(self.config.get("security", {}).get("api_token", ""))
        self.tenant = tenant or os.environ.get("LOCAL_AI_TENANT") or f"agent-{os.getpid()}"
        self.agent = os.environ.get("LOCAL_AI_AGENT_PROFILE") or os.environ.get("LOCAL_AI_AGENT") or "generic"
        self.auto_start = bool(auto_start)
        client_cfg = self.config.get("client", {})
        self.health_timeout = max(0.2, float(client_cfg.get("health_timeout_seconds", 0.75)))
        self.startup_wait = max(1.0, float(client_cfg.get("startup_wait_seconds", 15.0)))
        self.startup_poll = max(0.05, float(client_cfg.get("startup_poll_seconds", 0.15)))
        self.start_lock_stale = max(self.startup_wait, float(client_cfg.get("start_lock_stale_seconds", 20.0)))
        self.max_request_timeout = max(5.0, float(client_cfg.get("max_request_timeout_seconds", 1800.0)))
        self._transport_local = threading.local()
        self._connections_lock = threading.Lock()
        self._connections: dict[int, http.client.HTTPConnection] = {}
        self._flight_lock = threading.RLock()
        self._flights: dict[str, dict[str, Any]] = {}

    def _drop_connection(self) -> None:
        conn = getattr(self._transport_local, "connection", None)
        self._transport_local.connection = None
        with self._connections_lock:
            if conn is not None and self._connections.get(threading.get_ident()) is conn:
                self._connections.pop(threading.get_ident(), None)
        if conn is not None:
            try: conn.close()
            except Exception: pass

    def close(self) -> None:
        current = getattr(self._transport_local, "connection", None)
        self._transport_local.connection = None
        with self._connections_lock:
            connections = list(self._connections.values())
            self._connections.clear()
        if current is not None and current not in connections:
            connections.append(current)
        for conn in connections:
            try: conn.close()
            except Exception: pass

    def __del__(self) -> None:
        try: self.close()
        except Exception: pass

    def _connection(self, timeout: float) -> http.client.HTTPConnection:
        conn = getattr(self._transport_local, "connection", None)
        if conn is None:
            conn = http.client.HTTPConnection(self._http_host, self._http_port, timeout=timeout)
            self._transport_local.connection = conn
            with self._connections_lock:
                previous = self._connections.get(threading.get_ident())
                self._connections[threading.get_ident()] = conn
            if previous is not None and previous is not conn:
                try: previous.close()
                except Exception: pass
        elif getattr(conn, "sock", None) is not None:
            conn.sock.settimeout(timeout)
        return conn

    def _pooled_open(self, method: str, path: str, body: bytes | None, headers: dict[str, str], timeout: float) -> bytes:
        is_reused = False
        conn = getattr(self._transport_local, "connection", None)
        if conn is not None and getattr(conn, "sock", None) is not None:
            is_reused = True
        try:
            conn = self._connection(timeout)
            conn.request(method, path, body=body, headers=headers)
            response = conn.getresponse()
            raw = response.read()
        except (OSError, http.client.HTTPException):
            self._drop_connection()
            if is_reused:
                # Stale pooled keep-alive socket was closed by server idle timeout; retry once on fresh connection.
                try:
                    conn = self._connection(timeout)
                    conn.request(method, path, body=body, headers=headers)
                    response = conn.getresponse()
                    raw = response.read()
                except (OSError, http.client.HTTPException):
                    self._drop_connection()
                    raise
            else:
                raise
        if bool(getattr(response, "will_close", False)):
            self._drop_connection()
        if int(response.status) >= 400:
            response_headers = getattr(response, "headers", {})
            raise HTTPError(f"{self.base_url}{path}", int(response.status), str(response.reason), response_headers, io.BytesIO(raw))
        return raw

    @staticmethod
    def _error_response(exc: HTTPError, request_id: str) -> dict[str, Any]:
        try:
            parsed = json.loads(exc.read().decode("utf-8"))
            if isinstance(parsed, dict):
                parsed.setdefault("success", False)
                parsed.setdefault("status_code", exc.code)
                if exc.code in {408, 425, 429, 500, 502, 503, 504}:
                    parsed.setdefault("retryable", True)
                parsed.setdefault("request_id", request_id)
                return parsed
            return {"success": False, "error": str(parsed), "status_code": exc.code, "request_id": request_id}
        except Exception:
            return {"success": False, "error": f"HTTP {exc.code}: {exc.reason}", "status_code": exc.code, "request_id": request_id}

    def _singleflight(self, key: str, timeout: float, execute: Any) -> dict[str, Any]:
        with self._flight_lock:
            flight = self._flights.get(key)
            if flight is None:
                flight = {"event": threading.Event(), "result": None, "error": None}
                self._flights[key] = flight
                owner = True
            else:
                owner = False
        if owner:
            try:
                flight["result"] = execute()
            except BaseException as exc:
                flight["error"] = exc
            finally:
                with self._flight_lock:
                    self._flights.pop(key, None)
                flight["event"].set()
            if flight["error"] is not None:
                raise flight["error"]
            return flight["result"]
        if not flight["event"].wait(timeout):
            raise TimeoutError("coalesced hub request timed out")
        if flight["error"] is not None:
            raise flight["error"]
        result = copy.deepcopy(flight["result"])
        if isinstance(result, dict): result["coalesced"] = True
        return result

    def _online(self) -> bool:
        try:
            headers = {"X-LocalAI-Token": self.api_token} if self.api_token else {}
            with urlopen(Request(f"{self.base_url}/health", headers=headers), timeout=self.health_timeout) as response:
                return response.status == 200
        except HTTPError as exc:
            # 503 from the bounded HTTP admission gate proves the singleton is alive
            # but busy. Treat it as online so clients do not spawn/restart another hub.
            return exc.code == 503
        except Exception:
            return False

    def ensure_server(self) -> bool:
        if self._online():
            return True
        config_path = self.config["_config_path"]
        state_dir = Path(self.config["server"]["state_dir"])
        state_dir.mkdir(parents=True, exist_ok=True)
        if bool(self.config.get("headless", {}).get("respect_disabled_marker", True)) and (state_dir / "service.disabled").exists():
            return False
        # Once service.py has claimed this installation, only its supervisor may
        # create the hub. supervisor.pid is necessarily absent for a short window
        # during a managed restart; treating that window as an invitation for every
        # MCP client to spawn its own pythonw tree causes duplicate processes and an
        # unpredictable port owner.
        if (state_dir / "service.managed").exists():
            return False
        # Managed mode has one supervisor responsible for startup/recovery. During
        # its cold-start interval MCP clients must not bypass it with direct pythonw
        # spawns, otherwise every client can create a competing hub tree.
        try:
            supervisor_pid = int((state_dir / "supervisor.pid").read_text(encoding="utf-8").strip() or 0)
            if pid_alive(supervisor_pid):
                return False
        except (OSError, ValueError):
            pass
        # A listener can be alive before its health endpoint is ready (cold model/
        # index startup). Starting another hub in that gap creates duplicate pythonw
        # trees and competing supervisors; let the caller retry its request instead.
        port = int(self.config.get("server", {}).get("port", 11435))
        if find_listening_pid(port):
            return False
        lock_path = state_dir / "hub.start.lock"
        owner = False

        # Cross-process startup lock: several CLI agents often connect at once. Only
        # one should spawn the singleton; the others wait for its cheap /health.
        for _ in range(2):
            acquired_lock = False
            try:
                fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                acquired_lock = True
                try:
                    os.write(fd, f"{os.getpid()} {time.time()}".encode("ascii", errors="ignore"))
                finally:
                    os.close(fd)
                owner = True
                break
            except FileExistsError:
                # Dead/crashed startup owners should never consume the full wait budget.
                if not _live_start_lock(lock_path, self.start_lock_stale):
                    try:
                        lock_path.unlink(missing_ok=True)
                    except OSError:
                        pass
                    continue
                deadline = time.monotonic() + self.startup_wait
                while time.monotonic() < deadline:
                    if self._online():
                        return True
                    if not _live_start_lock(lock_path, self.start_lock_stale):
                        try:
                            lock_path.unlink(missing_ok=True)
                        except OSError:
                            pass
                        break
                    time.sleep(self.startup_poll)
                else:
                    return self._online()
                continue
            except OSError:
                if acquired_lock:
                    try:
                        lock_path.unlink(missing_ok=True)
                    except OSError:
                        pass
                continue

        if not owner:
            return self._online()
        try:
            if self._online():
                return True
            env = os.environ.copy()
            env["LOCAL_AI_CONFIG"] = config_path
            src_dir = str(Path(__file__).resolve().parents[1])
            env["PYTHONPATH"] = src_dir + (os.pathsep + env["PYTHONPATH"] if "PYTHONPATH" in env else "")
            py_exe = sys.executable
            if sys.platform == "win32":
                pyw_candidate = Path(sys.executable).parent / "pythonw.exe"
                if pyw_candidate.exists():
                    py_exe = str(pyw_candidate)
            cmd = [py_exe, "-m", "local_ai_hub", "--config", config_path]
            try:
                from .process_utils import hidden_run_kwargs
                kwargs = hidden_run_kwargs(detached=True)
            except Exception:
                kwargs = {}
            if sys.platform != "win32":
                kwargs["start_new_session"] = True
            process = subprocess.Popen(cmd, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, **kwargs)
            _retain_detached_child(process)
            deadline = time.monotonic() + self.startup_wait
            while time.monotonic() < deadline:
                if self._online():
                    return True
                time.sleep(self.startup_poll)
            return self._online()
        finally:
            lock_path.unlink(missing_ok=True)

    @staticmethod
    def _terminal_preflight(path: str, payload: dict[str, Any]) -> dict[str, Any] | None:
        """Reject deterministic client mistakes before opening a loopback socket."""
        if path == "/api/code/symbol" and not str(payload.get("symbol", payload.get("query", ""))).strip():
            return {"success": False, "terminal": True, "retryable": False, "preflight": True, "error": "symbol is required"}
        if path in {"/api/review/diff", "/api/repo/impact"}:
            root = str(payload.get("root", "")).strip()
            if root:
                try:
                    candidate = Path(root).expanduser().resolve(strict=False)
                    if candidate.is_dir() and not (candidate / ".git").exists():
                        return {
                            "success": False, "terminal": True, "retryable": False, "preflight": True,
                            "error": "operation requires a Git repository; use deterministic/code-index review for this root",
                        }
                except OSError:
                    pass
        return None

    def request(self, path: str, payload: dict[str, Any] | None = None, *, timeout: float = 30.0, replay_safe: bool = True) -> dict[str, Any]:
        timeout = min(self.max_request_timeout, max(0.2, float(timeout)))
        clean_payload = payload or {}
        preflight = self._terminal_preflight(path, clean_payload)
        if preflight is not None:
            return preflight
        body = json_dumps(clean_payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True, default=str).encode("utf-8")
        request_id = f"req_{uuid.uuid4().hex[:12]}"

        def once() -> dict[str, Any]:
            headers = {
                "Content-Type": "application/json", "X-LocalAI-Tenant": self.tenant,
                "X-LocalAI-Agent": self.agent, "X-LocalAI-Request-ID": request_id,
            }
            if self.api_token:
                headers["X-LocalAI-Token"] = self.api_token
            return json.loads(self._pooled_open("POST", path, body, headers, timeout).decode("utf-8"))

        def execute() -> dict[str, Any]:
            duplicate_deadline = time.monotonic() + timeout

            def request_until_terminal() -> dict[str, Any]:
                while True:
                    try:
                        return once()
                    except HTTPError as exc:
                        result = self._error_response(exc, request_id)
                        if not (result.get("in_progress") and result.get("retryable")):
                            return result
                        remaining = duplicate_deadline - time.monotonic()
                        if remaining <= 0:
                            result["retry_timeout"] = True
                            return result
                        retry_after = max(0.01, float(result.get("retry_after_seconds", 0.1) or 0.1))
                        time.sleep(min(retry_after, remaining))

            try:
                return request_until_terminal()
            except (URLError, OSError, http.client.HTTPException) as exc:
                self._drop_connection()
                # A local hub can disappear between ensure_server() and the response,
                # or a pooled keep-alive socket may be closed by the server idle timeout.
                # Replay-safe work retries once with a fresh connection for journal reuse.
                if replay_safe:
                    server_ready = self.ensure_server() if self.auto_start else True
                    if server_ready:
                        try: return request_until_terminal()
                        except HTTPError as retry_http: return self._error_response(retry_http, request_id)
                        except Exception as retry_exc: return {"success": False, "error": str(retry_exc), "request_id": request_id, "retried": True}
                return {"success": False, "error": str(exc), "request_id": request_id}
            except Exception as exc:
                return {"success": False, "error": str(exc), "request_id": request_id}

        key = f"POST|{self.tenant}|{self.agent}|{path}|{body.decode('utf-8')}"
        return self._singleflight(key, timeout, execute)

    def get(self, path: str, timeout: float = 15.0) -> dict[str, Any]:
        timeout = min(self.max_request_timeout, max(0.2, float(timeout)))
        request_id = f"req_{uuid.uuid4().hex[:12]}"
        headers = {
            "X-LocalAI-Tenant": self.tenant,
            "X-LocalAI-Agent": self.agent,
            "X-LocalAI-Request-ID": request_id,
        }
        if self.api_token:
            headers["X-LocalAI-Token"] = self.api_token

        def once() -> dict[str, Any]:
            result = json.loads(self._pooled_open("GET", path, None, headers, timeout).decode("utf-8"))
            return result if isinstance(result, dict) else {"success": False, "error": "invalid JSON response"}

        def execute() -> dict[str, Any]:
            try:
                return once()
            except HTTPError as exc:
                return self._error_response(exc, request_id)
            except (URLError, OSError, http.client.HTTPException) as exc:
                self._drop_connection()
                server_ready = self.ensure_server() if self.auto_start else True
                if server_ready:
                    try: return once()
                    except HTTPError as retry_http: return self._error_response(retry_http, request_id)
                    except Exception as retry_exc: return {"success": False, "error": str(retry_exc), "request_id": request_id, "retried": True}
                return {"success": False, "error": str(exc), "request_id": request_id}
            except Exception as exc:
                return {"success": False, "error": str(exc), "request_id": request_id}

        key = f"GET|{self.tenant}|{self.agent}|{path}"
        return self._singleflight(key, timeout, execute)

    def post(self, path: str, payload: dict[str, Any] | None = None, timeout: float = 360.0) -> dict[str, Any]:
        p = path.lower()
        is_mutating = (
            p == "/api/command"
            or p.startswith("/api/leases/")
            or p in {"/api/memory/put", "/api/memory/delete"}
            or p.startswith("/api/maintenance/")
            or (
                p.startswith("/api/agent-state/")
                and not (
                    p == "/api/agent-state/context"
                    or (p == "/api/agent-state/tasks" and isinstance(payload, dict) and payload.get("action") in {"get", "list", "count"})
                    or (p == "/api/agent-state/memory" and isinstance(payload, dict) and payload.get("action") in {"get", "find", "relation_find", "relation_traverse"})
                    or (p == "/api/agent-state/blackboard" and isinstance(payload, dict) and payload.get("action") in {"get", "list"})
                    or (p == "/api/agent-state/incidents" and isinstance(payload, dict) and payload.get("action") in {"find", "decision"})
                    or (p == "/api/agent-state/verification" and isinstance(payload, dict) and payload.get("action") in {"completion"})
                )
            )
        )
        replay_safe = not is_mutating
        return self.request(path, payload, timeout=timeout, replay_safe=replay_safe)

    def status(self, detail: str = "brief", scope: str = "process") -> dict[str, Any]:
        if detail == "agent_state":
            return self.get("/api/status?detail=agent_state")
        return self.get(f"/api/live/status?light=1&scope={scope}&detail={detail}")

    def coord(self, action: str, **kwargs: Any) -> dict[str, Any]:
        act = action.strip().lower().replace("-", "_")
        if act.startswith("task_"):
            task_action = act.replace("task_", "")
            payload = {"action": task_action, **kwargs}
            return self.post("/api/agent-state/tasks", payload)
        if act in ("memo_put", "memo_get", "memo_search", "memo_delete"):
            if act == "memo_put":
                return self.post("/api/memory/put", {
                    "root": kwargs.get("root", "."),
                    "key": kwargs.get("key", ""),
                    "value": kwargs.get("value", ""),
                    "ttl_seconds": kwargs.get("ttl_seconds", 604800),
                })
            if act == "memo_get":
                return self.post("/api/memory/get", {"root": kwargs.get("root", "."), "key": kwargs.get("key", "")})
            if act == "memo_search":
                return self.post("/api/memory/search", {"root": kwargs.get("root", "."), "query": kwargs.get("query", ""), "limit": kwargs.get("limit", 12)})
            if act == "memo_delete":
                return self.post("/api/memory/delete", {"root": kwargs.get("root", "."), "key": kwargs.get("key", "")})
        if act.startswith("memory_"):
            mem_action = act.replace("memory_", "")
            payload = {"action": mem_action, **kwargs}
            return self.post("/api/agent-state/memory", payload)
        if act.startswith("relation_"):
            payload = {"action": act, **kwargs}
            return self.post("/api/agent-state/memory", payload)
        if act == "incident_decision":

            return self.post("/api/agent-state/incidents", {
                "action": "decision",
                "fingerprint": kwargs.get("fingerprint") or {},
                "state_revision": kwargs.get("state_revision", kwargs.get("revision", "")),
            })
        if act == "context_compile":
            return self.post("/api/agent-state/context", {
                "action": "compile",
                "task_id": kwargs.get("task_id") or kwargs.get("query") or kwargs.get("task") or "",
                "token_budget": kwargs.get("token_budget") or kwargs.get("max_tokens") or 4000,
                "changed_paths": kwargs.get("changed_paths") or kwargs.get("paths") or [],
                "root": kwargs.get("root", "."),
            })
        if act == "verify_receipt":
            receipt = kwargs.get("receipt") or {
                "task_id": kwargs.get("task_id", ""),
                "criterion": kwargs.get("criterion", kwargs.get("key", "")),
                "passed": kwargs.get("passed", True),
            }
            if not str(receipt.get("task_id", "")).strip() or not str(receipt.get("criterion", "")).strip():
                return {
                    "success": False,
                    "terminal": True,
                    "retryable": False,
                    "error": "verify_receipt requires task_id and criterion",
                }
            return self.post("/api/agent-state/verification", {
                "action": "receipt",
                "receipt": receipt,
                "root": kwargs.get("root", "."),
            })
        if act == "verify_completion":
            return self.post("/api/agent-state/verification", {
                "action": "completion",
                "task_id": kwargs.get("task_id") or kwargs.get("query") or kwargs.get("task") or "",
                "root": kwargs.get("root", "."),
            })
        if act in {"negative_knowledge_record", "negative_knowledge_find"}:
            sub_act = "record" if act == "negative_knowledge_record" else "find"
            return self.post("/api/agent-state/incidents", {
                "action": sub_act,
                **kwargs,
            })
        if act in ("claim", "claim_batch"):
            return self.post("/api/leases/claim_batch", {
                "root": kwargs.get("root", "."),
                "paths": kwargs.get("paths") or [],
                "ttl_seconds": kwargs.get("ttl_seconds", 900),
                "purpose": kwargs.get("value", "agent edit"),
            })
        if act.startswith("blackboard_"):
            bb_action = act.replace("blackboard_", "")
            board_id = kwargs.get("board_id") or kwargs.get("task_id") or "default"
            payload = {
                "action": bb_action,
                "board_id": board_id,
                "section": kwargs.get("section") or kwargs.get("key"),
                "content": kwargs.get("content") or kwargs.get("value"),
                "author": kwargs.get("author", "agent"),
                "remote_sections": kwargs.get("remote_sections") or kwargs.get("sections") or {},
                "clock": kwargs.get("clock"),
            }
            if kwargs.get("expected_version") is not None:
                payload["expected_version"] = kwargs["expected_version"]
            return self.post("/api/agent-state/blackboard", payload)
        if act == "release":
            return self.post("/api/leases/release", {"lease_id": kwargs.get("lease_id", "")})
        if act == "leases":
            return self.get(f"/api/leases?root={quote(str(kwargs.get('root', '')))}")
        if act in {"worktree_lease", "worktree_claim"}:
            return self.post("/api/coord/worktree_lease", {
                "root": kwargs.get("root", "."),
                "branch": kwargs.get("branch") or kwargs.get("key") or kwargs.get("task_id"),
            })
        if act in {"worktree_release"}:
            return self.post("/api/coord/worktree_release", {
                "root": kwargs.get("root", "."),
                "worktree_path": kwargs.get("worktree_path") or kwargs.get("value") or kwargs.get("lease_id") or "",
                "delete_branch": kwargs.get("delete_branch", True),
                "branch": kwargs.get("branch") or kwargs.get("key") or kwargs.get("task_id"),
            })
        if act == "pubsub_publish":
            return self.post("/api/coord/pubsub_publish", {
                "root": kwargs.get("root", "."),
                "topic": kwargs.get("topic") or kwargs.get("key", "default"),
                "message": kwargs.get("message") or kwargs.get("value", ""),
                "publisher": kwargs.get("publisher") or kwargs.get("approver", "agent"),
            })
        if act == "pubsub_poll":
            return self.post("/api/coord/pubsub_poll", {
                "root": kwargs.get("root", "."),
                "topic": kwargs.get("topic") or kwargs.get("key", "default"),
                "since_timestamp": float(kwargs.get("since_timestamp", 0.0)),
                "limit": int(kwargs.get("limit", 50)),
            })
        if act == "simulate_merge":
            return self.post("/api/coord/simulate_merge", {
                "root": kwargs.get("root", "."),
                "source_branch": kwargs.get("source_branch") or kwargs.get("key") or kwargs.get("branch", ""),
                "target_branch": kwargs.get("target_branch") or kwargs.get("target_scope", "HEAD"),
            })
        return {"success": False, "error": f"unknown coord action '{action}'"}

    def context_compile(self, task_id: str, token_budget: int = 4000, changed_paths: list[str] | None = None, since_hash: str = "", compact: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "action": "compile",
            "task_id": task_id,
            "token_budget": token_budget,
            "changed_paths": changed_paths or [],
            "compact": compact,
        }
        if since_hash:
            payload["since_hash"] = since_hash
        return self.post("/api/agent-state/context", payload)

    def verify_receipt(self, task_id: str, criterion: str, passed: bool = True, command_id: str = "", evidence_id: str = "", details: dict[str, Any] | None = None) -> dict[str, Any]:
        return self.post("/api/agent-state/verification", {
            "action": "receipt",
            "receipt": {
                "task_id": task_id,
                "criterion": criterion,
                "passed": passed,
                "command_id": command_id,
                "evidence_id": evidence_id,
                "details": details or {},
            },
        })

    def verify_completion(self, task_id: str) -> dict[str, Any]:
        return self.post("/api/agent-state/verification", {
            "action": "completion",
            "task_id": task_id,
        })

    def create_task(self, goal: str, acceptance_criteria: list[str] | None = None, task_id: str = "", scope: str = "task", **kwargs: Any) -> dict[str, Any]:
        payload = {
            "action": "create",
            "goal": goal,
            "acceptance_criteria": acceptance_criteria or [],
            "task_id": task_id,
            "scope": scope,
            **kwargs,
        }
        return self.post("/api/agent-state/tasks", payload)

    def get_task(self, task_id: str) -> dict[str, Any]:
        return self.post("/api/agent-state/tasks", {"action": "get", "task_id": task_id})

    def list_tasks(self, status: str | None = None, limit: int = 100) -> dict[str, Any]:
        payload: dict[str, Any] = {"action": "list", "limit": limit}
        if status:
            payload["status"] = status
        return self.post("/api/agent-state/tasks", payload)

    def complete_task(self, task_id: str, reason: str = "completed by agent") -> dict[str, Any]:
        return self.post("/api/agent-state/tasks", {"action": "complete", "task_id": task_id, "reason": reason})

    def fail_task(self, task_id: str, reason: str = "failed by agent") -> dict[str, Any]:
        return self.post("/api/agent-state/tasks", {"action": "fail", "task_id": task_id, "reason": reason})

    def checkpoint_task(self, task_id: str, phase: str = "", next_action: str = "", affected_paths: list[str] | None = None, **kwargs: Any) -> dict[str, Any]:
        payload = {
            "action": "checkpoint",
            "task_id": task_id,
            "phase": phase,
            "next_action": next_action,
            "affected_paths": affected_paths or [],
            **kwargs,
        }
        return self.post("/api/agent-state/tasks", payload)

    def record_memory(self, key: str, value: Any, scope: str = "task", kind: str = "fact", **kwargs: Any) -> dict[str, Any]:
        payload = {
            "action": "record",
            "key": key,
            "value": value,
            "scope": scope,
            "kind": kind,
            **kwargs,
        }
        return self.post("/api/agent-state/memory", payload)

    def find_memory(self, scope: str | None = None, key: str | None = None, query: str | None = None, limit: int = 100) -> dict[str, Any]:
        payload: dict[str, Any] = {"action": "find", "limit": limit}
        if scope:
            payload["scope"] = scope
        if key:
            payload["key"] = key
        if query:
            payload["query"] = query
        return self.post("/api/agent-state/memory", payload)

    def continue_conversation(self, conversation_id: str, prompt: str, model: str | None = None, system: str | None = None, max_tokens: int | None = None, temperature: float | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "conversation_id": conversation_id,
            "prompt": prompt,
        }
        if model:
            payload["model"] = model
        if system:
            payload["system"] = system
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if temperature is not None:
            payload["temperature"] = temperature
        return self.post("/api/conversations/continue", payload)

    def doctor(self) -> dict[str, Any]:
        return self.get("/api/doctor")

    def logs(self, lines: int = 200) -> dict[str, Any]:
        return self.get(f"/api/logs/tail?lines={max(1, min(int(lines), 1000))}")

    def events(self, stream_id: str = "", after_seq: int = 0, limit: int = 100) -> dict[str, Any]:
        import urllib.parse
        params = []
        if stream_id:
            params.append(f"stream_id={urllib.parse.quote(stream_id)}")
        if after_seq > 0:
            params.append(f"after_seq={after_seq}")
        if limit != 100:
            params.append(f"limit={limit}")
        qs = ("?" + "&".join(params)) if params else ""
        return self.get(f"/api/agent-state/events{qs}")

    def stream_events(self, stream_id: str = "", kind: str = "", after_seq: int = 0, timeout: float = 30.0):
        """Yield parsed SSE events (event_type, data_dict) from /api/agent-state/events/stream."""
        import urllib.parse
        params = []
        if stream_id:
            params.append(f"stream_id={urllib.parse.quote(stream_id)}")
        if kind:
            params.append(f"kind={urllib.parse.quote(kind)}")
        if after_seq > 0:
            params.append(f"after_seq={after_seq}")
        if timeout > 0:
            params.append(f"timeout={timeout}")
        qs = ("?" + "&".join(params)) if params else ""
        url = f"{self.base_url}/api/agent-state/events/stream{qs}"
        req = Request(url, headers={"Authorization": f"Bearer {self.api_token}"} if self.api_token else {})
        with urlopen(req, timeout=timeout + 5.0 if timeout > 0 else 60.0) as resp:
            cur_event = ""
            cur_data = []
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
                if not line:
                    if cur_data:
                        data_str = "\n".join(cur_data)
                        try:
                            parsed_data = json.loads(data_str)
                        except Exception:
                            parsed_data = {"raw": data_str}
                        yield cur_event or "message", parsed_data
                    cur_event = ""
                    cur_data = []
                    continue
                if line.startswith(":"):
                    continue
                if line.startswith("event:"):
                    cur_event = line[len("event:"):].strip()
                elif line.startswith("data:"):
                    cur_data.append(line[len("data:"):].strip())

    def circular_dependencies(self, root: str = ".", language: str = "python") -> dict[str, Any]:
        return self.post("/api/repo/circular_dependencies", {"root": root, "language": language})

    def generate_types(self, root: str = ".", file: str = "", write_stub: bool = False) -> dict[str, Any]:
        return self.post("/api/repo/generate_types", {"root": root, "file": file, "write_stub": write_stub})

    def complexity(self, root: str = ".", path: str | None = None, max_results: int = 20) -> dict[str, Any]:
        return self.post("/api/repo/complexity", {"root": root, "path": path, "max_results": max_results})

    def api_spec(self, root: str = ".", framework: str | None = None) -> dict[str, Any]:
        return self.post("/api/repo/api_spec", {"root": root, "framework": framework})

    def dependency_slice(self, root: str = ".", symbol: str = "", path: str | None = None, depth: int = 2) -> dict[str, Any]:
        return self.post("/api/repo/dependency_slice", {"root": root, "symbol": symbol, "path": path, "depth": depth})

    def migration_drift(self, root: str = ".", db_path: str | None = None) -> dict[str, Any]:
        return self.post("/api/repo/migration_drift", {"root": root, "db_path": db_path})

    def package_audit(self, root: str = ".", lockfile_path: str | None = None) -> dict[str, Any]:
        return self.post("/api/repo/package_audit", {"root": root, "lockfile_path": lockfile_path})

    def diff_hunk_stage(self, cwd: str = ".", patch: str = "") -> dict[str, Any]:
        return self.post("/api/command/diff_hunk_stage", {"cwd": cwd, "root": cwd, "patch": patch})

    def flaky_detect(self, cwd: str = ".", command: str = "", runs: int = 5, timeout: int = 30) -> dict[str, Any]:
        return self.post("/api/command/flaky_detect", {"cwd": cwd, "root": cwd, "command": command, "runs": runs, "timeout": timeout})

    def webhook_replay(self, url: str = "", payload: dict[str, Any] | str = "", secret: str = "", signature_header: str = "X-Hub-Signature-256") -> dict[str, Any]:
        return self.post("/api/command/webhook_replay", {"url": url, "payload": payload, "secret": secret, "signature_header": signature_header})

    def eval_drift(self, suite_name: str = "default") -> dict[str, Any]:
        return self.post("/api/task/eval_drift", {"suite_name": suite_name})

    def git_diff(self, root: str = ".", path: str | None = None, staged: bool = False, max_lines: int = 1000) -> dict[str, Any]:
        return self.post("/api/git/diff", {"root": root, "path": path, "staged": staged, "max_lines": max_lines})

    def git_history_search(self, root: str = ".", query: str = "", max_commits: int = 20) -> dict[str, Any]:
        return self.post("/api/git/history_search", {"root": root, "query": query, "max_commits": max_commits})

    def hotspots(self, root: str = ".", days: int = 30, limit: int = 20) -> dict[str, Any]:
        return self.post("/api/repo/hotspots", {"root": root, "days": days, "limit": limit})

    def generate_tests_for_diff(self, root: str = ".", diff: str | None = None, path: str | None = None) -> dict[str, Any]:
        return self.post("/api/repo/generate_tests_for_diff", {"root": root, "diff": diff, "path": path})

    def cross_repo_contract(self, backend_root: str = ".", frontend_root: str = ".") -> dict[str, Any]:
        return self.post("/api/repo/cross_repo_contract", {"backend_root": backend_root, "frontend_root": frontend_root})

    def mock_server(self, action: str = "status", root: str = ".", spec_path: str | None = None, port: int = 11440) -> dict[str, Any]:
        return self.post("/api/command/mock_server", {"action": action, "root": root, "spec_path": spec_path, "port": port})

    def ingest_diagram(self, workspace: str = "default", image_path: str = "", caption: str = "", metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        return self.post("/api/rag/ingest_diagram", {"workspace": workspace, "image_path": image_path, "caption": caption, "metadata": metadata})

    def curate_training_dataset(self, output_path: str = "training_dataset.jsonl", min_receipts: int = 1, format: str = "jsonl") -> dict[str, Any]:
        return self.post("/api/agent-state/tasks", {"action": "curate_dataset", "output_path": output_path, "min_receipts": min_receipts, "format": format})
