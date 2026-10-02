from __future__ import annotations

from .json_utils import dumps as json_dumps

import json
import sqlite3
import threading
import time
from collections import deque
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .sqlite_support import connect_sqlite, initialize_wal, is_busy_error, retry_busy, quick_sanity_check


@dataclass
class BreakerState:
    failures: int = 0
    opened_at: float = 0.0
    half_open_probe: bool = False
    half_open_probe_since: float = 0.0


class CircuitBreakerRegistry:
    """Small per-resource circuit breaker used around fragile local runtimes.

    It fails fast while a model/runtime is repeatedly unhealthy, then permits one
    half-open probe after cooldown. State is process-local by design: restart is a
    clean recovery boundary and persistent health history remains in telemetry.
    """

    def __init__(self, failure_threshold: int = 3, cooldown_seconds: float = 20.0, probe_timeout_seconds: float = 30.0):
        self.failure_threshold = max(1, int(failure_threshold))
        self.cooldown_seconds = max(0.01, float(cooldown_seconds))
        self.probe_timeout_seconds = max(0.01, float(probe_timeout_seconds))
        self._lock = threading.RLock()
        self._states: dict[str, BreakerState] = {}

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        with self._lock:
            state = self._states.setdefault(key, BreakerState())
            if not state.opened_at:
                return True
            if now - state.opened_at < self.cooldown_seconds:
                return False
            if state.half_open_probe:
                # Auto-expire stale probe after timeout (prevents permanent lockdown)
                if now - state.half_open_probe_since > self.probe_timeout_seconds:
                    state.half_open_probe = False
                else:
                    return False
            state.half_open_probe = True
            state.half_open_probe_since = now
            return True

    def success(self, key: str) -> None:
        with self._lock:
            self._states[key] = BreakerState()

    def failure(self, key: str) -> None:
        with self._lock:
            state = self._states.setdefault(key, BreakerState())
            state.failures += 1
            state.half_open_probe = False
            if state.failures >= self.failure_threshold:
                state.opened_at = time.monotonic()

    def status(self) -> dict[str, Any]:
        now = time.monotonic()
        with self._lock:
            return {
                key: {
                    "failures": state.failures,
                    "open": bool(state.opened_at and now - state.opened_at < self.cooldown_seconds),
                    "cooldown_remaining_seconds": round(max(0.0, self.cooldown_seconds - (now - state.opened_at)), 2) if state.opened_at else 0.0,
                    "half_open_probe": state.half_open_probe,
                }
                for key, state in self._states.items()
            }


class RecoveryJournal:
    """Crash-safe metadata journal for in-progress high-level requests.

    Arbitrary Python callables cannot be replayed safely after a crash, so the
    journal intentionally does not pretend to resume them. Instead it makes
    abandoned requests explicit, lets watchdog/status diagnose them, and records
    clean completion/failure. Cached deterministic stages can be reused on retry.
    """

    def __init__(self, state_dir: Path, stale_seconds: int = 900):
        self.path = state_dir / "recovery.sqlite3"
        self.stale_seconds = max(30, int(stale_seconds))
        self._lock = threading.RLock()
        self._changed = threading.Condition(self._lock)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        self._reconcile_persisted_responses()
        # A newly constructed journal belongs to a fresh hub process. No request can
        # legitimately still be executing at this point, so any prior "running" row
        # is an interrupted request rather than work to wait for.
        self.mark_interrupted_on_startup()

    def _connect(self) -> sqlite3.Connection:
        return connect_sqlite(self.path, timeout_seconds=5.0)

    def _create_schema(self) -> None:
        with closing(self._connect()) as con:
            initialize_wal(con)
            con.execute("""CREATE TABLE IF NOT EXISTS requests (request_id TEXT PRIMARY KEY,tenant TEXT NOT NULL,action TEXT NOT NULL,state TEXT NOT NULL,created_at REAL NOT NULL,updated_at REAL NOT NULL,detail TEXT NOT NULL DEFAULT '',status_code INTEGER NOT NULL DEFAULT 0,response_json TEXT NOT NULL DEFAULT '')""")
            con.execute("CREATE INDEX IF NOT EXISTS idx_recovery_state ON requests(state, updated_at)"); con.commit()

    def _init_db(self) -> None:
        try:
            self._create_schema()
            with closing(self._connect()) as con:
                if not quick_sanity_check(con): raise sqlite3.DatabaseError("recovery sanity check failed")
        except sqlite3.DatabaseError as exc:
            # Lock contention is not corruption. Do not rename a valid journal just because
            # another process is finishing its final transaction during startup.
            if is_busy_error(exc):
                return
            # Request replay is an optimization; a damaged journal must never prevent hub startup.
            try: self.path.replace(self.path.with_suffix(f".corrupt-{int(time.time())}.sqlite3"))
            except OSError: pass
            self._create_schema()

    def _reconcile_persisted_responses(self) -> int:
        """Promote running rows whose terminal response was already committed."""
        with self._lock, closing(self._connect()) as con:
            rows = con.execute(
                "SELECT request_id,status_code,response_json FROM requests WHERE state='running' AND response_json<>''"
            ).fetchall()
            changed = 0
            for request_id, raw_status, raw_response in rows:
                try:
                    response = json.loads(raw_response)
                except Exception:
                    continue
                if response is None:
                    continue
                status_code = int(raw_status or 0)
                payload = response if isinstance(response, dict) else {}
                success = bool(payload.get("terminal") or payload.get("in_progress") or (status_code < 400 and payload.get("success") is not False))
                state = "done" if success else "failed"
                detail = "" if success else str(payload.get("error") or "request failed")[:500]
                cur = con.execute(
                    "UPDATE requests SET state=?,updated_at=?,detail=? WHERE request_id=? AND state='running' AND response_json=?",
                    (state, time.time(), detail, request_id, raw_response),
                )
                changed += int(cur.rowcount or 0)
            if changed:
                con.commit()
                self._changed.notify_all()
            else:
                con.rollback()
            return changed

    def begin(self, request_id: str, tenant: str, action: str) -> None:
        now = time.time()
        with self._lock:
            def write() -> None:
                with closing(self._connect()) as con:
                    con.execute(
                        """INSERT INTO requests(request_id,tenant,action,state,created_at,updated_at,detail,status_code,response_json)
                           VALUES(?,?,?,?,?,?,?,?,?)
                           ON CONFLICT(request_id) DO UPDATE SET
                             tenant=excluded.tenant, action=excluded.action, state='running',
                             created_at=excluded.created_at, updated_at=excluded.updated_at, detail='',
                             status_code=0, response_json=''""",
                        (request_id, tenant, action, "running", now, now, "", 0, ""),
                    )
                    con.commit()
            retry_busy(write, retries=2)

    def finish(self, request_id: str, success: bool, detail: str = "", *, status_code: int = 200, response: Any = None) -> None:
        response_json = ""
        if response is not None:
            try:
                encoded = json_dumps(response, ensure_ascii=False, separators=(",", ":"))
                # Responses should already be compact/artifact-backed. Avoid turning the
                # recovery journal into a second unbounded artifact store.
                if len(encoded.encode("utf-8")) <= 2_000_000:
                    response_json = encoded
            except Exception:
                response_json = ""
        with self._lock:
            def write() -> None:
                with closing(self._connect()) as con:
                    con.execute(
                        "UPDATE requests SET state=?, updated_at=?, detail=?, status_code=?, response_json=? WHERE request_id=?",
                        ("done" if success else "failed", time.time(), detail[:500], int(status_code), response_json, request_id),
                    )
                    con.commit()
            retry_busy(write, retries=2)
            self._changed.notify_all()

    def lookup(self, request_id: str, tenant: str, action: str) -> dict[str, Any] | None:
        with self._lock, closing(self._connect()) as con:
            row = con.execute(
                "SELECT tenant,action,state,status_code,response_json,updated_at FROM requests WHERE request_id=?",
                (request_id,),
            ).fetchone()
        if not row:
            return None
        if row[0] != tenant or row[1] != action:
            return {"state": "conflict", "error": "request id already belongs to a different tenant/action"}
        response = None
        if row[4]:
            try:
                response = json.loads(row[4])
            except Exception:
                response = None
        state = str(row[2])
        status_code = int(row[3] or 0)
        updated_at = float(row[5])
        if state == "running" and response is not None:
            payload = response if isinstance(response, dict) else {}
            success = bool(payload.get("terminal") or payload.get("in_progress") or (status_code < 400 and payload.get("success") is not False))
            repaired_state = "done" if success else "failed"
            detail = "" if success else str(payload.get("error") or "request failed")[:500]
            repaired_at = time.time()
            with self._lock, closing(self._connect()) as con:
                cur = con.execute(
                    "UPDATE requests SET state=?,updated_at=?,detail=? WHERE request_id=? AND state='running' AND response_json=?",
                    (repaired_state, repaired_at, detail, request_id, row[4]),
                )
                if cur.rowcount:
                    con.commit()
                    state = repaired_state
                    updated_at = repaired_at
                    self._changed.notify_all()
                else:
                    con.rollback()
                    current = con.execute("SELECT state,updated_at FROM requests WHERE request_id=?", (request_id,)).fetchone()
                    if current:
                        state = str(current[0])
                        updated_at = float(current[1])
        return {"state": state, "status_code": status_code, "response": response, "updated_at": updated_at}

    def wait_for(self, request_id: str, tenant: str, action: str, *, timeout_seconds: float) -> dict[str, Any] | None:
        """Wait briefly for active duplicate work, never re-executing it."""
        deadline = time.monotonic() + max(0.0, float(timeout_seconds))
        with self._changed:
            while True:
                current = self.lookup(request_id, tenant, action)
                if current is None or current.get("state") != "running":
                    if current is None:
                        return None
                    return {key: current.get(key) for key in ("state", "status_code", "response")}
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return {key: current.get(key) for key in ("state", "status_code", "response")}
                self._changed.wait(min(remaining, 0.05))

    def mark_interrupted_on_startup(self) -> int:
        with self._lock:
            changed = 0
            def write() -> None:
                nonlocal changed
                with closing(self._connect()) as con:
                    cur = con.execute(
                        "UPDATE requests SET state='abandoned', updated_at=?, detail='hub restarted before response completed' WHERE state='running'",
                        (time.time(),),
                    )
                    changed = int(cur.rowcount or 0)
                    con.commit()
            retry_busy(write, retries=2)
            return changed

    def mark_abandoned(self) -> int:
        cutoff = time.time() - self.stale_seconds
        with self._lock:
            changed = 0
            def write() -> None:
                nonlocal changed
                with closing(self._connect()) as con:
                    cur = con.execute(
                        "UPDATE requests SET state='abandoned', updated_at=? WHERE state='running' AND updated_at<?",
                        (time.time(), cutoff),
                    )
                    changed = int(cur.rowcount or 0)
                    con.commit()
            retry_busy(write, retries=2)
            return changed

    def status(self, limit: int = 20) -> dict[str, Any]:
        with closing(self._connect()) as con:
            counts = {row[0]: int(row[1]) for row in con.execute("SELECT state,COUNT(*) FROM requests GROUP BY state")}
            recent = [
                {"request_id": r[0], "tenant": r[1], "action": r[2], "state": r[3], "updated_at": r[4], "detail": r[5]}
                for r in con.execute(
                    "SELECT request_id,tenant,action,state,updated_at,detail FROM requests ORDER BY updated_at DESC LIMIT ?",
                    (max(1, min(int(limit), 100)),),
                )
            ]
        return {"counts": counts, "recent": recent}


class LatencyWindow:
    """Bounded rolling latency/failure window for adaptive decisions."""

    def __init__(self, size: int = 64):
        self._values: deque[tuple[float, bool]] = deque(maxlen=max(8, int(size)))
        self._lock = threading.Lock()

    def add(self, duration_ms: float, success: bool) -> None:
        with self._lock:
            self._values.append((float(duration_ms), bool(success)))

    def stats(self) -> dict[str, Any]:
        with self._lock:
            values = list(self._values)
        if not values:
            return {"count": 0, "avg_ms": 0.0, "failure_rate": 0.0}
        return {
            "count": len(values),
            "avg_ms": round(sum(x[0] for x in values) / len(values), 1),
            "failure_rate": round(sum(1 for x in values if not x[1]) / len(values), 4),
        }
