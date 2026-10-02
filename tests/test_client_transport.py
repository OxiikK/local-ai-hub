from __future__ import annotations

import threading
import time
from urllib.error import HTTPError
from io import BytesIO

from local_ai_hub import client as client_module
from local_ai_hub.client import HubClient


def _client(tmp_path):
    config = tmp_path / "config.toml"
    config.write_text(
        "[server]\nport=39123\nstate_dir='" + (tmp_path / "state").as_posix() + "'\n"
        "[client]\nmax_request_timeout_seconds=5\n",
        encoding="utf-8",
    )
    return HubClient(config_path=str(config), auto_start=False)


def test_client_reuses_one_http11_connection_for_sequential_requests(tmp_path, monkeypatch):
    class Response:
        status = 200
        reason = "OK"
        will_close = False
        headers = {}
        def read(self): return b'{"success":true}'

    class Connection:
        instances = []
        def __init__(self, *_args, **_kwargs): self.__class__.instances.append(self)
        def request(self, *_args, **_kwargs): pass
        def getresponse(self): return Response()
        def close(self): pass

    monkeypatch.setattr(client_module.http.client, "HTTPConnection", Connection)
    client = _client(tmp_path)

    assert client.get("/health")["success"] is True
    assert client.get("/health")["success"] is True
    assert len(Connection.instances) == 1


def test_client_coalesces_simultaneous_identical_requests(tmp_path, monkeypatch):
    class Response:
        status = 200
        reason = "OK"
        will_close = False
        headers = {}
        def read(self): return b'{"success":true}'

    class Connection:
        calls = 0
        lock = threading.Lock()
        def __init__(self, *_args, **_kwargs): pass
        def request(self, *_args, **_kwargs):
            with self.lock: self.__class__.calls += 1
            time.sleep(0.05)
        def getresponse(self): return Response()
        def close(self): pass

    monkeypatch.setattr(client_module.http.client, "HTTPConnection", Connection)
    client = _client(tmp_path)
    start = threading.Barrier(2)
    results = []

    def call():
        start.wait()
        results.append(client.get("/health"))

    threads = [threading.Thread(target=call) for _ in range(2)]
    for thread in threads: thread.start()
    for thread in threads: thread.join(1)

    assert Connection.calls == 1
    assert sorted(bool(item.get("coalesced")) for item in results) == [False, True]


def test_client_reconnects_on_idle_disconnect(tmp_path, monkeypatch):
    class Response:
        status = 200
        reason = "OK"
        will_close = False
        headers = {}
        def read(self): return b'{"success":true}'

    class FailingThenWorkingConnection:
        instances = []
        call_count = 0
        def __init__(self, *_args, **_kwargs):
            self.__class__.instances.append(self)
        def request(self, *_args, **_kwargs): pass
        def getresponse(self):
            self.__class__.call_count += 1
            if self.__class__.call_count == 1:
                raise client_module.http.client.RemoteDisconnected("Remote end closed connection")
            return Response()
        def close(self): pass

    monkeypatch.setattr(client_module.http.client, "HTTPConnection", FailingThenWorkingConnection)
    client = _client(tmp_path)
    res = client.get("/health")
    assert res["success"] is True
    assert len(FailingThenWorkingConnection.instances) == 2


def test_client_replays_duplicate_request_until_owner_finishes(tmp_path, monkeypatch):
    client = _client(tmp_path)
    calls = 0

    def pooled_open(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise HTTPError(
                "http://127.0.0.1/api/search",
                409,
                "Conflict",
                {},
                BytesIO(b'{"success":false,"retryable":true,"in_progress":true,"retry_after_seconds":0.01}'),
            )
        return b'{"success":true,"value":"owner-result"}'

    monkeypatch.setattr(client, "_pooled_open", pooled_open)
    result = client.request("/api/search", {"root": "repo"}, timeout=1)

    assert result == {"success": True, "value": "owner-result"}
    assert calls == 2


def test_client_waits_for_owner_after_replayed_request_returns_in_progress(tmp_path, monkeypatch):
    client = _client(tmp_path)
    calls = 0
    request_ids = []

    def pooled_open(*args, **_kwargs):
        nonlocal calls
        calls += 1
        request_ids.append(args[3]["X-LocalAI-Request-ID"])
        if calls == 1:
            raise client_module.http.client.RemoteDisconnected("connection closed after request")
        if calls == 2:
            raise HTTPError(
                "http://127.0.0.1/api/search",
                409,
                "Conflict",
                {},
                BytesIO(b'{"success":false,"retryable":true,"in_progress":true,"retry_after_seconds":0.01}'),
            )
        return b'{"success":true,"value":"owner-result"}'

    monkeypatch.setattr(client, "_pooled_open", pooled_open)
    result = client.request("/api/search", {"root": "repo"}, timeout=0.5)

    assert result == {"success": True, "value": "owner-result"}
    assert calls == 3
    assert len(set(request_ids)) == 1


def test_client_close_closes_connections_created_by_worker_threads(tmp_path, monkeypatch):
    class Response:
        status = 200
        reason = "OK"
        will_close = False
        headers = {}

        def read(self): return b'{"success":true}'

    class Connection:
        closed = 0

        def __init__(self, *_args, **_kwargs): pass
        def request(self, *_args, **_kwargs): pass
        def getresponse(self): return Response()
        def close(self): self.__class__.closed += 1

    monkeypatch.setattr(client_module.http.client, "HTTPConnection", Connection)
    client = _client(tmp_path)
    worker = threading.Thread(target=lambda: client.get("/health"))
    worker.start()
    worker.join(1)

    client.close()

    assert Connection.closed == 1
