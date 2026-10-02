"""Deterministic, advisory-only quality checks for local model text."""

from __future__ import annotations

import posixpath
import re
from collections.abc import Iterable, Mapping
from typing import Any


MAX_EVIDENCE_PATHS = 64
MAX_PATH_CHARS = 240
MAX_TASK_CHARS = 4096
MAX_OUTPUT_CHARS = 12_000
MAX_REPORTED_PATHS = 8

_QUEUED_STATES = {"queued", "pending", "running", "in_progress"}
_ERROR_MARKER = re.compile(
    r"^\s*(?:error|exception|traceback|fatal|failed|failure|"
    r"model\s+(?:error|unavailable|failed)|"
    r"(?:request|inference)\s+(?:error|failed|timed?\s*out)|"
    r"timed?\s*out|cancel(?:led|ed))\b",
    re.IGNORECASE,
)
_STATUS_MARKER = re.compile(r"^\s*(?:status\s*:\s*)?(queued|pending|running|in_progress)\s*$", re.IGNORECASE)
_PATH_TOKEN = re.compile(
    r"(?<![\w])(?:[A-Za-z]:[\\/])?[A-Za-z0-9_.-]+(?:[\\/][A-Za-z0-9_.-]+)+(?:[:#]\d+(?:-\d+)?)?"
)
_BARE_FILE = re.compile(
    r"(?<![\w./-])[A-Za-z0-9_-]+\.(?:py|js|jsx|ts|tsx|java|go|rs|cs|cpp|c|h|hpp|json|toml|yaml|yml|md|sql|sh|ps1|html|css)\b",
    re.IGNORECASE,
)
_URL = re.compile(r"\b(?:https?|file)://\S+", re.IGNORECASE)
_LINE_SUFFIX = re.compile(r"(?:[:#]\d+(?:-\d+)?)$")
_DETACHED_LINE_REFERENCE = re.compile(r"\b(?:at\s+)?line\s+\d+(?:\s*[-–]\s*\d+)?\b", re.IGNORECASE)
_QUALIFIED_LINE_REFERENCE = re.compile(
    r"(?<![\w./-])[A-Za-z0-9_.-]+(?:[\\/][A-Za-z0-9_.-]+)+\.(?:py|js|jsx|ts|tsx|java|go|rs|cs|cpp|c|h|hpp|json|toml|yaml|yml|md|sql|sh|ps1|html|css)[:#]\d+(?:-\d+)?\b",
    re.IGNORECASE,
)
_GENERIC_OUTPUT_MARKER = re.compile(
    r"\b(?:hardware or software|best practices|potential mismatch|it is important to note|"
    r"could be due to a variety of|in the context of|hope this helps)\b",
    re.IGNORECASE,
)


def assess_semantic_result(
    task: str,
    evidence_paths: Iterable[str],
    output: Any,
    *,
    context: str = "",
) -> dict[str, Any]:
    """Return a bounded quality decision without claiming semantic truth.

    The model result is never authoritative.  A usable result only means that
    this inexpensive shape-and-overlap gate found no obvious reason to bypass
    it; deterministic repository evidence still wins.
    """

    if not isinstance(task, str) or not task.strip():
        return _reject("malformed_task")
    if len(task) > MAX_TASK_CHARS:
        return _reject("task_too_large")

    paths, path_error = _bounded_paths(evidence_paths)
    if path_error:
        return _reject(path_error)
    if not paths:
        paths = _extract_paths(task + "\n" + context[:MAX_OUTPUT_CHARS])[:MAX_EVIDENCE_PATHS]

    text, output_error = _model_text(output)
    if output_error:
        return _reject(output_error)
    assert text is not None

    if len(text) > MAX_OUTPUT_CHARS:
        return _reject("output_too_large")
    stripped = text.strip()
    if not stripped:
        return _reject("empty_output")
    if "\x00" in text:
        return _reject("malformed_output")

    # A bare "at line N" claim is not evidence.  Exact source locations are
    # accepted only when they are attached to a path (for example ``auth.py:42``)
    # or supplied by the deterministic evidence layer.  This blocks the common
    # local-model failure mode of inventing line numbers while retaining useful
    # path-qualified references.
    # Models often write both a human-readable ``path line N`` label and a
    # machine-usable ``path:N`` reference in the same finding.  Reject only
    # outputs that contain no qualified path:line reference at all; otherwise
    # the path-overlap check below remains authoritative.
    if _DETACHED_LINE_REFERENCE.search(text) and not _QUALIFIED_LINE_REFERENCE.search(text):
        return _reject("unsupported_location")

    status = _STATUS_MARKER.fullmatch(stripped)
    if status:
        return _reject("queued" if status.group(1).casefold() in _QUEUED_STATES else "malformed_output")
    if _ERROR_MARKER.search(stripped):
        return _reject("model_error")
    if _GENERIC_OUTPUT_MARKER.search(stripped):
        return _reject("generic_output")

    unrelated = [
        path
        for path in _extract_paths(stripped)
        if not _matches_evidence(path, paths)
    ]
    if unrelated:
        return _reject("unrelated_output", paths=unrelated[:MAX_REPORTED_PATHS])

    return _accept()


def _bounded_paths(value: Iterable[str]) -> tuple[tuple[str, ...] | None, str | None]:
    if value is None or isinstance(value, (str, bytes, bytearray)):
        return None, "malformed_paths"
    try:
        iterator = iter(value)
    except TypeError:
        return None, "malformed_paths"

    paths: list[str] = []
    for index in range(MAX_EVIDENCE_PATHS + 1):
        try:
            path = next(iterator)
        except StopIteration:
            break
        except Exception:
            return None, "malformed_paths"
        if index >= MAX_EVIDENCE_PATHS:
            return None, "paths_too_many"
        if not isinstance(path, str) or not path.strip():
            return None, "malformed_paths"
        if len(path) > MAX_PATH_CHARS:
            return None, "path_too_long"
        normalized = _normalize_path(path)
        if normalized is None:
            return None, "unsafe_path"
        if normalized not in paths:
            paths.append(normalized)
    return tuple(paths), None


def _model_text(value: Any) -> tuple[str | None, str | None]:
    if value is None:
        return "", None
    if isinstance(value, Mapping):
        status = value.get("status")
        if isinstance(status, str) and status.strip().casefold() in _QUEUED_STATES:
            return None, "queued"
        if "error" in value or (isinstance(status, str) and status.strip().casefold() in {"error", "failed", "failure"}):
            return None, "model_error"
        for key in ("output", "text"):
            candidate = value.get(key)
            if isinstance(candidate, str):
                return candidate, None
        return None, "malformed_output"
    if not isinstance(value, str):
        return None, "malformed_output"
    return value, None


def _normalize_path(value: str) -> str | None:
    path = _LINE_SUFFIX.sub("", value.strip().strip("`'\"()[]{}<>,.;"))
    path = path.replace("\\", "/")
    if not path or path.startswith("/") or re.match(r"^[A-Za-z]:/", path) or path.startswith("//"):
        return None
    normalized = posixpath.normpath(path)
    if normalized in {"", "."} or normalized == ".." or normalized.startswith("../"):
        return None
    return normalized.casefold()


def _extract_paths(text: str) -> tuple[str, ...]:
    without_urls = _URL.sub(" ", text)
    candidates = list(_PATH_TOKEN.findall(without_urls))
    candidates.extend(_BARE_FILE.findall(without_urls))
    normalized: list[str] = []
    for candidate in candidates:
        value = _normalize_path(candidate)
        if value is not None and value not in normalized:
            normalized.append(value)
    return tuple(normalized)


def _matches_evidence(candidate: str, evidence_paths: tuple[str, ...] | None) -> bool:
    if not evidence_paths:
        return False
    candidate_name = posixpath.basename(candidate)
    for evidence in evidence_paths:
        if candidate == evidence or candidate.endswith("/" + evidence):
            return True
        if posixpath.basename(evidence) == candidate_name:
            return True
    return False


def _reject(reason: str, **extra: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        "usable": False,
        "reason": reason,
        "advisory_only": True,
        "bypass_reason": reason,
    }
    result.update(extra)
    return result


def _accept() -> dict[str, Any]:
    return {
        "usable": True,
        "reason": None,
        "advisory_only": True,
        "bypass_reason": None,
    }
