"""Persistent per-thread Slack response modes and the agent-facing toggle tool.

The agent interprets operator intent (normally via the Slack mention-only skill)
and calls ``slack_thread_response_mode``.  The transport adapter does not parse
natural-language control phrases; it only enforces the stored mode before a
message is admitted to the agent.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Optional

from gateway.session_context import get_session_env
from hermes_constants import get_hermes_home
from tools.registry import tool_error, tool_result
from utils import atomic_replace

logger = logging.getLogger(__name__)

_STATE_LOCK = threading.RLock()
_STATE_VERSION = 2
_MODE_MENTION_ONLY = "mention_only"
_MODE_NORMAL = "normal"
_VALID_MODES = {_MODE_MENTION_ONLY, _MODE_NORMAL}


def _state_path() -> Path:
    """Return the profile-scoped persistent Slack thread-mode state file."""
    return get_hermes_home() / "state" / "slack_thread_response_modes.json"


def _thread_key(channel_id: str, thread_ts: str) -> str:
    # JSON encoding avoids delimiter ambiguity while remaining deterministic.
    return json.dumps([str(channel_id), str(thread_ts)], ensure_ascii=True, separators=(",", ":"))


def _empty_state() -> dict[str, Any]:
    return {"version": _STATE_VERSION, "threads": {}}


def _load_state_unlocked(path: Path) -> dict[str, Any]:
    if not path.exists():
        return _empty_state()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        logger.warning("[Slack] Could not read thread response-mode state: %s", exc)
        return _empty_state()
    if not isinstance(payload, dict) or not isinstance(payload.get("threads"), dict):
        logger.warning("[Slack] Ignoring invalid thread response-mode state")
        return _empty_state()
    payload["version"] = _STATE_VERSION
    return payload


def _write_state_unlocked(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            tmp_path = Path(handle.name)
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(tmp_path, 0o600)
        atomic_replace(tmp_path, path)
    finally:
        if tmp_path is not None and tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


def get_thread_response_mode(channel_id: str, thread_ts: Optional[str]) -> str:
    """Return ``mention_only`` or ``normal`` for a Slack channel thread."""
    if not channel_id or not thread_ts:
        return _MODE_NORMAL
    key = _thread_key(channel_id, str(thread_ts))
    path = _state_path()
    with _STATE_LOCK:
        payload = _load_state_unlocked(path)
        entry = payload["threads"].get(key)
    if not isinstance(entry, dict):
        return _MODE_NORMAL
    mode = str(entry.get("mode") or _MODE_NORMAL)
    return mode if mode in _VALID_MODES else _MODE_NORMAL


def get_thread_response_state(
    channel_id: str,
    thread_ts: Optional[str],
) -> dict[str, Any]:
    """Return normalized persistent state for one Slack thread."""
    default = {"mode": _MODE_NORMAL, "last_ingested_ts": None}
    if not channel_id or not thread_ts:
        return default
    key = _thread_key(channel_id, str(thread_ts))
    path = _state_path()
    with _STATE_LOCK:
        payload = _load_state_unlocked(path)
        entry = payload["threads"].get(key)
    if not isinstance(entry, dict):
        return default
    mode = str(entry.get("mode") or _MODE_NORMAL)
    if mode not in _VALID_MODES:
        mode = _MODE_NORMAL
    cursor = entry.get("last_ingested_ts")
    return {
        "mode": mode,
        "last_ingested_ts": str(cursor) if cursor else None,
    }


def _slack_ts_decimal(value: str) -> Decimal:
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid Slack message timestamp: {value!r}") from exc


def advance_thread_history_cursor(
    channel_id: str,
    thread_ts: str,
    message_ts: str,
) -> bool:
    """Advance a mention-only thread's ingested-history cursor monotonically.

    Normal-mode or missing entries are left untouched so a disable performed
    during the turn cannot be undone by the processing-complete callback.
    """
    if not channel_id or not thread_ts:
        raise ValueError("Slack channel_id and thread_ts are required")
    candidate = str(message_ts or "").strip()
    candidate_value = _slack_ts_decimal(candidate)

    path = _state_path()
    key = _thread_key(channel_id, thread_ts)
    with _STATE_LOCK:
        payload = _load_state_unlocked(path)
        entry = payload["threads"].get(key)
        if not isinstance(entry, dict) or entry.get("mode") != _MODE_MENTION_ONLY:
            return False
        current = entry.get("last_ingested_ts")
        if current:
            try:
                if _slack_ts_decimal(str(current)) >= candidate_value:
                    return False
            except ValueError:
                logger.warning(
                    "[Slack] Replacing invalid persisted history cursor for %s:%s",
                    channel_id,
                    thread_ts,
                )
        entry["last_ingested_ts"] = candidate
        entry["cursor_updated_at"] = datetime.now(timezone.utc).isoformat()
        _write_state_unlocked(path, payload)
    return True


def set_thread_response_mode(channel_id: str, thread_ts: str, mode: str) -> str:
    """Persist a response mode for one Slack channel thread."""
    normalized = str(mode or "").strip().lower()
    if normalized not in _VALID_MODES:
        raise ValueError(f"Unsupported Slack thread response mode: {mode!r}")
    if not channel_id or not thread_ts:
        raise ValueError("Slack channel_id and thread_ts are required")

    path = _state_path()
    key = _thread_key(channel_id, thread_ts)
    with _STATE_LOCK:
        payload = _load_state_unlocked(path)
        threads = payload["threads"]
        if normalized == _MODE_NORMAL:
            threads.pop(key, None)
        else:
            previous = threads.get(key)
            entry = {
                "channel_id": str(channel_id),
                "thread_ts": str(thread_ts),
                "mode": normalized,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            if isinstance(previous, dict) and previous.get("last_ingested_ts"):
                entry["last_ingested_ts"] = str(previous["last_ingested_ts"])
            threads[key] = entry
        _write_state_unlocked(path, payload)
    return normalized


def _current_slack_thread() -> tuple[str, str] | None:
    platform = get_session_env("HERMES_SESSION_PLATFORM", "").strip().lower()
    channel_id = get_session_env("HERMES_SESSION_CHAT_ID", "").strip()
    thread_ts = get_session_env("HERMES_SESSION_THREAD_ID", "").strip()
    if platform != "slack" or not channel_id or not thread_ts:
        return None
    return channel_id, thread_ts


def _handle_slack_thread_response_mode(args: dict, **_kwargs: Any) -> str:
    current = _current_slack_thread()
    if current is None:
        return tool_error(
            "This tool can only change the current Slack channel thread. "
            "It is unavailable outside a Slack thread session."
        )

    action = str(args.get("action") or "status").strip().lower()
    channel_id, thread_ts = current
    if action == "status":
        mode = get_thread_response_mode(channel_id, thread_ts)
    elif action in {"enable", "mention_only"}:
        mode = set_thread_response_mode(channel_id, thread_ts, _MODE_MENTION_ONLY)
    elif action in {"disable", "normal"}:
        mode = set_thread_response_mode(channel_id, thread_ts, _MODE_NORMAL)
    else:
        return tool_error("action must be one of: enable, disable, status")

    return tool_result(
        {
            "success": True,
            "action": action,
            "mode": mode,
            "message": (
                "This Slack thread now accepts only explicit bot mentions."
                if mode == _MODE_MENTION_ONLY
                else "This Slack thread now uses normal response routing."
            ),
        }
    )


SLACK_THREAD_RESPONSE_MODE_SCHEMA = {
    "name": "slack_thread_response_mode",
    "description": (
        "Set or inspect the response mode for the CURRENT Slack thread. Use this "
        "after interpreting an operator request such as 'only respond when mentioned', "
        "'stay quiet unless tagged', or 'resume normal replies'. The skill/model "
        "interprets the natural-language intent; this tool only persists the toggle. "
        "When enabled, the Slack ingress adapter blocks future messages unless they "
        "contain an explicit Slack <@bot> mention."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["enable", "disable", "status"],
                "description": (
                    "enable = require explicit mentions; disable = restore normal "
                    "routing; status = inspect the current mode"
                ),
            }
        },
        "required": ["action"],
        "additionalProperties": False,
    },
}


__all__ = [
    "SLACK_THREAD_RESPONSE_MODE_SCHEMA",
    "_handle_slack_thread_response_mode",
    "advance_thread_history_cursor",
    "get_thread_response_mode",
    "get_thread_response_state",
    "set_thread_response_mode",
]
