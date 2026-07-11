from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from string import Formatter
from typing import Any, Mapping


PENDING = "PENDING"
DECIDED = "DECIDED"
OUTBOX_READY = "READY"
CHOICE_FALSE_LABEL = "AI 표시 안 함 (false)"
CHOICE_TRUE_LABEL = "AI 표시 승인 (true)"
DEFAULT_BUSY_TIMEOUT_MS = 5000
REQUIRED_COMMANDS = frozenset(
    {
        "approval_click",
        "post_binding",
        "ui_state",
        "worker_wakeup",
        "expire",
    }
)

APPROVAL_REQUEST_COLUMNS = {
    "approval_id",
    "item_id",
    "item_revision",
    "artifact_sha256",
    "metadata_sha256",
    "config_sha256",
    "status",
    "decision",
    "expires_at",
    "discord_guild_id",
    "channel_id",
    "message_id",
    "custom_id_false",
    "custom_id_true",
    "decided_by_user_id",
    "interaction_id",
    "decided_at",
    "ui_state",
    "ui_error",
    "request_title",
    "youtube_video_id",
    "row_hash",
}
QUEUE_ITEM_COLUMNS = {
    "item_id",
    "item_revision",
    "video_sha256",
    "metadata_sha256",
    "config_sha256",
}
OUTBOX_COLUMNS = {
    "job_id",
    "approval_id",
    "item_id",
    "item_revision",
    "action",
    "payload",
    "state",
    "attempts",
    "not_before",
    "created_at",
}


class ApprovalBridgeError(ValueError):
    """Raised when an approval bridge command or read check fails."""


@dataclass(frozen=True)
class QueueCommand:
    argv: list[str]


@dataclass(frozen=True)
class FileApprovalSettings:
    enabled: bool
    db_path: Path | None
    poll_interval_seconds: float
    approver_user_ids: set[str]
    commands: dict[str, QueueCommand]
    busy_timeout_ms: int

    @classmethod
    def from_config(cls, raw: Mapping[str, Any] | None) -> "FileApprovalSettings":
        cfg = raw if isinstance(raw, Mapping) else {}
        db_raw = str(cfg.get("db_path") or cfg.get("sqlite_path") or "").strip()
        approver_cfg = cfg.get("approver_user_ids")
        if approver_cfg is None:
            approver_cfg = os.getenv("GOLLASSUL_APPROVER_USER_IDS", "")
        return cls(
            enabled=_coerce_bool(cfg.get("enabled"), default=False),
            db_path=Path(db_raw).expanduser() if db_raw else None,
            poll_interval_seconds=max(0.05, _coerce_float(cfg.get("poll_interval_seconds"), 2.0)),
            approver_user_ids=_numeric_id_set(approver_cfg),
            commands=_coerce_commands(cfg.get("commands") or cfg),
            busy_timeout_ms=max(1, _coerce_int(cfg.get("busy_timeout_ms"), DEFAULT_BUSY_TIMEOUT_MS)),
        )

    def ready(self) -> bool:
        return bool(
            self.enabled
            and self.db_path
            and self.approver_user_ids
            and REQUIRED_COMMANDS.issubset(self.commands)
        )


@dataclass(frozen=True)
class PendingApproval:
    approval_id: str
    item_id: str
    item_revision: int
    artifact_sha256: str
    metadata_sha256: str
    config_sha256: str
    title: str
    video_id: str
    guild_id: str | None
    channel_id: str
    message_id: str | None
    custom_id_false: str
    custom_id_true: str
    expires_at: float
    decision: str | None = None


@dataclass(frozen=True)
class FailedUiRepair:
    approval: PendingApproval
    decision: bool


@dataclass(frozen=True)
class DecisionResult:
    approval_id: str
    item_id: str
    decision: bool
    approval_outbox_job_id: str | None
    idempotent: bool = False


def ensure_compatible(settings: FileApprovalSettings) -> tuple[bool, list[str]]:
    if settings.db_path is None:
        return False, ["missing db_path"]
    if not settings.db_path.exists():
        return False, [f"missing sqlite db: {settings.db_path}"]
    with connect(settings) as conn:
        missing: list[str] = []
        for table, columns in (
            ("approval_requests", APPROVAL_REQUEST_COLUMNS),
            ("queue_items", QUEUE_ITEM_COLUMNS),
            ("approval_outbox", OUTBOX_COLUMNS),
        ):
            existing = _table_columns(conn, table)
            if not existing:
                missing.append(f"missing table {table}")
                continue
            for column in sorted(columns - existing):
                missing.append(f"missing column {table}.{column}")
        return not missing, missing


def connect(settings: FileApprovalSettings) -> sqlite3.Connection:
    if settings.db_path is None:
        raise ApprovalBridgeError("missing db_path")
    conn = sqlite3.connect(str(settings.db_path), timeout=settings.busy_timeout_ms / 1000)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute(f"PRAGMA busy_timeout={int(settings.busy_timeout_ms)}")
    return conn


def pending_approvals(settings: FileApprovalSettings, *, now: float | None = None) -> list[PendingApproval]:
    ok, missing = ensure_compatible(settings)
    if not ok:
        raise ApprovalBridgeError("; ".join(missing))
    now_ts = time.time() if now is None else now
    with connect(settings) as conn:
        rows = conn.execute(
            """
            SELECT * FROM approval_requests
            WHERE status = ? AND expires_at > ?
            ORDER BY expires_at, approval_id
            """,
            (PENDING, now_ts),
        ).fetchall()
    return [_row_to_approval(row) for row in rows]


def failed_ui_repairs(settings: FileApprovalSettings) -> list[FailedUiRepair]:
    ok, missing = ensure_compatible(settings)
    if not ok:
        raise ApprovalBridgeError("; ".join(missing))
    with connect(settings) as conn:
        rows = conn.execute(
            """
            SELECT * FROM approval_requests
            WHERE status = ?
              AND ui_state = 'ui_update_failed'
              AND message_id IS NOT NULL
            ORDER BY decided_at, approval_id
            """,
            (DECIDED,),
        ).fetchall()
    return [
        FailedUiRepair(
            approval=_row_to_approval(row),
            decision=str(row["decision"]).lower() == "true",
        )
        for row in rows
    ]


def get_approval(settings: FileApprovalSettings, approval_id: str) -> PendingApproval:
    with connect(settings) as conn:
        row = conn.execute(
            "SELECT * FROM approval_requests WHERE approval_id = ?",
            (approval_id,),
        ).fetchone()
    if row is None:
        raise ApprovalBridgeError("approval request not found")
    return _row_to_approval(row)


def validate_click_binding(
    approval: PendingApproval,
    *,
    custom_id: str,
    decision: bool,
    guild_id: str | None,
    channel_id: str,
    message_id: str,
    now: float | None = None,
) -> None:
    now_ts = time.time() if now is None else now
    if approval.expires_at <= now_ts:
        raise ApprovalBridgeError("approval request expired")
    expected_custom_id = approval.custom_id_true if decision else approval.custom_id_false
    if custom_id != expected_custom_id:
        raise ApprovalBridgeError("custom_id binding mismatch")
    if approval.guild_id and str(guild_id or "") != approval.guild_id:
        raise ApprovalBridgeError("guild binding mismatch")
    if str(channel_id or "") != approval.channel_id:
        raise ApprovalBridgeError("channel binding mismatch")
    if str(message_id or "") != str(approval.message_id or ""):
        raise ApprovalBridgeError("message binding mismatch")


def run_approval_click(
    settings: FileApprovalSettings,
    approval: PendingApproval,
    *,
    decision: bool,
    user_id: str,
    interaction_id: str,
    custom_id: str,
    guild_id: str | None,
    channel_id: str,
    message_id: str,
) -> DecisionResult:
    payload = _run_configured_command(
        settings,
        "approval_click",
        {
            "approval_id": approval.approval_id,
            "request_id": approval.approval_id,
            "item_id": approval.item_id,
            "item_revision": str(approval.item_revision),
            "decision": "true" if decision else "false",
            "choice": "true" if decision else "false",
            "user_id": user_id,
            "interaction_id": interaction_id,
            "custom_id": custom_id,
            "guild_id": guild_id or "",
            "channel_id": channel_id,
            "message_id": message_id,
            "db_path": str(settings.db_path or ""),
            "state_dir": str((settings.db_path or Path("")).parent),
        },
    )
    if payload.get("ok") is False:
        raise ApprovalBridgeError(str(payload.get("error") or "approval click rejected"))
    return DecisionResult(
        approval_id=str(payload.get("approval_id") or payload.get("request_id") or approval.approval_id),
        item_id=str(payload.get("item_id") or approval.item_id),
        decision=str(payload.get("decision") or ("true" if decision else "false")).lower() == "true",
        approval_outbox_job_id=(
            str(payload.get("approval_outbox_job_id"))
            if payload.get("approval_outbox_job_id")
            else None
        ),
        idempotent=bool(payload.get("idempotent")),
    )


def run_post_binding(settings: FileApprovalSettings, approval: PendingApproval, *, guild_id: str | None, channel_id: str, message_id: str) -> None:
    _run_optional_command(
        settings,
        "post_binding",
        {
            "approval_id": approval.approval_id,
            "request_id": approval.approval_id,
            "item_id": approval.item_id,
            "guild_id": guild_id or "",
            "channel_id": channel_id,
            "message_id": message_id,
            "db_path": str(settings.db_path or ""),
            "state_dir": str((settings.db_path or Path("")).parent),
        },
    )


def run_ui_state(settings: FileApprovalSettings, approval: PendingApproval, *, state: str, error: str | None = None) -> None:
    _run_optional_command(
        settings,
        "ui_state",
        {
            "approval_id": approval.approval_id,
            "request_id": approval.approval_id,
            "item_id": approval.item_id,
            "ui_state": state,
            "error": error or "",
            "db_path": str(settings.db_path or ""),
            "state_dir": str((settings.db_path or Path("")).parent),
        },
    )


def run_worker_wakeup(settings: FileApprovalSettings, result: DecisionResult) -> None:
    _run_optional_command(
        settings,
        "worker_wakeup",
        {
            "approval_id": result.approval_id,
            "request_id": result.approval_id,
            "item_id": result.item_id,
            "decision": str(result.decision).lower(),
            "approval_outbox_job_id": result.approval_outbox_job_id or "",
            "outbox_job_id": result.approval_outbox_job_id or "",
            "db_path": str(settings.db_path or ""),
            "state_dir": str((settings.db_path or Path("")).parent),
        },
    )


def run_expire(settings: FileApprovalSettings) -> None:
    _run_optional_command(
        settings,
        "expire",
        {
            "db_path": str(settings.db_path or ""),
            "state_dir": str((settings.db_path or Path("")).parent),
        },
    )


def render_message(row: PendingApproval) -> str:
    parts = [
        "**Synthetic media approval**",
        f"Request ID: `{row.approval_id}`",
        f"Item ID: `{row.item_id}`",
        f"Title: {row.title}",
    ]
    if row.video_id:
        parts.append(f"Video ID: `{row.video_id}`")
    return "\n".join(parts)


def _run_optional_command(settings: FileApprovalSettings, command: str, mapping: Mapping[str, str]) -> dict[str, Any] | None:
    if command not in settings.commands:
        return None
    return _run_configured_command(settings, command, mapping)


def _run_configured_command(settings: FileApprovalSettings, command: str, mapping: Mapping[str, str]) -> dict[str, Any]:
    spec = settings.commands.get(command)
    if spec is None:
        raise ApprovalBridgeError(f"missing configured Queue command: {command}")
    argv = [_format_arg(arg, mapping) for arg in spec.argv]
    if not argv or not argv[0]:
        raise ApprovalBridgeError(f"invalid Queue command argv: {command}")
    completed = subprocess.run(
        argv,
        shell=False,
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )
    stdout = completed.stdout.strip()
    payload: dict[str, Any] = {}
    if stdout:
        try:
            payload = json.loads(stdout)
        except json.JSONDecodeError:
            payload = {"stdout": stdout}
    if completed.returncode != 0:
        message = completed.stderr.strip() or stdout or f"{command} failed"
        raise ApprovalBridgeError(message)
    return payload


def _row_to_approval(row: sqlite3.Row) -> PendingApproval:
    false_id = str(row["custom_id_false"] or "")
    true_id = str(row["custom_id_true"] or "")
    if not false_id or not true_id:
        raise ApprovalBridgeError("approval request is missing Queue-owned custom ids")
    return PendingApproval(
        approval_id=str(row["approval_id"]),
        item_id=str(row["item_id"]),
        item_revision=int(row["item_revision"]),
        artifact_sha256=str(row["artifact_sha256"] or ""),
        metadata_sha256=str(row["metadata_sha256"] or ""),
        config_sha256=str(row["config_sha256"] or ""),
        title=str(row["request_title"] or row["item_id"]),
        video_id=str(row["youtube_video_id"] or ""),
        guild_id=str(row["discord_guild_id"]) if row["discord_guild_id"] else None,
        channel_id=str(row["channel_id"] or ""),
        message_id=str(row["message_id"]) if row["message_id"] else None,
        custom_id_false=false_id,
        custom_id_true=true_id,
        expires_at=float(row["expires_at"]),
        decision=str(row["decision"]) if row["decision"] else None,
    )


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return {str(row["name"]) for row in rows}


def _numeric_id_set(value: Any) -> set[str]:
    return {item for item in (str(v).strip() for v in _as_list(value)) if item.isdigit()}


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",")]
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def _coerce_bool(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "on"}:
            return True
        if lowered in {"false", "0", "no", "off"}:
            return False
        return default
    return bool(value)


def _coerce_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_commands(value: Any) -> dict[str, QueueCommand]:
    if not isinstance(value, Mapping):
        return {}
    if isinstance(value.get("queue_commands"), Mapping):
        raw = value["queue_commands"]
    elif isinstance(value.get("commands"), Mapping):
        raw = value["commands"]
    else:
        raw = value
    out: dict[str, QueueCommand] = {}
    for command, spec in raw.items():
        key = str(command).strip()
        if key not in REQUIRED_COMMANDS:
            return {}
        if isinstance(spec, QueueCommand):
            argv = spec.argv
        else:
            argv = spec.get("argv") if isinstance(spec, Mapping) else spec
        if key and isinstance(argv, (list, tuple)) and argv and argv[0] and all(isinstance(arg, str) for arg in argv):
            out[key] = QueueCommand(argv=list(argv))
        else:
            return {}
    return out


def _format_arg(template: str, mapping: Mapping[str, str]) -> str:
    allowed = set(mapping)
    for _, field_name, _, _ in Formatter().parse(template):
        if field_name and field_name not in allowed:
            raise ApprovalBridgeError("Queue command argv contains unsupported placeholder")
    return template.format(**mapping)
