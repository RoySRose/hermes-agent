import asyncio
import json
import sqlite3
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import discord
import pytest

from gateway.config import PlatformConfig
from plugins.platforms.discord import file_approval as file_approval_module
from plugins.platforms.discord.adapter import DiscordAdapter, FileApprovalView
from plugins.platforms.discord.file_approval import (
    CHOICE_FALSE_LABEL,
    CHOICE_TRUE_LABEL,
    DECIDED,
    OUTBOX_READY,
    PENDING,
    FileApprovalSettings,
    get_approval,
    pending_approvals,
    run_approval_click,
    run_post_binding,
)


QUEUE_PATH = Path("/home/sungw/ecosystem/internal/maki/tools/gollassul-youtube-upload/gollassul_queue.py")


def _run_queue(state_dir: Path, *args: str) -> dict:
    completed = subprocess.run(
        [sys.executable, str(QUEUE_PATH), "--state-dir", str(state_dir), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return json.loads(completed.stdout)


def _settings(tmp_path: Path, state_dir: Path, *, approvers=("42",)) -> FileApprovalSettings:
    db_path = state_dir / "queue.db"
    return FileApprovalSettings.from_config(
        {
            "enabled": True,
            "db_path": str(db_path),
            "poll_interval_seconds": 0.01,
            "approver_user_ids": list(approvers),
            "commands": {
                "approval_click": {
                    "argv": [
                        sys.executable,
                        str(QUEUE_PATH),
                        "--state-dir",
                        "{state_dir}",
                        "decide-synthetic",
                        "--request-id",
                        "{approval_id}",
                        "--choice",
                        "{choice}",
                        "--user-id",
                        "{user_id}",
                        "--interaction-id",
                        "{interaction_id}",
                        "--custom-id",
                        "{custom_id}",
                        "--guild-id",
                        "{guild_id}",
                        "--channel-id",
                        "{channel_id}",
                        "--message-id",
                        "{message_id}",
                    ]
                },
                "post_binding": {
                    "argv": [
                        sys.executable,
                        str(QUEUE_PATH),
                        "--state-dir",
                        "{state_dir}",
                        "mark-posted",
                        "--approval-id",
                        "{approval_id}",
                        "--guild-id",
                        "{guild_id}",
                        "--channel-id",
                        "{channel_id}",
                        "--message-id",
                        "{message_id}",
                    ]
                },
                "ui_state": {
                    "argv": [
                        sys.executable,
                        str(QUEUE_PATH),
                        "--state-dir",
                        "{state_dir}",
                        "ui-state",
                        "--approval-id",
                        "{approval_id}",
                        "--ui-state",
                        "{ui_state}",
                        "--ui-error",
                        "{error}",
                    ]
                },
                "worker_wakeup": {
                    "argv": [
                        sys.executable,
                        str(QUEUE_PATH),
                        "--state-dir",
                        "{state_dir}",
                        "process-approval-outbox",
                        "--job-id",
                        "{approval_outbox_job_id}",
                    ]
                },
                "expire": {
                    "argv": [
                        sys.executable,
                        str(QUEUE_PATH),
                        "--state-dir",
                        "{state_dir}",
                        "sweep-expired",
                    ]
                },
            },
        }
    )


def _queue_state(tmp_path: Path, *, user_id="42", channel_id="123") -> tuple[Path, str]:
    state_dir = tmp_path / "queue-state"
    out = _run_queue(
        state_dir,
        "enqueue",
        "--title",
        "Synthetic video title",
        "--video-id",
        "vid-1",
        "--synthetic-media",
        "--allowed-user-id",
        user_id,
        "--channel-target",
        channel_id,
        "--video-sha256",
        "video-sha",
        "--config-sha256",
        "config-sha",
        "--metadata-sha256",
        "metadata-sha",
    )
    approval_id = Path(out["approval_request"]).stem
    return state_dir, approval_id


def _row(state_dir: Path, query: str, args=()):
    con = sqlite3.connect(state_dir / "queue.db")
    con.row_factory = sqlite3.Row
    row = con.execute(query, args).fetchone()
    con.close()
    return dict(row) if row else None


def _interaction(custom_id, *, user_id="42", message_id="987", channel_id="123", guild_id="555", interaction_id="int-1"):
    message = SimpleNamespace(id=message_id, channel=SimpleNamespace(id=channel_id), embeds=[])
    return SimpleNamespace(
        id=interaction_id,
        guild_id=guild_id,
        channel_id=channel_id,
        data={"custom_id": custom_id},
        user=SimpleNamespace(id=user_id, display_name=f"user-{user_id}", roles=[]),
        message=message,
        response=SimpleNamespace(edit_message=AsyncMock(), send_message=AsyncMock()),
    )


def _view(settings: FileApprovalSettings, approval_id: str) -> FileApprovalView:
    return FileApprovalView(get_approval(settings, approval_id), settings)


def _bind_posted(settings: FileApprovalSettings, approval_id: str, *, guild_id="555", channel_id="123", message_id="987") -> None:
    run_post_binding(
        settings,
        get_approval(settings, approval_id),
        guild_id=guild_id,
        channel_id=channel_id,
        message_id=message_id,
    )


def _process_approval_calls(calls):
    return [
        call
        for call in calls
        if "process-approval-outbox" in call and "--job-id" in call
    ]


def test_exact_two_button_labels_and_persistent_timeout(tmp_path):
    state_dir, approval_id = _queue_state(tmp_path)
    settings = _settings(tmp_path, state_dir)
    approval = pending_approvals(settings)[0]
    view = FileApprovalView(approval, settings)

    assert approval.approval_id == approval_id
    assert view.timeout is None
    assert [child.label for child in view.children] == [CHOICE_FALSE_LABEL, CHOICE_TRUE_LABEL]
    assert view.children[0].style == discord.ButtonStyle.success
    assert view.children[1].style == discord.ButtonStyle.danger
    assert len(view.children) == 2
    assert all(child.custom_id.startswith("gollassul.synthetic.") for child in view.children)


def test_ready_requires_complete_queue_command_allowlist(tmp_path):
    state_dir, _approval_id = _queue_state(tmp_path)
    valid = _settings(tmp_path, state_dir)
    assert valid.ready() is True
    for command in ("post_binding", "ui_state", "expire", "worker_wakeup"):
        assert str(QUEUE_PATH) in valid.commands[command].argv

    base = {
        "enabled": True,
        "db_path": str(state_dir / "queue.db"),
        "approver_user_ids": ["42"],
        "commands": {name: {"argv": cmd.argv} for name, cmd in valid.commands.items()},
    }
    missing_expire = dict(base)
    missing_expire["commands"] = dict(base["commands"])
    missing_expire["commands"].pop("expire")
    assert FileApprovalSettings.from_config(missing_expire).ready() is False

    malformed = dict(base)
    malformed["commands"] = dict(base["commands"])
    malformed["commands"]["expire"] = {"argv": [""]}
    assert FileApprovalSettings.from_config(malformed).ready() is False

    disallowed = dict(base)
    disallowed["commands"] = dict(base["commands"])
    disallowed["commands"]["youtube_reconcile"] = {"argv": [sys.executable, str(QUEUE_PATH), "list"]}
    assert FileApprovalSettings.from_config(disallowed).ready() is False


def test_queue_click_creates_ready_outbox_before_worker_wakeup(tmp_path):
    state_dir, approval_id = _queue_state(tmp_path)
    settings = _settings(tmp_path, state_dir)
    _bind_posted(settings, approval_id)
    approval = get_approval(settings, approval_id)

    result = run_approval_click(
        settings,
        approval,
        decision=True,
        user_id="42",
        interaction_id="int-ready",
        custom_id=approval.custom_id_true,
        guild_id="555",
        channel_id="123",
        message_id="987",
    )

    assert result.approval_outbox_job_id
    outbox = _row(state_dir, "SELECT state, not_before, created_at FROM approval_outbox WHERE approval_id=?", (approval_id,))
    assert outbox["state"] == OUTBOX_READY
    assert isinstance(outbox["not_before"], float)
    assert isinstance(outbox["created_at"], float)
    assert outbox["not_before"] <= outbox["created_at"]


@pytest.mark.asyncio
async def test_emit_callback_uses_queue_cli_and_processes_worker_once(tmp_path, monkeypatch):
    state_dir, approval_id = _queue_state(tmp_path)
    settings = _settings(tmp_path, state_dir)
    real_run = file_approval_module.subprocess.run
    command_calls = []

    def spy_run(argv, *args, **kwargs):
        command_calls.append(list(argv))
        return real_run(argv, *args, **kwargs)

    monkeypatch.setattr(file_approval_module.subprocess, "run", spy_run)
    adapter = DiscordAdapter(
        PlatformConfig(
            enabled=True,
            token="test-token",
            extra={
                "file_approval": {
                    "enabled": True,
                    "db_path": str(state_dir / "queue.db"),
                    "approver_user_ids": ["42"],
                    "commands": settings.commands,
                }
            },
        )
    )
    adapter._handle_message = AsyncMock(side_effect=AssertionError("agent dispatch forbidden"))
    message = SimpleNamespace(id=987, guild=SimpleNamespace(id=555))
    channel = SimpleNamespace(send=AsyncMock(return_value=message))
    adapter._client = SimpleNamespace(
        get_channel=MagicMock(return_value=channel),
        fetch_channel=AsyncMock(),
        add_view=MagicMock(),
    )

    await adapter._poll_file_approval_requests()
    await adapter._poll_file_approval_requests()

    channel.send.assert_called_once()
    adapter._client.add_view.assert_called_once()
    sent_view = channel.send.call_args.kwargs["view"]
    await sent_view.children[1].callback(_interaction(sent_view.children[1].custom_id))

    approval = _row(state_dir, "SELECT status, decision, ui_state FROM approval_requests WHERE approval_id=?", (approval_id,))
    assert approval == {"status": DECIDED, "decision": "true", "ui_state": "ui_updated"}
    outbox = _row(state_dir, "SELECT state, not_before, created_at FROM approval_outbox WHERE approval_id=?", (approval_id,))
    assert outbox["state"] == "DONE"
    assert isinstance(outbox["not_before"], float)
    assert isinstance(outbox["created_at"], float)
    assert outbox["not_before"] <= outbox["created_at"]
    wakeups = _process_approval_calls(command_calls)
    assert len(wakeups) == 1
    assert wakeups[0][-2:] == ["--job-id", _row(state_dir, "SELECT job_id FROM approval_outbox WHERE approval_id=?", (approval_id,))["job_id"]]
    adapter._handle_message.assert_not_called()


@pytest.mark.asyncio
async def test_concurrent_true_false_click_has_one_cas_winner(tmp_path, monkeypatch):
    state_dir, approval_id = _queue_state(tmp_path)
    settings = _settings(tmp_path, state_dir)
    _bind_posted(settings, approval_id)
    real_run = file_approval_module.subprocess.run
    command_calls = []

    def spy_run(argv, *args, **kwargs):
        command_calls.append(list(argv))
        return real_run(argv, *args, **kwargs)

    monkeypatch.setattr(file_approval_module.subprocess, "run", spy_run)
    view = _view(settings, approval_id)

    await asyncio.gather(
        view.children[0].callback(_interaction(view.children[0].custom_id, interaction_id="int-false")),
        view.children[1].callback(_interaction(view.children[1].custom_id, interaction_id="int-true")),
    )

    approval = _row(state_dir, "SELECT status, decision FROM approval_requests WHERE approval_id=?", (approval_id,))
    assert approval["status"] == DECIDED
    assert approval["decision"] in {"false", "true"}
    assert _row(state_dir, "SELECT COUNT(*) AS c FROM approval_outbox WHERE approval_id=?", (approval_id,))["c"] == 1
    assert len(_process_approval_calls(command_calls)) == 1


@pytest.mark.asyncio
async def test_ttl_boundary_rejects_expired_request_fail_closed(tmp_path, monkeypatch):
    state_dir, approval_id = _queue_state(tmp_path)
    settings = _settings(tmp_path, state_dir)
    _bind_posted(settings, approval_id)
    view = _view(settings, approval_id)
    monkeypatch.setattr(file_approval_module.time, "time", lambda: view.approval.expires_at)
    interaction = _interaction(view.children[1].custom_id)

    await view.children[1].callback(interaction)

    assert _row(state_dir, "SELECT status FROM approval_requests WHERE approval_id=?", (approval_id,))["status"] == PENDING
    assert _row(state_dir, "SELECT COUNT(*) AS c FROM approval_outbox WHERE approval_id=?", (approval_id,))["c"] == 0
    assert "expired" in interaction.response.send_message.call_args.args[0]
    assert interaction.response.send_message.call_args.kwargs["ephemeral"] is True


@pytest.mark.asyncio
async def test_interaction_replay_does_not_create_second_outbox_or_worker_wakeup(tmp_path, monkeypatch):
    state_dir, approval_id = _queue_state(tmp_path)
    settings = _settings(tmp_path, state_dir)
    _bind_posted(settings, approval_id)
    real_run = file_approval_module.subprocess.run
    command_calls = []

    def spy_run(argv, *args, **kwargs):
        command_calls.append(list(argv))
        return real_run(argv, *args, **kwargs)

    monkeypatch.setattr(file_approval_module.subprocess, "run", spy_run)
    view = _view(settings, approval_id)
    interaction = _interaction(view.children[1].custom_id, interaction_id="same-interaction")

    await view.children[1].callback(interaction)
    await view.children[1].callback(_interaction(view.children[1].custom_id, interaction_id="same-interaction"))

    assert _row(state_dir, "SELECT COUNT(*) AS c FROM approval_outbox WHERE approval_id=?", (approval_id,))["c"] == 1
    assert len(_process_approval_calls(command_calls)) == 1


@pytest.mark.asyncio
async def test_authorized_false_and_replay_idempotent_response(tmp_path):
    state_dir, approval_id = _queue_state(tmp_path)
    settings = _settings(tmp_path, state_dir)
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="test-token", extra={"file_approval": {"enabled": True, "db_path": str(state_dir / "queue.db"), "approver_user_ids": ["42"], "commands": settings.commands}}))
    adapter._client = SimpleNamespace(get_channel=MagicMock(), fetch_channel=AsyncMock())
    _bind_posted(settings, approval_id)
    view = _view(settings, approval_id)

    await view.children[0].callback(_interaction(view.children[0].custom_id))
    replay = _interaction(view.children[1].custom_id, interaction_id="int-replay")
    await view.children[1].callback(replay)

    approval = _row(state_dir, "SELECT status, decision FROM approval_requests WHERE approval_id=?", (approval_id,))
    assert approval == {"status": DECIDED, "decision": "false"}
    assert replay.response.send_message.call_args.kwargs["ephemeral"] is True
    count = _row(state_dir, "SELECT COUNT(*) AS c FROM approval_outbox WHERE approval_id=?", (approval_id,))["c"]
    assert count == 1


@pytest.mark.asyncio
async def test_exact_dedicated_auth_ignores_global_allow_all(tmp_path, monkeypatch):
    monkeypatch.setenv("DISCORD_ALLOW_ALL_USERS", "true")
    monkeypatch.setenv("GATEWAY_ALLOW_ALL_USERS", "true")
    state_dir, approval_id = _queue_state(tmp_path)
    settings = _settings(tmp_path, state_dir)
    _bind_posted(settings, approval_id)
    view = _view(settings, approval_id)
    interaction = _interaction(view.children[1].custom_id, user_id="999")

    await view.children[1].callback(interaction)

    assert _row(state_dir, "SELECT status FROM approval_requests WHERE approval_id=?", (approval_id,))["status"] == PENDING
    assert _row(state_dir, "SELECT COUNT(*) AS c FROM approval_outbox WHERE approval_id=?", (approval_id,))["c"] == 0
    interaction.response.send_message.assert_called_once()


@pytest.mark.asyncio
async def test_other_message_custom_id_replay_rejected_before_queue_cli(tmp_path):
    state_dir, approval_id = _queue_state(tmp_path)
    settings = _settings(tmp_path, state_dir)
    _bind_posted(settings, approval_id)
    view = _view(settings, approval_id)
    interaction = _interaction(view.children[1].custom_id, message_id="other")

    await view.children[1].callback(interaction)

    assert _row(state_dir, "SELECT status FROM approval_requests WHERE approval_id=?", (approval_id,))["status"] == PENDING
    assert _row(state_dir, "SELECT COUNT(*) AS c FROM approval_outbox WHERE approval_id=?", (approval_id,))["c"] == 0
    interaction.response.send_message.assert_called_once()


def test_restart_pending_view_restore(tmp_path):
    state_dir, approval_id = _queue_state(tmp_path)
    settings = _settings(tmp_path, state_dir)
    _bind_posted(settings, approval_id)
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="test-token", extra={"file_approval": {"enabled": True, "db_path": str(state_dir / "queue.db"), "approver_user_ids": ["42"], "commands": settings.commands}}))
    adapter._client = SimpleNamespace(add_view=MagicMock())

    adapter._restore_file_approval_view(pending_approvals(settings)[0])

    adapter._client.add_view.assert_called_once()
    restored_view = adapter._client.add_view.call_args.args[0]
    assert adapter._client.add_view.call_args.kwargs["message_id"] == 987
    assert restored_view.timeout is None
    assert [child.style for child in restored_view.children] == [discord.ButtonStyle.success, discord.ButtonStyle.danger]


@pytest.mark.asyncio
async def test_ui_edit_failure_repaired_without_second_decision_or_wakeup(tmp_path):
    state_dir, approval_id = _queue_state(tmp_path)
    settings = _settings(tmp_path, state_dir)
    _bind_posted(settings, approval_id)
    view = _view(settings, approval_id)
    interaction = _interaction(view.children[1].custom_id)
    interaction.response.edit_message = AsyncMock(side_effect=RuntimeError("discord edit failed"))

    await view.children[1].callback(interaction)
    assert _row(state_dir, "SELECT status, ui_state FROM approval_requests WHERE approval_id=?", (approval_id,)) == {"status": DECIDED, "ui_state": "ui_update_failed"}
    assert _row(state_dir, "SELECT state FROM approval_outbox WHERE approval_id=?", (approval_id,))["state"] == "DONE"

    repaired_message = SimpleNamespace(edit=AsyncMock())
    channel = SimpleNamespace(fetch_message=AsyncMock(return_value=repaired_message))
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="test-token", extra={"file_approval": {"enabled": True, "db_path": str(state_dir / "queue.db"), "approver_user_ids": ["42"], "commands": settings.commands}}))
    adapter._client = SimpleNamespace(get_channel=MagicMock(return_value=channel), fetch_channel=AsyncMock())

    await adapter._poll_file_approval_requests()

    repaired_message.edit.assert_called_once()
    repaired_view = repaired_message.edit.call_args.kwargs["view"]
    assert [child.style for child in repaired_view.children] == [discord.ButtonStyle.success, discord.ButtonStyle.danger]
    assert all(child.disabled for child in repaired_view.children)
    assert _row(state_dir, "SELECT ui_state FROM approval_requests WHERE approval_id=?", (approval_id,))["ui_state"] == "ui_updated"
    assert _row(state_dir, "SELECT COUNT(*) AS c FROM approval_outbox WHERE approval_id=?", (approval_id,))["c"] == 1


def test_no_direct_db_write_helpers_in_hermes_bridge():
    source = Path("plugins/platforms/discord/file_approval.py").read_text(encoding="utf-8")
    forbidden = [
        "UPDATE approval_requests",
        "INSERT INTO approval_outbox",
        "BEGIN IMMEDIATE",
        "SET status",
    ]
    for text in forbidden:
        assert text not in source


@pytest.mark.asyncio
async def test_file_approval_poller_lifecycle_cancels_and_awaits(tmp_path):
    state_dir, _approval_id = _queue_state(tmp_path)
    settings = _settings(tmp_path, state_dir)
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="test-token", extra={"file_approval": {"enabled": True, "db_path": str(state_dir / "queue.db"), "approver_user_ids": ["42"], "commands": settings.commands}}))
    adapter._client = object()
    cancelled = False

    async def loop():
        nonlocal cancelled
        try:
            while True:
                await asyncio.sleep(60)
        except asyncio.CancelledError:
            cancelled = True
            raise

    adapter._file_approval_loop = loop
    adapter._start_file_approval_poller()
    assert adapter._file_approval_task is not None
    await asyncio.sleep(0)

    await adapter._cancel_file_approval_poller()

    assert cancelled is True
    assert adapter._file_approval_task is None
