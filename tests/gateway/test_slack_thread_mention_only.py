"""Regression tests for skill-controlled Slack per-thread mention-only routing."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import PlatformConfig
from gateway.session_context import clear_session_vars, set_session_vars
import plugins.platforms.slack.adapter as _slack_mod

_slack_mod.SLACK_AVAILABLE = True

from plugins.platforms.slack.adapter import SlackAdapter  # noqa: E402
from plugins.platforms.slack.thread_mode import (  # noqa: E402
    _handle_slack_thread_response_mode,
    get_thread_response_mode,
    set_thread_response_mode,
)


CHANNEL = "C_THREAD"
THREAD = "1700000000.000001"
BOT = "U_BOT"
USER = "U_USER"
TEAM = "T_TEAM"


@pytest.fixture(autouse=True)
def isolated_hermes_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes-home"))


@pytest.fixture
def adapter():
    adapter = SlackAdapter(PlatformConfig(enabled=True, token="xoxb-test"))
    adapter._app = MagicMock()
    adapter._app.client = AsyncMock()
    adapter._bot_user_id = BOT
    adapter._running = True
    adapter.handle_message = AsyncMock()
    return adapter


def _thread_event(text: str, *, ts: str = "1700000000.000010") -> dict:
    return {
        "channel": CHANNEL,
        "channel_type": "channel",
        "team": TEAM,
        "user": USER,
        "text": text,
        "ts": ts,
        "thread_ts": THREAD,
    }


async def _handle(adapter: SlackAdapter, event: dict) -> None:
    with (
        patch.object(adapter, "_resolve_user_name", new=AsyncMock(return_value="tester")),
        patch.object(adapter, "_fetch_thread_context", new=AsyncMock(return_value="")),
        patch.object(adapter, "_fetch_thread_parent_text", new=AsyncMock(return_value=None)),
    ):
        await adapter._handle_slack_message(event)


def _call_mode_tool(action: str, *, platform: str = "slack") -> dict:
    tokens = set_session_vars(
        platform=platform,
        chat_id=CHANNEL,
        thread_id=THREAD,
        session_id="session-test",
    )
    try:
        return json.loads(_handle_slack_thread_response_mode({"action": action}))
    finally:
        clear_session_vars(tokens)


def test_agent_tool_enables_and_disables_persistent_thread_mode():
    enabled = _call_mode_tool("enable")
    assert enabled["success"] is True
    assert enabled["mode"] == "mention_only"
    assert get_thread_response_mode(CHANNEL, THREAD) == "mention_only"

    disabled = _call_mode_tool("disable")
    assert disabled["success"] is True
    assert disabled["mode"] == "normal"
    assert get_thread_response_mode(CHANNEL, THREAD) == "normal"


def test_agent_tool_rejects_non_slack_sessions():
    result = _call_mode_tool("enable", platform="discord")
    assert "error" in result
    assert get_thread_response_mode(CHANNEL, THREAD) == "normal"


@pytest.mark.asyncio
async def test_control_language_is_not_hardcoded_or_consumed_by_adapter(adapter):
    await _handle(adapter, _thread_event(f"<@{BOT}> 이 스레드에서는 앞으로 태그할 때만 나와"))

    adapter.handle_message.assert_awaited_once()
    assert get_thread_response_mode(CHANNEL, THREAD) == "normal"


@pytest.mark.asyncio
async def test_mention_only_thread_blocks_active_session_followups(adapter):
    set_thread_response_mode(CHANNEL, THREAD, "mention_only")
    adapter._mentioned_threads.add(THREAD)
    adapter._bot_message_ts.add(THREAD)

    await _handle(adapter, _thread_event("네 그렇게 진행하면 됩니다"))

    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_mention_only_thread_allows_explicit_slack_mention(adapter):
    set_thread_response_mode(CHANNEL, THREAD, "mention_only")

    await _handle(adapter, _thread_event(f"<@{BOT}> 이건 다시 확인해줘"))

    adapter.handle_message.assert_awaited_once()
    assert get_thread_response_mode(CHANNEL, THREAD) == "mention_only"
    assert THREAD not in adapter._mentioned_threads


@pytest.mark.asyncio
async def test_wake_word_pattern_does_not_bypass_explicit_mention_only_mode(adapter):
    set_thread_response_mode(CHANNEL, THREAD, "mention_only")

    with patch.object(adapter, "_slack_message_matches_mention_patterns", return_value=True):
        await _handle(adapter, _thread_event("오시야 이건 확인해줘"))

    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_mode_persists_across_adapter_instances(adapter):
    set_thread_response_mode(CHANNEL, THREAD, "mention_only")

    replacement = SlackAdapter(PlatformConfig(enabled=True, token="xoxb-test"))
    replacement._app = MagicMock()
    replacement._app.client = AsyncMock()
    replacement._bot_user_id = BOT
    replacement._running = True
    replacement.handle_message = AsyncMock()

    await _handle(replacement, _thread_event("재시작 뒤에도 답하지 마"))

    replacement.handle_message.assert_not_awaited()
