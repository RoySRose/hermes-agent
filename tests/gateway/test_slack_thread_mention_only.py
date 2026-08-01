"""Regression tests for Slack per-thread mention-only routing."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import PlatformConfig
import plugins.platforms.slack.adapter as _slack_mod

_slack_mod.SLACK_AVAILABLE = True

from plugins.platforms.slack.adapter import SlackAdapter, _MentionOnlyThreadState  # noqa: E402


CHANNEL = "C_THREAD"
THREAD = "1700000000.000001"
BOT = "U_BOT"
USER = "U_USER"
TEAM = "T_TEAM"


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


@pytest.mark.asyncio
async def test_enable_phrase_sets_thread_flag_and_consumes_message(adapter):
    await _handle(adapter, _thread_event(f"<@{BOT}> 이 쓰레드는 이제 태그할 때만 나와"))

    assert (CHANNEL, THREAD) in adapter._mention_only_threads
    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_mention_then_come_out_quiet_phrase_sets_thread_flag(adapter):
    await _handle(adapter, _thread_event(f"<@{BOT}> 멘션 하면 나와, 조용히해"))

    assert (CHANNEL, THREAD) in adapter._mention_only_threads
    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_mention_only_thread_blocks_active_session_followups(adapter):
    adapter._set_slack_thread_mention_only(CHANNEL, THREAD, enabled=True)
    adapter._mentioned_threads.add(THREAD)
    adapter._bot_message_ts.add(THREAD)

    await _handle(adapter, _thread_event("네 그렇게 진행하면 됩니다"))

    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_mention_only_thread_allows_explicit_slack_mention(adapter):
    adapter._set_slack_thread_mention_only(CHANNEL, THREAD, enabled=True)

    await _handle(adapter, _thread_event(f"<@{BOT}> 이건 다시 확인해줘"))

    adapter.handle_message.assert_awaited_once()
    assert (CHANNEL, THREAD) in adapter._mention_only_threads
    assert THREAD not in adapter._mentioned_threads


@pytest.mark.asyncio
async def test_disable_phrase_clears_thread_flag(adapter):
    adapter._set_slack_thread_mention_only(CHANNEL, THREAD, enabled=True)

    await _handle(adapter, _thread_event(f"<@{BOT}> 침묵 해제하고 다시 답해도 돼"))

    assert (CHANNEL, THREAD) not in adapter._mention_only_threads
    adapter.handle_message.assert_awaited_once()


def test_mention_only_thread_cleanup_prunes_stale_entries(adapter):
    adapter.config.extra["mention_only_thread_ttl_minutes"] = 0.01
    adapter._mention_only_threads[(CHANNEL, THREAD)] = _MentionOnlyThreadState(
        enabled_at=0.0,
        updated_at=0.0,
    )

    assert not adapter._slack_thread_is_mention_only(CHANNEL, THREAD)
    assert (CHANNEL, THREAD) not in adapter._mention_only_threads
