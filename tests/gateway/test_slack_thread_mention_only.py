"""Regression tests for skill-controlled Slack per-thread mention-only routing."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, ProcessingOutcome
from gateway.session import SessionSource
from gateway.session_context import clear_session_vars, set_session_vars
import plugins.platforms.slack.adapter as _slack_mod

_slack_mod.SLACK_AVAILABLE = True

from plugins.platforms.slack.adapter import SlackAdapter  # noqa: E402
from plugins.platforms.slack.thread_mode import (  # noqa: E402
    _handle_slack_thread_response_mode,
    advance_thread_history_cursor,
    get_thread_response_mode,
    get_thread_response_state,
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


async def _handle(
    adapter: SlackAdapter,
    event: dict,
    *,
    gap_context: str | None = "",
) -> None:
    with (
        patch.object(adapter, "_resolve_user_name", new=AsyncMock(return_value="tester")),
        patch.object(adapter, "_fetch_thread_context", new=AsyncMock(return_value="")),
        patch.object(
            adapter,
            "_fetch_thread_gap_context",
            new=AsyncMock(return_value=gap_context),
        ),
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


def _message_event(ts: str) -> MessageEvent:
    return MessageEvent(
        text="질문",
        source=SessionSource(
            platform=Platform.SLACK,
            user_id=USER,
            chat_id=CHANNEL,
            chat_type="thread",
            thread_id=THREAD,
        ),
        message_id=ts,
        metadata={
            "slack_history_cursor_candidate": ts,
            "slack_thread_ts": THREAD,
        },
    )


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


def test_history_cursor_persists_and_only_advances_monotonically():
    set_thread_response_mode(CHANNEL, THREAD, "mention_only")

    assert get_thread_response_state(CHANNEL, THREAD)["last_ingested_ts"] is None
    assert advance_thread_history_cursor(CHANNEL, THREAD, "1700000000.000010") is True
    assert get_thread_response_state(CHANNEL, THREAD)["last_ingested_ts"] == "1700000000.000010"

    assert advance_thread_history_cursor(CHANNEL, THREAD, "1700000000.000009") is False
    assert get_thread_response_state(CHANNEL, THREAD)["last_ingested_ts"] == "1700000000.000010"


@pytest.mark.asyncio
async def test_active_session_mention_wake_attaches_recovered_gap_as_channel_context(adapter):
    set_thread_response_mode(CHANNEL, THREAD, "mention_only")
    advance_thread_history_cursor(CHANNEL, THREAD, "1700000000.000001")

    with patch.object(adapter, "_has_active_session_for_thread", return_value=True):
        await _handle(
            adapter,
            _thread_event(f"<@{BOT}> 지금까지 논의 반영해서 답해줘"),
            gap_context="[Recovered Slack context]\n[alice] A안으로 진행",
        )

    adapter.handle_message.assert_awaited_once()
    event = adapter.handle_message.await_args.args[0]
    assert event.text == "지금까지 논의 반영해서 답해줘"
    assert event.channel_context == "[Recovered Slack context]\n[alice] A안으로 진행"
    assert event.metadata["slack_history_cursor_candidate"] == "1700000000.000010"
    assert event.metadata["slack_thread_ts"] == THREAD


@pytest.mark.asyncio
async def test_gap_fetch_failure_does_not_call_agent_or_advance_cursor(adapter):
    set_thread_response_mode(CHANNEL, THREAD, "mention_only")
    advance_thread_history_cursor(CHANNEL, THREAD, "1700000000.000001")
    adapter.send = AsyncMock()

    await _handle(
        adapter,
        _thread_event(f"<@{BOT}> 다시 확인해줘"),
        gap_context=None,
    )

    adapter.handle_message.assert_not_awaited()
    adapter.send.assert_awaited_once()
    assert get_thread_response_state(CHANNEL, THREAD)["last_ingested_ts"] == "1700000000.000001"


@pytest.mark.asyncio
async def test_successful_processing_advances_cursor_even_when_reactions_are_disabled(adapter):
    set_thread_response_mode(CHANNEL, THREAD, "mention_only")
    adapter.config.extra["reactions"] = False

    await adapter.on_processing_complete(
        _message_event("1700000000.000020"),
        ProcessingOutcome.SUCCESS,
    )

    assert get_thread_response_state(CHANNEL, THREAD)["last_ingested_ts"] == "1700000000.000020"


@pytest.mark.asyncio
async def test_failed_processing_keeps_cursor_for_replay(adapter):
    set_thread_response_mode(CHANNEL, THREAD, "mention_only")
    advance_thread_history_cursor(CHANNEL, THREAD, "1700000000.000010")

    await adapter.on_processing_complete(
        _message_event("1700000000.000020"),
        ProcessingOutcome.FAILURE,
    )

    assert get_thread_response_state(CHANNEL, THREAD)["last_ingested_ts"] == "1700000000.000010"


@pytest.mark.asyncio
async def test_fetch_thread_gap_context_orders_messages_and_excludes_current_and_self(adapter):
    adapter.set_authorization_check(
        lambda user_id, chat_type=None, chat_id=None: user_id != "U_UNVERIFIED"
    )
    adapter._resolve_user_name = AsyncMock(
        side_effect=lambda uid, **_: {
            "U_UNVERIFIED": "Alice",
            "U_VERIFIED": "Bob",
            "U_OTHER_BOT": "BuildBot",
        }.get(uid, uid)
    )
    adapter._app.client.conversations_replies = AsyncMock(
        return_value={
            "ok": True,
            "messages": [
                {"ts": "1700000000.000001", "user": USER, "text": "이전 입력"},
                {"ts": "1700000000.000002", "user": BOT, "bot_id": "B_SELF", "text": "이전 답변"},
                {"ts": "1700000000.000006", "user": "U_VERIFIED", "text": "두 번째 메시지"},
                {"ts": "1700000000.000004", "user": "U_UNVERIFIED", "text": "첫 번째 메시지"},
                {
                    "ts": "1700000000.000008",
                    "user": "U_OTHER_BOT",
                    "bot_id": "B_OTHER",
                    "text": "빌드 완료",
                },
                {"ts": "1700000000.000010", "user": USER, "text": f"<@{BOT}> 현재 질문"},
            ],
            "response_metadata": {"next_cursor": ""},
        }
    )

    context = await adapter._fetch_thread_gap_context(
        channel_id=CHANNEL,
        thread_ts=THREAD,
        current_ts="1700000000.000010",
        last_ingested_ts="1700000000.000001",
        team_id=TEAM,
    )

    assert context is not None
    assert "이전 입력" not in context
    assert "이전 답변" not in context
    assert "현재 질문" not in context
    assert context.index("첫 번째 메시지") < context.index("두 번째 메시지")
    assert context.index("두 번째 메시지") < context.index("빌드 완료")
    assert "[unverified] Alice" in context
    assert "BuildBot [bot]" in context
    awaited_call = adapter._app.client.conversations_replies.await_args
    assert awaited_call is not None
    call_kwargs = awaited_call.kwargs
    assert call_kwargs["oldest"] == "1700000000.000001"
    assert call_kwargs["latest"] == "1700000000.000010"
    assert call_kwargs["inclusive"] is False


def test_reenabling_mode_preserves_existing_history_cursor():
    set_thread_response_mode(CHANNEL, THREAD, "mention_only")
    advance_thread_history_cursor(CHANNEL, THREAD, "1700000000.000010")

    set_thread_response_mode(CHANNEL, THREAD, "mention_only")

    assert get_thread_response_state(CHANNEL, THREAD)["last_ingested_ts"] == "1700000000.000010"


@pytest.mark.asyncio
async def test_disable_during_turn_prevents_completion_hook_from_recreating_state(adapter):
    set_thread_response_mode(CHANNEL, THREAD, "mention_only")
    set_thread_response_mode(CHANNEL, THREAD, "normal")

    await adapter.on_processing_complete(
        _message_event("1700000000.000020"),
        ProcessingOutcome.SUCCESS,
    )

    assert get_thread_response_state(CHANNEL, THREAD) == {
        "mode": "normal",
        "last_ingested_ts": None,
    }


@pytest.mark.asyncio
async def test_fetch_thread_gap_context_paginates_and_deduplicates(adapter):
    adapter._resolve_user_name = AsyncMock(side_effect=lambda uid, **_: uid)
    adapter._app.client.conversations_replies = AsyncMock(
        side_effect=[
            {
                "ok": True,
                "messages": [
                    {"ts": "1700000000.000003", "user": "U_A", "text": "첫 페이지"},
                ],
                "response_metadata": {"next_cursor": "next-page"},
            },
            {
                "ok": True,
                "messages": [
                    {"ts": "1700000000.000003", "user": "U_A", "text": "중복"},
                    {"ts": "1700000000.000004", "user": "U_B", "text": "둘째 페이지"},
                ],
                "response_metadata": {"next_cursor": ""},
            },
        ]
    )

    context = await adapter._fetch_thread_gap_context(
        channel_id=CHANNEL,
        thread_ts=THREAD,
        current_ts="1700000000.000010",
        last_ingested_ts="1700000000.000001",
        team_id=TEAM,
    )

    assert context is not None
    assert "중복" in context
    assert "첫 페이지" not in context
    assert "둘째 페이지" in context
    assert adapter._app.client.conversations_replies.await_count == 2
    second_kwargs = adapter._app.client.conversations_replies.await_args_list[1].kwargs
    assert second_kwargs["cursor"] == "next-page"


@pytest.mark.asyncio
async def test_legacy_state_without_cursor_uses_latest_self_message_as_boundary(adapter):
    adapter._resolve_user_name = AsyncMock(side_effect=lambda uid, **_: uid)
    adapter._app.client.conversations_replies = AsyncMock(
        return_value={
            "ok": True,
            "messages": [
                {"ts": "1700000000.000002", "user": "U_A", "text": "이미 처리된 옛 대화"},
                {"ts": "1700000000.000003", "user": BOT, "text": "마지막 오시 답변"},
                {"ts": "1700000000.000004", "user": "U_B", "text": "복구해야 할 새 대화"},
            ],
            "response_metadata": {"next_cursor": ""},
        }
    )

    context = await adapter._fetch_thread_gap_context(
        channel_id=CHANNEL,
        thread_ts=THREAD,
        current_ts="1700000000.000010",
        last_ingested_ts=None,
        team_id=TEAM,
    )

    assert context is not None
    assert "이미 처리된 옛 대화" not in context
    assert "마지막 오시 답변" not in context
    assert "복구해야 할 새 대화" in context


@pytest.mark.asyncio
async def test_recovered_gap_reaches_gateway_model_input_as_context_block():
    from gateway.run import GatewayRunner

    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig(group_sessions_per_user=False)
    runner.adapters = {}
    setattr(runner, "_model", "test-model")
    setattr(runner, "_base_url", "")
    runner._has_setup_skill = lambda: False
    source = SessionSource(
        platform=Platform.SLACK,
        user_id=USER,
        user_name="tester",
        chat_id=CHANNEL,
        chat_type="thread",
        thread_id=THREAD,
    )
    recovered = "[Slack thread messages since your previous turn — context only.]\nAlice: A안"
    event = MessageEvent(
        text="그 내용 반영해서 답해줘",
        source=source,
        channel_context=recovered,
    )

    model_input = await runner._prepare_inbound_message_text(
        event=event,
        source=source,
        history=[],
    )

    assert model_input is not None
    assert model_input.startswith(recovered)
    # 태그가 발신자 라벨에 Slack user id 를 병기한다 — 라벨 상세에 결합하지 않는다
    assert "[New message]" in model_input
    assert "그 내용 반영해서 답해줘" in model_input
    assert "[tester" in model_input
