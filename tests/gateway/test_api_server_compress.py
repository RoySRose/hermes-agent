"""Tests for POST /api/sessions/{session_id}/compress — on-demand session compaction.

Covers:
- Route registered
- 404 on unknown/empty session
- Successful compress persists via SessionDB.replace_messages
- Busy (in-flight /v1/runs turn on the same session_id) -> 409
"""

from unittest.mock import MagicMock, patch, AsyncMock

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import (
    APIServerAdapter,
    cors_middleware,
    security_headers_middleware,
)


def _make_adapter(api_key: str = "") -> APIServerAdapter:
    extra = {}
    if api_key:
        extra["key"] = api_key
    config = PlatformConfig(enabled=True, extra=extra)
    return APIServerAdapter(config)


def _create_app(adapter: APIServerAdapter) -> web.Application:
    mws = [mw for mw in (cors_middleware, security_headers_middleware) if mw is not None]
    app = web.Application(middlewares=mws)
    app["api_server_adapter"] = adapter
    app.router.add_post("/api/sessions/{session_id}/compress", adapter._handle_compress_session)
    return app


def _make_history() -> list:
    return [
        {"role": "user", "content": "one"},
        {"role": "assistant", "content": "two"},
        {"role": "user", "content": "three"},
        {"role": "assistant", "content": "four"},
    ]


def _make_compressing_agent(session_id: str, compressed: list, rotated_session_id: str = None):
    """MagicMock standing in for a temp compression AIAgent."""
    agent = MagicMock()
    agent._cached_system_prompt = ""
    agent.tools = None
    agent.context_compressor.has_content_to_compress.return_value = True
    agent._compress_context.return_value = (compressed, "")
    agent.session_id = rotated_session_id or session_id
    return agent


@pytest.fixture
def adapter():
    return _make_adapter()


@pytest.fixture
def auth_adapter():
    return _make_adapter(api_key="sk-secret")


class TestCompressSessionRoute:
    @pytest.mark.asyncio
    async def test_route_is_registered(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_conversation_history_for_session", new_callable=AsyncMock, return_value=[]):
                resp = await cli.post("/api/sessions/unknown/compress", json={})
            assert resp.status == 404
            data = await resp.json()
            assert data["error"]["code"] == "session_not_found"

    @pytest.mark.asyncio
    async def test_requires_auth(self, auth_adapter):
        app = _create_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/api/sessions/any/compress", json={})
        assert resp.status == 401


class TestCompressSessionNotFound:
    @pytest.mark.asyncio
    async def test_404_when_session_has_no_messages(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_conversation_history_for_session", new_callable=AsyncMock, return_value=[]):
                resp = await cli.post("/api/sessions/empty-session/compress", json={})
            assert resp.status == 404
            data = await resp.json()
            assert data["error"]["code"] == "session_not_found"

    @pytest.mark.asyncio
    async def test_404_when_history_has_no_user_or_assistant_messages(self, adapter):
        """A session with only tool-role rows has nothing compressible."""
        app = _create_app(adapter)
        history = [{"role": "tool", "content": "result", "tool_call_id": "t1"}]
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_conversation_history_for_session", new_callable=AsyncMock, return_value=history):
                resp = await cli.post("/api/sessions/tool-only/compress", json={})
        assert resp.status == 404


class TestCompressSessionSuccess:
    @pytest.mark.asyncio
    async def test_compress_replaces_messages_same_session_id(self, adapter):
        """In-place / no-rotation case: replace_messages targets the same id, no reopen call."""
        history = _make_history()
        compressed = [{"role": "assistant", "content": "summary"}, history[-1]]
        session_id = "hub-channel-1"
        mock_agent = _make_compressing_agent(session_id, compressed)

        mock_db = MagicMock()
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_conversation_history_for_session", new_callable=AsyncMock, return_value=history), \
                 patch.object(adapter, "_create_agent", return_value=mock_agent), \
                 patch.object(adapter, "_ensure_session_db_async", new_callable=AsyncMock, return_value=mock_db):
                resp = await cli.post(f"/api/sessions/{session_id}/compress", json={})

            assert resp.status == 200
            data = await resp.json()
            assert data["ok"] is True
            assert data["session_id"] == session_id
            assert data["messages_before"] == len(history)
            assert data["messages_after"] == len(compressed)
        mock_db.replace_messages.assert_called_once_with(session_id, compressed)
        mock_db.reopen_session.assert_not_called()

    @pytest.mark.asyncio
    async def test_compress_reopens_session_on_rotation(self, adapter):
        """Rotated (legacy) case: replace_messages still targets the ORIGINAL
        url-path session_id (never the agent's new rotated id), and the
        original id is reopened since _compress_context ended it."""
        history = _make_history()
        compressed = [{"role": "assistant", "content": "summary"}]
        session_id = "hub-channel-2"
        mock_agent = _make_compressing_agent(session_id, compressed, rotated_session_id="20260728_000000_abcdef")

        mock_db = MagicMock()
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_conversation_history_for_session", new_callable=AsyncMock, return_value=history), \
                 patch.object(adapter, "_create_agent", return_value=mock_agent), \
                 patch.object(adapter, "_ensure_session_db_async", new_callable=AsyncMock, return_value=mock_db):
                resp = await cli.post(f"/api/sessions/{session_id}/compress", json={})

        assert resp.status == 200
        mock_db.replace_messages.assert_called_once_with(session_id, compressed)
        mock_db.reopen_session.assert_called_once_with(session_id)

    @pytest.mark.asyncio
    async def test_compress_noop_when_nothing_to_compress(self, adapter):
        history = _make_history()
        session_id = "hub-channel-3"
        mock_agent = _make_compressing_agent(session_id, history)
        mock_agent.context_compressor.has_content_to_compress.return_value = False

        mock_db = MagicMock()
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_conversation_history_for_session", new_callable=AsyncMock, return_value=history), \
                 patch.object(adapter, "_create_agent", return_value=mock_agent), \
                 patch.object(adapter, "_ensure_session_db_async", new_callable=AsyncMock, return_value=mock_db):
                resp = await cli.post(f"/api/sessions/{session_id}/compress", json={})

            assert resp.status == 200
            data = await resp.json()
            assert data["ok"] is True
            assert data["messages_before"] == data["messages_after"] == len(history)
        mock_db.replace_messages.assert_not_called()
        mock_agent._compress_context.assert_not_called()

    @pytest.mark.asyncio
    async def test_partial_compress_keeps_tail_verbatim(self, adapter):
        """keep_last N splits head/tail and rejoins the verbatim tail after
        the compressed head, mirroring the /compress "here N" boundary mode."""
        history = _make_history()  # 4 messages: u,a,u,a
        compressed_head = [{"role": "assistant", "content": "head summary"}]
        session_id = "hub-channel-4"
        mock_agent = _make_compressing_agent(session_id, compressed_head)

        mock_db = MagicMock()
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_conversation_history_for_session", new_callable=AsyncMock, return_value=history), \
                 patch.object(adapter, "_create_agent", return_value=mock_agent), \
                 patch.object(adapter, "_ensure_session_db_async", new_callable=AsyncMock, return_value=mock_db):
                resp = await cli.post(
                    f"/api/sessions/{session_id}/compress", json={"keep_last": 1}
                )

            assert resp.status == 200
            data = await resp.json()
            # head (summarized) + last 1 exchange (history[-2:], verbatim tail)
            assert data["messages_after"] == 1 + 2
        persisted = mock_db.replace_messages.call_args.args[1]
        assert persisted[0] == compressed_head[0]
        assert persisted[-2:] == history[-2:]

    @pytest.mark.asyncio
    async def test_focus_topic_forwarded_to_compress_context(self, adapter):
        history = _make_history()
        compressed = [{"role": "assistant", "content": "focused summary"}]
        session_id = "hub-channel-5"
        mock_agent = _make_compressing_agent(session_id, compressed)

        mock_db = MagicMock()
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_conversation_history_for_session", new_callable=AsyncMock, return_value=history), \
                 patch.object(adapter, "_create_agent", return_value=mock_agent), \
                 patch.object(adapter, "_ensure_session_db_async", new_callable=AsyncMock, return_value=mock_db):
                resp = await cli.post(
                    f"/api/sessions/{session_id}/compress", json={"focus": "deploy plan"}
                )

        assert resp.status == 200
        assert mock_agent._compress_context.call_args.kwargs["focus_topic"] == "deploy plan"


class TestCompressSessionBusy:
    @pytest.mark.asyncio
    async def test_busy_session_returns_409(self, adapter):
        history = _make_history()
        session_id = "hub-channel-busy"
        active_agent = MagicMock()
        active_agent.session_id = session_id
        adapter._active_run_agents["run_inflight"] = active_agent

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_conversation_history_for_session", new_callable=AsyncMock, return_value=history), \
                 patch.object(adapter, "_create_agent") as mock_create:
                resp = await cli.post(f"/api/sessions/{session_id}/compress", json={})

            assert resp.status == 409
            data = await resp.json()
            assert data["error"]["code"] == "session_busy"
            mock_create.assert_not_called()

    @pytest.mark.asyncio
    async def test_busy_check_is_scoped_to_session_id(self, adapter):
        """An in-flight run on a DIFFERENT session_id must not block this one."""
        history = _make_history()
        session_id = "hub-channel-free"
        other_agent = MagicMock()
        other_agent.session_id = "hub-channel-other"
        adapter._active_run_agents["run_other"] = other_agent
        compressed = [{"role": "assistant", "content": "summary"}]
        mock_agent = _make_compressing_agent(session_id, compressed)
        mock_db = MagicMock()

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_conversation_history_for_session", new_callable=AsyncMock, return_value=history), \
                 patch.object(adapter, "_create_agent", return_value=mock_agent), \
                 patch.object(adapter, "_ensure_session_db_async", new_callable=AsyncMock, return_value=mock_db):
                resp = await cli.post(f"/api/sessions/{session_id}/compress", json={})

        assert resp.status == 200
