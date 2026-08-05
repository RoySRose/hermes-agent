"""Tests for /v1/runs endpoints: start, status, events, and stop.

Covers:
- POST /v1/runs — start a run (202)
- GET /v1/runs/{run_id} — poll run status
- GET /v1/runs/{run_id}/events — SSE event stream
- POST /v1/runs/{run_id}/stop — interrupt a running agent
- Auth, error handling, and cleanup
"""

import asyncio
import json
import threading
import time
import uuid
from unittest.mock import MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import (
    APIServerAdapter,
    _approval_event_choices,
    cors_middleware,
    security_headers_middleware,
)
from tools import approval as approval_mod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("smart_denied", "allow_permanent", "expected"),
    [
        (False, True, ["once", "session", "always", "deny"]),
        (False, False, ["once", "session", "deny"]),
        (True, True, ["once", "deny"]),
        (True, False, ["once", "deny"]),
    ],
)
def test_approval_event_choices_follow_backend_capabilities(
    smart_denied, allow_permanent, expected
):
    assert _approval_event_choices(
        smart_denied=smart_denied,
        allow_permanent=allow_permanent,
    ) == expected


def _make_adapter(api_key: str = "") -> APIServerAdapter:
    """Create an adapter with optional API key."""
    extra = {}
    if api_key:
        extra["key"] = api_key
    config = PlatformConfig(enabled=True, extra=extra)
    adapter = APIServerAdapter(config)
    return adapter


def _create_runs_app(adapter: APIServerAdapter) -> web.Application:
    """Create an aiohttp app with /v1/runs routes registered."""
    mws = [mw for mw in (cors_middleware, security_headers_middleware) if mw is not None]
    app = web.Application(middlewares=mws)
    app["api_server_adapter"] = adapter
    app.router.add_post("/v1/runs", adapter._handle_runs)
    # Must precede the dynamic /v1/runs/{run_id} entry below, mirroring
    # production registration order in _http_route_table(): aiohttp's
    # UrlDispatcher resolves routes in registration order, so a static
    # "/v1/runs/meta" registered after the dynamic resource would be
    # swallowed by it (run_id="meta") instead of reaching this handler.
    app.router.add_get("/v1/runs/meta", adapter._handle_runs_meta)
    app.router.add_get("/v1/runs/{run_id}", adapter._handle_get_run)
    app.router.add_get("/v1/runs/{run_id}/events", adapter._handle_run_events)
    app.router.add_post("/v1/runs/{run_id}/approval", adapter._handle_run_approval)
    app.router.add_post("/v1/runs/{run_id}/stop", adapter._handle_stop_run)
    return app


def _make_slow_agent(**kwargs):
    """Create a mock agent that blocks in run_conversation until interrupted.

    Returns (mock_agent, agent_ready_event, interrupt_event) where
    agent_ready_event is set once run_conversation starts, and
    interrupt_event is set when interrupt() is called.
    """
    ready = threading.Event()
    interrupted = threading.Event()

    mock_agent = MagicMock()

    def _do_interrupt(message=None):
        interrupted.set()

    mock_agent.interrupt = MagicMock(side_effect=_do_interrupt)

    def _slow_run(user_message=None, conversation_history=None, task_id=None):
        ready.set()
        # Block until interrupt() is called
        interrupted.wait(timeout=10)
        return {"final_response": "interrupted"}

    mock_agent.run_conversation.side_effect = _slow_run
    mock_agent.session_prompt_tokens = 0
    mock_agent.session_completion_tokens = 0
    mock_agent.session_total_tokens = 0

    return mock_agent, ready, interrupted


def _seed_pending_approval(adapter: APIServerAdapter, run_id: str, **overrides):
    """Register a real pending approval for run_id and mark the run as
    waiting_for_approval, mirroring what an in-flight run's approval-request
    callback does. Uses the real _ApprovalEntry/_publish_run_approval code
    paths rather than hand-constructing internal state.

    Returns the queued _ApprovalEntry. Callers that don't fully resolve the
    approval must pop approval_mod._gateway_queues[run_id] themselves since
    it's shared module-level state.
    """
    payload = {
        "command": "bash -c pending-approval",
        "description": "pending approval",
        "pattern_keys": ["shell-c"],
        **overrides,
    }
    entry = approval_mod._ApprovalEntry(payload)
    adapter._run_approval_sessions[run_id] = run_id
    with approval_mod._lock:
        approval_mod._gateway_queues.setdefault(run_id, []).append(entry)
    adapter._publish_run_approval(run_id, dict(entry.data))
    return entry


@pytest.fixture
def adapter():
    return _make_adapter()


@pytest.fixture
def auth_adapter():
    return _make_adapter(api_key="sk-secret")


# ---------------------------------------------------------------------------
# POST /v1/runs — start a run
# ---------------------------------------------------------------------------


class TestStartRun:
    @pytest.mark.asyncio
    async def test_start_returns_202(self, adapter):
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent = MagicMock()
                mock_agent.run_conversation.return_value = {"final_response": "done"}
                mock_agent.session_prompt_tokens = 10
                mock_agent.session_completion_tokens = 5
                mock_agent.session_total_tokens = 15
                mock_create.return_value = mock_agent

                resp = await cli.post("/v1/runs", json={"input": "hello"})
                assert resp.status == 202
                data = await resp.json()
                assert data["status"] == "started"
                assert data["run_id"].startswith("run_")

                status_resp = await cli.get(f"/v1/runs/{data['run_id']}")
                assert status_resp.status == 200
                status = await status_resp.json()
                assert status["run_id"] == data["run_id"]
                assert status["status"] in {"queued", "running", "completed"}
                assert status["object"] == "hermes.run"

    @pytest.mark.asyncio
    async def test_start_binds_chat_id_for_delegation_wake_target(self, adapter):
        """/v1/runs must bind the raw session id as the api_server chat_id
        (like every other agent-entry route does via _run_agent): the async
        delegation dispatch reads HERMES_SESSION_CHAT_ID to pick its wake
        self-post target, and an empty binding forces background delegations
        on this route back to synchronous execution."""
        app = _create_runs_app(adapter)
        captured = {}

        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent = MagicMock()

                def _capture_run(user_message=None, conversation_history=None, task_id=None):
                    from tools.async_delegation import _current_origin_session_id

                    captured["origin_session_id"] = _current_origin_session_id()
                    return {"final_response": "done"}

                mock_agent.run_conversation.side_effect = _capture_run
                mock_agent.session_prompt_tokens = 0
                mock_agent.session_completion_tokens = 0
                mock_agent.session_total_tokens = 0
                mock_create.return_value = mock_agent

                resp = await cli.post(
                    "/v1/runs",
                    json={"input": "hello", "session_id": "runs-raw-sid"},
                )
                assert resp.status == 202
                data = await resp.json()
                run_id = data["run_id"]

                for _ in range(40):
                    status_resp = await cli.get(f"/v1/runs/{run_id}")
                    status = await status_resp.json()
                    if status["status"] == "completed":
                        break
                    await asyncio.sleep(0.05)

        assert captured.get("origin_session_id") == "runs-raw-sid", (
            "runs route must bind chat_id so delegation dispatch sees a wake target"
        )


    @pytest.mark.asyncio
    async def test_start_rejects_conflicting_route_and_request_provider(self):
        adapter = APIServerAdapter(
            PlatformConfig(
                enabled=True,
                extra={
                    "model_routes": {
                        "alias": {
                            "model": "route/model",
                            "provider": "openrouter",
                        }
                    }
                },
            )
        )
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                resp = await cli.post(
                    "/v1/runs",
                    json={
                        "input": "hello",
                        "model": "alias",
                        "provider": "minimax",
                    },
                )
                data = await resp.json()

        assert resp.status == 400
        assert "provider" in data["error"]["message"].lower()
        assert adapter._run_streams == {}
        assert adapter._run_statuses == {}
        mock_create.assert_not_called()

    @pytest.mark.asyncio
    async def test_start_passes_request_model_provider_options_to_create_agent(self, adapter):
        app = _create_runs_app(adapter)
        model_options = {"reasoning_effort": "medium", "service_tier": "priority"}
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent = MagicMock()
                mock_agent.run_conversation.return_value = {"final_response": "done"}
                mock_agent.session_prompt_tokens = 0
                mock_agent.session_completion_tokens = 0
                mock_agent.session_total_tokens = 0
                mock_create.return_value = mock_agent

                resp = await cli.post(
                    "/v1/runs",
                    json={
                        "input": "hello",
                        "model": "MiniMax-M3",
                        "provider": "minimax",
                        "model_options": model_options,
                    },
                )
                assert resp.status == 202
                for _ in range(20):
                    if mock_create.call_args is not None:
                        break
                    await asyncio.sleep(0.05)

        kwargs = mock_create.call_args.kwargs
        assert kwargs["requested_model"] == "MiniMax-M3"
        assert kwargs["requested_provider"] == "minimax"
        assert kwargs["model_options"] == model_options


# ---------------------------------------------------------------------------
# GET /v1/runs/{run_id} — poll run status
# ---------------------------------------------------------------------------


class TestRunStatus:

    @pytest.mark.asyncio
    async def test_status_reflects_explicit_session_id(self, adapter):
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent = MagicMock()
                mock_agent.run_conversation.return_value = {"final_response": "done"}
                mock_agent.session_prompt_tokens = 0
                mock_agent.session_completion_tokens = 0
                mock_agent.session_total_tokens = 0
                mock_create.return_value = mock_agent

                resp = await cli.post(
                    "/v1/runs",
                    json={"input": "hello", "session_id": "space-session"},
                )
                data = await resp.json()
                run_id = data["run_id"]

                for _ in range(20):
                    status_resp = await cli.get(f"/v1/runs/{run_id}")
                    status = await status_resp.json()
                    if status["status"] == "completed":
                        break
                    await asyncio.sleep(0.05)

                mock_agent.run_conversation.assert_called_once()
                assert mock_agent.run_conversation.call_args.kwargs["task_id"] == "space-session"
                assert status["session_id"] == "space-session"


# ---------------------------------------------------------------------------
# GET /v1/runs/{run_id}/events — SSE event stream
# ---------------------------------------------------------------------------


class TestRunEvents:
    @pytest.mark.asyncio
    async def test_events_stream_returns_completed(self, adapter):
        """Events stream should receive run.completed when agent finishes."""
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent = MagicMock()
                mock_agent.run_conversation.return_value = {"final_response": "Hello!"}
                mock_agent.session_prompt_tokens = 10
                mock_agent.session_completion_tokens = 5
                mock_agent.session_total_tokens = 15
                mock_create.return_value = mock_agent

                # Start run
                resp = await cli.post("/v1/runs", json={"input": "hello"})
                assert resp.status == 202
                data = await resp.json()
                run_id = data["run_id"]

                # Subscribe to events
                events_resp = await cli.get(f"/v1/runs/{run_id}/events")
                assert events_resp.status == 200
                body = await events_resp.text()

                # Should contain run.completed
                assert "run.completed" in body
                assert "Hello!" in body


    @pytest.mark.asyncio
    async def test_approval_resolve_all_is_scoped_to_target_run(self, auth_adapter):
        """Same client session_id must not let one run approve another run's queue."""
        app = _create_runs_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(auth_adapter, "_create_agent") as mock_create:
                victim_agent, victim_ready, victim_interrupted = _make_slow_agent()
                attacker_agent, attacker_ready, attacker_interrupted = _make_slow_agent()
                mock_create.side_effect = [victim_agent, attacker_agent]

                victim_resp = await cli.post(
                    "/v1/runs",
                    json={"input": "victim", "session_id": "shared-project"},
                    headers={"Authorization": "Bearer sk-secret"},
                )
                attacker_resp = await cli.post(
                    "/v1/runs",
                    json={"input": "attacker", "session_id": "shared-project"},
                    headers={"Authorization": "Bearer sk-secret"},
                )
                assert victim_resp.status == 202
                assert attacker_resp.status == 202
                victim_run = (await victim_resp.json())["run_id"]
                attacker_run = (await attacker_resp.json())["run_id"]

                victim_ready.wait(timeout=3.0)
                attacker_ready.wait(timeout=3.0)
                assert auth_adapter._run_approval_sessions[victim_run] == victim_run
                assert auth_adapter._run_approval_sessions[attacker_run] == attacker_run
                assert auth_adapter._run_approval_sessions[victim_run] != auth_adapter._run_approval_sessions[attacker_run]

                victim_entry = approval_mod._ApprovalEntry({
                    "command": "bash -c victim-danger",
                    "description": "victim approval",
                    "pattern_keys": ["shell-c"],
                })
                attacker_entry = approval_mod._ApprovalEntry({
                    "command": "bash -c attacker-danger",
                    "description": "attacker approval",
                    "pattern_keys": ["shell-c"],
                })
                with approval_mod._lock:
                    approval_mod._gateway_queues[victim_run] = [victim_entry]
                    approval_mod._gateway_queues[attacker_run] = [attacker_entry]

                approval_resp = await cli.post(
                    f"/v1/runs/{attacker_run}/approval",
                    json={"choice": "always", "resolve_all": True},
                    headers={"Authorization": "Bearer sk-secret"},
                )
                approval_data = await approval_resp.json()

                assert approval_resp.status == 200
                assert approval_data["resolved"] == 1
                assert attacker_entry.result == "always"
                assert attacker_entry.event.is_set()
                assert victim_entry.result is None
                assert not victim_entry.event.is_set()
                with approval_mod._lock:
                    assert approval_mod._gateway_queues[victim_run] == [victim_entry]
                    assert victim_run in approval_mod._gateway_queues
                    assert attacker_run not in approval_mod._gateway_queues

                # Clean up the synthetic pending victim approval and unblock the
                # slow test agents so their background run tasks can finish.
                with approval_mod._lock:
                    approval_mod._gateway_queues.pop(victim_run, None)
                victim_interrupted.set()
                attacker_interrupted.set()


# ---------------------------------------------------------------------------
# Run lifecycle TTL sweeping
# ---------------------------------------------------------------------------


class TestRunLifecycleSweep:

    @pytest.mark.asyncio
    async def test_expired_live_run_drops_transport_but_keeps_control_state(self, adapter):
        """Stream TTL bounds buffering without detaching a live run."""
        app = _create_runs_app(adapter)
        adapter._max_concurrent_runs = 1

        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent, agent_ready, _ = _make_slow_agent()
                mock_create.return_value = mock_agent

                start_resp = await cli.post("/v1/runs", json={"input": "hello"})
                assert start_resp.status == 202
                run_id = (await start_resp.json())["run_id"]
                assert agent_ready.wait(timeout=3.0)

                task = adapter._active_run_tasks[run_id]
                assert isinstance(task, asyncio.Task)
                assert not task.done()

                pending = approval_mod._ApprovalEntry({
                    "command": "bash -c long-running",
                    "description": "approval after stream TTL",
                    "pattern_keys": ["shell-c"],
                })
                with approval_mod._lock:
                    approval_mod._gateway_queues[run_id] = [pending]

                adapter._run_streams_created[run_id] -= adapter._RUN_STREAM_TTL + 1
                # Exercise one real sweeper iteration without waiting 60 seconds.
                with patch(
                    "gateway.platforms.api_server.asyncio.sleep",
                    side_effect=[None, asyncio.CancelledError()],
                ):
                    with pytest.raises(asyncio.CancelledError):
                        await adapter._sweep_orphaned_runs()

                assert adapter._active_run_tasks[run_id] is task
                assert adapter._active_run_agents[run_id] is mock_agent
                assert run_id not in adapter._run_streams
                assert run_id not in adapter._run_streams_created
                assert adapter._run_approval_sessions[run_id] == run_id

                limited = adapter._concurrency_limited_response()
                assert limited is not None
                assert limited.status == 429

                approval_resp = await cli.post(
                    f"/v1/runs/{run_id}/approval",
                    json={"choice": "once"},
                )
                assert approval_resp.status == 200
                assert pending.event.is_set()
                assert pending.result == "once"

                stop_resp = await cli.post(f"/v1/runs/{run_id}/stop")
                assert stop_resp.status == 200
                mock_agent.interrupt.assert_called_once_with("Stop requested via API")


# ---------------------------------------------------------------------------
# POST /v1/runs/{run_id}/stop — interrupt a running agent
# ---------------------------------------------------------------------------


class TestStopRun:

    @pytest.mark.asyncio
    async def test_stop_keeps_uncooperative_executor_tracked_until_exit(self, adapter):
        """Cancelling an asyncio wrapper must not hide its live executor thread."""
        app = _create_runs_app(adapter)
        run_can_finish = threading.Event()
        run_finished = threading.Event()

        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent = MagicMock()
                mock_agent.session_prompt_tokens = 0
                mock_agent.session_completion_tokens = 0
                mock_agent.session_total_tokens = 0
                started = threading.Event()

                def _run_conversation(*_args, **_kwargs):
                    started.set()
                    run_can_finish.wait(timeout=5)
                    run_finished.set()
                    return {"final_response": "late result"}

                mock_agent.run_conversation.side_effect = _run_conversation
                mock_create.return_value = mock_agent

                resp = await cli.post("/v1/runs", json={"input": "hello"})
                run_id = (await resp.json())["run_id"]
                assert started.wait(timeout=3)

                stop_resp = await cli.post(f"/v1/runs/{run_id}/stop")
                assert stop_resp.status == 200
                await asyncio.sleep(0.1)

                assert not run_finished.is_set()
                assert run_id in adapter._active_run_agents
                assert run_id in adapter._active_run_tasks
                assert adapter._run_statuses[run_id]["status"] == "stopping"

                run_can_finish.set()
                for _ in range(40):
                    if run_id not in adapter._active_run_tasks:
                        break
                    await asyncio.sleep(0.05)

                assert run_id not in adapter._active_run_agents
                assert run_id not in adapter._active_run_tasks
                assert adapter._run_statuses[run_id]["status"] == "cancelled"

    @pytest.mark.asyncio
    async def test_stop_running_agent(self, adapter):
        """Stop should interrupt the agent and cancel the task."""
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent, agent_ready, _ = _make_slow_agent()
                mock_create.return_value = mock_agent

                # Start run
                resp = await cli.post("/v1/runs", json={"input": "hello"})
                assert resp.status == 202
                data = await resp.json()
                run_id = data["run_id"]

                # Wait for agent to start running in the thread
                agent_ready.wait(timeout=3.0)
                await asyncio.sleep(0.1)

                # Verify agent ref is stored
                assert run_id in adapter._active_run_agents

                # Stop the run
                stop_resp = await cli.post(f"/v1/runs/{run_id}/stop")
                assert stop_resp.status == 200
                stop_data = await stop_resp.json()
                assert stop_data["run_id"] == run_id
                assert stop_data["status"] == "stopping"

                # Agent interrupt should have been called
                mock_agent.interrupt.assert_called_once_with("Stop requested via API")

                status_resp = await cli.get(f"/v1/runs/{run_id}")
                assert status_resp.status == 200
                status_data = await status_resp.json()
                assert status_data["status"] in {"stopping", "cancelled"}

                # Refs should be cleaned up
                await asyncio.sleep(0.2)
                assert run_id not in adapter._active_run_agents
                assert run_id not in adapter._active_run_tasks


    @pytest.mark.asyncio
    async def test_stop_sends_sentinel_to_events_stream(self, adapter):
        """After stop, the events stream should close."""
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent, agent_ready, _ = _make_slow_agent()
                mock_create.return_value = mock_agent

                # Start run
                resp = await cli.post("/v1/runs", json={"input": "hello"})
                assert resp.status == 202
                data = await resp.json()
                run_id = data["run_id"]

                agent_ready.wait(timeout=3.0)
                await asyncio.sleep(0.1)

                # Subscribe to events in background
                events_task = asyncio.ensure_future(
                    cli.get(f"/v1/runs/{run_id}/events")
                )

                await asyncio.sleep(0.1)

                # Stop the run
                stop_resp = await cli.post(f"/v1/runs/{run_id}/stop")
                assert stop_resp.status == 200

                # Events stream should close
                events_resp = await asyncio.wait_for(events_task, timeout=5.0)
                assert events_resp.status == 200
                body = await events_resp.text()
                # Stream should have received run.failed and closed
                assert "run.failed" in body or "stream closed" in body


class TestRunsProviderAuthFailure:
    @pytest.mark.asyncio
    async def test_status_reports_provider_auth_failure_distinctly(self, adapter):
        """/v1/runs builds its own agent via _create_agent() and does not
        route through _run_agent(), so the controlled "Provider
        authentication failed" message added there does not cover this
        endpoint. _handle_runs()'s own _ProviderAuthResolutionError branch
        must give the same distinguished message instead of the generic
        except-Exception "run failed" text."""
        from gateway.platforms.api_server import _ProviderAuthResolutionError

        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_create.side_effect = _ProviderAuthResolutionError(
                    "No credentials found for provider 'nous'"
                )

                resp = await cli.post("/v1/runs", json={"input": "hello"})
                assert resp.status == 202
                data = await resp.json()
                run_id = data["run_id"]

                for _ in range(40):
                    status_resp = await cli.get(f"/v1/runs/{run_id}")
                    status = await status_resp.json()
                    if status["status"] == "failed":
                        break
                    await asyncio.sleep(0.05)

                assert status["status"] == "failed"
                assert status["error"] == "⚠️ Provider authentication failed: No credentials found for provider 'nous'"
                assert status["last_event"] == "run.failed"


# ---------------------------------------------------------------------------
# [local-patch] run-admission-idempotency — fingerprint / sweep / reserve
# ---------------------------------------------------------------------------


class TestRunAdmissionFingerprint:
    def test_fingerprint_stable_for_identical_body(self, adapter):
        body = {"input": "hello", "model": "gpt-5", "session_id": "s1"}
        fp1 = adapter._run_admission_fingerprint(body, "gw-key")
        fp2 = adapter._run_admission_fingerprint(dict(body), "gw-key")
        assert fp1 == fp2

    @pytest.mark.parametrize(
        "field",
        [
            "input",
            "instructions",
            "previous_response_id",
            "conversation_history",
            "session_id",
            "model",
        ],
    )
    def test_fingerprint_changes_when_semantic_field_changes(self, adapter, field):
        base = {
            "input": "hello",
            "instructions": None,
            "previous_response_id": None,
            "conversation_history": None,
            "session_id": "s1",
            "model": "gpt-5",
        }
        baseline_fp = adapter._run_admission_fingerprint(base, "gw-key")
        mutated = dict(base)
        mutated[field] = "changed" if field != "conversation_history" else [{"role": "user", "content": "hi"}]
        assert adapter._run_admission_fingerprint(mutated, "gw-key") != baseline_fp

    def test_fingerprint_changes_with_gateway_session_key(self, adapter):
        body = {"input": "hello"}
        assert adapter._run_admission_fingerprint(body, "key-a") != adapter._run_admission_fingerprint(body, "key-b")
        assert adapter._run_admission_fingerprint(body, None) != adapter._run_admission_fingerprint(body, "key-a")

    def test_fingerprint_ignores_non_semantic_fields(self, adapter):
        """Fields outside the fixed semantic set (e.g. request-tracing
        metadata) must not perturb the fingerprint, or two functionally
        identical retries would be treated as conflicting."""
        body = {"input": "hello", "request_trace_id": "trace-1"}
        other = {"input": "hello", "request_trace_id": "trace-2"}
        assert adapter._run_admission_fingerprint(body, "k") == adapter._run_admission_fingerprint(other, "k")


class TestRunAdmissionSweepAndReserve:
    def test_reserve_records_entry_and_succeeds_under_capacity(self, adapter):
        ok = adapter._reserve_run_admission("key-1", "fp-1", "run_1")
        assert ok is True
        entry = adapter._run_admissions["key-1"]
        assert entry["fingerprint"] == "fp-1"
        assert entry["run_id"] == "run_1"
        assert entry["terminal"] is False
        assert entry["response_status"] == "started"

    def test_reserve_fails_closed_when_full_of_active_entries(self, adapter):
        adapter._reserve_run_admission("key-1", "fp-1", "run_1")
        adapter._MAX_RUN_ADMISSIONS = 1

        ok = adapter._reserve_run_admission("key-2", "fp-2", "run_2")

        assert ok is False
        assert "key-2" not in adapter._run_admissions

    def test_update_run_admission_status_marks_terminal_statuses(self, adapter):
        adapter._reserve_run_admission("key-1", "fp-1", "run_1")

        adapter._update_run_admission_status("run_1", "running")
        assert adapter._run_admissions["key-1"]["terminal"] is False

        adapter._update_run_admission_status("run_1", "completed")
        assert adapter._run_admissions["key-1"]["terminal"] is True

    def test_sweep_evicts_only_terminal_entries_past_ttl(self, adapter):
        """Terminal admissions expire by TTL; still-active admissions never
        do, regardless of age — evicting a live run's admission slot would
        let a retried request race a second run into existence."""
        adapter._reserve_run_admission("terminal-old", "fp-a", "run_a")
        adapter._reserve_run_admission("terminal-fresh", "fp-b", "run_b")
        adapter._reserve_run_admission("active-old", "fp-c", "run_c")
        adapter._update_run_admission_status("run_a", "completed")
        adapter._update_run_admission_status("run_b", "completed")

        stale = time.time() - adapter._RUN_ADMISSION_TTL - 1
        adapter._run_admissions["terminal-old"]["updated_at"] = stale
        adapter._run_admissions["active-old"]["updated_at"] = stale
        assert adapter._run_admissions["active-old"]["terminal"] is False

        adapter._sweep_run_admissions()

        assert "terminal-old" not in adapter._run_admissions
        assert "terminal-fresh" in adapter._run_admissions
        assert "active-old" in adapter._run_admissions

    def test_reserve_recovers_capacity_after_sweeping_expired_terminal_entries(self, adapter):
        adapter._reserve_run_admission("key-1", "fp-1", "run_1")
        adapter._update_run_admission_status("run_1", "completed")
        adapter._run_admissions["key-1"]["updated_at"] = time.time() - adapter._RUN_ADMISSION_TTL - 1
        adapter._MAX_RUN_ADMISSIONS = 1

        ok = adapter._reserve_run_admission("key-2", "fp-2", "run_2")

        assert ok is True
        assert "key-1" not in adapter._run_admissions
        assert "key-2" in adapter._run_admissions


# ---------------------------------------------------------------------------
# [local-patch] approval-admission-idempotency — fingerprint / sweep / reserve
# ---------------------------------------------------------------------------


class TestApprovalAdmissionFingerprint:
    def test_fingerprint_stable_for_identical_inputs(self, adapter):
        fp1 = adapter._approval_admission_fingerprint("run_1", "once", False, "appr_1")
        fp2 = adapter._approval_admission_fingerprint("run_1", "once", False, "appr_1")
        assert fp1 == fp2

    def test_fingerprint_changes_with_choice(self, adapter):
        fp1 = adapter._approval_admission_fingerprint("run_1", "once", False, "appr_1")
        fp2 = adapter._approval_admission_fingerprint("run_1", "deny", False, "appr_1")
        assert fp1 != fp2

    def test_fingerprint_changes_with_resolve_all(self, adapter):
        fp1 = adapter._approval_admission_fingerprint("run_1", "once", False, "appr_1")
        fp2 = adapter._approval_admission_fingerprint("run_1", "once", True, "appr_1")
        assert fp1 != fp2

    def test_fingerprint_changes_with_approval_id(self, adapter):
        fp1 = adapter._approval_admission_fingerprint("run_1", "once", False, "appr_1")
        fp2 = adapter._approval_admission_fingerprint("run_1", "once", False, "appr_2")
        assert fp1 != fp2

    def test_fingerprint_coerces_resolve_all_truthiness(self, adapter):
        """resolve_all is cast with bool(...) before hashing, so any truthy
        non-bool value must fingerprint identically to True."""
        fp_bool = adapter._approval_admission_fingerprint("run_1", "once", True, "appr_1")
        fp_truthy = adapter._approval_admission_fingerprint("run_1", "once", 1, "appr_1")
        assert fp_bool == fp_truthy


class TestApprovalAdmissionSweepAndReserve:
    def test_reserve_records_entry_and_succeeds_under_capacity(self, adapter):
        ok = adapter._reserve_approval_admission("key-1", "fp-1")
        assert ok is True
        assert adapter._approval_admissions["key-1"]["fingerprint"] == "fp-1"

    def test_reserve_fails_closed_when_full(self, adapter):
        adapter._reserve_approval_admission("key-1", "fp-1")
        adapter._MAX_APPROVAL_ADMISSIONS = 1

        ok = adapter._reserve_approval_admission("key-2", "fp-2")

        assert ok is False
        assert "key-2" not in adapter._approval_admissions

    def test_sweep_evicts_purely_by_age_no_terminal_concept(self, adapter):
        """Unlike run admissions, approval admissions have no terminal/active
        distinction — an approval decision is one-shot, so age alone governs
        eviction."""
        adapter._reserve_approval_admission("key-old", "fp-old")
        adapter._reserve_approval_admission("key-fresh", "fp-fresh")
        adapter._approval_admissions["key-old"]["created_at"] = (
            time.time() - adapter._RUN_ADMISSION_TTL - 1
        )

        adapter._sweep_approval_admissions()

        assert "key-old" not in adapter._approval_admissions
        assert "key-fresh" in adapter._approval_admissions

    def test_reserve_recovers_capacity_after_sweeping_expired_entries(self, adapter):
        adapter._reserve_approval_admission("key-1", "fp-1")
        adapter._approval_admissions["key-1"]["created_at"] = (
            time.time() - adapter._RUN_ADMISSION_TTL - 1
        )
        adapter._MAX_APPROVAL_ADMISSIONS = 1

        ok = adapter._reserve_approval_admission("key-2", "fp-2")

        assert ok is True
        assert "key-1" not in adapter._approval_admissions
        assert "key-2" in adapter._approval_admissions


# ---------------------------------------------------------------------------
# GET /v1/runs/meta
# ---------------------------------------------------------------------------


class TestRunsMeta:
    def test_meta_route_registered_before_dynamic_run_id_route(self, adapter):
        """Regression test for the ordering invariant documented in
        _http_route_table(): aiohttp resolves routes in registration order,
        so a static /v1/runs/meta registered after the dynamic
        /v1/runs/{run_id} GET route would never be reached (run_id="meta"
        would match the dynamic route first)."""
        routes = adapter._http_route_table()
        get_run_paths = [
            path for method, path, _handler in routes
            if method == "GET" and path.startswith("/v1/runs")
        ]
        assert get_run_paths.index("/v1/runs/meta") < get_run_paths.index("/v1/runs/{run_id}")

    @pytest.mark.asyncio
    async def test_meta_returns_generation_and_idempotency_info(self, adapter):
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/v1/runs/meta")
            data = await resp.json()

        assert resp.status == 200
        assert data["server_generation"] == adapter._server_generation
        assert data["idempotency"] == "v1"
        assert data["idempotency_ttl_seconds"] == adapter._RUN_ADMISSION_TTL

    @pytest.mark.asyncio
    async def test_meta_rejects_unauthenticated_request(self, auth_adapter):
        app = _create_runs_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/v1/runs/meta")

        assert resp.status == 401


# ---------------------------------------------------------------------------
# X-Hermes-Expected-Generation
# ---------------------------------------------------------------------------


class TestExpectedGeneration:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("bad_value", ["   ", "not-a-uuid", "x" * 65])
    async def test_start_run_rejects_invalid_expected_generation(self, adapter, bad_value):
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                resp = await cli.post(
                    "/v1/runs",
                    json={"input": "hello"},
                    headers={"X-Hermes-Expected-Generation": bad_value},
                )
                data = await resp.json()

        assert resp.status == 400
        assert data["error"]["message"] == "X-Hermes-Expected-Generation is invalid"
        mock_create.assert_not_called()

    @pytest.mark.asyncio
    async def test_start_run_rejects_mismatched_expected_generation(self, adapter):
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                resp = await cli.post(
                    "/v1/runs",
                    json={"input": "hello"},
                    headers={"X-Hermes-Expected-Generation": str(uuid.uuid4())},
                )
                data = await resp.json()

        assert resp.status == 409
        assert data["error"]["code"] == "server_generation_mismatch"
        mock_create.assert_not_called()

    @pytest.mark.asyncio
    async def test_start_run_accepts_matching_expected_generation(self, adapter):
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent = MagicMock()
                mock_agent.run_conversation.return_value = {"final_response": "done"}
                mock_agent.session_prompt_tokens = 0
                mock_agent.session_completion_tokens = 0
                mock_agent.session_total_tokens = 0
                mock_create.return_value = mock_agent

                resp = await cli.post(
                    "/v1/runs",
                    json={"input": "hello"},
                    headers={"X-Hermes-Expected-Generation": adapter._server_generation},
                )

        assert resp.status == 202


class TestApprovalExpectedGeneration:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("bad_value", ["   ", "not-a-uuid", "x" * 65])
    async def test_approval_rejects_invalid_expected_generation(self, adapter, bad_value):
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/v1/runs/run_x/approval",
                json={"choice": "once"},
                headers={"X-Hermes-Expected-Generation": bad_value},
            )
            data = await resp.json()

        assert resp.status == 400
        assert data["error"]["message"] == "X-Hermes-Expected-Generation is invalid"

    @pytest.mark.asyncio
    async def test_approval_rejects_mismatched_expected_generation(self, adapter):
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/v1/runs/run_x/approval",
                json={"choice": "once"},
                headers={"X-Hermes-Expected-Generation": str(uuid.uuid4())},
            )
            data = await resp.json()

        assert resp.status == 409
        assert data["error"]["code"] == "server_generation_mismatch"


# ---------------------------------------------------------------------------
# Idempotency-Key header validation
# ---------------------------------------------------------------------------


class TestIdempotencyKeyValidation:
    @pytest.mark.asyncio
    async def test_start_run_rejects_empty_idempotency_key_header(self, adapter):
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                resp = await cli.post(
                    "/v1/runs",
                    json={"input": "hello"},
                    headers={"Idempotency-Key": "   "},
                )
                data = await resp.json()

        assert resp.status == 400
        assert data["error"]["message"] == "Idempotency-Key must be nonempty"
        mock_create.assert_not_called()

    @pytest.mark.asyncio
    async def test_start_run_rejects_oversized_idempotency_key_header(self, adapter):
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                resp = await cli.post(
                    "/v1/runs",
                    json={"input": "hello"},
                    headers={"Idempotency-Key": "k" * 257},
                )
                data = await resp.json()

        assert resp.status == 400
        assert data["error"]["message"] == "Idempotency-Key exceeds maximum length"
        mock_create.assert_not_called()

    @pytest.mark.asyncio
    async def test_approval_rejects_empty_idempotency_key_header(self, adapter):
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/v1/runs/run_x/approval",
                json={"choice": "once"},
                headers={"Idempotency-Key": "   "},
            )
            data = await resp.json()

        assert resp.status == 400
        assert data["error"]["message"] == "Idempotency-Key must be nonempty"

    @pytest.mark.asyncio
    async def test_approval_rejects_oversized_idempotency_key_header(self, adapter):
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/v1/runs/run_x/approval",
                json={"choice": "once"},
                headers={"Idempotency-Key": "k" * 257},
            )
            data = await resp.json()

        assert resp.status == 400
        assert data["error"]["message"] == "Idempotency-Key exceeds maximum length"


# ---------------------------------------------------------------------------
# /v1/runs retry semantics + admission capacity
# ---------------------------------------------------------------------------


class TestRunIdempotentRetry:
    @pytest.mark.asyncio
    async def test_same_key_same_input_replay_does_not_create_second_run(self, adapter):
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent = MagicMock()
                mock_agent.run_conversation.return_value = {"final_response": "done"}
                mock_agent.session_prompt_tokens = 0
                mock_agent.session_completion_tokens = 0
                mock_agent.session_total_tokens = 0
                mock_create.return_value = mock_agent

                first = await cli.post(
                    "/v1/runs",
                    json={"input": "hello"},
                    headers={"Idempotency-Key": "retry-key-1"},
                )
                assert first.status == 202
                run_id = (await first.json())["run_id"]

                for _ in range(40):
                    status_resp = await cli.get(f"/v1/runs/{run_id}")
                    status = await status_resp.json()
                    if status["status"] == "completed":
                        break
                    await asyncio.sleep(0.05)
                assert status["status"] == "completed"

                second = await cli.post(
                    "/v1/runs",
                    json={"input": "hello"},
                    headers={"Idempotency-Key": "retry-key-1"},
                )
                assert second.status == 202
                second_data = await second.json()

        assert second_data["run_id"] == run_id
        mock_create.assert_called_once()

    @pytest.mark.asyncio
    async def test_same_key_different_input_returns_409(self, adapter):
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent = MagicMock()
                mock_agent.run_conversation.return_value = {"final_response": "done"}
                mock_agent.session_prompt_tokens = 0
                mock_agent.session_completion_tokens = 0
                mock_agent.session_total_tokens = 0
                mock_create.return_value = mock_agent

                first = await cli.post(
                    "/v1/runs",
                    json={"input": "hello"},
                    headers={"Idempotency-Key": "retry-key-2"},
                )
                assert first.status == 202

                second = await cli.post(
                    "/v1/runs",
                    json={"input": "different input"},
                    headers={"Idempotency-Key": "retry-key-2"},
                )
                data = await second.json()

                for _ in range(20):
                    if mock_create.call_count >= 1:
                        break
                    await asyncio.sleep(0.05)

        assert second.status == 409
        assert data["error"]["message"] == "Idempotency-Key was already used with different admission inputs"
        assert mock_create.call_count == 1

    @pytest.mark.asyncio
    async def test_admission_registry_full_returns_503(self, adapter):
        adapter._reserve_run_admission("filler-key", "filler-fp", "run_filler")
        adapter._MAX_RUN_ADMISSIONS = 1
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                resp = await cli.post(
                    "/v1/runs",
                    json={"input": "hello"},
                    headers={"Idempotency-Key": "new-key"},
                )
                data = await resp.json()

        assert resp.status == 503
        assert data["error"]["code"] == "run_admission_capacity"
        mock_create.assert_not_called()


# ---------------------------------------------------------------------------
# [local-patch] approval-admission-idempotency — POST /v1/runs/{run_id}/approval
# ---------------------------------------------------------------------------


class TestApprovalIdempotency:
    @pytest.mark.asyncio
    async def test_approval_requires_approval_id_when_keyed(self, adapter):
        """approval_id is optional for unkeyed (FIFO) requests, but a keyed
        request must name the exact approval it resolves."""
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/v1/runs/run_x/approval",
                json={"choice": "once"},
                headers={"Idempotency-Key": "some-key"},
            )
            data = await resp.json()

        assert resp.status == 400
        assert data["error"]["message"] == "approval_id must be a nonempty string within the maximum length"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("bad_approval_id", ["", "   ", "x" * 300])
    async def test_approval_rejects_invalid_approval_id_value(self, adapter, bad_approval_id):
        """The approval_id shape check fires unconditionally (with or
        without an Idempotency-Key) whenever the field is present."""
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/v1/runs/run_x/approval",
                json={"choice": "once", "approval_id": bad_approval_id},
            )
            data = await resp.json()

        assert resp.status == 400
        assert data["error"]["message"] == "approval_id must be a nonempty string within the maximum length"

    @pytest.mark.asyncio
    async def test_approval_keyed_request_cannot_resolve_all(self, adapter):
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/v1/runs/run_x/approval",
                json={"choice": "once", "approval_id": "appr_1", "resolve_all": True},
                headers={"Idempotency-Key": "some-key"},
            )
            data = await resp.json()

        assert resp.status == 400
        assert data["error"]["code"] == "invalid_approval_resolution_scope"

    @pytest.mark.asyncio
    async def test_approval_id_mismatch_returns_409(self, adapter):
        app = _create_runs_app(adapter)
        run_id = "run_mismatch_test"
        _seed_pending_approval(adapter, run_id)
        try:
            async with TestClient(TestServer(app)) as cli:
                resp = await cli.post(
                    f"/v1/runs/{run_id}/approval",
                    json={"choice": "once", "approval_id": "not-the-pending-one"},
                    headers={"Idempotency-Key": "mismatch-key"},
                )
                data = await resp.json()

            assert resp.status == 409
            assert data["error"]["code"] == "approval_id_mismatch"
        finally:
            with approval_mod._lock:
                approval_mod._gateway_queues.pop(run_id, None)

    @pytest.mark.asyncio
    async def test_approval_replay_returns_cached_response_without_reprocessing(self, adapter):
        app = _create_runs_app(adapter)
        run_id = "run_replay_test"
        entry = _seed_pending_approval(adapter, run_id)
        real_approval_id = entry.approval_id
        try:
            with patch(
                "tools.approval.resolve_gateway_approval_with_next",
                wraps=approval_mod.resolve_gateway_approval_with_next,
            ) as spy:
                async with TestClient(TestServer(app)) as cli:
                    body = {"choice": "once", "approval_id": real_approval_id}
                    headers = {"Idempotency-Key": "approval-replay-key"}

                    first = await cli.post(f"/v1/runs/{run_id}/approval", json=body, headers=headers)
                    first_data = await first.json()
                    assert first.status == 200
                    assert first_data["resolved"] == 1

                    second = await cli.post(f"/v1/runs/{run_id}/approval", json=body, headers=headers)
                    second_data = await second.json()

            assert second.status == 200
            assert second_data == first_data
            assert spy.call_count == 1
        finally:
            with approval_mod._lock:
                approval_mod._gateway_queues.pop(run_id, None)

    @pytest.mark.asyncio
    async def test_approval_rejected_when_admission_registry_full(self, adapter):
        app = _create_runs_app(adapter)
        run_id = "run_capacity_test"
        entry = _seed_pending_approval(adapter, run_id)
        adapter._reserve_approval_admission("filler-key", "filler-fp")
        adapter._MAX_APPROVAL_ADMISSIONS = 1
        try:
            async with TestClient(TestServer(app)) as cli:
                resp = await cli.post(
                    f"/v1/runs/{run_id}/approval",
                    json={"choice": "once", "approval_id": entry.approval_id},
                    headers={"Idempotency-Key": "capacity-key"},
                )
                data = await resp.json()

            assert resp.status == 503
            assert data["error"]["code"] == "approval_admission_capacity"
        finally:
            with approval_mod._lock:
                approval_mod._gateway_queues.pop(run_id, None)
