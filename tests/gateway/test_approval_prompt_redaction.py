"""Regression test for approval prompt credential redaction (issue #48456).

When Tirith flags a command for containing a credential-shaped pattern, the
gateway approval prompt must redact the credential from the command text
before sending it to the chat platform. Without this fix, the raw command
(with the credential in plaintext) is sent verbatim to Telegram/Discord/etc.,
undoing Tirith's redaction one layer up.

The redaction is wired through the module-level ``_redact_approval_command``
seam. These tests bind that seam -- the production wiring -- not just the
underlying ``redact_sensitive_text`` helper, so they fail if the redaction
call is removed from either approval path.

Credential fixtures are built at runtime from a benign prefix + a run of
``X`` characters (the same trick tests/agent/test_redact.py uses): they match
the redactor regexes so the assertions stay meaningful, but contain no real
or real-looking key, so secret scanners do not flag this file.
"""

from gateway.run import _redact_approval_command

# Synthetic, scanner-safe credential fixtures. Each matches its redactor
# regex (ghp_/sk-/JWT) but is unmistakably fake -- a run of X's, never a
# real or real-format key.
_FAKE_GHP = "ghp_" + "X" * 36
_FAKE_OPENAI = "sk-proj-" + "X" * 40
_FAKE_JWT = "eyJ" + "X" * 20 + "." + "eyJ" + "X" * 24 + "." + "X" * 30


class TestRedactApprovalCommand:
    """Contract for the approval-prompt redaction seam used by the gateway."""

    def test_redacts_github_pat(self):
        raw = "curl -H 'Authorization: token " + _FAKE_GHP + "' https://api.github.com/user"
        out = _redact_approval_command(raw)
        assert _FAKE_GHP not in out
        # command structure preserved so the operator can still judge the action
        assert "curl" in out
        assert "github.com" in out

    def test_redacts_openai_key(self):
        raw = "export OPENAI_API_KEY=" + _FAKE_OPENAI + " && python s.py"
        out = _redact_approval_command(raw)
        assert _FAKE_OPENAI not in out
        assert "python s.py" in out

    def test_redacts_bearer_token(self):
        raw = "curl -H 'Authorization: Bearer " + _FAKE_JWT + "' https://api.example.com"
        out = _redact_approval_command(raw)
        assert _FAKE_JWT not in out


    def test_forces_redaction_even_when_disabled(self, monkeypatch):
        """force=True must redact even if security.redact_secrets is off -- the
        approval prompt is a hard secret-egress boundary regardless of config."""
        raw = "curl -H 'Authorization: token " + _FAKE_GHP + "' https://api.github.com"
        # With redaction globally disabled, the seam must STILL redact (force=True).
        monkeypatch.setattr("agent.redact._REDACT_ENABLED", False, raising=False)
        out = _redact_approval_command(raw)
        assert _FAKE_GHP not in out


class TestApprovalCommandWiring:
    """Guard the production wiring on BOTH approval-notify transports:
    1. the chat-platform path (_approval_notify_sync in gateway/run.py),
       which redacts and reassigns the command inline before it reaches
       send_exec_approval, and
    2. the SSE/API path (gateway/platforms/api_server.py), where every
       outbound approval event is built by _approval_request_event (which
       redacts and reassigns the command into the event dict) and
       _publish_run_approval is the sole producer that turns such an event
       into an SSE put_nowait -- it must call the builder before
       enqueueing. _approval_notify is the callback registered for this
       path and must not enqueue directly, so there is no route around
       the builder.
    Uses AST (not char-offset string slicing) so a benign refactor doesn't
    cause a false failure, and so a discarded-result call
    (`_redact(cmd); send(cmd)`) does NOT pass."""

    def _find_function(self, module, func_name: str):
        """Parse `module`'s full source into an AST and return
        `(source, node)` for the (possibly nested) FunctionDef/
        AsyncFunctionDef named `func_name`. Walking the real AST (not a
        source slice) means the search is refactor-robust regardless of
        nesting depth or surrounding line churn."""
        import ast
        import inspect

        source = inspect.getsource(module)
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == func_name:
                return source, node
        raise AssertionError(f"function {func_name} not found in {module.__name__}")

    def _assert_redacts_then_uses(
        self, module, func_name: str, sink_substr: str,
        *, call_name: str = "_redact_approval_command",
    ):
        """Locate `func_name` in `module` and assert it contains an
        assignment `<x> = call_name(...)` whose result is then used by a
        statement matching `sink_substr` on a LATER line. `call_name`
        matches both a bare-name call (`_redact_approval_command(...)`) and
        a method call (`self._approval_request_event(...)`), so the same
        walk guards the module-level redaction seam and the SSE
        event-builder chokepoint. This rejects discarded-result calls (the
        call must be an assignment, not a bare expression)."""
        import ast

        source, target_fn = self._find_function(module, func_name)

        redact_line = None
        for node in ast.walk(target_fn):
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
                fn = node.value.func
                called = fn.id if isinstance(fn, ast.Name) else getattr(fn, "attr", None)
                if called == call_name:
                    redact_line = node.lineno
        assert redact_line is not None, (
            f"{func_name} must assign the result of {call_name}(...) "
            "(a discarded-result call would still leak the raw command)"
        )

        sink_line = None
        for node in ast.walk(target_fn):
            seg = ast.get_source_segment(source, node)
            if seg and sink_substr in seg and getattr(node, "lineno", 0) > redact_line:
                sink_line = node.lineno
                break
        assert sink_line is not None, (
            f"`{sink_substr}` sink not found after the redaction in {func_name}"
        )

    def _assert_no_bypass_call(self, module, func_name: str, forbidden_substr: str):
        """Locate `func_name` in `module` and assert no node in its body
        (other than the function definition itself) has a source segment
        containing `forbidden_substr` -- i.e. the function has no direct
        route to that sink and must delegate through the chokepoint
        instead."""
        import ast

        source, target_fn = self._find_function(module, func_name)

        bypass_line = None
        for node in ast.walk(target_fn):
            if node is target_fn:
                continue
            seg = ast.get_source_segment(source, node)
            if seg and forbidden_substr in seg:
                bypass_line = node.lineno
                break
        assert bypass_line is None, (
            f"{func_name} must not call `{forbidden_substr}` directly "
            "(that would bypass the redaction chokepoint)"
        )

    def test_chat_platform_path_redacts_before_send(self):
        import gateway.run as run

        self._assert_redacts_then_uses(run, "_approval_notify_sync", "send_exec_approval")

    def test_sse_api_path_redacts_before_enqueue(self):
        from gateway.platforms import api_server

        # The shared event builder assigns (not discards) the redacted
        # command into the outbound event.
        self._assert_redacts_then_uses(
            api_server, "_approval_request_event", "return event",
        )
        # The sole producer that turns an event into an SSE enqueue must
        # call the builder before putting it on the queue.
        self._assert_redacts_then_uses(
            api_server, "_publish_run_approval", "put_nowait",
            call_name="_approval_request_event",
        )
        # No bypass: the registered notify callback must not enqueue
        # directly.
        self._assert_no_bypass_call(api_server, "_approval_notify", "put_nowait")


class TestApprovalTextFallbackContract:
    def test_smart_deny_only_advertises_one_operation(self):
        from gateway.run import _format_exec_approval_fallback

        text = _format_exec_approval_fallback(
            "rm -rf /", "dangerous deletion", "/",
            allow_permanent=False, smart_denied=True,
        )
        assert "owner override" in text.lower()
        assert "one operation" in text.lower()
        assert "`/approve`" in text
        assert "approve session" not in text
        assert "approve always" not in text


