"""Executor test: a requires_confirmation tool never runs without a yes."""

from __future__ import annotations

from assistant.core.tools import Tool, ToolExecutor, ToolRegistry


def _registry_with_gated_tool(state: dict) -> ToolRegistry:
    def handler(**kwargs) -> str:
        state["ran"] = True
        return "did the thing"

    reg = ToolRegistry()
    reg.register(
        Tool(
            name="git_push",
            description="gated",
            parameters={"type": "object", "properties": {}},
            handler=handler,
            requires_confirmation=True,
        )
    )
    return reg


def test_gated_tool_not_run_when_declined():
    state = {"ran": False}
    asked = {"prompt": None}

    def confirm(prompt: str) -> bool:
        asked["prompt"] = prompt
        return False

    ex = ToolExecutor(_registry_with_gated_tool(state), confirm)
    result = ex.execute("git_push", {})

    assert state["ran"] is False
    assert "Cancelled" in result
    assert asked["prompt"] is not None  # user was actually asked


def test_gated_tool_runs_when_confirmed():
    state = {"ran": False}
    ex = ToolExecutor(_registry_with_gated_tool(state), lambda _: True)
    result = ex.execute("git_push", {})
    assert state["ran"] is True
    assert "did the thing" in result


def test_normal_tool_runs_without_asking():
    calls = {"confirm_asked": False}

    def confirm(_: str) -> bool:
        calls["confirm_asked"] = True
        return True

    reg = ToolRegistry()
    reg.register(
        Tool(
            name="rag_search",
            description="normal",
            parameters={"type": "object", "properties": {}},
            handler=lambda **k: "ok",
            requires_confirmation=False,
        )
    )
    ex = ToolExecutor(reg, confirm)
    assert ex.execute("rag_search", {}) == "ok"
    assert calls["confirm_asked"] is False  # not gated -> never prompts


def test_unknown_tool_reports_error():
    ex = ToolExecutor(ToolRegistry(), lambda _: True)
    assert "unknown tool" in ex.execute("nope", {}).lower()
