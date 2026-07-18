"""Agent loop control flow with a mocked LLM: iteration cap and clean stop."""

from __future__ import annotations

from pathlib import Path

import pytest
from rich.console import Console

from file_agent.agent import run_agent
from file_agent.tools import FileTools

QUIET = Console(quiet=True)


# --- minimal fakes mimicking the OpenAI response shape -------------------

class _Fn:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class _ToolCall:
    def __init__(self, id, name, arguments):
        self.id = id
        self.function = _Fn(name, arguments)


class _Msg:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls

    def model_dump(self, exclude_none=False):
        d = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            d["tool_calls"] = [
                {"id": t.id, "type": "function",
                 "function": {"name": t.function.name, "arguments": t.function.arguments}}
                for t in self.tool_calls
            ]
        return d


class _Choice:
    def __init__(self, msg):
        self.message = msg


class _Resp:
    def __init__(self, msg):
        self.choices = [_Choice(msg)]
        self.usage = None  # usage_dict tolerates None -> zeros


class AlwaysToolLLM:
    """Never stops: every turn asks for another tool call."""

    def __init__(self):
        self.calls = 0

    def chat(self, messages, tools=None, **kwargs):
        self.calls += 1
        return _Resp(_Msg(tool_calls=[
            _ToolCall(f"c{self.calls}", "list_files", '{"glob_pattern": "*.md"}')
        ]))


class ScriptedLLM:
    """One tool call, then a tool-free final answer."""

    def __init__(self):
        self.turn = 0

    def chat(self, messages, tools=None, **kwargs):
        self.turn += 1
        if self.turn == 1:
            return _Resp(_Msg(tool_calls=[
                _ToolCall("c1", "read_file", '{"path": "note.md"}')
            ]))
        return _Resp(_Msg(content="Готово: прочитал note.md."))


@pytest.fixture
def tools(tmp_path: Path) -> FileTools:
    (tmp_path / "note.md").write_text("content\n")
    return FileTools(tmp_path)


def test_loop_stops_at_iteration_limit(tools):
    llm = AlwaysToolLLM()
    result = run_agent(
        "loop forever", cfg=None, llm=llm, tools=tools,
        console=QUIET, max_iterations=4,
    )
    assert result.hit_limit is True
    assert result.iterations == 4
    assert llm.calls == 4  # exactly the cap, not more
    assert result.tools_called == ["list_files"] * 4


def test_loop_stops_when_model_answers(tools):
    result = run_agent(
        "read the note", cfg=None, llm=ScriptedLLM(), tools=tools,
        console=QUIET, max_iterations=15,
    )
    assert result.hit_limit is False
    assert result.iterations == 2
    assert result.tools_called == ["read_file"]
    assert "note.md" in result.answer


def test_dry_run_loop_collects_diff_without_writing(tmp_path):
    (tmp_path / "f.md").write_text("old\n")
    tools = FileTools(tmp_path, dry_run=True)

    class WriteThenStop:
        def __init__(self):
            self.turn = 0

        def chat(self, messages, tools=None, **kwargs):
            self.turn += 1
            if self.turn == 1:
                return _Resp(_Msg(tool_calls=[
                    _ToolCall("c1", "edit_file",
                              '{"path": "f.md", "old_text": "old", "new_text": "new"}')
                ]))
            return _Resp(_Msg(content="done"))

    result = run_agent(
        "change f", cfg=None, llm=WriteThenStop(), tools=tools,
        console=QUIET, max_iterations=15, dry_run=True,
    )
    assert (tmp_path / "f.md").read_text() == "old\n"  # disk untouched
    assert "+new" in result.diff and "-old" in result.diff
