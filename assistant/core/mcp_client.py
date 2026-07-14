"""Synchronous wrapper around an MCP stdio session.

The MCP Python SDK is async and its session must live inside the task that
created it. The CLI loop is synchronous, so we run the session on its own
event loop in a background thread and marshal calls in with
``run_coroutine_threadsafe``. The server process stays up for the whole
REPL instead of respawning per call.
"""

from __future__ import annotations

import asyncio
import os
import sys
import threading

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class McpClient:
    def __init__(self, params: StdioServerParameters, *, call_timeout: float = 30.0) -> None:
        self._params = params
        self._call_timeout = call_timeout
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._session: ClientSession | None = None
        self._stop: asyncio.Event | None = None
        self._ready = threading.Event()
        self._error: BaseException | None = None
        self.tools: list = []

    # --- lifecycle -------------------------------------------------------
    def start(self) -> "McpClient":
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self._ready.wait()
        if self._error is not None:
            raise self._error
        return self

    def _run(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._serve())
        except BaseException as e:  # surface startup failures to start()
            self._error = e
            self._ready.set()

    async def _serve(self) -> None:
        self._stop = asyncio.Event()
        async with stdio_client(self._params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                self._session = session
                self.tools = (await session.list_tools()).tools
                self._ready.set()
                await self._stop.wait()

    def stop(self) -> None:
        if self._loop and self._stop and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self._stop.set)
        if self._thread:
            self._thread.join(timeout=5)

    def __enter__(self) -> "McpClient":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    # --- calls -----------------------------------------------------------
    def call_tool(self, name: str, arguments: dict | None = None) -> str:
        if self._session is None or self._loop is None:
            raise RuntimeError("MCP client not started")
        fut = asyncio.run_coroutine_threadsafe(
            self._session.call_tool(name, arguments or {}), self._loop
        )
        res = fut.result(timeout=self._call_timeout)
        text = "\n".join(
            c.text for c in res.content if getattr(c, "text", None) is not None
        )
        if res.isError:
            raise RuntimeError(text or f"MCP tool {name} failed")
        return text


def default_server_params() -> StdioServerParameters:
    """Params to spawn this project's MCP server over stdio."""
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "assistant.mcp_server.server"],
        env=os.environ.copy(),
    )
