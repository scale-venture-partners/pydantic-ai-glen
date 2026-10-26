"""Glen capability: a `glen_search` tool plus automatic recording of every run."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
from dataclasses import dataclass, field
from typing import Any

import httpx
from pydantic_ai import RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import UserError
from pydantic_ai.run import AgentRunResult
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import AbstractToolset, FunctionToolset

from pydantic_ai_glen._messages import to_glen_messages

__all__ = ("Glen",)

log = logging.getLogger("glen")

DEFAULT_URL = "https://app.tryglen.com"
_RETRYABLE = (429, 500, 502, 503, 504)
_ATTEMPTS = 3

# Paid for on every request, so keep it short. Mirrors what the Claude Code plugin injects.
_INSTRUCTIONS = (
    "You have Glen, your team's shared memory of past agent conversations, decisions and "
    "incidents. Before you change, fix or extend a named system, call `glen_search` first: "
    "ask one full question that names the system, the behavior and the kind of answer you "
    "need (a rationale, an owner, a sequence of events), not keywords. Treat the answer as "
    "recalled memory that may be outdated, and cite it when you use it. When you share a "
    "GitHub pull request URL, also give its Glen review link: {url}/<owner>/<repo>/pull/<number>."
)


@dataclass
class Glen(AbstractCapability[AgentDepsT]):
    """Connect an agent to Glen's shared team memory.

    Adds a `glen_search` tool and records every run to Glen once it finishes. Recording
    runs in the background; call `await glen.flush()` before a short-lived process exits.
    Under `run_sync` the event loop closes with the run, so set `background=False`.
    """

    api_key: str | None = field(default=None, repr=False)
    """Defaults to `GLEN_API_KEY`."""

    org_id: str | None = None
    """Defaults to `GLEN_ORG_ID`."""

    url: str | None = None
    """Defaults to `GLEN_URL`, then https://app.tryglen.com."""

    search: bool = True
    """Expose the `glen_search` tool."""

    record: bool = True
    """Send each finished run to Glen."""

    instructions: bool = True
    """Tell the agent to search Glen before changing a system the team already worked on.
    Only used with `search`; turn it off if your own prompt says it."""

    background: bool = True
    """Record without delaying the run. When False, the run waits for the upload."""

    agent_name: str = "pydantic-ai"
    agent_version: str | None = None

    workspace: dict[str, str] | None = None
    """Optional `cwd`, `repoName`, `branch`, `commit`, `remoteUrl` for the recorded turns."""

    retry_backoff: float = 1.0
    """Base seconds between recording retries, doubled each attempt."""

    transport: httpx.AsyncBaseTransport | None = field(default=None, repr=False)
    """Override the HTTP transport (tests)."""

    _client: httpx.AsyncClient | None = field(default=None, init=False, repr=False)
    _pending: set[asyncio.Task[None]] = field(default_factory=set, init=False, repr=False)

    def __post_init__(self) -> None:
        self.api_key = self.api_key or os.environ.get("GLEN_API_KEY")
        self.org_id = self.org_id or os.environ.get("GLEN_ORG_ID")
        self.url = self.url or os.environ.get("GLEN_URL") or DEFAULT_URL
        if not self.api_key or not self.org_id:
            raise UserError(
                "Glen needs an API key and organization id: pass api_key/org_id "
                "or set GLEN_API_KEY and GLEN_ORG_ID."
            )

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.url or DEFAULT_URL,
                headers={
                    "x-api-key": self.api_key or "",
                    "x-glen-org-id": self.org_id or "",
                    "content-type": "application/json",
                },
                # Search reads memory and writes an answer, so reads can be slow.
                timeout=httpx.Timeout(10.0, read=120.0),
                transport=self.transport,
            )
        return self._client

    def get_instructions(self) -> str | None:
        if not (self.search and self.instructions):
            return None
        return _INSTRUCTIONS.format(url=self.url)

    def get_toolset(self) -> AbstractToolset[AgentDepsT] | None:
        if not self.search:
            return None
        toolset: FunctionToolset[AgentDepsT] = FunctionToolset()

        @toolset.tool
        async def glen_search(ctx: RunContext[AgentDepsT], query: str) -> str:
            """Search the team's shared memory: past agent conversations, decisions,
            incidents, and why things were built the way they were.

            Use it before changing or explaining a system the team already worked on.
            Ask one full question with context (system, behavior, what you need),
            not a keyword. Example: "why does the billing worker batch at 500 instead
            of streaming, and who decided".
            """
            return await self._search(query, ctx.conversation_id)

        return toolset

    async def _search(self, query: str, conversation_id: str | None) -> str:
        body: dict[str, Any] = {"query": query}
        if conversation_id:
            body["conversationId"] = conversation_id
        try:
            r = await self._http().post("/api/mcp/route/search", json=body)
        except httpx.HTTPError as e:
            log.warning("glen search failed: %s", e)
            return "Glen search is unavailable right now."
        if r.status_code != 200:
            log.warning("glen search returned %s: %s", r.status_code, r.text[:500])
            return f"Glen search failed ({r.status_code})."
        data = r.json()
        # An empty answer means the user turned Glen off or silent.
        if not data.get("answer"):
            return "Glen has no memory to share for this (it may be turned off)."
        gaps = data.get("gaps") or []
        suffix = "\n\nCould not verify:\n- " + "\n- ".join(gaps) if gaps else ""
        return data["answer"] + suffix

    async def after_run(
        self, ctx: RunContext[AgentDepsT], *, result: AgentRunResult[Any]
    ) -> AgentRunResult[Any]:
        if not self.record:
            return result
        try:
            body = self._turn_body(result)
        except Exception:
            log.exception("glen could not build the turn payload")
            return result
        if self.background:
            task = asyncio.create_task(self._send_turn(body))
            self._pending.add(task)
            task.add_done_callback(self._pending.discard)
        else:
            await self._send_turn(body)
        return result

    def _turn_body(self, result: AgentRunResult[Any]) -> dict[str, Any]:
        agent: dict[str, str] = {"name": self.agent_name}
        if self.agent_version:
            agent["version"] = self.agent_version
        context: dict[str, Any] = {"agent": agent}
        if self.workspace:
            context["workspace"] = self.workspace
        key = f"{result.conversation_id}\0{result.run_id}"
        return {
            "conversationId": result.conversation_id,
            "idempotencyKey": hashlib.sha256(key.encode()).hexdigest(),
            "messages": to_glen_messages(result.new_messages(), result.output),
            "additionalContext": context,
        }

    async def _send_turn(self, body: dict[str, Any]) -> None:
        # The idempotency key makes a retry safe: Glen drops a key it has already seen.
        for attempt in range(_ATTEMPTS):
            try:
                r = await self._http().post("/api/mcp/route/turn", json=body)
            except httpx.HTTPError as e:
                log.warning("glen record attempt %d failed: %s", attempt + 1, e)
            except Exception:
                log.exception("glen record failed")
                return
            else:
                if r.status_code == 200:
                    return
                if r.status_code not in _RETRYABLE:
                    log.warning("glen record rejected %s: %s", r.status_code, r.text[:500])
                    return
                log.warning("glen record attempt %d got %s", attempt + 1, r.status_code)
            if attempt < _ATTEMPTS - 1:
                await asyncio.sleep(self.retry_backoff * 2**attempt)

    async def flush(self) -> None:
        """Wait for in-flight recordings to finish."""
        if self._pending:
            await asyncio.gather(*self._pending, return_exceptions=True)

    async def aclose(self) -> None:
        """Flush recordings and close the HTTP client."""
        await self.flush()
        if self._client is not None:
            await self._client.aclose()
            self._client = None
