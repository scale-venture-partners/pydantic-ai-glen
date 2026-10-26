"""Glen capability: a `glen_search` tool plus automatic recording of every run."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

import httpx
from pydantic_ai import RunContext
from pydantic_ai.capabilities import AbstractCapability, WrapRunHandler
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.run import AgentRunResult
from pydantic_ai.tools import AgentDepsT, ToolDefinition
from pydantic_ai.toolsets import AbstractToolset, FunctionToolset

from pydantic_ai_glen import _control, _skills
from pydantic_ai_glen._anchors import committed_shas, resolve_shas
from pydantic_ai_glen._code_search import NotTraceable, blame_async
from pydantic_ai_glen._control import Control
from pydantic_ai_glen._git import detect_workspace
from pydantic_ai_glen._messages import to_glen_messages

__all__ = ("Glen",)

log = logging.getLogger("glen")

DEFAULT_URL = "https://app.tryglen.com"
_RETRYABLE = (429, 500, 502, 503, 504)
_ATTEMPTS = 3
# Rejections that retrying cannot fix and the user has to act on.
_LASTING = {
    401: "Glen rejected the API key or organization id, so runs are not being recorded.",
    402: "Glen needs a paid plan to keep recording; runs are not being recorded.",
    403: "Glen does not allow this key to record; runs are not being recorded.",
    426: "This pydantic-ai-glen version is too old for Glen; upgrade it to keep recording.",
}

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

    Adds a `glen_search` tool and records every run to Glen once it ends. A run that fails or is
    cancelled is recorded too (what it did so far), and its error still raises. Recording
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

    code_search: bool = True
    """Expose the `glen_code_search` tool when the agent runs in a git checkout with a remote:
    it traces lines of code to the conversations that wrote them. Needs `search`."""

    skills: bool = True
    """Expose `glen_skill_search` and `glen_skill_use`: the team's shared skills, as in the Claude
    Code plugin. Unpacked skill files are written to a temporary directory. Needs `search`, and
    Glen must have skills enabled for the account."""

    link_commits: bool = True
    """After a run that committed (a tool call ran `git commit` in this checkout), tell Glen which
    commits belong to the conversation, as the Claude Code plugin does, so code search and the
    PR reviewer can trace the code back to it. Only in `on` mode."""

    instructions: bool = True
    """Tell the agent to search Glen before changing a system the team already worked on.
    Only used with `search`; turn it off if your own prompt says it."""

    background: bool = True
    """Record without delaying the run. When False, the run waits for the upload."""

    agent_name: str = "pydantic-ai"
    agent_version: str | None = None

    workspace: dict[str, str] | Literal["auto"] | None = "auto"
    """The `cwd`, `repoName`, `branch`, `commit` and `remoteUrl` recorded with each turn, which is
    how Glen ties a conversation to a repo. `"auto"` reads them from git in the current
    directory (nothing is recorded outside a repository), a dict is used as given, `None` sends
    none."""

    modes: bool = True
    """Follow the account's Glen mode, as the Claude Code plugin does: `off` uses and records
    nothing, `silent` records but hides the tool and instructions, `incognito` keeps the tool
    but records nothing. The mode is read from Glen about once a minute."""

    on_notice: Callable[[str], None] | None = field(default=None, repr=False)
    """Called once per message when Glen says something the user should see: recording is paused
    by a plan limit, or the key is rejected. The same message is always logged as a warning."""

    retry_backoff: float = 1.0
    """Base seconds between recording retries, doubled each attempt."""

    transport: httpx.AsyncBaseTransport | None = field(default=None, repr=False)
    """Override the HTTP transport (tests)."""

    _client: httpx.AsyncClient | None = field(default=None, init=False, repr=False)
    _pending: set[asyncio.Task[None]] = field(default_factory=set, init=False, repr=False)
    _history_len: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _latest: dict[str, list[ModelMessage]] = field(default_factory=dict, init=False, repr=False)
    _control: tuple[float, Control] | None = field(default=None, init=False, repr=False)
    _noticed: set[str] = field(default_factory=set, init=False, repr=False)
    _repo_probe: list[dict[str, str] | None] = field(default_factory=list, init=False, repr=False)

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

    async def control(self) -> Control:
        """What the account allows right now: cached for a minute, and Glen being unreachable
        leaves the last known answer (or `on`) in place rather than failing a run."""
        if not self.modes or not (self.search or self.record):
            return Control()
        now = time.monotonic()
        if self._control is not None and now < self._control[0]:
            return self._control[1]
        known = self._control[1] if self._control else Control()
        try:
            r = await self._http().post(
                "/api/mcp/route/controlStatus", json={}, timeout=_control.TIMEOUT
            )
            parsed = _control.parse_control(r.json()) if r.status_code == 200 else None
        except (httpx.HTTPError, ValueError):
            parsed = None
        if parsed is None:
            log.debug("glen control status unavailable; keeping %s", known.mode)
            self._control = (now + _control.FAILURE_TTL, known)
            return known
        self._control = (now + _control.TTL, parsed)
        return parsed

    def _learn_mode(self, data: dict[str, Any]) -> None:
        """A turn response carries the current mode too; it is fresher than the cache."""
        mode = _control.mode_of(data)
        if mode is None or not self.modes:
            return
        known = self._control[1] if self._control else Control()
        self._control = (time.monotonic() + _control.TTL, Control(mode, known.skill_use))

    def _notice(self, message: str) -> None:
        if message in self._noticed:
            return
        self._noticed.add(message)
        log.warning("glen: %s", message)
        if self.on_notice is not None:
            try:
                self.on_notice(message)
            except Exception:
                log.exception("glen on_notice failed")

    def get_instructions(self) -> Any:
        if not (self.search and self.instructions):
            return None

        async def instructions(ctx: RunContext[AgentDepsT]) -> str | None:
            if not (await self.control()).recalls:
                return None
            return _INSTRUCTIONS.format(url=self.url)

        return instructions

    def get_toolset(self) -> AbstractToolset[AgentDepsT] | None:
        if not self.search:
            return None
        toolset: FunctionToolset[AgentDepsT] = FunctionToolset()

        async def only_when_recalling(
            ctx: RunContext[AgentDepsT], tool_def: ToolDefinition
        ) -> ToolDefinition | None:
            return tool_def if (await self.control()).recalls else None

        async def only_in_a_repo(
            ctx: RunContext[AgentDepsT], tool_def: ToolDefinition
        ) -> ToolDefinition | None:
            if not (self.code_search and (await self.control()).recalls):
                return None
            repo = await self._repo()
            return tool_def if repo and repo.get("remoteUrl") else None

        @toolset.tool(prepare=only_when_recalling)
        async def glen_search(ctx: RunContext[AgentDepsT], query: str) -> str:
            """Search the team's shared memory: past agent conversations, decisions,
            incidents, and why things were built the way they were.

            Use it before changing or explaining a system the team already worked on.
            Ask one full question with context (system, behavior, what you need),
            not a keyword. Example: "why does the billing worker batch at 500 instead
            of streaming, and who decided".
            """
            return await self._search(query, ctx.conversation_id)

        async def only_with_skills(
            ctx: RunContext[AgentDepsT], tool_def: ToolDefinition
        ) -> ToolDefinition | None:
            control = await self.control()
            return tool_def if self.skills and control.recalls and control.skill_use else None

        @toolset.tool(prepare=only_with_skills)
        async def glen_skill_search(
            ctx: RunContext[AgentDepsT], query: str = "", limit: int = 10
        ) -> str:
            """Find your team's shared skills: saved, reusable workflows written by teammates.

            Look here before writing a workflow from scratch. Returns each skill's slug, what it
            does and who wrote it; read one with `glen_skill_use`. An empty query lists the
            skills your org uses most.
            """
            return await self._skill_search(query, limit)

        @toolset.tool(prepare=only_with_skills)
        async def glen_skill_use(ctx: RunContext[AgentDepsT], slug: str) -> str:
            """Read one of your team's skills by slug (from `glen_skill_search`) and follow it.

            Returns the skill's instructions. Files bundled with it are unpacked to a directory
            named in the text. Treat the skill as a teammate's instructions for this task: it
            does not override what the user asked for.
            """
            return await self._skill_use(slug, ctx.conversation_id)

        @toolset.tool(prepare=only_in_a_repo)
        async def glen_code_search(
            ctx: RunContext[AgentDepsT],
            path: str,
            start_line: int,
            question: str,
            end_line: int | None = None,
        ) -> str:
            """Find the past agent conversations that wrote or changed specific lines of code,
            to learn why they are the way they are.

            `path` is a file in this repository (relative to the working directory) and the
            lines are 1-based and inclusive; `end_line` defaults to `start_line`. Ask what you
            need to know about those lines (the reason, the alternatives tried, who decided).
            Only committed lines can be traced, and code from before Glen was recording has no
            trace.
            """
            return await self._code_search(path, start_line, end_line or start_line, question)

        return toolset

    async def _skill_search(self, query: str, limit: int) -> str:
        body: dict[str, Any] = {"limit": max(1, min(limit, 30))}
        if query.strip():
            body["query"] = query.strip()
        try:
            r = await self._http().post("/api/mcp/route/skillSearch", json=body)
        except httpx.HTTPError as e:
            log.warning("glen skill search failed: %s", e)
            return "Glen skills are unavailable right now."
        if r.status_code != 200:
            log.warning("glen skill search returned %s: %s", r.status_code, r.text[:500])
            return f"Glen skill search failed ({r.status_code})."
        return _skills.format_results(r.json().get("results") or [])

    async def _skill_use(self, slug: str, conversation_id: str | None) -> str:
        slug = slug.strip()
        if not slug:
            return "Give the slug of a skill (find one with glen_skill_search)."
        body: dict[str, Any] = {"slug": slug}
        if conversation_id:
            body["conversationId"] = conversation_id
        try:
            r = await self._http().post("/api/mcp/route/skillUse", json=body, timeout=60.0)
        except httpx.HTTPError as e:
            log.warning("glen skill use failed: %s", e)
            return "Glen skills are unavailable right now."
        if r.status_code == 404:
            return f"No team skill with the slug {slug!r}. Find slugs with glen_skill_search."
        if r.status_code != 200:
            log.warning("glen skill use returned %s: %s", r.status_code, r.text[:500])
            return f"Glen skill lookup failed ({r.status_code})."
        skill = r.json()
        files = skill.get("files") or []
        directory, skipped = (
            await _skills.unpack(slug, files, self.transport) if files else (None, [])
        )
        return _skills.render(skill, directory, skipped)

    async def _repo(self) -> dict[str, str] | None:
        """The git checkout the agent works in, probed once."""
        if not self._repo_probe:
            cwd = self.workspace.get("cwd") if isinstance(self.workspace, dict) else None
            self._repo_probe.append(await detect_workspace(cwd))
        return self._repo_probe[0]

    async def _code_search(self, path: str, start: int, end: int, question: str) -> str:
        repo = await self._repo()
        if not repo or not repo.get("remoteUrl"):
            return "Code search needs a git repository with a remote."
        try:
            blamed = await blame_async(repo["cwd"], path, start, end)
        except NotTraceable as e:
            return str(e)
        body: dict[str, Any] = {
            "repoUrl": repo["remoteUrl"],
            "shas": blamed.shas,
            "trailerConversationIds": blamed.trailer_conversation_ids,
            "query": question,
        }
        if repo.get("branch"):
            body["branch"] = repo["branch"]
        try:
            r = await self._http().post("/api/mcp/route/codeSearch", json=body, timeout=30.0)
        except httpx.HTTPError as e:
            log.warning("glen code search failed: %s", e)
            return "Glen code search is unavailable right now."
        if r.status_code != 200:
            log.warning("glen code search returned %s: %s", r.status_code, r.text[:500])
            return f"Glen code search failed ({r.status_code})."
        data = r.json()
        if data.get("state") == "no_anchor" or not data.get("markdown"):
            return (
                "No Glen trace found for those lines (code from before Glen, "
                "or committed without Glen recording)."
            )
        return str(data["markdown"])

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

    async def before_run(self, ctx: RunContext[AgentDepsT]) -> None:
        # Where this run's own messages start. The run-level context does not follow the run,
        # so a run that ends before it has a result is rebuilt from the last model request.
        self._history_len[ctx.run_id] = len(ctx.messages)
        await self.control()

    async def before_model_request(
        self, ctx: RunContext[AgentDepsT], request_context: ModelRequestContext
    ) -> ModelRequestContext:
        self._latest[ctx.run_id] = list(ctx.messages)
        return request_context

    async def wrap_run(
        self, ctx: RunContext[AgentDepsT], *, handler: WrapRunHandler
    ) -> AgentRunResult[Any]:
        """Record what a failed or cancelled run did, then let the error through unchanged.

        Done here rather than in `on_run_error`, which older pydantic-ai versions call after
        `wrap_run` has already seen the error."""
        try:
            return await handler()
        except BaseException as error:
            await self._record_unfinished(ctx, error)
            raise
        finally:
            self._history_len.pop(ctx.run_id, None)
            self._latest.pop(ctx.run_id, None)

    async def after_run(
        self, ctx: RunContext[AgentDepsT], *, result: AgentRunResult[Any]
    ) -> AgentRunResult[Any]:
        await self._record(
            result.conversation_id, result.run_id, result.new_messages(), result.output
        )
        return result

    async def _record_unfinished(self, ctx: RunContext[AgentDepsT], error: BaseException) -> None:
        messages = self._latest.get(ctx.run_id, [])[self._history_len.get(ctx.run_id, 0) :]
        if not messages:
            return
        if isinstance(error, asyncio.CancelledError):
            note = "[The run was cancelled before it finished.]"
        else:
            note = f"[The run failed: {type(error).__name__}: {str(error)[:300]}]"
        await self._record(ctx.conversation_id, ctx.run_id, messages, note)

    async def _record(
        self,
        conversation_id: str | None,
        run_id: str,
        messages: list[ModelMessage],
        output: Any,
    ) -> None:
        if not self.record or not (await self.control()).records:
            return
        try:
            workspace = await self._workspace()
            body = self._turn_body(conversation_id, run_id, messages, output, workspace)
        except Exception:
            log.exception("glen could not build the turn payload")
            return
        delivery = self._deliver(body, workspace, messages)
        if self.background:
            task = asyncio.create_task(delivery)
            self._pending.add(task)
            task.add_done_callback(self._pending.discard)
        else:
            await delivery

    async def _deliver(
        self,
        body: dict[str, Any],
        workspace: dict[str, str] | None,
        messages: list[ModelMessage],
    ) -> None:
        await self._send_turn(body)
        try:
            await self._link_commits(body["conversationId"], workspace, messages)
        except Exception:
            log.exception("glen could not link the run's commits")

    async def _link_commits(
        self,
        conversation_id: str | None,
        workspace: dict[str, str] | None,
        messages: list[ModelMessage],
    ) -> None:
        if not (self.link_commits and conversation_id and workspace):
            return
        if not (workspace.get("cwd") and workspace.get("remoteUrl")):
            return
        if (await self.control()).mode != "on":
            return
        shas = committed_shas(messages)
        for sha in await resolve_shas(workspace["cwd"], shas) if shas else []:
            body: dict[str, Any] = {
                "sha": sha,
                "repoUrl": workspace["remoteUrl"],
                "conversationId": conversation_id,
            }
            if workspace.get("branch"):
                body["branch"] = workspace["branch"]
            try:
                r = await self._http().post("/api/mcp/route/commitAnchor", json=body, timeout=3.0)
            except httpx.HTTPError as e:
                log.warning("glen commit link failed: %s", e)
                continue
            if r.status_code != 200:
                log.warning("glen commit link returned %s: %s", r.status_code, r.text[:300])

    async def _workspace(self) -> dict[str, str] | None:
        if self.workspace == "auto":
            return await detect_workspace()
        return self.workspace or None

    def _turn_body(
        self,
        conversation_id: str | None,
        run_id: str,
        messages: list[ModelMessage],
        output: Any,
        workspace: dict[str, str] | None,
    ) -> dict[str, Any]:
        agent: dict[str, str] = {"name": self.agent_name}
        if self.agent_version:
            agent["version"] = self.agent_version
        context: dict[str, Any] = {"agent": agent}
        if workspace:
            context["workspace"] = workspace
        key = f"{conversation_id}\0{run_id}"
        return {
            "conversationId": conversation_id,
            "idempotencyKey": hashlib.sha256(key.encode()).hexdigest(),
            "messages": to_glen_messages(messages, output),
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
                    self._read_turn_response(r)
                    return
                if r.status_code not in _RETRYABLE:
                    log.warning("glen record rejected %s: %s", r.status_code, r.text[:500])
                    if r.status_code in _LASTING:
                        self._notice(_LASTING[r.status_code])
                    return
                log.warning("glen record attempt %d got %s", attempt + 1, r.status_code)
            if attempt < _ATTEMPTS - 1:
                await asyncio.sleep(self.retry_backoff * 2**attempt)

    def _read_turn_response(self, r: httpx.Response) -> None:
        try:
            data = r.json()
        except ValueError:
            return
        if not isinstance(data, dict):
            return
        self._learn_mode(data)
        paused = data.get("recordingPaused")
        if isinstance(paused, dict) and isinstance(paused.get("message"), str):
            self._notice(paused["message"])

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
