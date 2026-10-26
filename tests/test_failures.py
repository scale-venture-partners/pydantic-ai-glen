"""What the capability does when Glen misbehaves: errors are reported to the agent or logged, and
never fail the run."""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable
from typing import Any

import httpx
import pytest
from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from test_glen import (
    COMMIT,
    FakeGlen,
    commit_file,
    committing_agent,
    git_repo,
    make_glen,
    skill_call,
    text_model,
    tool_result,
)

from pydantic_ai_glen import Glen

Reply = httpx.Response | Exception | Callable[[httpx.Request], httpx.Response]


class Routes(FakeGlen):
    """FakeGlen with chosen paths overridden by a response, an exception or a function."""

    def __init__(self, overrides: dict[str, Reply], **kwargs: Any):
        super().__init__(**kwargs)
        self.overrides = overrides

    def __call__(self, request: httpx.Request) -> httpx.Response:
        reply = self.overrides.get(request.url.path)
        if reply is None:
            return super().__call__(request)
        self.requests.append(request)
        if isinstance(reply, Exception):
            raise reply
        return reply(request) if callable(reply) else reply


def bodies(fake: FakeGlen, path: str) -> list[dict[str, Any]]:
    return [json.loads(r.content) for r in fake.requests if r.url.path == path]


def tool_call(name: str, **args: Any) -> FunctionModel:
    return skill_call(name, **args)


async def run_tool(fake: FakeGlen, name: str, **args: Any) -> str:
    glen = make_glen(fake)
    result = await Agent(tool_call(name, **args), capabilities=[glen]).run("go")
    await glen.flush()
    return tool_result(result)


# --- search -----------------------------------------------------------------------------------


async def test_search_shows_what_could_not_be_verified_and_sends_the_conversation_id() -> None:
    fake = FakeGlen(answer="because Dana said so")
    glen = make_glen(fake)
    result = await Agent(tool_call("glen_search", query="why?"), capabilities=[glen]).run("go")

    assert tool_result(result) == "because Dana said so\n\nCould not verify:\n- no ticket found"
    [sent] = bodies(fake, "/api/mcp/route/search")
    assert sent == {"query": "why?", "conversationId": result.conversation_id}


async def test_search_without_gaps_has_no_suffix() -> None:
    fake = Routes(
        {"/api/mcp/route/search": httpx.Response(200, json={"answer": "plain", "gaps": []})}
    )

    assert await run_tool(fake, "glen_search", query="q") == "plain"


async def test_search_failure_status_is_reported_with_the_code(
    caplog: pytest.LogCaptureFixture,
) -> None:
    fake = Routes({"/api/mcp/route/search": httpx.Response(503, text="down")})

    with caplog.at_level(logging.WARNING, logger="glen"):
        assert await run_tool(fake, "glen_search", query="q") == "Glen search failed (503)."
    assert caplog.messages == ["glen search returned 503: down"]


async def test_search_transport_error_is_reported_and_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    fake = Routes({"/api/mcp/route/search": httpx.ConnectError("nope")})

    with caplog.at_level(logging.WARNING, logger="glen"):
        assert (
            await run_tool(fake, "glen_search", query="q")
            == "Glen search is unavailable right now."
        )
    assert caplog.messages == ["glen search failed: nope"]


# --- skills -----------------------------------------------------------------------------------


@pytest.mark.parametrize(("asked", "sent"), [(0, 1), (-5, 1), (7, 7), (30, 30), (500, 30)])
async def test_skill_search_limit_is_clamped(asked: int, sent: int) -> None:
    fake = FakeGlen()
    await run_tool(fake, "glen_skill_search", query="", limit=asked)

    assert bodies(fake, "/api/mcp/route/skillSearch") == [{"limit": sent}]


async def test_skill_search_query_is_trimmed_and_blank_means_no_query() -> None:
    fake = FakeGlen()
    await run_tool(fake, "glen_skill_search", query="  deploy  ")
    await run_tool(fake, "glen_skill_search", query="   ")

    assert bodies(fake, "/api/mcp/route/skillSearch") == [
        {"limit": 10, "query": "deploy"},
        {"limit": 10},
    ]


async def test_skill_search_with_no_results_says_so() -> None:
    assert await run_tool(FakeGlen(), "glen_skill_search") == "No team skills match."


async def test_skill_search_failures_are_reported(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="glen"):
        status = Routes({"/api/mcp/route/skillSearch": httpx.Response(500, text="oops")})
        assert await run_tool(status, "glen_skill_search") == "Glen skill search failed (500)."
        down = Routes({"/api/mcp/route/skillSearch": httpx.ReadTimeout("slow")})
        assert await run_tool(down, "glen_skill_search") == "Glen skills are unavailable right now."
    assert caplog.messages == [
        "glen skill search returned 500: oops",
        "glen skill search failed: slow",
    ]


async def test_skill_use_needs_a_slug_and_trims_it() -> None:
    fake = FakeGlen(skill={"name": "N", "content": "body"})

    blank = await run_tool(fake, "glen_skill_use", slug="   ")
    used = await run_tool(fake, "glen_skill_use", slug="  my-skill ")

    assert blank == "Give the slug of a skill (find one with glen_skill_search)."
    assert used.startswith('> Glen skill "N"')
    [sent] = bodies(fake, "/api/mcp/route/skillUse")
    assert sent["slug"] == "my-skill" and sent["conversationId"]


async def test_skill_use_failures_are_reported(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="glen"):
        status = Routes({"/api/mcp/route/skillUse": httpx.Response(500, text="oops")})
        assert await run_tool(status, "glen_skill_use", slug="s") == (
            "Glen skill lookup failed (500)."
        )
        down = Routes({"/api/mcp/route/skillUse": httpx.ReadTimeout("slow")})
        assert await run_tool(down, "glen_skill_use", slug="s") == (
            "Glen skills are unavailable right now."
        )
    assert caplog.messages == [
        "glen skill use returned 500: oops",
        "glen skill use failed: slow",
    ]


# --- code search ------------------------------------------------------------------------------


@pytest.fixture
def repo(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> Any:
    git_repo(tmp_path, remote="git@github.com:acme/widgets.git")
    commit_file(tmp_path, "a.py", "one\ntwo\n", "add a")
    monkeypatch.chdir(tmp_path)
    return tmp_path


async def test_code_search_sends_the_range_the_remote_and_the_branch(repo: Any) -> None:
    fake = FakeGlen()
    await run_tool(fake, "glen_code_search", path="a.py", start_line=1, end_line=2, question="why")
    await run_tool(fake, "glen_code_search", path="a.py", start_line=2, question="why")

    first, second = bodies(fake, "/api/mcp/route/codeSearch")
    assert first["repoUrl"] == second["repoUrl"] == "https://github.com/acme/widgets"
    assert first["query"] == "why" and first["branch"] == "main"
    assert len(first["shas"]) == 1 and first["trailerConversationIds"] == []


async def test_code_search_with_a_bad_range_is_explained(repo: Any) -> None:
    text = await run_tool(FakeGlen(), "glen_code_search", path="a.py", start_line=0, question="why")

    assert "line range is invalid" in text


@pytest.mark.parametrize("body", [{"state": "no_anchor", "markdown": "x"}, {"state": "ok"}])
async def test_code_search_without_a_trace_says_so(repo: Any, body: dict[str, Any]) -> None:
    fake = FakeGlen(code_search_json=body)

    text = await run_tool(fake, "glen_code_search", path="a.py", start_line=1, question="why")

    assert text.startswith("No Glen trace found for those lines")


async def test_code_search_failures_are_reported(
    repo: Any, caplog: pytest.LogCaptureFixture
) -> None:
    ask = {"path": "a.py", "start_line": 1, "question": "why"}
    with caplog.at_level(logging.WARNING, logger="glen"):
        status = Routes({"/api/mcp/route/codeSearch": httpx.Response(502, text="bad")})
        assert await run_tool(status, "glen_code_search", **ask) == (
            "Glen code search failed (502)."
        )
        down = Routes({"/api/mcp/route/codeSearch": httpx.ConnectError("nope")})
        assert await run_tool(down, "glen_code_search", **ask) == (
            "Glen code search is unavailable right now."
        )
    assert "glen code search returned 502: bad" in caplog.text
    assert "glen code search failed: nope" in caplog.text


async def test_code_search_is_off_when_the_flag_is_off(repo: Any) -> None:
    seen: list[str] = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.extend(t.name for t in info.function_tools)
        return ModelResponse(parts=[TextPart("ok")])

    await Agent(
        FunctionModel(respond), capabilities=[make_glen(FakeGlen(), code_search=False)]
    ).run("x")

    assert "glen_code_search" not in seen and "glen_search" in seen


async def test_code_search_called_directly_outside_a_repo_says_so(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    glen = make_glen(FakeGlen())

    assert await glen._code_search("a.py", 1, 1, "why") == (
        "Code search needs a git repository with a remote."
    )


# --- recording --------------------------------------------------------------------------------


async def test_the_idempotency_key_is_a_hash_of_conversation_and_run() -> None:
    fake = FakeGlen()
    glen = make_glen(fake)
    result = await Agent(text_model(), capabilities=[glen]).run("hi")
    await glen.flush()

    [sent] = bodies(fake, "/api/mcp/route/turn")
    expected = hashlib.sha256(f"{result.conversation_id}\0{result.run_id}".encode()).hexdigest()
    assert sent["idempotencyKey"] == expected
    assert sent["conversationId"] == result.conversation_id


async def test_the_agent_name_and_version_are_recorded() -> None:
    fake = FakeGlen()
    glen = make_glen(fake, agent_name="triage-bot", agent_version="1.2")
    await Agent(text_model(), capabilities=[glen]).run("hi")
    await glen.flush()
    bare = FakeGlen()
    glen2 = make_glen(bare)
    await Agent(text_model(), capabilities=[glen2]).run("hi")
    await glen2.flush()

    [sent] = bodies(fake, "/api/mcp/route/turn")
    [plain] = bodies(bare, "/api/mcp/route/turn")
    assert sent["additionalContext"]["agent"] == {"name": "triage-bot", "version": "1.2"}
    assert plain["additionalContext"]["agent"] == {"name": "pydantic-ai"}


async def test_every_retryable_status_is_retried_then_given_up_after_three_attempts() -> None:
    for status in (429, 500, 502, 503, 504):
        fake = FakeGlen(turn_statuses=[status, status, status, 200])
        glen = make_glen(fake)
        await Agent(text_model(), capabilities=[glen]).run("hi")
        await glen.flush()
        assert len(bodies(fake, "/api/mcp/route/turn")) == 3, status


async def test_retries_back_off_exponentially(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr("pydantic_ai_glen._capability.asyncio.sleep", fake_sleep)
    fake = FakeGlen(turn_statuses=[503, 503, 503])
    glen = Glen(api_key="k", org_id="o", transport=httpx.MockTransport(fake), retry_backoff=2.0)
    await Agent(text_model(), capabilities=[glen]).run("hi")
    await glen.flush()

    assert sleeps == [2.0, 4.0]


async def test_a_transport_error_is_retried_and_logged(caplog: pytest.LogCaptureFixture) -> None:
    calls = 0

    def flaky(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise httpx.ConnectError("reset")
        return httpx.Response(200, json={})

    fake = Routes({"/api/mcp/route/turn": flaky})
    glen = make_glen(fake)
    with caplog.at_level(logging.WARNING, logger="glen"):
        await Agent(text_model(), capabilities=[glen]).run("hi")
        await glen.flush()

    assert calls == 3
    assert "glen record attempt 1 failed: reset" in caplog.text


@pytest.mark.parametrize("status", [400, 404, 422])
async def test_other_rejections_are_logged_and_not_retried_or_announced(
    status: int, caplog: pytest.LogCaptureFixture
) -> None:
    notices: list[str] = []
    fake = FakeGlen(turn_statuses=[status])
    glen = make_glen(fake, on_notice=notices.append)
    with caplog.at_level(logging.WARNING, logger="glen"):
        await Agent(text_model(), capabilities=[glen]).run("hi")
        await glen.flush()

    assert len(bodies(fake, "/api/mcp/route/turn")) == 1
    assert notices == []
    assert f"glen record rejected {status}" in caplog.text


@pytest.mark.parametrize(
    ("status", "needle"),
    [(401, "API key"), (402, "paid plan"), (403, "does not allow"), (426, "too old")],
)
async def test_each_lasting_rejection_tells_the_user_what_to_do(status: int, needle: str) -> None:
    notices: list[str] = []
    fake = FakeGlen(turn_statuses=[status])
    glen = make_glen(fake, on_notice=notices.append)
    await Agent(text_model(), capabilities=[glen]).run("hi")
    await glen.flush()

    assert len(notices) == 1 and needle in notices[0]
    assert len(bodies(fake, "/api/mcp/route/turn")) == 1


async def test_a_failing_on_notice_callback_does_not_break_recording(
    caplog: pytest.LogCaptureFixture,
) -> None:
    def explode(message: str) -> None:
        raise RuntimeError("callback bug")

    fake = FakeGlen(turn_statuses=[401])
    glen = make_glen(fake, on_notice=explode)
    with caplog.at_level(logging.WARNING, logger="glen"):
        await Agent(text_model(), capabilities=[glen]).run("hi")
        await glen.flush()

    assert caplog.messages[1:] == [
        "glen: Glen rejected the API key or organization id, so runs are not being recorded.",
        "glen on_notice failed",
    ]


@pytest.mark.parametrize("body", ["not json", "[1, 2]", "null"])
async def test_a_turn_response_that_is_not_an_object_is_ignored(body: str) -> None:
    fake = Routes({"/api/mcp/route/turn": httpx.Response(200, text=body)})
    glen = make_glen(fake)
    await Agent(text_model(), capabilities=[glen]).run("hi")
    await glen.flush()
    await Agent(text_model(), capabilities=[glen]).run("again")
    await glen.flush()

    assert len(bodies(fake, "/api/mcp/route/turn")) == 2


async def test_a_pause_notice_needs_a_string_message() -> None:
    notices: list[str] = []
    for paused in ({"message": 5}, "paused", {"other": "x"}):
        fake = FakeGlen(turn_json={"recordingPaused": paused})
        glen = make_glen(fake, on_notice=notices.append)
        await Agent(text_model(), capabilities=[glen]).run("hi")
        await glen.flush()

    assert notices == []


async def test_a_mode_change_in_a_turn_response_is_ignored_when_modes_are_off() -> None:
    fake = FakeGlen(turn_json={"serverMode": "off"})
    glen = make_glen(fake, modes=False)
    agent = Agent(text_model(), capabilities=[glen])
    await agent.run("one")
    await glen.flush()
    await agent.run("two")
    await glen.flush()

    assert len(bodies(fake, "/api/mcp/route/turn")) == 2


async def test_a_turn_response_keeps_the_known_skill_flag_when_it_changes_the_mode() -> None:
    fake = FakeGlen(skill_use=False, turn_json={"serverMode": "on"})
    glen = make_glen(fake)
    await Agent(text_model(), capabilities=[glen]).run("hi")
    await glen.flush()

    assert (await glen.control()).skill_use is False


async def test_a_run_that_cannot_be_serialized_is_logged_and_does_not_fail(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def boom(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("bad payload")

    monkeypatch.setattr("pydantic_ai_glen._capability.to_glen_messages", boom)
    fake = FakeGlen()
    glen = make_glen(fake)
    with caplog.at_level(logging.ERROR, logger="glen"):
        result = await Agent(text_model(), capabilities=[glen]).run("hi")
        await glen.flush()

    assert result.output == "final answer"
    assert "glen could not build the turn payload" in caplog.text
    assert bodies(fake, "/api/mcp/route/turn") == []


async def test_an_unexpected_send_error_is_logged_and_not_retried(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    calls = 0

    def boom(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise RuntimeError("transport bug")

    fake = Routes({"/api/mcp/route/turn": boom})
    glen = make_glen(fake)
    with caplog.at_level(logging.ERROR, logger="glen"):
        await Agent(text_model(), capabilities=[glen]).run("hi")
        await glen.flush()

    assert calls == 1
    assert "glen record failed" in caplog.text


async def test_flush_and_aclose_wait_for_recording_and_close_the_client() -> None:
    gate_open = False

    def slow(request: httpx.Request) -> httpx.Response:
        nonlocal gate_open
        gate_open = True
        return httpx.Response(200, json={})

    fake = Routes({"/api/mcp/route/turn": slow})
    glen = make_glen(fake)
    await Agent(text_model(), capabilities=[glen]).run("hi")

    await glen.aclose()

    assert gate_open
    assert glen._client is None
    await glen.aclose()


async def test_flush_with_nothing_pending_returns() -> None:
    await make_glen(FakeGlen()).flush()


async def test_the_control_status_is_cached_and_a_failure_is_not_retried_at_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = 1000.0
    monkeypatch.setattr("pydantic_ai_glen._capability.time.monotonic", lambda: now)
    fake = FakeGlen(mode="off")
    glen = make_glen(fake)

    assert (await glen.control()).mode == "off"
    now += 59
    await glen.control()
    assert len(bodies(fake, "/api/mcp/route/controlStatus")) == 1
    now += 2
    await glen.control()
    assert len(bodies(fake, "/api/mcp/route/controlStatus")) == 2


async def test_a_failed_control_lookup_keeps_the_last_answer_for_the_failure_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = 1000.0
    monkeypatch.setattr("pydantic_ai_glen._capability.time.monotonic", lambda: now)
    state = {"fail": False}

    def status(request: httpx.Request) -> httpx.Response:
        if state["fail"]:
            raise httpx.ConnectError("down")
        return httpx.Response(200, json={"mode": "incognito"})

    fake = Routes({"/api/mcp/route/controlStatus": status})
    glen = make_glen(fake)
    assert (await glen.control()).mode == "incognito"
    state["fail"] = True
    now += 61
    assert (await glen.control()).mode == "incognito"
    now += 29
    await glen.control()
    assert len(bodies(fake, "/api/mcp/route/controlStatus")) == 2
    now += 2
    await glen.control()
    assert len(bodies(fake, "/api/mcp/route/controlStatus")) == 3


async def test_control_status_that_is_not_a_status_leaves_the_mode_on() -> None:
    for reply in (
        httpx.Response(200, text="<html>"),
        httpx.Response(200, json={"mode": "weird"}),
        httpx.Response(500, json={"mode": "off"}),
    ):
        glen = make_glen(Routes({"/api/mcp/route/controlStatus": reply}))
        assert (await glen.control()).mode == "on"


async def test_control_is_not_asked_when_neither_search_nor_record_is_used() -> None:
    fake = FakeGlen(mode="off")
    glen = make_glen(fake, search=False, record=False)

    assert (await glen.control()).mode == "on"
    assert fake.requests == []


async def test_control_is_asked_with_the_credentials_and_a_short_timeout() -> None:
    fake = FakeGlen()
    glen = make_glen(fake)
    await glen.control()

    [request] = fake.requests
    assert request.headers["x-api-key"] == "k"
    assert request.headers["x-glen-org-id"] == "org-1"
    assert request.extensions["timeout"]["read"] == 3.0


async def test_the_default_url_is_used_and_instructions_name_it() -> None:
    glen = Glen(api_key="k", org_id="o", transport=httpx.MockTransport(FakeGlen()))
    assert glen.url == "https://app.tryglen.com"
    custom = Glen(api_key="k", org_id="o", url="https://glen.internal")
    assert custom.url == "https://glen.internal"


# --- commit linking ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("reply", "needle"),
    [
        (httpx.Response(500, text="boom"), "glen commit link returned 500: boom"),
        (httpx.ConnectError("nope"), "glen commit link failed: nope"),
    ],
)
async def test_commit_link_failures_are_logged_and_the_run_still_succeeds(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    reply: Reply,
    needle: str,
) -> None:
    git_repo(tmp_path, remote="git@github.com:acme/widgets.git")
    monkeypatch.chdir(tmp_path)
    glen = make_glen(Routes({"/api/mcp/route/commitAnchor": reply}), background=False)
    with caplog.at_level(logging.WARNING, logger="glen"):
        result = await committing_agent(tmp_path, COMMIT, glen).run("go")

    assert result.output == "done"
    assert needle in caplog.text


@pytest.mark.parametrize(
    ("remote", "options"),
    [(None, {}), ("git@github.com:acme/widgets.git", {"link_commits": False})],
)
async def test_commits_are_not_linked_without_a_remote_or_with_the_flag_off(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
    remote: str | None,
    options: dict[str, Any],
) -> None:
    git_repo(tmp_path, remote=remote)
    monkeypatch.chdir(tmp_path)
    fake = FakeGlen()
    glen = make_glen(fake, background=False, **options)

    await committing_agent(tmp_path, COMMIT, glen).run("go")

    assert bodies(fake, "/api/mcp/route/turn")
    assert bodies(fake, "/api/mcp/route/commitAnchor") == []


async def test_an_unexpected_error_while_linking_commits_is_logged(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    git_repo(tmp_path, remote="git@github.com:acme/widgets.git")
    monkeypatch.chdir(tmp_path)

    async def boom(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("git exploded")

    monkeypatch.setattr("pydantic_ai_glen._capability.resolve_shas", boom)
    glen = make_glen(FakeGlen(), background=False)
    with caplog.at_level(logging.ERROR, logger="glen"):
        result = await committing_agent(tmp_path, COMMIT, glen).run("go")

    assert result.output == "done"
    assert "glen could not link the run's commits" in caplog.text


async def test_a_failed_run_notes_its_error_type_and_truncates_a_long_message() -> None:
    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart("boom", {})])
        return ModelResponse(parts=[TextPart("unreachable")])

    fake = FakeGlen()
    glen = make_glen(fake)
    agent: Agent[None, str] = Agent(FunctionModel(respond), capabilities=[glen])

    @agent.tool_plain
    def boom() -> str:
        raise ValueError("x" * 1000)

    with pytest.raises(ValueError):
        await agent.run("go")
    await glen.flush()

    [sent] = bodies(fake, "/api/mcp/route/turn")
    note = sent["messages"][-1]["content"]
    assert note == f"[The run failed: ValueError: {'x' * 300}]"


async def test_search_with_an_empty_answer_says_memory_may_be_off() -> None:
    fake = Routes({"/api/mcp/route/search": httpx.Response(200, json={"answer": ""})})

    assert await run_tool(fake, "glen_search", query="q") == (
        "Glen has no memory to share for this (it may be turned off)."
    )


async def test_search_gaps_are_listed_one_per_line() -> None:
    reply = httpx.Response(200, json={"answer": "a", "gaps": ["one", "two"]})
    fake = Routes({"/api/mcp/route/search": reply})

    assert await run_tool(fake, "glen_search", query="q") == "a\n\nCould not verify:\n- one\n- two"


async def test_error_bodies_are_logged_cut_to_500_characters(
    caplog: pytest.LogCaptureFixture,
) -> None:
    long = "y" * 600
    with caplog.at_level(logging.WARNING, logger="glen"):
        for path, tool, args in (
            ("search", "glen_search", {"query": "q"}),
            ("skillSearch", "glen_skill_search", {}),
            ("skillUse", "glen_skill_use", {"slug": "s"}),
        ):
            fake = Routes({f"/api/mcp/route/{path}": httpx.Response(500, text=long)})
            await run_tool(fake, tool, **args)

    assert [len(m.split(": ", 1)[1]) for m in caplog.messages] == [500, 500, 500]


async def test_skill_use_allows_a_minute_and_search_ten_seconds_to_connect() -> None:
    fake = FakeGlen(skill={"name": "N", "content": "c"})
    await run_tool(fake, "glen_skill_use", slug="s")

    [request] = [r for r in fake.requests if r.url.path == "/api/mcp/route/skillUse"]
    assert request.extensions["timeout"]["read"] == 60.0


async def test_a_skill_without_files_leaves_the_directory_placeholder_alone() -> None:
    fake = FakeGlen(skill={"name": "N", "content": "run ${GLEN_SKILL_DIR}/x", "files": []})

    assert (await run_tool(fake, "glen_skill_use", slug="s")).endswith("run ${GLEN_SKILL_DIR}/x")


class FailsFirst(AbstractCapability[None]):
    async def before_model_request(self, ctx: RunContext[None], request_context: Any) -> Any:
        raise RuntimeError("stopped before the model")


async def test_a_run_that_fails_before_any_model_request_records_nothing() -> None:
    fake = FakeGlen()
    glen = make_glen(fake)
    agent: Agent[None, str] = Agent(text_model(), capabilities=[FailsFirst(), glen])

    with pytest.raises(RuntimeError, match="stopped before the model"):
        await agent.run("hi")
    await glen.flush()

    assert bodies(fake, "/api/mcp/route/turn") == []


async def test_a_failed_run_is_keyed_by_its_own_run_id() -> None:
    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart("boom", {})])
        return ModelResponse(parts=[TextPart("unreachable")])

    fake = FakeGlen()
    glen = make_glen(fake)
    agent: Agent[None, str] = Agent(FunctionModel(respond), capabilities=[glen])
    run_ids: list[str] = []

    @agent.tool
    def boom(ctx: RunContext[None]) -> str:
        run_ids.append(ctx.run_id)
        raise ValueError("x")

    with pytest.raises(ValueError):
        await agent.run("go")
    await glen.flush()

    [sent] = bodies(fake, "/api/mcp/route/turn")
    key = hashlib.sha256(f"{sent['conversationId']}\0{run_ids[0]}".encode()).hexdigest()
    assert sent["idempotencyKey"] == key
