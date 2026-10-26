from __future__ import annotations

import asyncio
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic_ai import Agent
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from pydantic_ai_glen import Glen


class FakeGlen:
    """Stands in for Glen's HTTP API at the network boundary; everything else runs for real."""

    def __init__(
        self,
        turn_statuses: list[int] | None = None,
        answer: str = "because Dana said so",
        mode: str = "on",
        control_status: int = 200,
        turn_json: dict[str, Any] | None = None,
        code_search_json: dict[str, Any] | None = None,
        skill_use: bool = True,
        skills: list[dict[str, Any]] | None = None,
        skill: dict[str, Any] | None = None,
        skill_status: int = 200,
    ):
        self.requests: list[httpx.Request] = []
        self.turn_statuses = turn_statuses or []
        self.answer = answer
        self.mode = mode
        self.control_status = control_status
        self.turn_json = turn_json or {}
        self.skill_use = skill_use
        self.skills = skills or []
        self.skill = skill or {}
        self.skill_status = skill_status
        self.code_search_json = code_search_json or {
            "state": "anchor_found",
            "markdown": "# Why this code is the way it is\n\nDana asked for it.",
            "conversationCount": 1,
        }

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.url.path == "/api/mcp/route/search":
            return httpx.Response(
                200, json={"answer": self.answer, "gaps": ["no ticket found"], "citations": []}
            )
        if request.url.host == "files.example":
            return httpx.Response(200, content=b"#!/bin/sh\necho from-url\n")
        if request.url.path == "/api/mcp/route/skillSearch":
            return httpx.Response(200, json={"results": self.skills})
        if request.url.path == "/api/mcp/route/skillUse":
            return httpx.Response(self.skill_status, json=self.skill)
        if request.url.path == "/api/mcp/route/commitAnchor":
            return httpx.Response(200, json={"id": "anchor-1"})
        if request.url.path == "/api/mcp/route/codeSearch":
            return httpx.Response(200, json=self.code_search_json)
        if request.url.path == "/api/mcp/route/controlStatus":
            return httpx.Response(
                self.control_status,
                json={
                    "mode": self.mode,
                    "skillCreation": True,
                    "skillUse": self.skill_use,
                    "updatedAt": None,
                    "teamMemoryBlock": None,
                },
            )
        status = self.turn_statuses.pop(0) if self.turn_statuses else 200
        return httpx.Response(status, json=self.turn_json)

    def bodies(self, path: str) -> list[dict[str, Any]]:
        return [json.loads(r.content) for r in self.requests if r.url.path == path]


def make_glen(fake: FakeGlen, **kwargs: Any) -> Glen:
    return Glen(
        api_key="k",
        org_id="org-1",
        transport=httpx.MockTransport(fake),
        retry_backoff=0,
        **kwargs,
    )


def text_model(text: str = "final answer") -> FunctionModel:
    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart(text)])

    return FunctionModel(respond)


def search_then_answer() -> FunctionModel:
    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(
                parts=[ToolCallPart("glen_search", {"query": "why batch at 500?"})]
            )
        return ModelResponse(parts=[TextPart("answered")])

    return FunctionModel(respond)


def test_requires_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GLEN_API_KEY", raising=False)
    monkeypatch.delenv("GLEN_ORG_ID", raising=False)
    with pytest.raises(UserError):
        Glen()


def test_reads_credentials_from_env_and_hides_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GLEN_API_KEY", "secret-key")
    monkeypatch.setenv("GLEN_ORG_ID", "org-env")
    glen = Glen()
    assert glen.org_id == "org-env"
    assert glen.url == "https://app.tryglen.com"
    assert "secret-key" not in repr(glen)


async def test_run_is_recorded_with_headers_and_stable_ids() -> None:
    fake = FakeGlen()
    glen = make_glen(fake, agent_version="9.9", workspace={"repoName": "r"})
    agent = Agent(text_model(), capabilities=[glen])

    result = await agent.run("how do I deploy?")
    await glen.flush()

    [req] = [r for r in fake.requests if r.url.path == "/api/mcp/route/turn"]
    assert req.headers["x-api-key"] == "k"
    assert req.headers["x-glen-org-id"] == "org-1"
    [body] = fake.bodies("/api/mcp/route/turn")
    assert body["conversationId"] == result.conversation_id
    assert [m["role"] for m in body["messages"]] == ["user", "assistant"]
    assert body["messages"][0]["content"] == "how do I deploy?"
    assert body["messages"][1]["content"] == "final answer"
    assert body["additionalContext"] == {
        "agent": {"name": "pydantic-ai", "version": "9.9"},
        "workspace": {"repoName": "r"},
    }
    assert len(body["idempotencyKey"]) == 64


async def test_tool_steps_become_segments_and_search_hits_glen() -> None:
    fake = FakeGlen()
    glen = make_glen(fake)
    agent = Agent(search_then_answer(), capabilities=[glen])

    await agent.run("explain the worker")
    await glen.flush()

    [search] = fake.bodies("/api/mcp/route/search")
    assert search["query"] == "why batch at 500?"
    assert "conversationId" in search
    [turn] = fake.bodies("/api/mcp/route/turn")
    kinds = [s["kind"] for s in turn["messages"][-1]["segments"]]
    assert kinds == ["tool_call", "tool_result", "text"]
    tool_result = turn["messages"][-1]["segments"][1]["text"]
    assert "because Dana said so" in tool_result
    assert "no ticket found" in tool_result


async def test_each_run_gets_its_own_idempotency_key() -> None:
    fake = FakeGlen()
    glen = make_glen(fake)
    agent = Agent(text_model(), capabilities=[glen])

    await agent.run("one")
    await agent.run("two")
    await glen.flush()

    keys = {b["idempotencyKey"] for b in fake.bodies("/api/mcp/route/turn")}
    assert len(keys) == 2


async def test_retries_transient_failures_with_the_same_key() -> None:
    fake = FakeGlen(turn_statuses=[503, 429])
    glen = make_glen(fake)
    agent = Agent(text_model(), capabilities=[glen])

    await agent.run("hi")
    await glen.flush()

    turns = fake.bodies("/api/mcp/route/turn")
    assert len(turns) == 3
    assert len({t["idempotencyKey"] for t in turns}) == 1


async def test_rejected_turn_is_not_retried_and_does_not_fail_the_run(
    caplog: pytest.LogCaptureFixture,
) -> None:
    fake = FakeGlen(turn_statuses=[401])
    glen = make_glen(fake)
    agent = Agent(text_model(), capabilities=[glen])

    result = await agent.run("hi")
    await glen.flush()

    assert result.output == "final answer"
    assert len(fake.bodies("/api/mcp/route/turn")) == 1
    assert "glen record rejected 401" in caplog.text


async def test_foreground_recording_completes_before_run_returns() -> None:
    fake = FakeGlen()
    glen = make_glen(fake, background=False)
    agent = Agent(text_model(), capabilities=[glen])

    await agent.run("hi")

    assert len(fake.bodies("/api/mcp/route/turn")) == 1


def test_run_sync_records_in_foreground() -> None:
    fake = FakeGlen()
    glen = make_glen(fake, background=False)
    agent = Agent(text_model(), capabilities=[glen])

    agent.run_sync("hi")

    assert len(fake.bodies("/api/mcp/route/turn")) == 1


async def test_run_stream_is_recorded() -> None:
    async def stream(messages: list[ModelMessage], info: AgentInfo):  # type: ignore[no-untyped-def]
        yield "streamed"

    fake = FakeGlen()
    glen = make_glen(fake)
    agent = Agent(FunctionModel(stream_function=stream), capabilities=[glen])

    async with agent.run_stream("hi") as streamed:
        async for _ in streamed.stream_text():
            pass
    await glen.flush()

    [turn] = fake.bodies("/api/mcp/route/turn")
    assert turn["messages"][-1]["content"] == "streamed"


async def test_record_false_sends_nothing_and_search_false_hides_tool() -> None:
    fake = FakeGlen()
    glen = make_glen(fake, record=False, search=False)
    agent = Agent(text_model(), capabilities=[glen])

    await agent.run("hi")
    await glen.flush()

    assert fake.requests == []
    assert glen.get_toolset() is None


async def test_search_reports_http_errors_instead_of_raising() -> None:
    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down")

    glen = Glen(api_key="k", org_id="o", transport=httpx.MockTransport(boom))
    assert "unavailable" in await glen._search("q", None)


async def test_search_with_empty_answer_means_memory_is_off() -> None:
    glen = make_glen(FakeGlen(answer=""))
    assert "no memory" in await glen._search("q", None)


async def test_structured_output_and_multimodal_prompt_are_serialized() -> None:
    from pydantic import BaseModel
    from pydantic_ai import BinaryContent

    class Out(BaseModel):
        x: int

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert info.output_tools
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, {"x": 3})])

    fake = FakeGlen()
    glen = make_glen(fake)
    agent = Agent(FunctionModel(respond), output_type=Out, capabilities=[glen])

    await agent.run(["look at this", BinaryContent(b"\x89PNG", media_type="image/png")])
    await glen.flush()

    [turn] = fake.bodies("/api/mcp/route/turn")
    assert turn["messages"][0]["content"] == "look at this"
    assert json.loads(turn["messages"][-1]["content"]) == {"x": 3}


def capturing_model(seen: list[str | None]) -> FunctionModel:
    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append(info.instructions)
        return ModelResponse(parts=[TextPart("ok")])

    return FunctionModel(respond)


async def test_the_agent_is_told_to_search_glen_before_changing_a_system() -> None:
    seen: list[str | None] = []
    glen = make_glen(FakeGlen(), url="https://glen.example")
    await Agent(capturing_model(seen), capabilities=[glen]).run("fix billing")
    assert seen[0] is not None
    assert "glen_search" in seen[0]
    assert "https://glen.example/<owner>/<repo>/pull/<number>" in seen[0]


async def test_instructions_follow_the_search_tool_and_can_be_turned_off() -> None:
    for kwargs in ({"instructions": False}, {"search": False}):
        seen: list[str | None] = []
        glen = make_glen(FakeGlen(), **kwargs)
        await Agent(capturing_model(seen), capabilities=[glen]).run("fix billing")
        assert not seen[0] or "glen_search" not in seen[0]


def tool_then_fail() -> FunctionModel:
    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart("glen_search", {"query": "q"})])
        raise RuntimeError("model fell over")

    return FunctionModel(respond)


async def test_a_failed_run_is_recorded_and_the_error_still_raises() -> None:
    fake = FakeGlen()
    glen = make_glen(fake)
    agent = Agent(tool_then_fail(), capabilities=[glen])

    with pytest.raises(RuntimeError, match="model fell over"):
        await agent.run("investigate billing")
    await glen.flush()

    [body] = fake.bodies("/api/mcp/route/turn")
    user, assistant = body["messages"]
    assert user["content"] == "investigate billing"
    assert "RuntimeError: model fell over" in assistant["content"]
    assert [s["kind"] for s in assistant["segments"]] == ["tool_call", "tool_result"]


async def test_a_failed_run_records_only_its_own_messages() -> None:
    fake = FakeGlen()
    glen = make_glen(fake)
    first = Agent(text_model("earlier answer"), capabilities=[glen])
    history = (await first.run("an earlier question")).all_messages()
    await glen.flush()
    fake.requests.clear()

    with pytest.raises(RuntimeError):
        await Agent(tool_then_fail(), capabilities=[glen]).run(
            "the new question", message_history=history
        )
    await glen.flush()

    [body] = fake.bodies("/api/mcp/route/turn")
    assert [m["content"] for m in body["messages"] if m["role"] == "user"] == ["the new question"]


async def test_a_cancelled_run_is_recorded_and_stays_cancelled() -> None:
    fake = FakeGlen()
    glen = make_glen(fake)
    started = asyncio.Event()

    async def hang(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        started.set()
        await asyncio.sleep(60)
        raise AssertionError("not reached")

    task = asyncio.create_task(Agent(FunctionModel(hang), capabilities=[glen]).run("a long job"))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await glen.flush()

    [body] = fake.bodies("/api/mcp/route/turn")
    assert body["messages"][0]["content"] == "a long job"
    assert "cancelled" in body["messages"][-1]["content"]


async def test_a_failed_run_is_not_recorded_when_record_is_off() -> None:
    fake = FakeGlen()
    glen = make_glen(fake, record=False)
    with pytest.raises(RuntimeError):
        await Agent(tool_then_fail(), capabilities=[glen]).run("x")
    await glen.flush()
    assert fake.bodies("/api/mcp/route/turn") == []


def git_repo(path: Any, remote: str | None = None) -> None:
    def git(*args: str) -> None:
        subprocess.run(["git", "-C", str(path), *args], check=True, capture_output=True)

    path.mkdir(parents=True, exist_ok=True)
    git("init", "-q", "-b", "main")
    git("config", "user.email", "t@example.com")
    git("config", "user.name", "t")
    git("config", "commit.gpgsign", "false")
    git("commit", "-q", "--allow-empty", "-m", "first")
    if remote:
        git("remote", "add", "origin", remote)


async def recorded_workspace(fake: FakeGlen, glen: Glen) -> dict[str, str] | None:
    await Agent(text_model(), capabilities=[glen]).run("hi")
    await glen.flush()
    [body] = fake.bodies("/api/mcp/route/turn")
    return body["additionalContext"].get("workspace")


async def test_the_workspace_is_read_from_git_by_default(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    git_repo(tmp_path / "work", remote="https://user:token@GitHub.com/acme/widgets.git")
    monkeypatch.chdir(tmp_path / "work")
    fake = FakeGlen()

    workspace = await recorded_workspace(fake, make_glen(fake))

    assert workspace is not None
    assert workspace["repoName"] == "work"
    assert workspace["branch"] == "main"
    assert workspace["remoteUrl"] == "https://github.com/acme/widgets"
    assert len(workspace["commit"]) >= 7
    assert "token" not in json.dumps(workspace)


async def test_an_ssh_remote_becomes_an_https_url(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    git_repo(tmp_path / "work", remote="git@github.com:acme/widgets.git")
    monkeypatch.chdir(tmp_path / "work")
    fake = FakeGlen()

    workspace = await recorded_workspace(fake, make_glen(fake))

    assert workspace is not None
    assert workspace["remoteUrl"] == "https://github.com/acme/widgets"


async def test_outside_a_repository_no_workspace_is_sent(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
    fake = FakeGlen()

    assert await recorded_workspace(fake, make_glen(fake)) is None


async def test_an_explicit_workspace_wins_and_none_turns_it_off(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    git_repo(tmp_path / "work", remote="https://github.com/acme/widgets")
    monkeypatch.chdir(tmp_path / "work")

    fake = FakeGlen()
    explicit = {"repoName": "mine"}
    assert await recorded_workspace(fake, make_glen(fake, workspace=explicit)) == explicit

    fake = FakeGlen()
    assert await recorded_workspace(fake, make_glen(fake, workspace=None)) is None


def seeing_model(seen: list[dict[str, Any]]) -> FunctionModel:
    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append(
            {
                "tools": [t.name for t in info.function_tools],
                "instructions": info.instructions or "",
            }
        )
        return ModelResponse(parts=[TextPart("ok")])

    return FunctionModel(respond)


async def run_in_mode(mode: str) -> tuple[FakeGlen, dict[str, Any]]:
    fake = FakeGlen(mode=mode)
    glen = make_glen(fake)
    seen: list[dict[str, Any]] = []
    await Agent(seeing_model(seen), capabilities=[glen]).run("fix billing")
    await glen.flush()
    return fake, seen[0]


async def test_on_recalls_and_records() -> None:
    fake, seen = await run_in_mode("on")
    assert "glen_search" in seen["tools"] and "glen_search" in seen["instructions"]
    assert len(fake.bodies("/api/mcp/route/turn")) == 1


async def test_off_uses_and_records_nothing() -> None:
    fake, seen = await run_in_mode("off")
    assert "glen_search" not in seen["tools"] and "glen_search" not in seen["instructions"]
    assert fake.bodies("/api/mcp/route/turn") == []


async def test_silent_records_but_surfaces_nothing() -> None:
    fake, seen = await run_in_mode("silent")
    assert "glen_search" not in seen["tools"] and "glen_search" not in seen["instructions"]
    assert len(fake.bodies("/api/mcp/route/turn")) == 1


async def test_incognito_recalls_but_records_nothing() -> None:
    fake, seen = await run_in_mode("incognito")
    assert "glen_search" in seen["tools"] and "glen_search" in seen["instructions"]
    assert fake.bodies("/api/mcp/route/turn") == []


async def test_the_mode_is_looked_up_once_for_many_runs() -> None:
    fake = FakeGlen()
    glen = make_glen(fake)
    agent = Agent(text_model(), capabilities=[glen])
    for _ in range(3):
        await agent.run("hi")
    await glen.flush()
    assert len(fake.bodies("/api/mcp/route/controlStatus")) == 1


async def test_an_unreachable_control_status_leaves_glen_on() -> None:
    fake = FakeGlen(control_status=500)
    glen = make_glen(fake)
    seen: list[dict[str, Any]] = []
    await Agent(seeing_model(seen), capabilities=[glen]).run("hi")
    await glen.flush()
    assert "glen_search" in seen[0]["tools"]
    assert len(fake.bodies("/api/mcp/route/turn")) == 1


async def test_modes_false_never_asks_glen_for_the_mode() -> None:
    fake = FakeGlen(mode="off")
    glen = make_glen(fake, modes=False)
    await Agent(text_model(), capabilities=[glen]).run("hi")
    await glen.flush()
    assert fake.bodies("/api/mcp/route/controlStatus") == []
    assert len(fake.bodies("/api/mcp/route/turn")) == 1


async def test_a_turn_response_that_says_incognito_stops_the_next_recording() -> None:
    fake = FakeGlen(turn_json={"serverMode": "incognito"})
    glen = make_glen(fake)
    agent = Agent(text_model(), capabilities=[glen])
    await agent.run("first")
    await glen.flush()
    await agent.run("second")
    await glen.flush()
    assert len(fake.bodies("/api/mcp/route/turn")) == 1


async def test_a_paused_recording_is_reported_once() -> None:
    notices: list[str] = []
    fake = FakeGlen(turn_json={"recordingPaused": {"message": "Recording paused: plan limit."}})
    glen = make_glen(fake, on_notice=notices.append)
    agent = Agent(text_model(), capabilities=[glen])
    await agent.run("first")
    await agent.run("second")
    await glen.flush()
    assert notices == ["Recording paused: plan limit."]


async def test_a_rejected_key_is_reported_to_the_app_not_just_logged() -> None:
    notices: list[str] = []
    fake = FakeGlen(turn_statuses=[401, 401])
    glen = make_glen(fake, on_notice=notices.append)
    agent = Agent(text_model(), capabilities=[glen])
    await agent.run("first")
    await agent.run("second")
    await glen.flush()
    assert len(notices) == 1 and "rejected the API key" in notices[0]


def commit_file(repo: Any, name: str, text: str, message: str) -> str:
    (repo / name).write_text(text)
    subprocess.run(["git", "-C", str(repo), "add", name], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(repo), "commit", "-q", "-m", message], check=True, capture_output=True
    )
    done = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    )
    return done.stdout.strip()


def code_search_call(**args: Any) -> FunctionModel:
    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart("glen_code_search", args)])
        return ModelResponse(parts=[TextPart("done")])

    return FunctionModel(respond)


def tool_result(result: Any) -> str:
    returns = [
        part.content
        for message in result.all_messages()
        for part in getattr(message, "parts", [])
        if part.part_kind == "tool-return"
    ]
    return str(returns[-1])


async def test_code_search_traces_the_blamed_commit_and_its_trailer(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "work"
    git_repo(repo, remote="git@github.com:acme/widgets.git")
    sha = commit_file(
        repo,
        "retry.py",
        "RETRY_MS = 30_000\nbackoff = 2\n",
        "Raise retry backoff\n\nGlen-Conversation: conv-42",
    )
    monkeypatch.chdir(repo)
    fake = FakeGlen()
    glen = make_glen(fake)

    result = await Agent(
        code_search_call(path="retry.py", start_line=1, question="why 30s?"),
        capabilities=[glen],
    ).run("explain")

    [body] = fake.bodies("/api/mcp/route/codeSearch")
    assert body["shas"] == [sha]
    assert body["trailerConversationIds"] == ["conv-42"]
    assert body["repoUrl"] == "https://github.com/acme/widgets"
    assert body["branch"] == "main"
    assert body["query"] == "why 30s?"
    assert "Dana asked for it." in tool_result(result)


async def test_code_search_without_a_trace_says_so(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "work"
    git_repo(repo, remote="https://github.com/acme/widgets")
    commit_file(repo, "a.py", "x = 1\n", "add a")
    monkeypatch.chdir(repo)
    fake = FakeGlen(code_search_json={"state": "no_anchor", "markdown": "", "conversationCount": 0})

    result = await Agent(
        code_search_call(path="a.py", start_line=1, question="why?"),
        capabilities=[make_glen(fake)],
    ).run("explain")

    assert "No Glen trace found" in tool_result(result)


async def test_code_search_does_not_ask_glen_about_lines_that_are_not_committed(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "work"
    git_repo(repo, remote="https://github.com/acme/widgets")
    commit_file(repo, "a.py", "x = 1\n", "add a")
    (repo / "a.py").write_text("x = 1\ny = 2\n")
    monkeypatch.chdir(repo)
    fake = FakeGlen()

    result = await Agent(
        code_search_call(path="a.py", start_line=2, question="why?"),
        capabilities=[make_glen(fake)],
    ).run("explain")

    assert "not committed yet" in tool_result(result)
    assert fake.bodies("/api/mcp/route/codeSearch") == []


async def test_code_search_is_offered_only_in_a_repo_with_a_remote(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def tools_seen(**kwargs: Any) -> list[str]:
        seen: list[dict[str, Any]] = []
        await Agent(seeing_model(seen), capabilities=[make_glen(FakeGlen(), **kwargs)]).run("x")
        return seen[0]["tools"]

    with_remote = tmp_path / "with"
    git_repo(with_remote, remote="https://github.com/acme/widgets")
    monkeypatch.chdir(with_remote)
    assert "glen_code_search" in await tools_seen()
    assert "glen_code_search" not in await tools_seen(code_search=False)

    no_remote = tmp_path / "without"
    git_repo(no_remote)
    monkeypatch.chdir(no_remote)
    assert "glen_code_search" not in await tools_seen()

    monkeypatch.chdir(with_remote)
    off = FakeGlen(mode="off")
    seen: list[dict[str, Any]] = []
    await Agent(seeing_model(seen), capabilities=[make_glen(off)]).run("x")
    assert "glen_code_search" not in seen[0]["tools"]


def committing_agent(repo: Any, command: str, glen: Glen) -> Agent[None, str]:
    """An agent whose model runs `command` through a real shell tool in `repo`, then answers."""

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart("shell", {"command": command})])
        return ModelResponse(parts=[TextPart("done")])

    agent: Agent[None, str] = Agent(FunctionModel(respond), capabilities=[glen])

    @agent.tool_plain
    def shell(command: str) -> str:
        done = subprocess.run(
            command, shell=True, cwd=repo, capture_output=True, text=True, check=False
        )
        return done.stdout + done.stderr

    return agent


def head(repo: Any) -> str:
    done = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    )
    return done.stdout.strip()


COMMIT = (
    "echo hi > f.txt && git add f.txt && git commit -q -m x && git log -1 --format='[main %h] x'"
)


async def test_a_commit_the_agent_made_is_linked_to_the_conversation(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "work"
    git_repo(repo, remote="https://github.com/acme/widgets")
    monkeypatch.chdir(repo)
    fake = FakeGlen()
    glen = make_glen(fake)

    await committing_agent(repo, COMMIT, glen).run("commit it", conversation_id="conv-1")
    await glen.flush()

    [anchor] = fake.bodies("/api/mcp/route/commitAnchor")
    assert anchor == {
        "sha": head(repo),
        "repoUrl": "https://github.com/acme/widgets",
        "conversationId": "conv-1",
        "branch": "main",
    }
    paths = [r.url.path for r in fake.requests]
    assert paths.index("/api/mcp/route/turn") < paths.index("/api/mcp/route/commitAnchor")


async def test_commands_that_do_not_commit_link_nothing(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "work"
    git_repo(repo, remote="https://github.com/acme/widgets")
    monkeypatch.chdir(repo)
    for command in ("git status", "git commit --dry-run -m x", "echo '[main abc1234] x'"):
        fake = FakeGlen()
        glen = make_glen(fake)
        await committing_agent(repo, command, glen).run("go")
        await glen.flush()
        assert fake.bodies("/api/mcp/route/commitAnchor") == [], command


async def test_commit_linking_follows_the_mode_and_its_switch(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "work"
    git_repo(repo, remote="https://github.com/acme/widgets")
    monkeypatch.chdir(repo)
    for kwargs in ({"mode": "silent"}, {"mode": "off"}, {"link": False}):
        link = kwargs.pop("link", True)
        fake = FakeGlen(**kwargs)
        glen = make_glen(fake, link_commits=link)
        await committing_agent(repo, COMMIT, glen).run("go")
        await glen.flush()
        assert fake.bodies("/api/mcp/route/commitAnchor") == [], kwargs
        subprocess.run(["git", "-C", str(repo), "reset", "-q", "--hard", "HEAD~1"], check=False)


async def test_a_commit_from_a_failed_run_is_still_linked(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "work"
    git_repo(repo, remote="https://github.com/acme/widgets")
    monkeypatch.chdir(repo)
    fake = FakeGlen()
    glen = make_glen(fake)

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart("shell", {"command": COMMIT})])
        raise RuntimeError("ran out of budget")

    agent: Agent[None, str] = Agent(FunctionModel(respond), capabilities=[glen])

    @agent.tool_plain
    def shell(command: str) -> str:
        done = subprocess.run(command, shell=True, cwd=repo, capture_output=True, text=True)
        return done.stdout

    with pytest.raises(RuntimeError):
        await agent.run("go")
    await glen.flush()

    assert [a["sha"] for a in fake.bodies("/api/mcp/route/commitAnchor")] == [head(repo)]


def skill_call(name: str, **args: Any) -> FunctionModel:
    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart(name, args)])
        return ModelResponse(parts=[TextPart("done")])

    return FunctionModel(respond)


SKILL = {
    "slug": "debug-like-shawn",
    "name": "Debug like Shawn",
    "content": "Start with the logs. Helper: ${GLEN_SKILL_DIR}/check.sh",
    "dependencies": [
        {"name": "jq", "kind": "binary", "description": "json", "sourceUrl": "https://jq.example"}
    ],
    "updatedAt": "2026-10-01T09:00:00Z",
    "author": {"name": "Shawn"},
    "files": [],
    "prefetch": [],
}


async def test_skill_tools_follow_the_mode_the_flag_and_the_account() -> None:
    async def tools_seen(fake: FakeGlen, **kwargs: Any) -> list[str]:
        seen: list[dict[str, Any]] = []
        await Agent(seeing_model(seen), capabilities=[make_glen(fake, **kwargs)]).run("x")
        return seen[0]["tools"]

    assert {"glen_skill_search", "glen_skill_use"} <= set(await tools_seen(FakeGlen()))
    assert "glen_skill_use" not in await tools_seen(FakeGlen(), skills=False)
    assert "glen_skill_use" not in await tools_seen(FakeGlen(skill_use=False))
    assert "glen_skill_use" not in await tools_seen(FakeGlen(mode="silent"))
    assert "glen_skill_use" not in await tools_seen(FakeGlen(mode="off"))


async def test_skill_search_lists_what_the_org_has() -> None:
    fake = FakeGlen(
        skills=[
            {
                "slug": "debug-like-shawn",
                "name": "Debug like Shawn",
                "description": "How Shawn triages a failing job",
                "author": {"name": "Shawn"},
                "orgUseCount": 7,
            }
        ]
    )
    result = await Agent(
        skill_call("glen_skill_search", query="debug", limit=5), capabilities=[make_glen(fake)]
    ).run("go")

    assert fake.bodies("/api/mcp/route/skillSearch") == [{"limit": 5, "query": "debug"}]
    text = tool_result(result)
    assert "`debug-like-shawn`" in text and "by Shawn, used 7x" in text


async def test_using_a_skill_returns_its_instructions_with_provenance() -> None:
    fake = FakeGlen(skill=SKILL)
    result = await Agent(
        skill_call("glen_skill_use", slug="debug-like-shawn"), capabilities=[make_glen(fake)]
    ).run("go", conversation_id="conv-9")

    assert fake.bodies("/api/mcp/route/skillUse") == [
        {"slug": "debug-like-shawn", "conversationId": "conv-9"}
    ]
    text = tool_result(result)
    assert text.startswith('> Glen skill "Debug like Shawn" by Shawn, updated 2026-10-01')
    assert "Start with the logs." in text
    assert "- jq (binary): json (https://jq.example)" in text


async def test_bundled_files_are_unpacked_and_the_directory_is_substituted() -> None:
    skill = {
        **SKILL,
        "files": [
            {
                "path": "notes.md",
                "isExecutable": False,
                "contentType": "text/markdown",
                "content": "hi",
            },
            {
                "path": "bin/check.sh",
                "isExecutable": True,
                "contentType": "text/x-sh",
                "url": "https://files.example/check.sh",
            },
            {
                "path": "../evil.sh",
                "isExecutable": True,
                "contentType": "text/x-sh",
                "content": "x",
            },
        ],
    }
    fake = FakeGlen(skill=skill)
    result = await Agent(
        skill_call("glen_skill_use", slug="debug-like-shawn"), capabilities=[make_glen(fake)]
    ).run("go")

    text = tool_result(result)
    assert "${GLEN_SKILL_DIR}" not in text
    directory = text.split("Helper: ")[1].split("/check.sh")[0]
    assert (Path(directory) / "notes.md").read_text() == "hi"
    script = Path(directory) / "bin" / "check.sh"
    assert script.read_text().endswith("echo from-url\n") and os.access(script, os.X_OK)
    assert not (Path(directory).parent / "evil.sh").exists()
    assert "../evil.sh (unsafe path)" in text
    download = [r for r in fake.requests if r.url.host == "files.example"]
    assert download and "x-api-key" not in download[0].headers


async def test_an_unknown_skill_says_so() -> None:
    fake = FakeGlen(skill_status=404)
    result = await Agent(
        skill_call("glen_skill_use", slug="nope"), capabilities=[make_glen(fake)]
    ).run("go")
    assert "No team skill with the slug 'nope'" in tool_result(result)
