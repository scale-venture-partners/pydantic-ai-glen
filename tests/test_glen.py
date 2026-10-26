from __future__ import annotations

import asyncio
import json
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
        self, turn_statuses: list[int] | None = None, answer: str = "because Dana said so"
    ):
        self.requests: list[httpx.Request] = []
        self.turn_statuses = turn_statuses or []
        self.answer = answer

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.url.path == "/api/mcp/route/search":
            return httpx.Response(
                200, json={"answer": self.answer, "gaps": ["no ticket found"], "citations": []}
            )
        status = self.turn_statuses.pop(0) if self.turn_statuses else 200
        return httpx.Response(status, json={})

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
