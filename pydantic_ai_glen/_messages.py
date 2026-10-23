"""Convert Pydantic AI messages into the turn payload Glen's recording endpoint expects."""

from __future__ import annotations

from typing import Any

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_core import to_json

MAX_CONTENT = 100_000
MAX_SEGMENT = 30_000


def _output_text(output: Any) -> str:
    if isinstance(output, str):
        return output
    return to_json(output, fallback=str).decode()


def _prompt_text(content: Any) -> str:
    """Text of a user prompt; images, audio and other binary parts are dropped."""
    if isinstance(content, str):
        return content
    return "\n".join(item for item in content if isinstance(item, str))


def _tool_args(part: ToolCallPart) -> Any:
    try:
        return part.args_as_dict()
    except Exception:
        return {"raw": str(part.args)}


def _segments(response: ModelResponse) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for part in response.parts:
        if isinstance(part, TextPart) and part.content:
            out.append({"kind": "text", "text": part.content[:MAX_SEGMENT]})
        elif isinstance(part, ThinkingPart) and part.content:
            out.append({"kind": "thinking", "text": part.content[:MAX_SEGMENT]})
        elif isinstance(part, ToolCallPart):
            out.append(
                {"kind": "tool_call", "text": part.tool_name, "jsonContent": _tool_args(part)}
            )
    return out


def to_glen_messages(messages: list[ModelMessage], final_output: Any) -> list[dict[str, Any]]:
    """One user message per prompt, then one assistant message holding the final answer
    with every step of the run as ordered segments."""
    users: list[dict[str, Any]] = []
    segments: list[dict[str, Any]] = []
    for message in messages:
        if isinstance(message, ModelRequest):
            for part in message.parts:
                if isinstance(part, UserPromptPart):
                    text = _prompt_text(part.content)
                    if text:
                        users.append(
                            {
                                "role": "user",
                                "content": text[:MAX_CONTENT],
                                "occurredAt": part.timestamp.isoformat(),
                            }
                        )
                elif isinstance(part, ToolReturnPart):
                    segments.append(
                        {"kind": "tool_result", "text": part.model_response_str()[:MAX_SEGMENT]}
                    )
        elif isinstance(message, ModelResponse):
            segments.extend(_segments(message))
    assistant: dict[str, Any] = {
        "role": "assistant",
        "content": _output_text(final_output)[:MAX_CONTENT],
    }
    if segments:
        assistant["segments"] = segments
    return [*users, assistant]
