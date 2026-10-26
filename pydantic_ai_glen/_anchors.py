"""Find the commits an agent made in a run, so Glen can link them to the conversation.

The Claude Code plugin does this from a hook after each `git commit`. A library agent has no
such hook, so the finished run's tool calls are scanned instead: a tool call whose arguments
contain a `git commit` command, and the `[branch abc1234] message` line git printed in its result.
"""

from __future__ import annotations

import asyncio
import re
import subprocess
from typing import Any

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ToolCallPart,
    ToolReturnPart,
)

_VALUE_FLAGS = {"-C", "-c", "--git-dir", "--work-tree"}
_COMMIT_LINE = re.compile(r"^\[[^\]\n]*?\b([0-9a-f]{7,40})\]", re.MULTILINE)


def _segment_is_git_commit(segment: str) -> bool:
    tokens = segment.split()
    if "git" not in tokens:
        return False
    rest = tokens[tokens.index("git") + 1 :]
    subcommand = next(
        (
            t
            for i, t in enumerate(rest)
            if not t.startswith("-") and (i == 0 or rest[i - 1] not in _VALUE_FLAGS)
        ),
        None,
    )
    return subcommand == "commit" and "--dry-run" not in rest


def is_commit_command(command: str) -> bool:
    return any(_segment_is_git_commit(s) for s in re.split(r"[\n;&|]+", command))


def _strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for v in value.values() for s in _strings(v)]
    if isinstance(value, list):
        return [s for v in value for s in _strings(v)]
    return []


def committed_shas(messages: list[ModelMessage]) -> list[str]:
    """The (possibly abbreviated) SHAs `git commit` printed for commit commands the agent ran,
    oldest first."""
    commit_calls: set[str] = set()
    shas: list[str] = []
    for message in messages:
        if isinstance(message, ModelResponse):
            for call in message.parts:
                if isinstance(call, ToolCallPart):
                    try:
                        args = call.args_as_dict()
                    except Exception:
                        continue
                    if any(is_commit_command(s) for s in _strings(args)):
                        commit_calls.add(call.tool_call_id)
        elif isinstance(message, ModelRequest):
            for part in message.parts:
                if isinstance(part, ToolReturnPart) and part.tool_call_id in commit_calls:
                    for match in _COMMIT_LINE.finditer(part.model_response_str()):
                        if match.group(1) not in shas:
                            shas.append(match.group(1))
    return shas


def _full_sha(cwd: str, sha: str) -> str | None:
    try:
        done = subprocess.run(
            ["git", "-C", cwd, "rev-parse", "--verify", "--quiet", f"{sha}^{{commit}}"],
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    out = done.stdout.strip()
    return out if done.returncode == 0 and len(out) == 40 else None


async def resolve_shas(cwd: str, shas: list[str]) -> list[str]:
    """Full SHAs of the commits that exist in this checkout (an agent can commit somewhere
    else, such as a sandbox)."""
    found = await asyncio.gather(*(asyncio.to_thread(_full_sha, cwd, s) for s in shas))
    return [s for s in dict.fromkeys(found) if s]
