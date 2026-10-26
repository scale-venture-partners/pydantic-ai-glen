"""Trace lines of code to the agent conversations that wrote them (`glen code-search`).

The client does the git work (blame the lines, read each commit's `Glen-Conversation` trailer)
and Glen answers from the conversations behind those commits.
"""

from __future__ import annotations

import asyncio
import re
import subprocess
from dataclasses import dataclass

GIT_TIMEOUT = 4.0
ZERO_SHA = "0" * 40

_BLAME_SHA = re.compile(r"^([0-9a-f]{40}) \d+ \d+")
_TRAILER = re.compile(r"^Glen-Conversation:\s*(.+)$", re.IGNORECASE)


@dataclass(frozen=True)
class Blamed:
    shas: list[str]
    trailer_conversation_ids: list[str]


class NotTraceable(Exception):
    """The lines cannot be traced; the message is for the agent."""


def _git(cwd: str, args: list[str]) -> str | None:
    try:
        done = subprocess.run(
            ["git", "-C", cwd, *args],
            capture_output=True,
            text=True,
            timeout=GIT_TIMEOUT,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout if done.returncode == 0 else None


def blame(cwd: str, path: str, start: int, end: int) -> Blamed:
    if start < 1 or end < start:
        raise NotTraceable("The line range is invalid: start must be 1 or more and at most end.")
    out = _git(cwd, ["blame", "-w", "-M", "-C", "-L", f"{start},{end}", "--porcelain", "--", path])
    if out is None:
        raise NotTraceable(
            f"Could not read git history for {path}:{start}-{end} (untracked file or bad range)."
        )
    shas = list(
        dict.fromkeys(
            m.group(1)
            for line in out.splitlines()
            if (m := _BLAME_SHA.match(line)) and m.group(1) != ZERO_SHA
        )
    )
    if not shas:
        raise NotTraceable("Those lines are not committed yet, so there is no history to trace.")
    ids: list[str] = []
    for sha in shas:
        body = _git(cwd, ["show", "-s", "--format=%B", sha]) or ""
        for line in body.splitlines():
            if (m := _TRAILER.match(line.strip())) and m.group(1).strip() not in ids:
                ids.append(m.group(1).strip())
    return Blamed(shas, ids)


async def blame_async(cwd: str, path: str, start: int, end: int) -> Blamed:
    return await asyncio.to_thread(blame, cwd, path, start, end)
