"""The workspace Glen ties a conversation to: repo, branch, commit and remote, read from git."""

from __future__ import annotations

import asyncio
import os
import re
import subprocess
import time

CALL_TIMEOUT = 0.5
TOTAL_BUDGET = 1.2

_SSH = re.compile(r"^(?:ssh://)?[^@]+@([^:/]+)(?::\d+)?[:/](.+?)(?:\.git)?$")
_HTTPS = re.compile(r"^https?://(?:[^@/]+@)?([^/]+)/(.+?)(?:\.git)?/?\s*$")


def canonical_remote_url(raw: str) -> str | None:
    """`https://host/owner/repo` for an ssh or https remote, with any credentials dropped."""
    text = raw.strip()
    match = _SSH.match(text) or _HTTPS.match(text)
    if not match:
        return None
    return f"https://{match.group(1).lower()}/{match.group(2).rstrip('/')}"


def _git(cwd: str, args: list[str], deadline: float) -> str | None:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return None
    try:
        done = subprocess.run(
            ["git", "-C", cwd, *args],
            capture_output=True,
            text=True,
            timeout=min(remaining, CALL_TIMEOUT),
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip() or None if done.returncode == 0 else None


def _detect(cwd: str) -> dict[str, str] | None:
    deadline = time.monotonic() + TOTAL_BUDGET
    top = _git(cwd, ["rev-parse", "--show-toplevel"], deadline)
    if not top:
        return None
    workspace = {"cwd": cwd, "repoName": os.path.basename(top)}
    branch = _git(cwd, ["rev-parse", "--abbrev-ref", "HEAD"], deadline)
    if branch:
        workspace["branch"] = branch
    if commit := _git(cwd, ["rev-parse", "--short", "HEAD"], deadline):
        workspace["commit"] = commit
    remote_name = (
        _git(cwd, ["config", "--get", f"branch.{branch}.remote"], deadline)
        if branch and branch != "HEAD"
        else None
    )
    remote = (
        _git(cwd, ["config", "--get", f"remote.{remote_name}.url"], deadline)
        if remote_name
        else None
    ) or _git(cwd, ["config", "--get", "remote.origin.url"], deadline)
    if remote and (url := canonical_remote_url(remote)):
        workspace["remoteUrl"] = url
    return workspace


async def detect_workspace(cwd: str | None = None) -> dict[str, str] | None:
    """The git workspace of `cwd` (default: the current directory), or None when it is not a
    repository, git is missing or slow. Runs off the event loop and never raises."""
    try:
        return await asyncio.to_thread(_detect, cwd or os.getcwd())
    except Exception:
        return None
