"""Team skills: find one in the org's library and unpack it for the agent to follow.

A skill is markdown instructions, optionally with bundled files (scripts, templates). Like the
Claude Code plugin, bundled files are written to a directory and `${GLEN_SKILL_DIR}` in the
instructions is replaced with it.
"""

from __future__ import annotations

import stat
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

import httpx

DIR_PLACEHOLDER = "${GLEN_SKILL_DIR}"
MAX_FILES = 50
MAX_FILE_BYTES = 5_000_000
DOWNLOAD_TIMEOUT = 30.0


def format_results(results: list[dict[str, Any]]) -> str:
    if not results:
        return "No team skills match."
    lines = []
    for skill in results:
        author = (skill.get("author") or {}).get("name", "unknown")
        used = skill.get("orgUseCount", 0)
        lines.append(
            f"- `{skill.get('slug')}`: {skill.get('name')}: {skill.get('description')} "
            f"(by {author}, used {used}x in your org)"
        )
    return "\n".join(lines)


def _safe_target(root: Path, relative: str) -> Path | None:
    path = PurePosixPath(relative)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        return None
    target = (root / Path(*path.parts)).resolve()
    return target if root.resolve() in target.parents else None


async def unpack(
    slug: str, files: list[dict[str, Any]], transport: httpx.AsyncBaseTransport | None
) -> tuple[Path, list[str]]:
    """Write a skill's bundled files to a new directory. Returns it and what was skipped (a path
    that escapes the directory, too many files, a file over the size limit, a failed download)."""
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in slug)[:40]
    root = Path(tempfile.mkdtemp(prefix=f"glen-skill-{safe}-"))
    skipped: list[str] = []
    # No Glen credentials: a bundled file's URL is not Glen's to receive them.
    async with httpx.AsyncClient(transport=transport, timeout=DOWNLOAD_TIMEOUT) as plain:
        for file in files[:MAX_FILES]:
            relative = str(file.get("path", ""))
            target = _safe_target(root, relative)
            if target is None:
                skipped.append(f"{relative} (unsafe path)")
                continue
            try:
                if isinstance(file.get("content"), str):
                    data = file["content"].encode()
                else:
                    url = str(file.get("url", ""))
                    if not url.startswith("https://"):
                        skipped.append(f"{relative} (no https url)")
                        continue
                    response = await plain.get(url)
                    response.raise_for_status()
                    data = response.content
            except (httpx.HTTPError, OSError):
                skipped.append(f"{relative} (download failed)")
                continue
            if len(data) > MAX_FILE_BYTES:
                skipped.append(f"{relative} (too large)")
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            if file.get("isExecutable"):
                target.chmod(target.stat().st_mode | stat.S_IXUSR)
    skipped.extend(f"{f.get('path')} (over the {MAX_FILES} file limit)" for f in files[MAX_FILES:])
    return root, skipped


def render(skill: dict[str, Any], directory: Path | None, skipped: list[str]) -> str:
    content = str(skill.get("content", ""))
    if directory is not None:
        content = content.replace(DIR_PLACEHOLDER, str(directory))
    author = (skill.get("author") or {}).get("name", "unknown")
    updated = str(skill.get("updatedAt", ""))[:10]
    parts = [f'> Glen skill "{skill.get("name")}" by {author}, updated {updated}\n\n{content}']
    dependencies = skill.get("dependencies") or []
    if dependencies:
        parts.append(
            "\n## Requirements\n"
            + "\n".join(
                f"- {d.get('name')} ({d.get('kind')}): {d.get('description')} "
                f"({d.get('sourceUrl')})"
                for d in dependencies
            )
        )
    if skipped:
        parts.append("\n(Not unpacked: " + ", ".join(skipped) + ")")
    return "\n".join(parts)
