"""Direct tests of the pure helpers, which the end-to-end tests only reach through one path."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import BaseModel
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

from pydantic_ai_glen import _anchors, _code_search, _control, _git, _messages, _skills

# --- remote URLs -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("git@github.com:Acme/Repo.git", "https://github.com/Acme/Repo"),
        ("git@GitHub.com:acme/repo", "https://github.com/acme/repo"),
        ("ssh://git@github.com:22/acme/repo.git", "https://github.com/acme/repo"),
        ("ssh://git@github.com/acme/repo.git", "https://github.com/acme/repo"),
        ("https://github.com/acme/repo.git", "https://github.com/acme/repo"),
        ("https://github.com/acme/repo/", "https://github.com/acme/repo"),
        ("https://user:token@GitHub.com/acme/repo.git", "https://github.com/acme/repo"),
        ("http://example.com/a/b", "https://example.com/a/b"),
        ("  https://github.com/acme/repo.git \n", "https://github.com/acme/repo"),
        ("https://gitlab.com/group/sub/repo.git", "https://gitlab.com/group/sub/repo"),
    ],
)
def test_remote_urls_are_canonicalized_without_credentials(raw: str, expected: str) -> None:
    assert _git.canonical_remote_url(raw) == expected


@pytest.mark.parametrize("raw", ["", "not a url", "/local/path/repo", "file:///srv/repo.git"])
def test_unrecognised_remotes_are_dropped(raw: str) -> None:
    assert _git.canonical_remote_url(raw) is None


def _git_run(path: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(path), *args], check=True, capture_output=True)


def _init_repo(path: Path) -> None:
    _git_run(path, "init", "-q", "-b", "main")
    _git_run(path, "config", "user.email", "t@example.com")
    _git_run(path, "config", "user.name", "T")
    (path / "f.txt").write_text("x\n")
    _git_run(path, "add", ".")
    _git_run(path, "commit", "-q", "-m", "init")


async def test_detect_workspace_reads_branch_commit_and_the_branchs_own_remote(
    tmp_path: Path,
) -> None:
    _init_repo(tmp_path)
    _git_run(tmp_path, "remote", "add", "origin", "https://github.com/acme/origin.git")
    _git_run(tmp_path, "remote", "add", "other", "git@github.com:acme/other.git")
    _git_run(tmp_path, "config", "branch.main.remote", "other")

    workspace = await _git.detect_workspace(str(tmp_path))

    assert workspace is not None
    assert workspace["cwd"] == str(tmp_path)
    assert workspace["repoName"] == tmp_path.name
    assert workspace["branch"] == "main"
    assert len(workspace["commit"]) >= 7
    assert workspace["remoteUrl"] == "https://github.com/acme/other"


async def test_detect_workspace_falls_back_to_origin(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    _git_run(tmp_path, "remote", "add", "origin", "https://github.com/acme/origin.git")

    workspace = await _git.detect_workspace(str(tmp_path))

    assert workspace is not None
    assert workspace["remoteUrl"] == "https://github.com/acme/origin"


async def test_detect_workspace_without_a_remote_has_no_remote_url(tmp_path: Path) -> None:
    _init_repo(tmp_path)

    workspace = await _git.detect_workspace(str(tmp_path))

    assert workspace is not None
    assert "remoteUrl" not in workspace


async def test_detect_workspace_on_a_detached_head_skips_the_branch_remote(
    tmp_path: Path,
) -> None:
    _init_repo(tmp_path)
    _git_run(tmp_path, "remote", "add", "origin", "https://github.com/acme/origin.git")
    _git_run(tmp_path, "checkout", "-q", "--detach")

    workspace = await _git.detect_workspace(str(tmp_path))

    assert workspace is not None
    assert workspace["branch"] == "HEAD"
    assert workspace["remoteUrl"] == "https://github.com/acme/origin"


async def test_detect_workspace_outside_a_repo_is_none(tmp_path: Path) -> None:
    assert await _git.detect_workspace(str(tmp_path)) is None


async def test_detect_workspace_for_a_missing_directory_is_none(tmp_path: Path) -> None:
    assert await _git.detect_workspace(str(tmp_path / "nope")) is None


def test_git_is_not_run_once_the_deadline_has_passed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("git was run after the deadline")

    monkeypatch.setattr(_git.subprocess, "run", forbidden)

    assert _git._git(str(tmp_path), ["rev-parse", "HEAD"], deadline=0.0) is None


# --- commit detection ------------------------------------------------------------------------


@pytest.mark.parametrize(
    "command",
    [
        "git commit -m x",
        "git commit",
        "git -C /repo commit -m x",
        "git -c user.name=a commit -m x",
        "git --git-dir=/x/.git commit -m x",
        "cd repo && git add . && git commit -m x",
        "git add .; git commit -m x",
        "git add . | git commit -F -",
        "git add .\ngit commit -m x",
        "git --no-pager commit -m x",
    ],
)
def test_commit_commands_are_recognised(command: str) -> None:
    assert _anchors.is_commit_command(command)


@pytest.mark.parametrize(
    "command",
    [
        "git commit --dry-run",
        "git status",
        "git log --oneline",
        "git -C commit status",
        "git -c commit status",
        "echo commit",
        "commit",
        "git add . && git push",
        "",
    ],
)
def test_other_commands_are_not_commits(command: str) -> None:
    assert not _anchors.is_commit_command(command)


def _call(call_id: str, args: Any, name: str = "shell") -> ModelResponse:
    return ModelResponse(parts=[ToolCallPart(name, args, tool_call_id=call_id)])


def _ret(call_id: str, content: str, name: str = "shell") -> ModelRequest:
    return ModelRequest(parts=[ToolReturnPart(name, content, tool_call_id=call_id)])


def test_committed_shas_come_from_commit_calls_only_oldest_first_without_repeats() -> None:
    messages = [
        _call("a", {"command": "git commit -m one"}),
        _ret("a", "[main abc1234] one\n 1 file changed"),
        _call("b", {"command": "git status"}),
        _ret("b", "[main deadbee] looks like a commit line"),
        _call("c", {"cmd": ["git", "commit"], "nested": {"x": "git commit -m two"}}),
        _ret("c", "[feature/x (root-commit) 1234567890abcdef1234567890abcdef12345678] two"),
        _call("d", {"command": "git commit -m again"}),
        _ret("d", "[main abc1234] one"),
    ]

    assert _anchors.committed_shas(messages) == [
        "abc1234",
        "1234567890abcdef1234567890abcdef12345678",
    ]


def test_committed_shas_ignores_unparseable_arguments_and_unrelated_returns() -> None:
    messages = [
        _call("a", "{not json"),
        _ret("a", "[main abc1234] x"),
        _ret("zzz", "[main abc1234] x"),
        ModelRequest(parts=[UserPromptPart("[main abc1234] user text")]),
    ]

    assert _anchors.committed_shas(messages) == []


def test_committed_shas_reads_every_line_of_a_result() -> None:
    messages = [
        _call("a", {"command": "git commit -m x"}),
        _ret("a", "hook output\n[main abc1234] first\n[main bcd2345] second\n"),
    ]

    assert _anchors.committed_shas(messages) == ["abc1234", "bcd2345"]


async def test_resolve_shas_keeps_only_commits_in_this_checkout(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    full = subprocess.run(
        ["git", "-C", str(tmp_path), "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()

    resolved = await _anchors.resolve_shas(str(tmp_path), [full[:8], full, "0" * 40, "nonsense"])

    assert resolved == [full]


async def test_resolve_shas_outside_a_repo_is_empty(tmp_path: Path) -> None:
    assert await _anchors.resolve_shas(str(tmp_path), ["abc1234"]) == []


def test_full_sha_for_a_missing_directory_is_none(tmp_path: Path) -> None:
    assert _anchors._full_sha(str(tmp_path / "nope"), "abc1234") is None


# --- blame ------------------------------------------------------------------------------------


def test_blame_collects_shas_and_conversation_trailers(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    (tmp_path / "f.txt").write_text("x\ny\n")
    _git_run(tmp_path, "add", ".")
    _git_run(
        tmp_path,
        "commit",
        "-q",
        "-m",
        "second\n\nGlen-Conversation: conv-1\nglen-conversation: conv-2\nGlen-Conversation: conv-1",
    )

    blamed = _code_search.blame(str(tmp_path), "f.txt", 1, 2)

    assert len(blamed.shas) == 2
    assert all(len(s) == 40 for s in blamed.shas)
    assert blamed.trailer_conversation_ids == ["conv-1", "conv-2"]


def test_blame_single_line_range_is_inclusive(tmp_path: Path) -> None:
    _init_repo(tmp_path)

    assert len(_code_search.blame(str(tmp_path), "f.txt", 1, 1).shas) == 1


@pytest.mark.parametrize(("start", "end"), [(0, 1), (-3, 1), (3, 2)])
def test_blame_rejects_bad_ranges(tmp_path: Path, start: int, end: int) -> None:
    with pytest.raises(_code_search.NotTraceable, match="line range is invalid"):
        _code_search.blame(str(tmp_path), "f.txt", start, end)


def test_blame_of_an_untracked_file_cannot_be_traced(tmp_path: Path) -> None:
    _init_repo(tmp_path)

    with pytest.raises(_code_search.NotTraceable, match="Could not read git history"):
        _code_search.blame(str(tmp_path), "missing.txt", 1, 1)


def test_blame_of_uncommitted_lines_says_they_are_not_committed(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    (tmp_path / "f.txt").write_text("x\nnew line\n")

    with pytest.raises(_code_search.NotTraceable, match="not committed yet"):
        _code_search.blame(str(tmp_path), "f.txt", 2, 2)


def test_blame_outside_a_repo_cannot_be_traced(tmp_path: Path) -> None:
    (tmp_path / "f.txt").write_text("x\n")

    with pytest.raises(_code_search.NotTraceable):
        _code_search.blame(str(tmp_path), "f.txt", 1, 1)


# --- control mode -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("mode", "recalls", "records"),
    [
        ("on", True, True),
        ("incognito", True, False),
        ("silent", False, True),
        ("off", False, False),
    ],
)
def test_each_mode_allows_exactly_its_behaviour(mode: str, recalls: bool, records: bool) -> None:
    control = _control.Control(mode)
    assert (control.recalls, control.records) == (recalls, records)


@pytest.mark.parametrize(
    ("data", "mode"),
    [
        ({"serverMode": "silent"}, "silent"),
        ({"mode": "off"}, "off"),
        ({"serverMode": "off", "mode": "on"}, "off"),
        ({"serverMode": "bogus", "mode": "incognito"}, "incognito"),
        ({"serverIncognito": True}, "incognito"),
        ({"serverIncognito": False, "serverSilent": True}, "silent"),
        ({"serverIncognito": False, "serverSilent": False}, "on"),
        ({"serverIncognito": False}, "on"),
        ({"serverIncognito": True, "serverSilent": True}, "incognito"),
        ({"serverIncognito": "yes"}, None),
        ({"mode": "bogus"}, None),
        ({}, None),
    ],
)
def test_mode_is_read_the_way_the_plugin_reads_it(data: dict[str, Any], mode: str | None) -> None:
    assert _control.mode_of(data) == mode


def test_parse_control_reads_the_mode_and_skill_use_flag() -> None:
    assert _control.parse_control({"mode": "silent", "skillUse": False}) == _control.Control(
        "silent", False
    )
    assert _control.parse_control({"mode": "on"}) == _control.Control("on", True)
    assert _control.parse_control({"mode": "on", "skillUse": None}).skill_use is True  # type: ignore[union-attr]


@pytest.mark.parametrize("data", [None, [], "on", 3, {}, {"mode": "bogus"}])
def test_parse_control_rejects_what_is_not_a_status(data: Any) -> None:
    assert _control.parse_control(data) is None


# --- message conversion -----------------------------------------------------------------------


def test_text_and_structured_outputs_are_serialized() -> None:
    class Out(BaseModel):
        n: int

    assert _messages._output_text("hi") == "hi"
    assert _messages._output_text(Out(n=1)) == '{"n":1}'
    assert _messages._output_text({"a": object}).startswith('{"a":')


def test_prompt_text_keeps_strings_from_multimodal_prompts() -> None:
    assert _messages._prompt_text("hello") == "hello"
    assert _messages._prompt_text(["a", httpx.URL("https://x"), "b"]) == "a\nb"


def test_tool_args_that_cannot_be_read_are_kept_raw(monkeypatch: pytest.MonkeyPatch) -> None:
    part = ToolCallPart("t", "[1]")

    def boom(self: ToolCallPart) -> dict[str, Any]:
        raise ValueError("unreadable")

    assert _messages._tool_args(ToolCallPart("t", '{"a": 1}')) == {"a": 1}
    monkeypatch.setattr(ToolCallPart, "args_as_dict", boom)
    assert _messages._tool_args(part) == {"raw": "[1]"}


def test_a_run_becomes_user_messages_then_one_assistant_message_of_ordered_segments() -> None:
    messages = [
        ModelRequest(parts=[UserPromptPart("first"), UserPromptPart("")]),
        ModelResponse(
            parts=[
                ThinkingPart("hmm"),
                TextPart(""),
                TextPart("calling"),
                ToolCallPart("search", {"q": "x"}),
            ]
        ),
        _ret("1", "result text", name="search"),
        ModelResponse(parts=[TextPart("done")]),
    ]

    out = _messages.to_glen_messages(messages, "done")

    assert [m["role"] for m in out] == ["user", "assistant"]
    assert out[0]["content"] == "first"
    assert "occurredAt" in out[0]
    assert out[1]["content"] == "done"
    assert out[1]["segments"] == [
        {"kind": "thinking", "text": "hmm"},
        {"kind": "text", "text": "calling"},
        {"kind": "tool_call", "text": "search", "jsonContent": {"q": "x"}},
        {"kind": "tool_result", "text": "result text"},
        {"kind": "text", "text": "done"},
    ]


def test_a_run_without_steps_has_no_segments_key() -> None:
    out = _messages.to_glen_messages([ModelRequest(parts=[UserPromptPart("q")])], "a")

    assert out[1] == {"role": "assistant", "content": "a"}


def test_oversized_content_is_truncated_to_the_limits() -> None:
    big = "x" * (_messages.MAX_CONTENT + 10)
    long_segment = "y" * (_messages.MAX_SEGMENT + 10)
    messages = [
        ModelRequest(parts=[UserPromptPart(big)]),
        ModelResponse(parts=[TextPart(long_segment), ThinkingPart(long_segment)]),
        _ret("1", long_segment),
    ]

    out = _messages.to_glen_messages(messages, big)

    assert len(out[0]["content"]) == _messages.MAX_CONTENT
    assert len(out[1]["content"]) == _messages.MAX_CONTENT
    assert [len(s["text"]) for s in out[1]["segments"]] == [_messages.MAX_SEGMENT] * 3


# --- skills -----------------------------------------------------------------------------------


def test_skill_results_are_listed_with_author_and_usage() -> None:
    skills = [
        {
            "slug": "deploy",
            "name": "Deploy",
            "description": "ship it",
            "author": {"name": "Dana"},
            "orgUseCount": 4,
        },
        {"slug": "bare", "name": "Bare", "description": "d"},
    ]

    assert _skills.format_results(skills) == (
        "- `deploy`: Deploy: ship it (by Dana, used 4x in your org)\n"
        "- `bare`: Bare: d (by unknown, used 0x in your org)"
    )
    assert _skills.format_results([]) == "No team skills match."


@pytest.mark.parametrize(
    "relative",
    ["../escape.txt", "a/../../escape.txt", "/etc/passwd", "", "a/.."],
)
def test_paths_that_leave_the_directory_are_unsafe(tmp_path: Path, relative: str) -> None:
    assert _skills._safe_target(tmp_path, relative) is None


def test_nested_paths_stay_inside_the_directory(tmp_path: Path) -> None:
    target = _skills._safe_target(tmp_path, "scripts/run.sh")

    assert target == tmp_path.resolve() / "scripts" / "run.sh"


def test_a_directory_itself_is_not_a_safe_file_target(tmp_path: Path) -> None:
    assert _skills._safe_target(tmp_path, ".") is None


def _unpack(files: list[dict[str, Any]], handler: Any = None) -> Any:
    transport = httpx.MockTransport(handler or (lambda r: httpx.Response(200, content=b"data")))
    return _skills.unpack("my skill/../x!", files, transport)


async def test_unpack_writes_inline_and_downloaded_files_and_marks_executables() -> None:
    root, skipped = await _unpack(
        [
            {"path": "a.txt", "content": "inline"},
            {"path": "sub/b.sh", "url": "https://files.example/b.sh", "isExecutable": True},
        ]
    )

    assert skipped == []
    assert (root / "a.txt").read_text() == "inline"
    assert (root / "sub" / "b.sh").read_bytes() == b"data"
    assert not (root / "a.txt").stat().st_mode & 0o100
    assert (root / "sub" / "b.sh").stat().st_mode & 0o100
    assert root.name.startswith("glen-skill-my_skill____x_-")


async def test_unpack_skips_what_it_cannot_write_safely() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/boom":
            return httpx.Response(500)
        if request.url.path == "/big":
            return httpx.Response(200, content=b"x" * (_skills.MAX_FILE_BYTES + 1))
        return httpx.Response(200, content=b"ok")

    root, skipped = await _unpack(
        [
            {"path": "../out.txt", "content": "x"},
            {"path": "http.txt", "url": "http://files.example/x"},
            {"path": "nourl.txt"},
            {"path": "fail.txt", "url": "https://files.example/boom"},
            {"path": "big.txt", "url": "https://files.example/big"},
            {"path": "fine.txt", "url": "https://files.example/fine"},
        ],
        handler,
    )

    assert skipped == [
        "../out.txt (unsafe path)",
        "http.txt (no https url)",
        "nourl.txt (no https url)",
        "fail.txt (download failed)",
        "big.txt (too large)",
    ]
    assert [p.name for p in root.iterdir()] == ["fine.txt"]
    assert not (root.parent / "out.txt").exists()


async def test_unpack_stops_at_the_file_limit() -> None:
    files = [{"path": f"f{i}.txt", "content": "x"} for i in range(_skills.MAX_FILES + 2)]

    root, skipped = await _unpack(files)

    assert len(list(root.iterdir())) == _skills.MAX_FILES
    assert skipped == [
        f"f{_skills.MAX_FILES}.txt (over the {_skills.MAX_FILES} file limit)",
        f"f{_skills.MAX_FILES + 1}.txt (over the {_skills.MAX_FILES} file limit)",
    ]


async def test_unpack_sends_no_glen_credentials_to_file_hosts() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, content=b"x")

    await _unpack([{"path": "a", "url": "https://files.example/a"}], handler)

    assert "x-api-key" not in seen[0].headers
    assert "x-glen-org-id" not in seen[0].headers


def test_render_substitutes_the_directory_and_lists_requirements_and_skips(
    tmp_path: Path,
) -> None:
    skill = {
        "name": "Deploy",
        "author": {"name": "Dana"},
        "updatedAt": "2026-10-01T12:00:00Z",
        "content": "run ${GLEN_SKILL_DIR}/go.sh",
        "dependencies": [
            {"name": "jq", "kind": "binary", "description": "json", "sourceUrl": "https://jq"}
        ],
    }

    text = _skills.render(skill, tmp_path, ["x.txt (too large)", "y.txt (unsafe path)"])

    assert text == (
        '> Glen skill "Deploy" by Dana, updated 2026-10-01\n\n'
        f"run {tmp_path}/go.sh\n"
        "\n## Requirements\n- jq (binary): json (https://jq)\n"
        "\n(Not unpacked: x.txt (too large), y.txt (unsafe path))"
    )


def test_render_without_files_leaves_the_placeholder_and_extras_out() -> None:
    text = _skills.render({"name": "N", "content": "see ${GLEN_SKILL_DIR}"}, None, [])

    assert text == '> Glen skill "N" by unknown, updated \n\nsee ${GLEN_SKILL_DIR}'


def test_a_symlink_out_of_the_directory_is_not_a_safe_target(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    root = tmp_path / "root"
    root.mkdir()
    (root / "link").symlink_to(outside)

    assert _skills._safe_target(root, "link/x.txt") is None


async def test_unpack_creates_deeply_nested_directories() -> None:
    root, skipped = await _unpack([{"path": "a/b/c/d.txt", "content": "deep"}])

    assert skipped == []
    assert (root / "a" / "b" / "c" / "d.txt").read_text() == "deep"


async def test_unpack_accepts_a_file_of_exactly_the_size_limit() -> None:
    exact = b"x" * _skills.MAX_FILE_BYTES

    root, skipped = await _unpack(
        [{"path": "exact.bin", "url": "https://files.example/e"}],
        lambda r: httpx.Response(200, content=exact),
    )

    assert skipped == []
    assert (root / "exact.bin").stat().st_size == _skills.MAX_FILE_BYTES


async def test_unpack_names_the_directory_after_at_most_forty_characters_of_the_slug() -> None:
    root, _ = await _skills.unpack("a-b_c" + "z" * 60, [], None)

    assert root.name.startswith("glen-skill-a-b_c" + "z" * 35 + "-")


async def test_unpack_treats_a_file_without_a_path_as_unsafe() -> None:
    root, skipped = await _unpack([{"content": "x"}])

    assert skipped == [" (unsafe path)"]
    assert list(root.iterdir()) == []


async def test_unpack_downloads_with_a_thirty_second_timeout() -> None:
    seen: list[Any] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.extensions["timeout"])
        return httpx.Response(200, content=b"x")

    await _unpack([{"path": "a", "url": "https://files.example/a"}], handler)

    assert seen[0]["read"] == _skills.DOWNLOAD_TIMEOUT == 30.0


def test_render_without_content_is_empty_and_lists_each_requirement_on_its_own_line() -> None:
    skill = {
        "name": "N",
        "dependencies": [
            {"name": "a", "kind": "k", "description": "d", "sourceUrl": "u1"},
            {"name": "b", "kind": "k", "description": "d", "sourceUrl": "u2"},
        ],
    }

    text = _skills.render(skill, None, [])

    assert text.startswith('> Glen skill "N" by unknown, updated \n\n\n')
    assert "- a (k): d (u1)\n- b (k): d (u2)" in text
