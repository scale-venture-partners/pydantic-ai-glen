# Changelog

## Unreleased

- The agent is told to search Glen before changing a system the team already worked on, through the capability's instructions, as the Claude Code plugin does. `instructions=False` turns it off.
- A run that fails or is cancelled is now recorded (what it did so far), where before only finished runs were. Its error still raises. A background-agent deadline is a cancellation, so those runs are no longer lost.
- `workspace` defaults to `"auto"`: each recorded turn carries the repo, branch, commit and remote read from git in the current directory, as the Claude Code plugin sends them, so Glen can tie a conversation to a repo and its PRs. Pass `workspace=None` to send none, or a dict to set your own. The remote is sent as `https://host/owner/repo` with credentials dropped.
- The capability follows the account's Glen mode, as the Claude Code plugin does: `off` uses and records nothing, `silent` records but hides the tool and instructions, `incognito` keeps the tool but records nothing. The mode is read from Glen's control status about once a minute and updated from each turn response; if Glen cannot be reached the agent runs as `on`. `modes=False` turns this off. A paused recording (plan limit) and a rejected key (401, 402, 403, 426) are now reported once through `on_notice` as well as logged.
- `glen_code_search(path, start_line, question, end_line=None)`: a tool that traces lines of code to the agent conversations that wrote them, like `glen code-search` in the CLI. It blames the lines, reads each commit's `Glen-Conversation` trailer and asks Glen. It is offered only in a git checkout with a remote (and not in `off` or `silent` mode); `code_search=False` removes it.
- A run's commits are linked to its conversation, as the Claude Code plugin's `pr-link` hook does after a `git commit`: the finished run's tool calls are scanned for `git commit` commands and the `[branch sha]` line git printed, each commit that exists in the checkout is resolved to its full SHA, and Glen is told (`commitAnchor`) after the turn is recorded. Only in `on` mode, in a git checkout with a remote; `link_commits=False` turns it off. Failed and cancelled runs link what they committed too.
- Team skills, as in the Claude Code plugin: `glen_skill_search(query, limit)` and `glen_skill_use(slug)`. A skill's bundled files are unpacked to a new temporary directory (paths outside it are refused, 50 files and 5 MB per file at most, downloads sent without Glen credentials) and `${GLEN_SKILL_DIR}` in its instructions is replaced with it. Offered only when Glen allows skills for the account and the mode recalls; `skills=False` removes them. The per-turn skill suggestions are not surfaced.

## 0.1.0

First release: a Pydantic AI capability with a `glen_search` tool and automatic recording of
every run to Glen.
