# Changelog

## Unreleased

- The agent is told to search Glen before changing a system the team already worked on, through the capability's instructions, as the Claude Code plugin does. `instructions=False` turns it off.
- A run that fails or is cancelled is now recorded (what it did so far), where before only finished runs were. Its error still raises. A background-agent deadline is a cancellation, so those runs are no longer lost.
- `workspace` defaults to `"auto"`: each recorded turn carries the repo, branch, commit and remote read from git in the current directory, as the Claude Code plugin sends them, so Glen can tie a conversation to a repo and its PRs. Pass `workspace=None` to send none, or a dict to set your own. The remote is sent as `https://host/owner/repo` with credentials dropped.
- The capability follows the account's Glen mode, as the Claude Code plugin does: `off` uses and records nothing, `silent` records but hides the tool and instructions, `incognito` keeps the tool but records nothing. The mode is read from Glen's control status about once a minute and updated from each turn response; if Glen cannot be reached the agent runs as `on`. `modes=False` turns this off. A paused recording (plan limit) and a rejected key (401, 402, 403, 426) are now reported once through `on_notice` as well as logged.

## 0.1.0

First release: a Pydantic AI capability with a `glen_search` tool and automatic recording of
every run to Glen.
