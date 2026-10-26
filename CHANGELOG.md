# Changelog

## Unreleased

- The agent is told to search Glen before changing a system the team already worked on, through the capability's instructions, as the Claude Code plugin does. `instructions=False` turns it off.
- A run that fails or is cancelled is now recorded (what it did so far), where before only finished runs were. Its error still raises. A background-agent deadline is a cancellation, so those runs are no longer lost.

## 0.1.0

First release: a Pydantic AI capability with a `glen_search` tool and automatic recording of
every run to Glen.
