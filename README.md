# pydantic-ai-glen

A [Pydantic AI](https://ai.pydantic.dev/) capability that connects an agent to
[Glen](https://tryglen.com), a shared team memory. It does two things:

- **Search**: adds a `glen_search` tool so the agent can ask Glen a question.
- **Recording**: after every run, sends that run's conversation to Glen so it becomes
  searchable memory. Glen extracts memory in the background on its side.

## Install

```bash
pip install pydantic-ai-glen
```

Requires Python 3.10+ and `pydantic-ai-slim>=2.9`.

## Setup

1. Create an API key at https://app.tryglen.com/settings/api-keys.
2. In Glen, switch to the organization the agent should use, click **Connect an agent**,
   then **MCP server**, and copy the value next to `x-glen-org-id`.
3. Set both where the app runs (never commit them):

```
GLEN_API_KEY=<your key>
GLEN_ORG_ID=<your organization id>
```

`GLEN_URL` is optional and defaults to `https://app.tryglen.com`.

## Usage

```python
from pydantic_ai import Agent
from pydantic_ai_glen import Glen

glen = Glen()  # reads GLEN_API_KEY / GLEN_ORG_ID
agent = Agent("openai:gpt-5", capabilities=[glen])

result = await agent.run("why does the billing worker batch at 500?")
await glen.flush()  # before a short-lived process exits
```

Recording hooks into `after_run`, so `run`, `run_stream`, `iter` and `run_sync` are all
covered with no wrapper code.

| Option | Default | |
| --- | --- | --- |
| `api_key`, `org_id`, `url` | env vars | Constructor values win over the environment |
| `search` | `True` | Expose the `glen_search` tool |
| `record` | `True` | Send finished runs to Glen |
| `modes` | `True` | Follow the account's Glen mode: `off` uses and records nothing, `silent` records but hides the tool and instructions, `incognito` keeps the tool but records nothing |
| `on_notice` | `None` | Called once per message when Glen says something the user should see (recording paused by a plan limit, a rejected key). Always logged as a warning too |
| `instructions` | `True` | Tell the agent to search Glen before changing a system the team already worked on (needs `search`). Turn it off if your own prompt says it |
| `background` | `True` | Upload without delaying the run. Use `False` with `run_sync` |
| `agent_name`, `agent_version` | `"pydantic-ai"`, `None` | Reported to Glen |
| `workspace` | `"auto"` | `cwd`, `repoName`, `branch`, `commit`, `remoteUrl` for each recorded turn. `"auto"` reads them from git in the current directory (nothing outside a repository; a remote's credentials are dropped), a dict is used as given, `None` sends none |

Conversations are grouped by Pydantic AI's `conversation_id`. Pass your own thread id to
`agent.run(..., conversation_id=...)` to keep one Glen conversation across runs.

### Compared with the Claude Code plugin

The Claude Code plugin works through hooks. This is what carries over to a Pydantic AI agent:

| Claude Code plugin | This capability |
| --- | --- |
| Injects the team-memory rule at session start | Yes: added to the agent's instructions (`instructions=False` to opt out) |
| Records each prompt and turn | Records each run, including one that fails or is cancelled (what it did so far) |
| Honors on / off / silent / incognito and plan limits | Yes: the mode is read from Glen about once a minute (`modes=False` to ignore it); plan limits and a rejected key reach `on_notice` |
| Sends repo, branch and commit with each turn | Yes: read from git in the current directory (`workspace=None` to opt out) |
| Code search, team skills, commit and PR linking | Not yet |

### Behavior

- A Glen failure never fails a run. Search returns a short message to the model; recording
  retries transient errors (429/5xx) three times with the same idempotency key, then logs.
- A run that fails or is cancelled is recorded too (what it did so far, ending with a note of how it
  ended), and its error still raises. A run that fails before its first model request records nothing.
- Only this run's new messages are sent, with tool calls, tool results and thinking as
  ordered segments. System prompts and non-text prompt content (images, audio) are not sent.
- Everything recorded becomes real team memory. Don't point test runs at your real org.

### Known issue

Conversations recorded this way currently show up in Glen labelled as Claude Code sessions
(`claude-code:<conversationId>`). Only the label is affected; recording, search and memory
work normally.

## Try it

`examples/demo.py` is an interactive chat that searches Glen and records the conversation:

```bash
export GLEN_API_KEY=... GLEN_ORG_ID=...
uv run --with "pydantic-ai-slim[anthropic]" python examples/demo.py --model anthropic:claude-sonnet-5-5
uv run --with "pydantic-ai-slim[anthropic]" python examples/demo.py --model anthropic:claude-sonnet-5-5 --search-only
```

It asks before recording, since whatever you type becomes team memory. Use any model string
Pydantic AI supports (`--model` or `DEMO_MODEL`).

## Development

```bash
uv sync --extra dev
uv run ruff check . && uv run ruff format --check .
uv run pytest
```

Tests fake Glen's HTTP API with `httpx.MockTransport`; the agent, capability and message
conversion run for real.

See [CONTRIBUTING.md](CONTRIBUTING.md) and [SECURITY.md](SECURITY.md).

This is an independent integration; Pydantic AI and Glen are their respective owners' projects.

## License

MIT, see [LICENSE](LICENSE).
