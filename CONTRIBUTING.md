# Contributing

Issues and pull requests are welcome.

```bash
uv sync --extra dev
uv run ruff check . && uv run ruff format --check .
uv run pytest --cov
```

Tests fake Glen's HTTP API with `httpx.MockTransport`. Never point tests or experiments at a real
Glen organization: everything recorded becomes real team memory. Add a test with every behavior
change, and keep `pydantic-ai-slim>=2.9.0` working.

Mutation testing (slow, about 30 minutes; delete `mutants/` first so it sees current tests):

```bash
uv run mutmut run --max-children 8
uv run mutmut results
```

The remaining survivors are equivalent mutants (redundant guards, timeouts, defaults that behave the same).
