# Contributing

Issues and pull requests are welcome.

```bash
uv sync --extra dev
uv run ruff check . && uv run ruff format --check .
uv run pytest
```

Tests fake Glen's HTTP API with `httpx.MockTransport`. Never point tests or experiments at a real
Glen organization: everything recorded becomes real team memory. Add a test with every behavior
change, and keep `pydantic-ai-slim>=2.9.0` working.
