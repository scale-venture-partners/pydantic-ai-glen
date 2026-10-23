r"""Interactive demo: chat with an agent that can search Glen and records the chat to Glen.

    export GLEN_API_KEY=... GLEN_ORG_ID=...
    uv run --with "pydantic-ai-slim[anthropic]" python examples/demo.py \
        --model anthropic:claude-sonnet-5-5

Everything you type is recorded to your real Glen org and becomes team memory.
Use --search-only to try search without recording anything.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ToolCallPart

from pydantic_ai_glen import Glen

INSTRUCTIONS = (
    "You are a helpful engineering assistant. Before answering questions about systems, "
    "decisions or history the team may have worked on, call glen_search with one full "
    "question and base your answer on what it returns."
)


async def chat(agent: Agent, glen: Glen) -> None:
    history: list[ModelMessage] = []
    conversation_id: str | None = None
    print("Type a message, or an empty line to quit.\n")
    while True:
        try:
            prompt = await asyncio.to_thread(input, "you> ")
        except EOFError:
            break
        if not prompt.strip():
            break
        result = await agent.run(prompt, message_history=history, conversation_id=conversation_id)
        history = result.all_messages()
        conversation_id = result.conversation_id
        for message in result.new_messages():
            for part in getattr(message, "parts", []):
                if isinstance(part, ToolCallPart) and part.tool_name == "glen_search":
                    print(f"  [glen_search] {part.args_as_dict().get('query')}")
        print(f"\nagent> {result.output}\n")
    # Without this, a recording still in flight is dropped when the process exits.
    await glen.flush()
    if conversation_id and glen.record:
        print(
            f"Recorded as conversation {conversation_id}. See https://app.tryglen.com/conversations"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--model", default=os.environ.get("DEMO_MODEL"), required=not os.environ.get("DEMO_MODEL")
    )
    parser.add_argument("--search-only", action="store_true", help="do not record the chat to Glen")
    parser.add_argument("--yes", action="store_true", help="skip the recording confirmation")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)
    if not args.search_only and not args.yes:
        answer = input("This records your chat to your real Glen org. Continue? [y/N] ")
        if answer.strip().lower() != "y":
            sys.exit("Aborted. Re-run with --search-only to skip recording.")

    glen = Glen(record=not args.search_only, agent_name="pydantic-ai-glen-demo")
    agent = Agent(args.model, instructions=INSTRUCTIONS, capabilities=[glen])
    asyncio.run(chat(agent, glen))


if __name__ == "__main__":
    main()
