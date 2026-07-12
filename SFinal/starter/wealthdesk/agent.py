"""
wealthdesk/agent.py
-------------------
Builds and runs the WealthDesk LangGraph agent.

Merged Session 1 + Session 2: the graph is compiled with a checkpointer
so LangGraph persists conversation history across turns automatically.

Run with:
    python -m wealthdesk.agent
"""
import sqlite3
from uuid import uuid4

from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph

from .config import CHECKPOINT_DB
from .nodes import respond
from .state import WealthDeskState


def build_graph(checkpointer=None):
    """Build and compile the WealthDesk LangGraph graph."""
    builder = StateGraph(WealthDeskState)
    builder.add_node("respond", respond)
    builder.set_entry_point("respond")
    builder.add_edge("respond", END)

    if checkpointer is None:
        checkpointer = MemorySaver()

    return builder.compile(checkpointer=checkpointer)


# Module-level graph instance required by langgraph.json for LangGraph Studio.
graph = build_graph()


def run() -> None:
    conn = sqlite3.connect(str(CHECKPOINT_DB), check_same_thread=False)
    _graph    = build_graph(checkpointer=SqliteSaver(conn))
    thread_id = str(uuid4())
    config    = {"configurable": {"thread_id": thread_id}}

    print("=" * 55)
    print("  WealthDesk | Bharat National Bank")
    print("  Type 'quit' to exit")
    print("=" * 55)
    print(f"  Session: {thread_id[:8]}...")
    print("=" * 55)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nWealthDesk: Thank you for choosing Bharat National Bank. Goodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() == "quit":
            print("\nWealthDesk: Thank you for choosing Bharat National Bank. Goodbye!")
            break

        result = _graph.invoke(
            {"customer_message": user_input, "response": ""},
            config=config,
        )
        print(f"\nWealthDesk: {result['response']}")


if __name__ == "__main__":
    run()
