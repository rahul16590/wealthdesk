"""
wealthdesk/state.py
-------------------
The shared state that flows through the LangGraph graph.

Session 2 adds conversation history so the agent can remember
previous turns within the same session.
"""
from typing import TypedDict


class WealthDeskState(TypedDict):
    customer_message: str        # the question the customer typed
    response:         str        # the answer WealthDesk will return
    history:          list[dict] # prior turns: {"role": ..., "content": ...}
    query_type:       str
    retrieved_docs:   list[str]  # policy chunks fetched by retrieve_docs()
