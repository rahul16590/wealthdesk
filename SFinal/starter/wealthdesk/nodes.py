"""
wealthdesk/nodes.py
-------------------
Graph nodes for WealthDesk.

Session 2: the respond() node now carries conversation history across
turns so the LLM can refer back to earlier messages.
"""
from langchain_chroma import Chroma
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_huggingface import HuggingFaceEmbeddings

from .config import (
    SYSTEM_PROMPT, CLASSIFY_SYSTEM_PROMPT, ESCALATE_RESPONSE, DECLINE_RESPONSE,
    EMBED_MODEL, RETRIEVAL_K, VECTORSTORE_DIR,
)
from .state import WealthDeskState
from .tools import llm, classifier_llm

vectorstore = None  # shared across calls; initialised once by _init_vectorstore()


def _init_vectorstore() -> None:
    """Load ChromaDB + embeddings. No-op if already initialised."""
    global vectorstore
    if vectorstore is not None:
        return
    try:
        embeddings  = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
        vectorstore = Chroma(
            persist_directory=str(VECTORSTORE_DIR),
            embedding_function=embeddings,
        )
    except Exception as e:
        print(f"[WealthDesk] Could not load vectorstore: {e}")
        print("  Run 'python data/ingest.py' to create it.")


BLOCKLIST = [
    "ignore all previous",
    "forget everything",
    "you are now",
    "disregard your system",
    "act as",
    "jailbreak",
]

def classify(state: WealthDeskState) -> dict:
    """Call the LLM and return the agent's reply."""
    msg = state["customer_message"].strip()

    if any(phrase in msg.lower() for phrase in BLOCKLIST):
        return {"query_type": "OUT_OF_SCOPE", "retrieved_docs": []}

    if not msg or len(msg) < 10 or len(msg) > 500:
        return {"query_type": "OUT_OF_SCOPE", "retrieved_docs": []}

    messages = [
        SystemMessage(content=CLASSIFY_SYSTEM_PROMPT),
        HumanMessage(content=state["customer_message"]),
    ]

    try:
       result = classifier_llm.invoke(messages)
       query_type = result.content.strip().upper()
       if query_type not in {"SIMPLE","COMPLEX","OUT_OF_SCOPE"}:
          query_type = "SIMPLE"
    except Exception as e:
        print(f"[WealthDesk] Classification error: {e}")
        query_type = "SIMPLE"

    return {"query_type": query_type, "retrieved_docs": []}


def retrieve_docs(state: WealthDeskState) -> dict:
    """Query ChromaDB for policy chunks relevant to the customer's question."""
    _init_vectorstore()
    if vectorstore is None:
        return {"retrieved_docs": []}

    try:
        docs = vectorstore.similarity_search(state["customer_message"], k=RETRIEVAL_K)
        retrieved = [
            f"[{doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
            for doc in docs
        ]
    except Exception as e:
        print(f"[WealthDesk] Retrieval error: {e}")
        retrieved = []

    return {"retrieved_docs": retrieved}


def respond(state: WealthDeskState) -> dict:
    """Call the LLM with full conversation history and return the reply."""
    history   = state.get("history", [])
    retrieved = state.get("retrieved_docs", [])

    if retrieved:
        context_block  = "\n\n---\n\n".join(retrieved)
        system_content = (
            SYSTEM_PROMPT
            + "\n\nThe following sections from BNB's policy documents are relevant "
            "to the customer's question. Use this information in your answer:\n\n"
            + context_block
        )
    else:
        system_content = SYSTEM_PROMPT

    messages = [SystemMessage(content=system_content)]
    for turn in history:
        if turn["role"] == "user":
            messages.append(HumanMessage(content=turn["content"]))
        else:
            messages.append(AIMessage(content=turn["content"]))
    messages.append(HumanMessage(content=state["customer_message"]))

    try:
        result = llm.invoke(messages)
        response_text = result.content
    except Exception as e:
        print(f"[WealthDesk] LLM error: {e}")
        response_text = "I am temporarily unavailable. Please try again in a moment."

    new_history = history + [
        {"role": "user", "content": state["customer_message"]},
        {"role": "assistant", "content": response_text},
    ]
    return {"response": response_text, "history": new_history}


def escalate(state: WealthDeskState) -> dict:
    new_history = state.get("history", []) + [
        {"role": "user",      "content": state["customer_message"]},
        {"role": "assistant", "content": ESCALATE_RESPONSE},
    ]
    return {"response": ESCALATE_RESPONSE, "history": new_history}
 
def decline(state: WealthDeskState) -> dict:
    new_history = state.get("history", []) + [
        {"role": "user",      "content": state["customer_message"]},
        {"role": "assistant", "content": DECLINE_RESPONSE},
    ]
    return {"response": DECLINE_RESPONSE, "history": new_history}
 
def route_query(state: WealthDeskState)->str:
   query_type = state.get("query_type","SIMPLE")
   if query_type == "COMPLEX":
      return "escalate"
   if query_type == "OUT_OF_SCOPE":
      return "decline"
   return "retrieve_docs"