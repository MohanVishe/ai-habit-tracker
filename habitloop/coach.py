"""The chat coach.

Grounding here is structural rather than retrieval-based, and that is a
deliberate choice worth explaining.

A habit log is small — a few hundred rows. Chunking and embedding it would add
a vector store, an embedding model, and a whole class of retrieval failures, in
exchange for nothing: the entire *computed summary* fits in the context window
several times over. So the coach gets the complete picture every turn and there
is no retrieval step to get wrong.

What the model never sees is raw rows. It sees statistics computed in Python.
Asking a language model to count consecutive days across 300 records is asking
it to do the thing it is least reliable at, and it will produce a confident
wrong number rather than an error.
"""
from __future__ import annotations

import json

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

from .llm import get_llm
from .prompts import COACH


def answer(question: str, summary: dict, history: list[tuple[str, str]] | None = None) -> str:
    """Answer one question against the user's own statistics.

    `history` is a list of (role, text) pairs where role is 'user' or 'assistant'.
    """
    messages = []
    for role, text in (history or [])[-10:]:
        messages.append(HumanMessage(text) if role == "user" else AIMessage(text))

    prompt = ChatPromptTemplate.from_messages([
        ("system", COACH),
        MessagesPlaceholder("history"),
        ("human", "{question}"),
    ])

    chain = prompt | get_llm(temperature=0.2) | StrOutputParser()

    return chain.invoke({
        "summary": json.dumps(summary, indent=2, sort_keys=True),
        "history": messages,
        "question": question,
    })
