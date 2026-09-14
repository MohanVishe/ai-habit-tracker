"""LLM provider selection.

Three backends behind one factory. Groq is the default because it serves
open-weight models (Llama 3) fast and free; Ollama covers fully local; OpenAI
is there for comparison. Swapping providers is an env var, not a code change.
"""
from __future__ import annotations

import os

SUPPORTED = ("groq", "ollama", "openai")


def get_llm(temperature: float = 0.3):
    """Return a LangChain chat model.

    Temperature defaults low. This app asks the model to interpret numbers it
    has been given, not to invent anything — creativity here shows up as
    fabricated detail about the user's own history, which is the one failure
    mode that matters.
    """
    provider = os.getenv("LLM_PROVIDER", "groq").lower()

    if provider not in SUPPORTED:
        raise ValueError(f"LLM_PROVIDER must be one of {SUPPORTED}, got {provider!r}")

    if provider == "groq":
        from langchain_groq import ChatGroq

        if not os.getenv("GROQ_API_KEY"):
            raise RuntimeError("GROQ_API_KEY is not set. See .env.example.")
        return ChatGroq(
            model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            temperature=temperature,
        )

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=os.getenv("OLLAMA_MODEL", "llama3.1"),
            base_url=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
            temperature=temperature,
        )

    from langchain_openai import ChatOpenAI

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set. See .env.example.")
    return ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=temperature,
    )


def describe_provider() -> str:
    provider = os.getenv("LLM_PROVIDER", "groq").lower()
    model = {
        "groq": os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        "ollama": os.getenv("OLLAMA_MODEL", "llama3.1"),
        "openai": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    }.get(provider, "unknown")
    return f"{provider} · {model}"
