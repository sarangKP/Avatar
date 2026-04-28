"""
llm_wrapper.py
--------------
Thin LLM wrapper that streams sentence-by-sentence for low latency.
Supports:
  - Ollama  (local, zero-cost)
  - OpenAI / any OpenAI-compatible API  (e.g. Together, Groq, vLLM)
  - Echo    (no-op for testing pipeline without an LLM)

KEY CHANGE: All backends now return an Iterator[str] of sentences rather
than a single complete string.  This lets live_avatar_pipeline.py feed the
first sentence to TTS+FlashHead while the LLM is still generating the rest,
cutting perceived latency by 50–70 % on longer responses.

Install for Ollama:   https://ollama.com  then  ollama pull llama3
Install for OpenAI:   pip install openai

Usage
~~~~~
    from llm_wrapper import build_llm

    llm = build_llm(backend="ollama", model="llama3.2")
    for sentence in llm("What is the capital of France?"):
        print(sentence)   # streamed, sentence by sentence
"""

from __future__ import annotations

import re
from typing import Callable, Iterator

# ---------------------------------------------------------------------------
# System prompt — tuned for short sentences to minimise FlashHead chunk count
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a helpful, friendly AI avatar speaking directly to the user. "
    "Use short sentences of 10 words or fewer. "
    "Never join two clauses with a comma — always end the first clause with "
    "a period and start a new sentence. "
    "Be warm, conversational, and speak in first person."
)

# ---------------------------------------------------------------------------
# Sentence splitter
# ---------------------------------------------------------------------------
_SENTENCE_END = re.compile(r'(?<=[.!?])\s+')


def _flush_sentences(buffer: str) -> tuple[list[str], str]:
    """
    Split *buffer* on sentence boundaries.
    Returns (complete_sentences, leftover_fragment).
    """
    parts = _SENTENCE_END.split(buffer)
    if len(parts) == 1:
        # No sentence boundary found yet — keep buffering
        return [], buffer
    # Everything except the last fragment is a complete sentence
    complete = [p.strip() for p in parts[:-1] if p.strip()]
    leftover = parts[-1]
    return complete, leftover


# ---------------------------------------------------------------------------
# Backend implementations  (all return Iterator[str])
# ---------------------------------------------------------------------------

def _echo_llm(prompt: str) -> Iterator[str]:
    """Yield the prompt split into sentences — useful for testing without an LLM."""
    parts = _SENTENCE_END.split(prompt.strip())
    for part in parts:
        part = part.strip()
        if part:
            yield part


def _ollama_llm(model: str, host: str = "http://localhost:11434") -> Callable[[str], Iterator[str]]:
    """
    Streams tokens from Ollama's /api/chat endpoint and yields complete
    sentences as soon as a sentence boundary is detected in the token stream.
    No extra libraries required beyond `requests`.
    """
    import requests
    import json

    def call(prompt: str) -> Iterator[str]:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            "stream": True,
        }

        buffer = ""
        with requests.post(
            f"{host}/api/chat",
            json=payload,
            stream=True,
            timeout=60,
        ) as resp:
            resp.raise_for_status()
            for raw_line in resp.iter_lines():
                if not raw_line:
                    continue
                try:
                    data = json.loads(raw_line)
                except json.JSONDecodeError:
                    continue

                token = data.get("message", {}).get("content", "")
                buffer += token

                sentences, buffer = _flush_sentences(buffer)
                for sentence in sentences:
                    yield sentence

                # Ollama sets done=True on the final message
                if data.get("done"):
                    break

        # Flush any trailing fragment that lacked a terminal punctuation mark
        leftover = buffer.strip()
        if leftover:
            yield leftover

    return call


def _openai_llm(
    model: str,
    api_key: str | None,
    base_url: str | None,
) -> Callable[[str], Iterator[str]]:
    """
    Streams tokens from any OpenAI-compatible API and yields sentences.
    Requires: pip install openai
    """
    from openai import OpenAI

    client = OpenAI(
        api_key=api_key,
        base_url=base_url,   # None → default OpenAI endpoint
    )

    def call(prompt: str) -> Iterator[str]:
        buffer = ""
        stream = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            buffer += delta

            sentences, buffer = _flush_sentences(buffer)
            for sentence in sentences:
                yield sentence

        # Flush trailing fragment
        leftover = buffer.strip()
        if leftover:
            yield leftover

    return call


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

def build_llm(
    backend: str = "echo",
    model: str = "llama3.2",
    api_key: str | None = None,
    base_url: str | None = None,
    ollama_host: str = "http://localhost:11434",
) -> Callable[[str], Iterator[str]]:
    """
    Factory — returns a callable that takes a prompt string and yields
    sentences one-by-one as the LLM streams its response.

    Parameters
    ----------
    backend    : 'echo' | 'openai' | 'ollama'
    model      : model name (ignored for 'echo')
    api_key    : OpenAI API key (or set OPENAI_API_KEY env var)
    base_url   : override API base URL (e.g. 'https://api.groq.com/openai/v1')
    ollama_host: base URL for local Ollama server
    """
    if backend == "echo":
        print("[LLM] Using echo backend (no LLM, streaming sentences)")
        return _echo_llm

    elif backend == "ollama":
        print(f"[LLM] Using Ollama streaming backend  model={model}  host={ollama_host}")
        return _ollama_llm(model=model, host=ollama_host)

    elif backend == "openai":
        import os
        key = api_key or os.environ.get("OPENAI_API_KEY")
        print(f"[LLM] Using OpenAI-compatible streaming backend  model={model}  base_url={base_url or 'default'}")
        return _openai_llm(model=model, api_key=key, base_url=base_url)

    else:
        raise ValueError(f"Unknown LLM backend: {backend!r}. Choose echo | openai | ollama")