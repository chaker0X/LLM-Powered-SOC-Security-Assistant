"""Minimal LLM client with two interchangeable backends.

- ``ollama`` (default): local model served by Ollama, so no data leaves the machine.
- ``api``: any OpenAI-compatible endpoint. Optional and for development only.

Both expose the same ``chat()`` generator that yields text pieces as they are
produced, which lets the UI stream answers (important for speed on CPU).
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from typing import Any

import requests

from soc_assistant.config import load_config

Message = dict[str, str]  # {"role": "system"|"user"|"assistant", "content": "..."}


class LLMClient:
    def __init__(self, cfg: dict[str, Any] | None = None) -> None:
        self.cfg = cfg or load_config()["llm"]
        self.provider = self.cfg["provider"]
        if self.provider not in {"ollama", "api"}:
            raise ValueError(f"Unknown LLM provider: {self.provider!r}")

    @property
    def model(self) -> str:
        return self.cfg[self.provider]["model"]

    def chat(self, messages: list[Message], stream: bool = True,
             schema: dict | None = None) -> Iterator[str]:
        """`schema`: a JSON schema the answer must follow (constrained decoding), or None."""
        if self.provider == "ollama":
            yield from self._chat_ollama(messages, stream, schema)
        else:
            yield from self._chat_api(messages, stream, schema)

    def complete(self, messages: list[Message], schema: dict | None = None) -> str:
        """Convenience wrapper: return the full answer as one string."""
        return "".join(self.chat(messages, stream=False, schema=schema))

    # Backends

    def _chat_ollama(self, messages: list[Message], stream: bool, schema: dict | None) -> Iterator[str]:
        c = self.cfg["ollama"]
        payload = {
            "model": c["model"],
            "messages": messages,
            "stream": stream,
            "options": {"temperature": self.cfg["temperature"], "num_ctx": c["num_ctx"]},
        }
        if schema:
            payload["format"] = schema
        with requests.post(
            f"{c['base_url']}/api/chat", json=payload, stream=stream, timeout=self.cfg["timeout_s"]
        ) as r:
            r.raise_for_status()
            if not stream:
                yield r.json()["message"]["content"]
                return
            for line in r.iter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                if piece := chunk.get("message", {}).get("content"):
                    yield piece
                if chunk.get("done"):
                    break

    def _chat_api(self, messages: list[Message], stream: bool, schema: dict | None) -> Iterator[str]:
        c = self.cfg["api"]
        key = os.environ.get(c["api_key_env"])
        if not key:
            raise RuntimeError(f"Set the {c['api_key_env']} environment variable to use provider 'api'.")
        payload = {
            "model": c["model"],
            "messages": messages,
            "stream": stream,
            "temperature": self.cfg["temperature"],
        }
        if schema:
            payload["response_format"] = {"type": "json_schema",
                                          "json_schema": {"name": "answer", "schema": schema}}
        with requests.post(
            f"{c['base_url']}/chat/completions",
            json=payload,
            headers={"Authorization": f"Bearer {key}"},
            stream=stream,
            timeout=self.cfg["timeout_s"],
        ) as r:
            r.raise_for_status()
            if not stream:
                yield r.json()["choices"][0]["message"]["content"]
                return
            for line in r.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data: "):
                    continue
                data = line[len("data: "):]
                if data == "[DONE]":
                    break
                delta = json.loads(data)["choices"][0].get("delta", {})
                if piece := delta.get("content"):
                    yield piece
