"""Thin LLM wrapper supporting Anthropic and OpenAI with structured JSON output."""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Optional


def _log(msg: str):
    print(msg, file=sys.stderr, flush=True)


@dataclass
class LLMConfig:
    provider: str = "anthropic"
    model: str = "claude-sonnet-4-20250514"
    max_retries: int = 3
    timeout: int = 30
    max_output_tokens: int = 2000
    reasoning_effort: str | None = None


_OPENAI_MAX_COMPLETION_TOKENS_MODELS = ("o1", "o3", "o4", "gpt-5")


class LLMClient:
    """Unified LLM client supporting Anthropic and OpenAI."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self._client = None
        self._init_client()

    def _init_client(self):
        if self.config.provider == "anthropic":
            import anthropic
            self._client = anthropic.Anthropic(
                api_key=os.environ.get("ANTHROPIC_API_KEY"),
                timeout=self.config.timeout,
            )
        elif self.config.provider == "openai":
            import openai
            self._client = openai.OpenAI(
                api_key=os.environ.get("OPENAI_API_KEY"),
                timeout=self.config.timeout,
            )
        else:
            raise ValueError(f"Unknown provider: {self.config.provider}")

    def ask(self, prompt: str, system: str = "", json_mode: bool = False) -> str:
        """Send a prompt and get a text response. If json_mode, request JSON output."""
        for attempt in range(self.config.max_retries):
            try:
                start = time.time()
                response = self._call(prompt, system, json_mode)
                elapsed = time.time() - start
                _log(f"LLM call ({self.config.provider}/{self.config.model}): {elapsed:.1f}s, {len(response)} chars")
                return response
            except Exception as e:
                _log(f"LLM error (attempt {attempt + 1}/{self.config.max_retries}): {e}")
                if attempt == self.config.max_retries - 1:
                    raise
                time.sleep(1 * (attempt + 1))
        return ""

    def ask_json(self, prompt: str, system: str = "") -> dict | list:
        """Send a prompt and parse JSON from the response."""
        response = self.ask(prompt, system, json_mode=True)
        return self._extract_json(response)

    def send(self, messages: list[dict], system: str = "", json_mode: bool = False) -> str:
        """Send a multi-turn conversation and get a response."""
        for attempt in range(self.config.max_retries):
            try:
                start = time.time()
                response = self._call_multi(messages, system, json_mode)
                elapsed = time.time() - start
                _log(f"LLM call ({self.config.provider}/{self.config.model}): {elapsed:.1f}s, {len(response)} chars")
                return response
            except Exception as e:
                _log(f"LLM error (attempt {attempt + 1}/{self.config.max_retries}): {e}")
                if attempt == self.config.max_retries - 1:
                    raise
                time.sleep(1 * (attempt + 1))
        return ""

    def send_json(self, messages: list[dict], system: str = "") -> dict | list:
        """Send multi-turn and parse JSON response."""
        response = self.send(messages, system, json_mode=True)
        return self._extract_json(response)

    def _call(self, prompt: str, system: str, json_mode: bool) -> str:
        if self.config.provider == "anthropic":
            return self._call_anthropic(prompt, system, json_mode)
        else:
            return self._call_openai(prompt, system, json_mode)

    def _call_multi(self, messages: list[dict], system: str, json_mode: bool) -> str:
        if self.config.provider == "anthropic":
            return self._call_anthropic_multi(messages, system, json_mode)
        else:
            return self._call_openai_multi(messages, system, json_mode)

    def _call_anthropic(self, prompt: str, system: str, json_mode: bool) -> str:
        kwargs = {
            "model": self.config.model,
            "max_tokens": self.config.max_output_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system
        response = self._client.messages.create(**kwargs)
        return response.content[0].text

    def _openai_extra_params(self) -> dict:
        """Return model-specific OpenAI parameters."""
        n = self.config.max_output_tokens
        if any(self.config.model.startswith(p) for p in _OPENAI_MAX_COMPLETION_TOKENS_MODELS):
            params = {"max_completion_tokens": n}
        else:
            params = {"max_tokens": n}
        if self.config.reasoning_effort:
            params["reasoning_effort"] = self.config.reasoning_effort
        return params

    def _call_openai(self, prompt: str, system: str, json_mode: bool) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        kwargs = {
            "model": self.config.model,
            "messages": messages,
            **self._openai_extra_params(),
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        response = self._client.chat.completions.create(**kwargs)
        return response.choices[0].message.content

    def _call_anthropic_multi(self, messages: list[dict], system: str, json_mode: bool) -> str:
        kwargs = {
            "model": self.config.model,
            "max_tokens": self.config.max_output_tokens,
            "messages": messages,
        }
        if system:
            kwargs["system"] = system
        response = self._client.messages.create(**kwargs)
        return response.content[0].text

    def _call_openai_multi(self, messages: list[dict], system: str, json_mode: bool) -> str:
        api_messages = []
        if system:
            api_messages.append({"role": "system", "content": system})
        api_messages.extend(messages)
        kwargs = {
            "model": self.config.model,
            "messages": api_messages,
            **self._openai_extra_params(),
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        response = self._client.chat.completions.create(**kwargs)
        return response.choices[0].message.content

    @staticmethod
    def _extract_json(text: str) -> dict | list:
        """Extract JSON from LLM response, handling markdown code blocks."""
        text = text.strip()
        # Try direct parse first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        # Try extracting from markdown code block
        if "```" in text:
            blocks = text.split("```")
            for block in blocks[1::2]:  # odd-indexed blocks are inside backticks
                block = block.strip()
                if block.startswith("json"):
                    block = block[4:].strip()
                try:
                    return json.loads(block)
                except json.JSONDecodeError:
                    continue
        # Last resort: find first { or [ and parse from there
        for i, ch in enumerate(text):
            if ch in ('{', '['):
                try:
                    return json.loads(text[i:])
                except json.JSONDecodeError:
                    continue
        raise ValueError(f"Could not extract JSON from LLM response: {text[:200]}")
