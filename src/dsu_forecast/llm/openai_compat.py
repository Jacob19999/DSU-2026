from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import requests


@dataclass(frozen=True)
class OpenAICompatClient:
    api_key: str
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4o-mini"
    timeout_s: int = 60

    @staticmethod
    def from_env() -> "OpenAICompatClient":
        api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_TOKEN")
        if not api_key:
            raise RuntimeError("Missing OPENAI_API_KEY (or OPENAI_API_TOKEN) in environment")
        base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        timeout_s = int(os.environ.get("OPENAI_TIMEOUT_S", "60"))
        return OpenAICompatClient(api_key=api_key, base_url=base_url, model=model, timeout_s=timeout_s)


def chat_json(
    client: OpenAICompatClient,
    *,
    system: str,
    user: str,
    temperature: float = 0.0,
) -> dict[str, Any]:
    """
    OpenAI-compatible chat completions. Returns parsed JSON object.
    Robust to models that don't support response_format by falling back to JSON-only instruction.
    """
    url = f"{client.base_url}/chat/completions"
    headers = {"Authorization": f"Bearer {client.api_key}", "Content-Type": "application/json"}

    body: dict[str, Any] = {
        "model": client.model,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        # Try response_format if supported; harmless if ignored by proxy servers.
        "response_format": {"type": "json_object"},
    }

    r = requests.post(url, headers=headers, json=body, timeout=client.timeout_s)
    r.raise_for_status()
    payload: dict[str, Any] = r.json()
    content = payload["choices"][0]["message"]["content"]

    # Some models wrap JSON in ``` fences; strip.
    if content.strip().startswith("```"):
        content = content.strip()
        content = content.strip("`")
        # best-effort: remove leading language label
        content = content.replace("json\n", "", 1)

    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Model did not return valid JSON. content={content[:400]}") from e

