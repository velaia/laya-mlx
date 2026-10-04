"""Minimal client for the System One `/v1/systemone` endpoint.

Served locally by Ollama (Clef, Clef-Flash, ...) and remotely by OpenRouter
(TypeSafe's Jev). Not yet supported by the official `ollama` Python/JS
libraries, so we hit the REST endpoint directly. Mirrors the subset of
laya_mlx.Agent's interface our guardrail scripts use: `.predict(state,
questions)` -> same response shape laya_mlx returns (`{"answers": {...},
"usage": {...}}`).
"""

import os

import requests

DEFAULT_BASE_URL = "http://localhost:11434"
OPENROUTER_BASE_URL = "https://openrouter.ai/api"


class OllamaDecisionAgent:
    def __init__(self, model: str, base_url: str = DEFAULT_BASE_URL, timeout: float = 120.0, api_key: str | None = None):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        if api_key:
            self.session.headers["Authorization"] = f"Bearer {api_key}"

    def predict(self, state: str, questions: dict) -> dict:
        resp = self.session.post(
            f"{self.base_url}/v1/systemone",
            json={"model": self.model, "state": state, "questions": questions},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()

    # alias used by some of our scripts
    system_one = predict


def load(model: str, base_url: str = DEFAULT_BASE_URL) -> OllamaDecisionAgent:
    return OllamaDecisionAgent(model, base_url=base_url)


def load_backend(backend: str, model: str, base_url: str | None = None) -> OllamaDecisionAgent:
    """backend: "ollama" (local server) or "openrouter" (needs OPENROUTER_API_KEY)."""
    if backend == "openrouter":
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise SystemExit("--backend openrouter needs the OPENROUTER_API_KEY environment variable")
        return OllamaDecisionAgent(model, base_url=base_url or OPENROUTER_BASE_URL, api_key=api_key)
    return OllamaDecisionAgent(model, base_url=base_url or DEFAULT_BASE_URL)
