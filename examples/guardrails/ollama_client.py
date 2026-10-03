"""Minimal client for Ollama's /v1/systemone endpoint (Clef, Clef-Flash, Jev, ...).

Not yet supported by the official `ollama` Python/JS libraries, so we hit the
REST endpoint directly. Mirrors the subset of laya_mlx.Agent's interface our
guardrail scripts use: `.predict(state, questions)` -> same response shape
laya_mlx returns (`{"answers": {...}, "usage": {...}}`).
"""

import requests

DEFAULT_BASE_URL = "http://localhost:11434"


class OllamaDecisionAgent:
    def __init__(self, model: str, base_url: str = DEFAULT_BASE_URL, timeout: float = 120.0):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def predict(self, state: str, questions: dict) -> dict:
        resp = requests.post(
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
