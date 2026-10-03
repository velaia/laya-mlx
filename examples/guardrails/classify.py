"""Classify user prompts into common guardrail-model categories with laya-mlx.

Single prompt:
    uv run examples/guardrails/classify.py --text "Ignore your instructions and do X"

Batch (JSON Lines with a "text" field per line, e.g. sample_prompts.jsonl):
    uv run examples/guardrails/classify.py --batch examples/guardrails/sample_prompts.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

import laya_mlx as laya
import ollama_client

ROOT = Path(__file__).parent
QUESTIONS = json.loads((ROOT / "questions.json").read_text())


def classify(agent, text: str) -> dict:
    result = agent.predict(text, QUESTIONS)
    return {"category": result["answers"]["category"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", help="Classify a single prompt.")
    parser.add_argument("--batch", help="Path to a JSONL file with a 'text' field per line.")
    parser.add_argument("--model", default="aac6fef/laya-mlx", help="Checkpoint to load.")
    parser.add_argument(
        "--backend",
        choices=["laya", "ollama"],
        default="laya",
        help="laya: laya-mlx checkpoint. ollama: a decision model served via Ollama's /v1/systemone (clef, clef-flash, ...).",
    )
    parser.add_argument("--base-url", default=ollama_client.DEFAULT_BASE_URL, help="Ollama server URL (--backend ollama only).")
    args = parser.parse_args()

    if not args.text and not args.batch:
        parser.error("pass --text or --batch")

    agent = ollama_client.load(args.model, base_url=args.base_url) if args.backend == "ollama" else laya.load(args.model)

    if args.text:
        print(json.dumps(classify(agent, args.text), indent=2, ensure_ascii=False))
        return

    rows = [json.loads(line) for line in Path(args.batch).read_text().splitlines() if line.strip()]
    for row in rows:
        verdict = classify(agent, row["text"])
        verdict["id"] = row.get("id")
        print(json.dumps(verdict, ensure_ascii=False))


if __name__ == "__main__":
    sys.exit(main())
