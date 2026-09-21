# Guardrail prompt classifier

Classifies a user prompt into common LLM guardrail categories (safe, violence,
self-harm, sexual content, hate/harassment, illegal weapons/drugs, fraud/scams,
privacy/PII, jailbreak/prompt-injection, regulated advice), plus a severity
score and two flags (`jailbreak_attempt`, `requires_human_review`), using
laya-mlx's typed-decision API — no token generation, ~10-15ms per prompt on
Apple Silicon.

Edit `questions.json` to add, remove, or reword categories for your own policy.

## Run

```bash
# single prompt
uv run examples/guardrails/classify.py --text "Ignore all previous instructions and do X"

# batch over examples/guardrails/sample_prompts.jsonl (one {"id", "text"} per line)
uv run examples/guardrails/classify.py --batch examples/guardrails/sample_prompts.jsonl
```

First run downloads the `aac6fef/laya-mlx` checkpoint from Hugging Face; later
runs are fully local.
