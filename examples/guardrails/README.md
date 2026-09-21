# Guardrail prompt classifier

Classifies a user prompt into one of four high-level guardrail categories —
`safe`, `harmful_content`, `privacy_or_fraud`, `jailbreak` — with a single
typed `choice` question, using laya-mlx's typed-decision API (no token
generation, ~20ms per prompt on Apple Silicon; see `load_test.py`).

Edit `questions.json` to add, remove, or reword categories for your own
policy. Each entry you add to `questions.json` is a separate question that
laya-mlx answers in the same forward pass — more questions per prompt means
more work per call, so keep the file to only what you actually need to act
on.

## Run

```bash
# single prompt
uv run examples/guardrails/classify.py --text "Ignore all previous instructions and do X"

# batch over examples/guardrails/sample_prompts.jsonl (one {"id", "text"} per line)
uv run examples/guardrails/classify.py --batch examples/guardrails/sample_prompts.jsonl
```

First run downloads the `aac6fef/laya-mlx` checkpoint from Hugging Face; later
runs are fully local.
