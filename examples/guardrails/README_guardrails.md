# Guardrail classifier — throughput findings

Measured and estimated throughput for the single-question guardrail
classifier (`category`: `safe` / `harmful_content` / `privacy_or_fraud` /
`jailbreak`, one `choice` question, ~117 input tokens per prompt) defined in
`questions.json`.

## Measured (laya-mlx, MLX, `aac6fef/laya-mlx`, fp16)

| Hardware | Mode | Throughput |
|---|---|---:|
| Apple M1 Max (10-core, 64GB) | 1 process | 49.6 prompts/s |
| Apple M1 Max | 2 processes | 64.2 prompts/s |
| Apple M1 Max | 4 processes | 67.4 prompts/s |
| Apple M5 | 1 process | **117.2 prompts/s** |

`load_test.py` measures single-process throughput; `load_test_concurrent.py`
fans work across threads/processes. On M1 Max, throughput plateaus past 2
workers because the workload is GPU-bound — one Metal device serializes work
regardless of how many processes/threads submit it.

For reference, an earlier version of this example asked 4 separate questions
per prompt (`category`, `severity`, `jailbreak_attempt`,
`requires_human_review`) instead of just `category`. Each question is its
own item in the batched forward pass — not an extra output of one question —
so that version measured ~16.0 prompts/s single-process on M1 Max, about 3x
slower than the trimmed single-question version above for the same hardware.

## Comparison: Cloudflare Clef / Clef-Flash (via Ollama) vs. laya-mlx

Ollama added two decision models with the same `state` + typed `questions` ->
`answers` API as laya ([`/v1/systemone`](https://ollama.com/library/clef-flash)):
[`clef-flash`](https://ollama.com/library/clef-flash) (9B, fine-tuned from
Qwen3.5-9B) and [`clef`](https://ollama.com/library/clef) (27B), both from
Cloudflare, Apache 2.0. They're wire-compatible with our `questions.json`
schema, so `examples/guardrails/ollama_client.py` talks to them directly —
no prompt engineering needed. Both measured on the same M1 Max above.

Cloudflare's own published benchmarks (BFCL, API-Bank, BANKING77, CLINC150,
ANLI, RouterBench, When2Call, and 4 business-workflow decision tasks) don't
cover safety/guardrail classification, so they don't substitute for our own
dataset evaluation below — we ran it ourselves.

### Throughput (single-process, sequential requests)

| Model | Params | Backend | Throughput | vs. laya-mlx |
|---|---:|---|---:|---:|
| laya-mlx (`aac6fef/laya-mlx`) | 421M | MLX, native | 49.6 prompts/s | 1x |
| clef-flash | 9B | Ollama `/v1/systemone` | 1.8 prompts/s | ~28x slower |
| clef | 27B | Ollama `/v1/systemone` | 0.5 prompts/s | ~99x slower |

Cloudflare publishes 38.8ms median / 122.4ms p95 latency for clef-flash and
209.3ms / 238.6ms for clef — roughly 15x faster than what we measured here.
That gap is a hardware difference (their serving infra vs. this M1 Max via
Ollama/Metal), not a discrepancy in the model; the clef-vs-clef-flash speed
*ratio* we measured (~3.5x) is close to their published ~5.4x. As with the
T4/Xeon estimates above, both clef models are GPU-bound on a single device —
4-thread concurrency against the Ollama server does not raise the aggregate
ceiling.

### Accuracy (identical random 150-row sample per dataset, seed=42)

| Dataset | Metric | laya-mlx | clef-flash | clef |
|---|---|---:|---:|---:|
| wildguardmix | accuracy | 0.398 | 0.647 | **0.729** |
| wildguardmix | jailbreak recall | 0.176 | 0.426 | **0.574** |
| aegis-2.0 | accuracy | 0.619 | 0.782 | **0.810** |
| deepset-prompt-injections | jailbreak recall / F1 | 0.150 / 0.257 | 0.267 / 0.421 | **0.350 / 0.519** |
| jbb-behaviors | accuracy | **0.694** | 0.686 | 0.636 |

Takeaways:

- Both clef models are substantially more accurate than laya-mlx on 3 of 4
  datasets, and meaningfully better at the thing laya is weakest at —
  jailbreak detection — though none of the three is good at it in absolute
  terms (recall tops out at 0.574 for the 27B model).
- Accuracy scales with size (clef > clef-flash > laya-mlx) everywhere except
  jbb-behaviors, where laya-mlx edges out both — clef specifically
  over-predicts `harmful_content` on that dataset's `privacy_or_fraud` rows.
  This is a real result from the data, not a measurement artifact.
- The accuracy gain is expensive: clef-flash costs ~28x the latency of
  laya-mlx, clef ~99x. Whether that trade is worth it depends on whether a
  deployment is latency-bound or accuracy-bound.
- An earlier pass sampled the first N rows of each dataset instead of a
  random sample and got misleading numbers (e.g. laya-mlx's wildguardmix
  jailbreak recall came out at 54% instead of ~18%) because these datasets
  are grouped by subcategory, not shuffled. `evaluate.py --limit N` now
  draws a seeded random sample instead of taking a head slice.

### Reproducing

```bash
ollama pull clef-flash
ollama pull clef

uv run --with requests examples/guardrails/load_test.py --model clef-flash --backend ollama --num 100
uv run --with requests examples/guardrails/load_test_concurrent.py --model clef-flash --backend ollama --mode thread --workers 4 --num 80

uv run --with pandas --with pyarrow --with requests examples/guardrails/evaluate.py --model clef-flash --backend ollama --limit 150
uv run --with pandas --with pyarrow --with requests examples/guardrails/evaluate.py --model clef --backend ollama --limit 150
uv run --with pandas --with pyarrow --with requests examples/guardrails/evaluate.py --model aac6fef/laya-mlx --backend laya --limit 150
```

## Estimated — other hardware (not measured)

`laya-mlx` is MLX-only (Apple Silicon); it does not run on NVIDIA GPUs or
x86 CPUs. The numbers below apply to the upstream PyTorch/Transformers
project ([`NandhaKishorM/laya`](https://github.com/NandhaKishorM/laya)) on
that hardware, and are otherwise unverified.

### NVIDIA T4 — published upstream benchmark

Upstream publishes single-question latency on a T4, which maps directly
onto our current one-question-per-prompt workload:

| Checkpoint | Latency | Throughput |
|---|---:|---:|
| English (421M) | 39.5 ms | ~25.3 prompts/s |
| Multilingual (322M) | 32.8 ms | ~30.5 prompts/s |

Sources: [NandhaKishorM/laya](https://github.com/NandhaKishorM/laya),
[laya.convaiinnovations.com](https://laya.convaiinnovations.com/).

### Xeon (2025-class), single core — reasoned estimate

No published benchmark exists for this. Estimated via
FLOPs ≈ 2 × params × tokens ≈ 2 × 421M × 117 ≈ **98.5 GFLOPs/prompt**,
calibrated against the real T4 latency above (98.5 GFLOPs / 39.5 ms implies
~2.5 TFLOPS/s effective T4 utilization — a plausible figure, which is why
the same method is used for the CPU estimate below).

| Runtime path | Effective throughput | Est. prompts/s |
|---|---:|---:|
| Plain AVX-512 FP32 (PyTorch/oneDNN, no AMX) | ~40-80 GFLOPS/s | ~0.4-0.8 |
| AMX BF16 (Sapphire/Granite Rapids + ONNX Runtime/IPEX, tuned) | ~200-500 GFLOPS/s | ~2-5 |

Batch-1 latency rarely saturates AMX's fixed tile shapes well, which is why
even the optimized path tops out in the low single digits per core.

### Xeon (2025-class), 8 vCPUs + batching — reasoned estimate

Batching improves both per-core efficiency (fuller AMX/vector utilization)
and core scaling, at the cost of added per-prompt latency (a batch has to
fill before it runs) — the opposite tradeoff from the GPU/MLX results above.

| Runtime path | 8-core aggregate (~75-85% scaling efficiency) | Est. prompts/s |
|---|---:|---:|
| Plain AVX-512 FP32, threaded, no AMX | ~400-650 GFLOPS/s | ~4-7 |
| AMX BF16, ONNX Runtime/IPEX, batch 16-32, tuned | ~2.5-6 TFLOPS/s | ~25-60 |

The ~8-10x spread between the two Xeon paths is entirely about whether the
deployment actually engages AMX through a quantized/batched runtime, versus
running vanilla `transformers`/PyTorch on the same cores.

## Caveats

- T4 figures are real, published upstream numbers and apply directly since
  the workload is now exactly one question per prompt (no extrapolation).
- Xeon figures are order-of-magnitude estimates from a FLOPs model, not
  measurements. Actual numbers depend heavily on runtime (PyTorch vs. ONNX
  Runtime/IPEX), quantization, batch size, and thread/NUMA tuning.
- No multi-process/concurrency numbers exist yet for M5 — only the
  single-process figure above has been measured there.

## Reproducing the measured numbers

```bash
uv run examples/guardrails/load_test.py --num 300
uv run examples/guardrails/load_test_concurrent.py --mode process --workers 4 --num 800
```
