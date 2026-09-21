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
