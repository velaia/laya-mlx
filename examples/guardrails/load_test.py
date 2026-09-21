"""Throughput/latency load test for the guardrail classifier.

Each classify() call is one agent.predict() covering all 4 questions in
questions.json (one forward pass, since 4 <= batch_size). Prompts are
classified sequentially, one at a time, which matches a synchronous API
usage pattern; this is not a concurrency test.

Usage:
    uv run examples/guardrails/load_test.py --num 300
    uv run examples/guardrails/load_test.py --num 300 --dtype float16 --batch-size 16 --compile --cache-prompts
"""

import argparse
import itertools
import json
import statistics
import time
from pathlib import Path

import laya_mlx as laya

ROOT = Path(__file__).parent
QUESTIONS = json.loads((ROOT / "questions.json").read_text())
PROMPTS = [
    json.loads(line)["text"]
    for line in (ROOT / "sample_prompts.jsonl").read_text().splitlines()
    if line.strip()
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="aac6fef/laya-mlx")
    parser.add_argument("--num", type=int, default=200, help="Total prompts to classify.")
    parser.add_argument("--warmup", type=int, default=10, help="Untimed warmup calls.")
    parser.add_argument("--dtype", default="float16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--cache-prompts", action="store_true")
    parser.add_argument("--device", choices=["gpu", "cpu"], default=None)
    args = parser.parse_args()

    agent = laya.load(
        args.model,
        dtype=args.dtype,
        batch_size=args.batch_size,
        compile=args.compile,
        cache_prompts=args.cache_prompts,
        device=args.device,
    )

    prompt_cycle = itertools.cycle(PROMPTS)
    for _ in range(args.warmup):
        agent.predict(next(prompt_cycle), QUESTIONS)

    latencies = []
    start = time.perf_counter()
    for _ in range(args.num):
        text = next(prompt_cycle)
        t0 = time.perf_counter()
        agent.predict(text, QUESTIONS)
        latencies.append(time.perf_counter() - t0)
    total = time.perf_counter() - start

    latencies.sort()
    p50 = latencies[len(latencies) // 2]
    p95 = latencies[int(len(latencies) * 0.95)]
    p99 = latencies[min(len(latencies) - 1, int(len(latencies) * 0.99))]

    print(f"model:            {args.model}")
    print(f"dtype:            {args.dtype}")
    print(f"batch_size:       {args.batch_size}")
    print(f"compile:          {args.compile}")
    print(f"cache_prompts:    {args.cache_prompts}")
    print(f"prompts:          {args.num}  (warmup: {args.warmup})")
    print(f"total time:       {total:.3f}s")
    print(f"throughput:       {args.num / total:.1f} prompts/s")
    print(f"mean latency:     {statistics.mean(latencies) * 1000:.2f} ms")
    print(f"p50 latency:      {p50 * 1000:.2f} ms")
    print(f"p95 latency:      {p95 * 1000:.2f} ms")
    print(f"p99 latency:      {p99 * 1000:.2f} ms")


if __name__ == "__main__":
    main()
