"""Multi-threaded / multi-process throughput test for the guardrail classifier.

Single-process throughput is measured by load_test.py. This script fans work
out across several workers to see how much aggregate throughput you get from
the rest of your cores, using either:

  --mode thread    N threads share ONE Agent/model instance.
  --mode process   N processes each load their OWN Agent/model instance
                    (more memory, but avoids any single-instance contention).

Usage:
    uv run examples/guardrails/load_test_concurrent.py --workers 4 --mode process --num 400
    uv run examples/guardrails/load_test_concurrent.py --workers 4 --mode thread --num 400
"""

import argparse
import itertools
import json
import multiprocessing as mp
import statistics
import threading
import time
from pathlib import Path

ROOT = Path(__file__).parent
QUESTIONS = json.loads((ROOT / "questions.json").read_text())
PROMPTS = [
    json.loads(line)["text"]
    for line in (ROOT / "sample_prompts.jsonl").read_text().splitlines()
    if line.strip()
]


def report(label, results, total_num):
    starts = [r["start"] for r in results]
    ends = [r["end"] for r in results]
    elapsed = max(ends) - min(starts)
    all_latencies = sorted(lat for r in results for lat in r["latencies"])
    n = len(all_latencies)
    p50 = all_latencies[n // 2]
    p95 = all_latencies[int(n * 0.95)]
    p99 = all_latencies[min(n - 1, int(n * 0.99))]

    print(f"mode:             {label}")
    print(f"workers:          {len(results)}")
    print(f"prompts:          {total_num}")
    print(f"wall time:        {elapsed:.3f}s")
    print(f"aggregate:        {total_num / elapsed:.1f} prompts/s")
    print(f"per-worker mean:  {(total_num / elapsed) / len(results):.1f} prompts/s")
    print(f"mean latency:     {statistics.mean(all_latencies) * 1000:.2f} ms")
    print(f"p50 latency:      {p50 * 1000:.2f} ms")
    print(f"p95 latency:      {p95 * 1000:.2f} ms")
    print(f"p99 latency:      {p99 * 1000:.2f} ms")
    for r in sorted(results, key=lambda r: r["rank"]):
        n_r = len(r["latencies"])
        print(f"  worker {r['rank']}: {n_r} prompts in {r['end'] - r['start']:.3f}s "
              f"({n_r / (r['end'] - r['start']):.1f} prompts/s)")


def _run_prompts(agent, num_prompts, warmup):
    prompt_cycle = itertools.cycle(PROMPTS)
    for _ in range(warmup):
        agent.predict(next(prompt_cycle), QUESTIONS)
    start = time.perf_counter()
    latencies = []
    for _ in range(num_prompts):
        t0 = time.perf_counter()
        agent.predict(next(prompt_cycle), QUESTIONS)
        latencies.append(time.perf_counter() - t0)
    return start, time.perf_counter(), latencies


def _process_worker(rank, num_prompts, warmup, load_kwargs, barrier, queue):
    import laya_mlx as laya

    agent = laya.load(**load_kwargs)
    prompt_cycle = itertools.cycle(PROMPTS)
    for _ in range(warmup):
        agent.predict(next(prompt_cycle), QUESTIONS)
    barrier.wait()
    start = time.perf_counter()
    latencies = []
    for _ in range(num_prompts):
        t0 = time.perf_counter()
        agent.predict(next(prompt_cycle), QUESTIONS)
        latencies.append(time.perf_counter() - t0)
    queue.put({"rank": rank, "start": start, "end": time.perf_counter(), "latencies": latencies})


def _thread_worker(rank, agent, num_prompts, warmup, barrier, results, index):
    prompt_cycle = itertools.cycle(PROMPTS)
    for _ in range(warmup):
        agent.predict(next(prompt_cycle), QUESTIONS)
    barrier.wait()
    start = time.perf_counter()
    latencies = []
    for _ in range(num_prompts):
        t0 = time.perf_counter()
        agent.predict(next(prompt_cycle), QUESTIONS)
        latencies.append(time.perf_counter() - t0)
    results[index] = {"rank": rank, "start": start, "end": time.perf_counter(), "latencies": latencies}


def run_process_mode(args, load_kwargs, per_worker):
    barrier = mp.Barrier(args.workers)
    queue = mp.Queue()
    procs = [
        mp.Process(
            target=_process_worker,
            args=(i, per_worker, args.warmup, load_kwargs, barrier, queue),
        )
        for i in range(args.workers)
    ]
    for p in procs:
        p.start()
    results = [queue.get() for _ in procs]
    for p in procs:
        p.join()
    return results


def run_thread_mode(args, load_kwargs, per_worker):
    if args.backend == "ollama":
        import ollama_client

        agent = ollama_client.load(args.model, base_url=args.base_url)
    else:
        import laya_mlx as laya

        agent = laya.load(**load_kwargs)  # one shared model instance
    barrier = threading.Barrier(args.workers)
    results = [None] * args.workers
    threads = [
        threading.Thread(
            target=_thread_worker,
            args=(i, agent, per_worker, args.warmup, barrier, results, i),
        )
        for i in range(args.workers)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="aac6fef/laya-mlx")
    parser.add_argument("--num", type=int, default=400, help="Total prompts across all workers.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=10, help="Untimed warmup calls per worker.")
    parser.add_argument("--mode", choices=["thread", "process"], default="process")
    parser.add_argument("--dtype", default="float16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", choices=["gpu", "cpu"], default=None)
    parser.add_argument(
        "--backend",
        choices=["laya", "ollama"],
        default="laya",
        help="laya: laya-mlx checkpoint. ollama: a decision model served via Ollama's /v1/systemone "
        "(clef, clef-flash, ...) -- process mode unsupported, Ollama is already a shared server.",
    )
    parser.add_argument("--base-url", default="http://localhost:11434", help="Ollama server URL (--backend ollama only).")
    args = parser.parse_args()
    if args.backend == "ollama" and args.mode == "process":
        parser.error("--backend ollama only supports --mode thread (Ollama's server already shares the model)")

    load_kwargs = dict(
        model_id_or_path=args.model,
        dtype=args.dtype,
        batch_size=args.batch_size,
        device=args.device,
    )
    per_worker = args.num // args.workers
    total_num = per_worker * args.workers

    if args.mode == "process":
        results = run_process_mode(args, load_kwargs, per_worker)
    else:
        results = run_thread_mode(args, load_kwargs, per_worker)

    report(args.mode, results, total_num)


if __name__ == "__main__":
    main()
