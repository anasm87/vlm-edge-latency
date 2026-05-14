#!/usr/bin/env python3
"""
PyTorch VLM inference baseline — honest measurement, no torch.compile().

Same methodology as C++ benchmark: 10,000 frames, p50/p95/p99 over
wall-clock time including GPU synchronisation. torch.compile() is
deliberately omitted so this reflects what a standard Python PyTorch
deployment actually delivers in production.
"""

from __future__ import annotations

import argparse
import time
from statistics import mean
from typing import Optional

import torch


def percentile(latencies_ns: list[int], p: float) -> float:
    if not latencies_ns:
        return 0.0
    arr = sorted(latencies_ns)
    idx = int(p / 100.0 * (len(arr) - 1))
    return arr[idx] / 1e6  # ns → ms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PyTorch VLM inference latency baseline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="HuggingFace model ID or local path (e.g. google/gemma-3-4b-it)",
    )
    parser.add_argument("--frames", type=int, default=10_000)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument(
        "--image-size", type=int, default=224, help="Square image side length"
    )
    parser.add_argument(
        "--dtype",
        choices=["float16", "bfloat16", "float32"],
        default="float16",
        help="Model weight dtype",
    )
    parser.add_argument(
        "--check-p99",
        type=float,
        default=None,
        metavar="MS",
        help="Exit 1 if p99 exceeds this threshold (ms)",
    )
    return parser.parse_args()


def load_model(
    model_id: str, dtype_str: str, device: torch.device
) -> torch.nn.Module:
    from transformers import AutoModelForCausalLM  # type: ignore[import]

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map[dtype_str]

    print(f"Loading {model_id} ({dtype_str})...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()
    # Deliberately no torch.compile() — baseline must reflect unoptimised
    # production deployment.
    print("Model loaded.\n")
    return model


def make_dummy_inputs(
    image_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {
        "pixel_values": torch.randn(
            1, 3, image_size, image_size, dtype=dtype, device=device
        ),
        "input_ids": torch.randint(0, 32_000, (1, 16), device=device),
        "attention_mask": torch.ones(1, 16, dtype=torch.long, device=device),
    }


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("SpatialCore PyTorch Baseline Benchmark")
    print(f"  Model:  {args.model}")
    print(f"  Device: {device}")
    print(f"  Dtype:  {args.dtype}")
    print(f"  Frames: {args.frames}")
    print(f"  Warmup: {args.warmup}")
    print()

    model = load_model(args.model, args.dtype, device)
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    inputs = make_dummy_inputs(args.image_size, dtype_map[args.dtype], device)

    # ── Warmup ─────────────────────────────────────────────────────────────
    print(f"Warming up {args.warmup} frames...", end=" ", flush=True)
    with torch.no_grad():
        for _ in range(args.warmup):
            _ = model(**inputs)
            if device.type == "cuda":
                torch.cuda.synchronize()
    print("done\n")

    # ── Measurement ────────────────────────────────────────────────────────
    # time.perf_counter_ns() for nanosecond precision.
    # torch.cuda.synchronize() ensures GPU work is complete before t1.
    print(f"Measuring {args.frames} frames...", end=" ", flush=True)
    latencies_ns: list[int] = []

    with torch.no_grad():
        for _ in range(args.frames):
            t0 = time.perf_counter_ns()
            _ = model(**inputs)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter_ns()
            latencies_ns.append(t1 - t0)

    print("done\n")

    # ── Statistics ─────────────────────────────────────────────────────────
    p50 = percentile(latencies_ns, 50)
    p95 = percentile(latencies_ns, 95)
    p99 = percentile(latencies_ns, 99)
    mean_ms = mean(latencies_ns) / 1e6
    max_ms = max(latencies_ns) / 1e6

    W1, W2, W3 = 22, 18, 22
    sep = "=" * (W1 + W2 + W3)
    div = "-" * (W1 + W2 + W3)

    print(sep)
    print(
        f"{'Metric':<{W1}}"
        f"{'Python PyTorch (ms)':>{W2}}"
        f"{'C++ TRT (ms)':>{W3}}"
    )
    print(div)
    print(f"{'p50 latency':<{W1}}{p50:>{W2}.2f}{'~14':>{W3}}")
    print(f"{'p95 latency':<{W1}}{p95:>{W2}.2f}{'~20':>{W3}}")
    print(f"{'p99 latency':<{W1}}{p99:>{W2}.2f}{'~24':>{W3}}")
    print(f"{'Mean latency':<{W1}}{mean_ms:>{W2}.2f}{'~16':>{W3}}")
    print(f"{'Max latency':<{W1}}{max_ms:>{W2}.2f}{'~31':>{W3}}")
    print(sep)
    print(f"\nFrames measured: {args.frames}")
    print("Note: torch.compile() deliberately omitted — unoptimised baseline.")

    # ── CI threshold check ─────────────────────────────────────────────────
    check_p99: Optional[float] = args.check_p99
    if check_p99 is not None:
        if p99 > check_p99:
            print(
                f"\nFAIL: p99={p99:.2f}ms exceeds CI threshold {check_p99}ms",
                flush=True,
            )
            raise SystemExit(1)
        print(f"PASS: p99={p99:.2f}ms < CI threshold {check_p99}ms")


if __name__ == "__main__":
    main()
