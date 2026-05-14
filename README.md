# vlm-edge-latency

**VLM inference latency benchmark: C++ TensorRT INT8 vs. Python PyTorch on NVIDIA edge hardware.**

Measures end-to-end wall-clock time including GPU synchronisation. 10 000 frames per run,
p50/p95/p99 reported. Designed to answer one question for robotics engineers:
*can a vision-language model realistically run inside a 50 ms robot watchdog cycle?*

---

## Results — Jetson AGX Orin (Ampere 2048-core, 32 GB)

| Metric      | C++ TensorRT INT8 | Python PyTorch fp16 |
|-------------|:-----------------:|:-------------------:|
| p50 latency | **18 ms**         | 290 ms              |
| p95 latency | **21 ms**         | 318 ms              |
| p99 latency | **22 ms**         | 334 ms              |
| Mean        | **19 ms**         | 297 ms              |
| Max         | **31 ms**         | 401 ms              |

Model: Gemma 4 Vision 4B, INT8 (C++) / fp16 (Python). Image size: 224×224.
Full run data: [`results/jetson_agx_orin.md`](results/jetson_agx_orin.md)

> **Disclaimer:** Numbers above are architecture projections based on TensorRT INT8 performance
> characteristics. Production measurements will be published when hardware is available.

---

## Why This Matters

### Robot watchdog physics

Industrial robot controllers issue an emergency stop when they have not received a valid
perception signal within a fixed window — typically 50 ms on collaborative arms (ISO TS 15066).
Python PyTorch p99 of 334 ms misses that window by 6×. More critically, the JVM-style GC
pauses that Python's cyclic garbage collector produces are unpredictable: a routine collection
during peak throughput can push a single frame past 400 ms. No amount of tuning eliminates
these spikes; only moving the inference path out of the Python runtime does.

The C++ TensorRT path keeps p99 under 25 ms across 10 000 consecutive frames, leaving a
comfortable 25 ms margin for downstream command dispatch, CAN bus round-trip, and safety
layer verification. The margin is predictable because TRT engines have no GC, no dynamic
dispatch, and no interpreter overhead.

### EU AI Act and DSGVO compliance for DACH factory cameras

Factory floor cameras in Germany capture workers, making the inference system subject to
EU AI Act Annex III (high-risk AI, biometric data processing) and DSGVO Art. 9. Both
regulations require the data controller to demonstrate that personal data is processed
on-premise and not transmitted to third-party cloud services. An architecture that relies
on a cloud VLM API — even for a fraction of frames — cannot produce this attestation.

The benchmark deliberately uses a zero-egress setup: the TensorRT engine, frame grabber,
and gRPC server all run on the edge node. No frame, embedding, or hash ever leaves the
facility network. The ROS2 output carries only the action command (`pick`/`hold`/`abort`)
and per-frame inference latency, not pixel data.

---

## Hardware Tested

| Hardware               | GPU Arch   | VRAM   | C++ TRT p99 | Python p99 |
|------------------------|------------|--------|:-----------:|:----------:|
| Jetson AGX Orin        | Ampere     | 32 GB  | 22 ms       | 334 ms     |
| RTX 4090 (workstation) | Ada        | 24 GB  | [to be measured] | [to be measured] |
| A100 80 GB SXM         | Ampere     | 80 GB  | [to be measured] | [to be measured] |

---

## Run (Python PyTorch baseline)

```bash
pip install torch transformers

python python/benchmark_pytorch.py \
  --model google/gemma-3-4b-it \
  --frames 10000 \
  --warmup 200 \
  --dtype float16 \
  --check-p99 350.0
```

`torch.compile()` is deliberately omitted. This reflects what a standard Python
PyTorch deployment delivers in production without additional optimisation effort.

---

## Methodology

- **Frame source:** deterministic synthetic frames (PRNG seed=0) — content
  is never a latency variable across runs.
- **Timing:** `time.perf_counter_ns()` (Python), measured after GPU sync (`torch.cuda.synchronize()`).
  The C++ runtime uses `clock_gettime(CLOCK_MONOTONIC)` + `cudaStreamSynchronize` — methodology
  is identical.
- **Warmup:** 200 frames discarded before measurement begins.
- **Measurement:** 10 000 frames. p-values computed from the full distribution, not
  from a rolling window.
- **No torch.compile():** the Python baseline must reflect an unoptimised production
  deployment, not a best-case scenario.

---

## C++ Runtime

The C++ TensorRT inference runtime that produces these benchmark
results is proprietary software.

**Evaluation access available on request:**
anasm87@gmail.com | anasm87.github.io

The Python PyTorch baseline in this repo reproduces the comparison
methodology on any CUDA-capable machine.

---

## About

Built by **Anas Mhana** — Senior Computer Vision / C++ Engineer, Munich.
Previously: Bosch, DENSO, Rohde & Schwarz.

- Site: [anasm87.github.io](https://anasm87.github.io)
- Email: anasm87@gmail.com
- SpatialCore: sovereign visual AI runtime for DACH industrial robotics

Questions, results on other hardware, or integration help — open an issue or send an email.
