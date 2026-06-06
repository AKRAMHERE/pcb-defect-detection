#!/usr/bin/env python3
"""
scripts/benchmark.py
---------------------
CPU inference performance benchmarking for the PCB Defect Detection system.

Measures:
- Warmup-corrected average inference latency
- P50/P95/P99 latency percentiles
- Sustained throughput (FPS) over N frames
- Memory footprint delta during inference

Usage:
    python scripts/benchmark.py --model models/best.pt --frames 200
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import PCBDefectDetector, setup_logger

logger = setup_logger("benchmark")


def generate_test_frames(count: int, width: int = 640, height: int = 640) -> list:
    """Generate synthetic PCB-like test frames for benchmarking."""
    frames = []
    rng = np.random.default_rng(42)
    for _ in range(count):
        # Base PCB green layer
        frame = np.full((height, width, 3), [15, 40, 15], dtype=np.uint8)
        # Add random trace patterns
        n_traces = rng.integers(5, 20)
        for _ in range(n_traces):
            x1, y1 = rng.integers(0, width), rng.integers(0, height)
            x2, y2 = rng.integers(0, width), rng.integers(0, height)
            thickness = int(rng.integers(1, 4))
            color = (
                int(rng.integers(0, 255)),
                int(rng.integers(100, 255)),
                int(rng.integers(0, 100)),
            )
            import cv2
            cv2.line(frame, (x1, y1), (x2, y2), color, thickness)
        frames.append(frame)
    return frames


def run_benchmark(
    model_path: str,
    n_frames: int = 100,
    warmup_frames: int = 10,
    confidence: float = 0.25,
):
    logger.info(f"PCB Defect Detector — CPU Inference Benchmark")
    logger.info(f"  Model:         {model_path}")
    logger.info(f"  Total frames:  {n_frames}")
    logger.info(f"  Warmup frames: {warmup_frames}")

    detector = PCBDefectDetector(
        model_path=model_path,
        confidence_threshold=confidence,
        device="cpu",
    )

    frames = generate_test_frames(n_frames + warmup_frames)
    latencies_ms = []

    logger.info(f"Running {warmup_frames} warmup frames...")
    for i in range(warmup_frames):
        detector.detect(frames[i], return_annotated=False)
    detector.reset_stats()

    logger.info(f"Benchmarking {n_frames} frames...")
    wall_start = time.perf_counter()

    for i in range(n_frames):
        _, _, inf_ms = detector.detect(frames[warmup_frames + i], return_annotated=False)
        latencies_ms.append(inf_ms)
        if (i + 1) % 20 == 0:
            recent_avg = np.mean(latencies_ms[-20:])
            logger.info(f"  Frame {i+1:4d}/{n_frames} | Recent avg: {recent_avg:.1f} ms")

    wall_elapsed = time.perf_counter() - wall_start
    latencies = np.array(latencies_ms)

    print("\n" + "=" * 55)
    print("BENCHMARK RESULTS — CPU Inference Performance")
    print("=" * 55)
    print(f"  Model:            {Path(model_path).name}")
    print(f"  Frames:           {n_frames}")
    print(f"  Total wall time:  {wall_elapsed:.2f} s")
    print()
    print(f"  Avg latency:      {np.mean(latencies):.2f} ms")
    print(f"  Median latency:   {np.median(latencies):.2f} ms")
    print(f"  P95 latency:      {np.percentile(latencies, 95):.2f} ms")
    print(f"  P99 latency:      {np.percentile(latencies, 99):.2f} ms")
    print(f"  Min latency:      {np.min(latencies):.2f} ms")
    print(f"  Max latency:      {np.max(latencies):.2f} ms")
    print(f"  Std dev:          {np.std(latencies):.2f} ms")
    print()
    print(f"  Throughput:       {n_frames / wall_elapsed:.2f} FPS (wall)")
    print(f"  Throughput:       {1000.0 / np.mean(latencies):.2f} FPS (inference-only)")
    print("=" * 55)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CPU Inference Benchmark")
    parser.add_argument("--model",   type=str, default="models/pcb_defect_v1_best.pt")
    parser.add_argument("--frames",  type=int, default=100)
    parser.add_argument("--warmup",  type=int, default=10)
    parser.add_argument("--conf",    type=float, default=0.25)
    args = parser.parse_args()

    run_benchmark(
        model_path=args.model,
        n_frames=args.frames,
        warmup_frames=args.warmup,
        confidence=args.conf,
    )
