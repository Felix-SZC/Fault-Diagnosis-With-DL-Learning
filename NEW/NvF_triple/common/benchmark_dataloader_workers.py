from __future__ import annotations

import argparse
import time

import torch
from torch.utils.data import DataLoader

from common.utils.helpers import load_config
from test_NvF_fi_triple import get_dataset


def bench_once(
    dataset,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    warmup_batches: int,
    measure_batches: int,
) -> float:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(persistent_workers and num_workers > 0),
    )

    it = iter(loader)
    for _ in range(warmup_batches):
        try:
            next(it)
        except StopIteration:
            break

    start = time.perf_counter()
    n = 0
    for _ in range(measure_batches):
        try:
            _x, _y = next(it)
            n += 1
        except StopIteration:
            break
    elapsed = time.perf_counter() - start
    if n == 0:
        return float("inf")
    return elapsed / n


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark DataLoader num_workers throughput.")
    parser.add_argument("--config", type=str, default="configs/bench_NvF_LaoDA_fi_triple_joint.yaml")
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--workers", type=str, default="0,2,4,6,8")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--warmup_batches", type=int, default=5)
    parser.add_argument("--measure_batches", type=int, default=40)
    parser.add_argument("--pin_memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--persistent_workers", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    ds = get_dataset(data_cfg, split=args.split, filter_classes=None)

    worker_list = [int(x.strip()) for x in args.workers.split(",") if x.strip()]
    results: list[tuple[int, float]] = []

    print(f"Dataset split={args.split}, size={len(ds)}, batch_size={args.batch_size}")
    print("Benchmarking ...")
    for w in worker_list:
        t = bench_once(
            dataset=ds,
            batch_size=args.batch_size,
            num_workers=w,
            pin_memory=args.pin_memory,
            persistent_workers=args.persistent_workers,
            warmup_batches=args.warmup_batches,
            measure_batches=args.measure_batches,
        )
        if t == float("inf"):
            print(f"workers={w}: no batch measured")
        else:
            print(f"workers={w}: {t:.4f} s/batch")
        results.append((w, t))

    finite = [(w, t) for (w, t) in results if t != float("inf")]
    if not finite:
        print("No valid result.")
        return
    best_w, best_t = min(finite, key=lambda x: x[1])
    print(f"\nRecommended num_workers={best_w} (fastest {best_t:.4f} s/batch)")


if __name__ == "__main__":
    torch.set_num_threads(max(1, torch.get_num_threads()))
    main()
