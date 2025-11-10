import argparse
import statistics
import time
from typing import Callable, Iterable, Tuple

import numpy as np
from needle.backend import device as backend_device
from needle.backend import ndarray as nd


def ensure_divisible(value: int, tile: int) -> None:
    if value % tile != 0:
        raise ValueError(f"Size {value} must be divisible by tile size {tile}")


def time_many(
    fn: Callable[[], None], repeats: int, warmup: int = 1
) -> Tuple[float, float, float]:
    # warmup
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)  # ms
    return min(times), statistics.median(times), max(times)


def numpy_baseline_time(
    m: int, n: int, p: int, repeats: int, warmup: int
) -> Tuple[float, float, float]:
    a = np.random.randn(m, n).astype(np.float32)
    b = np.random.randn(n, p).astype(np.float32)

    def run() -> None:
        c = a @ b
        # use the result to avoid optimizer elision
        _ = c[0, 0]

    return time_many(run, repeats=repeats, warmup=warmup)


def cpu_tiled_time(
    m: int, n: int, p: int, repeats: int, warmup: int
) -> Tuple[float, float, float]:
    dev = backend_device.cpu()
    t = dev.__tile_size__
    ensure_divisible(m, t)
    ensure_divisible(n, t)
    ensure_divisible(p, t)

    # Create regular 2D NDArrays; __matmul__ will dispatch to tiled path automatically
    a = nd.array(np.random.randn(m, n).astype(np.float32), device=dev)
    b = nd.array(np.random.randn(n, p).astype(np.float32), device=dev)

    def run() -> None:
        c = a @ b
        # Force realization; copy back to host to ensure kernels have finished
        _ = c.numpy()[0, 0]

    return time_many(run, repeats=repeats, warmup=warmup)


def parse_sizes(s: str) -> Iterable[Tuple[int, int, int]]:
    """
    Parse sizes of the form:
      - single int: k  -> (k, k, k)
      - triple: m,n,p
      - multiple groups separated by spaces,
        e.g. "512 1024 2048" or "1024,1024,2048 2048,2048,2048"
    """
    groups = s.strip().split()
    for g in groups:
        parts = [int(x) for x in g.split(",")]
        if len(parts) == 1:
            k = parts[0]
            yield (k, k, k)
        elif len(parts) == 3:
            yield (parts[0], parts[1], parts[2])
        else:
            raise ValueError(f"Invalid size group: {g}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark matmul: NumPy vs C++ tiled backend"
    )
    parser.add_argument(
        "--sizes",
        type=str,
        default="512 1024 1536 2048",
        help='Space-separated sizes. Each can be "k" or "m,n,p".\
              Example: "1024 2048,2048,2048".',
    )
    parser.add_argument(
        "--repeats", type=int, default=5, help="Number of timed runs per case"
    )
    parser.add_argument("--warmup", type=int, default=1, help="Warmup runs per case")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()

    np.random.seed(args.seed)

    # Probe CPU backend availability
    cpu_available = True
    try:
        dev = backend_device.cpu()
        tile = dev.__tile_size__
    except Exception as e:
        cpu_available = False
        tile = None
        print(f"[warn] CPU backend unavailable: {e}")

    print("Backend(s):")
    print("  - NumPy")
    if cpu_available:
        print(f"  - C++ CPU (tiled), tile_size={tile}")
    print()

    header = f"{'Size (m,n,p)':>16} | {'NumPy ms (min/med/max)':>28}"
    if cpu_available:
        header += f" | {'CPU tiled ms (min/med/max)':>30} | {'speedup (med)':>13}"
    print(header)
    print("-" * len(header))

    for m, n, p in parse_sizes(args.sizes):
        np_min, np_med, np_max = numpy_baseline_time(
            m, n, p, repeats=args.repeats, warmup=args.warmup
        )
        row = f"{(m, n, p)!s:>16} | {np_min:7.2f}/{np_med:7.2f}/{np_max:7.2f}"
        if cpu_available:
            # skip sizes not divisible by tile
            if (m % tile != 0) or (n % tile != 0) or (p % tile != 0):
                row += f" | {'n/a':>30} | {'n/a':>13}"
            else:
                cpu_min, cpu_med, cpu_max = cpu_tiled_time(
                    m, n, p, repeats=args.repeats, warmup=args.warmup
                )
                speedup = np_med / cpu_med if cpu_med > 0 else float("inf")
                row += f" | {cpu_min:7.2f}/{cpu_med:7.2f}/{cpu_max:7.2f} | {speedup:>13.2f}x"
        print(row)


if __name__ == "__main__":
    main()
