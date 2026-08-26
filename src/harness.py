"""Wall-clock timing and peak-memory measurement.

Timing and memory are experimental results, not diagnostics, so the protocol is
fixed: each timed section runs N_TIMING_RUNS times and the median is reported.

Memory is measured in a separate, untimed run. Sampling RSS requires a polling
thread, and that thread contends for the GIL — keeping it out of the timed runs
is what lets `nothing extraneous inside a timed section` hold literally.
"""

import gc
import threading
import time
from statistics import median
from typing import Any, Callable

import psutil

from src.config import N_TIMING_RUNS

BYTES_PER_MB = 1024 * 1024
MEMORY_POLL_SECONDS = 0.01


def _total_rss(process: psutil.Process) -> int:
    """Resident memory of `process` plus any live children, in bytes."""
    total = process.memory_info().rss
    for child in process.children(recursive=True):
        try:
            total += child.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            # Workers come and go while we poll; a vanished one contributes 0.
            continue
    return total


def time_median(
    fn: Callable[[], Any], n_runs: int = N_TIMING_RUNS
) -> tuple[Any, dict[str, Any]]:
    """Run `fn` n_runs times and return its result plus the timing summary.

    Garbage is collected between runs, outside the clock, so that one run's
    cleanup is not charged to the next.
    """
    durations = []
    result = None
    for _ in range(n_runs):
        gc.collect()
        start = time.perf_counter()
        result = fn()
        durations.append(time.perf_counter() - start)

    return result, {
        "seconds_median": median(durations),
        "seconds_min": min(durations),
        "seconds_max": max(durations),
        "n_runs": n_runs,
    }


def peak_memory(
    fn: Callable[[], Any], poll_seconds: float = MEMORY_POLL_SECONDS
) -> tuple[Any, dict[str, Any]]:
    """Run `fn` once while polling RSS, returning its result plus peak usage.

    Reports both absolute peak RSS and the rise above the pre-run baseline; the
    baseline covers the interpreter and already-imported libraries, so the delta
    is the part attributable to `fn`.

    Child processes are counted too: statsforecast fits series in a worker pool,
    and measuring only the parent would attribute almost none of that memory to
    the model.
    """
    process = psutil.Process()
    gc.collect()
    baseline_bytes = _total_rss(process)
    peak_bytes = baseline_bytes
    stop = threading.Event()

    def poll() -> None:
        nonlocal peak_bytes
        while not stop.is_set():
            peak_bytes = max(peak_bytes, _total_rss(process))
            stop.wait(poll_seconds)

    sampler = threading.Thread(target=poll, daemon=True)
    sampler.start()
    try:
        result = fn()
    finally:
        stop.set()
        sampler.join()

    # A short call can finish between polls; take a final reading so the peak is
    # never below the RSS actually observed at completion.
    peak_bytes = max(peak_bytes, _total_rss(process))

    return result, {
        "peak_rss_mb": peak_bytes / BYTES_PER_MB,
        "baseline_rss_mb": baseline_bytes / BYTES_PER_MB,
        "peak_rss_delta_mb": (peak_bytes - baseline_bytes) / BYTES_PER_MB,
    }


def measure(
    fn: Callable[[], Any], n_runs: int = N_TIMING_RUNS
) -> tuple[Any, dict[str, Any]]:
    """Measure `fn` end to end: n_runs timed runs, then one memory-sampled run.

    `fn` therefore executes n_runs + 1 times in total and must be idempotent.
    Returns the result of the memory run alongside the combined measurements.
    """
    _, timing = time_median(fn, n_runs)
    result, memory = peak_memory(fn)
    return result, {**timing, **memory}
