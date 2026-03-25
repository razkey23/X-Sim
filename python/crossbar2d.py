#!/usr/bin/env python3
"""High-level Python facade for xbar_simulator_2d.

This module exposes:
- batched inference on 2-D inputs (samples, M)
- threaded batched inference across samples
- simple benchmark utilities to estimate thread-pool speedup
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Dict, Iterable, List, Tuple

import numpy as np

from . import xbar_simulator_2d as xs2


@dataclass
class BatchResult:
    iout: np.ndarray
    mac: np.ndarray


@dataclass
class SpeedupResult:
    workers: int
    seq_time_s: float
    thr_time_s: float
    speedup: float
    seq_throughput_samples_s: float
    thr_throughput_samples_s: float


class BatchCrossbarSim2D:
    """Python wrapper for 2-D crossbar inference.

    Input format:
    - x shape (samples, M), float32 preferred.

    Output format:
    - iout shape (samples, M, N)
    - mac shape (samples, N)
    """

    def __init__(self, m: int, n: int, bits_per_cell: int = 1, memristor: str = "jart") -> None:
        self.m = int(m)
        self.n = int(n)
        self.bits_per_cell = int(bits_per_cell)

        self._sim = xs2.CrossbarSimulator2D(self.m, self.n, self.bits_per_cell)
        self._initialize(memristor)

    def _initialize(self, memristor: str) -> None:
        key = str(memristor).strip().lower()
        if key == "jart":
            self._sim.initialize_jart()
        elif key == "fefet":
            self._sim.initialize_fefet()
        else:
            raise ValueError("memristor must be 'jart' or 'fefet'")

    @staticmethod
    def set_eigen_threads(num_threads: int) -> None:
        xs2.set_eigen_threads(int(num_threads))

    @staticmethod
    def get_eigen_threads() -> int:
        return int(xs2.get_eigen_threads())

    @staticmethod
    def module_threadpool_size() -> int:
        # Compile-time constant from the C++ simulator settings.
        return int(xs2.simulation_num_threads)

    def set_parasitic_resistances(
        self,
        rswl1: float,
        rswl2: float,
        rsbl1: float,
        rsbl2: float,
        rwl: float,
        rbl: float,
    ) -> None:
        self._sim.set_parasitic_resistances(
            float(rswl1), float(rswl2), float(rsbl1), float(rsbl2), float(rwl), float(rbl)
        )

    def set_weights(self, weights: np.ndarray) -> None:
        w = np.asarray(weights, dtype=np.int32)
        if w.shape != (self.m, self.n):
            raise ValueError(f"weights must have shape ({self.m}, {self.n}), got {w.shape}")
        self._sim.set_weights(w)

    def run_batch(self, x: np.ndarray, dt: float = xs2.simulation_time_step, method: str = "fixed-point") -> BatchResult:
        x2 = self._as_batch_input(x)
        iout, mac = self._sim.run_inference_2d(x2, dt=float(dt), method=str(method))
        return BatchResult(iout=np.asarray(iout), mac=np.asarray(mac))

    def run_batch_threaded(
        self,
        x: np.ndarray,
        num_workers: int = 0,
        dt: float = xs2.simulation_time_step,
        method: str = "fixed-point",
    ) -> BatchResult:
        x2 = self._as_batch_input(x)
        iout, mac = self._sim.run_inference_2d_threaded(
            x2, num_workers=int(num_workers), dt=float(dt), method=str(method)
        )
        return BatchResult(iout=np.asarray(iout), mac=np.asarray(mac))

    def benchmark_thread_pool_speedup(
        self,
        x: np.ndarray,
        worker_list: Iterable[int],
        dt: float = xs2.simulation_time_step,
        method: str = "fixed-point",
        warmup: int = 1,
    ) -> List[SpeedupResult]:
        """Compare sequential batch inference against threaded sample-level inference.

        Returns one SpeedupResult per workers value.
        """
        x2 = self._as_batch_input(x)
        samples = max(1, int(x2.shape[0]))

        for _ in range(max(0, int(warmup))):
            self._sim.run_inference_2d(x2, dt=float(dt), method=str(method))

        t0 = perf_counter()
        _ = self._sim.run_inference_2d(x2, dt=float(dt), method=str(method))
        t1 = perf_counter()
        seq_time = max(t1 - t0, 1e-12)
        seq_thr = samples / seq_time

        results: List[SpeedupResult] = []
        for workers in worker_list:
            w = int(workers)
            if w <= 0:
                continue

            for _ in range(max(0, int(warmup))):
                self._sim.run_inference_2d_threaded(x2, num_workers=w, dt=float(dt), method=str(method))

            s0 = perf_counter()
            _ = self._sim.run_inference_2d_threaded(x2, num_workers=w, dt=float(dt), method=str(method))
            s1 = perf_counter()
            thr_time = max(s1 - s0, 1e-12)
            thr_thr = samples / thr_time

            results.append(
                SpeedupResult(
                    workers=w,
                    seq_time_s=seq_time,
                    thr_time_s=thr_time,
                    speedup=seq_time / thr_time,
                    seq_throughput_samples_s=seq_thr,
                    thr_throughput_samples_s=thr_thr,
                )
            )

        return results

    def benchmark_thread_pool_speedup_table(
        self,
        x: np.ndarray,
        worker_list: Iterable[int],
        dt: float = xs2.simulation_time_step,
        method: str = "fixed-point",
        warmup: int = 1,
    ) -> Dict[str, List[float]]:
        """Return benchmark results in a dictionary format for quick logging."""
        rows = self.benchmark_thread_pool_speedup(
            x=x,
            worker_list=worker_list,
            dt=dt,
            method=method,
            warmup=warmup,
        )
        return {
            "workers": [r.workers for r in rows],
            "seq_time_s": [r.seq_time_s for r in rows],
            "thr_time_s": [r.thr_time_s for r in rows],
            "speedup": [r.speedup for r in rows],
            "seq_throughput_samples_s": [r.seq_throughput_samples_s for r in rows],
            "thr_throughput_samples_s": [r.thr_throughput_samples_s for r in rows],
        }

    def _as_batch_input(self, x: np.ndarray) -> np.ndarray:
        x2 = np.asarray(x, dtype=np.float32)
        if x2.ndim != 2:
            raise ValueError(f"input must be 2-D with shape (samples, {self.m}), got ndim={x2.ndim}")
        if x2.shape[1] != self.m:
            raise ValueError(f"input second dimension must equal M={self.m}, got {x2.shape[1]}")
        return np.ascontiguousarray(x2)


if __name__ == "__main__":
    # Tiny self-test / usage example.
    m, n = 64 , 64
    # m, n = 8, 8
    sim = BatchCrossbarSim2D(m, n, bits_per_cell=1, memristor="jart")

    rng = np.random.default_rng(0)
    weights = rng.integers(0, 2, size=(m, n), dtype=np.int32)
    sim.set_weights(weights)

    x = (rng.random((128, m)) > 0.5).astype(np.float32)
    x *= float(xs2.voltage_pulse_height)

    out_seq = sim.run_batch(x)
    num_workers = 16
    out_thr = sim.run_batch_threaded(x, num_workers=4)

    print("seq shapes:", out_seq.iout.shape, out_seq.mac.shape)
    print("thr shapes:", out_thr.iout.shape, out_thr.mac.shape)

    report = sim.benchmark_thread_pool_speedup(x, worker_list=[1, 2, 4], warmup=1)
    for row in report:
        print(
            f"workers={row.workers} "
            f"speedup={row.speedup:.3f} "
            f"seq_time={row.seq_time_s:.6f}s "
            f"thr_time={row.thr_time_s:.6f}s"
        )
