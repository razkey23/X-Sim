#!/usr/bin/env python3
"""Bridge function to replace software MVM with crossbar simulation.

This file is designed for integration from Network_mapping code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import time
from .crossbar2d import BatchCrossbarSim2D
from . import xbar_simulator_2d as xs2

ExecutionMode = Literal["sequential", "threadpool"]


@dataclass
class CrossbarMVMResult:
    mac: np.ndarray
    iout: Optional[np.ndarray] = None


def crossbar_mvm(
    inputs_dm: np.ndarray,
    weights_mn: np.ndarray,
    bit_cell_precision: int,
    wire_resistance: float,
    thread_pools: int,
    mode: ExecutionMode = "threadpool",
    memristor: str = "jart",
    method: str = "fixed-point",
    dt: float = xs2.simulation_time_step,
    return_iout: bool = False,
) -> CrossbarMVMResult:
    """Run crossbar-based batched MVM.

    Required inputs:
    1. inputs_dm: (D, M) crossbar input batch
    2. weights_mn: (M, N) crossbar weights
    3. bit_cell_precision: bits per cell
    4. wire_resistance: line resistance value used to set parasitics
    5. thread_pools: number of worker threads for sample-level threadpool mode

    Sequential option:
    - mode="sequential" runs the same batch without sample-level threadpool.

    Returns:
    - mac: (D, N) matrix (column MAC currents)
    - iout: optional (D, M, N) device-level currents
    """

    x = np.asarray(inputs_dm, dtype=np.float32)
    w = np.asarray(weights_mn, dtype=np.int32)

    if x.ndim != 2:
        raise ValueError(f"inputs_dm must be 2-D with shape (D, M), got ndim={x.ndim}")
    if w.ndim != 2:
        raise ValueError(f"weights_mn must be 2-D with shape (M, N), got ndim={w.ndim}")

    d, m = x.shape
    mw, n = w.shape
    if mw != m:
        raise ValueError(f"Shape mismatch: inputs M={m} but weights M={mw}")
    if bit_cell_precision < 1:
        raise ValueError("bit_cell_precision must be >= 1")
    if thread_pools < 0:
        raise ValueError("thread_pools must be >= 0")

    # The current wrapper treats non-zero inputs as active rows and drives
    # active rows with voltage_pulse_height.
    x_drive = np.where(x != 0, float(xs2.voltage_pulse_height), 0.0).astype(np.float32)

    sim = BatchCrossbarSim2D(m=m, n=n, bits_per_cell=int(bit_cell_precision), memristor=memristor)

    # Keep same parasitic pattern used in existing scripts while exposing
    # wire_resistance as the tunable parameter.
    rw = float(wire_resistance)
    sim.set_parasitic_resistances(rw, 1e20, 1e20, rw, rw, rw)
    sim.set_weights(w)

    mode_norm = str(mode).strip().lower()
    if mode_norm == "sequential":
        out = sim.run_batch(x_drive, dt=float(dt), method=method)
    elif mode_norm == "threadpool":
        out = sim.run_batch_threaded(x_drive, num_workers=int(thread_pools), dt=float(dt), method=method)
    else:
        raise ValueError("mode must be 'sequential' or 'threadpool'")

    if return_iout:
        return CrossbarMVMResult(mac=out.mac, iout=out.iout)
    return CrossbarMVMResult(mac=out.mac, iout=None)


def crossbar_mvm_mac(
    inputs_dm: np.ndarray,
    weights_mn: np.ndarray,
    bit_cell_precision: int,
    wire_resistance: float,
    thread_pools: int,
    mode: ExecutionMode = "threadpool",
    memristor: str = "jart",
    method: str = "fixed-point",
    dt: float = xs2.simulation_time_step,
) -> np.ndarray:
    """Convenience wrapper that returns only MAC outputs (D, N)."""
    result = crossbar_mvm(
        inputs_dm=inputs_dm,
        weights_mn=weights_mn,
        bit_cell_precision=bit_cell_precision,
        wire_resistance=wire_resistance,
        thread_pools=thread_pools,
        mode=mode,
        memristor=memristor,
        method=method,
        dt=dt,
        return_iout=False,
    )
    return result.mac


def _digitise(mac, adc_steps) -> np.ndarray:
    digital = np.floor((mac + 0.5*adc_steps)/adc_steps).astype(int)
    # digital = np.rint(mac / self.adc_steps).astype(int)
    return digital


if __name__ == "__main__":
    rng = np.random.default_rng(7)

    d, m, n = 256, 64, 64
    # d, m, n = 16, 32, 32
    workers = 4 
    cell_precision = 2
    x = (rng.random((d, m)) > 0.5).astype(np.float32)
    w = rng.integers(0, 2**cell_precision, size=(m, n), dtype=np.int32)
    wire_resistance = 0.00000000001
    
    
    # y_seq = crossbar_mvm_mac(
    #     inputs_dm=x,
    #     weights_mn=w,
    #     bit_cell_precision=1,
    #     wire_resistance=1.0,
    #     thread_pools=4,
    #     mode="sequential",
    # )
    out = x @ w
    step = 60/(2**cell_precision -1)
    # adc_steps = np.full(N,40)
    adc_steps = np.full(n, step)
    # print(adc_steps)
    
    # print("digital:", out)
    start_time = time.perf_counter()
    mac = crossbar_mvm_mac(
        inputs_dm=x,
        weights_mn=w,
        bit_cell_precision=cell_precision,
        wire_resistance=wire_resistance,
        thread_pools=workers,
        mode="threadpool",
    )
    end_time = time.perf_counter()
    print(f"Crossbar MVM time: {(end_time - start_time)*1000:.2f} ms")

    mac = mac*1e6
    
    values = _digitise(mac, adc_steps)
    # print(values-out)
    print(np.array_equal(out, values))
    # print("analog mac:", mac)
    # print("digitized values:", values)
    # # print("seq shape:", y_seq.shape)
    # print("thr shape:", y_thr.shape)
