#!/usr/bin/env python3
"""
Benchmark: run_inference vs transientInference
----------------------------------------------
For crossbar sizes 32, 64, 128:
  * generate 10 random (weights, inputs) pairs
  * run both solvers once per pair
  * report average |steady – transient| error and average runtimes
"""

from __future__ import annotations
import time
from statistics import mean
from pathlib import Path
import numpy as np
import xbar_simulator

# ──────────────────────────  constants  ──────────────────────────── #
SIZES          = (32, 64, 128)
SAMPLES        = 10
PARASITICS     = (3, 1e20, 1e20, 5, 3, 2)           # same as C++ example
VP_H, VP_W, VP_R, VP_F = 0.1, 50e-6, 5e-6, 5e-6     # read-pulse shape

WAVEFORM = np.array([[0.0,     0.0],
                     [VP_H,   VP_R],
                     [VP_H,   VP_W - VP_F],
                     [0.0,    VP_W]], dtype=float)

# ––––– small helpers ––––– #
rng = np.random.default_rng()

def random_bool_vec(n: int, p: float = 0.5) -> np.ndarray:
    """Random boolean vector with probability *p* of being True."""
    return rng.random(n) < p

def random_bool_mat(m: int, n: int, p: float = 0.5) -> np.ndarray:
    """Random boolean M×N matrix."""
    return rng.random((m, n)) < p

def build_sim(m: int) -> xbar_simulator.CrossbarSimulator:
    """Fresh simulator with JART devices and fixed parasitics."""
    sim = xbar_simulator.CrossbarSimulator(m, m)
    sim.initialize_jart()
    sim.set_parasitic_resistances(*PARASITICS)
    return sim

# ──────────────────────────  main benchmark  ─────────────────────── #
def main() -> None:
    hdr = "{:>7} | {:>12} | {:>12} | {:>12} | {:>12}"
    row = "{:7d} | {:12.3e} | {:12.3e} | {:12.3e} | {:12.3e}"
    print(hdr.format("size",
                     "mean-MAE (A)",
                     "mean-rel-MAE",
                     "⟨t⟩ steady (s)",
                     "⟨t⟩ transient (s)"))
    print("-"*67)

    for m in SIZES:
        mae_list      : list[float] = []   # absolute mean err per sample
        rel_mae_list  : list[float] = []   # relative MAE  (dimension-less)
        t_ss_list     : list[float] = []   # runtime run_inference
        t_tr_list     : list[float] = []   # runtime transientInference

        for _ in range(SAMPLES):
            # random data
            inputs  = random_bool_vec(m)
            weights = random_bool_mat(m, m)

            sim = build_sim(m)
            sim.set_weights(weights)

            # ––– steady-state –––
            t0 = time.perf_counter()
            _, mac_ss = sim.run_inference(inputs)      # (m,)
            t_ss_list.append(time.perf_counter() - t0)

            # ––– transient –––
            Vwl1 = inputs.copy()            # rows driven with pulse
            Vwl2 = np.zeros(m, bool)
            Vbl1 = np.zeros(m, bool)
            Vbl2 = np.zeros(m, bool)

            t0 = time.perf_counter()
            _, mac_tr = sim.transientInference(Vwl1, Vwl2,
                                               Vbl1, Vbl2,
                                               weights, WAVEFORM)
            mac_tr = np.asarray(mac_tr)
            t_tr_list.append(time.perf_counter() - t0)

            # ––– error metrics –––
            abs_err  = np.abs(mac_ss - mac_tr)
            mae      = abs_err.mean()
            rel_mae  = (abs_err / np.maximum(np.abs(mac_tr), 1e-30)).mean()

            mae_list.append(mae)
            rel_mae_list.append(rel_mae)

        # aggregate per crossbar size
        print(row.format(m,
                         mean(mae_list),
                         mean(rel_mae_list),
                         mean(t_ss_list),
                         mean(t_tr_list)))

    print("\nDone – 10 samples per size compared.")

# ──────────────────────────────────────────────────────────────────── #
if __name__ == "__main__":
    main()
