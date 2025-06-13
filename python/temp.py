#!/usr/bin/env python3
"""
LRS contribution on a 2×2 JART crossbar
---------------------------------------

For 10 random (input, weight) pairs:
    – run steady-state solver  (run_inference)
    – run transient solver     (transientInference)

Report the current through *each* cell that is simultaneously
 • driven (input = 1)  and
 • in LRS  (weight = True).

Requirements
------------
$ pip install numpy
$ python -c "import xbar_simulator"   # your just-built wheel / module
"""

from __future__ import annotations
import time, itertools, numpy as np
import xbar_simulator

# ───────────────────────── simulation constants ───────────────────────── #
M = N = 2                    # 2×2 crossbar only
SAMPLES = 10

PARASITICS = (1, 1e20, 1e20, 1, 1, 1)   # (Rswl1, Rswl2, Rsbl1, Rsbl2, Rwl, Rbl)

VP_H, VP_W, VP_R, VP_F = 0.1, 50e-6, 5e-6, 5e-6
WAVEFORM = np.array([[0.0, 0.0],
                     [VP_H, VP_R],
                     [VP_H, VP_W - VP_F],
                     [0.0, VP_W]], dtype=float)

rng = np.random.default_rng(42)        # reproducibility

def rand_bool_vec(n: int, p: float = .5) -> np.ndarray:  # Bernoulli(p)
    return rng.random(n) < p

def rand_bool_mat(m: int, n: int, p: float = .5) -> np.ndarray:
    return rng.random((m, n)) < p

def fresh_sim() -> xbar_simulator.CrossbarSimulator:
    sim = xbar_simulator.CrossbarSimulator(M, N)
    sim.initialize_jart()
    sim.set_parasitic_resistances(*PARASITICS)
    return sim

# ───────────────────────────────── main ───────────────────────────────── #
def main() -> None:
    dc_vals, tr_vals = [], []          # collect one entry per active-LRS cell
    t_dc_list, t_tr_list = [], []      # runtimes for information

    for k in range(1, SAMPLES+1):
        inp     = rand_bool_vec(M)           # 0 / 1 (False / True)
        w       = rand_bool_mat(M, N)

        # guarantee **at least one** driven-and-LRS cell to measure
        if not (inp[:, None] & w).any():
            idx_row = rng.integers(0, M)
            inp[idx_row] = True
            w[idx_row, rng.integers(0, N)] = True

        sim = fresh_sim()
        sim.set_weights(w)

        # 1) DC / steady-state
        tic = time.perf_counter()
        Icell_dc, _ = sim.run_inference(inp.astype(float), dt=0)   # tuple
        t_dc_list.append(time.perf_counter() - tic)

        # 2) full transient
        Vwl1 = inp.copy()                     # True where we apply the pulse
        zeros = np.zeros(M, bool)
        tic = time.perf_counter()
        Icell_tr, _ = sim.transientInference(Vwl1, zeros, zeros, zeros, w, WAVEFORM)
        t_tr_list.append(time.perf_counter() - tic)

        # 3) extract currents of interest
        mask = (inp[:, None]) & w             # shape (M, N)
        dc_vals.extend(np.asarray(Icell_dc)[mask])
        tr_vals.extend(np.asarray(Icell_tr)[mask])

        # nice per-sample log --------------------------------------------------
        print(f"sample {k:2d}:")
        for (i, j) in zip(*np.where(mask)):
            Idc = np.asarray(Icell_dc)[i, j]
            Itr = np.asarray(Icell_tr)[i, j]
            print(f"  cell ({i},{j})  Idc = {Idc:10.4e} A    Itr = {Itr:10.4e} A")
        print()

    # ───────── summary ─────────
    print("──────────────────  summary over 10 random trials  ──────────────────")
    print(f"total active-LRS cells measured : {len(dc_vals)}")
    print(f"⟨Idc⟩  = {np.mean(dc_vals):11.4e}  A")
    print(f"⟨Itr⟩  = {np.mean(tr_vals):11.4e}  A")
    print(f"⟨|Idc–Itr|⟩ = {np.mean(np.abs(np.asarray(dc_vals) - np.asarray(tr_vals))):.4e}  A")
    print()
    print(f"avg runtime  DC solver : {np.mean(t_dc_list):.3e}  s")
    print(f"avg runtime  transient : {np.mean(t_tr_list):.3e}  s")

# ────────────────────────────────────────────────────────────────────────── #
if __name__ == "__main__":
    main()