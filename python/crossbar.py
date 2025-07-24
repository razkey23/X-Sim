#!/usr/bin/env python3
from __future__ import annotations
import os, pickle, time, multiprocessing as mp
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Tuple, List
from statistics import mean
import dill
import numpy as np
from tqdm import tqdm
import xbar_simulator                                          # C++ bindings

# ----------------------------- constants ---------------------------------- #
# Rw=0.00001
# ideal_PARASITICS = (Rw, 1e20, 1e20, Rw, Rw, Rw)         # Rs, Rw, Rb, Cw, Cb, Cs
# Rw=1
# ir_PARASITICS = (Rw, 1e20, 1e20, Rw, Rw, Rw)         # Rs, Rw, Rb, Cw, Cb, Cs
# ir_Para

# column_divisor_path = os.path.abspath("/home/earapidis/Fast-Crossbar-Sim/python/column_divisors.pkl")
# column_divisor_path = os.path.abspath("/shares/bulk/earapidis/mac_optimization_divisors/monte_carlo/column_divisors.pkl")
# ADC_STEPS  = pickle.load(open(column_divisor_path , "rb"))

# ideal_current = 61.057
# ideal_steps = np.full_like(ADC_STEPS["gs"],ideal_current)
# ADC_STEPS["ideal"] = ideal_steps
# ADC_STEPS["no-comp"] = ideal_steps

# ----------------------------- 1. base class ------------------------------ #
class VectorSim(xbar_simulator.CrossbarSimulator):
    """
    A *single-vector* simulator.  One instance lives in one process.
    """
    def __init__(self, M: int, N: int,adc_steps_path:str, mode: str = "mapping_False-gs", transient=False):
        super().__init__(M, N)
        self.M, self.N, self.mode, self.transient = M, N, mode, transient
        self.initialize_jart()
        if mode=="ideal":
            Rw = 0.00001
        else:
            Rw = 1
        parasitics = (Rw, 1e20, 1e20, Rw, Rw, Rw)
            
        self.set_parasitic_resistances(*parasitics)
        if adc_steps_path!="":
            adc_steps  = dill.load(open(adc_steps_path , "rb"))
            self.adc_steps = adc_steps[self.mode]
        self.weights: np.ndarray | None = None             # set later
        

    # ----------------- helpers ------------------------------------------------
    def set_weights(self, w: np.ndarray) -> None:
        super().set_weights(w)
        self.weights = w

    def _solve(self, x: np.ndarray):
        """Return MAC (amps) for a single Boolean input row vector."""
        if self.transient:
            mem, mac = super().transientInference_mod(x)
        else:
            mem, mac = super().run_inference(x)
        mem = mem*1e6
        mac = mac*1e6
        return (mem,mac)                                     # A → µA

    def _digitise(self, mac: np.ndarray) -> np.ndarray:
        digital = np.floor((mac + 0.5*self.adc_steps)/self.adc_steps).astype(int)
        # digital = np.rint(mac / self.adc_steps).astype(int)
        return digital

    # ----------------- public API --------------------------------------------
    def run_vector(self, x: np.ndarray) -> np.ndarray:
        """One Boolean row vector → digital MAC."""
        _ , mac = self._solve(x)
        return self._digitise(mac)

    # convenience random-data helpers (unchanged logic, bug-fix reshape) ------
    def random_inputs(self, n_rows: int, p: float, seed=None) -> np.ndarray:
        rng   = np.random.default_rng(seed)
        k     = int(round((p if p<=1 else p/100) * self.M))
        rows  = np.zeros((n_rows, self.M), dtype=bool)
        for r in rows: r[rng.choice(self.M, k, replace=False)] = True
        if n_rows==1:
            rows = np.ravel(rows)
        return rows

    def random_weights(self, p: float, seed=None) -> np.ndarray:
        rng  = np.random.default_rng(seed)
        k    = int(round((p if p<=1 else p/100) * self.M * self.N))
        flat = np.zeros(self.M * self.N, dtype=bool)
        flat[rng.choice(flat.size, k, replace=False)] = True
        return flat.reshape(self.M, self.N)

    def mvm(self, x: np.ndarray) -> np.ndarray:
        return (x.astype(int) @ self.weights.astype(int))

# ----------------------------- 2. parallel wrapper ------------------------ #
# globals for worker processes
def _task(pair, weights, M, N,adc_steps_path, mode, transient):
    """Creates a new VectorSim inside each task."""
    idx, vec = pair
    sim = VectorSim(M, N,adc_steps_path, mode, transient)

    sim.set_weights(weights)
    if vec.ndim == 1:
        digital = sim.run_vector(vec)
        # _, mac = sim._solve(vec),  # µA
        # digital = sim._digitise(mac)
        return idx, digital
    else:
        _N_, M = vec.shape
        digital = np.empty((_N_,N))
        for id,_vec_ in enumerate(vec):
            _digital_ = sim.run_vector(_vec_)
            digital[id] = _digital_
        return idx, digital

def _collect_currents_(idx,x_per,w_per,M,N,num_runs,transient):
    adc_steps_path = ""
    mode = "mapping_False-gs"
    sim = VectorSim(M, N,adc_steps_path, mode=mode, transient=transient)
    x = sim.random_inputs(num_runs,x_per)
    w = sim.random_weights(w_per)

    sim.set_weights(w)
    collected_mac = []
    mvm = sim.mvm(x)

    if x.ndim == 1:
        _, mac = sim._solve(x)
        collected_mac.append(mac)
    else:
        _N_, M = x.shape
        for id,_vec_ in enumerate(x):
            _, mac = sim._solve(_vec_)
            collected_mac.append(mac)
    collected_mac = np.array(collected_mac)
    collected_mac = collected_mac.squeeze()
    # print(collected_mac.shape)
    # print(mvm.shape)
    output = np.stack((collected_mac,mvm))
    return idx, output


class Base():
    def __init__(self,M,N):
        self.M = M
        self.N = N
    def set_weights(self,weights):
        self.weights = weights
    def set_mode(self,mode):
        self.mode = mode
    
        
class ParallelSim(Base):
# class ParallelSim(VectorSim):
    """
    Front-end that slices a 2-D Boolean input matrix across processes,
    each of which owns a persistent VectorSim.
    """
    def __init__(self, M: int, N: int, mode="gs", transient=False, max_workers: int | None = None):
                
                super().__init__(M,N)
                self.set_mode(mode)
                self.transient = transient
                self.max_workers = max_workers or mp.cpu_count()
    
    mvm = VectorSim.mvm
    random_weights = VectorSim.random_weights
    random_inputs = VectorSim.random_inputs

    # ----------------------- main entry point --------------------------------
    def run(self, X: np.ndarray) -> np.ndarray:
        """
        Run in parallel: one VectorSim per task (no persistent sim per process).
        """
        if X.ndim == 1:
            return self.vector_sim.run_vector(X)

        indexed = list(enumerate(X))
        macs = [None] * len(indexed)

        # capture shared config/weights once
        args = (self.weights, self.M, self.N, self.mode, self.transient)
        # args = (self.vector_sim.weights, self.M, self.N, self.mode, self.transient)

        with ProcessPoolExecutor(
            max_workers=min(self.max_workers, len(indexed))
        ) as pool:
            futures = [
                pool.submit(_task, pair, *args)
                for pair in indexed
            ]
            for fut in as_completed(futures):
                idx, mac = fut.result()
                macs[idx] = mac

        return np.vstack(macs)

# ----------------------------- 3. benchmark -------------------------------- #
if __name__ == "__main__":
    M = N = 32
    P = 50          # sparsity %
    B = 10          # batch size
    RUNS = 1

    # print(_collect_currents_(0,P,P,M,N,RUNS,transient=False))

    mappings = [True,False]
    modes = ["gs","cs"]
    
    sim = VectorSim(M,N,"","ideal",False)
    x = sim.random_inputs(RUNS,P)
    w = sim.random_weights(P)
    sim.set_weights(w)
    mvm = sim.mvm(x)

    adc_steps_path = os.path.abspath("/shares/bulk/earapidis/dev/BinarizedNN/saved_models/lenet_5/model_1/adc_steps.pkl")
    for mapping in mappings:
        for mode in modes:
            _mode_ = f"mapping_{mapping}-{mode}"
            idx, digital = _task((0,x),w,M,N,adc_steps_path,mode=_mode_,transient=False)
            print(_mode_)
            correct = np.sum(digital==mvm)
            print(f"\tcorrect: {correct}/{N}")
