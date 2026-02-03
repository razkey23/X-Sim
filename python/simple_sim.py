# import xbar_simulator          
from Fast_Crossbar_Sim.python import xbar_simulator
# C++ bindings
# from . import xbar_simulator                                          # C++ bindings
import numpy as np
import random
import math
import torch
import itertools
def sample_without_replacement(n: int, r: int, k: int, seed=None) -> np.ndarray:
    if r > n:
        raise ValueError(f"r={r} cannot be larger than n={n}.")

    max_k = math.comb(n, r)
    rng = np.random.default_rng(seed)

    # If k is larger than max_k, we can:
    #  - generate all unique combinations (max_k rows)
    #  - then sample with replacement from them to get exactly k rows
    if k >= max_k:
        # all unique combinations as (max_k, r)
        combos = np.fromiter(
            (i for comb in itertools.combinations(range(n), r) for i in comb),
            dtype=int,
            count=max_k * r,
        ).reshape(max_k, r)

        if k == max_k:
            return combos

        # sample k rows with replacement from all unique combos
        idx = rng.integers(0, max_k, size=k)
        return combos[idx]

    # k < max_k: we want k unique combinations, sampled randomly
    possible_set: set[tuple[int, ...]] = set()

    while len(possible_set) < k:
        remaining = k - len(possible_set)
        batch_size = min(remaining * 3, 2000)  # smallish so Python loop is OK

        # Sample each row independently, without replacement within the row
        batch = [
            tuple(sorted(rng.choice(n, size=r, replace=False)))
            for _ in range(batch_size)
        ]

        for combo in batch:
            possible_set.add(combo)
            if len(possible_set) == k:
                break

    return np.array(list(possible_set), dtype=int)

def random_inputs(M:int, p: float,k: int=1, seed=None) -> np.ndarray:
    r     = int(round((p if p<=1 else p/100) * M))
    rows  = np.zeros((k, M), dtype=bool)
    active_rows = sample_without_replacement(M,r,k,seed)
    for i in range(k):
        rows[i,active_rows[i]] = True
    if k==1:
        rows = np.squeeze(rows)
    return rows

def random_weights(M:int,N:int, bits_per_cell: int, p: float,k: int=1, seed=None) -> np.ndarray:
    r = int(round((p if p<=1 else p/100) * M * N))
    flat_weights = np.zeros((k,M * N), dtype=int)
    active_cells = sample_without_replacement(M*N,r,k,seed)
    
    for i in range(k):
        for pos in active_cells[i]:
            cell_val = random.randint(1,2**bits_per_cell-1)
            flat_weights[i,pos] = cell_val

    weights = flat_weights.reshape((k,M,N))
    
    if k==1:
        weights = np.squeeze(weights)
    return weights

def _digitise(mac: torch.Tensor, adc_steps: torch.Tensor) -> np.ndarray:
    digital = torch.floor((mac + 0.5*adc_steps)/adc_steps).to(torch.int)
    # digital = np.rint(mac / self.adc_steps).astype(int)
    return digital


class Simple_Sim(xbar_simulator.CrossbarSimulator):
    """
    A *single-vector* simulator.  One instance lives in one process.
    """
    def __init__(self, M: int, N: int, bits_per_cell:int, Rw: int, transient=False):
        super().__init__(M, N,bits_per_cell)
        self.M, self.N, self.Rw, self.transient = M, N, Rw, transient
        self.initialize_jart()

        parasitics = (self.Rw, 1e20, 1e20, self.Rw, self.Rw, self.Rw)
        self.set_parasitic_resistances(*parasitics)

        self.weights: torch.Tensor | None = None             # set later
    
    def set_weights(self, w: torch.Tensor) -> None:
        super().set_weights(w)
        self.weights = w


    def _solve_one_(self, x: torch.Tensor):
        """Return MAC (amps) for a single Boolean input row vector."""
        # self.initialize_jart()
        self.set_weights(self.weights)
        if self.transient:
            mem, mac = super().transientInference_mod(x)
        else:
            mem, mac = super().run_inference(x)
        mem = mem*1e6
        mac = mac*1e6
        return (mem,mac)         
                                # 
    def _solve_(self, x: torch.Tensor):
        """Return MAC (amps) for multiple Boolean input row vectors."""
        if x.ndim == 1:
            xs = x.unsqueeze(0)
            squeeze_output = True
        elif x.ndim == 2:
            xs = x
            squeeze_output = False
        else:
            raise ValueError(f"x must be 1D or 2D, got shape {x.shape} (ndim={x.ndim})")

        mem_list = []
        mac_list = []

        for row in xs:
            mem, mac = self._solve_one_(row)
            mem_list.append(mem)
            mac_list.append(mac)
        mem = np.stack(mem_list, axis=0)
        mac = np.stack(mac_list, axis=0)
        if squeeze_output:
            # Preserve original behavior for 1D input
            mem = mem[0]
            mac = mac[0]
        return mem, mac
    
    def mvm(self, x: torch.Tensor) -> torch.Tensor:
        return (x.to(torch.int) @ self.weights.to(torch.int))
def print_matrix_mult(A, B):
    """Print [A] * [B] with A vertically, B in middle, and result as column on right"""
    A = np.asarray(A)
    B = np.asarray(B)
    
    # Compute result
    result = A @ B
    
    # Print each row with result column
    mid = len(A) // 2
    result_strs = ''.join(f' {x:d}' for x in result)
    # print(result_strs)
    
    for i in range(len(A)):
        b_row_str = ' '.join(f'{x:d}' for x in B[i])
        
        if i == mid:
                print(f"[{A[i]:d}]  *  [{b_row_str}] = [{result_strs}]")
        else:
                print(f"[{A[i]:d}]     [{b_row_str}]")
    
    print()

def run(M,N,bits_per_cell,Rw,x,w,transient=False):
    sim = Simple_Sim(M, N,bits_per_cell, Rw, transient=transient)
    sim.set_weights(w)
    mem, mac = sim._solve_(x)
    mac = torch.from_numpy(mac)
    return mem,mac
    
def run_simulation(M,N,bits_per_cell,Rw,x,w,steps,transient=False):
    mem, mac = run(M,N,bits_per_cell,Rw,x,w,transient)
    digital = _digitise(mac, steps)
    # print(digital)
    return digital

    
if __name__ == "__main__":
    M, N = 32,32
    # M, N = 3,3
    Rw = 0.0001
    # Rw = 1
    bits_per_cell = 2
    transient = True
    per_input = 0.5
    per_weight = 0.5
    num_simulations = 1
    
    # weights = np.empty((M,N), dtype=int)
    # p = 2**bits_per_cell 
    # for i in range(M):
    #     weights[i] = np.tile(np.arange(p), N // p)
    # inputs = np.ones((num_simulations,M), dtype=bool)

    inputs = random_inputs(M,per_input,num_simulations)
    weights = random_weights(M,N,bits_per_cell,per_weight,1)


    # print(inputs)
    # print(weights)
    mvm = inputs @ weights
    mvm = torch.from_numpy(mvm)
    
    step = 60/(2**bits_per_cell -1)
    print(step)
    # adc_steps = np.full(N,40)
    adc_steps = np.full(N,step)
    adc_steps = torch.from_numpy(adc_steps)
    
    inputs = torch.from_numpy(inputs)
    weights = torch.from_numpy(weights)
    digital = run_simulation(M,N,bits_per_cell,Rw,inputs,weights,adc_steps,transient)
    print(mvm-digital)
    # print(sim.voltage_pulse_height)