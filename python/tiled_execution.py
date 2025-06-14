import numpy as np
from crossbar import ParallelSim  # assuming this is in same directory
import time
from tqdm import tqdm
from typing import Dict, Any
import os
import pickle
def parallel_tiled_forward(inputs: np.ndarray,
                           weights: np.ndarray,
                           M: int, N: int,
                           mode="cs", transient=False,
                           max_workers: int = 8) -> np.ndarray:
    """
    Dot product using fixed-size (M×N) ParallelSim tiles.
    Edge tiles are padded with zeros so every block is exactly M×N.

    inputs  : (B, L) or (L,)
    weights : (L, C)
    returns : (B, C) or (C,)
    """
    single = inputs.ndim == 1
    if single:
        inputs = inputs[None, :]

    B, L = inputs.shape
    _, C = weights.shape
    out = np.zeros((B, C), dtype=int)

    # one simulator instance per tile size (re-used)
    sim = ParallelSim(M, N, mode=mode, transient=transient,
                      max_workers=max_workers)

    # reusable zero-filled buffers
    x_pad = np.zeros((B, M),  dtype=bool)
    w_pad = np.zeros((M, N),  dtype=bool)

    for r0 in tqdm(range(0, L, M),desc="running rows"):
        r1 = min(r0 + M, L)                  # actual rows in this slice
        r_len = r1 - r0

        # load input slice into padded buffer
        x_pad[:, :r_len] = inputs[:, r0:r1]
        x_pad[:, r_len:] = False             # zero out tail if reused

        # for c0 in tqdm(range(0, C, N),desc="running columns"):
        for c0 in range(0, C, N):
            c1 = min(c0 + N, C)              # actual cols in this slice
            c_len = c1 - c0

            # # load weight block into padded buffer
            # w_pad[:r_len, :c_len] = weights[r0:r1, c0:c1]
            # w_pad[:r_len, c_len:] = False
            # w_pad[r_len:, :]      = False
            w_pad[:r_len, :c_len] = weights[r0:r1, c0:c1]

            # Fill right padding columns (if any) with checkerboard
            if c_len < N:
                checker_cols = np.indices((r_len, N - c_len)).sum(axis=0) % 2 == 0
                w_pad[:r_len, c_len:] = checker_cols

            # Fill bottom padding rows (if any) with checkerboard
            if r_len < M:
                checker_rows = np.indices((M - r_len, N)).sum(axis=0) % 2 == 0
                w_pad[r_len:, :] = checker_rows
            # run simulation
            sim.set_weights(w_pad)
            y_pad = sim.run(x_pad)           # shape (B, N)

            # accumulate only the valid columns
            out[:, c0:c1] += y_pad[:, :c_len]

    return out[0] if single else out

def diff_stats(A: np.ndarray, B: np.ndarray) -> Dict[str, Any]:
    """
    Compare two same-shaped NumPy arrays element-wise.

    Returns a dictionary with:
      • row_abs_sum  : 1-D array of row-wise |A−B| sums
      • row_diff_cnt : 1-D array of row-wise (# of elements where A≠B)
      • total_abs_sum: scalar, sum of all |A−B|
      • total_diff_cnt: scalar, total # of differing elements
      • row_pct_diff : 1-D array, percentage of differing elements per row
    """
    if A.shape != B.shape:
        raise ValueError("A and B must have the same shape")

    abs_diff       = np.abs(A - B)
    bool_diff      = abs_diff != 0

    row_abs_sum    = abs_diff.sum(axis=1)
    row_diff_cnt   = bool_diff.sum(axis=1)

    total_abs_sum  = row_abs_sum.sum()
    total_diff_cnt = row_diff_cnt.sum()

    row_pct_diff   = row_diff_cnt / A.shape[1] * 100.0  # (%) per row

    return dict(
        row_abs_sum=row_abs_sum,
        row_diff_cnt=row_diff_cnt,
        total_abs_sum=total_abs_sum,
        total_diff_cnt=total_diff_cnt,
        row_pct_diff=row_pct_diff,
    )

def test_correctness(L,C,B,M,N,P,tran):
    psim = ParallelSim(L, C, mode="gs", transient=tran)

    inputs = psim.random_inputs(B,P)
    weights = psim.random_weights(P)

    # print(inputs.shape)
    # print(weights.shape)
    start = time.time()
    print(f"Running parallel tiled simulation... for gs transient {tran}")
    out_gs = parallel_tiled_forward(inputs, weights, M, N, mode="gs", transient=tran)
    print(f"time : {time.time() - start}")
    start = time.time()
    print(f"Running parallel tiled simulation... for cs transient {tran}")
    out_cs = parallel_tiled_forward(inputs, weights, M, N, mode="cs", transient=tran)
    print(f"time : {time.time() - start}")

    psim.set_weights(weights)
    ref = psim.mvm(inputs)

    # print("cs")
    cs_stats = diff_stats(ref,out_cs)
    # for k, v in cs_stats.items():
    #     print(f"{k}: {v}")
    
    # print("gs")
    gs_stats = diff_stats(ref,out_gs)
    # for k, v in gs_stats.items():
    #     print(f"{k}: {v}")
    
    overall = {"gs":gs_stats,"cs":cs_stats,"tran":tran}
    return overall

if __name__ == "__main__":
    iterations = 10
    # L, C = 128, 64     # large matrix
    L, C = 784, 512     # large matrix
    B = 2              # batch size
    M, N = 32,32      # tile size
    P = 50

    stats = []
    save_path = os.path.join("/home/earapidis/Fast-Crossbar-Sim/python/data/test.pkl")

    for i in range(iterations):
        for tran in [False,True]:
        # for tran in [True,False]:
            overall = test_correctness(L,C,B,M,N,P,tran)
            stats.append(overall)
            # print(overall)
    with open(save_path,"wb") as f:
        pickle.dump(stats,f)
    
    with open(save_path,"rb") as f:
        loaded_stats = pickle.load(f)
        print(loaded_stats)