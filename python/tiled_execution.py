import numpy as np
from crossbar import ParallelSim  # assuming this is in same directory
import time
from tqdm import tqdm

def parallel_tiled_forward(inputs: np.ndarray,
                           weights: np.ndarray,
                           M: int, N: int,
                           mode="cs", transient=False,
                           max_workers: int = 4) -> np.ndarray:
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

def test_correctness():
    # L, C = 256, 256    # large matrix
    L, C = 784, 512     # large matrix
    B = 1              # batch size
    M, N = 32,32      # tile size
    P = 50
    psim = ParallelSim(L, C, mode="gs", transient=False)

    inputs = psim.random_inputs(B,P)
    weights = psim.random_weights(P)

    print(inputs.shape)
    print(weights.shape)
    start = time.time()
    print("Running parallel tiled simulation...")
    out_cs = parallel_tiled_forward(inputs, weights, M, N, mode="cs", transient=False)
    out_gs = parallel_tiled_forward(inputs, weights, M, N, mode="gs", transient=False)

    print("Running reference matmul...")
    print(f"time : {time.time() - start}")
    psim.set_weights(weights)
    ref = psim.mvm(inputs)
    # ref = inputs.astype(int) @ weights.astype(int)
    diff_gs = ref - out_gs
    diff_cs = ref - out_cs
    idx = np.where(diff_gs!=0)
    print(idx, np.sum(np.abs(diff_gs)))
    idx = np.where(diff_cs!=0)
    print(idx, np.sum(np.abs(diff_cs)))

    # print(ref-out)


if __name__ == "__main__":
    test_correctness()
