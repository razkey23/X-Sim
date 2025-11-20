from . import xbar_simulator                                          # C++ bindings
import numpy as np

class Simple_Sim(xbar_simulator.CrossbarSimulator):
    """
    A *single-vector* simulator.  One instance lives in one process.
    """
    def __init__(self, M: int, N: int, Rw: int, transient=False):
        super().__init__(M, N)
        self.M, self.N, self.Rw, self.transient = M, N, Rw, transient
        self.initialize_jart()

        parasitics = (self.Rw, 1e20, 1e20, self.Rw, self.Rw, self.Rw)
        self.set_parasitic_resistances(*parasitics)

        self.weights: np.ndarray | None = None             # set later
    
    def set_weights(self, w: np.ndarray) -> None:
        super().set_weights(w)
        self.weights = w


    def _solve_one_(self, x: np.ndarray):
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
    def _solve_(self, x: np.ndarray):
        """Return MAC (amps) for multiple Boolean input row vectors."""
        if x.ndim == 1:
            xs = x[np.newaxis, :]
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
    
    def random_inputs(self, p: float,samples: int=1, seed=None) -> np.ndarray:
        rng   = np.random.default_rng(seed)
        k     = int(round((p if p<=1 else p/100) * self.M))
        rows  = np.zeros((samples, self.M), dtype=bool)
        for r in rows: r[rng.choice(self.M, k, replace=False)] = True
        if samples==1:
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
if __name__ == "__main__":
    M, N = 32,32
    Rw = 1
    sim = Simple_Sim(M, N, Rw, transient=False)

    inputs = sim.random_inputs(p=0.5, samples=5, seed=42)
    weights = sim.random_weights(p=0.5, seed=42)
    sim.set_weights(weights)
    mem, mac = sim._solve_(inputs)
    # print("Mem (µA):", mem)
    print("MAC (µA):", mac)

    mvm = sim.mvm(inputs)
    print("MVM (digital):", mvm)