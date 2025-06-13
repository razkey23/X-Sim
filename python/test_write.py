#!/usr/bin/env python
"""
test_fefet_write.py
===================

Complete programming cycle test for a single FeFET cell in a 2x2 crossbar:
1. Tracks resistance through multiple programming cycles
2. Only programs and reads the top-left cell (0,0)
3. Other cells remain in their initial state
"""

import numpy as np
import xbar_simulator as xs
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------
#  USER-TUNABLE PARAMETERS
# --------------------------------------------------------------------------
READ_WL   = 1.0      # word-line voltage during read   [V]
PROG_WL   = 3.0      # programming pulse amplitude     [V]
PROG_LEN  = 100e-9   # pulse width                     [s]
VDS_READ  = 0.1      # drain-source sense voltage      [V]
DT        = xs.simulation_time_step   # Δt exported by the C++ side
N_CYCLES  = 5        # number of complete cycles to run
# --------------------------------------------------------------------------

def id_read(sim, wl_volt):
    """Read the drain current with given WL level (dt=0 ⇒ no time advance)."""
    xs.voltage_pulse_height = wl_volt
    # Create input vector with VDS_READ applied to first bitline
    input_vec = np.array([VDS_READ, 0.0], dtype=np.float32)
    # Only read the current from the first cell (0,0)
    current = sim.run_inference(input_vec, dt=0.0)[0]   # A
    return current

def apply_program_pulse(sim):
    """
    Drive PROG_WL for PROG_LEN using multiple small
    run_inference() calls so that the FeFET time kernel integrates properly.
    """
    xs.voltage_pulse_height = PROG_WL
    n_steps = int(np.ceil(PROG_LEN / DT))
    # Apply programming voltage to first wordline, ground others
    input_vec = np.array([PROG_WL, 0.0], dtype=np.float32)
    sim.run_multiple_inferences(input_vec, n_steps, dt=DT)   # side-effect only

def amps_to_resistance(I_amp):
    """Convert ID to R using the assumed VDS_READ bias (Ω)."""
    if abs(I_amp) < 1e-18:
        return np.inf
    return VDS_READ / I_amp

def plot_resistance_cycle(resistances, states):
    """Plot the resistance values through the programming cycle."""
    plt.figure(figsize=(10, 6))
    # Filter out infinite values for plotting
    finite_resistances = [r if np.isfinite(r) else 1e15 for r in resistances]
    plt.semilogy(finite_resistances, 'b-o', label='Resistance')
    plt.axhline(y=1e6, color='r', linestyle='--', label='1 MΩ')
    plt.xlabel('Step')
    plt.ylabel('Resistance (Ω)')
    plt.title('FeFET Resistance Through Programming Cycle (Cell 0,0)')
    plt.grid(True)
    plt.legend()
    
    # Add state labels
    for i, state in enumerate(states):
        plt.text(i, finite_resistances[i], f'{"HRS" if state == 0 else "LRS"}', 
                ha='center', va='bottom')
    
    plt.savefig('fefet_programming_cycle.png')
    plt.close()

# --------------------------------------------------------------------------
#  MAIN
# --------------------------------------------------------------------------
if __name__ == "__main__":
    # 2 × 2 cross-bar -------------------------------------------------------
    sim = xs.CrossbarSimulator(2, 2)
    sim.initialize_fefet()
    
    # Set parasitic resistances (Rswl1, Rswl2, Rsbl1, Rsbl2, Rwl, Rbl)
    sim.set_parasitic_resistances(3, 1e20, 1e20, 5, 3, 2)

    # Initialize arrays to store results
    resistances = []
    states = []
    step = 0

    print("\nStarting FeFET Programming Cycle Test (Cell 0,0)")
    print("==============================================")
    print(f"Read voltage: {READ_WL}V")
    print(f"Program voltage: {PROG_WL}V")
    print(f"VDS read bias: {VDS_READ}V")

    # Start in HRS (Logic '0') for all cells
    weights = np.zeros((2, 2), dtype=bool)
    sim.set_weights(weights)
    I = id_read(sim, READ_WL)
    R = amps_to_resistance(I)
    resistances.append(R)
    states.append(0)
    print(f"\nStep {step}: Initial HRS")
    print(f"Current: {I:.4e} A")
    print(f"Resistance: {R:.4e} Ω")
    step += 1

    # Run multiple cycles
    for cycle in range(N_CYCLES):
        print(f"\nCycle {cycle + 1}/{N_CYCLES}")
        print("-----------------")

        # Program to LRS (Logic '1') for cell (0,0) only
        print("\nProgramming cell (0,0) to LRS...")
        weights[0, 0] = 1  # Set only cell (0,0) to Logic '1'
        sim.set_weights(weights)
        apply_program_pulse(sim)
        I = id_read(sim, READ_WL)
        R = amps_to_resistance(I)
        resistances.append(R)
        states.append(1)
        print(f"Step {step}: LRS")
        print(f"Current: {I:.4e} A")
        print(f"Resistance: {R:.4e} Ω")
        step += 1

        # Program back to HRS (Logic '0') for cell (0,0) only
        print("\nProgramming cell (0,0) to HRS...")
        weights[0, 0] = 0  # Set only cell (0,0) to Logic '0'
        sim.set_weights(weights)
        apply_program_pulse(sim)
        I = id_read(sim, READ_WL)
        R = amps_to_resistance(I)
        resistances.append(R)
        states.append(0)
        print(f"Step {step}: HRS")
        print(f"Current: {I:.4e} A")
        print(f"Resistance: {R:.4e} Ω")
        step += 1

    # Plot results
    plot_resistance_cycle(resistances, states)
    print("\nResults have been plotted to 'fefet_programming_cycle.png'")

    # Print summary statistics
    print("\nSummary Statistics")
    print("=================")
    hrs_resistances = [R for R, s in zip(resistances, states) if s == 0]
    lrs_resistances = [R for R, s in zip(resistances, states) if s == 1]
    
    print(f"Average HRS resistance: {np.mean(hrs_resistances):.4e} Ω")
    print(f"Average LRS resistance: {np.mean(lrs_resistances):.4e} Ω")
    if np.isfinite(np.mean(hrs_resistances)) and np.isfinite(np.mean(lrs_resistances)):
        print(f"Average R_off/R_on ratio: {np.mean(hrs_resistances)/np.mean(lrs_resistances):.2f}×")
    else:
        print("Could not calculate R_off/R_on ratio due to infinite resistances")
