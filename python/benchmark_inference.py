#!/usr/bin/env python3
import numpy as np
import xbar_simulator
import time
import os
import multiprocessing
from typing import List, Tuple

def create_random_test_case(size: int) -> Tuple[np.ndarray, np.ndarray]:
    """Create random input vector and weight matrix for testing."""
    weights = np.random.choice([True, False], size=(size, size))
    input_vector = np.ones(size, dtype=float) * xbar_simulator.voltage_pulse_height
    return input_vector, weights

def run_inference_benchmark(size: int, num_examples: int = 5) -> float:
    """Run inference benchmark for a given matrix size."""
    print(f"\nBenchmarking {size}x{size} crossbar...")
    
    
    # Initialize crossbar
    xbar = xbar_simulator.CrossbarSimulator(size, size)
    xbar.initialize_jart()
    xbar.set_parasitic_resistances(3, 1e20, 1e20, 5, 3, 2)
    
    xbar_simulator.simulation_num_threads = 2
    
    total_time = 0.0
    
    for i in range(num_examples):
        # Create random test case
        input_vector, weights = create_random_test_case(size)
        
        # Set weights
        xbar.set_weights(weights)
        
        # Time the inference
        start_time = time.time()
        output = xbar.run_inference(input_vector, dt=0)
        end_time = time.time()
        
        inference_time = end_time - start_time
        total_time += inference_time
        
        print(f"Example {i+1}/{num_examples}: {inference_time:.4f} seconds")
    
    avg_time = total_time / num_examples
    print(f"Average inference time for {size}x{size}: {avg_time:.4f} seconds")
    return avg_time

def main():
    # Matrix sizes to test
    sizes = [64, 128, 256]
    num_examples = 5
    
    print("Starting inference benchmarks...")
    print("Testing with", num_examples, "examples per size")
    
    results = []
    for size in sizes:
        avg_time = run_inference_benchmark(size, num_examples)
        results.append((size, avg_time))
    
    # Print summary
    print("\nBenchmark Summary:")
    print("-" * 40)
    print("Matrix Size | Average Time (s)")
    print("-" * 40)
    for size, time in results:
        print(f"{size:^11} | {time:^14.4f}")
    print("-" * 40)

if __name__ == "__main__":
    main() 