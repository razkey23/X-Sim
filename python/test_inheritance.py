#!/usr/bin/env python3
import numpy as np
import xbar_simulator

def load_vector(filename):
    """Load input vector from file with size header."""
    with open(filename, 'r') as f:
        size = int(f.readline().strip())
        vector = np.zeros(size)
        for i in range(size):
            vector[i] = float(f.readline().strip())
    return vector

def load_matrix(filename):
    """Load weight matrix from file with dimension header."""
    with open(filename, 'r') as f:
        # Read dimensions from first line
        rows, cols = map(int, f.readline().strip().split())
        # Read the matrix data
        matrix = np.zeros((rows, cols), dtype=bool)
        for i in range(rows):
            row = list(map(int, f.readline().strip().split()))
            matrix[i] = row
    return matrix

def main():
    # First test with a small 4x4 crossbar to verify functionality
    print("Testing with 4x4 crossbar first...")
    small_size = 4
    small_xbar = xbar_simulator.CrossbarSimulator(small_size, small_size)
    small_xbar.initialize_jart()
    small_xbar.set_parasitic_resistances(3, 1e20, 1e20, 5, 3, 2)
    
    # Create small test data
    small_weights = np.random.choice([True, False], size=(small_size, small_size))
    small_input = np.ones(small_size, dtype=float) * xbar_simulator.voltage_pulse_height
    
    # Test small crossbar
    small_xbar.set_weights(small_weights)
    small_output = small_xbar.run_inference(small_input)
    print("Small test successful!")
    print(f"Small output shape: {small_output.shape}")
    
    # Now try with the full data
    print("\nLoading full data...")
    try:
        input_vector = load_vector('../oracle/oracle_outs/input_vector.txt')
        weight_matrix = load_matrix('../oracle/oracle_outs/weight_matrix.txt')
        
        print(f"Input vector shape: {input_vector.shape}")
        print(f"Weight matrix shape: {weight_matrix.shape}")
        
        # Verify dimensions match
        if len(input_vector) != weight_matrix.shape[0]:
            raise ValueError("Input vector length doesn't match weight matrix rows")
        
        # Create crossbar simulator
        matrix_size = len(input_vector)
        xbar = xbar_simulator.CrossbarSimulator(matrix_size, matrix_size)
        
        # Initialize with JART memristors
        xbar.initialize_jart()
        #xbar_simulator.simulation_time_step = 0
        # Set parasitic resistances
        xbar.set_parasitic_resistances(3, 1e20, 1e20, 5, 3, 2)
        
        # Set the weights in the crossbar
        xbar.set_weights(weight_matrix)
        
        # Scale input vector by voltage pulse height
        input_scaled = input_vector * xbar_simulator.voltage_pulse_height
        
        # Run inference
        output = xbar.run_inference(input_scaled,dt=0)
        
        # Print results
        print("\nInput vector (first 10 elements):")
        print(input_vector[:10])
        print("\nWeight matrix (first 10x10 elements):")
        print(weight_matrix[:10, :10])
        print("\nOutput vector (first 10 elements):")
        print(output[:10])
        
        # Print statistics
        print("\nOutput statistics:")
        print(f"Mean: {np.mean(output):.6f}")
        print(f"Std: {np.std(output):.6f}")
        print(f"Min: {np.min(output):.6f}")
        print(f"Max: {np.max(output):.6f}")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        print("Please check the data files and their formats.")

if __name__ == "__main__":
    main() 