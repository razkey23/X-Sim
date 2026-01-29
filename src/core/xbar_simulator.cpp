#include "core/crossbar_simulator.h"
#include "core/nonlinear_crossbar_solver.h"
#include "core/simulation_settings.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <string>
#include <filesystem>

namespace fs = std::filesystem;

// Function to read a matrix from a binary file
std::vector<std::vector<int64_t>> readMatrixFromFile(const std::filesystem::path& filePath) {
    std::ifstream file(filePath, std::ios::binary);
    
    if (!file) {
        throw std::runtime_error("Failed to open file: " + filePath.string());
    }

    // Read rows and columns
    int64_t rows, cols;
    file.read(reinterpret_cast<char*>(&rows), sizeof(int64_t));
    file.read(reinterpret_cast<char*>(&cols), sizeof(int64_t));

    // Read data
    std::vector<std::vector<int64_t>> matrix(rows, std::vector<int64_t>(cols));

    // Read the data into the matrix
    for (int64_t i = 0; i < rows; ++i) {
        file.read(reinterpret_cast<char*>(matrix[i].data()), cols * sizeof(int64_t));
    }

    return matrix;
}

// Function to write output matrix to a binary file
void writeMatrixToFile(const std::vector<std::vector<float>>& matrix, const std::filesystem::path& filePath) {
    std::ofstream file(filePath, std::ios::binary);
    
    if (!file) {
        throw std::runtime_error("Failed to open file for writing: " + filePath.string());
    }

    int64_t rows = matrix.size();
    int64_t cols = matrix.empty() ? 0 : matrix[0].size();

    file.write(reinterpret_cast<const char*>(&rows), sizeof(int64_t));
    file.write(reinterpret_cast<const char*>(&cols), sizeof(int64_t));

    for (int64_t i = 0; i < rows; ++i) {
        file.write(reinterpret_cast<const char*>(matrix[i].data()), cols * sizeof(float));
    }
}

int main(int argc, char* argv[]) {
    // Check command line arguments
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <input_directory>" << std::endl;
        std::cout << "  The input directory should contain:" << std::endl;
        std::cout << "    - input.bin: Input voltage matrix" << std::endl;
        std::cout << "    - weight.bin: Weight matrix for the crossbar" << std::endl;
        return 1;
    }

    // Number of repeated inferences
    const int NUM_INFERENCES = 10000;

    // Set paths
    fs::path input_dir = fs::absolute(argv[1]);
    fs::path input_path = input_dir / "input.bin";
    fs::path weight_path = input_dir / "weight.bin";
    fs::path output_path = input_dir / "output.bin";
    fs::path mac_path = input_dir / "mac_iterations.bin";

    std::cout << "Reading input from: " << input_path << std::endl;
    std::cout << "Reading weights from: " << weight_path << std::endl;
    std::cout << "Will perform " << NUM_INFERENCES << " inferences with the same input" << std::endl;

    try {
        // Read input and weight data
        auto input_data = readMatrixFromFile(input_path);
        auto weight_data = readMatrixFromFile(weight_path);

        // Get dimensions
        int M = weight_data.size();
        int N = weight_data[0].size();
        std::cout << "Crossbar dimensions: " << M << "x" << N << std::endl;

        // Initialize crossbar
        CrossbarSimulator crossbar(M, N);

        // Convert weights to boolean matrix for crossbar
        std::vector<std::vector<int>> weights(M, std::vector<int>(N));
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                weights[i][j] = (weight_data[i][j] != 0);
            }
        }

        // Set weights in the crossbar
        crossbar.SetRRAM(weights);

        // Use only the first input vector
        if (input_data.empty()) {
            throw std::runtime_error("No input data found");
        }
        
        // Prepare output containers
        std::vector<std::vector<float>> mac_outputs_all_iterations;

        // Start timing
        auto start_time = std::chrono::high_resolution_clock::now();

        // Process the same input vector multiple times
        for (int iteration = 0; iteration < NUM_INFERENCES; iteration++) {
            std::cout << "Processing inference " << iteration+1 << "/" << NUM_INFERENCES << std::endl;
            
            // Set up input voltages
            Eigen::VectorXf Vappwl1 = Eigen::VectorXf::Zero(M);
            Eigen::VectorXf Vappwl2 = Eigen::VectorXf::Zero(M);
            Eigen::VectorXf Vappbl1 = Eigen::VectorXf::Zero(N);
            Eigen::VectorXf Vappbl2 = Eigen::VectorXf::Zero(N);
            
            // Set access transistors based on input (using first input vector)
            for (int j = 0; j < M; j++) {
                if (input_data[0][j] != 0) {
                    Vappwl1(j) = voltage_pulse_height;
                    crossbar.access_transistors[j] = std::vector<bool>(N, true);
                } else {
                    crossbar.access_transistors[j] = std::vector<bool>(N, false);
                }
            }

            // Initial voltage guess
            Eigen::VectorXf Vguess = Eigen::VectorXf::Zero(2*M*N);
            for (int i = 0; i < M; i++) {
                for (int j = 0; j < N; j++) {
                    Vguess(i*N + j) = Vappwl1(i);
                }
            }

            // Apply voltage and get output currents
            float dt = simulation_time_step;
            auto currents = crossbar.ApplyVoltage(Vguess, Vappwl1, Vappwl2, Vappbl1, Vappbl2, dt);
            
            // Calculate MAC outputs (sum of currents per column)
            std::vector<float> mac_outputs(N, 0.0f);
            for (int n = 0; n < N; n++) {
                for (int m = 0; m < M; m++) {
                    mac_outputs[n] += currents[m][n];
                }
            }
            
            // Store MAC results for this iteration
            mac_outputs_all_iterations.push_back(mac_outputs);
        }

        // End timing
        auto end_time = std::chrono::high_resolution_clock::now();
        auto execution_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
        std::cout << "Execution time: " << execution_time << " (ms)" << std::endl;

        // Write MAC outputs from all iterations
        std::ofstream outfile(mac_path, std::ios::binary);
        if (!outfile) {
            std::cout << "Failed to open file: " << mac_path << std::endl;
            return 1;
        }

        int64_t rows = mac_outputs_all_iterations.size();
        int64_t cols = mac_outputs_all_iterations.empty() ? 0 : mac_outputs_all_iterations[0].size();

        outfile.write(reinterpret_cast<const char*>(&rows), sizeof(int64_t));
        outfile.write(reinterpret_cast<const char*>(&cols), sizeof(int64_t));

        for (int64_t i = 0; i < rows; ++i) {
            outfile.write(reinterpret_cast<const char*>(mac_outputs_all_iterations[i].data()), cols * sizeof(float));
        }
        outfile.close();

        std::cout << "Results written to:" << std::endl;
        std::cout << "  - " << mac_path << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
} 