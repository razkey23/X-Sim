#ifndef CROSSBAR_SIMULATOR_H_
#define CROSSBAR_SIMULATOR_H_

#include "../memristor_model/JART_VCM_v1b_var.h"
#include "../crossbar_model/linear_crossbar_solver.h"
#include "../memristor_model/FeFET.h"
#include "../memristor_model/Memristor.h"
#include "../core/threadpool.h"
#include "../eigen/Eigen/Dense"
#include "../eigen/Eigen/Sparse"
#include <vector>
#include <string>

class CrossbarSimulator {
    public:
    int M;
    int N;
    int bits_per_cell;
    // std::vector<std::vector<JART_VCM_v1b_var>> RRAM;
    std::vector<std::vector<std::unique_ptr<Memristor>>> RRAM;
    std::vector<std::vector<bool>> access_transistors;

    float Rswl1;
    float Rswl2;
    float Rsbl1;
    float Rsbl2;
    float Rwl;
    float Rbl;
    

    Eigen::SparseMatrix<float> partial_G_ABCD;

    Eigen::ConjugateGradient<Eigen::SparseMatrix<float>> linear_solver;
    ThreadPool pool;
    
    CrossbarSimulator(int M, int N, int bits_per_cell_=1) : linear_solver(), pool(simulation_num_threads) {
        this->M = M;
        this->N = N;
        this->bits_per_cell = bits_per_cell_;

        // Initialize RRAM
        // RRAM = std::vector<std::vector<JART_VCM_v1b_var>>(M, std::vector<JART_VCM_v1b_var>(N, JART_VCM_v1b_var()));
        RRAM.resize(M);
        for (int i = 0; i < M; ++i) {
            RRAM[i].resize(N);
        }
        // Initialize access transistors to all on
        access_transistors = std::vector<std::vector<bool>>(M, std::vector<bool>(N, true));

        // Initialize parasitics to some default
        Rswl1 = 1;
        Rswl2 = INFINITY;
        Rsbl1 = INFINITY;
        Rsbl2 = 1;
        Rwl = 1;
        Rbl = 1;

        // Precompute partial_G_ABCD
        partial_G_ABCD = PartiallyPrecomputeG_ABCD(M, N, Rswl1, Rswl2, Rsbl1, Rsbl2, Rwl, Rbl);
    }
    
    template <typename MemristorType>
    void Initialize() {
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                RRAM[i][j] = std::make_unique<MemristorType>(this->bits_per_cell);
            }
        }
    }

    void SetAccessTransistors(std::vector<bool> gate_lines);
    void SetRRAM(std::vector<std::vector<int>> weights);

    Eigen::VectorXf NonlinearSolve(
        Eigen::VectorXf Vguess,
        const Eigen::VectorXf& Vappwl1, const Eigen::VectorXf& Vappwl2,
        const Eigen::VectorXf& Vappbl1, const Eigen::VectorXf& Vappbl2,
        std::string method = "fixed-point"
    );
    std::vector<std::vector<float>> ApplyVoltage(
        Eigen::VectorXf Vguess,
        const Eigen::VectorXf& Vappwl1, const Eigen::VectorXf& Vappwl2,
        const Eigen::VectorXf& Vappbl1, const Eigen::VectorXf& Vappbl2,
        float dt, std::string method = "fixed-point"
    );
    std::vector<float> CalculateIout(Eigen::VectorXf Vout);

    void Simulate(
        const std::vector<bool> Vwl1, const std::vector<bool> Vwl2,
        const std::vector<bool> Vbl1, const std::vector<bool> Vbl2,
        const std::vector<std::vector<int>> weights,
        const std::vector<std::array<float, 2>> waveform,
        const float dt,
        std::vector<std::vector<float>>& Iout,
        std::vector<float>& Iout_MAC
    ); 
};

#endif  // CROSSBAR_SIMULATOR_H_
