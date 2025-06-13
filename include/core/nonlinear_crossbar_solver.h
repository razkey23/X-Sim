#ifndef NONLINEAR_CROSSBAR_SOLVER_H_
#define NONLINEAR_CROSSBAR_SOLVER_H_

//#include "../memristor_model/JART_VCM_v1b_var.h"
#include "../memristor_model/Memristor.h"
#include "../core/threadpool.h"
#include "../eigen/Eigen/Dense"
#include "../eigen/Eigen/Sparse"

#include <vector>

Eigen::VectorXf BroydenInvSolve(
    // std::vector<std::vector<JART_VCM_v1b_var>> RRAM,
    std::vector<std::vector<std::unique_ptr<Memristor>>>& RRAM,
    std::vector<std::vector<bool>> access_transistors,
    Eigen::VectorXf Vguess, Eigen::SparseMatrix<float> G_ABCD,
    const Eigen::VectorXf E,
    const Eigen::VectorXf& Vappwl1, const Eigen::VectorXf& Vappwl2,
    const Eigen::VectorXf& Vappbl1, const Eigen::VectorXf& Vappbl2,
    const float Rswl1, const float Rswl2, const float Rsbl1, const float Rsbl2,
    const float Rwl, const float Rbl,
    Eigen::ConjugateGradient<Eigen::SparseMatrix<float>>& solver,
    const bool print = false
);

Eigen::VectorXf BroydenSolve(
    // std::vector<std::vector<JART_VCM_v1b_var>> RRAM,
    std::vector<std::vector<std::unique_ptr<Memristor>>>& RRAM,
    std::vector<std::vector<bool>> access_transistors,
    Eigen::VectorXf Vguess, Eigen::SparseMatrix<float> G_ABCD,
    const Eigen::VectorXf E,
    const Eigen::VectorXf& Vappwl1, const Eigen::VectorXf& Vappwl2,
    const Eigen::VectorXf& Vappbl1, const Eigen::VectorXf& Vappbl2,
    const float Rswl1, const float Rswl2, const float Rsbl1, const float Rsbl2,
    const float Rwl, const float Rbl,
    Eigen::ConjugateGradient<Eigen::SparseMatrix<float>>& solver,
    const bool print = false
);

Eigen::VectorXf NewtonRaphsonSolve(
    // std::vector<std::vector<JART_VCM_v1b_var>> RRAM,
    std::vector<std::vector<std::unique_ptr<Memristor>>>& RRAM,
    std::vector<std::vector<bool>> access_transistors,
    Eigen::VectorXf Vguess, Eigen::SparseMatrix<float> G_ABCD,
    const Eigen::VectorXf E,
    const Eigen::VectorXf& Vappwl1, const Eigen::VectorXf& Vappwl2,
    const Eigen::VectorXf& Vappbl1, const Eigen::VectorXf& Vappbl2,
    const float Rswl1, const float Rswl2, const float Rsbl1, const float Rsbl2,
    const float Rwl, const float Rbl,
    Eigen::ConjugateGradient<Eigen::SparseMatrix<float>>& solver,
    const bool print = false
);

Eigen::VectorXf FixedpointSolve(
    // std::vector<std::vector<JART_VCM_v1b_var>> RRAM,
    std::vector<std::vector<std::unique_ptr<Memristor>>>& RRAM,
    std::vector<std::vector<bool>> access_transistors,
    Eigen::VectorXf Vguess, Eigen::SparseMatrix<float> G_ABCD,
    const Eigen::VectorXf E,
    const Eigen::VectorXf& Vappwl1, const Eigen::VectorXf& Vappwl2,
    const Eigen::VectorXf& Vappbl1, const Eigen::VectorXf& Vappbl2,
    const float Rswl1, const float Rswl2, const float Rsbl1, const float Rsbl2,
    const float Rwl, const float Rbl,
    Eigen::ConjugateGradient<Eigen::SparseMatrix<float>>& solver,
    ThreadPool& pool,
    const bool print = false
);

#endif  // NONLINEAR_CROSSBAR_SOLVER_H_
