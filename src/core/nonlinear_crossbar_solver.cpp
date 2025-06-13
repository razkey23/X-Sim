#include "core/nonlinear_crossbar_solver.h"
#include "crossbar_model/linear_crossbar_solver.h"
#include "core/simulation_settings.h"
#include <omp.h>
#include <iostream>

// Note on the methods in this file:
//   Testing has found the fixed point to be fastest while having no additional drawbacks compared to the other methods
//   Additionally, the fixed point method seems to be the most stable


// Calculates the nodal voltages of a crossbar based on the crossbar and device resistances
// This function assumes the devices to be nonlinear, thus uses an iterative solving method to calculate the voltages
// Using this function with linear devices should still produce sufficient results, although slower than linear solvers
// The method used in this function is a variation on Broyden's method which instead calculates the inverse Jacobian
// There are two variations of this method: the 'good' and 'bad' Broyden's methods
// As using the 'bad' method is significantly faster and does not seem to have any drawbacks
// this method is used by default
// The 'good' method is also present in the code in the form of a comment
// The method in this function is based on: https://en.wikipedia.org/wiki/Broyden%27s_method
// The soving method takes an iteration limit of 100, after which the best guess is returned
Eigen::VectorXf BroydenInvSolve(
    std::vector<std::vector<std::unique_ptr<Memristor>>>& RRAM,  // Matrix containing the nonlinear deviceds
    std::vector<std::vector<bool>> access_transistors,
    Eigen::VectorXf Vguess,  // Initial guess for the nodal voltages. Supplying a zero vector acts as if no guess is given
    Eigen::SparseMatrix<float> G_ABCD,  // A partially precomputed version of the G_ABCD matrix
    const Eigen::VectorXf E,
    const Eigen::VectorXf& Vappwl1, const Eigen::VectorXf& Vappwl2,  // Applied voltages to the wordlines of the crossbar
    const Eigen::VectorXf& Vappbl1, const Eigen::VectorXf& Vappbl2,  // Applied voltages to the bitlines of the crossbar
    const float Rswl1, const float Rswl2, const float Rsbl1, const float Rsbl2,  // Resitances of the wordline and bitline voltage sources
    const float Rwl, const float Rbl,  // Wordline and bitline resistances of the crossbar
    Eigen::ConjugateGradient<Eigen::SparseMatrix<float>>& solver,
    const bool print  // Boolean variable to print some debug information, default false
) {
    if (RRAM.size() == 0) { return Eigen::VectorXf(0); }
    int M = RRAM.size();
    int N = RRAM[0].size();

    // Determine initial G
    Eigen::MatrixXf G(M, N);
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            if (access_transistors[i][j]) {
                float v = Vguess(i*N + j) - Vguess(i*N + j + M*N);
                G(i, j) = (float) 1./RRAM[i][j]->GetResistance(v);
            } else {
                G(i, j) = 0;
            }
        }
    }

    // Calculate initial Vout
    Eigen::VectorXf Vout = SolveCam(G, Vguess, G_ABCD, E, Vappwl1, Vappwl2, Vappbl1, Vappbl2, Rswl1, Rswl2, Rsbl1, Rsbl2, Rwl, Rbl, solver);
    Eigen::VectorXf Fv = Vout - Vguess;

    // Calculate initial inverse Jacobian (Scaled identity matrix, finite difference, or other)
    Eigen::MatrixXf B = Eigen::MatrixXf::Identity(2*M*N, 2*M*N);

    float a = 1.;

    int it_max = 100;
    int it = 0;
    while (true) {
        // For some very specific cases the SolveCam() funtion returns NANs as voltages
        // These circumstances are very rare and hard to recreate, therefore the exact cause for this bug is still unknown
        // Until this is fixed, the bug is circumvented by applying a small nudge to the current guess
        // Exstensive testing suggests that this is sufficient to find a solution, and at most one nudge per solving is encountered
        // In the case a 'nudge loop' is encountered, the best guess up untill that point is returned
        if (std::isnan(Fv.norm())) {
            if (print) { std::cout << "Nan norm detected, giving tiny nudge" << std::endl; }
            // std::cout << "Nan norm detected, giving tiny nudge" << std::endl;

            for (int i = 0; i < Vguess.size(); i++) {
                Vguess(i) += 1e-6 * ((Vguess(i) < 0) - (Vguess(i) > 0));
            }

            if (it >= it_max) {
                if (print) { std::cout << "Iteration limit reached: " << it << std::endl; }
                return Vguess;
            }

            it++;
            continue;
        }

        if (print) { std::cout << "Norm: " << Fv.norm() << std::endl; }
        // Check for convergence
        if (Fv.norm() < 1e-6 || it >= it_max) {
            if (print) {
                std::cout << "V:\n" << Vout << std::endl << std::endl;
                std::cout << "G:\n" << G << std::endl << std::endl;
                std::cout << "R:\n" << G.array().inverse() << std::endl << std::endl;
                std::cout << "Norm: " << (Vout - Vguess).norm() << std::endl;
                if (it >= it_max) {
                    std::cout << "Iteration limit reached: " << it << std::endl;
                } else {
                    std::cout << "Solved in " << it << " iterations" << std::endl;
                }
            }

            return Vout;
        }

        // Calculate dV
        Eigen::VectorXf dV = -B * Fv;

        // Updage Vguess
        Vguess += a * dV;
        
        // Determine G
        Eigen::MatrixXf G(M, N);
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                if (access_transistors[i][j]) {
                    float v = Vguess(i*N + j) - Vguess(i*N + j + M*N);
                    G(i, j) = (float) 1./RRAM[i][j]->GetResistance(v);
                } else {
                    G(i, j) = 0;
                }
            }
        }

        // Calculate V
        Vout = SolveCam(G, Vguess, G_ABCD, E, Vappwl1, Vappwl2, Vappbl1, Vappbl2, Rswl1, Rswl2, Rsbl1, Rsbl2, Rwl, Rbl, solver);
        Eigen::VectorXf Fv_new = Vout - Vguess;
        Eigen::VectorXf dF = Fv_new - Fv;

        // Update inverse Jacobian
        // B += ((dV - B * dF) / (dV.transpose() * B * dF + 1e-12)) * dV.transpose() * B; // 'Good' method
        B += ((dV - B * dF) / (dF.squaredNorm() + 1e-12)) * dF.transpose(); // 'Bad' method

        Fv = Fv_new;

        it += 1;
    }
}

// Calculates the nodal voltages of a crossbar based on the crossbar and device resistances
// This function assumes the devices to be nonlinear, thus uses an iterative solving method to calculate the voltages
// Using this function with linear devices should still produce sufficient results, although slower than linear solvers
// The method used in this function is Broyden's method
// The method in this function is based on: https://en.wikipedia.org/wiki/Broyden%27s_method
// The soving method takes an iteration limit of 100, after which the best guess is returned
Eigen::VectorXf BroydenSolve(
    // std::vector<std::vector<JART_VCM_v1b_var>> RRAM,  // Matrix containing the nonlinear deviceds
    std::vector<std::vector<std::unique_ptr<Memristor>>>& RRAM,
    std::vector<std::vector<bool>> access_transistors,
    Eigen::VectorXf Vguess,  // Initial guess for the nodal voltages. Supplying a zero vector acts as if no guess is given
    Eigen::SparseMatrix<float> G_ABCD,  // A partially precomputed version of the G_ABCD matrix
    const Eigen::VectorXf E,
    const Eigen::VectorXf& Vappwl1, const Eigen::VectorXf& Vappwl2,  // Applied voltages to the wordlines of the crossbar
    const Eigen::VectorXf& Vappbl1, const Eigen::VectorXf& Vappbl2,  // Applied voltages to the bitlines of the crossbar
    const float Rswl1, const float Rswl2, const float Rsbl1, const float Rsbl2,  // Resitances of the wordline and bitline voltage sources
    const float Rwl, const float Rbl,  // Wordline and bitline resistances of the crossbar
    Eigen::ConjugateGradient<Eigen::SparseMatrix<float>>& solver,
    const bool print  // Boolean variable to print some debug information, default false
) {
    if (RRAM.size() == 0) { return Eigen::VectorXf(0); }
    int M = RRAM.size();
    int N = RRAM[0].size();

    // Determine initial G
    Eigen::MatrixXf G(M, N);
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            if (access_transistors[i][j]) {
                float v = Vguess(i*N + j) - Vguess(i*N + j + M*N);
                G(i, j) = (float) 1./RRAM[i][j]->GetResistance(v);
            } else {
                G(i, j) = 0;
            }
        }
    }

    // Calculate initial Vout
    Eigen::VectorXf Vout = SolveCam(G, Vguess, G_ABCD, E, Vappwl1, Vappwl2, Vappbl1, Vappbl2, Rswl1, Rswl2, Rsbl1, Rsbl2, Rwl, Rbl, solver);
    Eigen::VectorXf Fv = Vout - Vguess;

    // Calculate initial Jacobian (Scaled identity matrix, finite difference, or other)
    Eigen::MatrixXf J = Eigen::MatrixXf::Identity(2*M*N, 2*M*N);

    float a = 1.;

    int it_max = 100;
    int it = 0;
    while (true) {
        // For some very specific cases the SolveCam() funtion returns NANs as voltages
        // These circumstances are very rare and hard to recreate, therefore the exact cause for this bug is still unknown
        // Until this is fixed, the bug is circumvented by applying a small nudge to the current guess
        // Exstensive testing suggests that this is sufficient to find a solution, and at most one nudge per solving is encountered
        // In the case a 'nudge loop' is encountered, the best guess up untill that point is returned
        if (std::isnan(Fv.norm())) {
            if (print) { std::cout << "Nan norm detected, giving tiny nudge" << std::endl; }
            // std::cout << "Nan norm detected, giving tiny nudge" << std::endl;

            for (int i = 0; i < Vguess.size(); i++) {
                Vguess(i) += 1e-6 * ((Vguess(i) < 0) - (Vguess(i) > 0));
            }

            if (it >= it_max) {
                if (print) { std::cout << "Iteration limit reached: " << it << std::endl; }
                return Vguess;
            }

            it++;
            continue;
        }

        if (print) { std::cout << "Norm: " << Fv.norm() << std::endl; }
        // Check for convergence
        if (Fv.norm() < 1e-6 || it >= it_max) {
            if (print) {
                std::cout << "V:\n" << Vout << std::endl << std::endl;
                std::cout << "G:\n" << G << std::endl << std::endl;
                std::cout << "R:\n" << G.array().inverse() << std::endl << std::endl;
                std::cout << "Norm: " << (Vout - Vguess).norm() << std::endl;
                if (it >= it_max) {
                    std::cout << "Iteration limit reached: " << it << std::endl;
                } else {
                    std::cout << "Solved in " << it << " iterations" << std::endl;
                }
            }

            return Vout;
        }

        // Calculate dV
        Eigen::VectorXf dV = J.partialPivLu().solve(-Fv);

        // Updage Vguess
        Vguess += a * dV;
        
        // Determine G
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                if (access_transistors[i][j]) {
                    float v = Vguess(i*N + j) - Vguess(i*N + j + M*N);
                    G(i, j) = (float) 1./RRAM[i][j]->GetResistance(v);
                } else {
                    G(i, j) = 0;
                }
            }
        }

        // Calculate V
        Vout = SolveCam(G, Vguess, G_ABCD, E, Vappwl1, Vappwl2, Vappbl1, Vappbl2, Rswl1, Rswl2, Rsbl1, Rsbl2, Rwl, Rbl, solver);
        Eigen::VectorXf Fv_new = Vout - Vguess;

        // Update Jacobian
        J += (((Fv_new - Fv) - J * dV) / (dV.squaredNorm() + 1e-12)) * dV.transpose();

        Fv = Fv_new;

        it += 1;
    }
}

// Calculates the nodal voltages of a crossbar based on the crossbar and device resistances
// This function assumes the devices to be nonlinear, thus uses an iterative solving method to calculate the voltages
// Using this function with linear devices should still produce sufficient results, although slower than linear solvers
// The method used in this function is the Newton-Raphon method and uses finite differences for partial differential approximation
// The method in this function is based on: https://en.wikipedia.org/wiki/Newton's_method
// The soving method takes an iteration limit of 100, after which the best guess is returned
Eigen::VectorXf NewtonRaphsonSolve(
    // std::vector<std::vector<JART_VCM_v1b_var>> RRAM,  // Matrix containing the nonlinear deviceds
    std::vector<std::vector<std::unique_ptr<Memristor>>>& RRAM,
    std::vector<std::vector<bool>> access_transistors,
    Eigen::VectorXf Vguess,  // Initial guess for the nodal voltages. Supplying a zero vector acts as if no guess is given
    Eigen::SparseMatrix<float> G_ABCD,  // A partially precomputed version of the G_ABCD matrix
    const Eigen::VectorXf E,
    const Eigen::VectorXf& Vappwl1, const Eigen::VectorXf& Vappwl2,  // Applied voltages to the wordlines of the crossbar
    const Eigen::VectorXf& Vappbl1, const Eigen::VectorXf& Vappbl2,  // Applied voltages to the bitlines of the crossbar
    const float Rswl1, const float Rswl2, const float Rsbl1, const float Rsbl2,  // Resitances of the wordline and bitline voltage sources
    const float Rwl, const float Rbl,  // Wordline and bitline resistances of the crossbar
    Eigen::ConjugateGradient<Eigen::SparseMatrix<float>>& solver,
    const bool print  // Boolean variable to print some debug information, default false
) {
    if (RRAM.size() == 0) { return Eigen::VectorXf(0); }
    int M = RRAM.size();
    int N = RRAM[0].size();

    Eigen::MatrixXf G(M, N);

    float a = 1.;

    int it_max = 100;
    int it = 0;
    while(true) {
        // Determine G
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                if (access_transistors[i][j]) {
                    float v = Vguess(i*N + j) - Vguess(i*N + j + M*N);
                    G(i, j) = (float) 1./RRAM[i][j]->GetResistance(v);
                } else {
                    G(i, j) = 0;
                }
            }
        }

        // Calculate Vout
        Eigen::VectorXf Vout = SolveCam(G, Vguess, G_ABCD, E, Vappwl1, Vappwl2, Vappbl1, Vappbl2, Rswl1, Rswl2, Rsbl1, Rsbl2, Rwl, Rbl, solver);
        Eigen::VectorXf Fv = Vout - Vguess;

        // For some very specific cases the SolveCam() funtion returns NANs as voltages
        // These circumstances are very rare and hard to recreate, therefore the exact cause for this bug is still unknown
        // Until this is fixed, the bug is circumvented by applying a small nudge to the current guess
        // Exstensive testing suggests that this is sufficient to find a solution, and at most one nudge per solving is encountered
        // In the case a 'nudge loop' is encountered, the best guess up untill that point is returned
        if (std::isnan(Fv.norm())) {
            if (print) { std::cout << "Nan norm detected, giving tiny nudge" << std::endl; }
            // std::cout << "Nan norm detected, giving tiny nudge" << std::endl;

            for (int i = 0; i < Vguess.size(); i++) {
                Vguess(i) += 1e-6 * ((Vguess(i) < 0) - (Vguess(i) > 0));
            }

            if (it >= it_max) {
                if (print) { std::cout << "Iteration limit reached: " << it << std::endl; }
                return Vguess;
            }

            it++;
            continue;
        }

        if (print) { std::cout << "Norm: " << Fv.norm() << std::endl; }
        // Check convergence
        if (Fv.norm() < 1e-6 || it >= it_max) {
            if (print) {
                std::cout << "V:\n" << Vout << std::endl << std::endl;
                std::cout << "G:\n" << G << std::endl << std::endl;
                std::cout << "R:\n" << G.array().inverse() << std::endl << std::endl;
                std::cout << "Norm: " << (Vout - Vguess).norm() << std::endl;
                if (it >= it_max) {
                    std::cout << "Iteration limit reached: " << it << std::endl;
                } else {
                    std::cout << "Solved in " << it << " iterations" << std::endl;
                }
            }

            return Vout;
        }

        // Calculate Jacobian
        Eigen::MatrixXf J = Eigen::MatrixXf::Zero(2*M*N, 2*M*N);
        for (int col = 0; col < 2*M*N; col++) {
            float delta = 1e-3;
            Vguess(col) += delta;

            
            for (int i = 0; i < M; i++) {
                for (int j = 0; j < N; j++) {
                    if (access_transistors[i][j]) {
                        float v = Vguess(i*N + j) - Vguess(i*N + j + M*N);
                        G(i, j) = (float) 1./RRAM[i][j]->GetResistance(v);
                    } else {
                        G(i, j) = 0;
                    }
                }
            }

            Eigen::VectorXf Fj = SolveCam(G, Vguess, G_ABCD, E, Vappwl1, Vappwl2, Vappbl1, Vappbl2, Rswl1, Rswl2, Rsbl1, Rsbl2, Rwl, Rbl, solver) - Vguess;

            J.col(col) = (Fj - Fv) / delta;

            Vguess(col) -= delta;
        }

        // Calculate dV
        Eigen::VectorXf dV = J.partialPivLu().solve(-Fv);

        // Update Vguess
        Vguess += a * dV;

        it++;
    }
}

// Calculates the nodal voltages of a crossbar based on the crossbar and device resistances
// This function assumes the devices to be nonlinear, thus uses an iterative solving method to calculate the voltages
// Using this function with linear devices should still produce sufficient results, although slower than linear solvers
// The method used in this function is a simple fixed point method
// The soving method takes an iteration limit of 100, after which the best guess is returned
Eigen::VectorXf FixedpointSolve(
    // std::vector<std::vector<JART_VCM_v1b_var>> RRAM,  // Matrix containing the nonlinear deviceds
    std::vector<std::vector<std::unique_ptr<Memristor>>>& RRAM,
    std::vector<std::vector<bool>> access_transistors,  // Matrix containing access transistors. Assumed to be ideal, thus true means on/closed, false means off/open
    Eigen::VectorXf Vguess,  // Initial guess for the nodal voltages. Supplying a zero vector acts as if no guess is given
    Eigen::SparseMatrix<float> G_ABCD,  // A partially precomputed version of the G_ABCD matrix
    const Eigen::VectorXf E,
    const Eigen::VectorXf& Vappwl1, const Eigen::VectorXf& Vappwl2,  // Applied voltages to the wordlines of the crossbar
    const Eigen::VectorXf& Vappbl1, const Eigen::VectorXf& Vappbl2,  // Applied voltages to the bitlines of the crossbar
    const float Rswl1, const float Rswl2, const float Rsbl1, const float Rsbl2,  // Resitances of the wordline and bitline voltage sources
    const float Rwl, const float Rbl,  // Wordline and bitline resistances of the crossbar
    Eigen::ConjugateGradient<Eigen::SparseMatrix<float>>& solver,
    ThreadPool& pool,
    const bool print  // Boolean variable to print some debug information, default false
) {
    if (RRAM.size() == 0) { return Eigen::VectorXf(0); }
    int M = RRAM.size();
    int N = RRAM[0].size();

    Eigen::MatrixXf G(M, N);

    float a = non_linear_fixed_point_a;

    int it_max = non_linear_fixed_point_it_max;
    int it = 0;
    while (true) {
        // Determine G
        /*#pragma omp parallel for 
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                if (access_transistors[i][j]) {
                    futures.emplace_back(pool.enqueue([&, i, j]() {
                            float v = Vguess(i*N + j) - Vguess(i*N + j + M*N);
                            G(i, j) = (float) 1./RRAM[i][j].GetResistance(v);
                    }));
                    //float v = Vguess(i*N + j) - Vguess(i*N + j + M*N);
                    //G(i, j) = (float) 1./RRAM[i][j]->GetResistance(v);
                } else {
                    G(i, j) = 0;
                }
            }
        }*/
        if (simulation_num_threads > 0) {
                std::vector<std::future<void>> futures;
                for (int i = 0; i < M; i++) {
                    for (int j = 0; j < N; j++) {
                        if (access_transistors[i][j]) {
                            futures.emplace_back(pool.enqueue([&, i, j]() {
                                float v = Vguess(i*N + j) - Vguess(i*N + j + M*N);
                                G(i, j) = (float) 1./RRAM[i][j]->GetResistance(v);
                            }));
                        } else {
                            G(i, j) = 0;
                        }
                    }
                }

                for (auto& fut : futures) {
                    fut.get();
                } 
            } else {
                for (int i = 0; i < M; i++) {
                    for (int j = 0; j < N; j++) {
                        if (access_transistors[i][j]) {
                            float v = Vguess(i*N + j) - Vguess(i*N + j + M*N);
                            G(i, j) = (float) 1./RRAM[i][j]->GetResistance(v);
                        } else {
                            G(i, j) = 0;
                        }
                    }
                }
            }
        
        // Calculate Vout
        Eigen::VectorXf Vout = SolveCam(G, Vguess, G_ABCD, E, Vappwl1, Vappwl2, Vappbl1, Vappbl2, Rswl1, Rswl2, Rsbl1, Rsbl2, Rwl, Rbl, solver);
        Eigen::VectorXf Fv = Vout - Vguess;

        // For some very specific cases the SolveCam() funtion returns NANs as voltages
        // These circumstances are very rare and hard to recreate, therefore the exact cause for this bug is still unknown
        // Until this is fixed, the bug is circumvented by applying a small nudge to the current guess
        // Exstensive testing suggests that this is sufficient to find a solution, and at most one nudge per solving is encountered
        // In the case a 'nudge loop' is encountered, the best guess up untill that point is returned
        if (std::isnan(Fv.norm())) {
            if (print) { std::cout << "Nan norm detected, giving tiny nudge" << std::endl; }
            // std::cout << "Nan norm detected, giving tiny nudge" << std::endl;

            for (int i = 0; i < Vguess.size(); i++) {
                Vguess(i) += non_linear_fixed_point_voltage_nudge * ((Vguess(i) < 0) - (Vguess(i) > 0));
            }

            if (it >= it_max) {
                if (print) { std::cout << "Iteration limit reached: " << it << std::endl; }
                return Vguess;
            }

            it++;
            continue;
        }

        if (print) { std::cout << "Norm: " << Fv.norm() << std::endl; }
        // std::cout << "Norm: " << Fv.norm() << std::endl;
        // Check convergence
        if (Fv.norm() < non_linear_fixed_point_criterion * sqrt(M*N) || it >= it_max) {
            if (print) {
                std::cout << "V:\n" << Vout << std::endl << std::endl;
                std::cout << "G:\n" << G << std::endl << std::endl;
                std::cout << "R:\n" << G.array().inverse() << std::endl << std::endl;
                std::cout << "Norm: " << Fv.norm() << std::endl;
                if (it >= it_max) {
                    std::cout << "Iteration limit reached: " << it << std::endl;
                } else {
                    std::cout << "Solved in " << it << " iterations" << std::endl;
                }
            }

            // if (it >= it_max) {
            //     std::cout << "Iteration limit reached: " << it << std::endl;
            //     std::cout << "Norm: " << Fv.norm() << std::endl;
            // }

            return Vout;
        }

        // Update Vguess
        Vguess += a * Fv;

        it++;
    }
}
