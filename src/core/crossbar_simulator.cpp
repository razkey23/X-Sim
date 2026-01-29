#include "core/crossbar_simulator.h"

#include "core/nonlinear_crossbar_solver.h"
#include "crossbar_model/linear_crossbar_solver.h"

#include <iostream>

/*void CrossbarSimulator::SetRRAM(std::vector<std::vector<bool>> weights) {
    assert(weights.size() == RRAM.size());
    assert(weights[0].size() == RRAM[0].size());

    for (int i = 0; i < weights.size(); i++) {
        for (int j = 0; j < weights[0].size(); j++) {
            if (weights[i][j]) { RRAM[i][j].Nreal = RRAM[i][j].Ndiscmax; }
            else { RRAM[i][j].Nreal = RRAM[i][j].Ndiscmin; }
            RRAM[i][j].Treal = RRAM[i][j].T0;
        }
    }
}*/

void CrossbarSimulator::SetAccessTransistors(std::vector<bool> gate_lines) {
    assert(gate_lines.size() == RRAM.size());
    for (int m = 0; m < M; m++) {
        if (gate_lines[m]) {
            access_transistors[m] = std::vector<bool>(N, true);
        } else {
            access_transistors[m] = std::vector<bool>(N, false);
        }
    }
}

void CrossbarSimulator::SetRRAM(std::vector<std::vector<int>> weights) {
    assert(weights.size() == RRAM.size());
    assert(weights[0].size() == RRAM[0].size());

    for (int i = 0; i < weights.size(); i++) {
        for (int j = 0; j < weights[0].size(); j++) {
            RRAM[i][j]->SetWeight(weights[i][j]);
        }
    }
}

Eigen::VectorXf CrossbarSimulator::NonlinearSolve(
    Eigen::VectorXf Vguess,
    const Eigen::VectorXf& Vappwl1, const Eigen::VectorXf& Vappwl2,
    const Eigen::VectorXf& Vappbl1, const Eigen::VectorXf& Vappbl2,
    std::string method
) {
    Eigen::VectorXf E = ComputeE(M, N, Vappwl1, Vappwl2, Vappbl1, Vappbl2, Rswl1, Rswl2, Rsbl1, Rsbl2, Rwl, Rbl);

    if (method == "fixed-point") {
        return FixedpointSolve(RRAM, access_transistors, Vguess, partial_G_ABCD, E, Vappwl1, Vappwl2, Vappbl1, Vappbl2, Rswl1, Rswl2, Rsbl1, Rsbl2, Rwl, Rbl, linear_solver,pool);
    } else if (method == "NewtonRaphson") {
        return NewtonRaphsonSolve(RRAM, access_transistors, Vguess, partial_G_ABCD, E, Vappwl1, Vappwl2, Vappbl1, Vappbl2, Rswl1, Rswl2, Rsbl1, Rsbl2, Rwl, Rbl, linear_solver);
    } else if (method == "Broyden") {
        return BroydenSolve(RRAM, access_transistors, Vguess, partial_G_ABCD, E, Vappwl1, Vappwl2, Vappbl1, Vappbl2, Rswl1, Rswl2, Rsbl1, Rsbl2, Rwl, Rbl, linear_solver);
    } else if (method == "BroydenInv") {
        return BroydenInvSolve(RRAM, access_transistors, Vguess, partial_G_ABCD, E, Vappwl1, Vappwl2, Vappbl1, Vappbl2, Rswl1, Rswl2, Rsbl1, Rsbl2, Rwl, Rbl, linear_solver);
    } else {
        return FixedpointSolve(RRAM, access_transistors, Vguess, partial_G_ABCD, E, Vappwl1, Vappwl2, Vappbl1, Vappbl2, Rswl1, Rswl2, Rsbl1, Rsbl2, Rwl, Rbl, linear_solver,pool);
    }
}

std::vector<std::vector<float>> CrossbarSimulator::ApplyVoltage(
    Eigen::VectorXf Vguess,
    const Eigen::VectorXf& Vappwl1, const Eigen::VectorXf& Vappwl2,
    const Eigen::VectorXf& Vappbl1, const Eigen::VectorXf& Vappbl2,
    float dt, std::string method
) {
    Eigen::VectorXf Vout = NonlinearSolve(Vguess, Vappwl1, Vappwl2, Vappbl1, Vappbl2, method);

    // Initialize Iout with zeros
    //std::vector<std::vector<float>> Iout(M, std::vector<float>(N, 0.0f));
    //std::mutex iout_mutex;
    std::vector<std::vector<float>> Iout(M, std::vector<float>(N));
    
    if (simulation_num_threads > 0) {
        std::vector<std::future<void>> futures;
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                if (access_transistors[i][j]) {
                    futures.emplace_back(pool.enqueue([&, i, j]() {
                        float v = Vout(i*N + j) - Vout(i*N + j + M*N);
                        Iout[i][j] = RRAM[i][j]->ApplyVoltage(v, dt);
                    }));
                } else {
                    Iout[i][j] = 0.;
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
                    float v = Vout(i*N + j) - Vout(i*N + j + M*N);
                    Iout[i][j] = RRAM[i][j]->ApplyVoltage(v, dt);
                } else {
                    Iout[i][j] = 0.;
                }
            }
        }
    }

    return Iout;
}

std::vector<float> CrossbarSimulator::CalculateIout(Eigen::VectorXf Vout) {
    std::vector<float> Iout;
    for (int j = 0; j < N; j++) {
        float Ioutj = 0.;
        for (int i = 0; i < M; i++) {
            float v = Vout(i*N + j) - Vout(i*N + j + M*N);
            Ioutj += RRAM[i][j]->ApplyVoltage(v, 0);
        }
        Iout.push_back(Ioutj);
    }
    return Iout;
}

void CrossbarSimulator::Simulate(
        const std::vector<bool> Vwl1, const std::vector<bool> Vwl2,  // Applied voltages to the wordlines of the crossbar
        const std::vector<bool> Vbl1, const std::vector<bool> Vbl2,  // Applied voltages to the bitlines of the crossbar
        const std::vector<std::vector<int>> weights,  // matrix of weights corresponding to each crossbar. Writing weights is not simulated, instead the SetRRAM() function is used
        const std::vector<std::array<float, 2>> waveform,  // Description of the waveform. Each element is a breakpoint consisting of a timestamp and a voltage. The wave is constructed by linearly interpolating between two breakpoints
        const float dt,  // Time step size used for simulation
        std::vector<std::vector<float>>& Iout,  // Output matrix for currents running through individual memristors. Will be cleared before use
        std::vector<float>& Iout_MAC  // Output vector for column currents. Will be cleared before use
) {
    SetRRAM(weights);
    SetAccessTransistors(Vwl1);
    
    Eigen::VectorXf Vappwl1 = Eigen::VectorXf::Zero(M);
    Eigen::VectorXf Vappwl2 = Eigen::VectorXf::Zero(M);
    Eigen::VectorXf Vappbl1 = Eigen::VectorXf::Zero(N);
    Eigen::VectorXf Vappbl2 = Eigen::VectorXf::Zero(N);

    Eigen::VectorXf Vguess = Eigen::VectorXf::Zero(2*M*N);

    float V = 0;
    float t = 0;

    std::vector<std::vector<std::vector<float>>> Iwave;

    // A voltage wave is applied to each row that is true in Vwl1
    // The wave is simulated by linearly interpolating between two breakpoints of the waveform definition
    for (int i = 0; i < waveform.size(); i++) {
        float dv = (waveform[i][0] - V) / ((waveform[i][1] - t) / dt);
        while (t < waveform[i][1]) {
            for (int i = 0; i < M; i++) {
                for (int j = 0; j < N; j++) {
                    Vguess(i*N + j) = Vappwl1(i);
                }
            }

            Iwave.push_back(ApplyVoltage(Vguess, Vappwl1, Vappwl2, Vappbl1, Vappbl2, dt));

            for (int j = 0; j < M; j++) {
                if (Vwl1[j]) {
                    Vappwl1(j) += dv;
                }
            }

            V += dv;
            t += dt;
        }
    }
    
    // Output current is calculed by taking the average current at the peaks of the waveform
    Iout.clear();
    for (int m = 0; m < M; m++) {
        std::vector<float> row;
        for (int n = 0; n < N; n++) {
            float Iavg = 0;
            for (int j = voltage_pulse_rise_time/simulation_time_step; j < (voltage_pulse_width - voltage_pulse_fall_time)/simulation_time_step; j++) {
                Iavg += Iwave[j][m][n];
            }
            Iavg /= (voltage_pulse_width - voltage_pulse_rise_time - voltage_pulse_fall_time) / simulation_time_step;
            row.push_back(Iavg);
        }
        Iout.push_back(row);
    }

    // Output MAC current is the sum of all memristor currents of a column 
    Iout_MAC.clear();
    for (int n = 0; n < N; n++) {
        float IMAC = 0;
        for (int m = 0; m < M; m++) {
            IMAC += Iout[m][n];
        }
        Iout_MAC.push_back(IMAC);
    }
}
