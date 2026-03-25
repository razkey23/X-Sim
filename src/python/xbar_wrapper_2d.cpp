#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>

#include <Eigen/Core>

#include "core/crossbar_simulator.h"
#include "core/simulation_settings.h"

#include <algorithm>
#include <future>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace py = pybind11;

// Re-export helpers so the module can be used independently from xbar_wrapper.cpp.
void set_eigen_threads_2d(int num_threads) {
    Eigen::setNbThreads(num_threads);
}

int get_eigen_threads_2d() {
    return Eigen::nbThreads();
}

enum class DeviceKind {
    Unknown = 0,
    JART = 1,
    FeFET = 2,
};

class PyXbarSimulator2D {
private:
    CrossbarSimulator simulator_;
    std::vector<std::vector<int>> weights_vec_;

    DeviceKind device_kind_ = DeviceKind::Unknown;

    float Rswl1_;
    float Rswl2_;
    float Rsbl1_;
    float Rsbl2_;
    float Rwl_;
    float Rbl_;

    void configure_simulator(CrossbarSimulator& sim) const {
        if (device_kind_ == DeviceKind::JART) {
            sim.Initialize<JART_VCM_v1b_var>();
        } else if (device_kind_ == DeviceKind::FeFET) {
            sim.Initialize<FeFET>();
        } else {
            throw std::runtime_error("Memristor type not initialized. Call initialize_jart() or initialize_fefet().");
        }

        sim.Rswl1 = Rswl1_;
        sim.Rswl2 = Rswl2_;
        sim.Rsbl1 = Rsbl1_;
        sim.Rsbl2 = Rsbl2_;
        sim.Rwl = Rwl_;
        sim.Rbl = Rbl_;
        sim.partial_G_ABCD = PartiallyPrecomputeG_ABCD(
            sim.M, sim.N,
            sim.Rswl1, sim.Rswl2,
            sim.Rsbl1, sim.Rsbl2,
            sim.Rwl, sim.Rbl
        );

        if (!weights_vec_.empty()) {
            sim.SetRRAM(weights_vec_);
        }
    }

    static void validate_input_2d_shape(const py::buffer_info& buf, int M) {
        if (buf.ndim != 2) {
            throw std::runtime_error("Input must be a 2-D array with shape (samples, M)");
        }
        if (buf.shape[1] != M) {
            throw std::runtime_error("Input second dimension must equal simulator M");
        }
    }

    static void run_single(
        CrossbarSimulator& sim,
        const float* vin,
        float dt,
        const std::string& method,
        std::vector<std::vector<float>>& Iout,
        std::vector<float>& Iout_MAC
    ) {
        const size_t M = static_cast<size_t>(sim.M);
        const size_t N = static_cast<size_t>(sim.N);

        Eigen::VectorXf Vappwl1 = Eigen::VectorXf::Zero(M);
        Eigen::VectorXf Vappwl2 = Eigen::VectorXf::Zero(M);
        Eigen::VectorXf Vappbl1 = Eigen::VectorXf::Zero(N);
        Eigen::VectorXf Vappbl2 = Eigen::VectorXf::Zero(N);

        for (size_t m = 0; m < M; ++m) {
            const bool drive = (vin[m] != 0.f);
            sim.access_transistors[m] = std::vector<bool>(N, drive);
            if (drive) {
                Vappwl1(m) = voltage_pulse_height;
            }
        }

        Eigen::VectorXf Vguess = Eigen::VectorXf::Zero(2 * M * N);
        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < N; ++j) {
                Vguess(i * N + j) = Vappwl1(i);
            }
        }

        Iout = sim.ApplyVoltage(Vguess, Vappwl1, Vappwl2, Vappbl1, Vappbl2, dt, method);

        Iout_MAC.assign(N, 0.f);
        for (size_t n = 0; n < N; ++n) {
            float acc = 0.f;
            for (size_t m = 0; m < M; ++m) {
                acc += Iout[m][n];
            }
            Iout_MAC[n] = acc;
        }
    }

public:
    PyXbarSimulator2D(int M, int N, int bits_per_cell)
        : simulator_(M, N, bits_per_cell),
          Rswl1_(simulator_.Rswl1),
          Rswl2_(simulator_.Rswl2),
          Rsbl1_(simulator_.Rsbl1),
          Rsbl2_(simulator_.Rsbl2),
          Rwl_(simulator_.Rwl),
          Rbl_(simulator_.Rbl) {}

    void initialize_jart() {
        simulator_.Initialize<JART_VCM_v1b_var>();
        device_kind_ = DeviceKind::JART;
    }

    void initialize_fefet() {
        simulator_.Initialize<FeFET>();
        device_kind_ = DeviceKind::FeFET;
    }

    void set_weights(py::array_t<int, py::array::c_style | py::array::forcecast> weights) {
        auto buf = weights.request();
        if (buf.ndim != 2) {
            throw std::runtime_error("Weights must be a 2-D array with shape (M, N)");
        }
        if (buf.shape[0] != simulator_.M || buf.shape[1] != simulator_.N) {
            throw std::runtime_error("Weight shape must match (M, N)");
        }

        const int* ptr = static_cast<int*>(buf.ptr);
        weights_vec_.assign(simulator_.M, std::vector<int>(simulator_.N));
        for (int i = 0; i < simulator_.M; ++i) {
            for (int j = 0; j < simulator_.N; ++j) {
                weights_vec_[i][j] = ptr[i * simulator_.N + j];
            }
        }

        simulator_.SetRRAM(weights_vec_);
    }

    void set_parasitic_resistances(float Rswl1, float Rswl2, float Rsbl1, float Rsbl2, float Rwl, float Rbl) {
        Rswl1_ = Rswl1;
        Rswl2_ = Rswl2;
        Rsbl1_ = Rsbl1;
        Rsbl2_ = Rsbl2;
        Rwl_ = Rwl;
        Rbl_ = Rbl;

        simulator_.Rswl1 = Rswl1_;
        simulator_.Rswl2 = Rswl2_;
        simulator_.Rsbl1 = Rsbl1_;
        simulator_.Rsbl2 = Rsbl2_;
        simulator_.Rwl = Rwl_;
        simulator_.Rbl = Rbl_;

        simulator_.partial_G_ABCD = PartiallyPrecomputeG_ABCD(
            simulator_.M, simulator_.N,
            simulator_.Rswl1, simulator_.Rswl2,
            simulator_.Rsbl1, simulator_.Rsbl2,
            simulator_.Rwl, simulator_.Rbl
        );
    }

    std::tuple<py::array_t<float>, py::array_t<float>> run_inference_2d(
        py::array_t<float, py::array::c_style | py::array::forcecast> input,
        float dt = simulation_time_step,
        std::string method = "fixed-point"
    ) {
        if (device_kind_ == DeviceKind::Unknown) {
            throw std::runtime_error("Memristor type not initialized. Call initialize_jart() or initialize_fefet().");
        }
        if (weights_vec_.empty()) {
            throw std::runtime_error("Weights are not set. Call set_weights() first.");
        }

        auto buf = input.request();
        validate_input_2d_shape(buf, simulator_.M);

        const size_t samples = static_cast<size_t>(buf.shape[0]);
        const size_t M = static_cast<size_t>(simulator_.M);
        const size_t N = static_cast<size_t>(simulator_.N);
        const float* xptr = static_cast<float*>(buf.ptr);

        const std::vector<py::ssize_t> iout_shape = {
            static_cast<py::ssize_t>(samples),
            static_cast<py::ssize_t>(M),
            static_cast<py::ssize_t>(N)
        };
        const std::vector<py::ssize_t> mac_shape = {
            static_cast<py::ssize_t>(samples),
            static_cast<py::ssize_t>(N)
        };

        py::array_t<float> Iout_np(iout_shape);
        py::array_t<float> IoutMAC_np(mac_shape);
        auto iout_buf = Iout_np.request();
        auto mac_buf = IoutMAC_np.request();

        float* iout_ptr = static_cast<float*>(iout_buf.ptr);
        float* mac_ptr = static_cast<float*>(mac_buf.ptr);

        std::vector<std::vector<float>> Iout;
        std::vector<float> Iout_MAC;

        for (size_t s = 0; s < samples; ++s) {
            const float* vin = xptr + s * M;
            run_single(simulator_, vin, dt, method, Iout, Iout_MAC);

            for (size_t m = 0; m < M; ++m) {
                for (size_t n = 0; n < N; ++n) {
                    const size_t idx = s * (M * N) + m * N + n;
                    iout_ptr[idx] = Iout[m][n];
                }
            }
            for (size_t n = 0; n < N; ++n) {
                mac_ptr[s * N + n] = Iout_MAC[n];
            }
        }

        return std::make_tuple(Iout_np, IoutMAC_np);
    }

    std::tuple<py::array_t<float>, py::array_t<float>> run_inference_2d_threaded(
        py::array_t<float, py::array::c_style | py::array::forcecast> input,
        int num_workers,
        float dt = simulation_time_step,
        std::string method = "fixed-point"
    ) {
        if (device_kind_ == DeviceKind::Unknown) {
            throw std::runtime_error("Memristor type not initialized. Call initialize_jart() or initialize_fefet().");
        }
        if (weights_vec_.empty()) {
            throw std::runtime_error("Weights are not set. Call set_weights() first.");
        }

        auto buf = input.request();
        validate_input_2d_shape(buf, simulator_.M);

        const size_t samples = static_cast<size_t>(buf.shape[0]);
        const size_t M = static_cast<size_t>(simulator_.M);
        const size_t N = static_cast<size_t>(simulator_.N);
        const float* xptr = static_cast<float*>(buf.ptr);

        if (samples == 0) {
            const std::vector<py::ssize_t> empty_iout_shape = {
                static_cast<py::ssize_t>(0),
                static_cast<py::ssize_t>(M),
                static_cast<py::ssize_t>(N)
            };
            const std::vector<py::ssize_t> empty_mac_shape = {
                static_cast<py::ssize_t>(0),
                static_cast<py::ssize_t>(N)
            };
            py::array_t<float> empty_iout(empty_iout_shape);
            py::array_t<float> empty_mac(empty_mac_shape);
            return std::make_tuple(empty_iout, empty_mac);
        }

        size_t workers = (num_workers > 0)
            ? static_cast<size_t>(num_workers)
            : static_cast<size_t>(std::max(1u, std::thread::hardware_concurrency()));
        workers = std::min(workers, samples);

        const std::vector<py::ssize_t> iout_shape = {
            static_cast<py::ssize_t>(samples),
            static_cast<py::ssize_t>(M),
            static_cast<py::ssize_t>(N)
        };
        const std::vector<py::ssize_t> mac_shape = {
            static_cast<py::ssize_t>(samples),
            static_cast<py::ssize_t>(N)
        };

        py::array_t<float> Iout_np(iout_shape);
        py::array_t<float> IoutMAC_np(mac_shape);
        auto iout_buf = Iout_np.request();
        auto mac_buf = IoutMAC_np.request();

        float* iout_ptr = static_cast<float*>(iout_buf.ptr);
        float* mac_ptr = static_cast<float*>(mac_buf.ptr);

        const size_t chunk = (samples + workers - 1) / workers;
        std::vector<std::future<void>> futures;
        futures.reserve(workers);

        for (size_t w = 0; w < workers; ++w) {
            const size_t start = w * chunk;
            if (start >= samples) {
                break;
            }
            const size_t end = std::min(samples, start + chunk);

            futures.emplace_back(std::async(std::launch::async, [&, start, end]() {
                // Parallel across samples is safe if each worker owns its own simulator state.
                CrossbarSimulator local_sim(simulator_.M, simulator_.N, simulator_.bits_per_cell);
                configure_simulator(local_sim);

                std::vector<std::vector<float>> Iout;
                std::vector<float> Iout_MAC;

                for (size_t s = start; s < end; ++s) {
                    const float* vin = xptr + s * M;
                    run_single(local_sim, vin, dt, method, Iout, Iout_MAC);

                    for (size_t m = 0; m < M; ++m) {
                        for (size_t n = 0; n < N; ++n) {
                            const size_t idx = s * (M * N) + m * N + n;
                            iout_ptr[idx] = Iout[m][n];
                        }
                    }
                    for (size_t n = 0; n < N; ++n) {
                        mac_ptr[s * N + n] = Iout_MAC[n];
                    }
                }
            }));
        }

        for (auto& f : futures) {
            f.get();
        }

        return std::make_tuple(Iout_np, IoutMAC_np);
    }
};

PYBIND11_MODULE(xbar_simulator_2d, m) {
    m.doc() = "2-D input Python bindings for the memristor crossbar simulator";

    m.def("set_eigen_threads", &set_eigen_threads_2d,
          "Set the number of threads for Eigen computations",
          py::arg("num_threads"));
    m.def("get_eigen_threads", &get_eigen_threads_2d,
          "Get the current number of threads for Eigen computations");

    py::class_<PyXbarSimulator2D>(m, "CrossbarSimulator2D")
        .def(py::init<int, int, int>(), py::arg("M"), py::arg("N"), py::arg("bits_per_cell") = 1)
        .def("set_weights", &PyXbarSimulator2D::set_weights,
             "Set crossbar weights as a 2-D matrix with shape (M, N)")
        .def("initialize_jart", &PyXbarSimulator2D::initialize_jart,
             "Initialize crossbar with JART memristors")
        .def("initialize_fefet", &PyXbarSimulator2D::initialize_fefet,
             "Initialize crossbar with FeFET memristors")
        .def("set_parasitic_resistances", &PyXbarSimulator2D::set_parasitic_resistances,
             py::arg("Rswl1"), py::arg("Rswl2"), py::arg("Rsbl1"), py::arg("Rsbl2"),
             py::arg("Rwl"), py::arg("Rbl"),
             "Set parasitic resistances")
        .def("run_inference_2d", &PyXbarSimulator2D::run_inference_2d,
             py::arg("input"), py::arg("dt") = simulation_time_step, py::arg("method") = "fixed-point",
             "Run sequential inference for a 2-D input matrix of shape (samples, M)")
        .def("run_inference_2d_threaded", &PyXbarSimulator2D::run_inference_2d_threaded,
             py::arg("input"), py::arg("num_workers") = 0,
             py::arg("dt") = simulation_time_step, py::arg("method") = "fixed-point",
             "Run threaded inference across samples using one simulator state per worker");

    m.attr("voltage_pulse_height") = voltage_pulse_height;
    m.attr("simulation_time_step") = simulation_time_step;
    m.attr("simulation_num_threads") = simulation_num_threads;
    m.attr("methods") = py::make_tuple("fixed-point", "NewtonRaphson", "Broyden", "BroydenInv");
}
