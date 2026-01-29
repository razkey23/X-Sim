#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <Eigen/Core>
#include "core/crossbar_simulator.h"
#include "core/simulation_settings.h"

namespace py = pybind11;

constexpr float VP_H = 0.1f;       // Voltage pulse height
constexpr float VP_W = 50e-6f;     // Pulse width
constexpr float VP_R =  5e-6f;     // Rise time
constexpr float VP_F =  5e-6f;     // Fall time

// Helper function to convert numpy array to vector of vectors
template <typename T>
std::vector<std::vector<T>> numpy_to_vector2d(py::array_t<T> array) {
    auto buf = array.request();
    if (buf.ndim != 2) {
        throw std::runtime_error("Input must be a 2D array");
    }
    
    size_t rows = buf.shape[0];
    size_t cols = buf.shape[1];
    T* ptr = static_cast<T*>(buf.ptr);
    
    std::vector<std::vector<T>> result(rows, std::vector<T>(cols));
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            result[i][j] = ptr[i * cols + j];
        }
    }
    
    return result;
}

// Helper function to convert vector of vectors to numpy array
template <typename T>
py::array_t<T> vector2d_to_numpy(const std::vector<std::vector<T>>& vec) {
    if (vec.empty()) {
        std::vector<size_t> shape = {0, 0};
        return py::array_t<T>(shape);
    }
    
    size_t rows = vec.size();
    size_t cols = vec[0].size();
    
    std::vector<size_t> shape = {rows, cols};
    py::array_t<T> result(shape);
    auto buf = result.request();
    T* ptr = static_cast<T*>(buf.ptr);
    
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            ptr[i * cols + j] = vec[i][j];
        }
    }
    
    return result;
}


// Helper function to convert numpy array to vector
template <typename T>
std::vector<T> numpy_to_vector1d(py::array_t<T> array) {
    auto buf = array.request();
    if (buf.ndim != 1) {
        throw std::runtime_error("Input must be a 1D array");
    }
    T* ptr = static_cast<T*>(buf.ptr);
    return std::vector<T>(ptr, ptr + buf.shape[0]);
}

// Helper function to convert vector to numpy array
template <typename T>
py::array_t<T> vector1d_to_numpy(const std::vector<T>& vec) {
    std::vector<size_t> shape = {vec.size()};
    py::array_t<T> result(shape);
    auto buf = result.request();
    T* ptr = static_cast<T*>(buf.ptr);
    std::copy(vec.begin(), vec.end(), ptr);
    return result;
}


// Helper function to convert numpy array to Eigen vector
Eigen::VectorXf numpy_to_eigen_vector(py::array_t<float> array) {
    auto buf = array.request();
    if (buf.ndim != 1) {
        throw std::runtime_error("Input must be a 1D array");
    }
    
    size_t size = buf.shape[0];
    float* ptr = static_cast<float*>(buf.ptr);
    
    Eigen::VectorXf result(size);
    for (size_t i = 0; i < size; i++) {
        result(i) = ptr[i];
    }
    
    return result;
}

// Function to set number of threads for Eigen
void set_eigen_threads(int num_threads) {
    Eigen::setNbThreads(num_threads);
}

// Function to get current number of threads for Eigen
int get_eigen_threads() {
    return Eigen::nbThreads();
}

class PyXbarSimulator {
private:
    CrossbarSimulator simulator;
    std::vector<std::vector<int>> weights_vec_;

public:
    PyXbarSimulator(int M, int N, int bits_per_cell) : simulator(M, N, bits_per_cell) {}
    
    void set_weights(py::array_t<int> weights) {
        weights_vec_ = numpy_to_vector2d<int>(weights);
        simulator.SetRRAM(weights_vec_);
    }

    void initialize_jart() {
        simulator.Initialize<JART_VCM_v1b_var>();
    }

    void initialize_fefet() {
        simulator.Initialize<FeFET>();
    }

   std::tuple<py::array_t<float>, py::array_t<float>>
    run_inference(py::array_t<float> input,
                float dt           = simulation_time_step,
                std::string method = "fixed-point")
    {
        if (input.ndim() != 1)
            throw std::runtime_error("Input must be a 1-D array");

        /* ----------  prepare voltages & access transistors ---------- */
        auto  buf = input.request();
        auto* vin = static_cast<float*>(buf.ptr);

        const size_t M = simulator.M;
        const size_t N = simulator.N;

        Eigen::VectorXf Vappwl1 = Eigen::VectorXf::Zero(M);
        Eigen::VectorXf Vappwl2 = Eigen::VectorXf::Zero(M);
        Eigen::VectorXf Vappbl1 = Eigen::VectorXf::Zero(N);
        Eigen::VectorXf Vappbl2 = Eigen::VectorXf::Zero(N);

        for (size_t m = 0; m < M; ++m) {
            bool drive = (m < buf.shape[0] && vin[m] != 0.f);
            simulator.access_transistors[m] = std::vector<bool>(N, drive);
            if (drive) Vappwl1(m) = voltage_pulse_height;
        }

        /* ----------  initial guess & non-linear solve  -------------- */
        Eigen::VectorXf Vguess = Eigen::VectorXf::Zero(2 * M * N);
        for (size_t i = 0; i < M; ++i)
            for (size_t j = 0; j < N; ++j)
                Vguess(i * N + j) = Vappwl1(i);

        auto Iout_vec2d = simulator.ApplyVoltage(Vguess,
                                                Vappwl1, Vappwl2,
                                                Vappbl1, Vappbl2,
                                                dt, method);   // (M×N)

        /* ----------  column MAC currents  --------------------------- */
        std::vector<float> Iout_MAC(N, 0.f);
        for (size_t n = 0; n < N; ++n)
            for (size_t m = 0; m < M; ++m)
                Iout_MAC[n] += Iout_vec2d[m][n];

        /* ----------  convert to NumPy & return  --------------------- */
        auto Iout_np     = vector2d_to_numpy<float>(Iout_vec2d);
        auto IoutMAC_np  = vector1d_to_numpy<float>(Iout_MAC);

        return std::make_tuple(Iout_np, IoutMAC_np);
    }
    
   py::array_t<float>
    run_multiple_inferences(py::array_t<float> input,
                            int  num_inferences,
                            float dt           = simulation_time_step,
                            std::string method = "fixed-point")
    {
        if (input.ndim() != 1)
            throw std::runtime_error("Input must be a 1-D array");

        const size_t N = simulator.N;
        std::vector<std::vector<float>> all_mac;   // one row per inference

        for (int k = 0; k < num_inferences; ++k)
        {
            /* run_inference now returns (Icell, Imac) */
            auto result   = this->run_inference(input, dt, method);
            auto mac_np   = std::get<1>(result);          // 1-D (N,)
            auto mac_buf  = mac_np.request();
            auto* mac_ptr = static_cast<float*>(mac_buf.ptr);

            all_mac.emplace_back(mac_ptr, mac_ptr + N);   // save this pass
        }

        return vector2d_to_numpy<float>(all_mac);         // shape (samples, N)
    }
    
    // Expose additional parameters and methods
    void set_parasitic_resistances(float Rswl1, float Rswl2, float Rsbl1, float Rsbl2, float Rwl, float Rbl) {
        simulator.Rswl1 = Rswl1;
        simulator.Rswl2 = Rswl2;
        simulator.Rsbl1 = Rsbl1;
        simulator.Rsbl2 = Rsbl2;
        simulator.Rwl = Rwl;
        simulator.Rbl = Rbl;
        
        // Recompute partial_G_ABCD
        simulator.partial_G_ABCD = PartiallyPrecomputeG_ABCD(
            simulator.M, simulator.N, 
            simulator.Rswl1, simulator.Rswl2, 
            simulator.Rsbl1, simulator.Rsbl2, 
            simulator.Rwl, simulator.Rbl
        );
    }

    std::tuple<py::array_t<float>, py::array_t<float>> transientInference(
        py::array_t<bool> Vwl1, py::array_t<bool> Vwl2,
        py::array_t<bool> Vbl1, py::array_t<bool> Vbl2,
        py::array_t<int> weights,
        py::array_t<float> waveform,
        float dt = simulation_time_step
    ) {
        // Convert numpy arrays to vectors
        auto Vwl1_vec = numpy_to_vector1d<bool>(Vwl1);
        auto Vwl2_vec = numpy_to_vector1d<bool>(Vwl2);
        auto Vbl1_vec = numpy_to_vector1d<bool>(Vbl1);
        auto Vbl2_vec = numpy_to_vector1d<bool>(Vbl2);
        auto weights_vec = numpy_to_vector2d<int>(weights);
        
        // Convert waveform to vector of arrays
        auto waveform_buf = waveform.request();
        if (waveform_buf.ndim != 2 || waveform_buf.shape[1] != 2) {
            throw std::runtime_error("Waveform must be a 2D array with shape (n, 2)");
        }
        std::vector<std::array<float, 2>> waveform_vec;
        float* ptr = static_cast<float*>(waveform_buf.ptr);
        for (size_t i = 0; i < waveform_buf.shape[0]; i++) {
            waveform_vec.push_back({ptr[i*2], ptr[i*2 + 1]});
        }

        // Prepare output containers
        std::vector<std::vector<float>> Iout;
        std::vector<float> Iout_MAC;

        // Run simulation
        simulator.Simulate(Vwl1_vec, Vwl2_vec, Vbl1_vec, Vbl2_vec, weights_vec, waveform_vec, dt, Iout, Iout_MAC);

        // Convert outputs to numpy arrays
        return std::make_tuple(vector2d_to_numpy<float>(Iout), vector1d_to_numpy<float>(Iout_MAC));
    }
    // In your pybind11 binding class…

    static const std::vector<std::array<float,2>> default_waveform;

    std::tuple<py::array_t<float>,py::array_t<float>>
    transientInference_mod(
        py::array_t<float> input,
        float dt           = simulation_time_step,
        const std::string &method = "fixed-point"
    ) {
        // 1) Validate input
        if (input.ndim() != 1)
            throw std::runtime_error("Input must be a 1-D array");

        // 2) Pull out the raw floats
        auto buf = input.request();
        auto* vin = static_cast<float*>(buf.ptr);

        // 3) Build control‐line booleans
        const size_t M = simulator.M;
        const size_t N = simulator.N;
        std::vector<bool> Vwl1(M), Vwl2(M,false), Vbl1(N,false), Vbl2(N,false);

        for (size_t i = 0; i < M; ++i) {
            bool drive = (i < buf.shape[0] && vin[i] != 0.f);
            Vwl1[i] = drive;
            // also set the access‐transistor mask inside the simulator
            simulator.access_transistors[i].assign(N, drive);
        }

        // 4) Grab the weights you previously loaded via set_weights()
        // const auto &weights_vec = weights_vec_;  // vector<vector<bool>>

        // 5) Use your stored default_waveform
        // const auto &waveform_vec = default_waveform;  // vector<array<float,2>>

        // 6) Prepare outputs
        std::vector<std::vector<float>> Iout;
        std::vector<float>               IoutMAC;

        // 7) Run the transient sim
        simulator.Simulate(
            Vwl1, Vwl2, Vbl1, Vbl2,
            weights_vec_,
            default_waveform,
            dt,
            Iout,
            IoutMAC
        );


        // 8) Convert back to NumPy and return
        return std::make_tuple(
        vector2d_to_numpy<float>(Iout),
        vector1d_to_numpy<float>(IoutMAC)
        );
}

};

const std::vector<std::array<float,2>> PyXbarSimulator::default_waveform = {
    {0.0f,      0.0f},
    {VP_H,      VP_R},
    {VP_H,      VP_W - VP_F},
    {0.0f,      VP_W}
};


PYBIND11_MODULE(xbar_simulator, m) {
    m.doc() = "Python bindings for the memristor crossbar simulator";
    
    // Add thread control functions
    m.def("set_eigen_threads", &set_eigen_threads, 
          "Set the number of threads for Eigen computations",
          py::arg("num_threads"));
    m.def("get_eigen_threads", &get_eigen_threads,
          "Get the current number of threads for Eigen computations");
    
    py::class_<PyXbarSimulator>(m, "CrossbarSimulator")
        .def(py::init<int, int, int>(), py::arg("M"), py::arg("N"), py::arg("bits_per_cell")=1)
        .def("set_weights", &PyXbarSimulator::set_weights, 
             "Set the weights of the crossbar (boolean matrix)")
        .def("initialize_jart", &PyXbarSimulator::initialize_jart,
             "Initialize the crossbar with JART memristors")
        .def("initialize_fefet", &PyXbarSimulator::initialize_fefet,
             "Initialize the crossbar with FeFET memristors")
        
        .def("run_inference", &PyXbarSimulator::run_inference, 
             py::arg("input"), py::arg("dt") = simulation_time_step, py::arg("method") = "fixed-point",
             "Run a single inference with the given input vector")
        .def("run_multiple_inferences", &PyXbarSimulator::run_multiple_inferences,
             py::arg("input"), py::arg("num_inferences"), py::arg("dt") = simulation_time_step,  py::arg("method") = "fixed-point",
             "Run multiple inferences with the same input vector")
        .def("set_parasitic_resistances", &PyXbarSimulator::set_parasitic_resistances,
             py::arg("Rswl1"), py::arg("Rswl2"), py::arg("Rsbl1"), py::arg("Rsbl2"), 
             py::arg("Rwl"), py::arg("Rbl"),
             "Set the parasitic resistances of the crossbar")
        .def("transientInference", &PyXbarSimulator::transientInference,
             py::arg("Vwl1"), py::arg("Vwl2"), py::arg("Vbl1"), py::arg("Vbl2"),
             py::arg("weights"), py::arg("waveform"), py::arg("dt") = simulation_time_step,
             "Run a transient simulation with the given voltage waveforms")
        .def("transientInference_mod", &PyXbarSimulator::transientInference_mod,
             py::arg("input"),
             py::arg("dt") = simulation_time_step,
             py::arg("method") = "fixed-point");
    // Expose simulation settings
    m.attr("voltage_pulse_height") = voltage_pulse_height;
    m.attr("simulation_time_step") = simulation_time_step;
    m.attr("simulation_num_threads") = simulation_num_threads;
    m.attr("methods") = py::make_tuple("fixed-point", "NewtonRaphson", "Broyden", "BroydenInv");
} 