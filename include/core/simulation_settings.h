#ifndef SIMULATION_SETTINGS_H_
#define SIMULATION_SETTINGS_H_

// JART VCM v1b var
const int memristor_solving_method = 1;  // Options: 0 for bisection, 1 for fixed-point

const double memristor_bisection_a = 0.5;            // 
const double memristor_bisection_criterion = 1e-6;   // Exit criterion of the bisection solver
const double memristor_bisection_it_max = 1e3;       // iteration limit of the bisection solver
const double memristor_bisection_min_domain = 1e-9;  // Minimum domain of the bisection solver. If the domain is smaller than this value, it is assumed there is no solution in the given domain

const double memristor_fixedpoint_criterion = 1e-6;  // Dampening factor of the fixed-point solver
const double memristor_fixedpoint_it_max = 1e3;      // Exit criterion of the fixed-point solver

const double memristor_dynamic_time_step_N_limit = 1e-2;   // Changes in N above this limit trigers the dynamic time step mechanism
const double memristor_dynamic_time_step_t_limit = 1e-12;   // Delta times below this value will prevent a time step from being triggered
const int memristor_dynamic_time_step_time_division = 2;  // The amount of subdivisions of the original time step that trigered the dynamic time step mechanism

const double memristor_get_resistance_voltage_threshold = 1e-6;  // The maximum amount of voltage for which the output current is assumed to be zero for the purpose of calculating resistance

// Non-linear solver
const float non_linear_fixed_point_a = 1.;               // Dampening factor of the fixed point solver
const float non_linear_fixed_point_it_max = 500;          // Iteration limit of the fixed point solver
const float non_linear_fixed_point_voltage_nudge = 1e-9;  // Amount the voltage guess will be nudged by if a NAN is returned from the linear solver
const float non_linear_fixed_point_criterion = 1e-6;      // Exit criterion of the fixed point solver

// Voltage pulse (as used in RRAM_validation.cpp)
const float voltage_pulse_width = 50e-6;     // Total voltage pulse width, including rise and fall time
const float voltage_pulse_height = 0.1;      // Maximum height of the voltage pulse
const float voltage_pulse_rise_time = 5e-6;  // Time for the voltage pulse to go from 0 to the pulse height
const float voltage_pulse_fall_time = 5e-6;  // Time for the voltage pulse to go from the pulse height to 0
const int simulation_num_threads = 4;        // Number of threads to use for the simulation
const float simulation_time_step = 1e-6;     // Delta time for each simulation step. Note: the memristor model can dynamically reduce this time step temporarilly if it detects large changes (see memristor_dynamic_time_step_N_limit)

#endif  // SIMULATION_SETTINGS_H_
