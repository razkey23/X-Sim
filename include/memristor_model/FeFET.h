#ifndef FEFET_H_
#define FEFET_H_


#include "Memristor.h"
#include <vector>
#include <random>
#include <cstddef>


/**
 * FeFET wrapper that mimics the API of a two‑terminal memristor while
 * internally keeping the 3‑terminal ferroelectric‑FET physics.
 *
 * The implementation is a line‑by‑line, C++ translation of the published
 * Verilog‑A model (see FeFET.cpp).  Only the public methods below are visible
 * to the rest of the cross‑bar simulator.
 */
class FeFET : public Memristor {
public:
    /**
         * @param ndom  Number of ferroelectric domains used in the stochastic model
         * @param seed  RNG seed so every cell instance can be decorrelated
    */
    explicit FeFET(std::size_t ndom = 20, unsigned seed = 1);

    /**
     * Apply a voltage for a duration dt.
     *   •  dt > 0  → treat V_applied as a **gate** write pulse (bit‑lines
     *                assumed grounded).
     *   •  dt == 0 → treat V_applied as the **drain‑source** read bias and
     *                return the corresponding channel current.
     * @return  Drain current I_D (A).
     */
    double ApplyVoltage(double V_applied, double dt) override;

    /**
     * Small‑signal resistance seen between drain and source during a read
     * operation at bias V_applied (absolute value typically ≤0.1 V).
     */
    double GetResistance(double V_applied) override;

    /**
     * Hard‑set the stored polarisation to logic ‘1’ (true) or ‘0’ (false).
     * Used by the simulator to preload weights instantly.
     */
    void SetWeight(int weight) override;

private:
     /* ===================== physical constants ===================== */
    static constexpr double kB   = 1.380649e-23;     // Boltzmann [J/K]
    static constexpr double q_e  = 1.602176634e-19;  // Elementary charge [C]
    static constexpr double eps0 = 8.854187817e-12;  // Vacuum permittivity [F/m]

    /* ======================= model parameters ===================== */
    // Geometry
    static constexpr double WIDTH  = 1e-6;   // [m]
    static constexpr double LENGTH = 1e-6;   // [m]
    static constexpr double TFE    = 0.8e-6; // ferroelectric thickness [m]
    static constexpr double TIL    = 0.1e-6; // interlayer oxide thickness [m]

    // Material / fitting
    static constexpr double VFB   = 0.0;            // flat‑band [V]
    static constexpr double NA    = 3e23;           // doping [1/m³] (≈3e17 cm⁻³)
    static constexpr double EPIS  = 11.8;           // Si relative permittivity
    static constexpr double EPIO  = 3.9;            // Interlayer oxide ε_r
    static constexpr double EPIFE = 28.0;           // FE ε_r
    static constexpr double MIU   = 50e-4;          // channel µ [m²/(V·s)]

    // Domain kinetics
    static constexpr double A_SHAPE   = 2.3;
    static constexpr double B_SHAPE   = 0.4;
    static constexpr double PR        = 25.0;       // remanent pol. [µC/cm²]
    static constexpr double TAU0      = 1.9e-8;     // prefactor [s]
    static constexpr double ALPHA_EXP = 3.0;
    static constexpr double BETA_SW   = 2.0;
    static constexpr double TIME_LIM  = 1e9;        // [s]

    /* ====================== run‑time state ======================== */
    const std::size_t ND_;
    std::vector<double> r_Ea_, r_voff_, vswitchlimit_;
    std::vector<double> St_, h_, hpre_, taus_;

    // global FE node
    double vfe_   = 0.0;   // internal FE voltage (previous solution)
    double vpre_  = 0.0;   // stored copy used by the Verilog‑A algorithm
    double t_prev_= 0.0;   // last time the domains were updated [s]

    // caches that depend only on geometry / constants
    double Cox_   = 0.0;   // gate‑oxide capacitance per unit area [F/m²]
    double gamma_ = 0.0;   // body factor parameter

    /* ====================== random number gen ===================== */
    std::mt19937_64 rng_;
    std::normal_distribution<double> norm_Ea_;
    std::normal_distribution<double> norm_voff_;
    std::uniform_real_distribution<double> uni01_;

     /* ====================== helper functions ====================== */
    double phi (double VGB, double VFB, double VCB) const;                 // surface potential solver
    double Qmos(double phis, double phid, double VGB) const;               // inversion charge density
    double Id  (double phis, double phid, double VGB) const;               // long‑channel ID
    double solve_vfe(double VGB, double VBS, double VBD);                  // bisection root solver

    double thermalVoltage() const;                                         // kT/q (V)
};

#endif  // FEFET_H_ 