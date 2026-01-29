#ifndef JART_VCM_v1b_var_H_
#define JART_VCM_v1b_var_H_
#include "Memristor.h"
#include <array>
#include <vector>
#include <iostream>

#ifndef M_PI
#define M_PI 3.1415927  // Define M_PI if not defined
#endif
const double P_Q = 1.6022e-19;    // Elementary charge [C]
const double P_K = 1.38065e-23;   // Boltzman constant [J/K]
const double P_EPS0 = 8.6549e-12; // Permittivity of a vacuum [F/m]
const double P_H = 6.626e-34;     // Planck constant [Js]

class JART_VCM_v1b_var : public Memristor {
    private:
    public:
        // ----- Pyisical constants do not change! -----
        const double Arichardson = 6.01e5; // Richardson's constant [A/m^2K^2]
        const double mdiel = 9.10938e-31;  // electron rest mass [kg]
        const double zvo = 2;              // oxygen vacancy charge number
        const double T0 = 293;             // ambient temperature [K]

        // ----- Fitting parameters -----
        double eps = 17;              // from [10:25], static hafnium oxide permittivity
        double epsphib = 5.5;         // hafnium oxide permittivity related to image force barrier lowering
        double phibn0 = 0.18;         // from [0.1:0.5], nominal schottky barrier height [eV]
        double phin = 0.1;            // from [0.1:0.3], energy level difference between the Fermi level in the oxide and the oxide conduction band edge [eV]
        double un = 4e-6;             // from [1e-6:1e-5], electron mobility [m^2/Vs]
        double Ndiscmax = 20;         // from [0.001:1100], maximum oxygen vacancy concentration in the disc [10^26/m^3]
        double Ndiscmin = 0.0001;      // from [0.0001:100], minimum oxygen vacancy concentration in the disc [10^26/m^3]
        double Ninit = 0.008;         // from [0.0001:1000], initial oxygen vacancy concentration in the disc [10^26/m^3]
        double Nplug = 20;            // from [0.001:100], oxygen vacancy concentration in the plug [10^26/m^3]
        double a = 0.25e-9;           // from [0.1e-9:1e-9], ion hopping distance [m]
        double nyo = 2e13;            // from [1e10:1e14], attempt frequency [Hz]
        double dWa = 1.35;            // from [0.8:1.5], activation energy [eV]
        // double Rth0 = 15.72e6;        // from [1e6:20e6], thermal resistance of the Hafnium Oxide [K/W]
        double Rth0 = 1e7;
        double rdet = 45e-9;          // from [5e-9:100e-9], radius of the filament [m]
        double rnew = 45e-9;          // from [5e-9:100e-9], radius of the filament [m]
        double lcell = 3;             // from [2:5], length of disc and plug region [nm]
        double ldet = 0.4;            // from [0.1:5], length of the disc region [nm]
        double lnew = 0.4;            // from [0.1:5], length of the disc region [nm]
        double Rtheff_scaling = 0.27; // from [0.1:1], scaling factor for RESET
        double RTiOx = 650;           // from [0:5000], series resistance of the TiOx layer [Ohm]
        double R0 = 719.2437;         // Resistance at T0 [Ohm]
        double Rthline = 90471.47;    // thermal conductivity of the Platinum and Titanium [W/mK]
        double alphaline = 3.92e-3;   // temperature coefficient [1/K]

        double eps_eff;     // static hafnium oxide permittivity
        double epsphib_eff; // hafnium oxide permittivity related to image force barrier lowering

        // ----- State variable -----

        // ----- Other internal variables -----
        int trig;         // Used to signify certain voltage crossings and limit the state variable
        double Ninitreal; // Not sure what this does, it is set to Ninit on startup and never changed again
        double rvar;      // radius of the fillament used in calculations, updated with the variability model
        double rold;      // radius of the fillament, only used in the variability model to update rvar
        double lvar;      // length of the disc region used in calculations, updated with the variability model
        double lold;      // length of the disc region, only used in the variability model to update lvar
        double Nold;      // oxygen vacancy of the disc region, only used in the variability model
        double Treal;     // homogeneous filament temperature [K]
        double A;         // cross section of the filament area
        double Rdisc;     // resistance of the disc region
        double Rplug;     // resistance of the plug region
        double Rseries;   // resistance of the series section
        double Rtheff;    // thermal resistance
        double V_prev;    // previous applied voltage, used to check for voltage crossings
        double V_schottky_prev;

        double Nreal; // oxygen vacancy concentration of the disc region [nm]

        double tmp;

        // ################ Ninit variables for MLC ##################
        int bits_per_cell = 1;
        double Ninit_LRS3    =  7.1327; 
        double Ninit_LRS2    =  0.5430;
        double Ninit_LRS1    =  0.1443;
        // double Ninit_LRS2    =  0.5991;
        // double Ninit_LRS1    =  0.1592;
        double Ninit_HRS     =  0.0002;

        void UpdateFilamentArea();
        void UpdateTemperature(double V_schottky, double V_discplugserial, double I_schottky);
        double ComputeSchottkyCurrent(double V_schottky);
        void UpdateResistance(double I_discplugserial);
        void UpdateConcentration(double I_ion, double dt);
        double ComputeIonCurrent(double V_applied, double V_schottky, double V_discplugserial);
        std::array<double, 3> SolveBisection(double V_low, double V_high, double V_applied);
        void MultiSolveBisection(double V_low, double V_high, double V_applied, std::vector<std::array<double, 3>> &roots);
        std::array<double, 3> SolveFixedpoint(double Vguess, double V_applied);
        std::array<double, 3> SolveBrent(double V_a, double V_b, double V_applied);

    // public:
        JART_VCM_v1b_var(int bits_per_cell_ = 1) {

            bits_per_cell = bits_per_cell_;

            Nreal = Ninit;
            Ninitreal = Ninit;
            trig = 1;
            rvar = rnew;
            rold = rnew;
            lvar = lnew;
            lold = lnew;
            Nold = Ninit;
            eps_eff = eps * P_EPS0;
            epsphib_eff = epsphib * P_EPS0;
            V_prev = 0;
            V_schottky_prev = 0;
            UpdateFilamentArea();
            UpdateResistance(0);
            UpdateTemperature(0, 0, 0);

            tmp = 0;
        }


        double ApplyVoltage(double V_applied, double dt) override;
        double GetResistance(double V_applied) override;
        void SetWeight(int weight) override;
};

#endif  // JART_VCM_v1b_var_H_
