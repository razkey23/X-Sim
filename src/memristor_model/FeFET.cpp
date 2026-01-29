#include "memristor_model/FeFET.h"

/***********************  ctor & initialisation  ************************/ 
FeFET::FeFET(std::size_t ndom, unsigned seed)
: ND_(ndom),
  r_Ea_(ND_), r_voff_(ND_), vswitchlimit_(ND_),
  St_(ND_), h_(ND_), hpre_(ND_), taus_(ND_),
  rng_(seed),
  norm_Ea_(A_SHAPE, B_SHAPE),
  norm_voff_(0.0, std::abs(VFB)),
  uni01_(0.0, 1.0)
{
    // Pre‑compute oxide constants (identical to Verilog‑A initial_step)
    Cox_   = EPIO * eps0 / TIL;
    gamma_ = std::sqrt(2.0 * q_e * NA * EPIS * eps0) / Cox_;

    // domain‑wise random parameters
    for (std::size_t i = 0; i < ND_; ++i)
    {
        r_Ea_[i]       = norm_Ea_(rng_);
        r_voff_[i]     = norm_voff_(rng_) * (VFB <= 0.0 ? +1.0 : -1.0);
        vswitchlimit_[i] = r_Ea_[i] / std::pow(std::log(TIME_LIM / TAU0), 1.0/ALPHA_EXP);
        St_[i]         = (rng_() & 1) ? +1.0 : -1.0;
        h_[i] = hpre_[i] = 0.0;
    }
}

double FeFET::ApplyVoltage(double V, double dt)
{
    /* --------------- 1. advance time ---------------- */
    const double t_now = t_prev_ + dt;

    /* distinguish between WRITE (dt>0) and READ (dt==0) */
    const bool is_write = (dt > 0.0);

    /* --------------- 2. gate bias ------------------- */
    const double VGB = is_write ? V        /* writing */
                                : 1.0;    /* reading : fixed word‑line */
    const double VDS = is_write ? 0.0      /* bit‑lines floated */
                                : V;       /* small read bias */
    const double VBS = 0.0;
    const double VBD = -VDS;               /* body at source potential */

    /* --------------- 3. solve internal FE node ------ */
    vfe_ = solve_vfe(VGB, VBS, VBD);

    /* --------------- 4. update domain kinetics ------ */
    double sumPol = 0.0;
    for (std::size_t i=0;i<ND_;++i)
    {
        double vswitch = (vfe_ + vpre_)*0.5 - r_voff_[i];
        taus_[i] = TAU0 * std::exp(std::pow(r_Ea_[i] / std::max(std::abs(vswitch), vswitchlimit_[i]), ALPHA_EXP));
        h_[i] = hpre_[i] + (t_now - t_prev_) * ((vswitch*St_[i]<=0)? +1.0 : -1.0) / taus_[i];

        double pswi = (hpre_[i] > h_[i]) ? -0.1
                      : 1.0 - std::exp(std::pow(hpre_[i], BETA_SW) - std::pow(h_[i], BETA_SW));
        if (h_[i] < 0.0 || pswi > uni01_(rng_)) h_[i] = 0.0;
        if (pswi > uni01_(rng_))               St_[i] = -St_[i];

        hpre_[i] = h_[i];
        sumPol   += St_[i];
    }

    /* --------------- 5. MOS electrostatics ---------- */
    const double phis = phi(VGB - vfe_, -VFB, -VBS);
    const double phid = phi(VGB - vfe_, -VFB, -VBD);
    const double Ids  = Id(phis, phid, VGB - vfe_);

    /* --------------- 6. book‑keeping --------------- */
    vpre_   = vfe_;
    t_prev_ = t_now;

    return Ids;   // during write Ids ≈ 0 A; during read real current
}

double FeFET::GetResistance(double V_applied)
{
    const double Id = ApplyVoltage(V_applied, /*dt=*/0.0);
    if (std::abs(Id) < 1e-15) return 1e15;
    return V_applied / Id;
}

void FeFET::SetWeight(int high)
{
    for (auto &s : St_) s = high ? +1.0 : -1.0;
    std::fill(h_.begin(),     h_.end(),     0.0);
    std::fill(hpre_.begin(),  hpre_.end(),  0.0);
    vfe_ = vpre_ = high ? -0.5 : +0.5;   // crude estimate of stable point
}

inline double FeFET::thermalVoltage() const
{
    return kB * 300.0 / q_e;   // assume 300 K, can be parameterised
}


double FeFET::Id(double phis, double phid, double VGB) const
{
    const double Vds = phid - phis;
    const double Vgt = VGB - gamma_*std::sqrt(std::max(phis, 0.0));
    if (Vgt <= 0.0) return 0.0;
    return MIU*Cox_*WIDTH/LENGTH * (Vgt*Vds - 0.5*Vds*Vds);
}

double FeFET::phi(double VGB, double VFB, double VCB) const
{
    // TODO: copy the iterative surface‑potential solver from the Verilog‑A
    // phi() function.  This stub returns a simple approximation suitable for
    // small‑signal reads.
    const double Vt = thermalVoltage();
    return std::min(VGB - VFB, 0.8*Vt);
}

double FeFET::Qmos(double phis, double phid, double VGB) const
{
    // TODO: verbatim equation port; for now, linear approx
    return Cox_ * (VGB - 0.5*(phis+phid));
}

double FeFET::solve_vfe(double VGB, double VBS, double VBD)
{
    double left = vpre_ - 10.0;
    double right= vpre_ + 10.0;
    double mid;

    auto Pnet = [&](double vtest){
        double sum=0.0; for(double s:St_) sum+=s;
        double phis = phi(VGB - vtest, -VFB, -VBS);
        double phid = phi(VGB - vtest, -VFB, -VBD);
        return PR*sum/ND_ + 1e6*vtest*EPIFE*eps0/TFE - 1e6*Qmos(phis, phid, VGB - vtest);
    };

    for(int i=0;i<50;++i){
        mid = 0.5*(left+right);
        ((Pnet(mid)*Pnet(left) <= 0) ? right : left) = mid;
    }
    return 0.5*(left+right);
}

