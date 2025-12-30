// PSBP_normal/cpp/psbp_core.hpp
#ifndef PSBP_CORE_HPP
#define PSBP_CORE_HPP

#ifndef INV_SQRT2
#define INV_SQRT2 0.7071067811865475244008443621048490392848359376887
#endif

#include <vector>
#include <algorithm>
#include <cmath>
#include <numbers>

namespace psbp {

// ============================================================================
// ESTRUCTURAS BÁSICAS
// ============================================================================

struct EtaResult {
    std::vector<std::vector<double>> eta;  // (n_batch, H)
    int n_batch;
    int H;
};

struct WeightsResult {
    std::vector<std::vector<double>> weights;  // (n, H)
    int n;
    int H;
};

struct AtomsResult {
    std::vector<double> theta_mu;     // (H,)
    std::vector<double> theta_sigma2; // (H,)
    int H;
};

struct ULatentResult {
    std::vector<std::vector<double>> u_latent;  // (n, H)
    int n;
    int H;
};

struct UpdateEllResult {
    std::vector<std::vector<int>> ell;  // (H, p)
    int H;
    int p;
};

struct UpdateAlphaResult {
    std::vector<double> alpha;       // (H,)
    std::vector<bool> acceptances;   // (H,)
    int H;
};

struct PsiResult {
    std::vector<std::vector<double>> psi;             // (H, p)
    std::vector<std::vector<bool>> acceptances_psi;   // (H, p)
    int H;
    int p;
};

// ============================================================================
// FUNCIONES PRINCIPALES
// ============================================================================

// 1. Cálculo de η (SIN γ - todas las variables activas)
EtaResult compute_eta(
    const std::vector<std::vector<double>>& X_batch,
    const std::vector<double>& alpha,
    const std::vector<std::vector<double>>& psi,
    const std::vector<std::vector<int>>& ell,
    const std::vector<std::vector<double>>& ell_grid
);

// 2. Pesos con probit
WeightsResult compute_weights_probit(
    const std::vector<std::vector<double>>& X_normalized,
    const std::vector<double>& alpha,
    const std::vector<std::vector<double>>& psi,
    const std::vector<std::vector<int>>& ell,
    const std::vector<std::vector<double>>& ell_grid
);

// 3. Variables latentes (normal truncada)
ULatentResult update_u_latent(
    const std::vector<std::vector<double>>& u_latent_current,
    const std::vector<int>& z,
    const std::vector<std::vector<double>>& X_normalized,
    const std::vector<double>& alpha,
    const std::vector<std::vector<double>>& psi,
    const std::vector<std::vector<int>>& ell,
    const std::vector<std::vector<double>>& ell_grid
);

// 4. Asignaciones con slice sampling
std::vector<int> update_assignments_slice(
    const std::vector<double>& u_slice,
    const std::vector<std::vector<double>>& w,
    const std::vector<double>& y_normalized,
    const std::vector<double>& theta_mu,
    const std::vector<double>& theta_sigma2,
    const std::vector<int>& z_current
);

// 5. Actualización de átomos
AtomsResult update_atoms(
    const std::vector<int>& z,
    const std::vector<double>& y_normalized,
    const std::vector<double>& theta_mu_current,
    const std::vector<double>& theta_sigma2_current,
    double mu0,
    double kappa0,
    double a0,
    double b0,
    int H
);

// 6. Actualización de localizaciones (probit)
UpdateEllResult update_ell_probit(
    const std::vector<std::vector<int>>& ell_current,
    const std::vector<int>& z,
    const std::vector<std::vector<double>>& X_normalized,
    const std::vector<double>& alpha,
    const std::vector<std::vector<double>>& psi,
    const std::vector<std::vector<double>>& u_latent,
    const std::vector<std::vector<double>>& ell_grid,
    int H,
    int n_grid
);

// 7. Actualización de ψ con MH (SIN γ)
PsiResult update_psi_mh(
    const std::vector<std::vector<double>>& psi_current,
    const std::vector<double>& mu_psi,
    const std::vector<double>& tau_psi,
    const std::vector<std::vector<double>>& u_latent,
    const std::vector<std::vector<double>>& X_normalized,
    const std::vector<double>& alpha,
    const std::vector<std::vector<int>>& ell,
    const std::vector<std::vector<double>>& ell_grid,
    const std::vector<int>& z,
    double mh_scale_psi,
    bool psi_positive,
    int H
);

// 8. α con probit (MH)
UpdateAlphaResult update_alpha_probit(
    const std::vector<double>& alpha_current,
    const std::vector<std::vector<double>>& u_latent,
    const std::vector<std::vector<double>>& X_normalized,
    const std::vector<std::vector<double>>& psi,
    const std::vector<std::vector<int>>& ell,
    const std::vector<std::vector<double>>& ell_grid,
    const std::vector<int>& z,
    double mu_alpha,
    double mh_scale,
    int H
);

// ============================================================================
// FUNCIONES AUXILIARES
// ============================================================================

inline double norm_cdf(double x) {
    return 0.5 * std::erfc(-x * INV_SQRT2);
}

inline double norm_pdf(double x, double mu, double sigma) {
    const double sqrt_2pi = std::sqrt(2.0 * std::numbers::pi);
    const double z = (x - mu) / sigma;
    return std::exp(-0.5 * z * z) / (sigma * sqrt_2pi);
}

// Muestreo
double sample_truncated_normal(double mu, double sigma, double lower, double upper);
double sample_invgamma(double a, double b);
double sample_normal(double mu, double sigma);

// Utilidades
double log_uniform();
int sample_categorical(const std::vector<double>& probs);
int sample_bernoulli(double p);

} // namespace psbp

#endif // PSBP_CORE_HPP