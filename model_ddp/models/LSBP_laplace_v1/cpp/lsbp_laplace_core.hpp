// lsbp_laplace_core.hpp
#ifndef LSBP_LAPLACE_CORE_HPP
#define LSBP_LAPLACE_CORE_HPP

#include <vector>
#include <cmath>
#include <algorithm>

namespace lsbp_laplace {

// ============================================================================
// Estructuras de resultado
// ============================================================================

struct EtaResult {
    std::vector<std::vector<double>> eta;
    int n_batch;
    int H;
};

struct WeightsResult {
    std::vector<std::vector<double>> weights;
    int n;
    int H;
};

struct AtomsResult {
    std::vector<double> theta_mu;
    std::vector<double> theta_b;
    int H;
};

struct LambdaResult {
    std::vector<std::vector<double>> lambda_latent;
    int n;
    int H;
};

struct AlphaUpdateResult {
    std::vector<double> alpha;
    std::vector<double> acceptance;
    int H;
};

struct PsiUpdateResult {
    std::vector<std::vector<double>> psi;
    std::vector<double> acceptance;
    int H;
    int p;
};

struct EllUpdateResult {
    std::vector<std::vector<int>> ell;
    int H;
    int p;
};

// ============================================================================
// Funciones principales
// ============================================================================

// 1. Calcular η_h(x)
EtaResult compute_eta(
    const std::vector<std::vector<double>>& X_batch,
    const std::vector<double>& alpha,
    const std::vector<std::vector<double>>& psi,
    const std::vector<std::vector<int>>& ell,
    const std::vector<std::vector<double>>& ell_grid
);

// 2. Calcular pesos w_h(x)
WeightsResult compute_weights(
    const std::vector<std::vector<double>>& X_normalized,
    const std::vector<double>& alpha,
    const std::vector<std::vector<double>>& psi,
    const std::vector<std::vector<int>>& ell,
    const std::vector<std::vector<double>>& ell_grid
);

// 3. Actualizar variables latentes λ_ih
LambdaResult update_lambda_latent(
    const std::vector<int>& z,
    const std::vector<double>& y_normalized,
    const std::vector<double>& theta_mu,
    const std::vector<double>& theta_b,
    const std::vector<std::vector<double>>& lambda_current,
    int H
);

// 4. Actualizar asignaciones z_i
std::vector<int> update_assignments(
    const std::vector<double>& u,
    const std::vector<std::vector<double>>& w,
    const std::vector<double>& y_normalized,
    const std::vector<double>& theta_mu,
    const std::vector<double>& theta_b,
    const std::vector<int>& z_current
);

// 5. Actualizar átomos θ_h = (μ_h, b_h)
AtomsResult update_atoms(
    const std::vector<int>& z,
    const std::vector<double>& y_normalized,
    const std::vector<std::vector<double>>& lambda_latent,
    const std::vector<double>& theta_mu_current,
    const std::vector<double>& theta_b_current,
    double mu0,
    double tau0,
    double a0,
    double beta0,
    int H
);

// 6. Actualizar α_h con Metropolis-Hastings
AlphaUpdateResult update_alpha(
    const std::vector<double>& alpha_current,
    const std::vector<int>& z,
    const std::vector<std::vector<double>>& X_normalized,
    const std::vector<std::vector<double>>& psi,
    const std::vector<std::vector<int>>& ell,
    const std::vector<std::vector<double>>& ell_grid,
    double mu,
    double mh_scale
);

// 7. Actualizar ψ_{hj} con Metropolis-Hastings
PsiUpdateResult update_psi(
    const std::vector<std::vector<double>>& psi_current,
    const std::vector<int>& z,
    const std::vector<double>& alpha,
    const std::vector<std::vector<double>>& X_normalized,
    const std::vector<std::vector<int>>& ell,
    const std::vector<std::vector<double>>& ell_grid,
    double mu_psi,
    double tau_psi_inv,
    double mh_scale
);

// 8. Actualizar ℓ_{hj} (muestreo discreto)
EllUpdateResult update_ell(
    const std::vector<std::vector<int>>& ell_current,
    const std::vector<int>& z,
    const std::vector<double>& alpha,
    const std::vector<std::vector<double>>& psi,
    const std::vector<std::vector<double>>& X_normalized,
    const std::vector<std::vector<double>>& ell_grid,
    int n_grid
);

// ============================================================================
// Funciones auxiliares
// ============================================================================

// Sigmoid/expit
inline double expit(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

// PDF de Laplace
inline double laplace_pdf(double x, double mu, double b) {
    return (1.0 / (2.0 * b)) * std::exp(-std::abs(x - mu) / b);
}

// Muestreo de Inverse-Gaussian
double sample_invgauss(double mu_ig, double lambda_ig);

// Muestreo de Normal
double sample_normal(double mu, double sigma);

// Muestreo de Gamma
double sample_gamma(double a, double b);

} // namespace lsbp_laplace

#endif // LSBP_LAPLACE_CORE_HPP