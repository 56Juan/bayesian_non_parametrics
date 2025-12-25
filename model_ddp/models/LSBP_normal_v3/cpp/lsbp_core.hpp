// model_ddp/models/LSBP_normal_v3/cpp/lsbp_core.hpp
#ifndef LSBP_CORE_HPP
#define LSBP_CORE_HPP

#include <vector>
#include <cmath>
#include <algorithm>
#include <numbers>

namespace lsbp {

// ============================================================================
// ESTRUCTURAS EXISTENTES
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
    std::vector<double> theta_sigma2;
    int H;
};

// ============================================================================
// NUEVAS ESTRUCTURAS PARA LAS 3 FUNCIONES OBJETIVO
// ============================================================================

// Resultado de update_ell
struct UpdateEllResult {
    std::vector<std::vector<int>> ell;  // Nueva matriz ell (H, p)
    int H;
    int p;
};

// Resultado de update_psi
struct UpdatePsiResult {
    std::vector<std::vector<double>> psi;  // Nueva matriz psi (H, p)
    std::vector<bool> acceptances;         // Vector de aceptaciones
    int H;
    int p;
};

// Resultado de update_alpha
struct UpdateAlphaResult {
    std::vector<double> alpha;      // Nuevo vector alpha (H,)
    std::vector<bool> acceptances;  // Vector de aceptaciones
    int H;
};

// ============================================================================
// FUNCIONES EXISTENTES
// ============================================================================

EtaResult compute_eta(
    const std::vector<std::vector<double>>& X_batch,
    const std::vector<double>& alpha,
    const std::vector<std::vector<double>>& psi,
    const std::vector<std::vector<int>>& ell,
    const std::vector<std::vector<double>>& ell_grid
);

WeightsResult compute_weights(
    const std::vector<std::vector<double>>& X_normalized,
    const std::vector<double>& alpha,
    const std::vector<std::vector<double>>& psi,
    const std::vector<std::vector<int>>& ell,
    const std::vector<std::vector<double>>& ell_grid
);

std::vector<int> update_assignments(
    const std::vector<double>& u,
    const std::vector<std::vector<double>>& w,
    const std::vector<double>& y_normalized,
    const std::vector<double>& theta_mu,
    const std::vector<double>& theta_sigma2,
    const std::vector<int>& z_current
);

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

// ============================================================================
// NUEVAS FUNCIONES OBJETIVO
// ============================================================================

/**
 * Update location parameters ℓ_hj using discrete sampling
 * 
 * For each cluster h and covariate j:
 *   - Compute log-likelihood for all grid positions
 *   - Sample new position from categorical distribution
 * 
 * Optimization: Vectorized distance calculations and log-likelihood evaluation
 */
UpdateEllResult update_ell(
    const std::vector<std::vector<int>>& ell_current,
    const std::vector<int>& z,
    const std::vector<std::vector<double>>& X_normalized,
    const std::vector<double>& alpha,
    const std::vector<std::vector<double>>& psi,
    const std::vector<std::vector<double>>& ell_grid,
    int H,
    int n_grid
);

/**
 * Update decay parameters ψ_hj using Metropolis-Hastings
 * 
 * For each cluster h and covariate j:
 *   - Propose new value from normal random walk
 *   - Compute acceptance ratio (likelihood + prior)
 *   - Accept/reject proposal
 * 
 * Optimization: Incremental eta calculation (reuse existing computation)
 */
UpdatePsiResult update_psi(
    const std::vector<std::vector<double>>& psi_current,
    const std::vector<int>& z,
    const std::vector<std::vector<double>>& X_normalized,
    const std::vector<double>& alpha,
    const std::vector<std::vector<int>>& ell,
    const std::vector<std::vector<double>>& ell_grid,
    double mu_psi,
    double tau_psi_inv,
    double mh_scale,
    int H
);

/**
 * Update intercept parameters α_h using Metropolis-Hastings
 * 
 * For each cluster h:
 *   - Propose new value from normal random walk
 *   - Compute acceptance ratio (likelihood + prior)
 *   - Accept/reject proposal
 * 
 * Optimization: Vectorized eta calculation for affected observations
 */
UpdateAlphaResult update_alpha(
    const std::vector<double>& alpha_current,
    const std::vector<int>& z,
    const std::vector<std::vector<double>>& X_normalized,
    const std::vector<std::vector<double>>& psi,
    const std::vector<std::vector<int>>& ell,
    const std::vector<std::vector<double>>& ell_grid,
    double mu,
    double mh_scale,
    int H
);

// ============================================================================
// FUNCIONES AUXILIARES
// ============================================================================

inline double expit(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

inline double norm_pdf(double x, double mu, double sigma) {
    const double sqrt_2pi = std::sqrt(2.0 * std::numbers::pi);
    double z = (x - mu) / sigma;
    return std::exp(-0.5 * z * z) / (sigma * sqrt_2pi);
}

double sample_invgamma(double a, double b);
double sample_normal(double mu, double sigma);

// Nuevas auxiliares para las 3 funciones
double log_uniform();  // log(U(0,1))
int sample_categorical(const std::vector<double>& probs);

} // namespace lsbp

#endif // LSBP_CORE_HPP