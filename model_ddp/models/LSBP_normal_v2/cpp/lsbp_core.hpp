// model_ddp/models/LSBP_normal_v2/cpp/lsbp_core.hpp
#ifndef LSBP_CORE_HPP
#define LSBP_CORE_HPP

#include <vector>
#include <cmath>
#include <algorithm>
#include <numbers>


namespace lsbp {

// Estructura para almacenar resultados de compute_eta
struct EtaResult {
    std::vector<std::vector<double>> eta;
    int n_batch;
    int H;
};

// Estructura para almacenar pesos
struct WeightsResult {
    std::vector<std::vector<double>> weights;
    int n;
    int H;
};

// Estructura para almacenar átomos actualizados
struct AtomsResult {
    std::vector<double> theta_mu;
    std::vector<double> theta_sigma2;
    int H;
};

// Función principal: calcular η_h(x) para todos los clusters
EtaResult compute_eta(
    const std::vector<std::vector<double>>& X_batch,
    const std::vector<double>& alpha,
    const std::vector<std::vector<double>>& psi,
    const std::vector<std::vector<int>>& ell,
    const std::vector<std::vector<double>>& ell_grid
);

// Función: calcular pesos mediante logit stick-breaking
WeightsResult compute_weights(
    const std::vector<std::vector<double>>& X_normalized,
    const std::vector<double>& alpha,
    const std::vector<std::vector<double>>& psi,
    const std::vector<std::vector<int>>& ell,
    const std::vector<std::vector<double>>& ell_grid
);

// Función: actualizar asignaciones z_i dado u_i
std::vector<int> update_assignments(
    const std::vector<double>& u,
    const std::vector<std::vector<double>>& w,
    const std::vector<double>& y_normalized,
    const std::vector<double>& theta_mu,
    const std::vector<double>& theta_sigma2,
    const std::vector<int>& z_current
);

// Función: actualizar átomos θ_h = (μ_h, σ²_h)
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

// Función auxiliar: sigmoid/expit
inline double expit(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

// Función auxiliar: PDF de la distribución normal
inline double norm_pdf(double x, double mu, double sigma) {
    const double sqrt_2pi = std::sqrt(2.0 * std::numbers::pi);
    double z = (x - mu) / sigma;
    return std::exp(-0.5 * z * z) / (sigma * sqrt_2pi);
}

// Función auxiliar: muestreo de Inverse-Gamma
double sample_invgamma(double a, double b);

// Función auxiliar: muestreo de Normal
double sample_normal(double mu, double sigma);

} // namespace lsbp

#endif // LSBP_CORE_HPP