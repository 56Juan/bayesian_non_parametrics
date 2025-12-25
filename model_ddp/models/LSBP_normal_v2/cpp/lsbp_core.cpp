// model_ddp/models/LSBP_normal_v2/cpp/lsbp_core.cpp
#include "lsbp_core.hpp"
#include <random>
#include <numeric>
#include <cmath>
#include <algorithm>
#include <limits>

namespace lsbp {

// Generador global de números aleatorios
static std::random_device rd;
static std::mt19937 gen(rd());

// ============================================================================
// Implementación de compute_eta
// ============================================================================
EtaResult compute_eta(
    const std::vector<std::vector<double>>& X_batch,
    const std::vector<double>& alpha,
    const std::vector<std::vector<double>>& psi,
    const std::vector<std::vector<int>>& ell,
    const std::vector<std::vector<double>>& ell_grid
) {
    int n_batch = X_batch.size();
    int p = X_batch[0].size();
    int H = alpha.size();
    
    EtaResult result;
    result.n_batch = n_batch;
    result.H = H;
    result.eta.resize(n_batch, std::vector<double>(H, 0.0));
    
    // Paralelizable con OpenMP
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n_batch; ++i) {
        for (int h = 0; h < H; ++h) {
            result.eta[i][h] = alpha[h];
            
            for (int j = 0; j < p; ++j) {
                double ell_hj_value = ell_grid[j][ell[h][j]];
                double dist = std::abs(X_batch[i][j] - ell_hj_value);
                result.eta[i][h] -= psi[h][j] * dist;
            }
        }
    }
    
    return result;
}

// ============================================================================
// Implementación de compute_weights
// ============================================================================
WeightsResult compute_weights(
    const std::vector<std::vector<double>>& X_normalized,
    const std::vector<double>& alpha,
    const std::vector<std::vector<double>>& psi,
    const std::vector<std::vector<int>>& ell,
    const std::vector<std::vector<double>>& ell_grid
) {
    int n = X_normalized.size();
    int H = alpha.size();
    
    // Calcular eta
    EtaResult eta_result = compute_eta(X_normalized, alpha, psi, ell, ell_grid);
    
    // Calcular v = expit(eta)
    std::vector<std::vector<double>> v(n, std::vector<double>(H));
    for (int i = 0; i < n; ++i) {
        for (int h = 0; h < H; ++h) {
            v[i][h] = expit(eta_result.eta[i][h]);
        }
    }
    
    // Stick-breaking
    WeightsResult result;
    result.n = n;
    result.H = H;
    result.weights.resize(n, std::vector<double>(H, 0.0));
    
    for (int i = 0; i < n; ++i) {
        double remaining = 1.0;
        for (int h = 0; h < H; ++h) {
            result.weights[i][h] = v[i][h] * remaining;
            remaining *= (1.0 - v[i][h]);
        }
        
        // Normalizar
        double sum = std::accumulate(result.weights[i].begin(), 
                                     result.weights[i].end(), 0.0);
        if (sum > 0) {
            for (int h = 0; h < H; ++h) {
                result.weights[i][h] /= sum;
            }
        }
    }
    
    return result;
}

// ============================================================================
// Implementación de update_assignments
// ============================================================================
std::vector<int> update_assignments(
    const std::vector<double>& u,
    const std::vector<std::vector<double>>& w,
    const std::vector<double>& y_normalized,
    const std::vector<double>& theta_mu,
    const std::vector<double>& theta_sigma2,
    const std::vector<int>& z_current
) {
    int n = u.size();
    int H = w[0].size();
    std::vector<int> z_new = z_current;
    
    std::uniform_real_distribution<> uniform(0.0, 1.0);
    
    for (int i = 0; i < n; ++i) {
        // Encontrar clusters activos: w_h(x_i) > u_i
        std::vector<int> candidates;
        for (int h = 0; h < H; ++h) {
            if (w[i][h] > u[i]) {
                candidates.push_back(h);
            }
        }
        
        if (candidates.empty()) {
            candidates.push_back(0);
        }
        
        // Calcular likelihood para cada candidato
        std::vector<double> likes(candidates.size());
        for (size_t c = 0; c < candidates.size(); ++c) {
            int h = candidates[c];
            double sigma = std::sqrt(theta_sigma2[h]);
            likes[c] = norm_pdf(y_normalized[i], theta_mu[h], sigma);
            likes[c] = std::max(likes[c], 1e-300);  // Clip
        }
        
        // Calcular probabilidades
        double sum_likes = std::accumulate(likes.begin(), likes.end(), 0.0);
        std::vector<double> probs(likes.size());
        
        if (sum_likes > 0) {
            for (size_t c = 0; c < likes.size(); ++c) {
                probs[c] = likes[c] / sum_likes;
            }
        } else {
            double uniform_prob = 1.0 / likes.size();
            for (size_t c = 0; c < likes.size(); ++c) {
                probs[c] = uniform_prob;
            }
        }
        
        // Muestrear usando distribución categórica
        std::discrete_distribution<> categorical(probs.begin(), probs.end());
        int selected_idx = categorical(gen);
        z_new[i] = candidates[selected_idx];
    }
    
    return z_new;
}

// ============================================================================
// Implementación de update_atoms
// ============================================================================
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
) {
    AtomsResult result;
    result.H = H;
    result.theta_mu.resize(H);
    result.theta_sigma2.resize(H);
    
    int n = y_normalized.size();
    
    for (int h = 0; h < H; ++h) {
        // Encontrar miembros del cluster h
        std::vector<double> members;
        for (int i = 0; i < n; ++i) {
            if (z[i] == h) {
                members.push_back(y_normalized[i]);
            }
        }
        
        int n_h = members.size();
        
        if (n_h > 0) {
            // Posterior Normal-Inverse-Gamma
            double y_bar = std::accumulate(members.begin(), members.end(), 0.0) / n_h;
            
            double ss = 0.0;
            for (double y : members) {
                ss += (y - y_bar) * (y - y_bar);
            }
            
            double kappa_n = kappa0 + n_h;
            double mu_n = (kappa0 * mu0 + n_h * y_bar) / kappa_n;
            double a_n = a0 + n_h / 2.0;
            double b_n = b0 + 0.5 * ss + 
                         (kappa0 * n_h * (y_bar - mu0) * (y_bar - mu0)) / (2.0 * kappa_n);
            
            // Muestrear σ²_h | y
            result.theta_sigma2[h] = sample_invgamma(a_n, b_n);
            result.theta_sigma2[h] = std::clamp(result.theta_sigma2[h], 0.01, 100.0);
            
            // Muestrear μ_h | σ²_h, y
            double sigma_mu = std::sqrt(result.theta_sigma2[h] / kappa_n);
            result.theta_mu[h] = sample_normal(mu_n, sigma_mu);
            
        } else {
            // Prior: muestrear de G₀
            result.theta_sigma2[h] = sample_invgamma(a0, b0);
            result.theta_sigma2[h] = std::clamp(result.theta_sigma2[h], 0.01, 100.0);
            
            double sigma_mu = std::sqrt(result.theta_sigma2[h] / kappa0);
            result.theta_mu[h] = sample_normal(mu0, sigma_mu);
        }
    }
    
    return result;
}

// ============================================================================
// Funciones auxiliares de muestreo
// ============================================================================

double sample_invgamma(double a, double b) {
    // InvGamma(a, b) se puede muestrear como 1/Gamma(a, 1/b)
    std::gamma_distribution<> gamma_dist(a, 1.0 / b);
    double gamma_sample = gamma_dist(gen);
    return 1.0 / gamma_sample;
}

double sample_normal(double mu, double sigma) {
    std::normal_distribution<> normal_dist(mu, sigma);
    return normal_dist(gen);
}

} // namespace lsbp