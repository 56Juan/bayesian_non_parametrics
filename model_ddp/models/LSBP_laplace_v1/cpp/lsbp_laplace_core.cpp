// lsbp_laplace_core.cpp
#include "lsbp_laplace_core.hpp"
#include <random>
#include <numeric>
#include <cmath>
#include <algorithm>
#include <limits>

namespace lsbp_laplace {

// Generador global de números aleatorios
static std::random_device rd;
static std::mt19937 gen(rd());

// ============================================================================
// 1. compute_eta
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
// 2. compute_weights
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
    
    EtaResult eta_result = compute_eta(X_normalized, alpha, psi, ell, ell_grid);
    
    std::vector<std::vector<double>> v(n, std::vector<double>(H));
    for (int i = 0; i < n; ++i) {
        for (int h = 0; h < H; ++h) {
            v[i][h] = expit(eta_result.eta[i][h]);
        }
    }
    
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
// 3. update_lambda_latent
// ============================================================================
LambdaResult update_lambda_latent(
    const std::vector<int>& z,
    const std::vector<double>& y_normalized,
    const std::vector<double>& theta_mu,
    const std::vector<double>& theta_b,
    const std::vector<std::vector<double>>& lambda_current,
    int H
) {
    int n = y_normalized.size();
    
    LambdaResult result;
    result.n = n;
    result.H = H;
    result.lambda_latent = lambda_current;
    
    for (int i = 0; i < n; ++i) {
        int h = z[i];
        
        double residual_abs = std::abs(y_normalized[i] - theta_mu[h]);
        residual_abs = std::max(residual_abs, 1e-6);
        
        // Parámetros Inverse-Gaussian
        // λ_ih ~ InvGauss(μ=b_h/|y_i-μ_h|, λ=b_h²)
        double mu_ig = theta_b[h] / residual_abs;
        double lambda_ig = theta_b[h] * theta_b[h];
        
        // Muestrear
        result.lambda_latent[i][h] = sample_invgauss(mu_ig, lambda_ig);
        result.lambda_latent[i][h] = std::clamp(result.lambda_latent[i][h], 0.001, 100.0);
    }
    
    return result;
}

// ============================================================================
// 4. update_assignments
// ============================================================================
std::vector<int> update_assignments(
    const std::vector<double>& u,
    const std::vector<std::vector<double>>& w,
    const std::vector<double>& y_normalized,
    const std::vector<double>& theta_mu,
    const std::vector<double>& theta_b,
    const std::vector<int>& z_current
) {
    int n = u.size();
    int H = w[0].size();
    std::vector<int> z_new = z_current;
    
    for (int i = 0; i < n; ++i) {
        std::vector<int> candidates;
        for (int h = 0; h < H; ++h) {
            if (w[i][h] > u[i]) {
                candidates.push_back(h);
            }
        }
        
        if (candidates.empty()) {
            candidates.push_back(0);
        }
        
        std::vector<double> likes(candidates.size());
        for (size_t c = 0; c < candidates.size(); ++c) {
            int h = candidates[c];
            likes[c] = laplace_pdf(y_normalized[i], theta_mu[h], theta_b[h]);
            likes[c] = std::max(likes[c], 1e-300);
        }
        
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
        
        std::discrete_distribution<> categorical(probs.begin(), probs.end());
        int selected_idx = categorical(gen);
        z_new[i] = candidates[selected_idx];
    }
    
    return z_new;
}

// ============================================================================
// 5. update_atoms
// ============================================================================
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
) {
    int n = y_normalized.size();
    
    AtomsResult result;
    result.H = H;
    result.theta_mu.resize(H);
    result.theta_b.resize(H);
    
    for (int h = 0; h < H; ++h) {
        std::vector<int> members_idx;
        for (int i = 0; i < n; ++i) {
            if (z[i] == h) {
                members_idx.push_back(i);
            }
        }
        
        int n_h = members_idx.size();
        
        if (n_h > 0) {
            // ==========================================
            // Actualizar μ_h (Gibbs con mezcla Normal)
            // ==========================================
            double sum_inv_lambda = 0.0;
            double sum_y_over_lambda = 0.0;
            
            for (int idx : members_idx) {
                double lambda_ih = lambda_latent[idx][h];
                sum_inv_lambda += 1.0 / lambda_ih;
                sum_y_over_lambda += y_normalized[idx] / lambda_ih;
            }
            
            double tau_post = tau0 + sum_inv_lambda;
            double mu_post = (tau0 * mu0 + sum_y_over_lambda) / tau_post;
            
            result.theta_mu[h] = sample_normal(mu_post, 1.0 / std::sqrt(tau_post));
            
            // ==========================================
            // Actualizar b_h (Gibbs condicional en λ)
            // ==========================================
            double a_post = a0 + n_h;
            double beta_post = beta0 + sum_inv_lambda;
            
            result.theta_b[h] = sample_gamma(a_post, beta_post);
            result.theta_b[h] = std::clamp(result.theta_b[h], 0.01, 100.0);
            
        } else {
            // Cluster vacío: prior
            result.theta_b[h] = sample_gamma(a0, beta0);
            result.theta_b[h] = std::clamp(result.theta_b[h], 0.01, 100.0);
            result.theta_mu[h] = sample_normal(mu0, 1.0 / std::sqrt(tau0));
        }
    }
    
    return result;
}

// ============================================================================
// 6. update_alpha
// ============================================================================
AlphaUpdateResult update_alpha(
    const std::vector<double>& alpha_current,
    const std::vector<int>& z,
    const std::vector<std::vector<double>>& X_normalized,
    const std::vector<std::vector<double>>& psi,
    const std::vector<std::vector<int>>& ell,
    const std::vector<std::vector<double>>& ell_grid,
    double mu,
    double mh_scale
) {
    int H = alpha_current.size();
    int n = z.size();
    int p = X_normalized[0].size();
    
    AlphaUpdateResult result;
    result.H = H;
    result.alpha = alpha_current;
    result.acceptance.resize(H - 1, 0.0);
    
    std::normal_distribution<> normal_dist(0.0, 1.0);
    std::uniform_real_distribution<> uniform_dist(0.0, 1.0);
    
    for (int h = 0; h < H - 1; ++h) {
        double alpha_prop = alpha_current[h] + mh_scale * normal_dist(gen);
        
        std::vector<int> affected;
        for (int i = 0; i < n; ++i) {
            if (z[i] >= h) {
                affected.push_back(i);
            }
        }
        
        double log_r;
        
        if (affected.empty()) {
            double log_prior_curr = -0.5 * std::pow(alpha_current[h] - mu, 2);
            double log_prior_prop = -0.5 * std::pow(alpha_prop - mu, 2);
            log_r = log_prior_prop - log_prior_curr;
        } else {
            // Calcular eta para affected
            std::vector<double> eta_curr(affected.size());
            std::vector<double> eta_prop(affected.size());
            
            for (size_t idx_local = 0; idx_local < affected.size(); ++idx_local) {
                int i = affected[idx_local];
                
                eta_curr[idx_local] = alpha_current[h];
                eta_prop[idx_local] = alpha_prop;
                
                for (int j = 0; j < p; ++j) {
                    double ell_hj_value = ell_grid[j][ell[h][j]];
                    double dist = std::abs(X_normalized[i][j] - ell_hj_value);
                    eta_curr[idx_local] -= psi[h][j] * dist;
                    eta_prop[idx_local] -= psi[h][j] * dist;
                }
            }
            
            // Calcular likelihood
            double log_like_curr = 0.0;
            double log_like_prop = 0.0;
            
            for (size_t idx_local = 0; idx_local < affected.size(); ++idx_local) {
                int idx_global = affected[idx_local];
                double v_curr = expit(eta_curr[idx_local]);
                double v_prop = expit(eta_prop[idx_local]);
                
                if (z[idx_global] == h) {
                    log_like_curr += std::log(std::clamp(v_curr, 1e-10, 1.0));
                    log_like_prop += std::log(std::clamp(v_prop, 1e-10, 1.0));
                } else {
                    log_like_curr += std::log(std::clamp(1.0 - v_curr, 1e-10, 1.0));
                    log_like_prop += std::log(std::clamp(1.0 - v_prop, 1e-10, 1.0));
                }
            }
            
            double log_prior_curr = -0.5 * std::pow(alpha_current[h] - mu, 2);
            double log_prior_prop = -0.5 * std::pow(alpha_prop - mu, 2);
            
            log_r = (log_like_prop + log_prior_prop) - (log_like_curr + log_prior_curr);
        }
        
        log_r = std::clamp(log_r, -50.0, 50.0);
        
        bool accept = std::log(uniform_dist(gen)) < log_r;
        if (accept) {
            result.alpha[h] = alpha_prop;
            result.acceptance[h] = 1.0;
        }
    }
    
    return result;
}

// ============================================================================
// 7. update_psi
// ============================================================================
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
) {
    int H = psi_current.size();
    int p = psi_current[0].size();
    int n = z.size();
    
    PsiUpdateResult result;
    result.H = H;
    result.p = p;
    result.psi = psi_current;
    
    std::normal_distribution<> normal_dist(0.0, 1.0);
    std::uniform_real_distribution<> uniform_dist(0.0, 1.0);
    
    for (int h = 0; h < H - 1; ++h) {
        for (int j = 0; j < p; ++j) {
            double psi_curr = psi_current[h][j];
            double psi_prop = psi_curr + mh_scale * normal_dist(gen);
            
            if (psi_prop < 0) {
                result.acceptance.push_back(0.0);
                continue;
            }
            
            std::vector<int> affected;
            for (int i = 0; i < n; ++i) {
                if (z[i] >= h) {
                    affected.push_back(i);
                }
            }
            
            double log_r;
            
            if (affected.empty()) {
                double log_prior_curr = -0.5 * std::pow(psi_curr - mu_psi, 2) / tau_psi_inv;
                double log_prior_prop = -0.5 * std::pow(psi_prop - mu_psi, 2) / tau_psi_inv;
                log_r = log_prior_prop - log_prior_curr;
            } else {
                double ell_hj_value = ell_grid[j][ell[h][j]];
                
                // Calcular eta
                std::vector<double> eta_curr(affected.size());
                std::vector<double> dist_j(affected.size());
                
                for (size_t idx_local = 0; idx_local < affected.size(); ++idx_local) {
                    int i = affected[idx_local];
                    eta_curr[idx_local] = alpha[h];
                    
                    for (int jj = 0; jj < p; ++jj) {
                        double ell_hjj_value = ell_grid[jj][ell[h][jj]];
                        double dist = std::abs(X_normalized[i][jj] - ell_hjj_value);
                        eta_curr[idx_local] -= psi_current[h][jj] * dist;
                        
                        if (jj == j) {
                            dist_j[idx_local] = dist;
                        }
                    }
                }
                
                // eta_prop = eta_curr + (psi_curr - psi_prop) * dist_j
                std::vector<double> eta_prop(affected.size());
                for (size_t idx_local = 0; idx_local < affected.size(); ++idx_local) {
                    eta_prop[idx_local] = eta_curr[idx_local] + 
                                         (psi_curr - psi_prop) * dist_j[idx_local];
                }
                
                // Calcular likelihood
                double log_like_curr = 0.0;
                double log_like_prop = 0.0;
                
                for (size_t idx_local = 0; idx_local < affected.size(); ++idx_local) {
                    int idx_global = affected[idx_local];
                    double v_curr = expit(eta_curr[idx_local]);
                    double v_prop = expit(eta_prop[idx_local]);
                    
                    if (z[idx_global] == h) {
                        log_like_curr += std::log(std::clamp(v_curr, 1e-10, 1.0));
                        log_like_prop += std::log(std::clamp(v_prop, 1e-10, 1.0));
                    } else {
                        log_like_curr += std::log(std::clamp(1.0 - v_curr, 1e-10, 1.0));
                        log_like_prop += std::log(std::clamp(1.0 - v_prop, 1e-10, 1.0));
                    }
                }
                
                double log_prior_curr = -0.5 * std::pow(psi_curr - mu_psi, 2) / tau_psi_inv;
                double log_prior_prop = -0.5 * std::pow(psi_prop - mu_psi, 2) / tau_psi_inv;
                
                log_r = (log_like_prop + log_prior_prop) - (log_like_curr + log_prior_curr);
            }
            
            log_r = std::clamp(log_r, -50.0, 50.0);
            
            bool accept = std::log(uniform_dist(gen)) < log_r;
            if (accept) {
                result.psi[h][j] = psi_prop;
                result.acceptance.push_back(1.0);
            } else {
                result.acceptance.push_back(0.0);
            }
        }
    }
    
    return result;
}

// ============================================================================
// 8. update_ell
// ============================================================================
EllUpdateResult update_ell(
    const std::vector<std::vector<int>>& ell_current,
    const std::vector<int>& z,
    const std::vector<double>& alpha,
    const std::vector<std::vector<double>>& psi,
    const std::vector<std::vector<double>>& X_normalized,
    const std::vector<std::vector<double>>& ell_grid,
    int n_grid
) {
    int H = ell_current.size();
    int p = ell_current[0].size();
    int n = z.size();
    
    EllUpdateResult result;
    result.H = H;
    result.p = p;
    result.ell = ell_current;
    
    std::uniform_int_distribution<> uniform_int(0, n_grid - 1);
    
    for (int h = 0; h < H - 1; ++h) {
        for (int j = 0; j < p; ++j) {
            std::vector<int> affected;
            for (int i = 0; i < n; ++i) {
                if (z[i] >= h) {
                    affected.push_back(i);
                }
            }
            
            if (affected.empty()) {
                result.ell[h][j] = uniform_int(gen);
                continue;
            }
            
            std::vector<double> log_likes(n_grid, 0.0);
            
            for (int m = 0; m < n_grid; ++m) {
                double ell_value = ell_grid[j][m];
                
                for (size_t idx_local = 0; idx_local < affected.size(); ++idx_local) {
                    int i = affected[idx_local];
                    
                    // Calcular eta con ℓ_{hj} = ell_value
                    double eta = alpha[h];
                    
                    for (int jj = 0; jj < p; ++jj) {
                        double ell_jj_value;
                        if (jj == j) {
                            ell_jj_value = ell_value;
                        } else {
                            ell_jj_value = ell_grid[jj][ell_current[h][jj]];
                        }
                        
                        double dist = std::abs(X_normalized[i][jj] - ell_jj_value);
                        eta -= psi[h][jj] * dist;
                    }
                    
                    double v = expit(eta);
                    
                    if (z[i] == h) {
                        log_likes[m] += std::log(std::clamp(v, 1e-10, 1.0));
                    } else {
                        log_likes[m] += std::log(std::clamp(1.0 - v, 1e-10, 1.0));
                    }
                }
            }
            
            // Normalizar y muestrear
            double max_log_like = *std::max_element(log_likes.begin(), log_likes.end());
            std::vector<double> probs(n_grid);
            double sum_probs = 0.0;
            
            for (int m = 0; m < n_grid; ++m) {
                probs[m] = std::exp(log_likes[m] - max_log_like);
                sum_probs += probs[m];
            }
            
            for (int m = 0; m < n_grid; ++m) {
                probs[m] /= sum_probs;
            }
            
            std::discrete_distribution<> categorical(probs.begin(), probs.end());
            result.ell[h][j] = categorical(gen);
        }
    }
    
    return result;
}

// ============================================================================
// Funciones auxiliares de muestreo
// ============================================================================

double sample_invgauss(double mu_ig, double lambda_ig) {
    // Muestreo de Inverse-Gaussian usando método de Michael, Schucany y Haas
    std::normal_distribution<> normal(0.0, 1.0);
    std::uniform_real_distribution<> uniform(0.0, 1.0);
    
    double nu = normal(gen);
    double y = nu * nu;
    
    double x = mu_ig + (mu_ig * mu_ig * y) / (2.0 * lambda_ig) - 
               (mu_ig / (2.0 * lambda_ig)) * 
               std::sqrt(4.0 * mu_ig * lambda_ig * y + mu_ig * mu_ig * y * y);
    
    double test = uniform(gen);
    if (test <= mu_ig / (mu_ig + x)) {
        return x;
    } else {
        return mu_ig * mu_ig / x;
    }
}

double sample_normal(double mu, double sigma) {
    std::normal_distribution<> normal_dist(mu, sigma);
    return normal_dist(gen);
}

double sample_gamma(double a, double b) {
    // b es el rate parameter (β)
    std::gamma_distribution<> gamma_dist(a, 1.0 / b);
    return gamma_dist(gen);
}

} // namespace lsbp_laplace