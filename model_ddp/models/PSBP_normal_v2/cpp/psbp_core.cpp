// PSBP_normal/cpp/psbp_core.cpp
#include "psbp_core.hpp"

#include <random>
#include <numeric>
#include <cmath>
#include <algorithm>
#include <limits>

namespace psbp {

// ============================================================================
// RNG (thread-local)
// ============================================================================

static thread_local std::mt19937 gen(std::random_device{}());

// ============================================================================
// FUNCIONES AUXILIARES
// ============================================================================

double log_uniform() {
    std::uniform_real_distribution<> unif(0.0, 1.0);
    return std::log(unif(gen));
}

int sample_categorical(const std::vector<double>& probs) {
    std::discrete_distribution<> dist(probs.begin(), probs.end());
    return dist(gen);
}

int sample_bernoulli(double p) {
    std::bernoulli_distribution dist(p);
    return dist(gen) ? 1 : 0;
}

double sample_truncated_normal(double mu, double sigma, double lower, double upper) {
    std::normal_distribution<> normal(0.0, 1.0);

    const double a = (lower - mu) / sigma;
    const double b = (upper - mu) / sigma;

    double z;
    do {
        z = normal(gen);
    } while (z < a || z > b);

    return mu + sigma * z;
}

double sample_invgamma(double a, double b) {
    std::gamma_distribution<> gamma_dist(a, 1.0 / b);
    return 1.0 / gamma_dist(gen);
}

double sample_normal(double mu, double sigma) {
    std::normal_distribution<> normal(mu, sigma);
    return normal(gen);
}

// ============================================================================
// 1. COMPUTE_ETA
// ============================================================================

EtaResult compute_eta(
    const std::vector<std::vector<double>>& X_batch,
    const std::vector<double>& alpha,
    const std::vector<std::vector<double>>& psi,
    const std::vector<std::vector<int>>& gamma,
    const std::vector<std::vector<int>>& ell,
    const std::vector<std::vector<double>>& ell_grid
) {
    const int n_batch = static_cast<int>(X_batch.size());
    const int p = static_cast<int>(X_batch[0].size());
    const int H = static_cast<int>(alpha.size());

    EtaResult result{ {}, n_batch, H };
    result.eta.assign(n_batch, std::vector<double>(H, 0.0));

    #pragma omp parallel for collapse(2) if(n_batch * H > 1000)
    for (int i = 0; i < n_batch; ++i) {
        for (int h = 0; h < H; ++h) {
            double eta = alpha[h];

            for (int j = 0; j < p; ++j) {
                if (gamma[h][j] == 1) {
                    const double ell_val = ell_grid[j][ell[h][j]];
                    const double dist = std::abs(X_batch[i][j] - ell_val);
                    eta -= psi[h][j] * dist;
                }
            }
            result.eta[i][h] = eta;
        }
    }

    return result;
}

// ============================================================================
// 2. COMPUTE_WEIGHTS_PROBIT
// ============================================================================

WeightsResult compute_weights_probit(
    const std::vector<std::vector<double>>& X_normalized,
    const std::vector<double>& alpha,
    const std::vector<std::vector<double>>& psi,
    const std::vector<std::vector<int>>& gamma,
    const std::vector<std::vector<int>>& ell,
    const std::vector<std::vector<double>>& ell_grid
) {
    const int n = static_cast<int>(X_normalized.size());
    const int H = static_cast<int>(alpha.size());

    EtaResult eta = compute_eta(X_normalized, alpha, psi, gamma, ell, ell_grid);

    WeightsResult result{ {}, n, H };
    result.weights.assign(n, std::vector<double>(H, 0.0));

    for (int i = 0; i < n; ++i) {
        double remaining = 1.0;

        for (int h = 0; h < H; ++h) {
            const double v = std::clamp(norm_cdf(eta.eta[i][h]), 1e-10, 1.0 - 1e-10);
            result.weights[i][h] = v * remaining;
            remaining *= (1.0 - v);
        }

        const double sum_w = std::accumulate(result.weights[i].begin(),
                                             result.weights[i].end(), 0.0);
        if (sum_w > 0.0) {
            for (double& w : result.weights[i]) {
                w /= sum_w;
            }
        }
    }

    return result;
}

// ============================================================================
// 3. UPDATE_U_LATENT
// ============================================================================

ULatentResult update_u_latent(
    const std::vector<std::vector<double>>& u_latent_current,
    const std::vector<int>& z,
    const std::vector<std::vector<double>>& X_normalized,
    const std::vector<double>& alpha,
    const std::vector<std::vector<double>>& psi,
    const std::vector<std::vector<int>>& gamma,
    const std::vector<std::vector<int>>& ell,
    const std::vector<std::vector<double>>& ell_grid
) {
    const int n = static_cast<int>(X_normalized.size());
    const int H = static_cast<int>(alpha.size());

    ULatentResult result{ u_latent_current, n, H };
    EtaResult eta = compute_eta(X_normalized, alpha, psi, gamma, ell, ell_grid);

    for (int i = 0; i < n; ++i) {
        for (int h = 0; h <= z[i]; ++h) {
            const double mu = eta.eta[i][h];

            if (z[i] == h) {
                result.u_latent[i][h] =
                    sample_truncated_normal(mu, 1.0, 0.0,
                        std::numeric_limits<double>::infinity());
            } else {
                result.u_latent[i][h] =
                    sample_truncated_normal(mu, 1.0,
                        -std::numeric_limits<double>::infinity(), 0.0);
            }
        }
    }

    return result;
}

// ============================================================================
// 4. UPDATE_ASSIGNMENTS_SLICE
// ============================================================================

std::vector<int> update_assignments_slice(
    const std::vector<double>& u_slice,
    const std::vector<std::vector<double>>& w,
    const std::vector<double>& y_normalized,
    const std::vector<double>& theta_mu,
    const std::vector<double>& theta_sigma2,
    const std::vector<int>& z_current
) {
    const int n = static_cast<int>(u_slice.size());
    const int H = static_cast<int>(w[0].size());

    std::vector<int> z_new = z_current;
    const double sqrt_2pi = std::sqrt(2.0 * std::numbers::pi);

    for (int i = 0; i < n; ++i) {
        std::vector<int> candidates;
        for (int h = 0; h < H; ++h) {
            if (w[i][h] > u_slice[i]) {
                candidates.push_back(h);
            }
        }

        if (candidates.empty()) {
            for (int h = 0; h < H; ++h) candidates.push_back(h);
        }

        std::vector<double> probs(candidates.size());
        for (size_t c = 0; c < candidates.size(); ++c) {
            const int h = candidates[c];
            const double sigma = std::sqrt(theta_sigma2[h]);
            const double diff = y_normalized[i] - theta_mu[h];

            // Likelihood
            double like = std::exp(-0.5 * diff * diff / (sigma * sigma)) / (sigma * sqrt_2pi);
            like = std::max(like, 1e-300);
            
            //  CRÍTICO: Multiplicar por peso
            probs[c] = w[i][h] * like;  // 
        }

        const double sum_probs = std::accumulate(probs.begin(), probs.end(), 0.0);
        if (sum_probs > 1e-300) {
            for (double& p : probs) p /= sum_probs;
        } else {
            for (double& p : probs) p = 1.0 / probs.size();
        }

        z_new[i] = candidates[sample_categorical(probs)];
    }

    return z_new;
}

// ============================================================================
// 5. UPDATE_ATOMS (SIN CAMBIOS)
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
        // Contar miembros del cluster
        int n_h = 0;
        double sum_y = 0.0;
        double sum_y2 = 0.0;
        
        for (int i = 0; i < n; ++i) {
            if (z[i] == h) {
                n_h++;
                sum_y += y_normalized[i];
                sum_y2 += y_normalized[i] * y_normalized[i];
            }
        }
        
        if (n_h > 0) {
            double y_bar = sum_y / n_h;
            double ss = sum_y2 - n_h * y_bar * y_bar;
            
            double kappa_n = kappa0 + n_h;
            double mu_n = (kappa0 * mu0 + n_h * y_bar) / kappa_n;
            double a_n = a0 + n_h / 2.0;
            double b_n = b0 + 0.5 * ss + (kappa0 * n_h * (y_bar - mu0) * (y_bar - mu0)) / (2.0 * kappa_n);
            
            // Muestrear σ²_h
            result.theta_sigma2[h] = sample_invgamma(a_n, b_n);
            result.theta_sigma2[h] = std::clamp(result.theta_sigma2[h], 0.01, 100.0);
            
            // Muestrear μ_h
            double sigma_mu = std::sqrt(result.theta_sigma2[h] / kappa_n);
            result.theta_mu[h] = sample_normal(mu_n, sigma_mu);
        } else {
            // Prior para cluster vacío
            result.theta_sigma2[h] = sample_invgamma(a0, b0);
            result.theta_sigma2[h] = std::clamp(result.theta_sigma2[h], 0.01, 100.0);
            
            double sigma_mu = std::sqrt(result.theta_sigma2[h] / kappa0);
            result.theta_mu[h] = sample_normal(mu0, sigma_mu);
        }
    }
    
    return result;
}

// ============================================================================
// 6. UPDATE_ELL_PROBIT (USANDO U_LATENT)
// ============================================================================

UpdateEllResult update_ell_probit(
    const std::vector<std::vector<int>>& ell_current,
    const std::vector<int>& z,
    const std::vector<std::vector<double>>& X_normalized,
    const std::vector<double>& alpha,
    const std::vector<std::vector<double>>& psi,
    const std::vector<std::vector<int>>& gamma,
    const std::vector<std::vector<double>>& u_latent,
    const std::vector<std::vector<double>>& ell_grid,
    int H,
    int n_grid
) {
    int p = ell_current[0].size();
    int n = X_normalized.size();
    
    UpdateEllResult result;
    result.H = H;
    result.p = p;
    result.ell = ell_current;
    
    std::uniform_int_distribution<> uniform_grid(0, n_grid - 1);
    
    for (int h = 0; h < H; ++h) {
        for (int j = 0; j < p; ++j) {
            if (gamma[h][j] == 0) {
                // Si variable inactiva, muestrear uniformemente
                result.ell[h][j] = uniform_grid(gen);
                continue;
            }
            
            // Encontrar observaciones donde z_i >= h
            std::vector<int> affected;
            for (int i = 0; i < n; ++i) {
                if (z[i] >= h) {
                    affected.push_back(i);
                }
            }
            
            if (affected.empty()) {
                result.ell[h][j] = uniform_grid(gen);
                continue;
            }
            
            // Pre-calcular eta base (sin contribución de j)
            std::vector<double> eta_base(affected.size());
            for (size_t idx = 0; idx < affected.size(); ++idx) {
                int i = affected[idx];
                eta_base[idx] = alpha[h];
                
                for (int jj = 0; jj < p; ++jj) {
                    if (jj != j && gamma[h][jj] == 1) {
                        double ell_value = ell_grid[jj][ell_current[h][jj]];
                        double dist = std::abs(X_normalized[i][jj] - ell_value);
                        eta_base[idx] -= psi[h][jj] * dist;
                    }
                }
            }
            
            // Evaluar cada posición del grid
            std::vector<double> log_likes(n_grid, 0.0);
            
            for (int m = 0; m < n_grid; ++m) {
                double ell_value = ell_grid[j][m];
                
                for (size_t idx = 0; idx < affected.size(); ++idx) {
                    int i = affected[idx];
                    double dist = std::abs(X_normalized[i][j] - ell_value);
                    double eta = eta_base[idx] - psi[h][j] * dist;
                    
                    // Log-likelihood usando u_{ih} ~ N(η, 1)
                    double u_val = u_latent[i][h];
                    double diff = u_val - eta;
                    log_likes[m] -= 0.5 * diff * diff;
                }
            }
            
            // Convertir a probabilidades
            double max_log = *std::max_element(log_likes.begin(), log_likes.end());
            std::vector<double> probs(n_grid);
            double sum_probs = 0.0;
            
            for (int m = 0; m < n_grid; ++m) {
                probs[m] = std::exp(log_likes[m] - max_log);
                sum_probs += probs[m];
            }
            
            for (int m = 0; m < n_grid; ++m) {
                probs[m] /= sum_probs;
            }
            
            result.ell[h][j] = sample_categorical(probs);
        }
    }
    
    return result;
}

// ============================================================================
// 7. UPDATE_PSI_GAMMA (MH + SPIKE-AND-SLAB)
// ============================================================================

PsiGammaResult update_psi_gamma(
    const std::vector<std::vector<double>>& psi_current,
    const std::vector<std::vector<int>>& gamma_current,
    const std::vector<double>& mu_psi,
    const std::vector<double>& tau_psi,
    const std::vector<double>& kappa,
    const std::vector<std::vector<double>>& u_latent,
    const std::vector<std::vector<double>>& X_normalized,
    const std::vector<double>& alpha,
    const std::vector<std::vector<int>>& ell,
    const std::vector<std::vector<double>>& ell_grid,
    const std::vector<int>& z,  //  AGREGAR
    double mh_scale_psi,
    bool psi_positive,
    int H
) {
    int p = psi_current[0].size();
    int n = X_normalized.size();
    
    PsiGammaResult result;
    result.H = H;
    result.p = p;
    result.psi = psi_current;
    result.gamma = gamma_current;
    result.acceptances_psi.resize(H, std::vector<bool>(p, false));
    
    std::normal_distribution<> proposal_dist(0.0, mh_scale_psi);
    
    for (int h = 0; h < H; ++h) {
        for (int j = 0; j < p; ++j) {
            // Paso 1: Actualizar γ_{hj}
            double log_prior_1 = std::log(kappa[j] + 1e-10);
            double log_prior_0 = std::log(1.0 - kappa[j] + 1e-10);
            
            double log_like_1 = 0.0;
            double log_like_0 = 0.0;
            
            //  CORRECCIÓN: Verificar z_i >= h
            for (int i = 0; i < n; ++i) {
                if (z[i] >= h) {  //  USAR z CORRECTAMENTE
                    double ell_value = ell_grid[j][ell[h][j]];
                    double dist = std::abs(X_normalized[i][j] - ell_value);
                    
                    // Calcular eta base (sin variable j)
                    double eta_base = alpha[h];
                    for (int jj = 0; jj < p; ++jj) {
                        if (jj != j && gamma_current[h][jj] == 1) {
                            double ell_jj = ell_grid[jj][ell[h][jj]];
                            double dist_jj = std::abs(X_normalized[i][jj] - ell_jj);
                            eta_base -= psi_current[h][jj] * dist_jj;
                        }
                    }
                    
                    // Con ψ actual (γ=1)
                    double eta_with = eta_base - psi_current[h][j] * dist;
                    log_like_1 -= 0.5 * (u_latent[i][h] - eta_with) * (u_latent[i][h] - eta_with);
                    
                    // Sin ψ (γ=0)
                    log_like_0 -= 0.5 * (u_latent[i][h] - eta_base) * (u_latent[i][h] - eta_base);
                }
            }
            
            double log_odds = (log_prior_1 + log_like_1) - (log_prior_0 + log_like_0);
            double p_gamma_1 = 1.0 / (1.0 + std::exp(-log_odds));
            result.gamma[h][j] = sample_bernoulli(p_gamma_1);
            
            // Paso 2: Actualizar ψ_{hj} si γ=1
            if (result.gamma[h][j] == 1) {
                double psi_curr = psi_current[h][j];
                double psi_prop = psi_curr + proposal_dist(gen);
                
                if (psi_positive && psi_prop < 0) {
                    result.acceptances_psi[h][j] = false;
                    continue;
                }
                
                double log_like_curr = 0.0;
                double log_like_prop = 0.0;
                
                //  CORRECCIÓN: Solo z_i >= h
                for (int i = 0; i < n; ++i) {
                    if (z[i] >= h) {  //  USAR z
                        double ell_value = ell_grid[j][ell[h][j]];
                        double dist = std::abs(X_normalized[i][j] - ell_value);
                        
                        // Calcular eta base
                        double eta_base = alpha[h];
                        for (int jj = 0; jj < p; ++jj) {
                            if (jj != j && result.gamma[h][jj] == 1) {
                                double ell_jj = ell_grid[jj][ell[h][jj]];
                                double dist_jj = std::abs(X_normalized[i][jj] - ell_jj);
                                eta_base -= psi_current[h][jj] * dist_jj;
                            }
                        }
                        
                        double eta_curr = eta_base - psi_curr * dist;
                        double eta_prop = eta_base - psi_prop * dist;
                        
                        log_like_curr -= 0.5 * (u_latent[i][h] - eta_curr) * (u_latent[i][h] - eta_curr);
                        log_like_prop -= 0.5 * (u_latent[i][h] - eta_prop) * (u_latent[i][h] - eta_prop);
                    }
                }
                
                double log_prior_curr = -0.5 * tau_psi[j] * (psi_curr - mu_psi[j]) * (psi_curr - mu_psi[j]);
                double log_prior_prop = -0.5 * tau_psi[j] * (psi_prop - mu_psi[j]) * (psi_prop - mu_psi[j]);
                
                double log_r = (log_like_prop + log_prior_prop) - (log_like_curr + log_prior_curr);
                
                bool accept = log_uniform() < log_r;
                if (accept) {
                    result.psi[h][j] = psi_prop;
                }
                result.acceptances_psi[h][j] = accept;
            } else {
                result.psi[h][j] = 0.0;
            }
        }
    }
    
    return result;
}

// ============================================================================
// 8. UPDATE_ALPHA_PROBIT (MH CON U_LATENT)
// ============================================================================

UpdateAlphaResult update_alpha_probit(
    const std::vector<double>& alpha_current,
    const std::vector<std::vector<double>>& u_latent,
    const std::vector<std::vector<double>>& X_normalized,
    const std::vector<std::vector<double>>& psi,
    const std::vector<std::vector<int>>& gamma,
    const std::vector<std::vector<int>>& ell,
    const std::vector<std::vector<double>>& ell_grid,
    const std::vector<int>& z,  // ✅ AGREGAR
    double mu_alpha,
    double mh_scale,
    int H
) {
    int n = X_normalized.size();
    
    UpdateAlphaResult result;
    result.H = H;
    result.alpha = alpha_current;
    result.acceptances.resize(H, false);
    
    std::normal_distribution<> proposal_dist(0.0, mh_scale);
    
    for (int h = 0; h < H; ++h) {
        double alpha_curr = alpha_current[h];
        double alpha_prop = alpha_curr + proposal_dist(gen);
        
        double log_like_curr = 0.0;
        double log_like_prop = 0.0;
        
        //  CORRECCIÓN: Solo z_i >= h
        for (int i = 0; i < n; ++i) {
            if (z[i] >= h) {  //  USAR z
                // Calcular offset de covariables
                double eta_offset = 0.0;
                for (int j = 0; j < static_cast<int>(X_normalized[i].size()); ++j) {
                    if (gamma[h][j] == 1) {
                        double ell_value = ell_grid[j][ell[h][j]];
                        double dist = std::abs(X_normalized[i][j] - ell_value);
                        eta_offset -= psi[h][j] * dist;
                    }
                }
                
                double eta_curr = alpha_curr + eta_offset;
                double eta_prop = alpha_prop + eta_offset;
                
                log_like_curr -= 0.5 * (u_latent[i][h] - eta_curr) * (u_latent[i][h] - eta_curr);
                log_like_prop -= 0.5 * (u_latent[i][h] - eta_prop) * (u_latent[i][h] - eta_prop);
            }
        }
        
        double log_prior_curr = -0.5 * (alpha_curr - mu_alpha) * (alpha_curr - mu_alpha);
        double log_prior_prop = -0.5 * (alpha_prop - mu_alpha) * (alpha_prop - mu_alpha);
        
        double log_r = (log_like_prop + log_prior_prop) - (log_like_curr + log_prior_curr);
        
        bool accept = log_uniform() < log_r;
        if (accept) {
            result.alpha[h] = alpha_prop;
        }
        result.acceptances[h] = accept;
    }
    
    return result;
}

// ============================================================================
// 9. UPDATE_GAMMA_PSI_GIBBS (TRUNCADO NORMAL POSTERIOR)
// ============================================================================

// ============================================================================
// 9. UPDATE_GAMMA_PSI_GIBBS (TRUNCATED NORMAL POSTERIOR)
// ============================================================================

PsiGammaResult update_gamma_psi_gibbs(
    const std::vector<std::vector<double>>& psi_current,
    const std::vector<std::vector<int>>& gamma_current,
    const std::vector<std::vector<double>>& u_latent,
    const std::vector<std::vector<double>>& X_normalized,
    const std::vector<double>& alpha,
    const std::vector<std::vector<int>>& ell,
    const std::vector<std::vector<double>>& ell_grid,
    const std::vector<int>& z,  // ✅ AGREGAR z
    const std::vector<double>& mu_psi,
    const std::vector<double>& tau_psi,
    const std::vector<double>& kappa,
    bool psi_positive,
    int H
) {
    int p = psi_current[0].size();
    int n = X_normalized.size();
    
    PsiGammaResult result;
    result.H = H;
    result.p = p;
    result.psi = psi_current;
    result.gamma = gamma_current;
    result.acceptances_psi.resize(H, std::vector<bool>(p, false));
    
    for (int h = 0; h < H; ++h) {
        for (int j = 0; j < p; ++j) {
            // ============================================================
            // PASO 1: ACTUALIZAR γ_{hj} (GIBBS)
            // ============================================================
            
            double log_prior_1 = std::log(kappa[j] + 1e-10);
            double log_prior_0 = std::log(1.0 - kappa[j] + 1e-10);
            
            // Calcular log-likelihood para γ=1 vs γ=0
            double log_like_1 = 0.0;
            double log_like_0 = 0.0;
            
            // Solo observaciones donde z_i >= h
            for (int i = 0; i < n; ++i) {
                if (z[i] >= h) {
                    double ell_value = ell_grid[j][ell[h][j]];
                    double dist = std::abs(X_normalized[i][j] - ell_value);
                    
                    // Calcular η base (sin variable j)
                    double eta_base = alpha[h];
                    for (int jj = 0; jj < p; ++jj) {
                        if (jj != j && gamma_current[h][jj] == 1) {
                            double ell_jj = ell_grid[jj][ell[h][jj]];
                            double dist_jj = std::abs(X_normalized[i][jj] - ell_jj);
                            eta_base -= psi_current[h][jj] * dist_jj;
                        }
                    }
                    
                    // Con ψ (γ=1)
                    double eta_with = eta_base - psi_current[h][j] * dist;
                    log_like_1 -= 0.5 * (u_latent[i][h] - eta_with) * (u_latent[i][h] - eta_with);
                    
                    // Sin ψ (γ=0)
                    log_like_0 -= 0.5 * (u_latent[i][h] - eta_base) * (u_latent[i][h] - eta_base);
                }
            }
            
            // Prior de ψ cuando γ=1
            if (psi_current[h][j] != 0.0 || gamma_current[h][j] == 1) {
                log_like_1 -= 0.5 * tau_psi[j] * std::pow(psi_current[h][j] - mu_psi[j], 2);
            }
            
            // Probabilidad posterior de γ=1
            double log_odds = (log_prior_1 + log_like_1) - (log_prior_0 + log_like_0);
            double p_gamma_1 = 1.0 / (1.0 + std::exp(-log_odds));
            
            // Samplear γ_{hj}
            result.gamma[h][j] = sample_bernoulli(p_gamma_1);
            
            if (result.gamma[h][j] == 0) {
                result.psi[h][j] = 0.0;
                result.acceptances_psi[h][j] = false;
                continue;
            }
            
            // ============================================================
            // PASO 2: ACTUALIZAR ψ_{hj} USANDO GIBBS (TRUNCATED NORMAL)
            // ============================================================
            
            // Encontrar observaciones afectadas
            std::vector<int> affected_idx;
            for (int i = 0; i < n; ++i) {
                if (z[i] >= h) {
                    affected_idx.push_back(i);
                }
            }
            
            if (affected_idx.empty()) {
                // Sin datos, samplear desde prior
                if (psi_positive) {
                    double sigma_psi = std::sqrt(1.0 / tau_psi[j]);
                    result.psi[h][j] = sample_truncated_normal(
                        mu_psi[j], sigma_psi, 0.0, 
                        std::numeric_limits<double>::infinity()
                    );
                } else {
                    double sigma_psi = std::sqrt(1.0 / tau_psi[j]);
                    result.psi[h][j] = sample_normal(mu_psi[j], sigma_psi);
                }
                result.acceptances_psi[h][j] = true;
                continue;
            }
            
            // Pre-calcular términos
            double sum_dist = 0.0;
            double sum_dist_sq = 0.0;
            double sum_u_residual_times_dist = 0.0;
            
            for (int idx : affected_idx) {
                int i = idx;
                
                // Calcular distancia
                double ell_value = ell_grid[j][ell[h][j]];
                double dist = std::abs(X_normalized[i][j] - ell_value);
                
                // Calcular residuo (sin contribución de ψ_hj)
                double u_residual = u_latent[i][h] - alpha[h];
                for (int jj = 0; jj < p; ++jj) {
                    if (jj != j && result.gamma[h][jj] == 1) {
                        double ell_jj = ell_grid[jj][ell[h][jj]];
                        double dist_jj = std::abs(X_normalized[i][jj] - ell_jj);
                        u_residual += psi_current[h][jj] * dist_jj;
                    }
                }
                
                sum_dist += dist;
                sum_dist_sq += dist * dist;
                sum_u_residual_times_dist += u_residual * dist;
            }
            
            // Posterior Normal: N(μ_post, σ²_post)
            // Derivado de:
            // u_{ih} - [α_h + otros términos] = -ψ_{hj} · dist_i + ε_i
            // donde ε_i ~ N(0, 1)
            
            double tau_post = tau_psi[j] + sum_dist_sq;  // Precisión posterior
            double mu_post = (tau_psi[j] * mu_psi[j] + sum_u_residual_times_dist) / tau_post;
            double sigma_post = std::sqrt(1.0 / tau_post);
            
            // Samplear desde posterior (truncado si psi_positive)
            if (psi_positive) {
                result.psi[h][j] = sample_truncated_normal(
                    mu_post, sigma_post, 0.0, 
                    std::numeric_limits<double>::infinity()
                );
                // Asegurar que sea positivo
                result.psi[h][j] = std::max(result.psi[h][j], 1e-6);
            } else {
                result.psi[h][j] = sample_normal(mu_post, sigma_post);
            }
            
            result.acceptances_psi[h][j] = true;
        }
    }
    
    return result;
}

} // namespace psbp