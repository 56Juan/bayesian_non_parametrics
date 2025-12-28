#include "lsbp_core.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <stdexcept>

LSBPCore::LSBPCore(unsigned int seed) 
    : rng_(seed), normal_dist_(0.0, 1.0), uniform_dist_(0.0, 1.0) {}

// ============================================================================
// SLICE SAMPLING
// ============================================================================
std::vector<double> LSBPCore::sample_slice_variables(
    const std::vector<int>& z,
    const std::vector<double>& w,
    int n
) {
    std::vector<double> u(n);
    
    for (int i = 0; i < n; ++i) {
        int zi = z[i];
        if (zi < 0 || zi >= static_cast<int>(w.size())) {
            throw std::runtime_error("Invalid cluster assignment");
        }
        u[i] = uniform_dist_(rng_) * w[zi];
    }
    
    return u;
}

// ============================================================================
// UPDATE ASSIGNMENTS
// ============================================================================
std::vector<int> LSBPCore::update_assignments(
    const MatrixXd& y_normalized,
    const MatrixXd& design_matrix,
    const MatrixXd& lambda_h,
    const MatrixXd& xi_h,
    const std::vector<double>& w,
    const std::vector<double>& u,
    int n,
    int H,
    int K
) {
    std::vector<int> z_new(n);
    
    for (int i = 0; i < n; ++i) {
        double y_val = y_normalized(i, 0);
        VectorXd d_row = design_matrix.row(i);
        
        // Encontrar candidatos: clusters con w_h > u_i
        std::vector<int> candidates;
        for (int h = 0; h < H; ++h) {
            if (w[h] > u[i]) {
                candidates.push_back(h);
            }
        }
        
        // Si no hay candidatos, usar el primero
        if (candidates.empty()) {
            candidates.push_back(0);
        }
        
        // Calcular likelihoods
        std::vector<double> likes(candidates.size());
        for (size_t idx = 0; idx < candidates.size(); ++idx) {
            int h = candidates[idx];
            
            // μ_h(x_i) = λ_h' d(x_i)
            double mu_h = lambda_h.row(h).dot(d_row);
            
            // log(σ²_h(x_i)) = ξ_h' d(x_i)
            double log_sigma2_h = xi_h.row(h).dot(d_row);
            double sigma2_h = std::exp(log_sigma2_h);
            sigma2_h = std::max(1e-6, std::min(sigma2_h, 1e6));
            
            // Likelihood
            double sigma_h = std::sqrt(sigma2_h);
            likes[idx] = std::exp(log_normal_pdf(y_val, mu_h, sigma_h));
            likes[idx] = std::max(likes[idx], 1e-300);
        }
        
        // Normalizar probabilidades
        double sum_likes = std::accumulate(likes.begin(), likes.end(), 0.0);
        std::vector<double> probs(likes.size());
        for (size_t idx = 0; idx < likes.size(); ++idx) {
            probs[idx] = likes[idx] / sum_likes;
        }
        
        // Muestrear categórica
        double u_sample = uniform_dist_(rng_);
        double cumsum = 0.0;
        size_t selected = 0;
        for (size_t idx = 0; idx < probs.size(); ++idx) {
            cumsum += probs[idx];
            if (u_sample <= cumsum) {
                selected = idx;
                break;
            }
        }
        
        z_new[i] = candidates[selected];
    }
    
    return z_new;
}

// ============================================================================
// UPDATE LAMBDA (Weighted Least Squares)
// ============================================================================
MatrixXd LSBPCore::update_lambda(
    const MatrixXd& y_normalized,
    const MatrixXd& design_matrix,
    const std::vector<int>& z,
    const MatrixXd& xi_h,
    const VectorXd& mu_lambda,
    const MatrixXd& Sigma_lambda,
    int n,
    int H,
    int K
) {
    MatrixXd lambda_new = MatrixXd::Zero(H, K);
    
    MatrixXd Sigma_inv = Sigma_lambda.inverse();
    VectorXd mu_prior_term = Sigma_inv * mu_lambda;
    
    for (int h = 0; h < H; ++h) {
        // Encontrar miembros del cluster
        std::vector<int> members;
        for (int i = 0; i < n; ++i) {
            if (z[i] == h) {
                members.push_back(i);
            }
        }
        
        if (members.empty()) {
            // Prior (cluster vacío)
            lambda_new.row(h) = sample_multivariate_normal(mu_lambda, Sigma_lambda);
        } else {
            // Datos del cluster
            int n_h = members.size();
            MatrixXd y_h(n_h, 1);
            MatrixXd d_h(n_h, K);
            VectorXd prec_h(n_h);
            
            for (int idx = 0; idx < n_h; ++idx) {
                int i = members[idx];
                y_h(idx, 0) = y_normalized(i, 0);
                d_h.row(idx) = design_matrix.row(i);
                
                // Calcular precisión
                double log_sigma2_h = xi_h.row(h).dot(design_matrix.row(i));
                double sigma2_h = std::exp(log_sigma2_h);
                sigma2_h = std::max(1e-6, std::min(sigma2_h, 1e6));
                prec_h(idx) = 1.0 / sigma2_h;
            }
            
            // Weighted least squares
            MatrixXd W = prec_h.asDiagonal();
            MatrixXd Sigma_post_inv = Sigma_inv + d_h.transpose() * W * d_h;
            MatrixXd Sigma_post = Sigma_post_inv.inverse();
            
            VectorXd mu_post = Sigma_post * (mu_prior_term + d_h.transpose() * W * y_h);
            
            lambda_new.row(h) = sample_multivariate_normal(mu_post, Sigma_post);
        }
    }
    
    return lambda_new;
}

// ============================================================================
// UPDATE XI (Metropolis-Hastings)
// ============================================================================
MatrixXd LSBPCore::update_xi(
    const MatrixXd& y_normalized,
    const MatrixXd& design_matrix,
    const MatrixXd& lambda_h,
    const MatrixXd& xi_h,
    const std::vector<int>& z,
    const VectorXd& mu_xi,
    const MatrixXd& Sigma_xi,
    double scale,
    int n,
    int H,
    int K
) {
    MatrixXd xi_new = xi_h;
    MatrixXd proposal_cov = scale * scale * MatrixXd::Identity(K, K);
    
    // Calcular desviaciones estándar del prior
    VectorXd sigma_xi_diag(K);
    for (int k = 0; k < K; ++k) {
        sigma_xi_diag(k) = std::sqrt(Sigma_xi(k, k));
    }
    
    for (int h = 0; h < H; ++h) {
        // Encontrar miembros
        std::vector<int> members;
        for (int i = 0; i < n; ++i) {
            if (z[i] == h) {
                members.push_back(i);
            }
        }
        
        if (members.empty()) {
            // Prior (cluster vacío)
            xi_new.row(h) = sample_multivariate_normal(mu_xi, Sigma_xi);
            continue;
        }
        
        // Propuesta
        VectorXd xi_curr = xi_h.row(h);
        VectorXd xi_prop = sample_multivariate_normal(xi_curr, proposal_cov);
        
        // Log-likelihood actual
        double log_like_curr = 0.0;
        for (int i : members) {
            VectorXd d_row = design_matrix.row(i);
            double y_val = y_normalized(i, 0);
            double mu_h = lambda_h.row(h).dot(d_row);
            
            double log_sigma2_curr = xi_curr.dot(d_row);
            double sigma2_curr = std::exp(log_sigma2_curr);
            sigma2_curr = std::max(1e-6, std::min(sigma2_curr, 1e6));
            
            log_like_curr += log_normal_pdf(y_val, mu_h, std::sqrt(sigma2_curr));
        }
        
        // Log-likelihood propuesta
        double log_like_prop = 0.0;
        for (int i : members) {
            VectorXd d_row = design_matrix.row(i);
            double y_val = y_normalized(i, 0);
            double mu_h = lambda_h.row(h).dot(d_row);
            
            double log_sigma2_prop = xi_prop.dot(d_row);
            double sigma2_prop = std::exp(log_sigma2_prop);
            sigma2_prop = std::max(1e-6, std::min(sigma2_prop, 1e6));
            
            log_like_prop += log_normal_pdf(y_val, mu_h, std::sqrt(sigma2_prop));
        }
        
        // Log-prior actual
        double log_prior_curr = 0.0;
        for (int k = 0; k < K; ++k) {
            log_prior_curr += log_normal_pdf(xi_curr(k), mu_xi(k), sigma_xi_diag(k));
        }
        
        // Log-prior propuesta
        double log_prior_prop = 0.0;
        for (int k = 0; k < K; ++k) {
            log_prior_prop += log_normal_pdf(xi_prop(k), mu_xi(k), sigma_xi_diag(k));
        }
        
        // Ratio de aceptación
        double log_alpha = (log_like_prop + log_prior_prop) - 
                          (log_like_curr + log_prior_curr);
        
        // Aceptar/rechazar
        if (std::log(uniform_dist_(rng_)) < log_alpha) {
            xi_new.row(h) = xi_prop;
        }
    }
    
    return xi_new;
}

// ============================================================================
// FUNCIONES AUXILIARES
// ============================================================================
double LSBPCore::log_normal_pdf(double x, double mean, double sd) {
    const double log_sqrt_2pi = 0.91893853320467267;
    double z = (x - mean) / sd;
    return -log_sqrt_2pi - std::log(sd) - 0.5 * z * z;
}

VectorXd LSBPCore::sample_multivariate_normal(
    const VectorXd& mean,
    const MatrixXd& cov
) {
    int K = mean.size();
    
    // Cholesky decomposition
    LLT<MatrixXd> llt(cov);
    if (llt.info() != Success) {
        // Fallback: add small diagonal
        MatrixXd cov_reg = cov + 1e-6 * MatrixXd::Identity(K, K);
        llt.compute(cov_reg);
    }
    MatrixXd L = llt.matrixL();
    
    // Sample standard normals
    VectorXd z(K);
    for (int k = 0; k < K; ++k) {
        z(k) = normal_dist_(rng_);
    }
    
    // Transform: mean + L * z
    return mean + L * z;
}