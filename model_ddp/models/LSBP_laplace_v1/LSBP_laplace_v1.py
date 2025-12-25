import numpy as np
from scipy.stats import norm, gamma, truncnorm, laplace, invgauss
from scipy.special import expit
import math

# Importar el módulo C++
try:
    from . import lsbp_laplace_cpp
    CPP_AVAILABLE = True
    print("Implementacion en C++ Exitosa")
except ImportError as e:
    CPP_AVAILABLE = False
    _IMPORT_ERROR = e
    print("Implementacion en C++ Fallida")

class LSBPLaplace:
    """
    Logit Stick-Breaking Process (LSBP) con kernel Laplace - OPTIMIZADO CON C++
    
    Funciones migradas a C++:
    - _compute_eta
    - _compute_weights
    - _update_lambda_latent
    - update_assignments
    - update_atoms
    - update_alpha
    - update_psi
    - update_ell
    """
    
    def __init__(self, y, X, H=20,
                 mu_prior=(0.0, 1.0),
                 mu0_prior=(0.0, 100.0),
                 tau0_prior=(2.0, 1.0),
                 a0_prior=(3.0, 1.0),
                 beta0_prior=(2.0, 1.0),
                 psi_prior=(0.0, 1.0),
                 n_grid=50,
                 verbose=True,
                 use_cpp=True):
        
        self.y = np.array(y)
        self.X = np.array(X)
        self.n, self.p = self.X.shape
        self.H = H
        self.verbose = verbose
        self.use_cpp = use_cpp and CPP_AVAILABLE
        
        if self.use_cpp:
            print("Using C++ acceleration for 8 functions (compute_eta, compute_weights, "
                  "update_lambda_latent, update_assignments, update_atoms, update_alpha, "
                  "update_psi, update_ell)")
        
        # Normalizar datos
        self.y_mean = np.mean(y)
        self.y_std = np.std(y)
        self.y_normalized = (self.y - self.y_mean) / self.y_std
        
        self.X_mean = np.mean(X, axis=0)
        self.X_std = np.std(X, axis=0)
        self.X_normalized = (self.X - self.X_mean) / self.X_std
        
        # Hiperpriors
        self.mu_mu, self.tau_mu_inv = mu_prior
        self.m0, self.s02 = mu0_prior
        self.alpha_tau, self.beta_tau = tau0_prior
        self.alpha_a, self.beta_a = a0_prior
        self.alpha_beta, self.beta_beta = beta0_prior
        self.mu_psi, self.tau_psi_inv = psi_prior
        
        # Grilla
        self.n_grid = n_grid
        self.ell_grid = self._create_grid()
        
        # MH scales
        self.mh_scales = {
            'alpha': 0.3,
            'psi': 0.2,
            'tau0': 0.2,
            'a0': 0.2
        }
        self.mh_acceptance = {
            'alpha': [],
            'psi': [],
            'tau0': [],
            'a0': []
        }
        
        # Storage
        self.trace = {
            'mu': [], 'mu0': [], 'tau0': [], 'a0': [], 'beta0': [],
            'z': [], 'theta_mu': [], 'theta_b': [], 'w': [],
            'n_clusters': [], 'alpha': [], 'psi': [], 'ell': []
        }
        
        self.initialize()
    
    def _create_grid(self):
        grid = np.zeros((self.p, self.n_grid))
        for j in range(self.p):
            x_min = self.X_normalized[:, j].min() - 0.5
            x_max = self.X_normalized[:, j].max() + 0.5
            grid[j, :] = np.linspace(x_min, x_max, self.n_grid)
        return grid
    
    def initialize(self):
        self.mu = np.random.normal(self.mu_mu, np.sqrt(self.tau_mu_inv))
        
        self.mu0 = np.random.normal(0, 1)
        self.tau0 = np.random.gamma(self.alpha_tau, 1.0/self.beta_tau)
        self.tau0 = np.clip(self.tau0, 0.1, 10.0)
        
        self.a0 = np.random.gamma(self.alpha_a, 1.0/self.beta_a)
        self.a0 = np.clip(self.a0, 1.0, 10.0)
        
        self.beta0 = np.random.gamma(self.alpha_beta, 1.0/self.beta_beta)
        self.beta0 = np.clip(self.beta0, 0.1, 10.0)
        
        self.alpha = np.random.normal(self.mu, 1.0, size=self.H)
        
        self.psi = np.zeros((self.H, self.p))
        for h in range(self.H):
            for j in range(self.p):
                a = (0 - self.mu_psi) / np.sqrt(self.tau_psi_inv)
                self.psi[h, j] = truncnorm.rvs(a, np.inf,
                                                loc=self.mu_psi,
                                                scale=np.sqrt(self.tau_psi_inv))
        
        self.ell = np.zeros((self.H, self.p), dtype=int)
        for h in range(self.H):
            for j in range(self.p):
                self.ell[h, j] = np.random.randint(0, self.n_grid)
        
        self.w = self._compute_weights()
        
        self.theta_mu = np.zeros(self.H)
        self.theta_b = np.zeros(self.H)
        for h in range(self.H):
            self.theta_b[h] = np.random.gamma(self.a0, 1.0/self.beta0)
            self.theta_b[h] = np.clip(self.theta_b[h], 0.01, 100.0)
            self.theta_mu[h] = np.random.normal(self.mu0, 1.0/np.sqrt(self.tau0))
        
        self.lambda_latent = np.ones((self.n, self.H))
        
        self.z = np.zeros(self.n, dtype=int)
        for i in range(self.n):
            likes = np.array([
                laplace.pdf(self.y_normalized[i], self.theta_mu[h], self.theta_b[h])
                for h in range(self.H)
            ])
            likes = np.clip(likes, 1e-300, None)
            probs = self.w[i, :] * likes
            probs /= probs.sum()
            self.z[i] = np.random.choice(self.H, p=probs)
        
        self._update_lambda_latent()
    
    def _compute_eta(self, X_batch):
        """Calcula η_h(x) - C++ o Python"""
        if self.use_cpp:
            result = lsbp_laplace_cpp.compute_eta(
                X_batch,
                self.alpha,
                self.psi,
                self.ell,
                self.ell_grid
            )
            return np.array(result.eta)
        else:
            n_batch = X_batch.shape[0]
            eta = np.zeros((n_batch, self.H))
            
            for h in range(self.H):
                eta[:, h] = self.alpha[h]
                for j in range(self.p):
                    ell_hj_value = self.ell_grid[j, self.ell[h, j]]
                    dist = np.abs(X_batch[:, j] - ell_hj_value)
                    eta[:, h] -= self.psi[h, j] * dist
            
            return eta
    
    def _compute_weights(self):
        """Calcula pesos w_h(x) - C++ o Python"""
        if self.use_cpp:
            result = lsbp_laplace_cpp.compute_weights(
                self.X_normalized,
                self.alpha,
                self.psi,
                self.ell,
                self.ell_grid
            )
            return np.array(result.weights)
        else:
            eta = self._compute_eta(self.X_normalized)
            v = expit(eta)
            
            w = np.zeros((self.n, self.H))
            for i in range(self.n):
                remaining = 1.0
                for h in range(self.H):
                    w[i, h] = v[i, h] * remaining
                    remaining *= (1 - v[i, h])
            
            w = w / w.sum(axis=1, keepdims=True)
            return w
    
    def _update_lambda_latent(self):
        """Actualizar λ_ih - C++ o Python"""
        if self.use_cpp:
            result = lsbp_laplace_cpp.update_lambda_latent(
                self.z,
                self.y_normalized,
                self.theta_mu,
                self.theta_b,
                self.lambda_latent,
                self.H
            )
            self.lambda_latent = np.array(result.lambda_latent)
        else:
            for i in range(self.n):
                h = self.z[i]
                residual_abs = np.abs(self.y_normalized[i] - self.theta_mu[h])
                residual_abs = np.clip(residual_abs, 1e-6, None)
                
                mu_ig = self.theta_b[h] / residual_abs
                lambda_ig = self.theta_b[h]**2
                
                self.lambda_latent[i, h] = invgauss.rvs(mu_ig / lambda_ig, scale=lambda_ig)
                self.lambda_latent[i, h] = np.clip(self.lambda_latent[i, h], 0.001, 100.0)
    
    def sample_slice_variables(self):
        u = np.zeros(self.n)
        for i in range(self.n):
            u[i] = np.random.uniform(0, self.w[i, self.z[i]])
        
        u_min = u.min()
        while self.H < 100:
            w_min = self.w.min(axis=0)
            if np.all(w_min < u_min):
                break
            
            H_new = self.H + 5
            
            alpha_new = np.random.normal(self.mu, 1.0, size=5)
            self.alpha = np.append(self.alpha, alpha_new)
            
            psi_new = np.zeros((5, self.p))
            for h in range(5):
                for j in range(self.p):
                    a = (0 - self.mu_psi) / np.sqrt(self.tau_psi_inv)
                    psi_new[h, j] = truncnorm.rvs(a, np.inf,
                                                   loc=self.mu_psi,
                                                   scale=np.sqrt(self.tau_psi_inv))
            self.psi = np.vstack([self.psi, psi_new])
            
            ell_new = np.random.randint(0, self.n_grid, size=(5, self.p))
            self.ell = np.vstack([self.ell, ell_new])
            
            theta_mu_new = np.zeros(5)
            theta_b_new = np.zeros(5)
            for h in range(5):
                theta_b_new[h] = np.random.gamma(self.a0, 1.0/self.beta0)
                theta_b_new[h] = np.clip(theta_b_new[h], 0.01, 100.0)
                theta_mu_new[h] = np.random.normal(self.mu0, 1.0/np.sqrt(self.tau0))
            
            self.theta_mu = np.append(self.theta_mu, theta_mu_new)
            self.theta_b = np.append(self.theta_b, theta_b_new)
            
            lambda_new = np.ones((self.n, 5))
            self.lambda_latent = np.hstack([self.lambda_latent, lambda_new])
            
            self.H = H_new
            self.w = self._compute_weights()
            break
        
        return u
    
    def update_assignments(self, u):
        """Actualizar z_i - C++ o Python"""
        if self.use_cpp:
            self.z = np.array(lsbp_laplace_cpp.update_assignments(
                u,
                self.w,
                self.y_normalized,
                self.theta_mu,
                self.theta_b,
                self.z
            ))
        else:
            for i in range(self.n):
                candidates = np.where(self.w[i, :] > u[i])[0]
                
                if len(candidates) == 0:
                    candidates = np.array([0])
                
                likes = laplace.pdf(self.y_normalized[i],
                                  self.theta_mu[candidates],
                                  self.theta_b[candidates])
                likes = np.clip(likes, 1e-300, None)
                probs = likes / likes.sum() if likes.sum() > 0 else np.ones_like(likes) / len(likes)
                
                self.z[i] = np.random.choice(candidates, p=probs)
    
    def update_atoms(self):
        """Actualizar θ_h - C++ o Python"""
        if self.use_cpp:
            result = lsbp_laplace_cpp.update_atoms(
                self.z,
                self.y_normalized,
                self.lambda_latent,
                self.theta_mu,
                self.theta_b,
                self.mu0,
                self.tau0,
                self.a0,
                self.beta0,
                self.H
            )
            self.theta_mu = np.array(result.theta_mu)
            self.theta_b = np.array(result.theta_b)
        else:
            for h in range(self.H):
                members_idx = np.where(self.z == h)[0]
                n_h = len(members_idx)
                
                if n_h > 0:
                    y_h = self.y_normalized[members_idx]
                    lambda_h = self.lambda_latent[members_idx, h]
                    
                    tau_post = self.tau0 + np.sum(1.0 / lambda_h)
                    mu_post = (self.tau0 * self.mu0 + np.sum(y_h / lambda_h)) / tau_post
                    
                    self.theta_mu[h] = np.random.normal(mu_post, 1.0/np.sqrt(tau_post))
                    
                    a_post = self.a0 + n_h
                    beta_post = self.beta0 + np.sum(1.0 / lambda_h)
                    
                    self.theta_b[h] = np.random.gamma(a_post, 1.0/beta_post)
                    self.theta_b[h] = np.clip(self.theta_b[h], 0.01, 100.0)
                else:
                    self.theta_b[h] = np.random.gamma(self.a0, 1.0/self.beta0)
                    self.theta_b[h] = np.clip(self.theta_b[h], 0.01, 100.0)
                    self.theta_mu[h] = np.random.normal(self.mu0, 1.0/np.sqrt(self.tau0))
        
        self._update_lambda_latent()
    
    def update_alpha(self):
        """Actualizar α_h - C++ o Python"""
        if self.use_cpp:
            result = lsbp_laplace_cpp.update_alpha(
                self.alpha,
                self.z,
                self.X_normalized,
                self.psi,
                self.ell,
                self.ell_grid,
                self.mu,
                self.mh_scales['alpha']
            )
            self.alpha = np.array(result.alpha)
            self.mh_acceptance['alpha'].extend(result.acceptance)
        else:
            for h in range(self.H - 1):
                alpha_prop = np.random.normal(self.alpha[h], self.mh_scales['alpha'])
                
                affected = np.where(self.z >= h)[0]
                
                if len(affected) == 0:
                    log_prior_curr = -0.5 * ((self.alpha[h] - self.mu)**2)
                    log_prior_prop = -0.5 * ((alpha_prop - self.mu)**2)
                    log_r = log_prior_prop - log_prior_curr
                else:
                    eta_curr = self._compute_eta_h(self.X_normalized[affected], h, self.alpha[h])
                    eta_prop = self._compute_eta_h(self.X_normalized[affected], h, alpha_prop)
                    
                    v_curr = expit(eta_curr)
                    v_prop = expit(eta_prop)
                    
                    log_like_curr = 0.0
                    log_like_prop = 0.0
                    
                    for idx_local, idx_global in enumerate(affected):
                        if self.z[idx_global] == h:
                            log_like_curr += np.log(np.clip(v_curr[idx_local], 1e-10, 1.0))
                            log_like_prop += np.log(np.clip(v_prop[idx_local], 1e-10, 1.0))
                        else:
                            log_like_curr += np.log(np.clip(1 - v_curr[idx_local], 1e-10, 1.0))
                            log_like_prop += np.log(np.clip(1 - v_prop[idx_local], 1e-10, 1.0))
                    
                    log_prior_curr = -0.5 * ((self.alpha[h] - self.mu)**2)
                    log_prior_prop = -0.5 * ((alpha_prop - self.mu)**2)
                    
                    log_r = (log_like_prop + log_prior_prop) - (log_like_curr + log_prior_curr)
                
                log_r = np.clip(log_r, -50, 50)
                
                accept = math.log(np.random.rand()) < log_r
                if accept:
                    self.alpha[h] = alpha_prop
                
                self.mh_acceptance['alpha'].append(float(accept))
    
    def _compute_eta_h(self, X_batch, h, alpha_h_value):
        """Calcula η_h(x) para cluster h con α_h dado"""
        n_batch = X_batch.shape[0]
        eta = np.full(n_batch, alpha_h_value)
        
        for j in range(self.p):
            ell_hj_value = self.ell_grid[j, self.ell[h, j]]
            dist = np.abs(X_batch[:, j] - ell_hj_value)
            eta -= self.psi[h, j] * dist
        
        return eta
    
    def update_psi(self):
        """Actualizar ψ_{hj} - C++ o Python"""
        if self.use_cpp:
            result = lsbp_laplace_cpp.update_psi(
                self.psi,
                self.z,
                self.alpha,
                self.X_normalized,
                self.ell,
                self.ell_grid,
                self.mu_psi,
                self.tau_psi_inv,
                self.mh_scales['psi']
            )
            self.psi = np.array(result.psi)
            self.mh_acceptance['psi'].extend(result.acceptance)
        else:
            for h in range(self.H - 1):
                for j in range(self.p):
                    psi_curr = self.psi[h, j]
                    psi_prop = np.random.normal(psi_curr, self.mh_scales['psi'])
                    
                    if psi_prop < 0:
                        continue
                    
                    affected = np.where(self.z >= h)[0]
                    
                    if len(affected) == 0:
                        log_prior_curr = -0.5 * ((psi_curr - self.mu_psi)**2) / self.tau_psi_inv
                        log_prior_prop = -0.5 * ((psi_prop - self.mu_psi)**2) / self.tau_psi_inv
                        log_r = log_prior_prop - log_prior_curr
                    else:
                        ell_hj_value = self.ell_grid[j, self.ell[h, j]]
                        dist = np.abs(self.X_normalized[affected, j] - ell_hj_value)
                        
                        eta_curr = self.alpha[h] - np.sum(
                            [self.psi[h, jj] * np.abs(
                                self.X_normalized[affected, jj] -
                                self.ell_grid[jj, self.ell[h, jj]]
                            ) for jj in range(self.p)], axis=0
                        )
                        
                        delta_eta = (psi_curr - psi_prop) * dist
                        eta_prop = eta_curr + delta_eta
                        
                        v_curr = expit(eta_curr)
                        v_prop = expit(eta_prop)
                        
                        log_like_curr = 0.0
                        log_like_prop = 0.0
                        for idx_local, idx in enumerate(affected):
                            if self.z[idx] == h:
                                log_like_curr += np.log(np.clip(v_curr[idx_local], 1e-10, 1.0))
                                log_like_prop += np.log(np.clip(v_prop[idx_local], 1e-10, 1.0))
                            else:
                                log_like_curr += np.log(np.clip(1 - v_curr[idx_local], 1e-10, 1.0))
                                log_like_prop += np.log(np.clip(1 - v_prop[idx_local], 1e-10, 1.0))
                        
                        log_prior_curr = -0.5 * ((psi_curr - self.mu_psi)**2) / self.tau_psi_inv
                        log_prior_prop = -0.5 * ((psi_prop - self.mu_psi)**2) / self.tau_psi_inv
                        
                        log_r = (log_like_prop + log_prior_prop) - (log_like_curr + log_prior_curr)
                    
                    log_r = np.clip(log_r, -50, 50)
                    
                    accept = math.log(np.random.rand()) < log_r
                    if accept:
                        self.psi[h, j] = psi_prop
                    
                    self.mh_acceptance['psi'].append(float(accept))
    
    def update_ell(self):
        """Actualizar ℓ_{hj} - C++ o Python"""
        if self.use_cpp:
            result = lsbp_laplace_cpp.update_ell(
                self.ell,
                self.z,
                self.alpha,
                self.psi,
                self.X_normalized,
                self.ell_grid,
                self.n_grid
            )
            self.ell = np.array(result.ell)
        else:
            for h in range(self.H - 1):
                for j in range(self.p):
                    affected = np.where(self.z >= h)[0]
                    
                    if len(affected) == 0:
                        self.ell[h, j] = np.random.randint(0, self.n_grid)
                        continue
                    
                    log_likes = np.zeros(self.n_grid)
                    
                    for m in range(self.n_grid):
                        ell_value = self.ell_grid[j, m]
                        dist = np.abs(self.X_normalized[affected, j] - ell_value)
                        
                        eta = self.alpha[h] - self.psi[h, j] * dist
                        for jj in range(self.p):
                            if jj != j:
                                ell_jj_value = self.ell_grid[jj, self.ell[h, jj]]
                                eta -= self.psi[h, jj] * np.abs(
                                    self.X_normalized[affected, jj] - ell_jj_value
                                )
                        
                        v = expit(eta)
                        
                        for idx_local, idx in enumerate(affected):
                            if self.z[idx] == h:
                                log_likes[m] += np.log(np.clip(v[idx_local], 1e-10, 1.0))
                            else:
                                log_likes[m] += np.log(np.clip(1 - v[idx_local], 1e-10, 1.0))
                    
                    log_likes = log_likes - np.max(log_likes)
                    probs = np.exp(log_likes)
                    probs /= probs.sum()
                    
                    self.ell[h, j] = np.random.choice(self.n_grid, p=probs)
    
    def update_weights(self):
        """Recalcular pesos"""
        self.w = self._compute_weights()
    
    def update_mu(self):
        """Actualizar μ (hiperparámetro de α_h)"""
        tau_post = len(self.alpha) + 1.0 / self.tau_mu_inv
        mu_post = (np.sum(self.alpha) + self.mu_mu / self.tau_mu_inv) / tau_post
        
        self.mu = np.random.normal(mu_post, 1.0 / np.sqrt(tau_post))
        self.mu = np.clip(self.mu, -10, 10)
    
    def update_mu0(self):
        """Actualizar μ₀"""
        active_clusters = np.unique(self.z)
        if len(active_clusters) == 0:
            return
        
        mu_active = self.theta_mu[active_clusters]
        
        precision_post = self.tau0 * len(active_clusters) + 1.0 / self.s02
        s0n2 = 1.0 / precision_post
        m0n = s0n2 * (self.tau0 * np.sum(mu_active) + self.m0 / self.s02)
        
        s0n2 = np.clip(s0n2, 1e-6, 1e6)
        m0n = np.clip(m0n, -100, 100)
        
        self.mu0 = np.random.normal(m0n, math.sqrt(s0n2))
        self.mu0 = np.clip(self.mu0, -50, 50)
    
    def update_tau0(self):
        """Actualizar τ₀ con MH"""
        log_tau = math.log(self.tau0)
        log_tau_prop = np.random.normal(log_tau, self.mh_scales['tau0'])
        tau_prop = math.exp(log_tau_prop)
        tau_prop = np.clip(tau_prop, 0.01, 100.0)
        
        active_clusters = np.unique(self.z)
        if len(active_clusters) == 0:
            return
        
        mu_active = self.theta_mu[active_clusters]
        diff_sq = (mu_active - self.mu0)**2
        
        K = len(active_clusters)
        log_like_curr = (0.5 * K * np.log(self.tau0) - 
                        0.5 * self.tau0 * np.sum(diff_sq))
        log_like_prop = (0.5 * K * np.log(tau_prop) - 
                        0.5 * tau_prop * np.sum(diff_sq))
        
        log_prior_curr = ((self.alpha_tau - 1) * math.log(self.tau0) - 
                         self.beta_tau * self.tau0)
        log_prior_prop = ((self.alpha_tau - 1) * math.log(tau_prop) - 
                         self.beta_tau * tau_prop)
        
        log_r = (log_like_prop + log_prior_prop) - (log_like_curr + log_prior_curr)
        log_r = np.clip(log_r, -50, 50)
        
        accept = math.log(np.random.rand()) < log_r
        if accept:
            self.tau0 = tau_prop
        
        self.mh_acceptance['tau0'].append(float(accept))
    
    def update_a0(self):
        """Actualizar a₀ con MH"""
        log_a = math.log(self.a0)
        log_a_prop = np.random.normal(log_a, self.mh_scales['a0'])
        a_prop = math.exp(log_a_prop)
        a_prop = np.clip(a_prop, 0.5, 20.0)
        
        active_clusters = np.unique(self.z)
        if len(active_clusters) == 0:
            return
        
        b_active = self.theta_b[active_clusters]
        
        K = len(active_clusters)
        log_like_curr = (self.a0 * K * np.log(self.beta0) - 
                        K * math.lgamma(self.a0) + 
                        (self.a0 - 1) * np.sum(np.log(b_active)) - 
                        self.beta0 * np.sum(b_active))
        log_like_prop = (a_prop * K * np.log(self.beta0) - 
                        K * math.lgamma(a_prop) + 
                        (a_prop - 1) * np.sum(np.log(b_active)) - 
                        self.beta0 * np.sum(b_active))
        
        log_prior_curr = (self.alpha_a - 1) * math.log(self.a0) - self.beta_a * self.a0
        log_prior_prop = (self.alpha_a - 1) * math.log(a_prop) - self.beta_a * a_prop
        
        log_r = (log_like_prop + log_prior_prop) - (log_like_curr + log_prior_curr)
        log_r = np.clip(log_r, -50, 50)
        
        accept = math.log(np.random.rand()) < log_r
        if accept:
            self.a0 = a_prop
        
        self.mh_acceptance['a0'].append(float(accept))
    
    def update_beta0(self):
        """Actualizar β₀ con Gibbs (conjugado)"""
        active_clusters = np.unique(self.z)
        if len(active_clusters) == 0:
            return
        
        b_active = self.theta_b[active_clusters]
        
        alpha_post = self.alpha_beta + len(active_clusters) * self.a0
        beta_post = self.beta_beta + np.sum(b_active)
        
        self.beta0 = np.random.gamma(alpha_post, 1.0 / beta_post)
        self.beta0 = np.clip(self.beta0, 0.01, 100.0)
    
    def adapt_mh_scales(self, iteration):
        """Adaptar escalas MH durante burn-in"""
        if iteration > 50 and iteration % 50 == 0:
            for param in ['alpha', 'psi', 'tau0', 'a0']:
                if len(self.mh_acceptance[param]) > 0:
                    recent = self.mh_acceptance[param][-50:]
                    acc_rate = np.mean(recent)
                    
                    if acc_rate < 0.15:
                        self.mh_scales[param] *= 0.8
                    elif acc_rate > 0.4:
                        self.mh_scales[param] *= 1.2
                    
                    self.mh_scales[param] = np.clip(self.mh_scales[param], 0.01, 1.0)
    
    def run(self, iterations=1000, burnin=500):
        """Ejecuta la cadena de Markov para LSBP-Laplace con Slice Sampling"""
        
        for it in range(iterations):
            u = self.sample_slice_variables()
            self.update_assignments(u)
            self.update_atoms()
            
            self.update_alpha()
            self.update_psi()
            self.update_ell()
            self.update_weights()
            
            self.update_mu()
            self.update_mu0()
            self.update_tau0()
            self.update_a0()
            self.update_beta0()
            
            if it < burnin:
                self.adapt_mh_scales(it)
            
            if it >= burnin:
                active_clusters = len(np.unique(self.z))
                
                mu_original = self.theta_mu * self.y_std + self.y_mean
                b_original = self.theta_b * self.y_std
                
                self.trace['mu'].append(self.mu)
                self.trace['mu0'].append(self.mu0 * self.y_std + self.y_mean)
                self.trace['tau0'].append(self.tau0 / (self.y_std ** 2))
                self.trace['a0'].append(self.a0)
                self.trace['beta0'].append(self.beta0 * self.y_std)
                self.trace['z'].append(self.z.copy())
                self.trace['theta_mu'].append(mu_original.copy())
                self.trace['theta_b'].append(b_original.copy())
                self.trace['w'].append(self.w.copy())
                self.trace['n_clusters'].append(active_clusters)
                self.trace['alpha'].append(self.alpha.copy())
                self.trace['psi'].append(self.psi.copy())
                self.trace['ell'].append(self.ell.copy())
            
            if self.verbose and (it + 1) % 100 == 0:
                active = len(np.unique(self.z))
                acc_alpha = np.mean(self.mh_acceptance['alpha'][-100:]) if len(self.mh_acceptance['alpha']) >= 100 else 0
                acc_psi = np.mean(self.mh_acceptance['psi'][-100:]) if len(self.mh_acceptance['psi']) >= 100 else 0
                acc_tau = np.mean(self.mh_acceptance['tau0'][-100:]) if len(self.mh_acceptance['tau0']) >= 100 else 0
                acc_a = np.mean(self.mh_acceptance['a0'][-100:]) if len(self.mh_acceptance['a0']) >= 100 else 0
                
                print(f"Iter {it+1}/{iterations}: K_eff={active}, H={self.H}, "
                      f"μ={self.mu:.2f}, μ₀={self.mu0:.2f}, τ₀={self.tau0:.2f}, "
                      f"a₀={self.a0:.2f}, β₀={self.beta0:.2f}")
                print(f"  Acceptance: α={acc_alpha:.2f}, ψ={acc_psi:.2f}, "
                      f"τ={acc_tau:.2f}, a={acc_a:.2f}")
        
        return self.trace
    
    def get_posterior_summary(self):
        """Calcula resúmenes de la distribución posterior"""
        summary = {
            'mu': (np.mean(self.trace['mu']), np.std(self.trace['mu'])),
            'mu0': (np.mean(self.trace['mu0']), np.std(self.trace['mu0'])),
            'tau0': (np.mean(self.trace['tau0']), np.std(self.trace['tau0'])),
            'a0': (np.mean(self.trace['a0']), np.std(self.trace['a0'])),
            'beta0': (np.mean(self.trace['beta0']), np.std(self.trace['beta0'])),
            'n_clusters': (np.mean(self.trace['n_clusters']), 
                          np.std(self.trace['n_clusters']))
        }
        return summary
    
    def predict_density(self, y_new, X_new, n_samples=100):
        """Estima la densidad predictiva f(y_new | X_new, data)"""
        n_new = X_new.shape[0]
        y_grid = np.array(y_new)
        density = np.zeros((len(y_grid), n_new))
        
        X_new_norm = (X_new - self.X_mean) / self.X_std
        
        n_post = len(self.trace['z'])
        indices = np.linspace(0, n_post - 1, n_samples, dtype=int)
        
        for idx in indices:
            alpha_sample = self.trace['alpha'][idx]
            psi_sample = self.trace['psi'][idx]
            ell_sample = self.trace['ell'][idx]
            theta_mu_sample = self.trace['theta_mu'][idx]
            theta_b_sample = self.trace['theta_b'][idx]
            
            H_sample = len(alpha_sample)
            eta = np.zeros((n_new, H_sample))
            
            for h in range(H_sample):
                eta[:, h] = alpha_sample[h]
                for j in range(self.p):
                    ell_hj_value = self.ell_grid[j, ell_sample[h, j]]
                    dist = np.abs(X_new_norm[:, j] - ell_hj_value)
                    eta[:, h] -= psi_sample[h, j] * dist
            
            v = expit(eta)
            w = np.zeros((n_new, H_sample))
            for i in range(n_new):
                remaining = 1.0
                for h in range(H_sample):
                    w[i, h] = v[i, h] * remaining
                    remaining *= (1 - v[i, h])
            
            w = w / w.sum(axis=1, keepdims=True)
            
            for i in range(n_new):
                for y_idx, y_val in enumerate(y_grid):
                    for h in range(H_sample):
                        density[y_idx, i] += (w[i, h] * 
                            laplace.pdf(y_val, theta_mu_sample[h], theta_b_sample[h]))
        
        density /= n_samples
        
        return density
    
    def predict_mean(self, X_new, n_samples=100, return_full_uncertainty=True):
        """Estima E[Y|X] y desviación estándar con heterocedasticidad"""
        n_new = X_new.shape[0]
        
        X_new_norm = (X_new - self.X_mean) / self.X_std
        
        n_post = len(self.trace['z'])
        indices = np.linspace(0, n_post - 1, n_samples, dtype=int)
        
        predictions_mean = np.zeros((n_samples, n_new))
        predictions_var = np.zeros((n_samples, n_new))
        
        for s, idx in enumerate(indices):
            alpha_sample = self.trace['alpha'][idx]
            psi_sample = self.trace['psi'][idx]
            ell_sample = self.trace['ell'][idx]
            theta_mu_sample = self.trace['theta_mu'][idx]
            theta_b_sample = self.trace['theta_b'][idx]
            
            H_sample = len(alpha_sample)
            eta = np.zeros((n_new, H_sample))
            
            for h in range(H_sample):
                eta[:, h] = alpha_sample[h]
                for j in range(self.p):
                    ell_hj_value = self.ell_grid[j, ell_sample[h, j]]
                    dist = np.abs(X_new_norm[:, j] - ell_hj_value)
                    eta[:, h] -= psi_sample[h, j] * dist
            
            v = expit(eta)
            w = np.zeros((n_new, H_sample))
            for i in range(n_new):
                remaining = 1.0
                for h in range(H_sample):
                    w[i, h] = v[i, h] * remaining
                    remaining *= (1 - v[i, h])
            
            w = w / w.sum(axis=1, keepdims=True)
            
            mean_sample = np.sum(w * theta_mu_sample[np.newaxis, :H_sample], axis=1)
            
            second_moment = np.sum(
                w * (2 * theta_b_sample[np.newaxis, :H_sample]**2 + 
                     theta_mu_sample[np.newaxis, :H_sample]**2),
                axis=1
            )
            var_sample = second_moment - mean_sample**2
            var_sample = np.maximum(var_sample, 1e-8)
            
            predictions_mean[s, :] = mean_sample
            predictions_var[s, :] = var_sample
        
        mean_pred = np.mean(predictions_mean, axis=0)
        
        if return_full_uncertainty:
            var_within = np.mean(predictions_var, axis=0)
            var_between = np.var(predictions_mean, axis=0)
            
            total_var = var_within + var_between
            std_pred = np.sqrt(total_var)
        else:
            std_pred = np.std(predictions_mean, axis=0)
        
        return mean_pred, std_pred