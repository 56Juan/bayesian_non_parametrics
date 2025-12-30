import numpy as np
from scipy.stats import norm, invgamma, gamma, truncnorm
from scipy.special import ndtr  # CDF normal (Φ)
import math

# Importar el módulo C++ para PSBP
try:
    from . import psbp_cpp
    CPP_AVAILABLE = True
    print("Implementación PSBP en C++ Exitosa")
except ImportError as e:
    CPP_AVAILABLE = False
    _IMPORT_ERROR = e
    print(f"Implementación PSBP en C++ Fallida: {e}")


class PSBPNormal:
    """
    Probit Stick-Breaking Process (PSBP) con kernel Normal-Inverse-Gamma.
    VERSIÓN FIJADA: Todas las variables están activas (sin selección de variables).
    
    Modelo:
    y_i | z_i = h, μ_h, σ²_h ~ N(μ_h, σ²_h)
    z_i | x_i, {w_h(x_i)} ~ Categorical(w_1(x_i), ..., w_T(x_i))
    w_h(x) = v_h(x) ∏_{ℓ<h} (1 - v_ℓ(x))
    v_h(x) = Φ(η_h(x))
    η_h(x)= α_h - Σ_j ψ_{hj} · |x_j - ℓ_{hj}|
    
    Priors:
    α_h ~ N(μ_α, σ²_α)
    ψ_{hj} ~ N⁺(μ_ψ_j, τ⁻¹_ψ_j)  (truncado en 0 si psi_positive=True)
    ℓ_{hj} ~ Discrete-Uniform{ℓ*_{jm}}_{m=1}^{M_j}
    σ²_h ~ InvGamma(a₀, b₀)
    μ_h | σ²_h ~ N(μ₀, σ²_h / κ₀)
    """
    
    def __init__(self, y, X, H=20, 
                 mu_prior=(0.0, 1),           # μ_α ~ N(m_α, s²_α)
                 mu0_prior=(0.0, 100),          # μ₀ ~ N(m₀, s²₀)
                 kappa0_prior=(2.0, 1.0),       # κ₀ ~ Gamma(α_κ, β_κ)
                 a0_prior=(2.5, 1.0),           # a₀ ~ Gamma(α_a, β_a)
                 b0_prior=(2.0, 1.0),           # b₀ ~ Gamma(α_b, β_b)
                 psi_prior_mean=None,           # μ_ψ_j
                 psi_prior_prec=None,           # τ_ψ_j
                 mu_psi_hyperprior=(0.0, 10.0), # μ_ψ_j ~ N(m_ψ_j, s²_ψ_j)
                 tau_psi_hyperprior=(2.0, 0.5), # τ_ψ_j ~ Gamma(α_τ_j, β_τ_j)
                 estimate_psi_hyperpriors=True, 
                 n_grid=50, 
                 verbose=True, 
                 use_cpp=True, 
                 psi_positive=True,
                 normalize=False):
        
        self.y = np.array(y).flatten()
        self.X = np.array(X)
        if self.X.ndim == 1:
            self.X = self.X.reshape(-1, 1)
        self.n, self.p = self.X.shape
        self.H = H
        self.verbose = verbose
        self.use_cpp = use_cpp and CPP_AVAILABLE
        self.normalize = normalize
        self.psi_positive = psi_positive
        
        if self.use_cpp:
            print("Using C++ acceleration")
        
        # NORMALIZACIÓN CONDICIONAL
        if self.normalize:
            self.y_mean = np.mean(y)
            self.y_std = np.std(y)
            if self.y_std < 1e-10:
                self.y_std = 1.0
            self.y_normalized = (self.y - self.y_mean) / self.y_std
            
            self.X_mean = np.mean(X, axis=0)
            self.X_std = np.std(X, axis=0)
            self.X_std[self.X_std < 1e-10] = 1.0
            self.X_normalized = (self.X - self.X_mean) / self.X_std
        else:
            self.y_mean = 0.0
            self.y_std = 1.0
            self.y_normalized = self.y.copy()
            
            self.X_mean = np.zeros(self.p)
            self.X_std = np.ones(self.p)
            self.X_normalized = self.X.copy()
        
        # Hiperparámetros
        self.mu_mu, self.tau_mu_inv = mu_prior
        self.m0, self.s02 = mu0_prior
        self.alpha_kappa, self.beta_kappa = kappa0_prior
        self.alpha_a, self.beta_a = a0_prior
        self.alpha_b, self.beta_b = b0_prior
        self.m_psi, self.s2_psi = mu_psi_hyperprior
        self.alpha_tau, self.beta_tau = tau_psi_hyperprior
        self.estimate_psi_hyperpriors = estimate_psi_hyperpriors
        
        self.mu_psi = np.full(self.p, self.m_psi) if psi_prior_mean is None else np.array(psi_prior_mean)
        self.tau_psi = np.ones(self.p) if psi_prior_prec is None else np.array(psi_prior_prec)
        
        # Grilla para ℓ
        self.n_grid = n_grid
        self.ell_grid = self._create_grid()
        
        # MH
        self.mh_scales = {'kappa0': 0.2, 'a0': 0.2, 'alpha': 0.1, 'psi': 0.1}
        self.mh_acceptance = {'kappa0': [], 'a0': [], 'alpha': [], 'psi': []}
        
        # Storage
        self.trace = {
            'mu': [], 'mu0': [], 'kappa0': [], 'a0': [], 'b0': [],
            'z': [], 'theta_mu': [], 'theta_sigma2': [], 'w': [],
            'n_clusters': [], 'alpha': [], 'psi': [], 'ell': []
        }
        if self.estimate_psi_hyperpriors:
            self.trace['mu_psi'] = []
            self.trace['tau_psi'] = []
        
        self.initialize()
    
    def _create_grid(self):
        grid = np.zeros((self.p, self.n_grid))
        for j in range(self.p):
            x_min = self.X_normalized[:, j].min() - 0.5
            x_max = self.X_normalized[:, j].max() + 0.5
            grid[j, :] = np.linspace(x_min, x_max, self.n_grid)
        return grid
    
    def initialize(self):
        # Hiperparámetros
        self.mu = np.random.normal(self.mu_mu, np.sqrt(self.tau_mu_inv))
        self.mu0 = np.random.normal(self.m0, np.sqrt(self.s02))
        self.kappa0 = np.clip(np.random.gamma(self.alpha_kappa, 1.0/self.beta_kappa), 0.1, 10.0)
        self.a0 = np.clip(np.random.gamma(self.alpha_a, 1.0/self.beta_a), 1.0, 10.0)
        self.b0 = np.clip(np.random.gamma(self.alpha_b, 1.0/self.beta_b), 0.1, 10.0)
        
        # Parámetros de dependencia (TODAS LAS VARIABLES ACTIVAS)
        self.alpha = np.random.normal(self.mu, 1.0, size=self.H)
        self.psi = np.zeros((self.H, self.p))
        
        for h in range(self.H):
            for j in range(self.p):
                if self.psi_positive:
                    a = (0 - self.mu_psi[j]) / np.sqrt(1.0/self.tau_psi[j])
                    self.psi[h, j] = max(truncnorm.rvs(a, np.inf, loc=self.mu_psi[j], 
                                          scale=np.sqrt(1.0/self.tau_psi[j])), 1e-6)
                else:
                    self.psi[h, j] = np.random.normal(self.mu_psi[j], np.sqrt(1.0/self.tau_psi[j]))
        
        self.ell = np.random.randint(0, self.n_grid, size=(self.H, self.p))
        self.w = self._compute_weights()
        
        # Átomos
        self.theta_mu = np.zeros(self.H)
        self.theta_sigma2 = np.zeros(self.H)
        for h in range(self.H):
            self.theta_sigma2[h] = np.clip(invgamma.rvs(self.a0, scale=self.b0), 0.01, 100.0)
            self.theta_mu[h] = np.random.normal(self.mu0, math.sqrt(self.theta_sigma2[h] / self.kappa0))
        
        # Asignaciones
        self.z = np.zeros(self.n, dtype=int)
        for i in range(self.n):
            likes = np.clip([norm.pdf(self.y_normalized[i], self.theta_mu[h], 
                            math.sqrt(self.theta_sigma2[h])) for h in range(self.H)], 1e-300, None)
            probs = self.w[i, :] * likes
            probs = probs / probs.sum() if probs.sum() > 1e-300 else np.ones(self.H) / self.H
            self.z[i] = np.random.choice(self.H, p=probs)
        
        # Variables latentes u_{ih} (Data Augmentation para probit)
        self.u_latent = np.zeros((self.n, self.H))
        self._update_u_latent()
    
    def _compute_eta(self, X_batch=None):
        """η_h(x) = α_h - Σ_j ψ_{hj}·|x_j - ℓ_{hj}| (TODAS LAS VARIABLES ACTIVAS)"""
        if X_batch is None:
            X_batch = self.X_normalized
        
        if self.use_cpp:
            result = psbp_cpp.compute_eta(
                X_batch.astype(np.float64),
                self.alpha.astype(np.float64),
                self.psi.astype(np.float64),
                self.ell.astype(np.int32),
                self.ell_grid.astype(np.float64)
            )
            return np.array(result.eta)
        
        n_batch = X_batch.shape[0]
        eta = np.zeros((n_batch, self.H))
        for h in range(self.H):
            eta[:, h] = self.alpha[h]
            for j in range(self.p):
                ell_hj_value = self.ell_grid[j, self.ell[h, j]]
                dist = np.abs(X_batch[:, j] - ell_hj_value)
                eta[:, h] -= self.psi[h, j] * dist
        return eta
    
    def _compute_weights(self, X_batch=None):
        """Calcula pesos con enlace probit: v_h(x) = Φ(η_h(x))"""
        if X_batch is None:
            X_batch = self.X_normalized
        
        if self.use_cpp:
            result = psbp_cpp.compute_weights_probit(
                X_batch.astype(np.float64),
                self.alpha.astype(np.float64),
                self.psi.astype(np.float64),
                self.ell.astype(np.int32),
                self.ell_grid.astype(np.float64)
            )
            return np.array(result.weights)
        
        eta = self._compute_eta(X_batch)
        v = ndtr(eta)  # Φ(η) - función probit
        
        n_batch = X_batch.shape[0]
        w = np.zeros((n_batch, self.H))
        for i in range(n_batch):
            remaining = 1.0
            for h in range(self.H):
                w[i, h] = v[i, h] * remaining
                remaining *= (1 - v[i, h])
                if remaining < 1e-15:
                    break
        
        row_sums = w.sum(axis=1, keepdims=True)
        return w / np.where(row_sums < 1e-15, 1.0, row_sums)
    
    def _update_u_latent(self):
        """Data augmentation para probit: u_{ih} ~ TruncatedNormal(η_h(x_i), 1, truncation)"""
        if self.use_cpp:
            result = psbp_cpp.update_u_latent(
                self.u_latent.astype(np.float64),
                self.z.astype(np.int32),
                self.X_normalized.astype(np.float64),
                self.alpha.astype(np.float64),
                self.psi.astype(np.float64),
                self.ell.astype(np.int32),
                self.ell_grid.astype(np.float64)
            )
            self.u_latent = np.array(result.u_latent)
            return
        
        eta = self._compute_eta(self.X_normalized)
        for i in range(self.n):
            z_i = self.z[i]
            for h in range(self.H):
                if h > z_i:
                    continue
                eta_ih = eta[i, h]
                if z_i == h:
                    # u > 0
                    a = (0 - eta_ih) / 1.0
                    self.u_latent[i, h] = truncnorm.rvs(a, np.inf, loc=eta_ih, scale=1.0)
                elif z_i > h:
                    # u < 0
                    b = (0 - eta_ih) / 1.0
                    self.u_latent[i, h] = truncnorm.rvs(-np.inf, b, loc=eta_ih, scale=1.0)
    
    def sample_slice_variables(self):
        u = np.zeros(self.n)
        for i in range(self.n):
            w_z = self.w[i, self.z[i]]
            u[i] = np.random.uniform(0, max(w_z, 1e-10))
        
        u_min = u.min()
        for _ in range(3):
            if self.H >= 100 or np.all(self.w.min(axis=0) < u_min):
                break
            self._expand_clusters(n_new=3)
        return u
    
    def _expand_clusters(self, n_new=3):
        H_new = self.H + n_new
        self.alpha = np.append(self.alpha, np.random.normal(self.mu, 1.0, size=n_new))
        
        psi_new = np.zeros((n_new, self.p))
        for h in range(n_new):
            for j in range(self.p):
                if self.psi_positive:
                    a = (0 - self.mu_psi[j]) / np.sqrt(1.0/self.tau_psi[j])
                    psi_new[h, j] = max(truncnorm.rvs(a, np.inf, loc=self.mu_psi[j],
                                        scale=np.sqrt(1.0/self.tau_psi[j])), 1e-6)
                else:
                    psi_new[h, j] = np.random.normal(self.mu_psi[j], np.sqrt(1.0/self.tau_psi[j]))
        
        self.psi = np.vstack([self.psi, psi_new])
        self.ell = np.vstack([self.ell, np.random.randint(0, self.n_grid, size=(n_new, self.p))])
        
        theta_mu_new = np.zeros(n_new)
        theta_sigma2_new = np.zeros(n_new)
        for h in range(n_new):
            theta_sigma2_new[h] = np.clip(invgamma.rvs(self.a0, scale=self.b0), 0.01, 100.0)
            theta_mu_new[h] = np.random.normal(self.mu0, math.sqrt(theta_sigma2_new[h] / self.kappa0))
        
        self.theta_mu = np.append(self.theta_mu, theta_mu_new)
        self.theta_sigma2 = np.append(self.theta_sigma2, theta_sigma2_new)
        self.H = H_new
        self.w = self._compute_weights()
        self.u_latent = np.hstack([self.u_latent, np.zeros((self.n, n_new))])
    
    def update_assignments(self, u):
        """Actualiza asignaciones z_i ~ Categorical(w_1(x_i), ..., w_T(x_i))"""
        if self.use_cpp:
            self.z = np.array(psbp_cpp.update_assignments(
                u.astype(np.float64),
                self.w.astype(np.float64),
                self.y_normalized.astype(np.float64),
                self.theta_mu.astype(np.float64),
                self.theta_sigma2.astype(np.float64),
                self.z.astype(np.int32)
            ))
            return
        
        for i in range(self.n):
            candidates = np.where(self.w[i, :] > u[i])[0]
            if len(candidates) == 0:
                candidates = np.arange(self.H)
            
            likes = np.clip(norm.pdf(self.y_normalized[i], self.theta_mu[candidates],
                           np.sqrt(self.theta_sigma2[candidates])), 1e-300, None)
            probs = self.w[i, candidates] * likes
            probs = probs / probs.sum() if probs.sum() > 1e-300 else np.ones_like(likes) / len(likes)
            self.z[i] = candidates[np.random.choice(len(candidates), p=probs)]
    
    def update_atoms(self):
        """Actualiza átomos (μ_h, σ²_h) ~ Normal-Inverse-Gamma"""
        if self.use_cpp:
            result = psbp_cpp.update_atoms(
                self.z.astype(np.int32),
                self.y_normalized.astype(np.float64),
                self.theta_mu.astype(np.float64),
                self.theta_sigma2.astype(np.float64),
                float(self.mu0),
                float(self.kappa0),
                float(self.a0),
                float(self.b0),
                int(self.H)
            )
            self.theta_mu = np.array(result.theta_mu)
            self.theta_sigma2 = np.array(result.theta_sigma2)
            return
        
        for h in range(self.H):
            members = self.y_normalized[self.z == h]
            n_h = len(members)
            if n_h > 0:
                y_bar = members.mean()
                ss = np.sum((members - y_bar)**2)
                kappa_n = self.kappa0 + n_h
                mu_n = (self.kappa0 * self.mu0 + n_h * y_bar) / kappa_n
                a_n = self.a0 + n_h / 2.0
                b_n = self.b0 + 0.5 * ss + (self.kappa0 * n_h * (y_bar - self.mu0)**2) / (2 * kappa_n)
                self.theta_sigma2[h] = np.clip(invgamma.rvs(a_n, scale=b_n), 0.01, 100.0)
                self.theta_mu[h] = np.random.normal(mu_n, math.sqrt(self.theta_sigma2[h] / kappa_n))
            else:
                self.theta_sigma2[h] = np.clip(invgamma.rvs(self.a0, scale=self.b0), 0.01, 100.0)
                self.theta_mu[h] = np.random.normal(self.mu0, math.sqrt(self.theta_sigma2[h] / self.kappa0))
    
    def update_alpha(self):
        """Actualiza α_h ~ N(μ_α, σ²_α) usando MH"""
        if self.use_cpp:
            result = psbp_cpp.update_alpha(
                self.alpha.astype(np.float64),
                self.u_latent.astype(np.float64),
                self.X_normalized.astype(np.float64),
                self.psi.astype(np.float64),
                self.ell.astype(np.int32),
                self.ell_grid.astype(np.float64),
                self.z.astype(np.int32),
                float(self.mu),
                float(self.mh_scales['alpha']),
                int(self.H)
            )
            self.alpha = np.array(result.alpha)
            self.mh_acceptance['alpha'].extend(result.acceptances)
            return
        
        for h in range(self.H):
            affected_idx = np.where(self.z >= h)[0]
            if len(affected_idx) == 0:
                self.alpha[h] = np.random.normal(self.mu, 1.0)
                continue
            
            correction = np.zeros(len(affected_idx))
            for j in range(self.p):
                ell_hj_value = self.ell_grid[j, self.ell[h, j]]
                dist = np.abs(self.X_normalized[affected_idx, j] - ell_hj_value)
                correction += self.psi[h, j] * dist
            
            u_affected = self.u_latent[affected_idx, h]
            n_affected = len(affected_idx)
            tau_post = n_affected + 1.0
            mu_post = (np.sum(u_affected + correction) + self.mu) / tau_post
            self.alpha[h] = np.random.normal(mu_post, 1.0 / np.sqrt(tau_post))
    
    def update_psi(self):
        """Actualiza ψ_{hj} ~ N⁺(μ_ψ_j, τ⁻¹_ψ_j) usando MH"""
        if self.use_cpp:
            result = psbp_cpp.update_psi(
                self.psi.astype(np.float64),
                self.mu_psi.astype(np.float64),
                self.tau_psi.astype(np.float64),
                self.u_latent.astype(np.float64),
                self.X_normalized.astype(np.float64),
                self.alpha.astype(np.float64),
                self.ell.astype(np.int32),
                self.ell_grid.astype(np.float64),
                self.z.astype(np.int32),
                float(self.mh_scales['psi']),
                bool(self.psi_positive),
                int(self.H)
            )
            self.psi = np.array(result.psi)
            for h in range(self.H):
                self.mh_acceptance['psi'].extend(result.acceptances_psi[h])
            return
        
        for h in range(self.H):
            for j in range(self.p):
                psi_curr = self.psi[h, j]
                psi_prop = np.random.normal(psi_curr, self.mh_scales['psi'])
                
                if self.psi_positive and psi_prop < 0:
                    self.mh_acceptance['psi'].append(False)
                    continue
                
                affected_idx = np.where(self.z >= h)[0]
                if len(affected_idx) == 0:
                    log_prior_curr = -0.5 * self.tau_psi[j] * (psi_curr - self.mu_psi[j])**2
                    log_prior_prop = -0.5 * self.tau_psi[j] * (psi_prop - self.mu_psi[j])**2
                    log_r = log_prior_prop - log_prior_curr
                else:
                    ell_hj_value = self.ell_grid[j, self.ell[h, j]]
                    dist = np.abs(self.X_normalized[affected_idx, j] - ell_hj_value)
                    
                    eta_base = self.alpha[h]
                    for jj in range(self.p):
                        if jj != j:
                            ell_jj_value = self.ell_grid[jj, self.ell[h, jj]]
                            dist_jj = np.abs(self.X_normalized[affected_idx, jj] - ell_jj_value)
                            eta_base -= self.psi[h, jj] * dist_jj
                    
                    u_affected = self.u_latent[affected_idx, h]
                    
                    log_like_curr = -0.5 * np.sum((u_affected - (eta_base - psi_curr * dist))**2)
                    log_like_prop = -0.5 * np.sum((u_affected - (eta_base - psi_prop * dist))**2)
                    
                    log_prior_curr = -0.5 * self.tau_psi[j] * (psi_curr - self.mu_psi[j])**2
                    log_prior_prop = -0.5 * self.tau_psi[j] * (psi_prop - self.mu_psi[j])**2
                    
                    log_r = (log_like_prop + log_prior_prop) - (log_like_curr + log_prior_curr)
                
                accept = np.log(np.random.rand()) < log_r
                if accept:
                    self.psi[h, j] = psi_prop
                self.mh_acceptance['psi'].append(accept)
    
    def update_ell(self):
        """Actualiza ℓ_{hj} ~ Discrete-Uniform{ℓ*_{jm}}"""
        if self.use_cpp:
            result = psbp_cpp.update_ell(
                self.ell.astype(np.int32),
                self.z.astype(np.int32),
                self.X_normalized.astype(np.float64),
                self.alpha.astype(np.float64),
                self.psi.astype(np.float64),
                self.u_latent.astype(np.float64),
                self.ell_grid.astype(np.float64),
                int(self.H),
                int(self.n_grid)
            )
            self.ell = np.array(result.ell)
            return
        
        for h in range(self.H):
            for j in range(self.p):
                affected_idx = np.where(self.z >= h)[0]
                if len(affected_idx) == 0:
                    self.ell[h, j] = np.random.randint(0, self.n_grid)
                    continue
                
                log_likes = np.zeros(self.n_grid)
                for m in range(self.n_grid):
                    ell_value = self.ell_grid[j, m]
                    dist = np.abs(self.X_normalized[affected_idx, j] - ell_value)
                    eta = self.alpha[h] - self.psi[h, j] * dist
                    for jj in range(self.p):
                        if jj != j:
                            ell_jj_value = self.ell_grid[jj, self.ell[h, jj]]
                            eta -= self.psi[h, jj] * np.abs(self.X_normalized[affected_idx, jj] - ell_jj_value)
                    log_likes[m] = -0.5 * np.sum((self.u_latent[affected_idx, h] - eta)**2)
                
                log_likes -= np.max(log_likes)
                probs = np.exp(log_likes)
                probs /= probs.sum()
                self.ell[h, j] = np.random.choice(self.n_grid, p=probs)
    
    def update_weights(self):
        self.w = self._compute_weights()
    
    def update_mu(self):
        """Actualiza μ_α ~ N(m_α, s²_α)"""
        tau_post = len(self.alpha) + 1.0 / self.tau_mu_inv
        mu_post = (np.sum(self.alpha) + self.mu_mu / self.tau_mu_inv) / tau_post
        self.mu = np.random.normal(mu_post, 1.0 / np.sqrt(tau_post))
    
    def update_mu0(self):
        """Actualiza μ₀ ~ N(m₀, s²₀)"""
        active_clusters = np.unique(self.z)
        if len(active_clusters) == 0:
            return
        mu_active = self.theta_mu[active_clusters]
        sigma2_active = self.theta_sigma2[active_clusters]
        inv_sigma2 = np.clip(1.0 / sigma2_active, 1e-6, 1e6)
        precision_post = self.kappa0 * np.sum(inv_sigma2) + 1.0 / self.s02
        s0n2 = 1.0 / precision_post
        m0n = s0n2 * (self.kappa0 * np.sum(mu_active * inv_sigma2) + self.m0 / self.s02)
        self.mu0 = np.random.normal(m0n, math.sqrt(s0n2))
    
    def update_kappa0(self):
        """Actualiza κ₀ ~ Gamma(α_κ, β_κ) usando MH"""
        log_kappa = math.log(self.kappa0)
        log_kappa_prop = np.random.normal(log_kappa, self.mh_scales['kappa0'])
        kappa_prop = np.clip(math.exp(log_kappa_prop), 0.01, 100.0)
        
        active_clusters = np.unique(self.z)
        if len(active_clusters) == 0:
            return
        
        mu_active = self.theta_mu[active_clusters]
        sigma2_active = self.theta_sigma2[active_clusters]
        diff_sq = (mu_active - self.mu0)**2
        inv_sigma2 = np.clip(1.0 / sigma2_active, 1e-6, 1e6)
        
        K = len(active_clusters)
        log_like_curr = 0.5 * K * np.log(self.kappa0) - 0.5 * self.kappa0 * np.sum(diff_sq * inv_sigma2)
        log_like_prop = 0.5 * K * np.log(kappa_prop) - 0.5 * kappa_prop * np.sum(diff_sq * inv_sigma2)
        log_prior_curr = (self.alpha_kappa - 1) * math.log(self.kappa0) - self.beta_kappa * self.kappa0
        log_prior_prop = (self.alpha_kappa - 1) * math.log(kappa_prop) - self.beta_kappa * kappa_prop
        
        log_r = (log_like_prop + log_prior_prop) - (log_like_curr + log_prior_curr)
        accept = math.log(np.random.rand()) < log_r
        if accept:
            self.kappa0 = kappa_prop
        self.mh_acceptance['kappa0'].append(float(accept))
    
    def update_mu_psi(self):
        """Actualiza μ_ψ_j ~ N(m_ψ_j, s²_ψ_j) (si estimate_psi_hyperpriors=True)"""
        if not self.estimate_psi_hyperpriors:
            return
        
        for j in range(self.p):
            psi_active = self.psi[:, j]
            n_active = len(psi_active)
            
            tau_post = n_active * self.tau_psi[j] + 1.0 / self.s2_psi
            m_post = (self.tau_psi[j] * np.sum(psi_active) + self.m_psi / self.s2_psi) / tau_post
            s_post = 1.0 / np.sqrt(tau_post)
            self.mu_psi[j] = np.random.normal(m_post, s_post)
    
    def update_tau_psi(self):
        """Actualiza τ_ψ_j ~ Gamma(α_τ_j, β_τ_j) (si estimate_psi_hyperpriors=True)"""
        if not self.estimate_psi_hyperpriors:
            return
        
        for j in range(self.p):
            psi_active = self.psi[:, j]
            n_active = len(psi_active)
            
            alpha_post = self.alpha_tau + n_active / 2.0
            ss = np.sum((psi_active - self.mu_psi[j])**2)
            beta_post = self.beta_tau + 0.5 * ss
            self.tau_psi[j] = gamma.rvs(alpha_post, scale=1.0/beta_post)
    
    def adapt_mh_scales(self, iteration):
        """Adapta escalas MH durante burnin"""
        if iteration > 50 and iteration % 50 == 0:
            for param in ['kappa0', 'a0', 'alpha', 'psi']:
                if len(self.mh_acceptance[param]) > 0:
                    recent = self.mh_acceptance[param][-50:]
                    acc_rate = np.mean(recent)
                    
                    if acc_rate < 0.15:
                        self.mh_scales[param] *= 0.8
                    elif acc_rate > 0.4:
                        self.mh_scales[param] *= 1.2
                    
                    self.mh_scales[param] = np.clip(self.mh_scales[param], 0.01, 1.0)
    
    def update_a0(self):
        """Actualiza a₀ ~ Gamma(α_a, β_a) usando MH"""
        log_a = math.log(self.a0)
        log_a_prop = np.random.normal(log_a, self.mh_scales['a0'])
        a_prop = np.clip(math.exp(log_a_prop), 0.5, 20.0)
        active_clusters = np.unique(self.z)
        if len(active_clusters) == 0:
            return
        sigma2_active = self.theta_sigma2[active_clusters]
        ratio = np.clip(self.b0 / sigma2_active, 1e-10, 1e10)
        log_ratio = np.log(ratio)
        K = len(active_clusters)
        log_like_curr = self.a0 * np.sum(log_ratio) - K * math.lgamma(self.a0) - (self.a0 + 1) * np.sum(np.log(sigma2_active))
        log_like_prop = a_prop * np.sum(log_ratio) - K * math.lgamma(a_prop) - (a_prop + 1) * np.sum(np.log(sigma2_active))
        log_r = (log_like_prop + (self.alpha_a - 1) * math.log(a_prop) - self.beta_a * a_prop) - (log_like_curr + (self.alpha_a - 1) * math.log(self.a0) - self.beta_a * self.a0)
        if math.log(np.random.rand()) < log_r:
            self.a0 = a_prop
        self.mh_acceptance['a0'].append(float(math.log(np.random.rand()) < log_r))
    
    def update_b0(self):
        """Actualiza b₀ ~ Gamma(α_b, β_b)"""
        active_clusters = np.unique(self.z)
        if len(active_clusters) == 0:
            return
        sigma2_active = self.theta_sigma2[active_clusters]
        alpha_post = self.alpha_b + len(active_clusters) * self.a0
        beta_post = self.beta_b + np.sum(np.clip(1.0 / sigma2_active, 1e-6, 1e6))
        self.b0 = np.clip(np.random.gamma(alpha_post, 1.0 / beta_post), 0.01, 100.0)
    
    def run(self, iterations=1000, burnin=500):
        for it in range(iterations):
            u = self.sample_slice_variables()
            self.update_assignments(u)
            self.update_atoms()
            self._update_u_latent()
            self.update_alpha()
            self.update_psi()
            self.update_ell()
            self.update_weights()
            self.update_mu()
            self.update_mu0()
            self.update_kappa0()
            self.update_a0()
            self.update_b0()
            
            if self.estimate_psi_hyperpriors:
                self.update_mu_psi()
                self.update_tau_psi()
            
            if it < burnin:
                self.adapt_mh_scales(it)
            
            if it >= burnin:
                self.trace['mu'].append(self.mu)
                # TRANSFORMACIÓN CONDICIONAL
                if self.normalize:
                    self.trace['mu0'].append(self.mu0 * self.y_std + self.y_mean)
                    self.trace['b0'].append(self.b0 * (self.y_std ** 2))
                    self.trace['theta_mu'].append((self.theta_mu * self.y_std + self.y_mean).copy())
                    self.trace['theta_sigma2'].append((self.theta_sigma2 * (self.y_std ** 2)).copy())
                else:
                    self.trace['mu0'].append(self.mu0)
                    self.trace['b0'].append(self.b0)
                    self.trace['theta_mu'].append(self.theta_mu.copy())
                    self.trace['theta_sigma2'].append(self.theta_sigma2.copy())
                
                self.trace['kappa0'].append(self.kappa0)
                self.trace['a0'].append(self.a0)
                self.trace['z'].append(self.z.copy())
                self.trace['w'].append(self.w.copy())
                self.trace['n_clusters'].append(len(np.unique(self.z)))
                self.trace['alpha'].append(self.alpha.copy())
                self.trace['psi'].append(self.psi.copy())
                self.trace['ell'].append(self.ell.copy())
                
                if self.estimate_psi_hyperpriors:
                    self.trace['mu_psi'].append(self.mu_psi.copy())
                    self.trace['tau_psi'].append(self.tau_psi.copy())
            
            if self.verbose and (it + 1) % 100 == 0:
                n_clusters = len(np.unique(self.z))
                # PRINT CONDICIONAL
                if self.normalize:
                    mu0_print = self.mu0 * self.y_std + self.y_mean
                    b0_print = self.b0 * (self.y_std ** 2)
                else:
                    mu0_print = self.mu0
                    b0_print = self.b0
                    
                print(f"Iter {it+1}/{iterations}: "
                    f"K_eff={n_clusters}, H={self.H}, "
                    f"μ={self.mu:.2f}, "
                    f"μ₀={mu0_print:.2f}, "
                    f"κ₀={self.kappa0:.2f}, "
                    f"a₀={self.a0:.2f}, "
                    f"b₀={b0_print:.2f}")
        return self.trace
    
    def get_posterior_summary(self):
        """Resúmenes posteriores"""
        summary = {
            'mu': (np.mean(self.trace['mu']), np.std(self.trace['mu'])),
            'mu0': (np.mean(self.trace['mu0']), np.std(self.trace['mu0'])),
            'kappa0': (np.mean(self.trace['kappa0']), np.std(self.trace['kappa0'])),
            'a0': (np.mean(self.trace['a0']), np.std(self.trace['a0'])),
            'b0': (np.mean(self.trace['b0']), np.std(self.trace['b0'])),
            'n_clusters': (np.mean(self.trace['n_clusters']), np.std(self.trace['n_clusters']))
        }
        return summary
    
    def predict_density(self, y_new, X_new, n_samples=100):
        """Densidad predictiva f(y_new | X_new, data)"""
        n_new = X_new.shape[0]
        y_grid = np.array(y_new)
        density = np.zeros((len(y_grid), n_new))
        
        # NORMALIZACIÓN CONDICIONAL
        if self.normalize:
            X_new_norm = (X_new - self.X_mean) / self.X_std
        else:
            X_new_norm = X_new.copy()
        
        n_post = len(self.trace['z'])
        indices = np.linspace(0, n_post - 1, min(n_samples, n_post), dtype=int)
        
        for idx in indices:
            H_sample = len(self.trace['alpha'][idx])
            eta = np.zeros((n_new, H_sample))
            
            for h in range(H_sample):
                eta[:, h] = self.trace['alpha'][idx][h]
                for j in range(self.p):
                    ell_hj_value = self.ell_grid[j, self.trace['ell'][idx][h, j]]
                    dist = np.abs(X_new_norm[:, j] - ell_hj_value)
                    eta[:, h] -= self.trace['psi'][idx][h, j] * dist
            
            v = ndtr(eta)
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
                            norm.pdf(y_val, self.trace['theta_mu'][idx][h], 
                                   np.sqrt(self.trace['theta_sigma2'][idx][h])))
        
        density /= len(indices)
        return density
    
    def predict_with_decomposition(self, X_new, n_samples=100):
        """Predicción con descomposición de incertidumbre"""
        n_new = X_new.shape[0]
        
        # NORMALIZACIÓN CONDICIONAL
        if self.normalize:
            X_new_norm = (X_new - self.X_mean) / self.X_std
        else:
            X_new_norm = X_new.copy()
        
        n_post = len(self.trace['z'])
        indices = np.linspace(0, n_post - 1, min(n_samples, n_post), dtype=int)
        
        predictions_mean = np.zeros((len(indices), n_new))
        predictions_var = np.zeros((len(indices), n_new))
        
        for s, idx in enumerate(indices):
            H_sample = len(self.trace['alpha'][idx])
            eta = np.zeros((n_new, H_sample))
            
            for h in range(H_sample):
                eta[:, h] = self.trace['alpha'][idx][h]
                for j in range(self.p):
                    ell_hj_value = self.ell_grid[j, self.trace['ell'][idx][h, j]]
                    dist = np.abs(X_new_norm[:, j] - ell_hj_value)
                    eta[:, h] -= self.trace['psi'][idx][h, j] * dist
            
            v = ndtr(eta)
            w = np.zeros((n_new, H_sample))
            for i in range(n_new):
                remaining = 1.0
                for h in range(H_sample):
                    w[i, h] = v[i, h] * remaining
                    remaining *= (1 - v[i, h])
            w = w / w.sum(axis=1, keepdims=True)
            
            mean_sample = np.sum(w * self.trace['theta_mu'][idx][np.newaxis, :H_sample], axis=1)
            second_moment = np.sum(w * (self.trace['theta_sigma2'][idx][np.newaxis, :H_sample] + 
                                       self.trace['theta_mu'][idx][np.newaxis, :H_sample]**2), axis=1)
            var_sample = np.maximum(second_moment - mean_sample**2, 1e-8)
            
            predictions_mean[s, :] = mean_sample
            predictions_var[s, :] = var_sample
        
        mean_pred = np.mean(predictions_mean, axis=0)
        var_within = np.mean(predictions_var, axis=0)
        var_between = np.var(predictions_mean, axis=0)
        
        return {
            'mean': mean_pred,
            'std_total': np.sqrt(var_within + var_between),
            'std_aleatoric': np.sqrt(var_within),
            'std_epistemic': np.sqrt(var_between),
            'var_within': var_within,
            'var_between': var_between
        }
    
    def predict_quantiles(self, X_new, quantiles=[0.025, 0.25, 0.5, 0.75, 0.975], 
                          n_samples=100, n_pred_samples=1000):
        """Cuantiles predictivos"""
        n_new = X_new.shape[0]
        
        # NORMALIZACIÓN CONDICIONAL
        if self.normalize:
            X_new_norm = (X_new - self.X_mean) / self.X_std
        else:
            X_new_norm = X_new.copy()
        
        n_post = len(self.trace['z'])
        indices = np.linspace(0, n_post - 1, min(n_samples, n_post), dtype=int)
        y_samples = np.zeros((n_pred_samples, n_new))
        
        for i in range(n_pred_samples):
            post_idx = np.random.choice(indices)
            H_sample = len(self.trace['alpha'][post_idx])
            eta = np.zeros((n_new, H_sample))
            
            for h in range(H_sample):
                eta[:, h] = self.trace['alpha'][post_idx][h]
                for j in range(self.p):
                    ell_hj_value = self.ell_grid[j, self.trace['ell'][post_idx][h, j]]
                    dist = np.abs(X_new_norm[:, j] - ell_hj_value)
                    eta[:, h] -= self.trace['psi'][post_idx][h, j] * dist
            
            v = ndtr(eta)
            w = np.zeros((n_new, H_sample))
            for k in range(n_new):
                remaining = 1.0
                for h in range(H_sample):
                    w[k, h] = v[k, h] * remaining
                    remaining *= (1 - v[k, h])
            w = w / w.sum(axis=1, keepdims=True)
            
            for k in range(n_new):
                cluster = np.random.choice(H_sample, p=w[k, :])
                y_samples[i, k] = np.random.normal(
                    self.trace['theta_mu'][post_idx][cluster],
                    np.sqrt(self.trace['theta_sigma2'][post_idx][cluster])
                )
        
        results = {'mean': np.mean(y_samples, axis=0)}
        for q in quantiles:
            results[f'q_{q}'] = np.quantile(y_samples, q, axis=0)
        
        return results
    
    def predict_mean(self, X_new, n_samples=100, return_full_uncertainty=True):
        """Predice E[Y|X] y desviación estándar"""
        n_new = X_new.shape[0]
        
        # NORMALIZACIÓN CONDICIONAL
        if self.normalize:
            X_new_norm = (X_new - self.X_mean) / self.X_std
        else:
            X_new_norm = X_new.copy()
        
        n_post = len(self.trace['z'])
        indices = np.linspace(0, n_post - 1, min(n_samples, n_post), dtype=int)
        predictions_mean = np.zeros((len(indices), n_new))
        predictions_var = np.zeros((len(indices), n_new))
        
        for s, idx in enumerate(indices):
            eta = np.zeros((n_new, len(self.trace['alpha'][idx])))
            for h in range(len(self.trace['alpha'][idx])):
                eta[:, h] = self.trace['alpha'][idx][h]
                for j in range(self.p):
                    ell_hj_value = self.ell_grid[j, self.trace['ell'][idx][h, j]]
                    eta[:, h] -= self.trace['psi'][idx][h, j] * np.abs(X_new_norm[:, j] - ell_hj_value)
            
            v = ndtr(eta)  # Φ
            w = np.zeros((n_new, len(self.trace['alpha'][idx])))
            for i in range(n_new):
                remaining = 1.0
                for h in range(len(self.trace['alpha'][idx])):
                    w[i, h] = v[i, h] * remaining
                    remaining *= (1 - v[i, h])
            w = w / w.sum(axis=1, keepdims=True)
            
            predictions_mean[s, :] = np.sum(w * self.trace['theta_mu'][idx][np.newaxis, :len(self.trace['alpha'][idx])], axis=1)
            predictions_var[s, :] = np.sum(w * (self.trace['theta_sigma2'][idx][np.newaxis, :len(self.trace['alpha'][idx])] + self.trace['theta_mu'][idx][np.newaxis, :len(self.trace['alpha'][idx])]**2), axis=1) - predictions_mean[s, :]**2
        
        mean_pred = np.mean(predictions_mean, axis=0)
        if return_full_uncertainty:
            std_pred = np.sqrt(np.mean(predictions_var, axis=0) + np.var(predictions_mean, axis=0))
        else:
            std_pred = np.std(predictions_mean, axis=0)
        return mean_pred, std_pred