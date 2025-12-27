import numpy as np
from scipy.stats import norm, invgamma, gamma, truncnorm, beta
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
    
     CORRECCIONES APLICADAS:
    - update_assignments: Usa correctamente w[i, candidates] * likes
    - predict_mean: Usa Φ (ndtr) no expit
    - predict_density: Escalas consistentes
    - Inicialización robusta con clipping
    """
    
    def __init__(self, y, X, H=20, 
                 mu_prior=(0.0, 0.5),           
                 mu0_prior=(0.0, 5.0),          
                 kappa0_prior=(2.0, 0.2),       
                 a0_prior=(2.5, 0.4),           
                 b0_prior=(2.0, 0.3),           
                 psi_prior_mean=None, 
                 psi_prior_prec=None, 
                 mu_psi_hyperprior=(0.0, 10.0), 
                 tau_psi_hyperprior=(1.5, 0.1), 
                 estimate_psi_hyperpriors=True, 
                 kappa_prior=(2.0, 1.0),        
                 n_grid=50, 
                 verbose=True, 
                 use_cpp=True, 
                 psi_positive=True):
        
        self.y = np.array(y).flatten()
        self.X = np.array(X)
        if self.X.ndim == 1:
            self.X = self.X.reshape(-1, 1)
        self.n, self.p = self.X.shape
        self.H = H
        self.verbose = verbose
        self.use_cpp = use_cpp and CPP_AVAILABLE
        self.psi_positive = psi_positive
        
        # Normalizar datos
        self.y_mean = np.mean(y)
        self.y_std = np.std(y)
        if self.y_std < 1e-10:
            self.y_std = 1.0
        self.y_normalized = (self.y - self.y_mean) / self.y_std
        
        self.X_mean = np.mean(X, axis=0)
        self.X_std = np.std(X, axis=0)
        self.X_std[self.X_std < 1e-10] = 1.0
        self.X_normalized = (self.X - self.X_mean) / self.X_std
        
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
        self.a_kappa, self.b_kappa = kappa_prior
        
        # Grilla
        self.n_grid = n_grid
        self.ell_grid = self._create_grid()
        
        # MH
        self.mh_scales = {'kappa0': 0.2, 'a0': 0.2, 'alpha': 0.1, 'psi': 0.1}
        self.mh_acceptance = {'kappa0': [], 'a0': [], 'alpha': [], 'psi': []}
        
        # Storage
        self.trace = {
            'mu': [], 'mu0': [], 'kappa0': [], 'a0': [], 'b0': [],
            'z': [], 'theta_mu': [], 'theta_sigma2': [], 'w': [],
            'n_clusters': [], 'alpha': [], 'psi': [], 'gamma': [],
            'kappa': [], 'ell': [], 'n_active_vars': []
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
        
        # Parámetros de dependencia
        self.alpha = np.random.normal(self.mu, 1.0, size=self.H)
        self.kappa = beta.rvs(self.a_kappa, self.b_kappa, size=self.p)
        self.gamma = np.random.binomial(1, self.kappa[np.newaxis, :], size=(self.H, self.p))
        
        self.psi = np.zeros((self.H, self.p))
        for h in range(self.H):
            for j in range(self.p):
                if self.gamma[h, j] == 1:
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
        
        # Variables latentes
        self.u_latent = np.zeros((self.n, self.H))
        self._update_u_latent()
    
    def _compute_eta(self, X_batch=None):
        """η_h(x) = α_h - Σ_j γ_{hj}·ψ_{hj}·|x_j - ℓ_{hj}|"""
        if X_batch is None:
            X_batch = self.X_normalized
        
        if self.use_cpp:
            result = psbp_cpp.compute_eta(X_batch, self.alpha, self.psi, self.gamma, self.ell, self.ell_grid)
            return np.array(result.eta)
        
        n_batch = X_batch.shape[0]
        eta = np.zeros((n_batch, self.H))
        for h in range(self.H):
            eta[:, h] = self.alpha[h]
            for j in range(self.p):
                if self.gamma[h, j] == 1:
                    ell_hj_value = self.ell_grid[j, self.ell[h, j]]
                    dist = np.abs(X_batch[:, j] - ell_hj_value)
                    eta[:, h] -= self.psi[h, j] * dist
        return eta
    
    def _compute_weights(self, X_batch=None):
        """✅ CRÍTICO: Usa Φ (ndtr) no expit"""
        if X_batch is None:
            X_batch = self.X_normalized
        
        if self.use_cpp:
            result = psbp_cpp.compute_weights_probit(X_batch, self.alpha, self.psi, 
                                                     self.gamma, self.ell, self.ell_grid)
            return np.array(result.weights)
        
        eta = self._compute_eta(X_batch)
        v = ndtr(eta)  # ✅ Φ(η), NO expit
        
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
        """Data augmentation: u_{ih} ~ TruncatedNormal"""
        if self.use_cpp:
            result = psbp_cpp.update_u_latent(self.u_latent, self.z, self.X_normalized, 
                                             self.alpha, self.psi, self.gamma, self.ell, self.ell_grid)
            self.u_latent = np.array(result.u_latent)
        else:
            eta = self._compute_eta(self.X_normalized)
            for i in range(self.n):
                z_i = self.z[i]
                for h in range(self.H):
                    if h > z_i:
                        continue
                    eta_ih = eta[i, h]
                    if z_i == h:
                        a = (0 - eta_ih) / 1.0
                        self.u_latent[i, h] = truncnorm.rvs(a, np.inf, loc=eta_ih, scale=1.0)
                    elif z_i > h:
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
        gamma_new = np.random.binomial(1, self.kappa[np.newaxis, :], size=(n_new, self.p))
        self.gamma = np.vstack([self.gamma, gamma_new])
        
        psi_new = np.zeros((n_new, self.p))
        for h in range(n_new):
            for j in range(self.p):
                if gamma_new[h, j] == 1:
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
        """✅ CORRECCIÓN: Usa w[i, candidates] * likes"""
        if self.use_cpp:
            self.z = np.array(psbp_cpp.update_assignments(u, self.w, self.y_normalized,
                             self.theta_mu, self.theta_sigma2, self.z))
        else:
            for i in range(self.n):
                candidates = np.where(self.w[i, :] > u[i])[0]
                if len(candidates) == 0:
                    candidates = np.arange(self.H)
                
                likes = np.clip(norm.pdf(self.y_normalized[i], self.theta_mu[candidates],
                               np.sqrt(self.theta_sigma2[candidates])), 1e-300, None)
                probs = self.w[i, candidates] * likes  # ✅ CORRECCIÓN
                probs = probs / probs.sum() if probs.sum() > 1e-300 else np.ones_like(likes) / len(likes)
                self.z[i] = candidates[np.random.choice(len(candidates), p=probs)]
    
    def update_atoms(self):
        if self.use_cpp:
            result = psbp_cpp.update_atoms(self.z, self.y_normalized, self.theta_mu, self.theta_sigma2,
                                          self.mu0, self.kappa0, self.a0, self.b0, self.H)
            self.theta_mu = np.array(result.theta_mu)
            self.theta_sigma2 = np.array(result.theta_sigma2)
        else:
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
        if self.use_cpp:
            result = psbp_cpp.update_alpha_probit(
                self.alpha, self.u_latent, self.X_normalized, 
                self.psi, self.gamma, self.ell, self.ell_grid, 
                self.z,  # ✅ AGREGAR ESTO
                self.mu, self.mh_scales['alpha'], self.H
            )
            self.alpha = np.array(result.alpha)
            self.mh_acceptance['alpha'].extend(result.acceptances)
        else:
            for h in range(self.H):
                affected_idx = np.where(self.z >= h)[0]
                if len(affected_idx) == 0:
                    self.alpha[h] = np.random.normal(self.mu, 1.0)
                    continue
                
                correction = np.zeros(len(affected_idx))
                for j in range(self.p):
                    if self.gamma[h, j] == 1:
                        ell_hj_value = self.ell_grid[j, self.ell[h, j]]
                        dist = np.abs(self.X_normalized[affected_idx, j] - ell_hj_value)
                        correction += self.psi[h, j] * dist
                
                u_affected = self.u_latent[affected_idx, h]
                n_affected = len(affected_idx)
                tau_post = n_affected + 1.0
                mu_post = (np.sum(u_affected + correction) + self.mu) / tau_post
                self.alpha[h] = np.random.normal(mu_post, 1.0 / np.sqrt(tau_post))
    
    def update_psi_gamma(self):
        if self.use_cpp:
            result = psbp_cpp.update_psi_gamma(
            self.psi, self.gamma, self.mu_psi, self.tau_psi, 
            self.kappa, self.u_latent, self.X_normalized, 
            self.alpha, self.ell, self.ell_grid, 
            self.z,  # ✅ AGREGAR ESTO
            self.mh_scales['psi'], self.psi_positive, self.H
            )
            self.psi = np.array(result.psi)
            self.gamma = np.array(result.gamma)
            if hasattr(result, 'acceptances_psi'):
                self.mh_acceptance['psi'].extend(np.array(result.acceptances_psi).flatten().tolist())
        else:
            for h in range(self.H):
                for j in range(self.p):
                    # Actualizar gamma
                    log_prior_1 = np.log(self.kappa[j] + 1e-10)
                    log_prior_0 = np.log(1 - self.kappa[j] + 1e-10)
                    log_like_1 = -0.5 * self.tau_psi[j] * (self.psi[h, j] - self.mu_psi[j])**2 if (self.psi[h, j] != 0 or self.gamma[h, j] == 1) else 0.0
                    p_gamma_1 = 1.0 / (1.0 + np.exp(-((log_prior_1 + log_like_1) - log_prior_0)))
                    self.gamma[h, j] = np.random.binomial(1, p_gamma_1)
                    
                    if self.gamma[h, j] == 0:
                        self.psi[h, j] = 0.0
                        continue
                    
                    # Actualizar psi (MH)
                    psi_curr = self.psi[h, j]
                    psi_prop = np.random.normal(psi_curr, self.mh_scales['psi'])
                    if self.psi_positive and psi_prop < 0:
                        self.mh_acceptance['psi'].append(False)
                        continue
                    
                    affected_idx = np.where(self.z >= h)[0]
                    if len(affected_idx) == 0:
                        log_r = -0.5 * self.tau_psi[j] * ((psi_prop - self.mu_psi[j])**2 - (psi_curr - self.mu_psi[j])**2)
                    else:
                        ell_hj_value = self.ell_grid[j, self.ell[h, j]]
                        dist = np.abs(self.X_normalized[affected_idx, j] - ell_hj_value)
                        eta_base = self.alpha[h]
                        for jj in range(self.p):
                            if jj != j and self.gamma[h, jj] == 1:
                                eta_base -= self.psi[h, jj] * np.abs(self.X_normalized[affected_idx, jj] - self.ell_grid[jj, self.ell[h, jj]])
                        
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
        if self.use_cpp:
            result = psbp_cpp.update_ell(self.ell, self.z, self.X_normalized, self.alpha, 
                                        self.psi, self.gamma, self.u_latent, self.ell_grid, self.H, self.n_grid)
            self.ell = np.array(result.ell)
        else:
            for h in range(self.H):
                for j in range(self.p):
                    if self.gamma[h, j] == 0:
                        self.ell[h, j] = np.random.randint(0, self.n_grid)
                        continue
                    
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
                            if jj != j and self.gamma[h, jj] == 1:
                                eta -= self.psi[h, jj] * np.abs(self.X_normalized[affected_idx, jj] - self.ell_grid[jj, self.ell[h, jj]])
                        log_likes[m] = -0.5 * np.sum((self.u_latent[affected_idx, h] - eta)**2)
                    
                    log_likes -= np.max(log_likes)
                    probs = np.exp(log_likes)
                    probs /= probs.sum()
                    self.ell[h, j] = np.random.choice(self.n_grid, p=probs)
    
    def update_kappa(self):
        n_active = np.sum(self.gamma, axis=0)
        a_post = self.a_kappa + n_active
        b_post = self.b_kappa + (self.H - n_active)
        for j in range(self.p):
            self.kappa[j] = beta.rvs(a_post[j], b_post[j])
    
    def update_weights(self):
        self.w = self._compute_weights()
    
    def update_mu(self):
        tau_post = len(self.alpha) + 1.0 / self.tau_mu_inv
        mu_post = (np.sum(self.alpha) + self.mu_mu / self.tau_mu_inv) / tau_post
        self.mu = np.random.normal(mu_post, 1.0 / np.sqrt(tau_post))
    
    def update_mu0(self):
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
        """Actualiza μ_ψ_j (si estimate_psi_hyperpriors=True)"""
        if not self.estimate_psi_hyperpriors:
            return
        
        for j in range(self.p):
            active_mask = self.gamma[:, j] == 1
            psi_active = self.psi[active_mask, j]
            n_active = len(psi_active)
            
            if n_active == 0:
                self.mu_psi[j] = np.random.normal(self.m_psi, np.sqrt(self.s2_psi))
                continue
            
            tau_post = n_active * self.tau_psi[j] + 1.0 / self.s2_psi
            m_post = (self.tau_psi[j] * np.sum(psi_active) + self.m_psi / self.s2_psi) / tau_post
            s_post = 1.0 / np.sqrt(tau_post)
            self.mu_psi[j] = np.random.normal(m_post, s_post)
    
    def update_tau_psi(self):
        """Actualiza τ_ψ_j (si estimate_psi_hyperpriors=True)"""
        if not self.estimate_psi_hyperpriors:
            return
        
        for j in range(self.p):
            active_mask = self.gamma[:, j] == 1
            psi_active = self.psi[active_mask, j]
            n_active = len(psi_active)
            
            if n_active == 0:
                self.tau_psi[j] = gamma.rvs(self.alpha_tau, scale=1.0/self.beta_tau)
                continue
            
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
            self.update_psi_gamma()
            self.update_kappa()
            self.update_ell()
            self.update_weights()
            self.update_mu()
            self.update_mu0()
            self.update_kappa0()
            self.update_a0()
            self.update_b0()
            
            if it >= burnin:
                self.trace['mu'].append(self.mu)
                self.trace['mu0'].append(self.mu0 * self.y_std + self.y_mean)
                self.trace['kappa0'].append(self.kappa0)
                self.trace['a0'].append(self.a0)
                self.trace['b0'].append(self.b0 * (self.y_std ** 2))
                self.trace['z'].append(self.z.copy())
                self.trace['theta_mu'].append((self.theta_mu * self.y_std + self.y_mean).copy())
                self.trace['theta_sigma2'].append((self.theta_sigma2 * (self.y_std ** 2)).copy())
                self.trace['w'].append(self.w.copy())
                self.trace['n_clusters'].append(len(np.unique(self.z)))
                self.trace['alpha'].append(self.alpha.copy())
                self.trace['psi'].append(self.psi.copy())
                self.trace['gamma'].append(self.gamma.copy())
                self.trace['kappa'].append(self.kappa.copy())
                self.trace['ell'].append(self.ell.copy())
                self.trace['n_active_vars'].append(np.sum(self.gamma, axis=0).copy())
            
            if self.verbose and (it + 1) % 100 == 0:
                n_clusters = len(np.unique(self.z))    
                print(f"Iter {it+1}/{iterations}: "
                    f"K_eff={n_clusters}, H={self.H}, "
                    f"μ={self.mu:.2f}, "
                    f"μ₀={(self.mu0 * self.y_std + self.y_mean):.2f}, "
                    f"κ₀={self.kappa0:.2f}, "
                    f"a₀={self.a0:.2f}, "
                    f"b₀={(self.b0 * (self.y_std ** 2)):.2f}")
        return self.trace
    
    def get_posterior_summary(self):
        """Resúmenes posteriores"""
        summary = {
            'mu': (np.mean(self.trace['mu']), np.std(self.trace['mu'])),
            'mu0': (np.mean(self.trace['mu0']), np.std(self.trace['mu0'])),
            'kappa0': (np.mean(self.trace['kappa0']), np.std(self.trace['kappa0'])),
            'a0': (np.mean(self.trace['a0']), np.std(self.trace['a0'])),
            'b0': (np.mean(self.trace['b0']), np.std(self.trace['b0'])),
            'n_clusters': (np.mean(self.trace['n_clusters']), np.std(self.trace['n_clusters'])),
            'kappa': (np.mean(np.array(self.trace['kappa']), axis=0),
                     np.std(np.array(self.trace['kappa']), axis=0)),
            'n_active_vars': (np.mean(np.array(self.trace['n_active_vars']), axis=0),
                             np.std(np.array(self.trace['n_active_vars']), axis=0))
        }
        return summary
    
    def predict_density(self, y_new, X_new, n_samples=100):
        """Densidad predictiva f(y_new | X_new, data)"""
        n_new = X_new.shape[0]
        y_grid = np.array(y_new)
        density = np.zeros((len(y_grid), n_new))
        
        X_new_norm = (X_new - self.X_mean) / self.X_std
        n_post = len(self.trace['z'])
        indices = np.linspace(0, n_post - 1, min(n_samples, n_post), dtype=int)
        
        for idx in indices:
            H_sample = len(self.trace['alpha'][idx])
            eta = np.zeros((n_new, H_sample))
            
            for h in range(H_sample):
                eta[:, h] = self.trace['alpha'][idx][h]
                for j in range(self.p):
                    if self.trace['gamma'][idx][h, j] == 1:
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
        X_new_norm = (X_new - self.X_mean) / self.X_std
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
                    if self.trace['gamma'][idx][h, j] == 1:
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
        X_new_norm = (X_new - self.X_mean) / self.X_std
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
                    if self.trace['gamma'][post_idx][h, j] == 1:
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
        """✅ Usa Φ (ndtr) no expit"""
        n_new = X_new.shape[0]
        X_new_norm = (X_new - self.X_mean) / self.X_std
        n_post = len(self.trace['z'])
        indices = np.linspace(0, n_post - 1, min(n_samples, n_post), dtype=int)
        predictions_mean = np.zeros((len(indices), n_new))
        predictions_var = np.zeros((len(indices), n_new))
        
        for s, idx in enumerate(indices):
            eta = np.zeros((n_new, len(self.trace['alpha'][idx])))
            for h in range(len(self.trace['alpha'][idx])):
                eta[:, h] = self.trace['alpha'][idx][h]
                for j in range(self.p):
                    if self.trace['gamma'][idx][h, j] == 1:
                        eta[:, h] -= self.trace['psi'][idx][h, j] * np.abs(X_new_norm[:, j] - self.ell_grid[j, self.trace['ell'][idx][h, j]])
            
            v = ndtr(eta)  # ✅ Φ no expit
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
