import numpy as np
from scipy.stats import norm, invgamma, gamma, truncnorm
from scipy.special import expit
import math

# Importar el módulo C++
try:
    from . import lsbp_cpp
    CPP_AVAILABLE = True
    print("Implementacion en C++ Exitosa")
except ImportError as e:
    CPP_AVAILABLE = False
    _IMPORT_ERROR = e
    print("Implementacion en C++ Fallida")


class LSBPNormal:
    """
    Logit Stick-Breaking Process (LSBP) con kernel Normal-Inversa-Gamma.
    
    Proceso de Dirichlet Dependiente donde los pesos de la mezcla varían
    con las covariables mediante un enlace logit y función kernel de dependencia.
    
    Modelo:
    -------
    y_i | z_i=h, μ_h, σ²_h ~ N(μ_h, σ²_h)
    z_i | x_i ~ Categorical(w_1(x_i), ..., w_T(x_i))
    
    w_h(x) = v_h(x) ∏_{ℓ<h} (1 - v_ℓ(x))
    v_h(x) = logit⁻¹(η_h(x)) = exp(η_h(x)) / [1 + exp(η_h(x))]
    η_h(x) = α_h - Σ_j ψ_{hj} |x_j - ℓ_{hj}|
    
    Usa Slice Sampling con todos los hiperparámetros aleatorios.
    """
    
    def __init__(self, y, X, H=20,
                 mu_prior=(0.0, 1.0),
                 mu0_prior=(0.0, 100.0),
                 kappa0_prior=(2.0, 1.0),
                 a0_prior=(3.0, 1.0),
                 b0_prior=(2.0, 1.0),
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
            print("Using C++ acceleration")
        
        self.y_mean = np.mean(y)
        self.y_std = np.std(y)
        self.y_normalized = (self.y - self.y_mean) / self.y_std
        
        self.X_mean = np.mean(X, axis=0)
        self.X_std = np.std(X, axis=0)
        self.X_normalized = (self.X - self.X_mean) / self.X_std
        
        self.mu_mu, self.tau_mu_inv = mu_prior
        self.m0, self.s02 = mu0_prior
        self.alpha_kappa, self.beta_kappa = kappa0_prior
        self.alpha_a, self.beta_a = a0_prior
        self.alpha_b, self.beta_b = b0_prior
        self.mu_psi, self.tau_psi_inv = psi_prior
        
        self.n_grid = n_grid
        self.ell_grid = self._create_grid()
        
        self.mh_scales = {
            'alpha': 0.3,
            'psi': 0.2,
            'kappa0': 0.2,
            'a0': 0.2
        }
        self.mh_acceptance = {
            'alpha': [],
            'psi': [],
            'kappa0': [],
            'a0': []
        }
        
        self.trace = {
            'mu': [], 'mu0': [], 'kappa0': [], 'a0': [], 'b0': [],
            'z': [], 'theta_mu': [], 'theta_sigma2': [], 'w': [], 
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
        self.kappa0 = np.random.gamma(self.alpha_kappa, 1.0/self.beta_kappa)
        self.kappa0 = np.clip(self.kappa0, 0.1, 10.0)
        
        self.a0 = np.random.gamma(self.alpha_a, 1.0/self.beta_a)
        self.a0 = np.clip(self.a0, 1.0, 10.0)
        
        self.b0 = np.random.gamma(self.alpha_b, 1.0/self.beta_b)
        self.b0 = np.clip(self.b0, 0.1, 10.0)
        
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
        self.theta_sigma2 = np.zeros(self.H)
        for h in range(self.H):
            self.theta_sigma2[h] = invgamma.rvs(self.a0, scale=self.b0)
            self.theta_sigma2[h] = np.clip(self.theta_sigma2[h], 0.01, 100.0)
            self.theta_mu[h] = np.random.normal(
                self.mu0, 
                math.sqrt(self.theta_sigma2[h] / self.kappa0)
            )
        
        self.z = np.zeros(self.n, dtype=int)
        for i in range(self.n):
            likes = np.array([
                norm.pdf(self.y_normalized[i], self.theta_mu[h], 
                        math.sqrt(self.theta_sigma2[h])) 
                for h in range(self.H)
            ])
            likes = np.clip(likes, 1e-300, None)
            probs = self.w[i, :] * likes
            probs /= probs.sum()
            self.z[i] = np.random.choice(self.H, p=probs)
    
    def _compute_eta(self, X_batch):
        if self.use_cpp:
            result = lsbp_cpp.compute_eta(
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
        if self.use_cpp:
            result = lsbp_cpp.compute_weights(
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
            theta_sigma2_new = np.zeros(5)
            for h in range(5):
                theta_sigma2_new[h] = invgamma.rvs(self.a0, scale=self.b0)
                theta_sigma2_new[h] = np.clip(theta_sigma2_new[h], 0.01, 100.0)
                theta_mu_new[h] = np.random.normal(
                    self.mu0,
                    math.sqrt(theta_sigma2_new[h] / self.kappa0)
                )
            self.theta_mu = np.append(self.theta_mu, theta_mu_new)
            self.theta_sigma2 = np.append(self.theta_sigma2, theta_sigma2_new)
            
            self.H = H_new
            self.w = self._compute_weights()
            break
        
        return u
    
    def update_assignments(self, u):
        if self.use_cpp:
            self.z = np.array(lsbp_cpp.update_assignments(
                u,
                self.w,
                self.y_normalized,
                self.theta_mu,
                self.theta_sigma2,
                self.z
            ))
        else:
            for i in range(self.n):
                candidates = np.where(self.w[i, :] > u[i])[0]
                
                if len(candidates) == 0:
                    candidates = np.array([0])
                
                likes = norm.pdf(self.y_normalized[i], 
                               self.theta_mu[candidates],
                               np.sqrt(self.theta_sigma2[candidates]))
                likes = np.clip(likes, 1e-300, None)
                
                probs = likes / likes.sum() if likes.sum() > 0 else np.ones_like(likes) / len(likes)
                
                self.z[i] = np.random.choice(candidates, p=probs)
    
    def update_atoms(self):
        if self.use_cpp:
            result = lsbp_cpp.update_atoms(
                self.z,
                self.y_normalized,
                self.theta_mu,
                self.theta_sigma2,
                self.mu0,
                self.kappa0,
                self.a0,
                self.b0,
                self.H
            )
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
                    b_n = (self.b0 + 0.5 * ss + 
                           (self.kappa0 * n_h * (y_bar - self.mu0)**2) / (2 * kappa_n))
                    
                    self.theta_sigma2[h] = invgamma.rvs(a_n, scale=b_n)
                    self.theta_sigma2[h] = np.clip(self.theta_sigma2[h], 0.01, 100.0)
                    
                    self.theta_mu[h] = np.random.normal(
                        mu_n, 
                        math.sqrt(self.theta_sigma2[h] / kappa_n)
                    )
                else:
                    self.theta_sigma2[h] = invgamma.rvs(self.a0, scale=self.b0)
                    self.theta_sigma2[h] = np.clip(self.theta_sigma2[h], 0.01, 100.0)
                    self.theta_mu[h] = np.random.normal(
                        self.mu0,
                        math.sqrt(self.theta_sigma2[h] / self.kappa0)
                    )
    
    def update_alpha(self):
        if self.use_cpp:
            result = lsbp_cpp.update_alpha(
                self.alpha,
                self.z,
                self.X_normalized,
                self.psi,
                self.ell,
                self.ell_grid,
                self.mu,
                self.mh_scales['alpha'],
                self.H
            )
            self.alpha = np.array(result.alpha)
            # Registrar aceptaciones
            self.mh_acceptance['alpha'].extend(result.acceptances)
        else:
            # Código Python original (sin cambios)
            for h in range(self.H - 1):
                alpha_prop = np.random.normal(self.alpha[h], self.mh_scales['alpha'])
                affected = np.where(self.z >= h)[0]

                if len(affected) == 0:
                    log_prior_curr = -0.5 * ((self.alpha[h] - self.mu)**2)
                    log_prior_prop = -0.5 * ((alpha_prop - self.mu)**2)
                    log_r = log_prior_prop - log_prior_curr
                else:
                    eta_curr = self._compute_eta_h(self.X_normalized[affected], h, 
                                                    self.alpha[h])
                    eta_prop = self._compute_eta_h(self.X_normalized[affected], h, 
                                                    alpha_prop)

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

                    log_prior_curr = -0.5 * ((self.alpha[h] - self.mu)**2)
                    log_prior_prop = -0.5 * ((alpha_prop - self.mu)**2)

                    log_r = (log_like_prop + log_prior_prop) - (log_like_curr + log_prior_curr)

                log_r = np.clip(log_r, -50, 50)

                accept = math.log(np.random.rand()) < log_r
                if accept:
                    self.alpha[h] = alpha_prop

                self.mh_acceptance['alpha'].append(float(accept))
    
    def _compute_eta_h(self, X_batch, h, alpha_h_value):
        n_batch = X_batch.shape[0]
        eta = np.full(n_batch, alpha_h_value)
        
        for j in range(self.p):
            ell_hj_value = self.ell_grid[j, self.ell[h, j]]
            dist = np.abs(X_batch[:, j] - ell_hj_value)
            eta -= self.psi[h, j] * dist
        
        return eta
    
    def update_psi(self):
        if self.use_cpp:
            result = lsbp_cpp.update_psi(
                self.psi,
                self.z,
                self.X_normalized,
                self.alpha,
                self.ell,
                self.ell_grid,
                self.mu_psi,
                self.tau_psi_inv,
                self.mh_scales['psi'],
                self.H
            )
            self.psi = np.array(result.psi)
            # Registrar aceptaciones
            self.mh_acceptance['psi'].extend(result.acceptances)
        else:
            # Código Python original (sin cambios)
            for h in range(self.H - 1):
                for j in range(self.p):
                    psi_curr = self.psi[h, j]
                    psi_prop = np.random.normal(psi_curr, self.mh_scales['psi'])
                    
                    if psi_prop < 0:
                        self.mh_acceptance['psi'].append(False)
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
        if self.use_cpp:
            result = lsbp_cpp.update_ell(
                self.ell,
                self.z,
                self.X_normalized,
                self.alpha,
                self.psi,
                self.ell_grid,
                self.H,
                self.n_grid
            )
            self.ell = np.array(result.ell)
        else:
            # Código Python original (sin cambios)
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
        self.w = self._compute_weights()
    
    def update_mu(self):
        tau_post = len(self.alpha) + 1.0 / self.tau_mu_inv
        mu_post = (np.sum(self.alpha) + self.mu_mu / self.tau_mu_inv) / tau_post
        
        self.mu = np.random.normal(mu_post, 1.0 / np.sqrt(tau_post))
        self.mu = np.clip(self.mu, -10, 10)
    
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
        
        s0n2 = np.clip(s0n2, 1e-6, 1e6)
        m0n = np.clip(m0n, -100, 100)
        
        self.mu0 = np.random.normal(m0n, math.sqrt(s0n2))
        self.mu0 = np.clip(self.mu0, -50, 50)
    
    def update_kappa0(self):
        log_kappa = math.log(self.kappa0)
        log_kappa_prop = np.random.normal(log_kappa, self.mh_scales['kappa0'])
        kappa_prop = math.exp(log_kappa_prop)
        kappa_prop = np.clip(kappa_prop, 0.01, 100.0)
        
        active_clusters = np.unique(self.z)
        if len(active_clusters) == 0:
            return
        
        mu_active = self.theta_mu[active_clusters]
        sigma2_active = self.theta_sigma2[active_clusters]
        
        diff_sq = (mu_active - self.mu0)**2
        inv_sigma2 = np.clip(1.0 / sigma2_active, 1e-6, 1e6)
        
        log_like_curr = (0.5 * len(active_clusters) * np.log(self.kappa0) - 
                        0.5 * self.kappa0 * np.sum(diff_sq * inv_sigma2))
        log_like_prop = (0.5 * len(active_clusters) * np.log(kappa_prop) - 
                        0.5 * kappa_prop * np.sum(diff_sq * inv_sigma2))
        
        log_prior_curr = ((self.alpha_kappa - 1) * math.log(self.kappa0) - 
                         self.beta_kappa * self.kappa0)
        log_prior_prop = ((self.alpha_kappa - 1) * math.log(kappa_prop) - 
                         self.beta_kappa * kappa_prop)
        
        log_r = (log_like_prop + log_prior_prop) - (log_like_curr + log_prior_curr)
        log_r = np.clip(log_r, -50, 50)
        
        accept = math.log(np.random.rand()) < log_r
        if accept:
            self.kappa0 = kappa_prop
        
        self.mh_acceptance['kappa0'].append(float(accept))
    
    def update_a0(self):
        log_a = math.log(self.a0)
        log_a_prop = np.random.normal(log_a, self.mh_scales['a0'])
        a_prop = math.exp(log_a_prop)
        a_prop = np.clip(a_prop, 0.5, 20.0)
        
        active_clusters = np.unique(self.z)
        if len(active_clusters) == 0:
            return
        
        sigma2_active = self.theta_sigma2[active_clusters]
        
        ratio = np.clip(self.b0 / sigma2_active, 1e-10, 1e10)
        log_ratio = np.log(ratio)
        
        K = len(active_clusters)
        log_like_curr = (self.a0 * np.sum(log_ratio) - 
                        K * math.lgamma(self.a0) - 
                        (self.a0 + 1) * np.sum(np.log(sigma2_active)))
        log_like_prop = (a_prop * np.sum(log_ratio) - 
                        K * math.lgamma(a_prop) - 
                        (a_prop + 1) * np.sum(np.log(sigma2_active)))
        
        log_prior_curr = (self.alpha_a - 1) * math.log(self.a0) - self.beta_a * self.a0
        log_prior_prop = (self.alpha_a - 1) * math.log(a_prop) - self.beta_a * a_prop
        
        log_r = (log_like_prop + log_prior_prop) - (log_like_curr + log_prior_curr)
        log_r = np.clip(log_r, -50, 50)
        
        accept = math.log(np.random.rand()) < log_r
        if accept:
            self.a0 = a_prop
        
        self.mh_acceptance['a0'].append(float(accept))
    
    def update_b0(self):
        active_clusters = np.unique(self.z)
        if len(active_clusters) == 0:
            return
        
        sigma2_active = self.theta_sigma2[active_clusters]
        
        alpha_post = self.alpha_b + len(active_clusters) * self.a0
        inv_sigma2 = np.clip(1.0 / sigma2_active, 1e-6, 1e6)
        beta_post = self.beta_b + np.sum(inv_sigma2)
        
        self.b0 = np.random.gamma(alpha_post, 1.0 / beta_post)
        self.b0 = np.clip(self.b0, 0.01, 100.0)
    
    def adapt_mh_scales(self, iteration):
        if iteration > 50 and iteration % 50 == 0:
            for param in ['alpha', 'psi', 'kappa0', 'a0']:
                if len(self.mh_acceptance[param]) > 0:
                    recent = self.mh_acceptance[param][-50:]
                    acc_rate = np.mean(recent)
                    
                    if acc_rate < 0.15:
                        self.mh_scales[param] *= 0.8
                    elif acc_rate > 0.4:
                        self.mh_scales[param] *= 1.2
                    
                    self.mh_scales[param] = np.clip(self.mh_scales[param], 0.01, 1.0)
    
    def run(self, iterations=1000, burnin=500):
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
            self.update_kappa0()
            self.update_a0()
            self.update_b0()
            
            if it < burnin:
                self.adapt_mh_scales(it)
            
            if it >= burnin:
                active_clusters = len(np.unique(self.z))
                
                mu_original = self.theta_mu * self.y_std + self.y_mean
                sigma2_original = self.theta_sigma2 * (self.y_std ** 2)
                
                self.trace['mu'].append(self.mu)
                self.trace['mu0'].append(self.mu0 * self.y_std + self.y_mean)
                self.trace['kappa0'].append(self.kappa0)
                self.trace['a0'].append(self.a0)
                self.trace['b0'].append(self.b0 * (self.y_std ** 2))
                self.trace['z'].append(self.z.copy())
                self.trace['theta_mu'].append(mu_original.copy())
                self.trace['theta_sigma2'].append(sigma2_original.copy())
                self.trace['w'].append(self.w.copy())
                self.trace['n_clusters'].append(active_clusters)
                self.trace['alpha'].append(self.alpha.copy())
                self.trace['psi'].append(self.psi.copy())
                self.trace['ell'].append(self.ell.copy())
            
            if self.verbose and (it + 1) % 100 == 0:
                active = len(np.unique(self.z))
                acc_alpha = np.mean(self.mh_acceptance['alpha'][-100:]) if len(self.mh_acceptance['alpha']) >= 100 else 0
                acc_psi = np.mean(self.mh_acceptance['psi'][-100:]) if len(self.mh_acceptance['psi']) >= 100 else 0
                acc_k = np.mean(self.mh_acceptance['kappa0'][-100:]) if len(self.mh_acceptance['kappa0']) >= 100 else 0
                acc_a = np.mean(self.mh_acceptance['a0'][-100:]) if len(self.mh_acceptance['a0']) >= 100 else 0
                
                print(f"Iter {it+1}/{iterations}: K_eff={active}, H={self.H}, "
                      f"μ={self.mu:.2f}, μ₀={self.mu0:.2f}, κ₀={self.kappa0:.2f}, "
                      f"a₀={self.a0:.2f}, b₀={self.b0:.2f}")
                print(f"  Acceptance: α={acc_alpha:.2f}, ψ={acc_psi:.2f}, "
                      f"κ={acc_k:.2f}, a={acc_a:.2f}")
        
        return self.trace
    
    def get_posterior_summary(self):
        summary = {
            'mu': (np.mean(self.trace['mu']), np.std(self.trace['mu'])),
            'mu0': (np.mean(self.trace['mu0']), np.std(self.trace['mu0'])),
            'kappa0': (np.mean(self.trace['kappa0']), np.std(self.trace['kappa0'])),
            'a0': (np.mean(self.trace['a0']), np.std(self.trace['a0'])),
            'b0': (np.mean(self.trace['b0']), np.std(self.trace['b0'])),
            'n_clusters': (np.mean(self.trace['n_clusters']), 
                          np.std(self.trace['n_clusters']))
        }
        return summary
    
    def predict_density(self, y_new, X_new, n_samples=100):
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
            theta_sigma2_sample = self.trace['theta_sigma2'][idx]
            
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
            
            for i in range(n_new):
                for y_idx, y_val in enumerate(y_grid):
                    for h in range(H_sample):
                        density[y_idx, i] += (w[i, h] * 
                            norm.pdf(y_val, theta_mu_sample[h], 
                                   np.sqrt(theta_sigma2_sample[h])))
        
        density /= n_samples
        
        return density
    
    def predict_mean(self, X_new, n_samples=100, return_full_uncertainty=True):
        """
        Predice E[Y|X] y la desviación estándar predictiva total.
        
        Parámetros:
        -----------
        X_new : array (n_new, p)
            Covariables para predicción
        
        n_samples : int
            Número de muestras posteriores a usar
        
        return_full_uncertainty : bool
            Si True, devuelve la incertidumbre predictiva total (epistémica + aleatoria)
            Si False, solo devuelve la incertidumbre epistémica (variabilidad de E[Y|X])
        
        Retorna:
        --------
        mean_pred : array (n_new,)
            Media predictiva E[Y|X]
        
        std_pred : array (n_new,)
            Desviación estándar predictiva:
            - Si return_full_uncertainty=True: sqrt(Var[Y|X]) (incertidumbre total)
            - Si return_full_uncertainty=False: std(E[Y|X]) (solo epistémica)
        
        Fórmulas:
        ---------
        E[Y|X] = Σ_h w_h(x) · μ_h
        
        Var[Y|X] = Σ_h w_h(x) · [σ²_h + μ²_h] - [E[Y|X]]²
                 = E[μ²_h + σ²_h] - [E[μ_h]]²  (Ley de Varianza Total)
                 = Var_between + E[Var_within]
        """
        n_new = X_new.shape[0]
        
        # Normalizar covariables
        X_new_norm = (X_new - self.X_mean) / self.X_std
        
        # Seleccionar muestras posteriores
        n_post = len(self.trace['z'])
        indices = np.linspace(0, n_post - 1, n_samples, dtype=int)
        
        # Almacenar predicciones
        predictions_mean = np.zeros((n_samples, n_new))
        predictions_var = np.zeros((n_samples, n_new))
        
        for s, idx in enumerate(indices):
            # Extraer parámetros de la muestra posterior
            alpha_sample = self.trace['alpha'][idx]
            psi_sample = self.trace['psi'][idx]
            ell_sample = self.trace['ell'][idx]
            theta_mu_sample = self.trace['theta_mu'][idx]
            theta_sigma2_sample = self.trace['theta_sigma2'][idx]
            
            H_sample = len(alpha_sample)
            
            # ========================================
            # Calcular pesos w_h(x) para X_new
            # ========================================
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
            
            # Normalizar pesos (por seguridad numérica)
            w = w / w.sum(axis=1, keepdims=True)
            
            # ========================================
            # Calcular E[Y|X] y Var[Y|X] para esta muestra
            # ========================================
            
            # E[Y|X] = Σ_h w_h(x) · μ_h
            mean_sample = np.sum(w * theta_mu_sample[np.newaxis, :H_sample], axis=1)
            
            # Var[Y|X] = Σ_h w_h(x) · [σ²_h + μ²_h] - [E[Y|X]]²
            second_moment = np.sum(
                w * (theta_sigma2_sample[np.newaxis, :H_sample] + 
                     theta_mu_sample[np.newaxis, :H_sample]**2), 
                axis=1
            )
            var_sample = second_moment - mean_sample**2
            
            # Asegurar varianza no negativa (por errores numéricos)
            var_sample = np.maximum(var_sample, 1e-8)
            
            predictions_mean[s, :] = mean_sample
            predictions_var[s, :] = var_sample
        
        # ========================================
        # Agregación sobre muestras posteriores
        # ========================================
        
        # Media predictiva (promedio de E[Y|X] sobre el posterior)
        mean_pred = np.mean(predictions_mean, axis=0)
        
        if return_full_uncertainty:
            # Incertidumbre total: promedio de Var[Y|X] + varianza de E[Y|X]
            # Var_total = E[Var[Y|X]] + Var[E[Y|X]]
            var_within = np.mean(predictions_var, axis=0)  # E[Var[Y|X]]
            var_between = np.var(predictions_mean, axis=0)  # Var[E[Y|X]]
            
            total_var = var_within + var_between
            std_pred = np.sqrt(total_var)
        else:
            # Solo incertidumbre epistémica (variabilidad de E[Y|X])
            std_pred = np.std(predictions_mean, axis=0)
        
        return mean_pred, std_pred


    def predict_with_decomposition(self, X_new, n_samples=100):
        """
        Versión extendida que devuelve descomposición completa de incertidumbre.

        Retorna:
        --------
        results : dict
            - 'mean': E[Y|X]
            - 'std_total': Desviación estándar total
            - 'std_aleatoric': Incertidumbre aleatoria (heterocedasticidad)
            - 'std_epistemic': Incertidumbre epistémica (sobre parámetros)
            - 'var_within': Varianza dentro de clusters E[σ²_h]
            - 'var_between': Varianza entre clusters Var[μ_h]
        """
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
            theta_sigma2_sample = self.trace['theta_sigma2'][idx]

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
                w * (theta_sigma2_sample[np.newaxis, :H_sample] + 
                     theta_mu_sample[np.newaxis, :H_sample]**2), 
                axis=1
            )
            var_sample = second_moment - mean_sample**2
            var_sample = np.maximum(var_sample, 1e-8)

            predictions_mean[s, :] = mean_sample
            predictions_var[s, :] = var_sample

        # Descomposición de incertidumbre
        mean_pred = np.mean(predictions_mean, axis=0)
        var_within = np.mean(predictions_var, axis=0)  # Aleatoria
        var_between = np.var(predictions_mean, axis=0)  # Epistémica

        return {
            'mean': mean_pred,
            'std_total': np.sqrt(var_within + var_between),
            'std_aleatoric': np.sqrt(var_within),
            'std_epistemic': np.sqrt(var_between),
            'var_within': var_within,
            'var_between': var_between
        }


    def predict_quantiles(self, X_new, quantiles=[0.025, 0.25, 0.5, 0.75, 0.975], n_samples=100):
        """
        Predice cuantiles de la distribución predictiva P(Y|X).
        Útil para intervalos de credibilidad sin asumir normalidad.

        Parámetros:
        -----------
        X_new : array (n_new, p)
            Covariables

        quantiles : list
            Cuantiles a calcular (por defecto: 2.5%, 25%, 50%, 75%, 97.5%)

        n_samples : int
            Muestras posteriores

        Retorna:
        --------
        results : dict
            Diccionario con claves: 'mean', 'q_0.025', 'q_0.25', etc.
        """
        n_new = X_new.shape[0]
        X_new_norm = (X_new - self.X_mean) / self.X_std

        n_post = len(self.trace['z'])
        indices = np.linspace(0, n_post - 1, n_samples, dtype=int)

        # Simular de la distribución predictiva
        n_pred_samples = 1000
        y_samples = np.zeros((n_pred_samples, n_new))

        for i in range(n_pred_samples):
            # Elegir una muestra posterior aleatoria
            post_idx = np.random.choice(indices)

            alpha_sample = self.trace['alpha'][post_idx]
            psi_sample = self.trace['psi'][post_idx]
            ell_sample = self.trace['ell'][post_idx]
            theta_mu_sample = self.trace['theta_mu'][post_idx]
            theta_sigma2_sample = self.trace['theta_sigma2'][post_idx]

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

            for k in range(n_new):
                remaining = 1.0
                for h in range(H_sample):
                    w[k, h] = v[k, h] * remaining
                    remaining *= (1 - v[k, h])

            w = w / w.sum(axis=1, keepdims=True)

            # Para cada observación, muestrear de la mezcla
            for k in range(n_new):
                # Elegir cluster según w[k, :]
                cluster = np.random.choice(H_sample, p=w[k, :])
                # Muestrear Y de ese cluster
                y_samples[i, k] = np.random.normal(
                    theta_mu_sample[cluster],
                    np.sqrt(theta_sigma2_sample[cluster])
                )

        # Calcular cuantiles
        results = {'mean': np.mean(y_samples, axis=0)}
        for q in quantiles:
            results[f'q_{q}'] = np.quantile(y_samples, q, axis=0)

        return results