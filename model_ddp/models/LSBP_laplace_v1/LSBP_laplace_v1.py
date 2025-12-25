import numpy as np
from scipy.stats import norm, gamma, truncnorm, expon, laplace
from scipy.special import expit
import math

class LSBPLaplace:
    """
    Logit Stick-Breaking Process (LSBP) con kernel Laplace.
    
    Usa representación de mezcla de escala para mantener conjugación:
    Laplace(y | μ, b) = ∫ N(y | μ, λ) Exp(λ | 2b²) dλ
    
    Modelo:
    -------
    y_i | z_i=h, μ_h, b_h ~ Laplace(μ_h, b_h)
    z_i | x_i ~ Categorical(w_1(x_i), ..., w_T(x_i))
    
    w_h(x) = v_h(x) ∏_{ℓ<h} (1 - v_ℓ(x))
    v_h(x) = logit⁻¹(η_h(x))
    η_h(x) = α_h - Σ_j ψ_{hj} |x_j - ℓ_{hj}|
    
    Medida base G₀ (conjugada vía mezcla de escala):
    ------------------------------------------------
    b_h ~ Gamma(a₀, β₀)           [escala Laplace]
    μ_h ~ N(μ₀, τ₀⁻¹)              [localización]
    λ_ih ~ Exp(2b_h²)              [variables latentes de mezcla]
    
    Hiperparámetros:
    ---------------
    μ₀ ~ N(m₀, s₀²)
    τ₀ ~ Gamma(α_τ, β_τ)
    a₀ ~ Gamma(α_a, β_a)
    β₀ ~ Gamma(α_β, β_β)
    """
    
    def __init__(self, y, X, H=20,
                 mu_prior=(0.0, 1.0),           # μ (intercepto stick-breaking)
                 mu0_prior=(0.0, 100.0),        # μ₀ (localización base)
                 tau0_prior=(2.0, 1.0),         # τ₀ (precisión localización)
                 a0_prior=(3.0, 1.0),           # a₀ (shape escala)
                 beta0_prior=(2.0, 1.0),        # β₀ (rate escala)
                 psi_prior=(0.0, 1.0),          # ψ_{hj} (decaimiento kernel)
                 n_grid=50,
                 verbose=True):
        """
        Parámetros:
        -----------
        y : array (n,)
            Respuesta observada
        X : array (n, p)
            Matriz de covariables
        H : int
            Truncamiento inicial
        mu_prior : tuple
            (μ_μ, τ⁻¹_μ) para prior de μ
        mu0_prior : tuple
            (m₀, s₀²) para prior de μ₀
        tau0_prior : tuple
            (α_τ, β_τ) para prior de τ₀
        a0_prior : tuple
            (α_a, β_a) para prior de a₀
        beta0_prior : tuple
            (α_β, β_β) para prior de β₀
        psi_prior : tuple
            (μ_ψ, τ⁻¹_ψ) para prior de ψ_{hj}
        n_grid : int
            Puntos en grilla para ℓ_{hj}
        """
        self.y = np.array(y)
        self.X = np.array(X)
        self.n, self.p = self.X.shape
        self.H = H
        self.verbose = verbose
        
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
        
        # Grilla para centros
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
        """Crea grilla uniforme para cada predictor"""
        grid = np.zeros((self.p, self.n_grid))
        for j in range(self.p):
            x_min = self.X_normalized[:, j].min() - 0.5
            x_max = self.X_normalized[:, j].max() + 0.5
            grid[j, :] = np.linspace(x_min, x_max, self.n_grid)
        return grid
    
    def initialize(self):
        """Inicializa todos los parámetros"""
        
        # Hiperparámetro μ (intercepto stick-breaking)
        self.mu = np.random.normal(self.mu_mu, np.sqrt(self.tau_mu_inv))
        
        # Hiperparámetros de G₀
        self.mu0 = np.random.normal(0, 1)
        self.tau0 = np.random.gamma(self.alpha_tau, 1.0/self.beta_tau)
        self.tau0 = np.clip(self.tau0, 0.1, 10.0)
        
        self.a0 = np.random.gamma(self.alpha_a, 1.0/self.beta_a)
        self.a0 = np.clip(self.a0, 1.0, 10.0)
        
        self.beta0 = np.random.gamma(self.alpha_beta, 1.0/self.beta_beta)
        self.beta0 = np.clip(self.beta0, 0.1, 10.0)
        
        # Parámetros de dependencia
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
        
        # Calcular pesos
        self.w = self._compute_weights()
        
        # Átomos θ_h = (μ_h, b_h)
        self.theta_mu = np.zeros(self.H)
        self.theta_b = np.zeros(self.H)
        for h in range(self.H):
            self.theta_b[h] = np.random.gamma(self.a0, 1.0/self.beta0)
            self.theta_b[h] = np.clip(self.theta_b[h], 0.01, 100.0)
            self.theta_mu[h] = np.random.normal(self.mu0, 1.0/np.sqrt(self.tau0))
        
        # Variables latentes de mezcla λ_ih para representación Gaussian
        # λ_ih ~ Exp(2b_h²)
        self.lambda_latent = np.ones((self.n, self.H))
        
        # Asignaciones iniciales
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
        
        # Inicializar λ dado z
        self._update_lambda_latent()
    
    def _compute_eta(self, X_batch):
        """Calcula η_h(x) = α_h - Σ_j ψ_{hj} |x_j - ℓ_{hj}|"""
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
        """Calcula pesos w_h(x_i) mediante logit stick-breaking"""
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
        """
        Actualizar variables latentes λ_ih de la representación de mezcla.
        
        Para y_i asignado a cluster h:
        λ_ih | y_i, μ_h, b_h ~ InverseGaussian(μ=b_h/|y_i - μ_h|, λ=b_h²)
        
        Usamos aproximación: λ_ih ~ Exp(2b_h²) truncada y ajustada
        """
        for i in range(self.n):
            h = self.z[i]
            residual = abs(self.y_normalized[i] - self.theta_mu[h])
            
            # Parámetro de la exponencial
            rate = 2 * self.theta_b[h]**2
            
            # Muestrear de Exponencial
            self.lambda_latent[i, h] = expon.rvs(scale=1.0/rate)
            self.lambda_latent[i, h] = np.clip(self.lambda_latent[i, h], 0.001, 100.0)
    
    def sample_slice_variables(self):
        """Paso 1: Generar variables u_i ~ Uniform(0, w_{z_i}(x_i))"""
        u = np.zeros(self.n)
        for i in range(self.n):
            u[i] = np.random.uniform(0, self.w[i, self.z[i]])
        
        u_min = u.min()
        while self.H < 100:
            w_min = self.w.min(axis=0)
            if np.all(w_min < u_min):
                break
            
            # Expandir
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
            
            # Expandir λ_latent
            lambda_new = np.ones((self.n, 5))
            self.lambda_latent = np.hstack([self.lambda_latent, lambda_new])
            
            self.H = H_new
            self.w = self._compute_weights()
            break
        
        return u
    
    def update_assignments(self, u):
        """Paso 2: Actualizar z_i dado u_i"""
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
        """
        Paso 3: Actualizar θ_h = (μ_h, b_h)
        
        Usando representación de mezcla de escala:
        - μ_h | y, b_h, λ ~ N(posterior)
        - b_h | y, μ_h ~ posterior (vía Metropolis-Hastings o Gibbs aproximado)
        """
        for h in range(self.H):
            members_idx = np.where(self.z == h)[0]
            n_h = len(members_idx)
            
            if n_h > 0:
                y_h = self.y_normalized[members_idx]
                
                # Actualizar μ_h usando representación gaussiana
                # y_i | μ_h, λ_ih ~ N(μ_h, λ_ih)
                # μ_h | τ₀, μ₀ ~ N(μ₀, 1/τ₀)
                
                lambda_h = self.lambda_latent[members_idx, h]
                precision_h = 1.0 / lambda_h
                
                tau_post = self.tau0 + np.sum(precision_h)
                mu_post = (self.tau0 * self.mu0 + np.sum(y_h * precision_h)) / tau_post
                
                self.theta_mu[h] = np.random.normal(mu_post, 1.0/np.sqrt(tau_post))
                
                # Actualizar b_h
                # Posterior: b_h | y, μ_h, a₀, β₀
                # Usamos que |y_i - μ_h|/b_h ~ Exp(1) implica
                # Σ|y_i - μ_h|/b_h ~ Gamma(n_h, 1)
                
                residuals = np.abs(y_h - self.theta_mu[h])
                sum_residuals = np.sum(residuals)
                
                # Posterior Gamma
                a_post = self.a0 + n_h
                beta_post = self.beta0 + sum_residuals
                
                self.theta_b[h] = np.random.gamma(a_post, 1.0/beta_post)
                self.theta_b[h] = np.clip(self.theta_b[h], 0.01, 100.0)
            
            else:
                # Cluster vacío: muestrear de G₀
                self.theta_b[h] = np.random.gamma(self.a0, 1.0/self.beta0)
                self.theta_b[h] = np.clip(self.theta_b[h], 0.01, 100.0)
                self.theta_mu[h] = np.random.normal(self.mu0, 1.0/np.sqrt(self.tau0))
        
        # Actualizar variables latentes λ
        self._update_lambda_latent()
    
    def update_alpha(self):
        """Paso 4: Actualizar α_h con MH"""
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
                for idx in affected:
                    if self.z[idx] == h:
                        log_like_curr += np.log(np.clip(v_curr[idx], 1e-10, 1.0))
                        log_like_prop += np.log(np.clip(v_prop[idx], 1e-10, 1.0))
                    else:
                        log_like_curr += np.log(np.clip(1 - v_curr[idx], 1e-10, 1.0))
                        log_like_prop += np.log(np.clip(1 - v_prop[idx], 1e-10, 1.0))
                
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
        """Paso 5: Actualizar ψ_{hj} con MH"""
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
        """Paso 6: Actualizar ℓ_{hj} discretamente"""
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
        """Paso 7: Recalcular pesos"""
        self.w = self._compute_weights()
    
    def update_mu(self):
        """Paso 8: Actualizar μ (hiperparámetro de α_h)"""
        tau_post = len(self.alpha) + 1.0 / self.tau_mu_inv
        mu_post = (np.sum(self.alpha) + self.mu_mu / self.tau_mu_inv) / tau_post
        
        self.mu = np.random.normal(mu_post, 1.0 / np.sqrt(tau_post))
        self.mu = np.clip(self.mu, -10, 10)
    
    def update_mu0(self):
        """Paso 9: Actualizar μ₀"""
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
        """Paso 10: Actualizar τ₀ con MH"""
        log_tau = math.log(self.tau0)
        log_tau_prop = np.random.normal(log_tau, self.mh_scales['tau0'])
        tau_prop = math.exp(log_tau_prop)
        tau_prop = np.clip(tau_prop, 0.01, 100.0)
        
        active_clusters = np.unique(self.z)
        if len(active_clusters) == 0:
            return
        
        mu_active = self.theta_mu[active_clusters]
        diff_sq = (mu_active - self.mu0)**2
        
        # Log-likelihood
        K = len(active_clusters)
        log_like_curr = (0.5 * K * np.log(self.tau0) - 
                        0.5 * self.tau0 * np.sum(diff_sq))
        log_like_prop = (0.5 * K * np.log(tau_prop) - 
                        0.5 * tau_prop * np.sum(diff_sq))
        
        # Log-prior Gamma
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
        """Paso 11: Actualizar a₀ con MH"""
        log_a = math.log(self.a0)
        log_a_prop = np.random.normal(log_a, self.mh_scales['a0'])
        a_prop = math.exp(log_a_prop)
        a_prop = np.clip(a_prop, 0.5, 20.0)
        
        active_clusters = np.unique(self.z)
        if len(active_clusters) == 0:
            return
        
        b_active = self.theta_b[active_clusters]
        
        # Log-likelihood de Gamma
        K = len(active_clusters)
        log_like_curr = (self.a0 * np.sum(np.log(self.beta0)) - 
                        K * math.lgamma(self.a0) + 
                        (self.a0 - 1) * np.sum(np.log(b_active)) - 
                        self.beta0 * np.sum(b_active))
        log_like_prop = (a_prop * np.sum(np.log(self.beta0)) - 
                        K * math.lgamma(a_prop) + 
                        (a_prop - 1) * np.sum(np.log(b_active)) - 
                        self.beta0 * np.sum(b_active))
        
        # Log-prior Gamma
        log_prior_curr = (self.alpha_a - 1) * math.log(self.a0) - self.beta_a * self.a0
        log_prior_prop = (self.alpha_a - 1) * math.log(a_prop) - self.beta_a * a_prop
        
        log_r = (log_like_prop + log_prior_prop) - (log_like_curr + log_prior_curr)
        log_r = np.clip(log_r, -50, 50)
        
        accept = math.log(np.random.rand()) < log_r
        if accept:
            self.a0 = a_prop
        
        self.mh_acceptance['a0'].append(float(accept))
    
    def update_beta0(self):
        """Paso 12: Actualizar β₀ con Gibbs (conjugado)"""
        active_clusters = np.unique(self.z)
        if len(active_clusters) == 0:
            return
        
        b_active = self.theta_b[active_clusters]
        
        # Posterior Gamma
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
        """
        Ejecuta la cadena de Markov para LSBP-Laplace con Slice Sampling
        
        Parámetros:
        -----------
        iterations : int
            Número total de iteraciones
        burnin : int
            Número de iteraciones de burn-in
        
        Retorna:
        --------
        trace : dict
            Trazas de todos los parámetros
        """
        
        for it in range(iterations):
            # Ciclo principal de Slice Sampling
            u = self.sample_slice_variables()
            self.update_assignments(u)
            self.update_atoms()
            
            # Actualizar parámetros de dependencia
            self.update_alpha()
            self.update_psi()
            self.update_ell()
            self.update_weights()
            
            # Actualizar hiperparámetros
            self.update_mu()
            self.update_mu0()
            self.update_tau0()
            self.update_a0()
            self.update_beta0()
            
            # Adaptar propuestas durante burn-in
            if it < burnin:
                self.adapt_mh_scales(it)
            
            # Guardar trazas (después del burn-in)
            if it >= burnin:
                active_clusters = len(np.unique(self.z))
                
                # Desnormalizar parámetros
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
            
            # Verbose
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
        """
        Calcula resúmenes de la distribución posterior
        
        Retorna:
        --------
        summary : dict
            Medias y desviaciones estándar posteriores
        """
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
        """
        Estima la densidad predictiva f(y_new | X_new, data)
        
        Parámetros:
        -----------
        y_new : array
            Valores de y donde evaluar la densidad
        X_new : array (n_new, p)
            Covariables para predicción
        n_samples : int
            Número de muestras posteriores a usar
        
        Retorna:
        --------
        density : array (len(y_new), n_new)
            Densidad predictiva estimada
        """
        n_new = X_new.shape[0]
        y_grid = np.array(y_new)
        density = np.zeros((len(y_grid), n_new))
        
        # Normalizar X_new
        X_new_norm = (X_new - self.X_mean) / self.X_std
        
        # Seleccionar muestras posteriores
        n_post = len(self.trace['z'])
        indices = np.linspace(0, n_post - 1, n_samples, dtype=int)
        
        for idx in indices:
            # Recuperar parámetros
            alpha_sample = self.trace['alpha'][idx]
            psi_sample = self.trace['psi'][idx]
            ell_sample = self.trace['ell'][idx]
            theta_mu_sample = self.trace['theta_mu'][idx]
            theta_b_sample = self.trace['theta_b'][idx]
            
            # Calcular pesos para X_new
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
            
            # Calcular densidad como mezcla de Laplace
            for i in range(n_new):
                for y_idx, y_val in enumerate(y_grid):
                    for h in range(H_sample):
                        density[y_idx, i] += (w[i, h] * 
                            laplace.pdf(y_val, theta_mu_sample[h], theta_b_sample[h]))
        
        # Promediar sobre muestras posteriores
        density /= n_samples
        
        return density
    
    def predict_mean(self, X_new, n_samples=100):
        """
        Estima la media condicional E[y | X_new, data]
        
        Para Laplace: E[y | μ, b] = μ
        
        Parámetros:
        -----------
        X_new : array (n_new, p)
            Covariables para predicción
        n_samples : int
            Número de muestras posteriores
        
        Retorna:
        --------
        mean_pred : array (n_new,)
            Media condicional estimada
        std_pred : array (n_new,)
            Desviación estándar de la predicción
        """
        n_new = X_new.shape[0]
        predictions = np.zeros((n_samples, n_new))
        
        # Normalizar X_new
        X_new_norm = (X_new - self.X_mean) / self.X_std
        
        # Seleccionar muestras posteriores
        n_post = len(self.trace['z'])
        indices = np.linspace(0, n_post - 1, n_samples, dtype=int)
        
        for s, idx in enumerate(indices):
            # Recuperar parámetros
            alpha_sample = self.trace['alpha'][idx]
            psi_sample = self.trace['psi'][idx]
            ell_sample = self.trace['ell'][idx]
            theta_mu_sample = self.trace['theta_mu'][idx]
            
            # Calcular pesos
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
            
            # Media condicional = suma de pesos × medias de clusters
            # Para Laplace: E[Y|μ,b] = μ
            predictions[s, :] = np.sum(w * theta_mu_sample[np.newaxis, :H_sample], axis=1)
        
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        
        return mean_pred, std_pred
    
    def predict_quantile(self, X_new, q=0.5, n_samples=100):
        """
        Estima cuantiles condicionales Q_q(y | X_new, data)
        
        Para Laplace(μ, b):
        - Mediana: μ
        - Q_q: μ + b * sign(q - 0.5) * ln(1 - 2|q - 0.5|)
        
        Parámetros:
        -----------
        X_new : array (n_new, p)
            Covariables para predicción
        q : float
            Cuantil a estimar (e.g., 0.5 para mediana)
        n_samples : int
            Número de muestras posteriores
        
        Retorna:
        --------
        quantile_pred : array (n_new,)
            Cuantil condicional estimado
        """
        n_new = X_new.shape[0]
        quantiles = np.zeros((n_samples, n_new))
        
        # Normalizar X_new
        X_new_norm = (X_new - self.X_mean) / self.X_std
        
        # Seleccionar muestras posteriores
        n_post = len(self.trace['z'])
        indices = np.linspace(0, n_post - 1, n_samples, dtype=int)
        
        for s, idx in enumerate(indices):
            # Recuperar parámetros
            alpha_sample = self.trace['alpha'][idx]
            psi_sample = self.trace['psi'][idx]
            ell_sample = self.trace['ell'][idx]
            theta_mu_sample = self.trace['theta_mu'][idx]
            theta_b_sample = self.trace['theta_b'][idx]
            
            # Calcular pesos
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
            
            # Aproximar cuantil de la mezcla
            # Para cada observación, encontrar cuantil numéricamente
            for i in range(n_new):
                # Calcular cuantiles de cada componente
                if q == 0.5:
                    # Mediana de Laplace = μ
                    component_quantiles = theta_mu_sample[:H_sample]
                else:
                    # Cuantil general de Laplace
                    component_quantiles = (theta_mu_sample[:H_sample] + 
                                          theta_b_sample[:H_sample] * 
                                          np.sign(q - 0.5) * 
                                          np.log(1 - 2 * np.abs(q - 0.5)))
                
                # Aproximar cuantil de mezcla como promedio ponderado
                quantiles[s, i] = np.sum(w[i, :H_sample] * component_quantiles)
        
        quantile_pred = np.mean(quantiles, axis=0)
        
        return quantile_pred