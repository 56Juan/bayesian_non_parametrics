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
    
    Proceso de Dirichlet Dependiente donde los pesos de la mezcla varían
    con las covariables mediante un enlace probit y función kernel de dependencia.
    
    Incluye selección de variables mediante spike-and-slab priors (γ_{hj}).
    
    Modelo:
    -------
    y_i | z_i=h, μ_h, σ²_h ~ N(μ_h, σ²_h)
    z_i | x_i ~ Categorical(w_1(x_i), ..., w_T(x_i))
    
    w_h(x) = v_h(x) ∏_{ℓ<h} (1 - v_ℓ(x))
    v_h(x) = Φ(η_h(x)) donde Φ es la CDF normal estándar
    η_h(x) = α_h - Σ_j γ_{hj} · ψ_{hj} · |x_j - ℓ_{hj}|
    
    Variables latentes para conjugación:
    u_{ih} | z, η ~ TruncatedNormal(η_h(x_i), 1)
    
    Usa Slice Sampling + Gibbs donde sea posible.
    """
    
    def __init__(self, y, X, H=20,
                 mu_prior=(0.0, 1.0),           # μ_α
                 mu0_prior=(0.0, 100.0),        # μ₀
                 kappa0_prior=(2.0, 1.0),       # κ₀
                 a0_prior=(3.0, 1.0),           # a₀
                 b0_prior=(2.0, 1.0),           # b₀
                 psi_prior_mean=None,           # μ_ψ_j (por variable)
                 psi_prior_prec=None,           # τ_ψ_j (por variable)
                 mu_psi_hyperprior=(0.0, 1.0),  # (m_ψ, s²_ψ)
                 tau_psi_hyperprior=(2.0, 1.0), # (α_τ, β_τ)
                 estimate_psi_hyperpriors=False, # Si True, estima μ_ψ y τ_ψ
                 kappa_prior=(1.0, 1.0),        # κ_j ~ Beta(a_κ, b_κ)
                 n_grid=50,
                 verbose=True,
                 use_cpp=True,
                 psi_positive=True):            # Si True, ψ > 0 (truncado)
        
        self.y = np.array(y).flatten()
        self.X = np.array(X)
        self.n, self.p = self.X.shape
        self.H = H
        self.verbose = verbose
        self.use_cpp = use_cpp and CPP_AVAILABLE
        self.psi_positive = psi_positive
        
        if self.use_cpp:
            print("Usando aceleración C++ para PSBP")
        
        # Normalizar datos
        self.y_mean = np.mean(y)
        self.y_std = np.std(y)
        if self.y_std == 0:
            self.y_std = 1.0
        self.y_normalized = (self.y - self.y_mean) / self.y_std
        
        self.X_mean = np.mean(X, axis=0)
        self.X_std = np.std(X, axis=0)
        self.X_std[self.X_std == 0] = 1.0
        self.X_normalized = (self.X - self.X_mean) / self.X_std
        
        # Hiperpriors
        self.mu_mu, self.tau_mu_inv = mu_prior
        self.m0, self.s02 = mu0_prior
        self.alpha_kappa, self.beta_kappa = kappa0_prior
        self.alpha_a, self.beta_a = a0_prior
        self.alpha_b, self.beta_b = b0_prior
        
        # Hiperpriors para ψ
        self.m_psi, self.s2_psi = mu_psi_hyperprior
        self.alpha_tau, self.beta_tau = tau_psi_hyperprior
        self.estimate_psi_hyperpriors = estimate_psi_hyperpriors
        
        # Priors específicos por variable (para ψ)
        if psi_prior_mean is None:
            self.mu_psi = np.full(self.p, self.m_psi)
        else:
            self.mu_psi = np.array(psi_prior_mean)
        
        if psi_prior_prec is None:
            self.tau_psi = np.ones(self.p)
        else:
            self.tau_psi = np.array(psi_prior_prec)
        
        # Prior para selección de variables (κ_j)
        self.a_kappa, self.b_kappa = kappa_prior
        
        # Grilla para centros
        self.n_grid = n_grid
        self.ell_grid = self._create_grid()
        
        # MH scales
        self.mh_scales = {
            'kappa0': 0.2,
            'a0': 0.2,
            'alpha': 0.1,
            'psi': 0.1
        }
        self.mh_acceptance = {
            'kappa0': [],
            'a0': [],
            'alpha': [],
            'psi': []
        }
        
        # Storage
        self.trace = {
            'mu': [], 'mu0': [], 'kappa0': [], 'a0': [], 'b0': [],
            'z': [], 'theta_mu': [], 'theta_sigma2': [], 'w': [],
            'n_clusters': [], 'alpha': [], 'psi': [], 'gamma': [],
            'kappa': [], 'ell': [], 'n_active_vars': []
        }
        
        # Si se estiman hiperpriors de ψ, agregar al trace
        if self.estimate_psi_hyperpriors:
            self.trace['mu_psi'] = []
            self.trace['tau_psi'] = []
        
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
        
        # Hiperparámetro μ (media de α_h)
        self.mu = np.random.normal(self.mu_mu, np.sqrt(self.tau_mu_inv))
        
        # Hiperparámetros de G₀
        self.mu0 = np.random.normal(self.m0, np.sqrt(self.s02))
        self.kappa0 = np.random.gamma(self.alpha_kappa, 1.0/self.beta_kappa)
        self.kappa0 = np.clip(self.kappa0, 0.1, 10.0)
        
        self.a0 = np.random.gamma(self.alpha_a, 1.0/self.beta_a)
        self.a0 = np.clip(self.a0, 1.0, 10.0)
        
        self.b0 = np.random.gamma(self.alpha_b, 1.0/self.beta_b)
        self.b0 = np.clip(self.b0, 0.1, 10.0)
        
        # Parámetros de dependencia
        self.alpha = np.random.normal(self.mu, 1.0, size=self.H)
        
        # Selección de variables (spike-and-slab)
        self.kappa = beta.rvs(self.a_kappa, self.b_kappa, size=self.p)
        self.gamma = np.random.binomial(1, self.kappa[np.newaxis, :], size=(self.H, self.p))
        
        # ψ_{hj} (solo no-cero donde γ_{hj}=1)
        self.psi = np.zeros((self.H, self.p))
        for h in range(self.H):
            for j in range(self.p):
                if self.gamma[h, j] == 1:
                    if self.psi_positive:
                        # Truncado positivo
                        a = (0 - self.mu_psi[j]) / np.sqrt(1.0/self.tau_psi[j])
                        self.psi[h, j] = truncnorm.rvs(
                            a, np.inf,
                            loc=self.mu_psi[j],
                            scale=np.sqrt(1.0/self.tau_psi[j])
                        )
                    else:
                        # No truncado
                        self.psi[h, j] = np.random.normal(
                            self.mu_psi[j],
                            np.sqrt(1.0/self.tau_psi[j])
                        )
        
        # Localizaciones en grilla
        self.ell = np.random.randint(0, self.n_grid, size=(self.H, self.p))
        
        # Calcular pesos iniciales
        self.w = self._compute_weights()
        
        # Átomos θ_h = (μ_h, σ²_h)
        self.theta_mu = np.zeros(self.H)
        self.theta_sigma2 = np.zeros(self.H)
        for h in range(self.H):
            self.theta_sigma2[h] = invgamma.rvs(self.a0, scale=self.b0)
            self.theta_sigma2[h] = np.clip(self.theta_sigma2[h], 0.01, 100.0)
            self.theta_mu[h] = np.random.normal(
                self.mu0,
                math.sqrt(self.theta_sigma2[h] / self.kappa0)
            )
        
        # Asignaciones iniciales
        self.z = np.zeros(self.n, dtype=int)
        for i in range(self.n):
            likes = np.array([
                norm.pdf(self.y_normalized[i], self.theta_mu[h],
                        math.sqrt(self.theta_sigma2[h]))
                for h in range(self.H)
            ])
            likes = np.clip(likes, 1e-300, None)
            probs = self.w[i, :] * likes
            probs_sum = probs.sum()
            if probs_sum > 0:
                probs /= probs_sum
            else:
                probs = np.ones(self.H) / self.H
            self.z[i] = np.random.choice(self.H, p=probs)
        
        # Variables latentes u_{ih} (data augmentation para probit)
        self.u_latent = np.zeros((self.n, self.H))
        self._update_u_latent()
    
    # =========================================================================
    # FUNCIONES ACELERADAS CON C++
    # =========================================================================
    
    def _compute_eta(self, X_batch=None):
        """Calcula η_h(x) con aceleración C++ si está disponible"""
        if X_batch is None:
            X_batch = self.X_normalized
            
        if self.use_cpp:
            result = psbp_cpp.compute_eta(
                X_batch,
                self.alpha,
                self.psi,
                self.gamma,
                self.ell,
                self.ell_grid
            )
            return np.array(result.eta)
        else:
            # Implementación Python
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
        """Calcula pesos con probit stick-breaking"""
        if X_batch is None:
            X_batch = self.X_normalized
            
        if self.use_cpp:
            result = psbp_cpp.compute_weights_probit(
                X_batch,
                self.alpha,
                self.psi,
                self.gamma,
                self.ell,
                self.ell_grid
            )
            return np.array(result.weights)
        else:
            eta = self._compute_eta(X_batch)
            v = ndtr(eta)  # CDF normal estándar = Φ(η)
            
            n_batch = X_batch.shape[0]
            w = np.zeros((n_batch, self.H))
            
            for i in range(n_batch):
                remaining = 1.0
                for h in range(self.H):
                    w[i, h] = v[i, h] * remaining
                    remaining *= (1 - v[i, h])
            
            # Normalizar
            w = w / w.sum(axis=1, keepdims=True)
            return w
    
    def _update_u_latent(self):
        """Actualiza variables latentes truncadas normales"""
        if self.use_cpp:
            result = psbp_cpp.update_u_latent(
                self.u_latent,
                self.z,
                self.X_normalized,
                self.alpha,
                self.psi,
                self.gamma,
                self.ell,
                self.ell_grid
            )
            self.u_latent = np.array(result.u_latent)
        else:
            # Implementación Python
            eta = self._compute_eta(self.X_normalized)
            
            for i in range(self.n):
                z_i = self.z[i]
                for h in range(self.H):
                    if h > z_i:
                        continue
                        
                    eta_ih = eta[i, h]
                    
                    if z_i == h:
                        # u_{ih} ~ TN(η, 1, [0, ∞))
                        a = (0 - eta_ih) / 1.0
                        self.u_latent[i, h] = truncnorm.rvs(
                            a, np.inf, loc=eta_ih, scale=1.0
                        )
                    elif z_i > h:
                        # u_{ih} ~ TN(η, 1, (-∞, 0])
                        b = (0 - eta_ih) / 1.0
                        self.u_latent[i, h] = truncnorm.rvs(
                            -np.inf, b, loc=eta_ih, scale=1.0
                        )
    
    def sample_slice_variables(self):
        """Genera variables de slice para expandir clusters"""
        u = np.zeros(self.n)
        for i in range(self.n):
            w_z = self.w[i, self.z[i]]
            if w_z > 0:
                u[i] = np.random.uniform(0, w_z)
            else:
                u[i] = np.random.uniform(0, 1e-10)
        
        u_min = u.min()
        
        # Expandir solo si es necesario
        max_attempts = 3
        attempt = 0
        
        while attempt < max_attempts and self.H < 100:
            w_min = self.w.min(axis=0)
            
            if np.all(w_min < u_min):
                break
            
            # Expandir con clusters nuevos
            self._expand_clusters(n_new=3)
            attempt += 1
        
        return u
    
    def _expand_clusters(self, n_new=3):
        """Expande el modelo con clusters adicionales"""
        H_new = self.H + n_new
        
        # Nuevos α
        alpha_new = np.random.normal(self.mu, 1.0, size=n_new)
        self.alpha = np.append(self.alpha, alpha_new)
        
        # Nuevos γ
        gamma_new = np.random.binomial(1, self.kappa[np.newaxis, :], 
                                       size=(n_new, self.p))
        self.gamma = np.vstack([self.gamma, gamma_new])
        
        # Nuevos ψ
        psi_new = np.zeros((n_new, self.p))
        for h in range(n_new):
            for j in range(self.p):
                if gamma_new[h, j] == 1:
                    if self.psi_positive:
                        a = (0 - self.mu_psi[j]) / np.sqrt(1.0/self.tau_psi[j])
                        psi_new[h, j] = truncnorm.rvs(
                            a, np.inf,
                            loc=self.mu_psi[j],
                            scale=np.sqrt(1.0/self.tau_psi[j])
                        )
                    else:
                        psi_new[h, j] = np.random.normal(
                            self.mu_psi[j],
                            np.sqrt(1.0/self.tau_psi[j])
                        )
        self.psi = np.vstack([self.psi, psi_new])
        
        # Nuevos ℓ
        ell_new = np.random.randint(0, self.n_grid, size=(n_new, self.p))
        self.ell = np.vstack([self.ell, ell_new])
        
        # Nuevos átomos
        theta_mu_new = np.zeros(n_new)
        theta_sigma2_new = np.zeros(n_new)
        for h in range(n_new):
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
        
        # Expandir u_latent
        u_new = np.zeros((self.n, n_new))
        self.u_latent = np.hstack([self.u_latent, u_new])
    
    def update_assignments(self, u):
        """Actualiza asignaciones de clusters"""
        if self.use_cpp:
            self.z = np.array(psbp_cpp.update_assignments_slice(
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
                    candidates = np.arange(self.H)
                
                likes = norm.pdf(self.y_normalized[i],
                               self.theta_mu[candidates],
                               np.sqrt(self.theta_sigma2[candidates]))
                likes = np.clip(likes, 1e-300, None)
                
                probs = likes
                probs_sum = probs.sum()
                if probs_sum > 0:
                    probs /= probs_sum
                else:
                    probs = np.ones_like(likes) / len(likes)
                
                self.z[i] = candidates[np.random.choice(len(candidates), p=probs)]
    
    def update_atoms(self):
        """Actualiza parámetros de los clusters"""
        if self.use_cpp:
            result = psbp_cpp.update_atoms(
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
                    # Cluster vacío
                    self.theta_sigma2[h] = invgamma.rvs(self.a0, scale=self.b0)
                    self.theta_sigma2[h] = np.clip(self.theta_sigma2[h], 0.01, 100.0)
                    self.theta_mu[h] = np.random.normal(
                        self.mu0,
                        math.sqrt(self.theta_sigma2[h] / self.kappa0)
                    )
    
    def update_alpha(self):
        """Actualiza interceptos α_h"""
        if self.use_cpp:
            result = psbp_cpp.update_alpha_probit(
                self.alpha,
                self.u_latent,
                self.X_normalized,
                self.psi,
                self.gamma,
                self.ell,
                self.ell_grid,
                self.mu,
                self.mh_scales['alpha'],
                self.H
            )
            self.alpha = np.array(result.alpha)
            self.mh_acceptance['alpha'].extend(result.acceptances)
        else:
            for h in range(self.H):
                affected_idx = np.where(self.z >= h)[0]
                
                if len(affected_idx) == 0:
                    self.alpha[h] = np.random.normal(self.mu, 1.0)
                    continue
                
                # Calcular corrección Σ γ·ψ·|x-ℓ|
                correction = np.zeros(len(affected_idx))
                for j in range(self.p):
                    if self.gamma[h, j] == 1:
                        ell_hj_value = self.ell_grid[j, self.ell[h, j]]
                        dist = np.abs(self.X_normalized[affected_idx, j] - ell_hj_value)
                        correction += self.psi[h, j] * dist
                
                # u_{ih} = α_h - correction + ε
                u_affected = self.u_latent[affected_idx, h]
                
                # Posterior Normal
                n_affected = len(affected_idx)
                tau_post = n_affected + 1.0
                mu_post = (np.sum(u_affected + correction) + self.mu) / tau_post
                
                self.alpha[h] = np.random.normal(mu_post, 1.0 / np.sqrt(tau_post))
    
    def update_psi_gamma(self):
        """Actualiza ψ y γ combinados (spike-and-slab)"""
        if self.use_cpp:
            result = psbp_cpp.update_psi_gamma(
                self.psi,
                self.gamma,
                self.mu_psi,
                self.tau_psi,
                self.kappa,
                self.u_latent,
                self.X_normalized,
                self.alpha,
                self.ell,
                self.ell_grid,
                self.mh_scales['psi'],
                self.psi_positive,
                self.H
            )
            self.psi = np.array(result.psi)
            self.gamma = np.array(result.gamma)
            if hasattr(result, 'acceptances_psi'):
                # Aplanar matriz de aceptaciones
                acceptances_flat = np.array(result.acceptances_psi).flatten()
                self.mh_acceptance['psi'].extend(acceptances_flat.tolist())
        else:
            # Implementación Python
            for h in range(self.H):
                for j in range(self.p):
                    # Actualizar γ_{hj}
                    log_prior_1 = np.log(self.kappa[j] + 1e-10)
                    log_prior_0 = np.log(1 - self.kappa[j] + 1e-10)
                    
                    if self.psi[h, j] != 0 or self.gamma[h, j] == 1:
                        log_like_1 = -0.5 * self.tau_psi[j] * (self.psi[h, j] - self.mu_psi[j])**2
                        log_like_0 = 0.0
                    else:
                        log_like_1 = 0.0
                        log_like_0 = 0.0
                    
                    log_odds = (log_prior_1 + log_like_1) - (log_prior_0 + log_like_0)
                    p_gamma_1 = 1.0 / (1.0 + np.exp(-log_odds))
                    
                    self.gamma[h, j] = np.random.binomial(1, p_gamma_1)
                    
                    # Si γ=0, forzar ψ=0
                    if self.gamma[h, j] == 0:
                        self.psi[h, j] = 0.0
                        continue
                    
                    # Actualizar ψ_{hj} si γ=1 (MH)
                    psi_curr = self.psi[h, j]
                    psi_prop = np.random.normal(psi_curr, self.mh_scales['psi'])
                    
                    if self.psi_positive and psi_prop < 0:
                        self.mh_acceptance['psi'].append(False)
                        continue
                    
                    # Solo observaciones donde z_i >= h
                    affected_idx = np.where(self.z >= h)[0]
                    
                    if len(affected_idx) == 0:
                        log_prior_curr = -0.5 * self.tau_psi[j] * (psi_curr - self.mu_psi[j])**2
                        log_prior_prop = -0.5 * self.tau_psi[j] * (psi_prop - self.mu_psi[j])**2
                        log_r = log_prior_prop - log_prior_curr
                    else:
                        # Calcular log-verosimilitud
                        ell_hj_value = self.ell_grid[j, self.ell[h, j]]
                        dist = np.abs(self.X_normalized[affected_idx, j] - ell_hj_value)
                        
                        # η base sin contribución de j
                        eta_base = self.alpha[h]
                        for jj in range(self.p):
                            if jj != j and self.gamma[h, jj] == 1:
                                ell_jj_value = self.ell_grid[jj, self.ell[h, jj]]
                                dist_jj = np.abs(self.X_normalized[affected_idx, jj] - ell_jj_value)
                                eta_base -= self.psi[h, jj] * dist_jj
                        
                        # Para todas las observaciones afectadas
                        eta_curr = eta_base - psi_curr * dist
                        eta_prop = eta_base - psi_prop * dist
                        
                        # u ~ N(η, 1)
                        u_affected = self.u_latent[affected_idx, h]
                        log_like_curr = -0.5 * np.sum((u_affected - eta_curr)**2)
                        log_like_prop = -0.5 * np.sum((u_affected - eta_prop)**2)
                        
                        # Priors
                        log_prior_curr = -0.5 * self.tau_psi[j] * (psi_curr - self.mu_psi[j])**2
                        log_prior_prop = -0.5 * self.tau_psi[j] * (psi_prop - self.mu_psi[j])**2
                        
                        log_r = (log_like_prop + log_prior_prop) - (log_like_curr + log_prior_curr)
                    
                    accept = np.log(np.random.rand()) < log_r
                    if accept:
                        self.psi[h, j] = psi_prop
                    
                    self.mh_acceptance['psi'].append(accept)
    
    def update_ell(self):
        """Actualiza localizaciones ℓ_{hj}"""
        if self.use_cpp:
            result = psbp_cpp.update_ell_probit(
                self.ell,
                self.z,
                self.X_normalized,
                self.alpha,
                self.psi,
                self.gamma,
                self.u_latent,
                self.ell_grid,
                self.H,
                self.n_grid
            )
            self.ell = np.array(result.ell)
        else:
            for h in range(self.H):
                for j in range(self.p):
                    if self.gamma[h, j] == 0:
                        self.ell[h, j] = np.random.randint(0, self.n_grid)
                        continue
                    
                    # Solo observaciones donde z_i >= h
                    affected_idx = np.where(self.z >= h)[0]
                    
                    if len(affected_idx) == 0:
                        self.ell[h, j] = np.random.randint(0, self.n_grid)
                        continue
                    
                    # Calcular log-likelihood para cada posición del grid
                    log_likes = np.zeros(self.n_grid)
                    
                    for m in range(self.n_grid):
                        ell_value = self.ell_grid[j, m]
                        dist = np.abs(self.X_normalized[affected_idx, j] - ell_value)
                        
                        # η con este ℓ
                        eta = self.alpha[h] - self.psi[h, j] * dist
                        for jj in range(self.p):
                            if jj != j and self.gamma[h, jj] == 1:
                                ell_jj_value = self.ell_grid[jj, self.ell[h, jj]]
                                dist_jj = np.abs(self.X_normalized[affected_idx, jj] - ell_jj_value)
                                eta -= self.psi[h, jj] * dist_jj
                        
                        # Log-likelihood usando u_{ih} ~ N(η, 1)
                        u_affected = self.u_latent[affected_idx, h]
                        log_likes[m] = -0.5 * np.sum((u_affected - eta)**2)
                    
                    # Convertir a probabilidades
                    log_likes = log_likes - np.max(log_likes)
                    probs = np.exp(log_likes)
                    probs /= probs.sum()
                    
                    self.ell[h, j] = np.random.choice(self.n_grid, p=probs)
    
    def update_kappa(self):
        """Actualiza κ_j (probabilidad de selección de variables)"""
        n_active = np.sum(self.gamma, axis=0)
        
        a_post = self.a_kappa + n_active
        b_post = self.b_kappa + (self.H - n_active)
        
        for j in range(self.p):
            self.kappa[j] = beta.rvs(a_post[j], b_post[j])
    
    def update_weights(self):
        """Recalcula pesos"""
        self.w = self._compute_weights()
    
    def update_mu(self):
        """Actualiza μ (hiperparámetro de α_h)"""
        tau_post = len(self.alpha) + 1.0 / self.tau_mu_inv
        mu_post = (np.sum(self.alpha) + self.mu_mu / self.tau_mu_inv) / tau_post
        
        self.mu = np.random.normal(mu_post, 1.0 / np.sqrt(tau_post))
    
    def update_mu0(self):
        """Actualiza μ₀"""
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
        """Actualiza κ₀ con MH"""
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
        
        accept = math.log(np.random.rand()) < log_r
        if accept:
            self.kappa0 = kappa_prop
        
        self.mh_acceptance['kappa0'].append(float(accept))
    
    def update_a0(self):
        """Actualiza a₀ con MH"""
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
        
        accept = math.log(np.random.rand()) < log_r
        if accept:
            self.a0 = a_prop
        
        self.mh_acceptance['a0'].append(float(accept))
    
    def update_b0(self):
        """Actualiza b₀ con Gibbs"""
        active_clusters = np.unique(self.z)
        if len(active_clusters) == 0:
            return
        
        sigma2_active = self.theta_sigma2[active_clusters]
        
        alpha_post = self.alpha_b + len(active_clusters) * self.a0
        inv_sigma2 = np.clip(1.0 / sigma2_active, 1e-6, 1e6)
        beta_post = self.beta_b + np.sum(inv_sigma2)
        
        self.b0 = np.random.gamma(alpha_post, 1.0 / beta_post)
        self.b0 = np.clip(self.b0, 0.01, 100.0)
    
    def update_mu_psi(self):
        """Actualiza μ_ψ_j (si se estiman hiperpriors)"""
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
            m_post = (self.tau_psi[j] * np.sum(psi_active) + 
                      self.m_psi / self.s2_psi) / tau_post
            s_post = 1.0 / np.sqrt(tau_post)
            
            self.mu_psi[j] = np.random.normal(m_post, s_post)
    
    def update_tau_psi(self):
        """Actualiza τ_ψ_j (si se estiman hiperpriors)"""
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
        """Adapta escalas MH durante burn-in"""
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
    
    def run(self, iterations=1000, burnin=500):
        """Ejecuta el MCMC para PSBP"""
        
        for it in range(iterations):
            # Slice sampling
            u = self.sample_slice_variables()
            self.update_assignments(u)
            self.update_atoms()
            
            # Variables latentes (probit data augmentation)
            self._update_u_latent()
            
            # Parámetros de dependencia
            self.update_alpha()
            self.update_psi_gamma()  # Actualiza ψ y γ juntos
            self.update_kappa()      # Actualiza κ después de γ
            self.update_ell()
            self.update_weights()
            
            # Hiperparámetros
            self.update_mu()
            self.update_mu0()
            self.update_kappa0()
            self.update_a0()
            self.update_b0()
            
            # Hiperpriors de ψ si se estiman
            if self.estimate_psi_hyperpriors:
                self.update_mu_psi()
                self.update_tau_psi()
            
            # Adaptar propuestas durante burn-in
            if it < burnin:
                self.adapt_mh_scales(it)
            
            # Guardar trazas
            if it >= burnin:
                active_clusters = len(np.unique(self.z))
                n_active_vars = np.sum(self.gamma, axis=0)
                
                # Desnormalizar parámetros
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
                self.trace['gamma'].append(self.gamma.copy())
                self.trace['kappa'].append(self.kappa.copy())
                self.trace['ell'].append(self.ell.copy())
                self.trace['n_active_vars'].append(n_active_vars.copy())
                
                if self.estimate_psi_hyperpriors:
                    self.trace['mu_psi'].append(self.mu_psi.copy())
                    self.trace['tau_psi'].append(self.tau_psi.copy())
            
            # Verbose
            if self.verbose and (it + 1) % 100 == 0:
                active = len(np.unique(self.z))
                n_active = np.mean(np.sum(self.gamma, axis=0))
                acc_k = np.mean(self.mh_acceptance['kappa0'][-100:]) if len(self.mh_acceptance['kappa0']) >= 100 else 0
                acc_a = np.mean(self.mh_acceptance['a0'][-100:]) if len(self.mh_acceptance['a0']) >= 100 else 0
                acc_alpha = np.mean(self.mh_acceptance['alpha'][-100:]) if len(self.mh_acceptance['alpha']) >= 100 else 0
                acc_psi = np.mean(self.mh_acceptance['psi'][-100:]) if len(self.mh_acceptance['psi']) >= 100 else 0
                
                print(f"Iter {it+1}/{iterations}: K_eff={active}, H={self.H}, "
                      f"μ={self.mu:.2f}, μ₀={self.mu0:.2f}, κ₀={self.kappa0:.2f}, "
                      f"a₀={self.a0:.2f}, b₀={self.b0:.2f}")
                print(f"  Active vars: {n_active:.1f}/{self.p}, "
                      f"Acceptance: α={acc_alpha:.2f}, ψ={acc_psi:.2f}, "
                      f"κ={acc_k:.2f}, a={acc_a:.2f}")
        
        return self.trace
    
    # =========================================================================
    # FUNCIONES DE PREDICCIÓN (igual que antes)
    # =========================================================================
    
    def get_posterior_summary(self):
        """Calcula resúmenes de la distribución posterior"""
        summary = {
            'mu': (np.mean(self.trace['mu']), np.std(self.trace['mu'])),
            'mu0': (np.mean(self.trace['mu0']), np.std(self.trace['mu0'])),
            'kappa0': (np.mean(self.trace['kappa0']), np.std(self.trace['kappa0'])),
            'a0': (np.mean(self.trace['a0']), np.std(self.trace['a0'])),
            'b0': (np.mean(self.trace['b0']), np.std(self.trace['b0'])),
            'n_clusters': (np.mean(self.trace['n_clusters']), 
                          np.std(self.trace['n_clusters'])),
            'kappa': (np.mean(np.array(self.trace['kappa']), axis=0),
                     np.std(np.array(self.trace['kappa']), axis=0)),
            'n_active_vars': (np.mean(np.array(self.trace['n_active_vars']), axis=0),
                             np.std(np.array(self.trace['n_active_vars']), axis=0))
        }
        return summary
    
    def predict_density(self, y_new, X_new, n_samples=100):
        """Estima la densidad predictiva f(y_new | X_new, data)."""
        n_new = X_new.shape[0]
        y_grid = np.array(y_new)
        density = np.zeros((len(y_grid), n_new))
        
        X_new_norm = (X_new - self.X_mean) / self.X_std
        n_post = len(self.trace['z'])
        indices = np.linspace(0, n_post - 1, n_samples, dtype=int)
        
        for idx in indices:
            alpha_sample = self.trace['alpha'][idx]
            psi_sample = self.trace['psi'][idx]
            gamma_sample = self.trace['gamma'][idx]
            ell_sample = self.trace['ell'][idx]
            theta_mu_sample = self.trace['theta_mu'][idx]
            theta_sigma2_sample = self.trace['theta_sigma2'][idx]
            
            H_sample = len(alpha_sample)
            eta = np.zeros((n_new, H_sample))
            
            for h in range(H_sample):
                eta[:, h] = alpha_sample[h]
                for j in range(self.p):
                    if gamma_sample[h, j] == 1:
                        ell_hj_value = self.ell_grid[j, ell_sample[h, j]]
                        dist = np.abs(X_new_norm[:, j] - ell_hj_value)
                        eta[:, h] -= psi_sample[h, j] * dist
            
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
                            norm.pdf(y_val, theta_mu_sample[h], 
                                   np.sqrt(theta_sigma2_sample[h])))
        
        density /= n_samples
        return density
    
    def predict_mean(self, X_new, n_samples=100, return_full_uncertainty=True):
        """Predice E[Y|X] y la desviación estándar predictiva total."""
        n_new = X_new.shape[0]
        X_new_norm = (X_new - self.X_mean) / self.X_std
        
        n_post = len(self.trace['z'])
        indices = np.linspace(0, n_post - 1, n_samples, dtype=int)
        
        predictions_mean = np.zeros((n_samples, n_new))
        predictions_var = np.zeros((n_samples, n_new))
        
        for s, idx in enumerate(indices):
            alpha_sample = self.trace['alpha'][idx]
            psi_sample = self.trace['psi'][idx]
            gamma_sample = self.trace['gamma'][idx]
            ell_sample = self.trace['ell'][idx]
            theta_mu_sample = self.trace['theta_mu'][idx]
            theta_sigma2_sample = self.trace['theta_sigma2'][idx]
            
            H_sample = len(alpha_sample)
            eta = np.zeros((n_new, H_sample))
            
            for h in range(H_sample):
                eta[:, h] = alpha_sample[h]
                for j in range(self.p):
                    if gamma_sample[h, j] == 1:
                        ell_hj_value = self.ell_grid[j, ell_sample[h, j]]
                        dist = np.abs(X_new_norm[:, j] - ell_hj_value)
                        eta[:, h] -= psi_sample[h, j] * dist
            
            v = ndtr(eta)
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
        
        mean_pred = np.mean(predictions_mean, axis=0)
        
        if return_full_uncertainty:
            var_within = np.mean(predictions_var, axis=0)
            var_between = np.var(predictions_mean, axis=0)
            total_var = var_within + var_between
            std_pred = np.sqrt(total_var)
        else:
            std_pred = np.std(predictions_mean, axis=0)
        
        return mean_pred, std_pred
    
    def predict_with_decomposition(self, X_new, n_samples=100):
        """Versión extendida que devuelve descomposición completa de incertidumbre."""
        n_new = X_new.shape[0]
        X_new_norm = (X_new - self.X_mean) / self.X_std
        
        n_post = len(self.trace['z'])
        indices = np.linspace(0, n_post - 1, n_samples, dtype=int)
        
        predictions_mean = np.zeros((n_samples, n_new))
        predictions_var = np.zeros((n_samples, n_new))
        
        for s, idx in enumerate(indices):
            alpha_sample = self.trace['alpha'][idx]
            psi_sample = self.trace['psi'][idx]
            gamma_sample = self.trace['gamma'][idx]
            ell_sample = self.trace['ell'][idx]
            theta_mu_sample = self.trace['theta_mu'][idx]
            theta_sigma2_sample = self.trace['theta_sigma2'][idx]
            
            H_sample = len(alpha_sample)
            eta = np.zeros((n_new, H_sample))
            
            for h in range(H_sample):
                eta[:, h] = alpha_sample[h]
                for j in range(self.p):
                    if gamma_sample[h, j] == 1:
                        ell_hj_value = self.ell_grid[j, ell_sample[h, j]]
                        dist = np.abs(X_new_norm[:, j] - ell_hj_value)
                        eta[:, h] -= psi_sample[h, j] * dist
            
            v = ndtr(eta)
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
        """Predice cuantiles de la distribución predictiva P(Y|X)."""
        n_new = X_new.shape[0]
        X_new_norm = (X_new - self.X_mean) / self.X_std
        
        n_post = len(self.trace['z'])
        indices = np.linspace(0, n_post - 1, n_samples, dtype=int)
        
        y_samples = np.zeros((n_pred_samples, n_new))
        
        for i in range(n_pred_samples):
            post_idx = np.random.choice(indices)
            
            alpha_sample = self.trace['alpha'][post_idx]
            psi_sample = self.trace['psi'][post_idx]
            gamma_sample = self.trace['gamma'][post_idx]
            ell_sample = self.trace['ell'][post_idx]
            theta_mu_sample = self.trace['theta_mu'][post_idx]
            theta_sigma2_sample = self.trace['theta_sigma2'][post_idx]
            
            H_sample = len(alpha_sample)
            eta = np.zeros((n_new, H_sample))
            
            for h in range(H_sample):
                eta[:, h] = alpha_sample[h]
                for j in range(self.p):
                    if gamma_sample[h, j] == 1:
                        ell_hj_value = self.ell_grid[j, ell_sample[h, j]]
                        dist = np.abs(X_new_norm[:, j] - ell_hj_value)
                        eta[:, h] -= psi_sample[h, j] * dist
            
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
                    theta_mu_sample[cluster],
                    np.sqrt(theta_sigma2_sample[cluster])
                )
        
        results = {'mean': np.mean(y_samples, axis=0)}
        for q in quantiles:
            results[f'q_{q}'] = np.quantile(y_samples, q, axis=0)
        
        return results