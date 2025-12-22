import numpy as np
from scipy.stats import norm, invgamma, gamma, truncnorm
from scipy.special import expit  # Función logit⁻¹
import math

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
                 mu_prior=(0.0, 1.0),           # μ ~ N(μ_μ, τ⁻¹_μ)
                 mu0_prior=(0.0, 100.0),        # μ₀ ~ N(m₀, s₀²)
                 kappa0_prior=(2.0, 1.0),       # κ₀ ~ Gamma(α_κ, β_κ)
                 a0_prior=(3.0, 1.0),           # a₀ ~ Gamma(α_a, β_a)
                 b0_prior=(2.0, 1.0),           # b₀ ~ Gamma(α_b, β_b)
                 psi_prior=(0.0, 1.0),          # ψ_{hj} ~ N⁺(μ_ψ, τ⁻¹_ψ)
                 n_grid=50,                      # Número de puntos en grilla para ℓ
                 verbose=True):
        """
        Parámetros:
        -----------
        y : array (n,)
            Respuesta observada
        X : array (n, p)
            Matriz de covariables (n observaciones, p predictores)
        H : int
            Truncamiento stick-breaking inicial
        mu_prior : tuple
            (μ_μ, τ⁻¹_μ) para el prior de μ (intercepto stick-breaking)
        mu0_prior : tuple
            (m₀, s₀²) para el prior de μ₀ (media base)
        kappa0_prior : tuple
            (α_κ, β_κ) para el prior de κ₀ (precisión relativa)
        a0_prior : tuple
            (α_a, β_a) para el prior de a₀ (shape de σ²)
        b0_prior : tuple
            (α_b, β_b) para el prior de b₀ (scale de σ²)
        psi_prior : tuple
            (μ_ψ, τ⁻¹_ψ) para el prior de ψ_{hj} (decaimiento kernel)
        n_grid : int
            Número de puntos en la grilla para los centros ℓ_{hj}
        verbose : bool
            Imprimir progreso
        """
        self.y = np.array(y)
        self.X = np.array(X)
        self.n, self.p = self.X.shape
        self.H = H
        self.verbose = verbose
        
        # Normalizar datos para estabilidad numérica
        self.y_mean = np.mean(y)
        self.y_std = np.std(y)
        self.y_normalized = (self.y - self.y_mean) / self.y_std
        
        self.X_mean = np.mean(X, axis=0)
        self.X_std = np.std(X, axis=0)
        self.X_normalized = (self.X - self.X_mean) / self.X_std
        
        # Hiperpriors
        self.mu_mu, self.tau_mu_inv = mu_prior
        self.m0, self.s02 = mu0_prior
        self.alpha_kappa, self.beta_kappa = kappa0_prior
        self.alpha_a, self.beta_a = a0_prior
        self.alpha_b, self.beta_b = b0_prior
        self.mu_psi, self.tau_psi_inv = psi_prior
        
        # Grilla para centros ℓ_{hj}
        self.n_grid = n_grid
        self.ell_grid = self._create_grid()
        
        # Escalas adaptativas para Metropolis-Hastings
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
        
        # Storage para trazas
        self.trace = {
            'mu': [], 'mu0': [], 'kappa0': [], 'a0': [], 'b0': [],
            'z': [], 'theta_mu': [], 'theta_sigma2': [], 'w': [], 
            'n_clusters': [], 'alpha': [], 'psi': [], 'ell': []
        }
        
        self.initialize()
    
    def _create_grid(self):
        """
        Crea grilla uniforme de puntos para cada predictor.
        Grilla sobre datos normalizados: típicamente [-3.5, 3.5]
        """
        grid = np.zeros((self.p, self.n_grid))
        for j in range(self.p):
            x_min = self.X_normalized[:, j].min() - 0.5
            x_max = self.X_normalized[:, j].max() + 0.5
            grid[j, :] = np.linspace(x_min, x_max, self.n_grid)
        return grid
    
    def initialize(self):
        """Inicializa todos los parámetros del modelo"""
        
        # Hiperparámetro μ (intercepto global stick-breaking)
        self.mu = np.random.normal(self.mu_mu, np.sqrt(self.tau_mu_inv))
        
        # Hiperparámetros de G₀ (medida base Normal-Inverse-Gamma)
        self.mu0 = np.random.normal(0, 1)
        self.kappa0 = np.random.gamma(self.alpha_kappa, 1.0/self.beta_kappa)
        self.kappa0 = np.clip(self.kappa0, 0.1, 10.0)
        
        self.a0 = np.random.gamma(self.alpha_a, 1.0/self.beta_a)
        self.a0 = np.clip(self.a0, 1.0, 10.0)
        
        self.b0 = np.random.gamma(self.alpha_b, 1.0/self.beta_b)
        self.b0 = np.clip(self.b0, 0.1, 10.0)
        
        # Parámetros de dependencia espacial
        # α_h ~ N(μ, 1)
        self.alpha = np.random.normal(self.mu, 1.0, size=self.H)
        
        # ψ_{hj} ~ N⁺(μ_ψ, τ⁻¹_ψ)
        self.psi = np.zeros((self.H, self.p))
        for h in range(self.H):
            for j in range(self.p):
                # Truncated normal (lower=0)
                a = (0 - self.mu_psi) / np.sqrt(self.tau_psi_inv)
                self.psi[h, j] = truncnorm.rvs(a, np.inf, 
                                                loc=self.mu_psi, 
                                                scale=np.sqrt(self.tau_psi_inv))
        
        # ℓ_{hj} ~ Discrete-Uniform sobre grilla
        self.ell = np.zeros((self.H, self.p), dtype=int)
        for h in range(self.H):
            for j in range(self.p):
                self.ell[h, j] = np.random.randint(0, self.n_grid)
        
        # Calcular pesos iniciales w_h(x_i) para cada observación
        self.w = self._compute_weights()
        
        # Átomos θ_h = (μ_h, σ²_h) ~ G₀
        self.theta_mu = np.zeros(self.H)
        self.theta_sigma2 = np.zeros(self.H)
        for h in range(self.H):
            self.theta_sigma2[h] = invgamma.rvs(self.a0, scale=self.b0)
            self.theta_sigma2[h] = np.clip(self.theta_sigma2[h], 0.01, 100.0)
            self.theta_mu[h] = np.random.normal(
                self.mu0, 
                math.sqrt(self.theta_sigma2[h] / self.kappa0)
            )
        
        # Asignaciones iniciales z_i
        self.z = np.zeros(self.n, dtype=int)
        for i in range(self.n):
            # Likelihood para cada cluster
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
        """
        Calcula η_h(x) = α_h - Σ_j ψ_{hj} |x_j - ℓ_{hj}|
        
        Parámetros:
        -----------
        X_batch : array (n_batch, p)
            Batch de covariables (normalizadas)
        
        Retorna:
        --------
        eta : array (n_batch, H)
            Predictor lineal para cada observación y cluster
        """
        n_batch = X_batch.shape[0]
        eta = np.zeros((n_batch, self.H))
        
        for h in range(self.H):
            eta[:, h] = self.alpha[h]
            for j in range(self.p):
                # Centro del cluster h para predictor j
                ell_hj_value = self.ell_grid[j, self.ell[h, j]]
                # Distancia L1: |x_j - ℓ_{hj}|
                dist = np.abs(X_batch[:, j] - ell_hj_value)
                # Acumular: -ψ_{hj} * distancia
                eta[:, h] -= self.psi[h, j] * dist
        
        return eta
    
    def _compute_weights(self):
        """
        Calcula pesos dependientes w_h(x_i) mediante logit stick-breaking
        
        w_h(x) = v_h(x) ∏_{ℓ<h} (1 - v_ℓ(x))
        v_h(x) = logit⁻¹(η_h(x)) = exp(η_h(x)) / [1 + exp(η_h(x))]
        
        Retorna:
        --------
        w : array (n, H)
            Pesos para cada observación y cluster
        """
        # Calcular η_h(x_i) para todas las observaciones
        eta = self._compute_eta(self.X_normalized)
        
        # Calcular v_h(x_i) = logit⁻¹(η_h(x_i))
        v = expit(eta)  # expit(x) = 1 / (1 + exp(-x))
        
        # Stick-breaking: w_h(x) = v_h(x) ∏_{ℓ<h} (1 - v_ℓ(x))
        w = np.zeros((self.n, self.H))
        for i in range(self.n):
            remaining = 1.0
            for h in range(self.H):
                w[i, h] = v[i, h] * remaining
                remaining *= (1 - v[i, h])
        
        # Normalizar por seguridad
        w = w / w.sum(axis=1, keepdims=True)
        
        return w
    
    def sample_slice_variables(self):
        """
        Paso 1: Generar variables auxiliares u_i ~ Uniform(0, w_{z_i}(x_i))
        
        Retorna:
        --------
        u : array (n,)
            Variables auxiliares
        """
        u = np.zeros(self.n)
        for i in range(self.n):
            u[i] = np.random.uniform(0, self.w[i, self.z[i]])
        
        # Verificar si necesitamos expandir truncamiento
        u_min = u.min()
        while self.H < 100:  # Límite de seguridad
            # Verificar si algún peso mínimo está por debajo de u_min
            w_min = self.w.min(axis=0)
            if np.all(w_min < u_min):
                break
                
            # Expandir: agregar nuevos clusters
            H_new = self.H + 5
            
            # α_{H_new} ~ N(μ, 1)
            alpha_new = np.random.normal(self.mu, 1.0, size=5)
            self.alpha = np.append(self.alpha, alpha_new)
            
            # ψ_{H_new, j} ~ N⁺(μ_ψ, τ⁻¹_ψ)
            psi_new = np.zeros((5, self.p))
            for h in range(5):
                for j in range(self.p):
                    a = (0 - self.mu_psi) / np.sqrt(self.tau_psi_inv)
                    psi_new[h, j] = truncnorm.rvs(a, np.inf,
                                                   loc=self.mu_psi,
                                                   scale=np.sqrt(self.tau_psi_inv))
            self.psi = np.vstack([self.psi, psi_new])
            
            # ℓ_{H_new, j}
            ell_new = np.random.randint(0, self.n_grid, size=(5, self.p))
            self.ell = np.vstack([self.ell, ell_new])
            
            # θ_{H_new} ~ G₀
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
            
            # Recalcular pesos con nuevos clusters
            self.w = self._compute_weights()
            break  # Solo una expansión por iteración
        
        return u
    
    def update_assignments(self, u):
        """
        Paso 2: Actualizar asignaciones z_i dado u_i
        
        z_i ~ Categorical sobre {h : w_h(x_i) > u_i}
        """
        for i in range(self.n):
            # Clusters activos: w_h(x_i) > u_i
            candidates = np.where(self.w[i, :] > u[i])[0]
            
            if len(candidates) == 0:
                candidates = np.array([0])
            
            # Likelihood para cada cluster candidato
            likes = norm.pdf(self.y_normalized[i], 
                           self.theta_mu[candidates],
                           np.sqrt(self.theta_sigma2[candidates]))
            likes = np.clip(likes, 1e-300, None)
            
            # Probabilidades proporcionales a likelihood
            probs = likes / likes.sum() if likes.sum() > 0 else np.ones_like(likes) / len(likes)
            
            self.z[i] = np.random.choice(candidates, p=probs)
    
    def update_atoms(self):
        """
        Paso 3: Actualizar átomos θ_h = (μ_h, σ²_h)
        
        Para clusters con datos: usar posterior NIG conjugado
        Para clusters vacíos: muestrear de prior G₀
        """
        for h in range(self.H):
            members = self.y_normalized[self.z == h]
            n_h = len(members)
            
            if n_h > 0:
                # Posterior Normal-Inverse-Gamma
                y_bar = members.mean()
                ss = np.sum((members - y_bar)**2)
                
                kappa_n = self.kappa0 + n_h
                mu_n = (self.kappa0 * self.mu0 + n_h * y_bar) / kappa_n
                a_n = self.a0 + n_h / 2.0
                b_n = (self.b0 + 0.5 * ss + 
                       (self.kappa0 * n_h * (y_bar - self.mu0)**2) / (2 * kappa_n))
                
                # Muestrear σ²_h | y
                self.theta_sigma2[h] = invgamma.rvs(a_n, scale=b_n)
                self.theta_sigma2[h] = np.clip(self.theta_sigma2[h], 0.01, 100.0)
                
                # Muestrear μ_h | σ²_h, y
                self.theta_mu[h] = np.random.normal(
                    mu_n, 
                    math.sqrt(self.theta_sigma2[h] / kappa_n)
                )
            else:
                # Prior: muestrear de G₀
                self.theta_sigma2[h] = invgamma.rvs(self.a0, scale=self.b0)
                self.theta_sigma2[h] = np.clip(self.theta_sigma2[h], 0.01, 100.0)
                self.theta_mu[h] = np.random.normal(
                    self.mu0,
                    math.sqrt(self.theta_sigma2[h] / self.kappa0)
                )
    
    def update_alpha(self):
        """
        Paso 4: Actualizar α_h usando Metropolis-Hastings
        
        α_h ~ N(μ, 1) [prior]
        Likelihood: producto sobre i con z_i ≥ h de términos logit
        """
        for h in range(self.H - 1):  # No actualizar último (fijo a 1)
            # Propuesta
            alpha_prop = np.random.normal(self.alpha[h], self.mh_scales['alpha'])
            
            # Observaciones asignadas a clusters h o superiores
            affected = np.where(self.z >= h)[0]
            
            if len(affected) == 0:
                # No hay datos, solo prior
                log_prior_curr = -0.5 * ((self.alpha[h] - self.mu)**2)
                log_prior_prop = -0.5 * ((alpha_prop - self.mu)**2)
                log_r = log_prior_prop - log_prior_curr
            else:
                # Calcular log-likelihood
                # Necesitamos recalcular v_h(x_i) con nuevo α_h
                eta_curr = self._compute_eta_h(self.X_normalized[affected], h, 
                                                self.alpha[h])
                eta_prop = self._compute_eta_h(self.X_normalized[affected], h, 
                                                alpha_prop)
                
                v_curr = expit(eta_curr)
                v_prop = expit(eta_prop)
                
                # Log-likelihood: suma de log(v_h) si z_i=h, log(1-v_h) si z_i>h
                log_like_curr = 0.0
                log_like_prop = 0.0
                for idx in affected:
                    if self.z[idx] == h:
                        log_like_curr += np.log(np.clip(v_curr[idx], 1e-10, 1.0))
                        log_like_prop += np.log(np.clip(v_prop[idx], 1e-10, 1.0))
                    else:
                        log_like_curr += np.log(np.clip(1 - v_curr[idx], 1e-10, 1.0))
                        log_like_prop += np.log(np.clip(1 - v_prop[idx], 1e-10, 1.0))
                
                # Log-prior
                log_prior_curr = -0.5 * ((self.alpha[h] - self.mu)**2)
                log_prior_prop = -0.5 * ((alpha_prop - self.mu)**2)
                
                log_r = (log_like_prop + log_prior_prop) - (log_like_curr + log_prior_curr)
            
            log_r = np.clip(log_r, -50, 50)
            
            # Aceptar/rechazar
            accept = math.log(np.random.rand()) < log_r
            if accept:
                self.alpha[h] = alpha_prop
            
            self.mh_acceptance['alpha'].append(float(accept))
    
    def _compute_eta_h(self, X_batch, h, alpha_h_value):
        """
        Calcula η_h(x) para un cluster específico h con α_h dado
        
        Usado en MH para evaluar propuestas
        """
        n_batch = X_batch.shape[0]
        eta = np.full(n_batch, alpha_h_value)
        
        for j in range(self.p):
            ell_hj_value = self.ell_grid[j, self.ell[h, j]]
            dist = np.abs(X_batch[:, j] - ell_hj_value)
            eta -= self.psi[h, j] * dist
        
        return eta
    
    def update_psi(self):
        """
        Paso 5: Actualizar ψ_{hj} usando Metropolis-Hastings
        
        ψ_{hj} ~ N⁺(μ_ψ, τ⁻¹_ψ) [prior truncado en 0]
        """
        for h in range(self.H - 1):
            for j in range(self.p):
                # Propuesta con restricción ψ ≥ 0
                psi_curr = self.psi[h, j]
                psi_prop = np.random.normal(psi_curr, self.mh_scales['psi'])
                
                if psi_prop < 0:
                    continue  # Rechazar automáticamente
                
                # Observaciones afectadas
                affected = np.where(self.z >= h)[0]
                
                if len(affected) == 0:
                    # Solo prior truncado
                    log_prior_curr = -0.5 * ((psi_curr - self.mu_psi)**2) / self.tau_psi_inv
                    log_prior_prop = -0.5 * ((psi_prop - self.mu_psi)**2) / self.tau_psi_inv
                    log_r = log_prior_prop - log_prior_curr
                else:
                    # Calcular likelihood con nueva ψ_{hj}
                    ell_hj_value = self.ell_grid[j, self.ell[h, j]]
                    dist = np.abs(self.X_normalized[affected, j] - ell_hj_value)
                    
                    # η_h con psi actual y propuesta
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
                    
                    # Log-likelihood
                    log_like_curr = 0.0
                    log_like_prop = 0.0
                    for idx_local, idx in enumerate(affected):
                        if self.z[idx] == h:
                            log_like_curr += np.log(np.clip(v_curr[idx_local], 1e-10, 1.0))
                            log_like_prop += np.log(np.clip(v_prop[idx_local], 1e-10, 1.0))
                        else:
                            log_like_curr += np.log(np.clip(1 - v_curr[idx_local], 1e-10, 1.0))
                            log_like_prop += np.log(np.clip(1 - v_prop[idx_local], 1e-10, 1.0))
                    
                    # Log-prior truncado
                    log_prior_curr = -0.5 * ((psi_curr - self.mu_psi)**2) / self.tau_psi_inv
                    log_prior_prop = -0.5 * ((psi_prop - self.mu_psi)**2) / self.tau_psi_inv
                    
                    log_r = (log_like_prop + log_prior_prop) - (log_like_curr + log_prior_curr)
                
                log_r = np.clip(log_r, -50, 50)
                
                accept = math.log(np.random.rand()) < log_r
                if accept:
                    self.psi[h, j] = psi_prop
                
                self.mh_acceptance['psi'].append(float(accept))
    
    def update_ell(self):
        """
        Paso 6: Actualizar ℓ_{hj} de forma discreta sobre la grilla
        
        ℓ_{hj} ~ Discrete-Uniform{ℓ*_{jm}}_{m=1}^{M_j}
        """
        for h in range(self.H - 1):
            for j in range(self.p):
                # Observaciones afectadas
                affected = np.where(self.z >= h)[0]
                
                if len(affected) == 0:
                    # Uniforme sobre grilla
                    self.ell[h, j] = np.random.randint(0, self.n_grid)
                    continue
                
                # Calcular log-likelihood para cada punto de la grilla
                log_likes = np.zeros(self.n_grid)
                
                for m in range(self.n_grid):
                    # Calcular η_h con ℓ_{hj} = ℓ*_{jm}
                    ell_value = self.ell_grid[j, m]
                    dist = np.abs(self.X_normalized[affected, j] - ell_value)
                    
                    # η_h completo (todos los predictores)
                    eta = self.alpha[h] - self.psi[h, j] * dist
                    for jj in range(self.p):
                        if jj != j:
                            ell_jj_value = self.ell_grid[jj, self.ell[h, jj]]
                            eta -= self.psi[h, jj] * np.abs(
                                self.X_normalized[affected, jj] - ell_jj_value
                            )
                    
                    v = expit(eta)
                    
                    # Log-likelihood
                    for idx_local, idx in enumerate(affected):
                        if self.z[idx] == h:
                            log_likes[m] += np.log(np.clip(v[idx_local], 1e-10, 1.0))
                        else:
                            log_likes[m] += np.log(np.clip(1 - v[idx_local], 1e-10, 1.0))
                
                # Convertir a probabilidades (prior uniforme se cancela)
                log_likes = log_likes - np.max(log_likes)  # Estabilidad
                probs = np.exp(log_likes)
                probs /= probs.sum()
                
                # Muestrear nuevo ℓ_{hj}
                self.ell[h, j] = np.random.choice(self.n_grid, p=probs)
    
    def update_weights(self):
        """
        Paso 7: Recalcular pesos w_h(x_i) con parámetros actualizados
        """
        self.w = self._compute_weights()
    
    def update_mu(self):
        """
        Paso 8: Actualizar μ (hiperparámetro de α_h)
        
        μ ~ N(μ_μ, τ⁻¹_μ) [prior]
        α_h | μ ~ N(μ, 1) [likelihood]
        """
        # Posterior Normal conjugado
        tau_post = len(self.alpha) + 1.0 / self.tau_mu_inv
        mu_post = (np.sum(self.alpha) + self.mu_mu / self.tau_mu_inv) / tau_post
        
        self.mu = np.random.normal(mu_post, 1.0 / np.sqrt(tau_post))
        self.mu = np.clip(self.mu, -10, 10)
    
    def update_mu0(self):
        """
        Paso 9: Actualizar μ₀ (hiperparámetro de la media base)
        
        μ₀ ~ N(m₀, s₀²) [prior]
        μ_h | μ₀, σ²_h, κ₀ ~ N(μ₀, σ²_h/κ₀) [likelihood]
        """
        # Solo clusters activos
        active_clusters = np.unique(self.z)
        if len(active_clusters) == 0:
            return
        
        mu_active = self.theta_mu[active_clusters]
        sigma2_active = self.theta_sigma2[active_clusters]
        
        # Posterior Normal conjugado
        inv_sigma2 = np.clip(1.0 / sigma2_active, 1e-6, 1e6)
        precision_post = self.kappa0 * np.sum(inv_sigma2) + 1.0 / self.s02
        s0n2 = 1.0 / precision_post
        m0n = s0n2 * (self.kappa0 * np.sum(mu_active * inv_sigma2) + self.m0 / self.s02)
        
        s0n2 = np.clip(s0n2, 1e-6, 1e6)
        m0n = np.clip(m0n, -100, 100)
        
        self.mu0 = np.random.normal(m0n, math.sqrt(s0n2))
        self.mu0 = np.clip(self.mu0, -50, 50)
    
    def update_kappa0(self):
        """
        Paso 10: Actualizar κ₀ usando Metropolis-Hastings
        
        κ₀ ~ Gamma(α_κ, β_κ) [prior]
        μ_h | μ₀, σ²_h, κ₀ ~ N(μ₀, σ²_h/κ₀) [likelihood]
        """
        log_kappa = math.log(self.kappa0)
        log_kappa_prop = np.random.normal(log_kappa, self.mh_scales['kappa0'])
        kappa_prop = math.exp(log_kappa_prop)
        kappa_prop = np.clip(kappa_prop, 0.01, 100.0)
        
        # Solo clusters activos
        active_clusters = np.unique(self.z)
        if len(active_clusters) == 0:
            return
        
        mu_active = self.theta_mu[active_clusters]
        sigma2_active = self.theta_sigma2[active_clusters]
        
        diff_sq = (mu_active - self.mu0)**2
        inv_sigma2 = np.clip(1.0 / sigma2_active, 1e-6, 1e6)
        
        # Log-likelihood
        log_like_curr = (0.5 * len(active_clusters) * np.log(self.kappa0) - 
                        0.5 * self.kappa0 * np.sum(diff_sq * inv_sigma2))
        log_like_prop = (0.5 * len(active_clusters) * np.log(kappa_prop) - 
                        0.5 * kappa_prop * np.sum(diff_sq * inv_sigma2))
        
        # Log-prior Gamma
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
        """
        Paso 11: Actualizar a₀ usando Metropolis-Hastings
        
        a₀ ~ Gamma(α_a, β_a) [prior]
        σ²_h | a₀, b₀ ~ InvGamma(a₀, b₀) [likelihood]
        """
        log_a = math.log(self.a0)
        log_a_prop = np.random.normal(log_a, self.mh_scales['a0'])
        a_prop = math.exp(log_a_prop)
        a_prop = np.clip(a_prop, 0.5, 20.0)
        
        # Solo clusters activos
        active_clusters = np.unique(self.z)
        if len(active_clusters) == 0:
            return
        
        sigma2_active = self.theta_sigma2[active_clusters]
        
        # Log-likelihood de Inverse-Gamma
        ratio = np.clip(self.b0 / sigma2_active, 1e-10, 1e10)
        log_ratio = np.log(ratio)
        
        K = len(active_clusters)
        log_like_curr = (self.a0 * np.sum(log_ratio) - 
                        K * math.lgamma(self.a0) - 
                        (self.a0 + 1) * np.sum(np.log(sigma2_active)))
        log_like_prop = (a_prop * np.sum(log_ratio) - 
                        K * math.lgamma(a_prop) - 
                        (a_prop + 1) * np.sum(np.log(sigma2_active)))
        
        # Log-prior Gamma
        log_prior_curr = (self.alpha_a - 1) * math.log(self.a0) - self.beta_a * self.a0
        log_prior_prop = (self.alpha_a - 1) * math.log(a_prop) - self.beta_a * a_prop
        
        log_r = (log_like_prop + log_prior_prop) - (log_like_curr + log_prior_curr)
        log_r = np.clip(log_r, -50, 50)
        
        accept = math.log(np.random.rand()) < log_r
        if accept:
            self.a0 = a_prop
        
        self.mh_acceptance['a0'].append(float(accept))
    
    def update_b0(self):
        """
        Paso 12: Actualizar b₀ con Gibbs (conjugado)
        
        b₀ ~ Gamma(α_b, β_b) [prior]
        σ²_h | a₀, b₀ ~ InvGamma(a₀, b₀) [likelihood]
        """
        active_clusters = np.unique(self.z)
        if len(active_clusters) == 0:
            return
        
        sigma2_active = self.theta_sigma2[active_clusters]
        
        # Posterior Gamma
        alpha_post = self.alpha_b + len(active_clusters) * self.a0
        inv_sigma2 = np.clip(1.0 / sigma2_active, 1e-6, 1e6)
        beta_post = self.beta_b + np.sum(inv_sigma2)
        
        self.b0 = np.random.gamma(alpha_post, 1.0 / beta_post)
        self.b0 = np.clip(self.b0, 0.01, 100.0)
    
    def adapt_mh_scales(self, iteration):
        """
        Adaptar escalas de propuestas MH durante burn-in
        Target acceptance rate: ~0.234 para propuestas univariadas
        """
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
        """
        Ejecuta la cadena de Markov para LSBP con Slice Sampling
        
        Parámetros:
        -----------
        iterations : int
            Número total de iteraciones
        burnin : int
            Número de iteraciones de burn-in (no se guardan)
        
        Retorna:
        --------
        trace : dict
            Diccionario con trazas de todos los parámetros
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
            self.update_kappa0()
            self.update_a0()
            self.update_b0()
            
            # Adaptar propuestas durante burn-in
            if it < burnin:
                self.adapt_mh_scales(it)
            
            # Guardar trazas (después del burn-in)
            if it >= burnin:
                active_clusters = len(np.unique(self.z))
                
                # Desnormalizar parámetros para guardar
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
            
            # Verbose
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
        """
        Calcula resúmenes de la distribución posterior
        
        Retorna:
        --------
        summary : dict
            Diccionario con medias y desviaciones estándar posteriores
        """
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
        
        # Seleccionar muestras posteriores uniformemente
        n_post = len(self.trace['z'])
        indices = np.linspace(0, n_post - 1, n_samples, dtype=int)
        
        for idx in indices:
            # Recuperar parámetros de esta iteración
            alpha_sample = self.trace['alpha'][idx]
            psi_sample = self.trace['psi'][idx]
            ell_sample = self.trace['ell'][idx]
            theta_mu_sample = self.trace['theta_mu'][idx]
            theta_sigma2_sample = self.trace['theta_sigma2'][idx]
            
            # Calcular pesos para X_new
            H_sample = len(alpha_sample)
            eta = np.zeros((n_new, H_sample))
            
            for h in range(H_sample):
                eta[:, h] = alpha_sample[h]
                for j in range(self.p):
                    ell_hj_value = self.ell_grid[j, ell_sample[h, j]]
                    # Desnormalizar para usar grilla original
                    ell_hj_orig = ell_hj_value * self.X_std[j] + self.X_mean[j]
                    dist = np.abs(X_new_norm[:, j] - ell_hj_value)
                    eta[:, h] -= psi_sample[h, j] * dist
            
            v = expit(eta)
            w = np.zeros((n_new, H_sample))
            for i in range(n_new):
                remaining = 1.0
                for h in range(H_sample):
                    w[i, h] = v[i, h] * remaining
                    remaining *= (1 - v[i, h])
            
            # Calcular densidad como mezcla
            for i in range(n_new):
                for y_idx, y_val in enumerate(y_grid):
                    for h in range(H_sample):
                        density[y_idx, i] += (w[i, h] * 
                            norm.pdf(y_val, theta_mu_sample[h], 
                                   np.sqrt(theta_sigma2_sample[h])))
        
        # Promediar sobre muestras posteriores
        density /= n_samples
        
        return density
    
    def predict_mean(self, X_new, n_samples=100):
        """
        Estima la media condicional E[y | X_new, data]
        
        Parámetros:
        -----------
        X_new : array (n_new, p)
            Covariables para predicción
        n_samples : int
            Número de muestras posteriores a usar
        
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
            predictions[s, :] = np.sum(w * theta_mu_sample[np.newaxis, :H_sample], axis=1)
        
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        
        return mean_pred, std_pred