import numpy as np
from scipy.stats import laplace, gamma as gamma_dist, norm
import math

class DPMLaplace:
    """
    Dirichlet Process Mixture con kernel Laplace.
    Usa Slice Sampling (Walker 2007) con todos los hiperparámetros aleatorios.
    
    Jerarquía:
    y_i | θ_i ~ Laplace(μ_i, b_i)
    θ_i = (μ_i, b_i) | G ~ G
    G | M, G₀ ~ DP(M, G₀)
    
    Base measure G₀:
    μ | b ~ N(μ₀, b/κ₀)
    b ~ Gamma(a₀, β₀)
    
    Hiperparámetros aleatorios:
    M ~ Gamma(α_M, β_M)
    μ₀ ~ N(m₀, s₀²)
    κ₀ ~ Gamma(α_κ, β_κ)
    a₀ ~ Gamma(α_a, β_a)
    β₀ ~ Gamma(α_β, β_β)
    """
    
    def __init__(self, y, H=20,
                 M_prior=(2.0, 1.0),
                 mu0_prior=(0.0, 100.0),
                 kappa0_prior=(2.0, 1.0),
                 a0_prior=(3.0, 1.0),
                 beta0_prior=(2.0, 1.0),
                 verbose=True):
        """
        Parámetros:
        -----------
        y : array
            Datos observados
        H : int
            Truncamiento stick-breaking inicial
        """
        self.y = np.array(y)
        self.n = len(y)
        self.H = H
        self.verbose = verbose
        
        # Normalizar datos para estabilidad
        self.y_mean = np.mean(y)
        self.y_std = np.std(y)
        self.y_normalized = (self.y - self.y_mean) / self.y_std
        
        # Hiperpriors
        self.alpha_M, self.beta_M = M_prior
        self.m0, self.s02 = mu0_prior
        self.alpha_kappa, self.beta_kappa = kappa0_prior
        self.alpha_a, self.beta_a = a0_prior
        self.alpha_beta, self.beta_beta = beta0_prior
        
        # Propuestas adaptativas para MH
        self.mh_scales = {
            'mu': 0.3,      # Para μ_h en clusters
            'b': 0.2,       # Para b_h en clusters (log-scale)
            'kappa0': 0.2,
            'a0': 0.2,
            'beta0': 0.2
        }
        self.mh_acceptance = {
            'mu': [],
            'b': [],
            'kappa0': [],
            'a0': [],
            'beta0': []
        }
        
        # Storage para trazas
        self.trace = {
            'M': [], 'mu0': [], 'kappa0': [], 'a0': [], 'beta0': [],
            'z': [], 'mu': [], 'b': [], 'w': [], 'n_clusters': []
        }
        
        self.initialize()
    
    def initialize(self):
        """Inicializa todos los parámetros del modelo"""
        
        # Parámetro de concentración
        self.M = np.random.gamma(self.alpha_M, 1.0/self.beta_M)
        
        # Hiperparámetros de G₀
        self.mu0 = np.random.normal(0, 1)
        self.kappa0 = np.random.gamma(self.alpha_kappa, 1.0/self.beta_kappa)
        self.kappa0 = np.clip(self.kappa0, 0.1, 10.0)
        
        self.a0 = np.random.gamma(self.alpha_a, 1.0/self.beta_a)
        self.a0 = np.clip(self.a0, 1.0, 10.0)
        
        self.beta0 = np.random.gamma(self.alpha_beta, 1.0/self.beta_beta)
        self.beta0 = np.clip(self.beta0, 0.1, 10.0)
        
        # Pesos stick-breaking
        self.v = np.random.beta(1, self.M, size=self.H)
        self.w = self._stick_breaking(self.v)
        
        # Parámetros de clusters (muestrear de G₀)
        self.mu = np.zeros(self.H)
        self.b = np.zeros(self.H)
        for h in range(self.H):
            # Primero b, luego μ | b
            self.b[h] = gamma_dist.rvs(self.a0, scale=1.0/self.beta0)
            self.b[h] = np.clip(self.b[h], 0.01, 100.0)
            self.mu[h] = np.random.normal(self.mu0, math.sqrt(self.b[h] / self.kappa0))
        
        # Asignaciones iniciales
        self.z = np.zeros(self.n, dtype=int)
        for i in range(self.n):
            likes = np.array([laplace.pdf(self.y_normalized[i], self.mu[h], self.b[h]) 
                            for h in range(self.H)])
            likes = np.clip(likes, 1e-300, None)
            probs = self.w * likes
            probs /= probs.sum()
            self.z[i] = np.random.choice(self.H, p=probs)
    
    def _stick_breaking(self, v):
        """Convierte v_h en pesos w_h mediante stick-breaking"""
        w = np.zeros_like(v)
        remaining = 1.0
        for h in range(len(v)):
            w[h] = v[h] * remaining
            remaining *= (1 - v[h])
        return w / w.sum()
    
    def sample_slice_variables(self):
        """Paso 2: Generar variables auxiliares u_i"""
        u = np.array([np.random.uniform(0, self.w[self.z[i]]) for i in range(self.n)])
        
        # Expandir truncamiento si es necesario
        u_min = u.min()
        while self.w.sum() < 1 - u_min or len(self.w) < 3:
            # Agregar nuevos sticks
            v_new = np.random.beta(1, self.M)
            self.v = np.append(self.v, v_new)
            
            # Muestrear nuevo cluster de G₀
            b_new = gamma_dist.rvs(self.a0, scale=1.0/self.beta0)
            b_new = np.clip(b_new, 0.01, 100.0)
            mu_new = np.random.normal(self.mu0, math.sqrt(b_new / self.kappa0))
            
            self.b = np.append(self.b, b_new)
            self.mu = np.append(self.mu, mu_new)
            self.H += 1
            
            # Recalcular pesos
            self.w = self._stick_breaking(self.v)
        
        return u
    
    def update_assignments(self, u):
        """Paso 3: Actualizar asignaciones z_i"""
        for i in range(self.n):
            candidates = np.where(self.w > u[i])[0]
            
            if len(candidates) == 0:
                candidates = np.array([0])
            
            # Likelihood Laplace
            likes = laplace.pdf(self.y_normalized[i], self.mu[candidates], self.b[candidates])
            likes = np.clip(likes, 1e-300, None)
            probs = likes / likes.sum() if likes.sum() > 0 else np.ones_like(likes) / len(likes)
            
            self.z[i] = np.random.choice(candidates, p=probs)
    
    def update_clusters(self):
        """
        Paso 4: Actualizar θ_h = (μ_h, b_h) con Metropolis-Hastings
        No hay conjugación exacta para Laplace
        """
        
        for h in range(self.H):
            members = self.y_normalized[self.z == h]
            n_h = len(members)
            
            if n_h > 0:
                # Cluster con observaciones: MH para μ_h y b_h
                
                # Update μ_h | b_h, y
                mu_prop = np.random.normal(self.mu[h], self.mh_scales['mu'])
                
                # Log-likelihood Laplace: -|y - μ|/b - log(2b)
                log_like_curr = -np.sum(np.abs(members - self.mu[h]) / self.b[h])
                log_like_prop = -np.sum(np.abs(members - mu_prop) / self.b[h])
                
                # Log-prior: μ | b ~ N(μ₀, b/κ₀)
                log_prior_curr = -0.5 * self.kappa0 * (self.mu[h] - self.mu0)**2 / self.b[h]
                log_prior_prop = -0.5 * self.kappa0 * (mu_prop - self.mu0)**2 / self.b[h]
                
                log_r = (log_like_prop + log_prior_prop) - (log_like_curr + log_prior_curr)
                log_r = np.clip(log_r, -50, 50)
                
                accept_mu = math.log(np.random.rand()) < log_r
                if accept_mu:
                    self.mu[h] = mu_prop
                self.mh_acceptance['mu'].append(float(accept_mu))
                
                # Update b_h | μ_h, y (en log-scale)
                log_b = math.log(self.b[h])
                log_b_prop = np.random.normal(log_b, self.mh_scales['b'])
                b_prop = math.exp(log_b_prop)
                b_prop = np.clip(b_prop, 0.01, 100.0)
                
                # Log-likelihood Laplace
                log_like_curr = -np.sum(np.abs(members - self.mu[h]) / self.b[h]) - n_h * math.log(2 * self.b[h])
                log_like_prop = -np.sum(np.abs(members - self.mu[h]) / b_prop) - n_h * math.log(2 * b_prop)
                
                # Log-prior: b ~ Gamma(a₀, β₀) y μ | b ~ N(μ₀, b/κ₀)
                log_prior_b_curr = (self.a0 - 1) * math.log(self.b[h]) - self.beta0 * self.b[h]
                log_prior_b_prop = (self.a0 - 1) * math.log(b_prop) - self.beta0 * b_prop
                
                log_prior_mu_curr = -0.5 * math.log(self.b[h]) - 0.5 * self.kappa0 * (self.mu[h] - self.mu0)**2 / self.b[h]
                log_prior_mu_prop = -0.5 * math.log(b_prop) - 0.5 * self.kappa0 * (self.mu[h] - self.mu0)**2 / b_prop
                
                log_r = (log_like_prop + log_prior_b_prop + log_prior_mu_prop) - \
                        (log_like_curr + log_prior_b_curr + log_prior_mu_curr)
                log_r = np.clip(log_r, -50, 50)
                
                accept_b = math.log(np.random.rand()) < log_r
                if accept_b:
                    self.b[h] = b_prop
                self.mh_acceptance['b'].append(float(accept_b))
            
            else:
                # Cluster vacío: muestrear de G₀
                self.b[h] = gamma_dist.rvs(self.a0, scale=1.0/self.beta0)
                self.b[h] = np.clip(self.b[h], 0.01, 100.0)
                self.mu[h] = np.random.normal(self.mu0, math.sqrt(self.b[h] / self.kappa0))
    
    def update_weights(self):
        """Paso 5: Actualizar v_h y recalcular w_h"""
        n_h = np.array([np.sum(self.z == h) for h in range(self.H)])
        
        for h in range(self.H):
            tail_count = n_h[h+1:].sum() if h+1 < self.H else 0
            self.v[h] = np.random.beta(1 + n_h[h], self.M + tail_count)
        
        self.w = self._stick_breaking(self.v)
    
    def update_concentration(self):
        """Paso 6: Actualizar M"""
        log_terms = np.sum(np.log(np.clip(1 - self.v, 1e-10, 1.0)))
        
        alpha_post = self.alpha_M + self.H - 1
        beta_post = self.beta_M - log_terms
        beta_post = max(beta_post, 0.01)
        
        self.M = np.random.gamma(alpha_post, 1.0/beta_post)
        self.M = np.clip(self.M, 0.01, 100.0)
    
    def update_hyperparameters(self):
        """Paso 7: Actualizar hiperparámetros μ₀, κ₀, a₀, β₀"""
        
        active_clusters = np.unique(self.z)
        if len(active_clusters) == 0:
            return
        
        mu_active = self.mu[active_clusters]
        b_active = self.b[active_clusters]
        
        # Actualizar μ₀ con Gibbs (conjugado condicional)
        inv_b = np.clip(1.0 / b_active, 1e-6, 1e6)
        precision_post = self.kappa0 * np.sum(inv_b) + 1.0 / self.s02
        s0n2 = 1.0 / precision_post
        m0n = s0n2 * (self.kappa0 * np.sum(mu_active * inv_b) + self.m0 / self.s02)
        
        s0n2 = np.clip(s0n2, 1e-6, 1e6)
        m0n = np.clip(m0n, -100, 100)
        
        self.mu0 = np.random.normal(m0n, math.sqrt(s0n2))
        self.mu0 = np.clip(self.mu0, -50, 50)
        
        # Actualizar κ₀, a₀, β₀ con MH
        self._update_kappa0()
        self._update_a0()
        self._update_beta0()
    
    def _update_kappa0(self):
        """Actualizar κ₀ con Metropolis-Hastings"""
        log_kappa = math.log(self.kappa0)
        log_kappa_prop = np.random.normal(log_kappa, self.mh_scales['kappa0'])
        kappa_prop = math.exp(log_kappa_prop)
        kappa_prop = np.clip(kappa_prop, 0.01, 100.0)
        
        active_clusters = np.unique(self.z)
        mu_active = self.mu[active_clusters]
        b_active = self.b[active_clusters]
        
        diff_sq = (mu_active - self.mu0)**2
        inv_b = np.clip(1.0 / b_active, 1e-6, 1e6)
        
        # Log-likelihood: producto de normales N(μ₀, b/κ)
        log_like_curr = -0.5 * len(active_clusters) * math.log(self.kappa0) - \
                        0.5 * self.kappa0 * np.sum(diff_sq * inv_b)
        log_like_prop = -0.5 * len(active_clusters) * math.log(kappa_prop) - \
                        0.5 * kappa_prop * np.sum(diff_sq * inv_b)
        
        # Log-prior
        log_prior_curr = (self.alpha_kappa - 1) * math.log(self.kappa0) - self.beta_kappa * self.kappa0
        log_prior_prop = (self.alpha_kappa - 1) * math.log(kappa_prop) - self.beta_kappa * kappa_prop
        
        log_r = (log_like_prop + log_prior_prop) - (log_like_curr + log_prior_curr)
        log_r = np.clip(log_r, -50, 50)
        
        accept = math.log(np.random.rand()) < log_r
        if accept:
            self.kappa0 = kappa_prop
        
        self.mh_acceptance['kappa0'].append(float(accept))
    
    def _update_a0(self):
        """Actualizar a₀ con Metropolis-Hastings"""
        log_a = math.log(self.a0)
        log_a_prop = np.random.normal(log_a, self.mh_scales['a0'])
        a_prop = math.exp(log_a_prop)
        a_prop = np.clip(a_prop, 0.5, 20.0)
        
        active_clusters = np.unique(self.z)
        b_active = self.b[active_clusters]
        
        # Log-likelihood: producto de Gamma(a₀, β₀)
        log_like_curr = len(active_clusters) * (self.a0 * math.log(self.beta0) - math.lgamma(self.a0)) + \
                        (self.a0 - 1) * np.sum(np.log(b_active)) - self.beta0 * np.sum(b_active)
        log_like_prop = len(active_clusters) * (a_prop * math.log(self.beta0) - math.lgamma(a_prop)) + \
                        (a_prop - 1) * np.sum(np.log(b_active)) - self.beta0 * np.sum(b_active)
        
        # Log-prior
        log_prior_curr = (self.alpha_a - 1) * math.log(self.a0) - self.beta_a * self.a0
        log_prior_prop = (self.alpha_a - 1) * math.log(a_prop) - self.beta_a * a_prop
        
        log_r = (log_like_prop + log_prior_prop) - (log_like_curr + log_prior_curr)
        log_r = np.clip(log_r, -50, 50)
        
        accept = math.log(np.random.rand()) < log_r
        if accept:
            self.a0 = a_prop
        
        self.mh_acceptance['a0'].append(float(accept))
    
    def _update_beta0(self):
        """Actualizar β₀ con Gibbs (conjugado)"""
        active_clusters = np.unique(self.z)
        b_active = self.b[active_clusters]
        
        alpha_post = self.alpha_beta + len(active_clusters) * self.a0
        beta_post = self.beta_beta + np.sum(b_active)
        
        self.beta0 = np.random.gamma(alpha_post, 1.0/beta_post)
        self.beta0 = np.clip(self.beta0, 0.01, 100.0)
    
    def adapt_mh_scales(self, iteration):
        """Adaptar escalas de propuestas MH (durante burn-in)"""
        if iteration > 50 and iteration % 50 == 0:
            for param in ['mu', 'b', 'kappa0', 'a0', 'beta0']:
                recent = self.mh_acceptance[param][-50:]
                if len(recent) > 0:
                    acc_rate = np.mean(recent)
                    if acc_rate < 0.15:
                        self.mh_scales[param] *= 0.8
                    elif acc_rate > 0.4:
                        self.mh_scales[param] *= 1.2
                    
                    # Diferentes rangos según el parámetro
                    if param in ['mu']:
                        self.mh_scales[param] = np.clip(self.mh_scales[param], 0.01, 2.0)
                    else:
                        self.mh_scales[param] = np.clip(self.mh_scales[param], 0.01, 1.0)
    
    def run(self, iterations=1000, burnin=500):
        """
        Ejecuta la cadena de Markov con Slice Sampling
        """
        
        for it in range(iterations):
            # Ciclo principal de Slice Sampling
            u = self.sample_slice_variables()
            self.update_assignments(u)
            self.update_clusters()
            self.update_weights()
            self.update_concentration()
            self.update_hyperparameters()
            
            # Adaptar propuestas durante burn-in
            if it < burnin:
                self.adapt_mh_scales(it)
            
            # Guardar trazas (después del burnin)
            if it >= burnin:
                active_clusters = len(np.unique(self.z))
                
                # Desnormalizar parámetros para guardar
                mu_original = self.mu * self.y_std + self.y_mean
                b_original = self.b * self.y_std
                
                self.trace['M'].append(self.M)
                self.trace['mu0'].append(self.mu0 * self.y_std + self.y_mean)
                self.trace['kappa0'].append(self.kappa0)
                self.trace['a0'].append(self.a0)
                self.trace['beta0'].append(self.beta0 / self.y_std)
                self.trace['z'].append(self.z.copy())
                self.trace['mu'].append(mu_original.copy())
                self.trace['b'].append(b_original.copy())
                self.trace['w'].append(self.w.copy())
                self.trace['n_clusters'].append(active_clusters)
            
            # Verbose
            if self.verbose and (it + 1) % 100 == 0:
                active = len(np.unique(self.z))
                acc_mu = np.mean(self.mh_acceptance['mu'][-100:]) if self.mh_acceptance['mu'] else 0
                acc_b = np.mean(self.mh_acceptance['b'][-100:]) if self.mh_acceptance['b'] else 0
                acc_k = np.mean(self.mh_acceptance['kappa0'][-100:]) if self.mh_acceptance['kappa0'] else 0
                acc_a = np.mean(self.mh_acceptance['a0'][-100:]) if self.mh_acceptance['a0'] else 0
                print(f"Iter {it+1}/{iterations}: K_eff={active}, M={self.M:.2f}, "
                      f"μ₀={self.mu0:.2f}, κ₀={self.kappa0:.2f}, a₀={self.a0:.2f}, β₀={self.beta0:.2f}")
                print(f"  [Acc: μ={acc_mu:.2f}, b={acc_b:.2f}, κ={acc_k:.2f}, a={acc_a:.2f}]")
        
        return self.trace
    
    def get_posterior_summary(self):
        """Calcula resúmenes de la posterior"""
        summary = {
            'M': (np.mean(self.trace['M']), np.std(self.trace['M'])),
            'mu0': (np.mean(self.trace['mu0']), np.std(self.trace['mu0'])),
            'kappa0': (np.mean(self.trace['kappa0']), np.std(self.trace['kappa0'])),
            'a0': (np.mean(self.trace['a0']), np.std(self.trace['a0'])),
            'beta0': (np.mean(self.trace['beta0']), np.std(self.trace['beta0'])),
            'n_clusters': (np.mean(self.trace['n_clusters']), np.std(self.trace['n_clusters']))
        }
        return summary