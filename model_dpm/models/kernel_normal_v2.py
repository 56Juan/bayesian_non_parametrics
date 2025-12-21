import numpy as np
from scipy.stats import norm, invgamma, beta, gamma
import math

class DPMNormal:
    """
    Dirichlet Process Mixture con kernel Normal-Inversa-Gamma.
    Usa Slice Sampling (Walker 2007) con todos los hiperparámetros aleatorios.
    Versión estabilizada numéricamente.
    """
    
    def __init__(self, y, H=20,
                 M_prior=(2.0, 1.0),
                 mu0_prior=(0.0, 100.0),  # Aumentado s0²
                 kappa0_prior=(2.0, 1.0),  # Más informativo
                 a0_prior=(3.0, 1.0),      # Más informativo
                 b0_prior=(2.0, 1.0),
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
        self.alpha_b, self.beta_b = b0_prior
        
        # Propuestas adaptativas para MH
        self.mh_scales = {
            'kappa0': 0.2,
            'a0': 0.2
        }
        self.mh_acceptance = {
            'kappa0': [],
            'a0': []
        }
        
        # Storage para trazas
        self.trace = {
            'M': [], 'mu0': [], 'kappa0': [], 'a0': [], 'b0': [],
            'z': [], 'mu': [], 'sigma2': [], 'w': [], 'n_clusters': []
        }
        
        self.initialize()
    
    def initialize(self):
        """Inicializa todos los parámetros del modelo"""
        
        # Parámetro de concentración
        self.M = np.random.gamma(self.alpha_M, 1.0/self.beta_M)
        
        # Hiperparámetros de G₀ - inicialización robusta
        self.mu0 = np.random.normal(0, 1)  # Cerca del centro de datos normalizados
        self.kappa0 = np.random.gamma(self.alpha_kappa, 1.0/self.beta_kappa)
        self.kappa0 = np.clip(self.kappa0, 0.1, 10.0)  # Bounds razonables
        
        self.a0 = np.random.gamma(self.alpha_a, 1.0/self.beta_a)
        self.a0 = np.clip(self.a0, 1.0, 10.0)  # Asegurar > 0
        
        self.b0 = np.random.gamma(self.alpha_b, 1.0/self.beta_b)
        self.b0 = np.clip(self.b0, 0.1, 10.0)
        
        # Pesos stick-breaking
        self.v = np.random.beta(1, self.M, size=self.H)
        self.w = self._stick_breaking(self.v)
        
        # Parámetros de clusters (muestrear de G₀)
        self.mu = np.zeros(self.H)
        self.sigma2 = np.zeros(self.H)
        for h in range(self.H):
            self.sigma2[h] = invgamma.rvs(self.a0, scale=self.b0)
            self.sigma2[h] = np.clip(self.sigma2[h], 0.01, 100.0)  # Evitar extremos
            self.mu[h] = np.random.normal(self.mu0, math.sqrt(self.sigma2[h] / self.kappa0))
        
        # Asignaciones iniciales
        self.z = np.zeros(self.n, dtype=int)
        for i in range(self.n):
            likes = np.array([norm.pdf(self.y_normalized[i], self.mu[h], 
                                      math.sqrt(self.sigma2[h])) 
                            for h in range(self.H)])
            likes = np.clip(likes, 1e-300, None)  # Evitar underflow
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
        return w / w.sum()  # Normalizar
    
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
            sigma2_new = invgamma.rvs(self.a0, scale=self.b0)
            sigma2_new = np.clip(sigma2_new, 0.01, 100.0)
            mu_new = np.random.normal(self.mu0, math.sqrt(sigma2_new / self.kappa0))
            
            self.sigma2 = np.append(self.sigma2, sigma2_new)
            self.mu = np.append(self.mu, mu_new)
            self.H += 1
            
            # Recalcular pesos
            self.w = self._stick_breaking(self.v)
        
        return u
    
    def update_assignments(self, u):
        """Paso 3: Actualizar asignaciones z_i"""
        for i in range(self.n):
            # Clusters activos (w_h > u_i)
            candidates = np.where(self.w > u[i])[0]
            
            if len(candidates) == 0:
                candidates = np.array([0])
            
            # Likelihood para cada cluster candidato
            likes = norm.pdf(self.y_normalized[i], self.mu[candidates], 
                           np.sqrt(self.sigma2[candidates]))
            likes = np.clip(likes, 1e-300, None)
            probs = likes / likes.sum() if likes.sum() > 0 else np.ones_like(likes) / len(likes)
            
            self.z[i] = np.random.choice(candidates, p=probs)
    
    def update_clusters(self):
        """Paso 4: Actualizar θ_h = (μ_h, σ²_h) usando conjugación NIG"""
        
        for h in range(self.H):
            members = self.y_normalized[self.z == h]
            n_h = len(members)
            
            if n_h > 0:
                # Cluster con observaciones: posterior NIG
                y_bar = members.mean()
                ss = np.sum((members - y_bar)**2)
                
                # Parámetros posteriores
                kappa_n = self.kappa0 + n_h
                mu_n = (self.kappa0 * self.mu0 + n_h * y_bar) / kappa_n
                a_n = self.a0 + n_h / 2.0
                b_n = self.b0 + 0.5 * ss + (self.kappa0 * n_h * (y_bar - self.mu0)**2) / (2 * kappa_n)
                
                # Muestrear σ²_h | y
                self.sigma2[h] = invgamma.rvs(a_n, scale=b_n)
                self.sigma2[h] = np.clip(self.sigma2[h], 0.01, 100.0)
                
                # Muestrear μ_h | σ²_h, y
                self.mu[h] = np.random.normal(mu_n, math.sqrt(self.sigma2[h] / kappa_n))
            
            else:
                # Cluster vacío: muestrear de G₀
                self.sigma2[h] = invgamma.rvs(self.a0, scale=self.b0)
                self.sigma2[h] = np.clip(self.sigma2[h], 0.01, 100.0)
                self.mu[h] = np.random.normal(self.mu0, math.sqrt(self.sigma2[h] / self.kappa0))
    
    def update_weights(self):
        """Paso 5: Actualizar v_h y recalcular w_h"""
        
        # Contar observaciones por cluster
        n_h = np.array([np.sum(self.z == h) for h in range(self.H)])
        
        for h in range(self.H):
            # Suma de observaciones en clusters posteriores
            tail_count = n_h[h+1:].sum() if h+1 < self.H else 0
            
            # Posterior Beta
            self.v[h] = np.random.beta(1 + n_h[h], self.M + tail_count)
        
        # Recalcular pesos
        self.w = self._stick_breaking(self.v)
    
    def update_concentration(self):
        """Paso 6: Actualizar M (Escobar & West style)"""
        
        # log(1 - v_h) para la likelihood
        log_terms = np.sum(np.log(np.clip(1 - self.v, 1e-10, 1.0)))
        
        # Posterior Gamma
        alpha_post = self.alpha_M + self.H - 1
        beta_post = self.beta_M - log_terms
        beta_post = max(beta_post, 0.01)  # Evitar negativos
        
        self.M = np.random.gamma(alpha_post, 1.0/beta_post)
        self.M = np.clip(self.M, 0.01, 100.0)
    
    def update_hyperparameters(self):
        """Paso 7: Actualizar hiperparámetros μ₀, κ₀, a₀, b₀"""
        
        K_eff = len(np.unique(self.z))  # Solo clusters activos
        
        if K_eff == 0:
            return
        
        # Actualizar μ₀ - con estabilización
        active_clusters = np.unique(self.z)
        mu_active = self.mu[active_clusters]
        sigma2_active = self.sigma2[active_clusters]
        
        # Evitar divisiones problemáticas
        inv_sigma2 = np.clip(1.0 / sigma2_active, 1e-6, 1e6)
        
        precision_post = self.kappa0 * np.sum(inv_sigma2) + 1.0 / self.s02
        s0n2 = 1.0 / precision_post
        m0n = s0n2 * (self.kappa0 * np.sum(mu_active * inv_sigma2) + self.m0 / self.s02)
        
        # Bounds check
        s0n2 = np.clip(s0n2, 1e-6, 1e6)
        m0n = np.clip(m0n, -100, 100)
        
        self.mu0 = np.random.normal(m0n, math.sqrt(s0n2))
        self.mu0 = np.clip(self.mu0, -50, 50)
        
        # Actualizar κ₀ (Metropolis-Hastings en log-scale)
        self._update_kappa0()
        
        # Actualizar a₀ y b₀
        self._update_a0()
        self._update_b0()
    
    def _update_kappa0(self):
        """Actualizar κ₀ con Metropolis-Hastings"""
        log_kappa = math.log(self.kappa0)
        log_kappa_prop = np.random.normal(log_kappa, self.mh_scales['kappa0'])
        kappa_prop = math.exp(log_kappa_prop)
        kappa_prop = np.clip(kappa_prop, 0.01, 100.0)
        
        # Log-likelihood - solo clusters activos
        active_clusters = np.unique(self.z)
        mu_active = self.mu[active_clusters]
        sigma2_active = self.sigma2[active_clusters]
        
        diff_sq = (mu_active - self.mu0)**2
        inv_sigma2 = np.clip(1.0 / sigma2_active, 1e-6, 1e6)
        
        log_like_curr = -0.5 * self.kappa0 * np.sum(diff_sq * inv_sigma2)
        log_like_prop = -0.5 * kappa_prop * np.sum(diff_sq * inv_sigma2)
        
        # Log-prior
        log_prior_curr = (self.alpha_kappa - 1) * math.log(self.kappa0) - self.beta_kappa * self.kappa0
        log_prior_prop = (self.alpha_kappa - 1) * math.log(kappa_prop) - self.beta_kappa * kappa_prop
        
        log_r = (log_like_prop + log_prior_prop) - (log_like_curr + log_prior_curr)
        log_r = np.clip(log_r, -50, 50)  # Evitar overflow
        
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
        
        # Log-likelihood - solo clusters activos
        active_clusters = np.unique(self.z)
        sigma2_active = self.sigma2[active_clusters]
        
        # Evitar log(0) o divisiones problemáticas
        ratio = np.clip(self.b0 / sigma2_active, 1e-10, 1e10)
        log_ratio = np.log(ratio)
        
        log_like_curr = self.a0 * np.sum(log_ratio) - len(active_clusters) * math.lgamma(self.a0)
        log_like_prop = a_prop * np.sum(log_ratio) - len(active_clusters) * math.lgamma(a_prop)
        
        # Log-prior
        log_prior_curr = (self.alpha_a - 1) * math.log(self.a0) - self.beta_a * self.a0
        log_prior_prop = (self.alpha_a - 1) * math.log(a_prop) - self.beta_a * a_prop
        
        log_r = (log_like_prop + log_prior_prop) - (log_like_curr + log_prior_curr)
        log_r = np.clip(log_r, -50, 50)
        
        accept = math.log(np.random.rand()) < log_r
        if accept:
            self.a0 = a_prop
        
        self.mh_acceptance['a0'].append(float(accept))
    
    def _update_b0(self):
        """Actualizar b₀ con Gibbs (conjugado)"""
        active_clusters = np.unique(self.z)
        sigma2_active = self.sigma2[active_clusters]
        
        alpha_post = self.alpha_b + len(active_clusters) * self.a0
        inv_sigma2 = np.clip(1.0 / sigma2_active, 1e-6, 1e6)
        beta_post = self.beta_b + np.sum(inv_sigma2)
        
        self.b0 = np.random.gamma(alpha_post, 1.0/beta_post)
        self.b0 = np.clip(self.b0, 0.01, 100.0)
    
    def adapt_mh_scales(self, iteration):
        """Adaptar escalas de propuestas MH (durante burn-in)"""
        if iteration > 50 and iteration % 50 == 0:
            for param in ['kappa0', 'a0']:
                recent = self.mh_acceptance[param][-50:]
                if len(recent) > 0:
                    acc_rate = np.mean(recent)
                    # Target: 0.234 para propuestas normales
                    if acc_rate < 0.15:
                        self.mh_scales[param] *= 0.8
                    elif acc_rate > 0.4:
                        self.mh_scales[param] *= 1.2
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
                sigma2_original = self.sigma2 * (self.y_std ** 2)
                
                self.trace['M'].append(self.M)
                self.trace['mu0'].append(self.mu0 * self.y_std + self.y_mean)
                self.trace['kappa0'].append(self.kappa0)
                self.trace['a0'].append(self.a0)
                self.trace['b0'].append(self.b0 * (self.y_std ** 2))
                self.trace['z'].append(self.z.copy())
                self.trace['mu'].append(mu_original.copy())
                self.trace['sigma2'].append(sigma2_original.copy())
                self.trace['w'].append(self.w.copy())
                self.trace['n_clusters'].append(active_clusters)
            
            # Verbose
            if self.verbose and (it + 1) % 100 == 0:
                active = len(np.unique(self.z))
                acc_k = np.mean(self.mh_acceptance['kappa0'][-100:]) if self.mh_acceptance['kappa0'] else 0
                acc_a = np.mean(self.mh_acceptance['a0'][-100:]) if self.mh_acceptance['a0'] else 0
                print(f"Iter {it+1}/{iterations}: K_eff={active}, M={self.M:.2f}, "
                      f"μ₀={self.mu0:.2f}, κ₀={self.kappa0:.2f}, a₀={self.a0:.2f}, b₀={self.b0:.2f} "
                      f"[Acc: κ={acc_k:.2f}, a={acc_a:.2f}]")
        
        return self.trace
    
    def get_posterior_summary(self):
        """Calcula resúmenes de la posterior"""
        summary = {
            'M': (np.mean(self.trace['M']), np.std(self.trace['M'])),
            'mu0': (np.mean(self.trace['mu0']), np.std(self.trace['mu0'])),
            'kappa0': (np.mean(self.trace['kappa0']), np.std(self.trace['kappa0'])),
            'a0': (np.mean(self.trace['a0']), np.std(self.trace['a0'])),
            'b0': (np.mean(self.trace['b0']), np.std(self.trace['b0'])),
            'n_clusters': (np.mean(self.trace['n_clusters']), np.std(self.trace['n_clusters']))
        }
        return summary