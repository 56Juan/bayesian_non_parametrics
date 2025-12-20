import numpy as np
from scipy.stats import norm, invgamma, beta, gamma
import math

class DPMNormal:
    """
    Dirichlet Process Mixture con kernel Normal-Inversa-Gamma.
    Usa Slice Sampling (Walker 2007) con todos los hiperparámetros aleatorios.
    
    Jerarquía:
    y_i | θ_i ~ N(μ_i, σ²_i)
    θ_i = (μ_i, σ²_i) | G ~ G
    G | M, G₀ ~ DP(M, G₀)
    G₀ = NIG(μ₀, κ₀, a₀, b₀)
    M ~ Gamma(α_M, β_M)
    μ₀ ~ N(m₀, s₀²)
    κ₀ ~ Gamma(α_κ, β_κ)
    a₀ ~ Gamma(α_a, β_a)
    b₀ ~ Gamma(α_b, β_b)
    """
    
    def __init__(self, y, H=20,
                 M_prior=(2.0, 1.0),
                 mu0_prior=(0.0, 10.0),
                 kappa0_prior=(1.0, 1.0),
                 a0_prior=(2.0, 1.0),
                 b0_prior=(2.0, 1.0),
                 verbose=True):
        """
        Parámetros:
        -----------
        y : array
            Datos observados
        H : int
            Truncamiento stick-breaking inicial
        M_prior : tuple
            (α_M, β_M) para Gamma prior del parámetro de concentración
        mu0_prior : tuple
            (m₀, s₀²) para Normal prior de μ₀
        kappa0_prior : tuple
            (α_κ, β_κ) para Gamma prior de κ₀
        a0_prior : tuple
            (α_a, β_a) para Gamma prior de a₀
        b0_prior : tuple
            (α_b, β_b) para Gamma prior de b₀
        """
        self.y = np.array(y)
        self.n = len(y)
        self.H = H
        self.verbose = verbose
        
        # Hiperpriors
        self.alpha_M, self.beta_M = M_prior
        self.m0, self.s02 = mu0_prior
        self.alpha_kappa, self.beta_kappa = kappa0_prior
        self.alpha_a, self.beta_a = a0_prior
        self.alpha_b, self.beta_b = b0_prior
        
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
        
        # Hiperparámetros de G₀
        self.mu0 = np.random.normal(self.m0, math.sqrt(self.s02))
        self.kappa0 = np.random.gamma(self.alpha_kappa, 1.0/self.beta_kappa)
        self.a0 = np.random.gamma(self.alpha_a, 1.0/self.beta_a)
        self.b0 = np.random.gamma(self.alpha_b, 1.0/self.beta_b)
        
        # Pesos stick-breaking
        self.v = np.random.beta(1, self.M, size=self.H)
        self.w = self._stick_breaking(self.v)
        
        # Parámetros de clusters (muestrear de G₀)
        self.mu = np.zeros(self.H)
        self.sigma2 = np.zeros(self.H)
        for h in range(self.H):
            self.sigma2[h] = invgamma.rvs(self.a0, scale=self.b0)
            self.mu[h] = np.random.normal(self.mu0, math.sqrt(self.sigma2[h] / self.kappa0))
        
        # Asignaciones iniciales (usando probabilidades basadas en likelihood)
        self.z = np.zeros(self.n, dtype=int)
        for i in range(self.n):
            likes = np.array([norm.pdf(self.y[i], self.mu[h], math.sqrt(self.sigma2[h])) 
                            for h in range(self.H)])
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
            likes = norm.pdf(self.y[i], self.mu[candidates], np.sqrt(self.sigma2[candidates]))
            probs = likes / likes.sum() if likes.sum() > 0 else np.ones_like(likes) / len(likes)
            
            self.z[i] = np.random.choice(candidates, p=probs)
    
    def update_clusters(self):
        """Paso 4: Actualizar θ_h = (μ_h, σ²_h) usando conjugación NIG"""
        
        for h in range(self.H):
            members = self.y[self.z == h]
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
                
                # Muestrear μ_h | σ²_h, y
                self.mu[h] = np.random.normal(mu_n, math.sqrt(self.sigma2[h] / kappa_n))
            
            else:
                # Cluster vacío: muestrear de G₀
                self.sigma2[h] = invgamma.rvs(self.a0, scale=self.b0)
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
        log_terms = np.sum(np.log(1 - self.v + 1e-16))
        
        # Posterior Gamma
        alpha_post = self.alpha_M + self.H - 1
        beta_post = self.beta_M - log_terms
        
        self.M = np.random.gamma(alpha_post, 1.0/beta_post)
    
    def update_hyperparameters(self):
        """Paso 7: Actualizar hiperparámetros μ₀, κ₀, a₀, b₀"""
        
        K_eff = self.H
        
        # Actualizar μ₀
        s0n2 = 1.0 / (K_eff * self.kappa0 / self.sigma2.sum() + 1.0 / self.s02)
        m0n = s0n2 * (self.kappa0 * np.sum(self.mu / self.sigma2) + self.m0 / self.s02)
        self.mu0 = np.random.normal(m0n, math.sqrt(s0n2))
        
        # Actualizar κ₀ (Metropolis-Hastings en log-scale)
        log_kappa = math.log(self.kappa0)
        log_kappa_prop = np.random.normal(log_kappa, 0.1)
        kappa_prop = math.exp(log_kappa_prop)
        
        # Log-likelihood
        log_like_curr = -0.5 * self.kappa0 * np.sum((self.mu - self.mu0)**2 / self.sigma2)
        log_like_prop = -0.5 * kappa_prop * np.sum((self.mu - self.mu0)**2 / self.sigma2)
        
        # Log-prior
        log_prior_curr = (self.alpha_kappa - 1) * math.log(self.kappa0) - self.beta_kappa * self.kappa0
        log_prior_prop = (self.alpha_kappa - 1) * math.log(kappa_prop) - self.beta_kappa * kappa_prop
        
        log_r = (log_like_prop + log_prior_prop) - (log_like_curr + log_prior_curr)
        
        if math.log(np.random.rand()) < log_r:
            self.kappa0 = kappa_prop
        
        # Actualizar a₀ y b₀ (MH también)
        self._update_a0()
        self._update_b0()
    
    def _update_a0(self):
        """Actualizar a₀ con Metropolis-Hastings"""
        log_a = math.log(self.a0)
        log_a_prop = np.random.normal(log_a, 0.1)
        a_prop = math.exp(log_a_prop)
        
        # Log-likelihood
        log_like_curr = self.a0 * np.sum(np.log(self.b0 / self.sigma2)) - np.sum(np.log(gamma_func(self.a0)))
        log_like_prop = a_prop * np.sum(np.log(self.b0 / self.sigma2)) - np.sum(np.log(gamma_func(a_prop)))
        
        # Log-prior
        log_prior_curr = (self.alpha_a - 1) * math.log(self.a0) - self.beta_a * self.a0
        log_prior_prop = (self.alpha_a - 1) * math.log(a_prop) - self.beta_a * a_prop
        
        log_r = (log_like_prop + log_prior_prop) - (log_like_curr + log_prior_curr)
        
        if math.log(np.random.rand()) < log_r:
            self.a0 = a_prop
    
    def _update_b0(self):
        """Actualizar b₀ con Gibbs (conjugado)"""
        alpha_post = self.alpha_b + self.H * self.a0
        beta_post = self.beta_b + np.sum(1.0 / self.sigma2)
        
        self.b0 = np.random.gamma(alpha_post, 1.0/beta_post)
    
    def run(self, iterations=1000, burnin=500):
        """
        Ejecuta la cadena de Markov con Slice Sampling
        
        Parámetros:
        -----------
        iterations : int
            Número total de iteraciones
        burnin : int
            Iteraciones de burn-in a descartar
        """
        
        for it in range(iterations):
            # Ciclo principal de Slice Sampling
            u = self.sample_slice_variables()
            self.update_assignments(u)
            self.update_clusters()
            self.update_weights()
            self.update_concentration()
            self.update_hyperparameters()
            
            # Guardar trazas (después del burnin)
            if it >= burnin:
                active_clusters = len(np.unique(self.z))
                
                self.trace['M'].append(self.M)
                self.trace['mu0'].append(self.mu0)
                self.trace['kappa0'].append(self.kappa0)
                self.trace['a0'].append(self.a0)
                self.trace['b0'].append(self.b0)
                self.trace['z'].append(self.z.copy())
                self.trace['mu'].append(self.mu.copy())
                self.trace['sigma2'].append(self.sigma2.copy())
                self.trace['w'].append(self.w.copy())
                self.trace['n_clusters'].append(active_clusters)
            
            # Verbose
            if self.verbose and (it + 1) % 100 == 0:
                active = len(np.unique(self.z))
                print(f"Iter {it+1}/{iterations}: K_eff={active}, M={self.M:.2f}, "
                      f"μ₀={self.mu0:.2f}, κ₀={self.kappa0:.2f}, a₀={self.a0:.2f}, b₀={self.b0:.2f}")
        
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


def gamma_func(x):
    """Función gamma usando lgamma"""
    return math.exp(math.lgamma(x))


# Ejemplo de uso
#if __name__ == "__main__":
#    # Datos simulados: mezcla de 3 normales
#    np.random.seed(42)
#    data = np.concatenate([
#        np.random.normal(-5, 1, 50),
#        np.random.normal(0, 0.5, 100),
#        np.random.normal(5, 1.5, 50)
#    ])
#    
#    # Crear modelo
#    model = DPMNormal(data, H=15, verbose=True)
#    
#    # Ejecutar
#    trace = model.run(iterations=1000, burnin=500)
#    
#    # Resumen
#    summary = model.get_posterior_summary()
#    print("\n=== Resumen Posterior ===")
#    for param, (mean, std) in summary.items():
#        print(f"{param}: {mean:.3f} ± {std:.3f}")