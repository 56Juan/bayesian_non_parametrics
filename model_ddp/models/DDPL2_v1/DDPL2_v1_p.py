import numpy as np
from scipy.stats import norm, invgamma, gamma, wishart, invwishart, beta
from scipy.interpolate import BSpline
import math

# Importar el módulo C++
try:
    from . import ddp2_cpp
    CPP_AVAILABLE = True
    print("Implementacion en C++ Exitosa")
except ImportError as e:
    CPP_AVAILABLE = False
    _IMPORT_ERROR = e
    print("Implementacion en C++ Fallida")


class DDPLinearSpline2:
    """
    Dependent Dirichlet Process con regresión lineal en B-splines.
    
    Modelo jerárquico completo:
    ---------------------------
    Likelihood:
        y_i | z_i=h, λ_h, ξ_h ~ N(μ_h(x_i), σ²_h(x_i))
        μ_h(x_i) = λ_h' d(x_i)
        log(σ²_h(x_i)) = ξ_h' d(x_i)
    
    Asignaciones:
        z_i | {w_h} ~ Categorical(w_1, ..., w_T)
    
    Pesos stick-breaking:
        w_h = v_h * ∏_{ℓ<h} (1 - v_ℓ)
        v_h ~ Beta(1, M)
        M ~ Gamma(a_M, b_M)
    
    Coeficientes (átomos):
        λ_h | μ_λ, Σ_λ ~ N_K(μ_λ, Σ_λ)
        ξ_h | μ_ξ, Σ_ξ ~ N_K(μ_ξ, Σ_ξ)
    
    Priors jerárquicos (NIW):
        μ_λ | Σ_λ, m_λ, κ_λ ~ N_K(m_λ, Σ_λ/κ_λ)
        Σ_λ ~ Inv-Wishart(ν_λ, Ψ_λ)
        
        μ_ξ | Σ_ξ, m_ξ, κ_ξ ~ N_K(m_ξ, Σ_ξ/κ_ξ)
        Σ_ξ ~ Inv-Wishart(ν_ξ, Ψ_ξ)
    
    Hiperparámetros Nivel 1 (aleatorios):
        m_λ ~ N_K(μ_m, Σ_m)
        κ_λ ~ Gamma(α_κ, β_κ)
        ν_λ ~ Gamma(α_ν, β_ν)
        Ψ_λ ~ Wishart(ν_Ψ, Ω_Ψ)
        
        m_ξ ~ N_K(μ_m, Σ_m)
        κ_ξ ~ Gamma(α_κ, β_κ)
        ν_ξ ~ Gamma(α_ν, β_ν)
        Ψ_ξ ~ Wishart(ν_Ψ, Ω_Ψ)
        
        a_M ~ Gamma(α_aM, β_aM)
        b_M ~ Gamma(α_bM, β_bM)
    
    Hiperparámetros Nivel 2 (fijos):
        μ_m = 0_K, Σ_m = 100·I_K
        α_κ = 2, β_κ = 1
        α_ν = 2, β_ν = 0.1
        ν_Ψ = K + 2, Ω_Ψ = I_K
        α_aM = 2, β_aM = 1
        α_bM = 1, β_bM = 1
    
    Splines (configuración fija):
        d(x_i) = [1, b₁(x_i1)', ..., b_p(x_ip)']'
        Knots en cuantiles, grado fijo, número de knots fijo
    """
    
    def __init__(self, y, X, H=15,
                 degree=3,
                 n_knots=5,
                 mu_m=None,
                 Sigma_m=None,
                 alpha_kappa=2.0,
                 beta_kappa=1.0,
                 alpha_nu=2.0,
                 beta_nu=0.1,
                 nu_Psi=None,
                 Omega_Psi=None,
                 alpha_aM=2.0,
                 beta_aM=1.0,
                 alpha_bM=1.0,
                 beta_bM=1.0,
                 verbose=True):
        """
        Inicializa el modelo DDP con splines.
        
        Parámetros:
        -----------
        y : array (n,)
            Variable respuesta
        X : array (n, p)
            Covariables
        H : int
            Número inicial de clusters (truncamiento)
        degree : int
            Grado de B-splines (1=lineal, 2=cuadrático, 3=cúbico)
        n_knots : int
            Número de knots internos por covariable
        mu_m, Sigma_m : array, array
            Hiperparámetros Nivel 2 para m_λ y m_ξ
        alpha_kappa, beta_kappa : float
            Hiperparámetros Nivel 2 para κ_λ y κ_ξ
        alpha_nu, beta_nu : float
            Hiperparámetros Nivel 2 para ν_λ y ν_ξ
        nu_Psi, Omega_Psi : float, array
            Hiperparámetros Nivel 2 para Ψ_λ y Ψ_ξ
        alpha_aM, beta_aM : float
            Hiperparámetros Nivel 2 para a_M
        alpha_bM, beta_bM : float
            Hiperparámetros Nivel 2 para b_M
        verbose : bool
            Imprimir mensajes de progreso
        """
        
        # Datos
        self.y = np.array(y)
        self.X = np.array(X)
        self.n, self.p = self.X.shape
        self.H = H
        self.verbose = verbose
        
        # Normalización para estabilidad numérica
        self.y_mean = np.mean(y)
        self.y_std = np.std(y)
        self.y_normalized = (self.y - self.y_mean) / self.y_std
        
        self.X_mean = np.mean(X, axis=0)
        self.X_std = np.std(X, axis=0)
        self.X_normalized = (self.X - self.X_mean) / self.X_std
        
        # Parámetros de splines (FIJOS durante MCMC)
        self.degree = degree
        self.n_knots = n_knots
        
        # Construir bases B-spline
        self.spline_bases = self._build_spline_bases()
        self.K = self.spline_bases['K']  # Dimensión expandida
        
        # Hiperparámetros Nivel 2 (FIJOS)
        self.mu_m = mu_m if mu_m is not None else np.zeros(self.K)
        self.Sigma_m = Sigma_m if Sigma_m is not None else 100.0 * np.eye(self.K)
        self.alpha_kappa = alpha_kappa
        self.beta_kappa = beta_kappa
        self.alpha_nu = alpha_nu
        self.beta_nu = beta_nu
        self.nu_Psi = nu_Psi if nu_Psi is not None else self.K + 2
        self.Omega_Psi = Omega_Psi if Omega_Psi is not None else np.eye(self.K)
        self.alpha_aM = alpha_aM
        self.beta_aM = beta_aM
        self.alpha_bM = alpha_bM
        self.beta_bM = beta_bM
        
        # Almacenamiento de muestras posteriores
        self.trace = {
            'z': [],           # Asignaciones
            'lambda': [],      # Coeficientes de media
            'xi': [],          # Coeficientes de log-varianza
            'w': [],           # Pesos stick-breaking
            'v': [],           # Variables stick-breaking
            'mu_lambda': [],   # Media de λ
            'Sigma_lambda': [], # Covarianza de λ
            'mu_xi': [],       # Media de ξ
            'Sigma_xi': [],    # Covarianza de ξ
            'M': [],           # Concentración DP
            'a_M': [],         # Shape de M
            'b_M': [],         # Rate de M
            'm_lambda': [],    # Media base de μ_λ
            'kappa_lambda': [], # Escala de μ_λ
            'nu_lambda': [],   # Grados de libertad de Σ_λ
            'Psi_lambda': [],  # Escala de Σ_λ
            'm_xi': [],        # Media base de μ_ξ
            'kappa_xi': [],    # Escala de μ_ξ
            'nu_xi': [],       # Grados de libertad de Σ_ξ
            'Psi_xi': [],      # Escala de Σ_ξ
            'n_clusters': []   # Número de clusters activos
        }
        
        if self.verbose:
            print(f"Inicializando DDPLinearSpline2")
            print(f"  n={self.n}, p={self.p}, H={self.H}")
            print(f"  Grado spline={self.degree}, knots internos={self.n_knots}")
            print(f"  Dimensión expandida K={self.K}")
        
        # Inicializar todos los parámetros
        self.initialize()
    
    def _build_spline_bases(self):
        """
        Construye las bases B-spline para cada covariable.
        
        Retorna:
        --------
        dict con:
            'knots': lista de arrays de knots por covariable
            'K_j': lista con número de bases por covariable
            'K': dimensión total (1 + sum(K_j))
            'design_matrix': matriz d(X) de dimensión (n, K)
        """
        knots_list = []
        K_j_list = []
        
        for j in range(self.p):
            x_j = self.X_normalized[:, j]
            x_min, x_max = x_j.min(), x_j.max()
            
            # Knots internos en cuantiles (FIJOS)
            internal_knots = np.quantile(
                x_j, 
                np.linspace(0, 1, self.n_knots + 2)[1:-1]
            )
            
            # Knots completos: repetir bordes (degree+1) veces
            knots = np.concatenate([
                [x_min] * (self.degree + 1),
                internal_knots,
                [x_max] * (self.degree + 1)
            ])
            
            knots_list.append(knots)
            # Número de bases B-spline
            K_j = len(knots) - self.degree - 1
            K_j_list.append(K_j)
        
        # Dimensión total: intercepto + sum(K_j)
        K_total = 1 + sum(K_j_list)
        
        # Construir matriz de diseño d(X)
        design_matrix = self._compute_design_matrix(
            self.X_normalized, 
            knots_list, 
            K_j_list
        )
        
        return {
            'knots': knots_list,
            'K_j': K_j_list,
            'K': K_total,
            'design_matrix': design_matrix
        }
    
    def _compute_design_matrix(self, X, knots_list, K_j_list):
        """
        Computa la matriz de diseño expandida d(X).
        
        d(X) = [1, b₁(x₁)', ..., b_p(x_p)']'
        
        Parámetros:
        -----------
        X : array (n, p)
            Matriz de covariables
        knots_list : lista de arrays
            Knots por covariable
        K_j_list : lista de ints
            Número de bases por covariable
        
        Retorna:
        --------
        d : array (n, K)
            Matriz de diseño expandida
        """
        n = X.shape[0]
        K_total = 1 + sum(K_j_list)
        
        d = np.zeros((n, K_total))
        d[:, 0] = 1.0  # Intercepto
        
        col_idx = 1
        for j in range(self.p):
            knots = knots_list[j]
            K_j = K_j_list[j]
            
            # Evaluar cada base B-spline
            for k in range(K_j):
                # Coeficientes: vector canónico
                c = np.zeros(K_j)
                c[k] = 1.0
                
                # Crear objeto BSpline de scipy
                bspline = BSpline(knots, c, self.degree, extrapolate=False)
                
                # Evaluar en X[:, j]
                d[:, col_idx] = bspline(X[:, j])
                
                # Manejar NaN (extrapolación fuera de rango)
                d[:, col_idx] = np.nan_to_num(d[:, col_idx], nan=0.0)
                
                col_idx += 1
        
        return d
    
    def initialize(self):
        """
        Inicializa todos los parámetros del modelo muestreando de los priors.
        
        Orden jerárquico:
        1. Hiperparámetros Nivel 1 (aleatorios)
        2. Hiperparámetros Nivel 0 (μ, Σ)
        3. Parámetros de concentración (M)
        4. Pesos stick-breaking (v, w)
        5. Átomos (λ_h, ξ_h)
        6. Asignaciones (z_i)
        """
        if self.verbose:
            print("Inicializando parámetros...")
        
        # ===== Hiperparámetros Nivel 1 para M =====
        self.a_M = np.random.gamma(self.alpha_aM, 1.0 / self.beta_aM)
        self.a_M = np.clip(self.a_M, 0.1, 10.0)
        
        self.b_M = np.random.gamma(self.alpha_bM, 1.0 / self.beta_bM)
        self.b_M = np.clip(self.b_M, 0.1, 10.0)
        
        # ===== Concentración del DP =====
        self.M = np.random.gamma(self.a_M, 1.0 / self.b_M)
        self.M = np.clip(self.M, 0.1, 10.0)
        
        # ===== Hiperparámetros Nivel 1 para λ (coeficientes de media) =====
        self.m_lambda = np.random.multivariate_normal(self.mu_m, self.Sigma_m)
        
        self.kappa_lambda = np.random.gamma(self.alpha_kappa, 1.0 / self.beta_kappa)
        self.kappa_lambda = np.clip(self.kappa_lambda, 0.1, 10.0)
        
        self.nu_lambda = np.random.gamma(self.alpha_nu, 1.0 / self.beta_nu)
        self.nu_lambda = np.clip(self.nu_lambda, self.K + 1, 50.0)
        
        self.Psi_lambda = wishart.rvs(df=self.nu_Psi, scale=self.Omega_Psi)
        
        # ===== Hiperparámetros Nivel 1 para ξ (coeficientes de log-varianza) =====
        self.m_xi = np.random.multivariate_normal(self.mu_m, self.Sigma_m)
        
        self.kappa_xi = np.random.gamma(self.alpha_kappa, 1.0 / self.beta_kappa)
        self.kappa_xi = np.clip(self.kappa_xi, 0.1, 10.0)
        
        self.nu_xi = np.random.gamma(self.alpha_nu, 1.0 / self.beta_nu)
        self.nu_xi = np.clip(self.nu_xi, self.K + 1, 50.0)
        
        self.Psi_xi = wishart.rvs(df=self.nu_Psi, scale=self.Omega_Psi)
        
        # ===== Hiperparámetros Nivel 0 (μ_λ, Σ_λ) =====
        # Σ_λ ~ Inv-Wishart(ν_λ, Ψ_λ)
        self.Sigma_lambda = invwishart.rvs(df=self.nu_lambda, scale=self.Psi_lambda)
        
        # μ_λ | Σ_λ ~ N(m_λ, Σ_λ/κ_λ)
        self.mu_lambda = np.random.multivariate_normal(
            self.m_lambda, 
            self.Sigma_lambda / self.kappa_lambda
        )
        
        # ===== Hiperparámetros Nivel 0 (μ_ξ, Σ_ξ) =====
        # Σ_ξ ~ Inv-Wishart(ν_ξ, Ψ_ξ)
        self.Sigma_xi = invwishart.rvs(df=self.nu_xi, scale=self.Psi_xi)
        
        # μ_ξ | Σ_ξ ~ N(m_ξ, Σ_ξ/κ_ξ)
        self.mu_xi = np.random.multivariate_normal(
            self.m_xi,
            self.Sigma_xi / self.kappa_xi
        )
        
        # ===== Pesos stick-breaking =====
        # v_h ~ Beta(1, M)
        self.v = np.random.beta(1, self.M, size=self.H)
        self.w = self._compute_weights()
        
        # ===== Coeficientes por cluster (átomos) =====
        self.lambda_h = np.zeros((self.H, self.K))
        self.xi_h = np.zeros((self.H, self.K))
        
        for h in range(self.H):
            # λ_h ~ N(μ_λ, Σ_λ)
            self.lambda_h[h, :] = np.random.multivariate_normal(
                self.mu_lambda, self.Sigma_lambda
            )
            # ξ_h ~ N(μ_ξ, Σ_ξ)
            self.xi_h[h, :] = np.random.multivariate_normal(
                self.mu_xi, self.Sigma_xi
            )
        
        # ===== Asignaciones iniciales =====
        self.z = np.zeros(self.n, dtype=int)
        d = self.spline_bases['design_matrix']
        
        for i in range(self.n):
            likes = np.zeros(self.H)
            
            for h in range(self.H):
                # μ_h(x_i) = λ_h' d(x_i)
                mu_h = np.dot(self.lambda_h[h, :], d[i, :])
                
                # log(σ²_h(x_i)) = ξ_h' d(x_i)
                log_sigma2_h = np.dot(self.xi_h[h, :], d[i, :])
                sigma2_h = np.exp(log_sigma2_h)
                sigma2_h = np.clip(sigma2_h, 1e-6, 1e6)
                
                # Likelihood: y_i ~ N(μ_h(x_i), σ²_h(x_i))
                likes[h] = norm.pdf(
                    self.y_normalized[i], 
                    mu_h, 
                    np.sqrt(sigma2_h)
                )
            
            likes = np.clip(likes, 1e-300, None)
            probs = self.w * likes
            probs /= probs.sum()
            
            self.z[i] = np.random.choice(self.H, p=probs)
        
        if self.verbose:
            active = len(np.unique(self.z))
            print(f"Inicialización completa. Clusters activos: {active}")
    
    def _compute_weights(self):
        """
        Calcula pesos stick-breaking: w_h = v_h * ∏_{ℓ<h} (1 - v_ℓ).
        
        Retorna:
        --------
        w : array (H,)
            Pesos normalizados
        """
        w = np.zeros(self.H)
        remaining = 1.0
        
        for h in range(self.H):
            w[h] = self.v[h] * remaining
            remaining *= (1 - self.v[h])
        
        # Normalizar por seguridad numérica
        w /= w.sum()
        return w
    
    def sample_slice_variables(self):
        """
        Slice sampling para truncamiento adaptativo.
        
        Muestrea u_i ~ Uniform(0, w_{z_i}) para cada observación.
        Si min(w_h) > min(u_i), expandir número de clusters.
        
        Retorna:
        --------
        u : array (n,)
            Variables slice
        """
        u = np.zeros(self.n)
        for i in range(self.n):
            u[i] = np.random.uniform(0, self.w[self.z[i]])
        
        u_min = u.min()
        
        # Expandir si es necesario (truncamiento adaptativo)
        while self.H < 100:
            if self.w.min() < u_min:
                break
            
            # Agregar 5 nuevos clusters
            H_new = self.H + 5
            
            # Nuevas variables stick-breaking
            v_new = np.random.beta(1, self.M, size=5)
            self.v = np.append(self.v, v_new)
            
            # Nuevos coeficientes
            lambda_new = np.zeros((5, self.K))
            xi_new = np.zeros((5, self.K))
            
            for h in range(5):
                lambda_new[h, :] = np.random.multivariate_normal(
                    self.mu_lambda, self.Sigma_lambda
                )
                xi_new[h, :] = np.random.multivariate_normal(
                    self.mu_xi, self.Sigma_xi
                )
            
            self.lambda_h = np.vstack([self.lambda_h, lambda_new])
            self.xi_h = np.vstack([self.xi_h, xi_new])
            
            self.H = H_new
            self.w = self._compute_weights()
            break
        
        return u
    
    def update_assignments(self, u):
        """
        Actualiza asignaciones z_i dado slice variables u.
        
        Solo considera clusters h donde w_h > u_i (slice constraint).
        
        Parámetros:
        -----------
        u : array (n,)
            Variables slice
        """
        d = self.spline_bases['design_matrix']
        
        for i in range(self.n):
            # Candidatos: clusters con peso > u_i
            candidates = np.where(self.w > u[i])[0]
            
            if len(candidates) == 0:
                candidates = np.array([0])
            
            # Calcular likelihood para candidatos
            likes = np.zeros(len(candidates))
            for idx, h in enumerate(candidates):
                mu_h = np.dot(self.lambda_h[h, :], d[i, :])
                log_sigma2_h = np.dot(self.xi_h[h, :], d[i, :])
                sigma2_h = np.exp(log_sigma2_h)
                sigma2_h = np.clip(sigma2_h, 1e-6, 1e6)
                
                likes[idx] = norm.pdf(
                    self.y_normalized[i], 
                    mu_h, 
                    np.sqrt(sigma2_h)
                )
            
            likes = np.clip(likes, 1e-300, None)
            probs = likes / likes.sum()
            
            self.z[i] = np.random.choice(candidates, p=probs)
    
    def update_v(self):
        """
        Actualiza variables stick-breaking v_h.
        
        Posterior conjugado: v_h ~ Beta(1 + n_h, M + n_{>h})
        donde n_h = #{i: z_i = h} y n_{>h} = #{i: z_i > h}
        """
        for h in range(self.H - 1):
            n_h = np.sum(self.z == h)
            n_greater = np.sum(self.z > h)
            
            alpha_post = 1 + n_h
            beta_post = self.M + n_greater
            
            self.v[h] = np.random.beta(alpha_post, beta_post)
        
        # Recalcular pesos
        self.w = self._compute_weights()
    
    def update_lambda(self):
        """
        Actualiza coeficientes λ_h (CONJUGADO: Normal).
        
        Posterior: λ_h ~ N(μ_post, Σ_post)
        
        Nota: No es conjugado puro porque σ²_h depende de ξ_h,
        pero dado ξ_h es un problema de regresión lineal ponderada.
        """
        d = self.spline_bases['design_matrix']
        
        Sigma_inv = np.linalg.inv(self.Sigma_lambda)
        mu_prior_term = Sigma_inv @ self.mu_lambda
        
        for h in range(self.H):
            members_idx = np.where(self.z == h)[0]
            
            if len(members_idx) > 0:
                # Datos del cluster
                y_h = self.y_normalized[members_idx]
                d_h = d[members_idx, :]
                
                # Precisiones (dependen de ξ_h)
                prec_h = np.zeros(len(members_idx))
                for idx, i in enumerate(members_idx):
                    log_sigma2_h = np.dot(self.xi_h[h, :], d[i, :])
                    sigma2_h = np.exp(log_sigma2_h)
                    sigma2_h = np.clip(sigma2_h, 1e-6, 1e6)
                    prec_h[idx] = 1.0 / sigma2_h
                
                # Posterior (weighted least squares)
                W = np.diag(prec_h)
                Sigma_post_inv = Sigma_inv + d_h.T @ W @ d_h
                Sigma_post = np.linalg.inv(Sigma_post_inv)
                
                mu_post = Sigma_post @ (mu_prior_term + d_h.T @ W @ y_h)
                
                self.lambda_h[h, :] = np.random.multivariate_normal(
                    mu_post, Sigma_post
                )
            else:
                # Prior (cluster vacío)
                self.lambda_h[h, :] = np.random.multivariate_normal(
                    self.mu_lambda, self.Sigma_lambda
                )
    
    def update_xi(self):
        """
        Actualiza coeficientes ξ_h (NO CONJUGADO: Metropolis-Hastings).
        
        Usa propuesta Normal centrada en valor actual con covarianza escalada.
        """
        d = self.spline_bases['design_matrix']
        scale = 0.1  # Escala de propuesta (puede adaptarse)
        
        for h in range(self.H):
            members_idx = np.where(self.z == h)[0]
            
            if len(members_idx) == 0:
                # Prior (cluster vacío)
                self.xi_h[h, :] = np.random.multivariate_normal(
                    self.mu_xi, self.Sigma_xi
                )
                continue
            
            # Propuesta: random walk Normal
            xi_prop = np.random.multivariate_normal(
                self.xi_h[h, :], 
                scale**2 * np.eye(self.K)
            )
            
            # Log-likelihood actual
            log_like_curr = 0.0
            for i in members_idx:
                mu_h = np.dot(self.lambda_h[h, :], d[i, :])
                log_sigma2_curr = np.dot(self.xi_h[h, :], d[i, :])
                sigma2_curr = np.exp(log_sigma2_curr)
                sigma2_curr = np.clip(sigma2_curr, 1e-6, 1e6)
                
                log_like_curr += norm.logpdf(
                    self.y_normalized[i], mu_h, np.sqrt(sigma2_curr)
                )
            
            # Log-likelihood propuesta
            log_like_prop = 0.0
            for i in members_idx:
                mu_h = np.dot(self.lambda_h[h, :], d[i, :])
                log_sigma2_prop = np.dot(xi_prop, d[i, :])
                sigma2_prop = np.exp(log_sigma2_prop)
                sigma2_prop = np.clip(sigma2_prop, 1e-6, 1e6)
                
                log_like_prop += norm.logpdf(
                    self.y_normalized[i], mu_h, np.sqrt(sigma2_prop)
                )
            
            # Log-prior actual
            log_prior_curr = norm.logpdf(
                self.xi_h[h, :], self.mu_xi, 
                np.sqrt(np.diag(self.Sigma_xi))
            ).sum()
            
            # Log-prior propuesta
            log_prior_prop = norm.logpdf(
                xi_prop, self.mu_xi,
                np.sqrt(np.diag(self.Sigma_xi))
            ).sum()
            
            # Ratio de aceptación (log-scale)
            log_alpha = (log_like_prop + log_prior_prop) - \
                       (log_like_curr + log_prior_curr)
            
            # Aceptar/rechazar
            if np.log(np.random.uniform()) < log_alpha:
                self.xi_h[h, :] = xi_prop
    
    def update_mu_lambda(self):
        """
        Actualiza μ_λ (CONJUGADO: Normal).
        
        Posterior: μ_λ ~ N(μ_post, Σ_post)
        """
        # Precisión del prior
        Sigma_prior_inv = np.linalg.inv(self.Sigma_lambda / self.kappa_lambda)
        
        # Precisión de la likelihood (suma sobre todos los λ_h)
        Sigma_like_inv = self.H * np.linalg.inv(self.Sigma_lambda)
        
        # Posterior
        Sigma_post_inv = Sigma_prior_inv + Sigma_like_inv
        Sigma_post = np.linalg.inv(Sigma_post_inv)
        
        # Media posterior
        mu_post = Sigma_post @ (
            Sigma_prior_inv @ self.m_lambda + 
            Sigma_like_inv @ self.lambda_h.sum(axis=0)
        )
        
        self.mu_lambda = np.random.multivariate_normal(mu_post, Sigma_post)
    
    def update_Sigma_lambda(self):
        """
        Actualiza Σ_λ (CONJUGADO: Inverse-Wishart).
        
        Posterior: Σ_λ ~ Inv-Wishart(ν_post, Ψ_post)
        """
        # Grados de libertad posterior
        nu_post = self.nu_lambda + self.H
        
        # Matriz de escala posterior
        Psi_post = self.Psi_lambda.copy()
        
        # Suma de productos externos
        for h in range(self.H):
            diff = self.lambda_h[h, :] - self.mu_lambda
            Psi_post += np.outer(diff, diff)
        
        # Muestrear
        self.Sigma_lambda = invwishart.rvs(df=nu_post, scale=Psi_post)
    
    def update_mu_xi(self):
        """
        Actualiza μ_ξ (CONJUGADO: Normal).
        
        Posterior: μ_ξ ~ N(μ_post, Σ_post)
        """
        # Precisión del prior
        Sigma_prior_inv = np.linalg.inv(self.Sigma_xi / self.kappa_xi)
        
        # Precisión de la likelihood
        Sigma_like_inv = self.H * np.linalg.inv(self.Sigma_xi)
        
        # Posterior
        Sigma_post_inv = Sigma_prior_inv + Sigma_like_inv
        Sigma_post = np.linalg.inv(Sigma_post_inv)
        
        # Media posterior
        mu_post = Sigma_post @ (
            Sigma_prior_inv @ self.m_xi + 
            Sigma_like_inv @ self.xi_h.sum(axis=0)
        )
        
        self.mu_xi = np.random.multivariate_normal(mu_post, Sigma_post)
    
    def update_Sigma_xi(self):
        """
        Actualiza Σ_ξ (CONJUGADO: Inverse-Wishart).
        
        Posterior: Σ_ξ ~ Inv-Wishart(ν_post, Ψ_post)
        """
        # Grados de libertad posterior
        nu_post = self.nu_xi + self.H
        
        # Matriz de escala posterior
        Psi_post = self.Psi_xi.copy()
        
        # Suma de productos externos
        for h in range(self.H):
            diff = self.xi_h[h, :] - self.mu_xi
            Psi_post += np.outer(diff, diff)
        
        # Muestrear
        self.Sigma_xi = invwishart.rvs(df=nu_post, scale=Psi_post)
    
    def update_M(self):
        """
        Actualiza M (CONJUGADO: Gamma).
        
        Posterior: M ~ Gamma(a_post, b_post)
        """
        # Número de clusters activos
        K_active = len(np.unique(self.z))
        
        # Parámetros posterior
        a_post = self.a_M + K_active
        b_post = self.b_M - np.sum(np.log(1 - self.v[:-1]))
        
        self.M = np.random.gamma(a_post, 1.0 / b_post)
        self.M = np.clip(self.M, 0.1, 10.0)
    
    def update_m_lambda(self):
        """
        Actualiza m_λ (CONJUGADO: Normal).
        
        Posterior: m_λ ~ N(μ_post, Σ_post)
        """
        # Precisión del prior
        Sigma_prior_inv = np.linalg.inv(self.Sigma_m)
        
        # Precisión de la likelihood
        Sigma_like_inv = self.kappa_lambda * np.linalg.inv(self.Sigma_lambda)
        
        # Posterior
        Sigma_post_inv = Sigma_prior_inv + Sigma_like_inv
        Sigma_post = np.linalg.inv(Sigma_post_inv)
        
        # Media posterior
        mu_post = Sigma_post @ (
            Sigma_prior_inv @ self.mu_m + 
            Sigma_like_inv @ self.mu_lambda
        )
        
        self.m_lambda = np.random.multivariate_normal(mu_post, Sigma_post)
    
    def update_kappa_lambda(self):
        """
        Actualiza κ_λ (CONJUGADO: Gamma).
        
        Posterior: κ_λ ~ Gamma(α_post, β_post)
        """
        # Parámetros posterior
        alpha_post = self.alpha_kappa + self.K / 2.0
        
        diff = self.mu_lambda - self.m_lambda
        Sigma_inv = np.linalg.inv(self.Sigma_lambda)
        
        beta_post = self.beta_kappa + 0.5 * diff @ Sigma_inv @ diff
        
        self.kappa_lambda = np.random.gamma(alpha_post, 1.0 / beta_post)
        self.kappa_lambda = np.clip(self.kappa_lambda, 0.1, 10.0)
    
    def update_nu_lambda(self):
        """
        Actualiza ν_λ (NO CONJUGADO: Metropolis-Hastings).
        
        Propuesta: random walk en log-scale.
        """
        # Propuesta en log-scale
        log_nu_curr = np.log(self.nu_lambda)
        log_nu_prop = np.random.normal(log_nu_curr, 0.1)
        nu_prop = np.exp(log_nu_prop)
        
        # Restricción: ν > K
        if nu_prop <= self.K:
            return
        
        # Log-likelihood (Inv-Wishart para Σ_λ)
        log_like_curr = invwishart.logpdf(
            self.Sigma_lambda, df=self.nu_lambda, scale=self.Psi_lambda
        )
        log_like_prop = invwishart.logpdf(
            self.Sigma_lambda, df=nu_prop, scale=self.Psi_lambda
        )
        
        # Log-prior (Gamma)
        log_prior_curr = gamma.logpdf(
            self.nu_lambda, a=self.alpha_nu, scale=1.0/self.beta_nu
        )
        log_prior_prop = gamma.logpdf(
            nu_prop, a=self.alpha_nu, scale=1.0/self.beta_nu
        )
        
        # Jacobiano de la transformación log
        log_jacobian = log_nu_prop - log_nu_curr
        
        # Ratio de aceptación
        log_alpha = (log_like_prop + log_prior_prop) - \
                   (log_like_curr + log_prior_curr) + log_jacobian
        
        if np.log(np.random.uniform()) < log_alpha:
            self.nu_lambda = np.clip(nu_prop, self.K + 1, 50.0)
    
    def update_Psi_lambda(self):
        """
        Actualiza Ψ_λ (CONJUGADO: Wishart).
        
        Posterior: Ψ_λ ~ Wishart(ν_post, Ω_post)
        """
        # Grados de libertad posterior
        nu_post = self.nu_Psi + self.nu_lambda
        Sigma_inv = np.linalg.inv(self.Sigma_lambda)
        Psi_post = np.linalg.inv(np.linalg.inv(self.Omega_Psi) + Sigma_inv)
        
        self.Psi_lambda = wishart.rvs(df=nu_post, scale=Psi_post)
    
    def update_m_xi(self):
        """
        Actualiza m_ξ (CONJUGADO: Normal).
        
        Posterior: m_ξ ~ N(μ_post, Σ_post)
        """
        # Precisión del prior
        Sigma_prior_inv = np.linalg.inv(self.Sigma_m)
        
        # Precisión de la likelihood
        Sigma_like_inv = self.kappa_xi * np.linalg.inv(self.Sigma_xi)
        
        # Posterior
        Sigma_post_inv = Sigma_prior_inv + Sigma_like_inv
        Sigma_post = np.linalg.inv(Sigma_post_inv)
        
        # Media posterior
        mu_post = Sigma_post @ (
            Sigma_prior_inv @ self.mu_m + 
            Sigma_like_inv @ self.mu_xi
        )
        
        self.m_xi = np.random.multivariate_normal(mu_post, Sigma_post)
    
    def update_kappa_xi(self):
        """
        Actualiza κ_ξ (CONJUGADO: Gamma).
        
        Posterior: κ_ξ ~ Gamma(α_post, β_post)
        """
        # Parámetros posterior
        alpha_post = self.alpha_kappa + self.K / 2.0
        
        diff = self.mu_xi - self.m_xi
        Sigma_inv = np.linalg.inv(self.Sigma_xi)
        
        beta_post = self.beta_kappa + 0.5 * diff @ Sigma_inv @ diff
        
        self.kappa_xi = np.random.gamma(alpha_post, 1.0 / beta_post)
        self.kappa_xi = np.clip(self.kappa_xi, 0.1, 10.0)
    
    def update_nu_xi(self):
        """
        Actualiza ν_ξ (NO CONJUGADO: Metropolis-Hastings).
        
        Propuesta: random walk en log-scale.
        """
        # Propuesta en log-scale
        log_nu_curr = np.log(self.nu_xi)
        log_nu_prop = np.random.normal(log_nu_curr, 0.1)
        nu_prop = np.exp(log_nu_prop)
        
        # Restricción: ν > K
        if nu_prop <= self.K:
            return
        
        # Log-likelihood (Inv-Wishart para Σ_ξ)
        log_like_curr = invwishart.logpdf(
            self.Sigma_xi, df=self.nu_xi, scale=self.Psi_xi
        )
        log_like_prop = invwishart.logpdf(
            self.Sigma_xi, df=nu_prop, scale=self.Psi_xi
        )
        
        # Log-prior (Gamma)
        log_prior_curr = gamma.logpdf(
            self.nu_xi, a=self.alpha_nu, scale=1.0/self.beta_nu
        )
        log_prior_prop = gamma.logpdf(
            nu_prop, a=self.alpha_nu, scale=1.0/self.beta_nu
        )
        
        # Jacobiano
        log_jacobian = log_nu_prop - log_nu_curr
        
        # Ratio de aceptación
        log_alpha = (log_like_prop + log_prior_prop) - \
                   (log_like_curr + log_prior_curr) + log_jacobian
        
        if np.log(np.random.uniform()) < log_alpha:
            self.nu_xi = np.clip(nu_prop, self.K + 1, 50.0)
    
    def update_Psi_xi(self):
        """
        Actualiza Ψ_ξ (CONJUGADO: Wishart).
        
        Posterior: Ψ_ξ ~ Wishart(ν_post, Ω_post)
        """
        # Grados de libertad posterior
        nu_post = self.nu_Psi + self.nu_xi
        Sigma_inv = np.linalg.inv(self.Sigma_xi)
        Psi_post = np.linalg.inv(np.linalg.inv(self.Omega_Psi) + Sigma_inv)
        
        self.Psi_xi = wishart.rvs(df=nu_post, scale=Psi_post)
    
    def update_a_M(self):
        """
        Actualiza a_M (CONJUGADO: Gamma).
        
        Posterior: a_M ~ Gamma(α_post, β_post)
        """
        # Número de clusters activos
        K_active = len(np.unique(self.z))
        
        # Parámetros posterior
        alpha_post = self.alpha_aM + self.a_M
        beta_post = self.beta_aM + self.M
        
        self.a_M = np.random.gamma(alpha_post, 1.0 / beta_post)
        self.a_M = np.clip(self.a_M, 0.1, 10.0)
    
    def update_b_M(self):
        """
        Actualiza b_M (CONJUGADO: Gamma).
        
        Posterior: b_M ~ Gamma(α_post, β_post)
        """
        alpha_post = self.alpha_bM + self.a_M
        beta_post = self.beta_bM + self.M
        
        self.b_M = np.random.gamma(alpha_post, 1.0 / beta_post)
        self.b_M = np.clip(self.b_M, 0.1, 10.0)

    def fit(self, n_iter=1000, burn_in=500, thin=1):
        """
        Ejecuta el muestreador de Gibbs.
        
        Parámetros:
        -----------
        n_iter : int
            Número total de iteraciones MCMC
        burn_in : int
            Número de iteraciones de calentamiento (descartadas)
        thin : int
            Factor de adelgazamiento (guardar cada thin iteraciones)
        """
        if self.verbose:
            print(f"\nIniciando MCMC: {n_iter} iteraciones")
            print(f"  Burn-in: {burn_in}, Thin: {thin}")
        
        for iteration in range(n_iter):
            # ===== Actualizar asignaciones y estructura =====
            u = self.sample_slice_variables()
            self.update_assignments(u)
            self.update_v()
            
            # ===== Actualizar coeficientes (átomos) =====
            self.update_lambda()
            self.update_xi()
            
            # ===== Actualizar hiperparámetros Nivel 0 (λ) =====
            self.update_mu_lambda()
            self.update_Sigma_lambda()
            
            # ===== Actualizar hiperparámetros Nivel 0 (ξ) =====
            self.update_mu_xi()
            self.update_Sigma_xi()
            
            # ===== Actualizar concentración =====
            self.update_M()
            
            # ===== Actualizar hiperparámetros Nivel 1 (λ) =====
            self.update_m_lambda()
            self.update_kappa_lambda()
            self.update_nu_lambda()
            self.update_Psi_lambda()
            
            # ===== Actualizar hiperparámetros Nivel 1 (ξ) =====
            self.update_m_xi()
            self.update_kappa_xi()
            self.update_nu_xi()
            self.update_Psi_xi()
            
            # ===== Actualizar hiperparámetros Nivel 1 (M) =====
            self.update_a_M()
            self.update_b_M()
            
            # ===== Guardar muestras (después de burn-in) =====
            if iteration >= burn_in and (iteration - burn_in) % thin == 0:
                self.trace['z'].append(self.z.copy())
                self.trace['lambda'].append(self.lambda_h.copy())
                self.trace['xi'].append(self.xi_h.copy())
                self.trace['w'].append(self.w.copy())
                self.trace['v'].append(self.v.copy())
                self.trace['mu_lambda'].append(self.mu_lambda.copy())
                self.trace['Sigma_lambda'].append(self.Sigma_lambda.copy())
                self.trace['mu_xi'].append(self.mu_xi.copy())
                self.trace['Sigma_xi'].append(self.Sigma_xi.copy())
                self.trace['M'].append(self.M)
                self.trace['a_M'].append(self.a_M)
                self.trace['b_M'].append(self.b_M)
                self.trace['m_lambda'].append(self.m_lambda.copy())
                self.trace['kappa_lambda'].append(self.kappa_lambda)
                self.trace['nu_lambda'].append(self.nu_lambda)
                self.trace['Psi_lambda'].append(self.Psi_lambda.copy())
                self.trace['m_xi'].append(self.m_xi.copy())
                self.trace['kappa_xi'].append(self.kappa_xi)
                self.trace['nu_xi'].append(self.nu_xi)
                self.trace['Psi_xi'].append(self.Psi_xi.copy())
                self.trace['n_clusters'].append(len(np.unique(self.z)))
            
            # ===== Progreso =====
            if self.verbose and (iteration + 1) % 100 == 0:
                n_active = len(np.unique(self.z))
                print(f"  Iter {iteration + 1}/{n_iter} - "
                      f"Clusters activos: {n_active}, M: {self.M:.3f}")
        
        if self.verbose:
            print(f"\nMCMC completado. Muestras guardadas: {len(self.trace['z'])}")
    
    def predict(self, X_new, return_std=True, n_samples=None):
        """
        Predice para nuevas observaciones X_new.
        
        Parámetros:
        -----------
        X_new : array (n_new, p)
            Nuevas covariables
        return_std : bool
            Si True, retorna también la desviación estándar predictiva
        n_samples : int or None
            Número de muestras posteriores a usar (None = todas)
        
        Retorna:
        --------
        y_pred : array (n_new,)
            Media predictiva (desnormalizada)
        y_std : array (n_new,) (opcional)
            Desviación estándar predictiva (desnormalizada)
        """
        if len(self.trace['z']) == 0:
            raise ValueError("Modelo no entrenado. Ejecutar fit() primero.")
        
        # Normalizar X_new
        X_new_norm = (X_new - self.X_mean) / self.X_std
        
        # Calcular matriz de diseño para X_new
        d_new = self._compute_design_matrix(
            X_new_norm,
            self.spline_bases['knots'],
            self.spline_bases['K_j']
        )
        
        n_new = X_new.shape[0]
        n_post = len(self.trace['z']) if n_samples is None else n_samples
        n_post = min(n_post, len(self.trace['z']))
        
        # Almacenar predicciones por muestra
        y_samples = np.zeros((n_post, n_new))
        
        for s in range(n_post):
            lambda_s = self.trace['lambda'][s]
            xi_s = self.trace['xi'][s]
            w_s = self.trace['w'][s]
            H_s = lambda_s.shape[0]
            
            for i in range(n_new):
                # Calcular media y varianza por componente
                mu_components = np.zeros(H_s)
                sigma2_components = np.zeros(H_s)
                
                for h in range(H_s):
                    mu_components[h] = np.dot(lambda_s[h, :], d_new[i, :])
                    log_sigma2_h = np.dot(xi_s[h, :], d_new[i, :])
                    sigma2_components[h] = np.exp(log_sigma2_h)
                    sigma2_components[h] = np.clip(sigma2_components[h], 1e-6, 1e6)
                
                # Media predictiva: E[y|x] = Σ w_h μ_h(x)
                y_samples[s, i] = np.dot(w_s[:H_s], mu_components)
        
        # Media sobre muestras posteriores
        y_pred_norm = y_samples.mean(axis=0)
        
        # Desnormalizar
        y_pred = y_pred_norm * self.y_std + self.y_mean
        
        if return_std:
            # Desviación estándar predictiva
            y_std_norm = y_samples.std(axis=0)
            y_std = y_std_norm * self.y_std
            return y_pred, y_std
        else:
            return y_pred
    
    def predict_components(self, X_new, n_samples=None):
        """
        Predice media y desviación por componente para X_new.
        
        Parámetros:
        -----------
        X_new : array (n_new, p)
            Nuevas covariables
        n_samples : int or None
            Número de muestras posteriores a usar
        
        Retorna:
        --------
        dict con:
            'mu': array (n_samples, n_new, H) - medias por componente
            'sigma': array (n_samples, n_new, H) - desviaciones por componente
            'weights': array (n_samples, H) - pesos de componentes
        """
        if len(self.trace['z']) == 0:
            raise ValueError("Modelo no entrenado. Ejecutar fit() primero.")
        
        # Normalizar X_new
        X_new_norm = (X_new - self.X_mean) / self.X_std
        
        # Calcular matriz de diseño para X_new
        d_new = self._compute_design_matrix(
            X_new_norm,
            self.spline_bases['knots'],
            self.spline_bases['K_j']
        )
        
        n_new = X_new.shape[0]
        n_post = len(self.trace['z']) if n_samples is None else n_samples
        n_post = min(n_post, len(self.trace['z']))
        H_max = max(t.shape[0] for t in self.trace['lambda'])
        
        # Almacenar predicciones
        mu_components = np.zeros((n_post, n_new, H_max))
        sigma_components = np.zeros((n_post, n_new, H_max))
        weights = np.zeros((n_post, H_max))
        
        for s in range(n_post):
            lambda_s = self.trace['lambda'][s]
            xi_s = self.trace['xi'][s]
            w_s = self.trace['w'][s]
            H_s = lambda_s.shape[0]
            
            weights[s, :H_s] = w_s[:H_s]
            
            for i in range(n_new):
                for h in range(H_s):
                    # Media: μ_h(x) = λ_h' d(x)
                    mu_h = np.dot(lambda_s[h, :], d_new[i, :])
                    mu_components[s, i, h] = mu_h * self.y_std + self.y_mean
                    
                    # Desviación: σ_h(x) = exp(ξ_h' d(x) / 2)
                    log_sigma2_h = np.dot(xi_s[h, :], d_new[i, :])
                    sigma2_h = np.exp(log_sigma2_h)
                    sigma2_h = np.clip(sigma2_h, 1e-6, 1e6)
                    sigma_components[s, i, h] = np.sqrt(sigma2_h) * self.y_std
        
        return {
            'mu': mu_components,
            'sigma': sigma_components,
            'weights': weights
        }
    
    def get_cluster_summary(self):
        """
        Resume información sobre los clusters encontrados.
        
        Retorna:
        --------
        dict con:
            'n_clusters_mean': media de clusters activos
            'n_clusters_std': desviación de clusters activos
            'cluster_sizes': tamaños promedio por cluster
            'concentration_mean': media de M
        """
        if len(self.trace['z']) == 0:
            raise ValueError("Modelo no entrenado. Ejecutar fit() primero.")
        
        n_clusters_trace = np.array(self.trace['n_clusters'])
        M_trace = np.array(self.trace['M'])
        
        # Tamaños de cluster promedio
        z_last = self.trace['z'][-1]
        unique, counts = np.unique(z_last, return_counts=True)
        cluster_sizes = dict(zip(unique, counts))
        
        return {
            'n_clusters_mean': n_clusters_trace.mean(),
            'n_clusters_std': n_clusters_trace.std(),
            'n_clusters_mode': np.bincount(n_clusters_trace).argmax(),
            'cluster_sizes': cluster_sizes,
            'concentration_mean': M_trace.mean(),
            'concentration_std': M_trace.std()
        }


# ===== Ejemplo de uso =====
if __name__ == "__main__":
    # Generar datos sintéticos
    np.random.seed(42)
    n = 300
    
    # Dos covariables
    X = np.column_stack([
        np.random.uniform(-3, 3, n),
        np.random.uniform(-3, 3, n)
    ])
    
    # Función verdadera: mezcla de dos componentes
    z_true = (X[:, 0] + X[:, 1] > 0).astype(int)
    
    # Componente 0: lineal simple
    mu0 = 2*X[:, 0] - X[:, 1]
    sigma0 = 0.5
    
    # Componente 1: no lineal
    mu1 = np.sin(X[:, 0]) + 0.5*X[:, 1]**2
    sigma1 = 1.0 + 0.3*np.abs(X[:, 0])
    
    # Generar respuestas
    y = np.where(
        z_true == 0,
        np.random.normal(mu0, sigma0),
        np.random.normal(mu1, sigma1)
    )
    
    # Ajustar modelo
    print("="*60)
    print("Ejemplo: DDPLinearSpline2")
    print("="*60)
    
    model = DDPLinearSpline2(
        y=y, 
        X=X, 
        H=10,
        degree=3,
        n_knots=5,
        verbose=True
    )
    
    model.fit(n_iter=1000, burn_in=500, thin=2)
    
