import numpy as np
from scipy.stats import norm, invgamma, gamma, wishart, invwishart, beta
from scipy.interpolate import BSpline
import math

# Importar módulo C++ compilado
try:
    from . import ddp2_cpp
    CPP_AVAILABLE = True
    print("Módulo C++ cargado exitosamente")
except ImportError:
    CPP_AVAILABLE = False
    print("Módulo C++ no disponible, usando implementación Python pura")


class DDPLinearSpline2:
    """
    Dependent Dirichlet Process con regresión lineal en B-splines - Versión con estandarización robusta.
    
    MEJORA CLAVE: Estandarización en tres niveles para máxima estabilidad numérica:
    1. Datos crudos (X, y) → escala [0, 1] aproximadamente
    2. Bases spline d(X) → estandarización columna por columna
    3. Priors adaptativos → ajustados a la escala estandarizada
    """
    
    def __init__(self, y, X, H=10,
                 degree=3,
                 n_knots=3,
                 standardize_splines=True,
                 mu_m=None,
                 Sigma_m=None,
                 alpha_kappa=5.0,
                 beta_kappa=5.0,
                 alpha_nu=10.0,
                 beta_nu=2,
                 nu_Psi=None,
                 Omega_Psi=None,
                 alpha_aM=2.0,
                 beta_aM=2.0,
                 alpha_bM=2.0,
                 beta_bM=2.0,
                 use_cpp=True,
                 seed=42,
                 verbose=True):
        """
        Inicializa el modelo DDP con estandarización robusta.
        
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
        standardize_splines : bool
            Si True, estandariza las bases spline columna por columna (RECOMENDADO)
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
        use_cpp : bool
            Si True y el módulo C++ está disponible, usa implementación C++
        seed : int
            Semilla para reproducibilidad
        verbose : bool
            Imprimir mensajes de progreso
        """
        
        # Datos
        self.y = np.array(y)
        self.X = np.array(X)
        self.n, self.p = self.X.shape
        self.H = H
        self.verbose = verbose
        self.seed = seed
        self.standardize_splines = standardize_splines
        np.random.seed(seed)
        
        # Configurar uso de C++
        self.use_cpp = use_cpp and CPP_AVAILABLE
        if self.use_cpp:
            self.cpp_core = ddp2_cpp.LSBPCore(seed)
            if self.verbose:
                print(f"Usando aceleración C++ (seed={seed})")
        else:
            if use_cpp and not CPP_AVAILABLE:
                print("Advertencia: C++ solicitado pero no disponible")
            if self.verbose:
                print("Usando implementación Python pura")
        
        # ===== NIVEL 1: Estandarización de datos crudos =====
        self.y_mean = np.mean(y)
        self.y_std = np.std(y)
        if self.y_std < 1e-10:
            self.y_std = 1.0
        self.y_normalized = (self.y - self.y_mean) / self.y_std
        
        self.X_mean = np.mean(X, axis=0)
        self.X_std = np.std(X, axis=0)
        self.X_std[self.X_std < 1e-10] = 1.0
        self.X_normalized = (self.X - self.X_mean) / self.X_std
        
        # Parámetros de splines (FIJOS durante MCMC)
        self.degree = degree
        self.n_knots = n_knots
        
        # ===== NIVEL 2: Construir y estandarizar bases B-spline =====
        self.spline_bases = self._build_spline_bases()
        self.K = self.spline_bases['K']
        
        # Hiperparámetros Nivel 2 (ajustados a escala estandarizada)
        self.mu_m = mu_m if mu_m is not None else np.zeros(self.K)
        self.Sigma_m = Sigma_m if Sigma_m is not None else 5.0 * np.eye(self.K)
        self.alpha_kappa = alpha_kappa
        self.beta_kappa = beta_kappa
        self.alpha_nu = alpha_nu
        self.beta_nu = beta_nu
        self.nu_Psi = nu_Psi if nu_Psi is not None else self.K + 2
        self.Omega_Psi = Omega_Psi if Omega_Psi is not None else 2.0 * np.eye(self.K)
        self.alpha_aM = alpha_aM
        self.beta_aM = beta_aM
        self.alpha_bM = alpha_bM
        self.beta_bM = beta_bM
        
        # Almacenamiento de muestras posteriores
        self.trace = {
            'z': [],
            'lambda': [],
            'xi': [],
            'w': [],
            'v': [],
            'mu_lambda': [],
            'Sigma_lambda': [],
            'mu_xi': [],
            'Sigma_xi': [],
            'M': [],
            'a_M': [],
            'b_M': [],
            'm_lambda': [],
            'kappa_lambda': [],
            'nu_lambda': [],
            'Psi_lambda': [],
            'm_xi': [],
            'kappa_xi': [],
            'nu_xi': [],
            'Psi_xi': [],
            'n_clusters': []
        }
        
        if self.verbose:
            print(f"Inicializando DDPLinearSpline2 con estandarización robusta")
            print(f"  n={self.n}, p={self.p}, H={self.H}")
            print(f"  Grado spline={self.degree}, knots internos={self.n_knots}")
            print(f"  Dimensión expandida K={self.K}")
            print(f"  Estandarización de splines: {self.standardize_splines}")
        
        # Inicializar todos los parámetros
        self.initialize()
    
    def _build_spline_bases(self):
        """
        Construye y opcionalmente estandariza las bases B-spline.
        
        Retorna dict con:
            'knots': lista de arrays de knots por covariable
            'K_j': lista con número de bases por covariable
            'K': dimensión total (1 + sum(K_j))
            'design_matrix': matriz d(X) de dimensión (n, K) ESTANDARIZADA
            'spline_mean': media de cada columna spline (para predicción)
            'spline_std': std de cada columna spline (para predicción)
        """
        knots_list = []
        K_j_list = []
        
        for j in range(self.p):
            x_j = self.X_normalized[:, j]
            x_min, x_max = x_j.min(), x_j.max()
            
            internal_knots = np.quantile(
                x_j, 
                np.linspace(0, 1, self.n_knots + 2)[1:-1]
            )
            
            knots = np.concatenate([
                [x_min] * (self.degree + 1),
                internal_knots,
                [x_max] * (self.degree + 1)
            ])
            
            knots_list.append(knots)
            K_j = len(knots) - self.degree - 1
            K_j_list.append(K_j)
        
        K_total = 1 + sum(K_j_list)
        
        # Construir matriz de diseño SIN estandarizar
        design_matrix_raw = self._compute_design_matrix_raw(
            self.X_normalized, 
            knots_list, 
            K_j_list
        )
        
        # ===== ESTANDARIZACIÓN DE SPLINES (columna por columna) =====
        if self.standardize_splines:
            # NO estandarizar el intercepto (columna 0)
            spline_mean = np.zeros(K_total)
            spline_std = np.ones(K_total)
            
            # Estandarizar solo las columnas spline (1 a K-1)
            for k in range(1, K_total):
                col_mean = np.mean(design_matrix_raw[:, k])
                col_std = np.std(design_matrix_raw[:, k])
                
                if col_std < 1e-10:
                    col_std = 1.0
                
                spline_mean[k] = col_mean
                spline_std[k] = col_std
                
                design_matrix_raw[:, k] = (design_matrix_raw[:, k] - col_mean) / col_std
            
            if self.verbose:
                print(f"  Splines estandarizadas: mean={spline_mean[1:].mean():.3f}, std={spline_std[1:].mean():.3f}")
        else:
            spline_mean = np.zeros(K_total)
            spline_std = np.ones(K_total)
        
        return {
            'knots': knots_list,
            'K_j': K_j_list,
            'K': K_total,
            'design_matrix': design_matrix_raw,
            'spline_mean': spline_mean,
            'spline_std': spline_std
        }
    
    def _compute_design_matrix_raw(self, X, knots_list, K_j_list):
        """
        Computa la matriz de diseño expandida d(X) SIN estandarizar.
        """
        n = X.shape[0]
        K_total = 1 + sum(K_j_list)
        
        d = np.zeros((n, K_total))
        d[:, 0] = 1.0
        
        col_idx = 1
        for j in range(self.p):
            knots = knots_list[j]
            K_j = K_j_list[j]
            
            for k in range(K_j):
                c = np.zeros(K_j)
                c[k] = 1.0
                
                bspline = BSpline(knots, c, self.degree, extrapolate=False)
                d[:, col_idx] = bspline(X[:, j])
                d[:, col_idx] = np.nan_to_num(d[:, col_idx], nan=0.0)
                
                col_idx += 1
        
        return d
    
    def _compute_design_matrix(self, X, knots_list, K_j_list):
        """
        Computa matriz de diseño para nuevos datos X, aplicando la MISMA estandarización.
        """
        d_raw = self._compute_design_matrix_raw(X, knots_list, K_j_list)
        
        if self.standardize_splines:
            for k in range(1, self.K):
                d_raw[:, k] = (d_raw[:, k] - self.spline_bases['spline_mean'][k]) / self.spline_bases['spline_std'][k]
        
        return d_raw
    
    def initialize(self):
        """
        Inicializa todos los parámetros del modelo.
        """
        if self.verbose:
            print("Inicializando parámetros...")
        
        # Hiperparámetros Nivel 1 para M
        self.a_M = np.random.gamma(self.alpha_aM, 1.0 / self.beta_aM)
        self.a_M = np.clip(self.a_M, 0.1, 10.0)
        
        self.b_M = np.random.gamma(self.alpha_bM, 1.0 / self.beta_bM)
        self.b_M = np.clip(self.b_M, 0.1, 10.0)
        
        # Concentración del DP
        self.M = np.random.gamma(self.a_M, 1.0 / self.b_M)
        self.M = np.clip(self.M, 0.1, 10.0)
        
        # Hiperparámetros Nivel 1 para λ
        self.m_lambda = np.random.multivariate_normal(self.mu_m, self.Sigma_m)
        
        self.kappa_lambda = np.random.gamma(self.alpha_kappa, 1.0 / self.beta_kappa)
        self.kappa_lambda = np.clip(self.kappa_lambda, 0.1, 10.0)
        
        self.nu_lambda = np.random.gamma(self.alpha_nu, 1.0 / self.beta_nu)
        self.nu_lambda = np.clip(self.nu_lambda, self.K + 1, 50.0)
        
        self.Psi_lambda = wishart.rvs(df=self.nu_Psi, scale=self.Omega_Psi)
        
        # Hiperparámetros Nivel 1 para ξ
        self.m_xi = np.random.multivariate_normal(self.mu_m, self.Sigma_m)
        
        self.kappa_xi = np.random.gamma(self.alpha_kappa, 1.0 / self.beta_kappa)
        self.kappa_xi = np.clip(self.kappa_xi, 0.1, 10.0)
        
        self.nu_xi = np.random.gamma(self.alpha_nu, 1.0 / self.beta_nu)
        self.nu_xi = np.clip(self.nu_xi, self.K + 1, 50.0)
        
        self.Psi_xi = wishart.rvs(df=self.nu_Psi, scale=self.Omega_Psi)
        
        # Hiperparámetros Nivel 0 (μ_λ, Σ_λ)
        self.Sigma_lambda = invwishart.rvs(df=self.nu_lambda, scale=self.Psi_lambda)
        self.mu_lambda = np.random.multivariate_normal(
            self.m_lambda, 
            self.Sigma_lambda / self.kappa_lambda
        )
        
        # Hiperparámetros Nivel 0 (μ_ξ, Σ_ξ)
        self.Sigma_xi = invwishart.rvs(df=self.nu_xi, scale=self.Psi_xi)
        self.mu_xi = np.random.multivariate_normal(
            self.m_xi,
            self.Sigma_xi / self.kappa_xi
        )
        
        # Pesos stick-breaking
        self.v = np.random.beta(1, self.M, size=self.H)
        self.w = self._compute_weights()
        
        # Coeficientes por cluster
        self.lambda_h = np.zeros((self.H, self.K))
        self.xi_h = np.zeros((self.H, self.K))
        
        for h in range(self.H):
            self.lambda_h[h, :] = np.random.multivariate_normal(
                self.mu_lambda, self.Sigma_lambda
            )
            self.xi_h[h, :] = np.random.multivariate_normal(
                self.mu_xi, self.Sigma_xi
            )
        
        # Asignaciones iniciales
        self.z = np.zeros(self.n, dtype=int)
        d = self.spline_bases['design_matrix']
        
        for i in range(self.n):
            likes = np.zeros(self.H)
            
            for h in range(self.H):
                mu_h = np.dot(self.lambda_h[h, :], d[i, :])
                log_sigma2_h = np.dot(self.xi_h[h, :], d[i, :])
                sigma2_h = np.exp(log_sigma2_h)
                sigma2_h = np.clip(sigma2_h, 1e-6, 1e6)
                
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
        Calcula pesos stick-breaking.
        """
        w = np.zeros(self.H)
        remaining = 1.0
        
        for h in range(self.H):
            w[h] = self.v[h] * remaining
            remaining *= (1 - self.v[h])
        
        w /= w.sum()
        return w
    
    def sample_slice_variables(self):
        """
        Slice sampling para truncamiento adaptativo.
        """
        if self.use_cpp:
            u = np.array(self.cpp_core.sample_slice_variables(
                self.z.tolist(), self.w.tolist(), self.n
            ))
        else:
            u = np.zeros(self.n)
            for i in range(self.n):
                u[i] = np.random.uniform(0, self.w[self.z[i]])
        
        u_min = u.min()
        
        while self.H < 100:
            if self.w.min() < u_min:
                break
            
            H_new = self.H + 5
            v_new = np.random.beta(1, self.M, size=5)
            self.v = np.append(self.v, v_new)
            
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
        Actualiza asignaciones z_i.
        """
        if self.use_cpp:
            y_norm_2d = self.y_normalized.reshape(-1, 1)
            z_new = self.cpp_core.update_assignments(
                y_norm_2d,
                self.spline_bases['design_matrix'],
                self.lambda_h,
                self.xi_h,
                self.w.tolist(),
                u.tolist(),
                self.n,
                self.H,
                self.K
            )
            self.z = np.array(z_new, dtype=int)
        else:
            d = self.spline_bases['design_matrix']
            
            for i in range(self.n):
                candidates = np.where(self.w > u[i])[0]
                
                if len(candidates) == 0:
                    candidates = np.array([0])
                
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
        """
        for h in range(self.H - 1):
            n_h = np.sum(self.z == h)
            n_greater = np.sum(self.z > h)
            
            alpha_post = 1 + n_h
            beta_post = self.M + n_greater
            
            self.v[h] = np.random.beta(alpha_post, beta_post)
        
        self.w = self._compute_weights()
    
    def update_lambda(self):
        """
        Actualiza coeficientes λ_h.
        """
        if self.use_cpp:
            y_norm_2d = self.y_normalized.reshape(-1, 1)
            self.lambda_h = self.cpp_core.update_lambda(
                y_norm_2d,
                self.spline_bases['design_matrix'],
                self.z.tolist(),
                self.xi_h,
                self.mu_lambda,
                self.Sigma_lambda,
                self.n,
                self.H,
                self.K
            )
        else:
            d = self.spline_bases['design_matrix']
            
            Sigma_inv = np.linalg.inv(self.Sigma_lambda)
            mu_prior_term = Sigma_inv @ self.mu_lambda
            
            for h in range(self.H):
                members_idx = np.where(self.z == h)[0]
                
                if len(members_idx) > 0:
                    y_h = self.y_normalized[members_idx]
                    d_h = d[members_idx, :]
                    
                    prec_h = np.zeros(len(members_idx))
                    for idx, i in enumerate(members_idx):
                        log_sigma2_h = np.dot(self.xi_h[h, :], d[i, :])
                        sigma2_h = np.exp(log_sigma2_h)
                        sigma2_h = np.clip(sigma2_h, 1e-6, 1e6)
                        prec_h[idx] = 1.0 / sigma2_h
                    
                    W = np.diag(prec_h)
                    Sigma_post_inv = Sigma_inv + d_h.T @ W @ d_h
                    Sigma_post = np.linalg.inv(Sigma_post_inv)
                    
                    mu_post = Sigma_post @ (mu_prior_term + d_h.T @ W @ y_h)
                    
                    self.lambda_h[h, :] = np.random.multivariate_normal(
                        mu_post, Sigma_post
                    )
                else:
                    self.lambda_h[h, :] = np.random.multivariate_normal(
                        self.mu_lambda, self.Sigma_lambda
                    )
    
    def update_xi(self):
        """
        Actualiza coeficientes ξ_h.
        """
        scale = 0.1
        
        if self.use_cpp:
            y_norm_2d = self.y_normalized.reshape(-1, 1)
            self.xi_h = self.cpp_core.update_xi(
                y_norm_2d,
                self.spline_bases['design_matrix'],
                self.lambda_h,
                self.xi_h,
                self.z.tolist(),
                self.mu_xi,
                self.Sigma_xi,
                scale,
                self.n,
                self.H,
                self.K
            )
        else:
            d = self.spline_bases['design_matrix']
            
            for h in range(self.H):
                members_idx = np.where(self.z == h)[0]
                
                if len(members_idx) == 0:
                    self.xi_h[h, :] = np.random.multivariate_normal(
                        self.mu_xi, self.Sigma_xi
                    )
                    continue
                
                xi_prop = np.random.multivariate_normal(
                    self.xi_h[h, :], 
                    scale**2 * np.eye(self.K)
                )
                
                log_like_curr = 0.0
                for i in members_idx:
                    mu_h = np.dot(self.lambda_h[h, :], d[i, :])
                    log_sigma2_curr = np.dot(self.xi_h[h, :], d[i, :])
                    sigma2_curr = np.exp(log_sigma2_curr)
                    sigma2_curr = np.clip(sigma2_curr, 1e-6, 1e6)
                    
                    log_like_curr += norm.logpdf(
                        self.y_normalized[i], mu_h, np.sqrt(sigma2_curr)
                    )
                
                log_like_prop = 0.0
                for i in members_idx:
                    mu_h = np.dot(self.lambda_h[h, :], d[i, :])
                    log_sigma2_prop = np.dot(xi_prop, d[i, :])
                    sigma2_prop = np.exp(log_sigma2_prop)
                    sigma2_prop = np.clip(sigma2_prop, 1e-6, 1e6)
                    
                    log_like_prop += norm.logpdf(
                        self.y_normalized[i], mu_h, np.sqrt(sigma2_prop)
                    )
                
                log_prior_curr = norm.logpdf(
                    self.xi_h[h, :], self.mu_xi, 
                    np.sqrt(np.diag(self.Sigma_xi))
                ).sum()
                
                log_prior_prop = norm.logpdf(
                    xi_prop, self.mu_xi,
                    np.sqrt(np.diag(self.Sigma_xi))
                ).sum()
                
                log_alpha = (log_like_prop + log_prior_prop) - \
                           (log_like_curr + log_prior_curr)
                
                if np.log(np.random.uniform()) < log_alpha:
                    self.xi_h[h, :] = xi_prop
    
    def update_mu_lambda(self):
        """
        Actualiza μ_λ con regularización robusta.
        """
        try:
            Sigma_prior_inv = np.linalg.inv(self.Sigma_lambda / self.kappa_lambda)
        except np.linalg.LinAlgError:
            Sigma_reg = self.Sigma_lambda + 1e-6 * np.eye(self.K)
            Sigma_prior_inv = np.linalg.inv(Sigma_reg / self.kappa_lambda)
        
        try:
            Sigma_like_inv = self.H * np.linalg.inv(self.Sigma_lambda)
        except np.linalg.LinAlgError:
            Sigma_reg = self.Sigma_lambda + 1e-6 * np.eye(self.K)
            Sigma_like_inv = self.H * np.linalg.inv(Sigma_reg)
        
        Sigma_post_inv = Sigma_prior_inv + Sigma_like_inv
        Sigma_post_inv = (Sigma_post_inv + Sigma_post_inv.T) / 2.0
        
        try:
            Sigma_post = np.linalg.inv(Sigma_post_inv)
        except np.linalg.LinAlgError:
            epsilon = 1e-6 * np.trace(Sigma_post_inv) / self.K
            Sigma_post_inv += epsilon * np.eye(self.K)
            Sigma_post = np.linalg.inv(Sigma_post_inv)
        
        Sigma_post = (Sigma_post + Sigma_post.T) / 2.0
        
        mu_post = Sigma_post @ (
            Sigma_prior_inv @ self.m_lambda + 
            Sigma_like_inv @ self.lambda_h.sum(axis=0)
        )
        
        if np.any(np.isinf(mu_post)) or np.any(np.isnan(mu_post)):
            self.mu_lambda = self.m_lambda.copy()
            return
        
        try:
            self.mu_lambda = np.random.multivariate_normal(mu_post, Sigma_post)
        except Exception as e:
            self.mu_lambda = mu_post
    
    def update_Sigma_lambda(self):
        """
        Actualiza Σ_λ con control robusto de escala.
        """
        nu_post = self.nu_lambda + self.H
        
        if nu_post <= self.K:
            nu_post = self.K + 2
        
        Psi_post = self.Psi_lambda.copy()
        
        for h in range(self.H):
            diff = self.lambda_h[h, :] - self.mu_lambda
            outer = np.outer(diff, diff)
            
            if np.any(np.isinf(outer)) or np.any(np.isnan(outer)):
                continue
            
            Psi_post += outer
        
        if np.any(np.isinf(Psi_post)) or np.any(np.isnan(Psi_post)):
            self.Sigma_lambda = self.Psi_lambda / (self.nu_lambda - self.K - 1)
            return
        
        trace_val = np.trace(Psi_post)
        if trace_val > 1e10 or trace_val < 1e-10:
            scale_factor = np.sqrt(self.K / trace_val)
            Psi_post *= scale_factor
        
        Psi_post = (Psi_post + Psi_post.T) / 2.0
        
        try:
            np.linalg.cholesky(Psi_post)
        except np.linalg.LinAlgError:
            epsilon = 1e-4 * np.mean(np.diag(Psi_post))
            Psi_post += epsilon * np.eye(self.K)
        
        try:
            self.Sigma_lambda = invwishart.rvs(df=nu_post, scale=Psi_post)
            
            if np.any(np.isinf(self.Sigma_lambda)) or np.any(np.isnan(self.Sigma_lambda)):
                raise ValueError("Resultado contiene inf/nan")
            
            trace_result = np.trace(self.Sigma_lambda)
            if trace_result > 1e6:
                self.Sigma_lambda *= (1e6 / trace_result)
                
        except (np.linalg.LinAlgError, ValueError) as e:
            self.Sigma_lambda = self.Psi_lambda / (self.nu_lambda - self.K - 1)
    
    def update_mu_xi(self):
        """
        Actualiza μ_ξ con regularización robusta.
        """
        try:
            Sigma_prior_inv = np.linalg.inv(self.Sigma_xi / self.kappa_xi)
        except np.linalg.LinAlgError:
            Sigma_reg = self.Sigma_xi + 1e-6 * np.eye(self.K)
            Sigma_prior_inv = np.linalg.inv(Sigma_reg / self.kappa_xi)
        
        try:
            Sigma_like_inv = self.H * np.linalg.inv(self.Sigma_xi)
        except np.linalg.LinAlgError:
            Sigma_reg = self.Sigma_xi + 1e-6 * np.eye(self.K)
            Sigma_like_inv = self.H * np.linalg.inv(Sigma_reg)
        
        Sigma_post_inv = Sigma_prior_inv + Sigma_like_inv
        Sigma_post_inv = (Sigma_post_inv + Sigma_post_inv.T) / 2.0
        
        try:
            Sigma_post = np.linalg.inv(Sigma_post_inv)
        except np.linalg.LinAlgError:
            epsilon = 1e-6 * np.trace(Sigma_post_inv) / self.K
            Sigma_post_inv += epsilon * np.eye(self.K)
            Sigma_post = np.linalg.inv(Sigma_post_inv)
        
        Sigma_post = (Sigma_post + Sigma_post.T) / 2.0
        
        mu_post = Sigma_post @ (
            Sigma_prior_inv @ self.m_xi + 
            Sigma_like_inv @ self.xi_h.sum(axis=0)
        )
        
        if np.any(np.isinf(mu_post)) or np.any(np.isnan(mu_post)):
            self.mu_xi = self.m_xi.copy()
            return
        
        try:
            self.mu_xi = np.random.multivariate_normal(mu_post, Sigma_post)
        except Exception as e:
            self.mu_xi = mu_post
    
    def update_Sigma_xi(self):
        """
        Actualiza Σ_ξ con control robusto de escala.
        """
        nu_post = self.nu_xi + self.H
        
        if nu_post <= self.K:
            nu_post = self.K + 2
        
        Psi_post = self.Psi_xi.copy()
        
        for h in range(self.H):
            diff = self.xi_h[h, :] - self.mu_xi
            outer = np.outer(diff, diff)
            
            if np.any(np.isinf(outer)) or np.any(np.isnan(outer)):
                continue
            
            Psi_post += outer
        
        if np.any(np.isinf(Psi_post)) or np.any(np.isnan(Psi_post)):
            self.Sigma_xi = self.Psi_xi / (self.nu_xi - self.K - 1)
            return
        
        trace_val = np.trace(Psi_post)
        if trace_val > 1e10 or trace_val < 1e-10:
            scale_factor = np.sqrt(self.K / trace_val)
            Psi_post *= scale_factor
        
        Psi_post = (Psi_post + Psi_post.T) / 2.0
        
        try:
            np.linalg.cholesky(Psi_post)
        except np.linalg.LinAlgError:
            epsilon = 1e-4 * np.mean(np.diag(Psi_post))
            Psi_post += epsilon * np.eye(self.K)
        
        try:
            self.Sigma_xi = invwishart.rvs(df=nu_post, scale=Psi_post)
            
            if np.any(np.isinf(self.Sigma_xi)) or np.any(np.isnan(self.Sigma_xi)):
                raise ValueError("Resultado contiene inf/nan")
            
            trace_result = np.trace(self.Sigma_xi)
            if trace_result > 1e6:
                self.Sigma_xi *= (1e6 / trace_result)
                
        except (np.linalg.LinAlgError, ValueError) as e:
            self.Sigma_xi = self.Psi_xi / (self.nu_xi - self.K - 1)
    
    def update_M(self):
        """
        Actualiza M con control de v cercano a 1.
        """
        K_active = len(np.unique(self.z))
        
        a_post = self.a_M + K_active
        
        log_terms = []
        for v_val in self.v[:-1]:
            if v_val >= 0.9999:
                log_terms.append(-10.0)
            else:
                log_terms.append(np.log(1 - v_val))
        
        b_post = self.b_M - np.sum(log_terms)
        b_post = np.clip(b_post, 0.1, 100.0)
        
        self.M = np.random.gamma(a_post, 1.0 / b_post)
        self.M = np.clip(self.M, 0.1, 10.0)
    
    def update_m_lambda(self):
        """
        Actualiza m_λ.
        """
        Sigma_prior_inv = np.linalg.inv(self.Sigma_m)
        Sigma_like_inv = self.kappa_lambda * np.linalg.inv(self.Sigma_lambda)
        
        Sigma_post_inv = Sigma_prior_inv + Sigma_like_inv
        Sigma_post = np.linalg.inv(Sigma_post_inv)
        
        mu_post = Sigma_post @ (
            Sigma_prior_inv @ self.mu_m + 
            Sigma_like_inv @ self.mu_lambda
        )
        
        self.m_lambda = np.random.multivariate_normal(mu_post, Sigma_post)
    
    def update_kappa_lambda(self):
        """
        Actualiza κ_λ.
        """
        alpha_post = self.alpha_kappa + self.K / 2.0
        
        diff = self.mu_lambda - self.m_lambda
        Sigma_inv = np.linalg.inv(self.Sigma_lambda)
        
        beta_post = self.beta_kappa + 0.5 * diff @ Sigma_inv @ diff
        
        self.kappa_lambda = np.random.gamma(alpha_post, 1.0 / beta_post)
        self.kappa_lambda = np.clip(self.kappa_lambda, 0.1, 10.0)
    
    def update_nu_lambda(self):
        """
        Actualiza ν_λ.
        """
        log_nu_curr = np.log(self.nu_lambda)
        log_nu_prop = np.random.normal(log_nu_curr, 0.1)
        nu_prop = np.exp(log_nu_prop)
        
        if nu_prop <= self.K:
            return
        
        log_like_curr = invwishart.logpdf(
            self.Sigma_lambda, df=self.nu_lambda, scale=self.Psi_lambda
        )
        log_like_prop = invwishart.logpdf(
            self.Sigma_lambda, df=nu_prop, scale=self.Psi_lambda
        )
        
        log_prior_curr = gamma.logpdf(
            self.nu_lambda, a=self.alpha_nu, scale=1.0/self.beta_nu
        )
        log_prior_prop = gamma.logpdf(
            nu_prop, a=self.alpha_nu, scale=1.0/self.beta_nu
        )
        
        log_jacobian = log_nu_prop - log_nu_curr
        
        log_alpha = (log_like_prop + log_prior_prop) - \
                   (log_like_curr + log_prior_curr) + log_jacobian
        
        if np.log(np.random.uniform()) < log_alpha:
            self.nu_lambda = np.clip(nu_prop, self.K + 1, 50.0)
    
    def update_Psi_lambda(self):
        """
        Actualiza Ψ_λ con control de estabilidad.
        """
        nu_post = self.nu_Psi + self.nu_lambda
        
        if nu_post <= self.K:
            nu_post = self.K + 2
        
        try:
            Sigma_inv = np.linalg.inv(self.Sigma_lambda)
            Omega_inv = np.linalg.inv(self.Omega_Psi)
        except np.linalg.LinAlgError:
            return
        
        Psi_post_inv = Omega_inv + Sigma_inv
        Psi_post_inv = (Psi_post_inv + Psi_post_inv.T) / 2.0
        
        try:
            Psi_post = np.linalg.inv(Psi_post_inv)
        except np.linalg.LinAlgError:
            epsilon = 1e-6 * np.trace(Psi_post_inv) / self.K
            Psi_post_inv += epsilon * np.eye(self.K)
            Psi_post = np.linalg.inv(Psi_post_inv)
        
        Psi_post = (Psi_post + Psi_post.T) / 2.0
        
        trace_val = np.trace(Psi_post)
        if trace_val > 1e6 or trace_val < 1e-6:
            return
        
        try:
            self.Psi_lambda = wishart.rvs(df=nu_post, scale=Psi_post)
        except Exception:
            pass
    
    def update_m_xi(self):
        """
        Actualiza m_ξ.
        """
        Sigma_prior_inv = np.linalg.inv(self.Sigma_m)
        Sigma_like_inv = self.kappa_xi * np.linalg.inv(self.Sigma_xi)
        
        Sigma_post_inv = Sigma_prior_inv + Sigma_like_inv
        Sigma_post = np.linalg.inv(Sigma_post_inv)
        
        mu_post = Sigma_post @ (
            Sigma_prior_inv @ self.mu_m + 
            Sigma_like_inv @ self.mu_xi
        )
        
        self.m_xi = np.random.multivariate_normal(mu_post, Sigma_post)
    
    def update_kappa_xi(self):
        """
        Actualiza κ_ξ.
        """
        alpha_post = self.alpha_kappa + self.K / 2.0
        
        diff = self.mu_xi - self.m_xi
        Sigma_inv = np.linalg.inv(self.Sigma_xi)
        
        beta_post = self.beta_kappa + 0.5 * diff @ Sigma_inv @ diff
        
        self.kappa_xi = np.random.gamma(alpha_post, 1.0 / beta_post)
        self.kappa_xi = np.clip(self.kappa_xi, 0.1, 10.0)
    
    def update_nu_xi(self):
        """
        Actualiza ν_ξ.
        """
        log_nu_curr = np.log(self.nu_xi)
        log_nu_prop = np.random.normal(log_nu_curr, 0.1)
        nu_prop = np.exp(log_nu_prop)
        
        if nu_prop <= self.K:
            return
        
        log_like_curr = invwishart.logpdf(
            self.Sigma_xi, df=self.nu_xi, scale=self.Psi_xi
        )
        log_like_prop = invwishart.logpdf(
            self.Sigma_xi, df=nu_prop, scale=self.Psi_xi
        )
        
        log_prior_curr = gamma.logpdf(
            self.nu_xi, a=self.alpha_nu, scale=1.0/self.beta_nu
        )
        log_prior_prop = gamma.logpdf(
            nu_prop, a=self.alpha_nu, scale=1.0/self.beta_nu
        )
        
        log_jacobian = log_nu_prop - log_nu_curr
        
        log_alpha = (log_like_prop + log_prior_prop) - \
                   (log_like_curr + log_prior_curr) + log_jacobian
        
        if np.log(np.random.uniform()) < log_alpha:
            self.nu_xi = np.clip(nu_prop, self.K + 1, 50.0)
    
    def update_Psi_xi(self):
        """
        Actualiza Ψ_ξ con control de estabilidad.
        """
        nu_post = self.nu_Psi + self.nu_xi
        
        if nu_post <= self.K:
            nu_post = self.K + 2
        
        try:
            Sigma_inv = np.linalg.inv(self.Sigma_xi)
            Omega_inv = np.linalg.inv(self.Omega_Psi)
        except np.linalg.LinAlgError:
            return
        
        Psi_post_inv = Omega_inv + Sigma_inv
        Psi_post_inv = (Psi_post_inv + Psi_post_inv.T) / 2.0
        
        try:
            Psi_post = np.linalg.inv(Psi_post_inv)
        except np.linalg.LinAlgError:
            epsilon = 1e-6 * np.trace(Psi_post_inv) / self.K
            Psi_post_inv += epsilon * np.eye(self.K)
            Psi_post = np.linalg.inv(Psi_post_inv)
        
        Psi_post = (Psi_post + Psi_post.T) / 2.0
        
        trace_val = np.trace(Psi_post)
        if trace_val > 1e6 or trace_val < 1e-6:
            return
        
        try:
            self.Psi_xi = wishart.rvs(df=nu_post, scale=Psi_post)
        except Exception:
            pass
    
    def update_a_M(self):
        """
        Actualiza a_M.
        """
        K_active = len(np.unique(self.z))
        
        alpha_post = self.alpha_aM + self.a_M
        beta_post = self.beta_aM + self.M
        
        self.a_M = np.random.gamma(alpha_post, 1.0 / beta_post)
        self.a_M = np.clip(self.a_M, 0.1, 10.0)
    
    def update_b_M(self):
        """
        Actualiza b_M.
        """
        alpha_post = self.alpha_bM + self.a_M
        beta_post = self.beta_bM + self.M
        
        self.b_M = np.random.gamma(alpha_post, 1.0 / beta_post)
        self.b_M = np.clip(self.b_M, 0.1, 10.0)

    def fit(self, n_iter=1000, burn_in=500, thin=1):
        """
        Ejecuta el muestreador de Gibbs.
        """
        if self.verbose:
            print(f"\nIniciando MCMC: {n_iter} iteraciones")
            print(f"  Burn-in: {burn_in}, Thin: {thin}")
        
        for iteration in range(n_iter):
            u = self.sample_slice_variables()
            self.update_assignments(u)
            self.update_v()
            
            self.update_lambda()
            self.update_xi()
            
            self.update_mu_lambda()
            self.update_Sigma_lambda()
            
            self.update_mu_xi()
            self.update_Sigma_xi()
            
            self.update_M()
            
            self.update_m_lambda()
            self.update_kappa_lambda()
            self.update_nu_lambda()
            self.update_Psi_lambda()
            
            self.update_m_xi()
            self.update_kappa_xi()
            self.update_nu_xi()
            self.update_Psi_xi()
            
            self.update_a_M()
            self.update_b_M()
            
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
            
            if self.verbose and (iteration + 1) % 100 == 0:
                n_active = len(np.unique(self.z))
                print(f"  Iter {iteration + 1}/{n_iter} - "
                      f"Clusters activos: {n_active}, M: {self.M:.3f}")
        
        if self.verbose:
            print(f"\nMCMC completado. Muestras guardadas: {len(self.trace['z'])}")
    
    def predict_mean(self, X_new, return_std=True, n_samples=None):
        """
        Predice para nuevas observaciones X_new.
        
        IMPORTANTE: Aplica la MISMA transformación de estandarización que en entrenamiento.
        
        Parámetros:
        -----------
        X_new : array (n_new, p)
            Nuevas covariables (en escala ORIGINAL)
        return_std : bool
            Si True, retorna también la desviación estándar predictiva
        n_samples : int or None
            Número de muestras posteriores a usar (None = todas)
        
        Retorna:
        --------
        y_pred : array (n_new,)
            Media predictiva (en escala ORIGINAL de y)
        y_std : array (n_new,) (opcional)
            Desviación estándar predictiva (en escala ORIGINAL de y)
        """
        if len(self.trace['z']) == 0:
            raise ValueError("Modelo no entrenado. Ejecutar fit() primero.")
        
        # PASO 1: Normalizar X_new usando parámetros de ENTRENAMIENTO
        X_new_norm = (X_new - self.X_mean) / self.X_std
        
        # PASO 2: Calcular matriz de diseño aplicando MISMA estandarización spline
        d_new = self._compute_design_matrix(
            X_new_norm,
            self.spline_bases['knots'],
            self.spline_bases['K_j']
        )
        
        n_new = X_new.shape[0]
        n_post = len(self.trace['z']) if n_samples is None else n_samples
        n_post = min(n_post, len(self.trace['z']))
        
        y_samples = np.zeros((n_post, n_new))
        
        for s in range(n_post):
            lambda_s = self.trace['lambda'][s]
            xi_s = self.trace['xi'][s]
            w_s = self.trace['w'][s]
            H_s = lambda_s.shape[0]
            
            for i in range(n_new):
                mu_components = np.zeros(H_s)
                sigma2_components = np.zeros(H_s)
                
                for h in range(H_s):
                    mu_components[h] = np.dot(lambda_s[h, :], d_new[i, :])
                    log_sigma2_h = np.dot(xi_s[h, :], d_new[i, :])
                    sigma2_components[h] = np.exp(log_sigma2_h)
                    sigma2_components[h] = np.clip(sigma2_components[h], 1e-6, 1e6)
                
                y_samples[s, i] = np.dot(w_s[:H_s], mu_components)
        
        y_pred_norm = y_samples.mean(axis=0)
        
        # PASO 3: Desnormalizar a escala ORIGINAL
        y_pred = y_pred_norm * self.y_std + self.y_mean
        
        if return_std:
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
            Nuevas covariables (en escala ORIGINAL)
        n_samples : int or None
            Número de muestras posteriores a usar
        
        Retorna:
        --------
        dict con:
            'mu': array (n_samples, n_new, H) - medias por componente (escala ORIGINAL)
            'sigma': array (n_samples, n_new, H) - desviaciones por componente (escala ORIGINAL)
            'weights': array (n_samples, H) - pesos de componentes
        """
        if len(self.trace['z']) == 0:
            raise ValueError("Modelo no entrenado. Ejecutar fit() primero.")
        
        # Normalizar X_new
        X_new_norm = (X_new - self.X_mean) / self.X_std
        
        # Calcular matriz de diseño con estandarización
        d_new = self._compute_design_matrix(
            X_new_norm,
            self.spline_bases['knots'],
            self.spline_bases['K_j']
        )
        
        n_new = X_new.shape[0]
        n_post = len(self.trace['z']) if n_samples is None else n_samples
        n_post = min(n_post, len(self.trace['z']))
        H_max = max(t.shape[0] for t in self.trace['lambda'])
        
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
                    # Media en escala normalizada, luego desnormalizar
                    mu_h_norm = np.dot(lambda_s[h, :], d_new[i, :])
                    mu_components[s, i, h] = mu_h_norm * self.y_std + self.y_mean
                    
                    # Desviación
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
        """
        if len(self.trace['z']) == 0:
            raise ValueError("Modelo no entrenado. Ejecutar fit() primero.")
        
        n_clusters_trace = np.array(self.trace['n_clusters'])
        M_trace = np.array(self.trace['M'])
        
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