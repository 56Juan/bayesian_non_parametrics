"""
sampler.py
----------
Gibbs sampler completo para el modelo PSBP-FAR(c) funcional.
Implementa los 8 pasos derivados de Chung & Dunson (2009).

Pasos del sampler
-----------------
1. Actualizar Z_{il}^*  — aumentación de datos (Albert & Chib)
2. Actualizar S_i       — asignaciones de componente
3. Actualizar (B_l, Sigma_l) — átomos MNIW conjugado
4. Actualizar alpha_l   — interceptos de pesos
5. Actualizar mu        — hiperparámetro de alpha_l
6. Actualizar Gamma_lj  — localizaciones (muestreo discreto)
7. Actualizar psi_lj    — escalas (normal truncada N^+)
8. Media predictiva in-sample
"""
from __future__ import annotations
import numpy as np
from numpy.linalg import cholesky, solve, LinAlgError
from scipy.stats import norm, invwishart

from model_psbp_fd.data_functions.preprocessors import standardize, build_lags


class GibbsSampler:
    """
    Encapsula el estado y la lógica del Gibbs sampler.

    Parameters
    ----------
    K, c, L, M    : dimensiones del modelo
    hyperparams   : dict con sobreescritura de hiperparámetros
    rng           : numpy Generator para reproducibilidad
    """

    def __init__(
        self,
        K: int,
        c: int,
        L: int,
        M: int,
        hyperparams: dict,
        rng: np.random.Generator,
    ) -> None:
        self.K   = K
        self.c   = c
        self.L   = L
        self.M   = M
        self.rng = rng
        self.hyperparams_: dict = {}   # poblado en _init_hyperparams()
        self._hp = hyperparams         # overrides del usuario

    # ------------------------------------------------------------------
    # Interfaz pública
    # ------------------------------------------------------------------

    def run(
        self,
        THETA: np.ndarray,
        nsim: int,
        burn: int,
    ) -> dict[str, np.ndarray]:
        """
        Ejecuta nsim iteraciones y devuelve cadenas post-burn.

        Returns
        -------
        chains : dict con arrays de forma (nsim-burn, ...)
        """
        self._prepare_data(THETA)
        self._init_hyperparams()
        self._init_state()
        self._alloc_storage(nsim)

        for gt in range(nsim):
            self._step1_update_Z()
            self._step2_update_S()
            self._step3_update_atoms()
            self._step4_update_alpha()
            self._step5_update_mu()
            self._step6_update_Gamma()
            self._step7_update_psi()
            self._step8_predictive()
            self._store(gt)

            if (gt + 1) % 200 == 0:
                print(f"  Iter {gt+1}/{nsim} | Componentes activos: {self.Si.max()}")

        return self._extract_chains(burn, nsim)

    # ------------------------------------------------------------------
    # Preparación
    # ------------------------------------------------------------------

    def _prepare_data(self, THETA: np.ndarray) -> None:
        THETA_s, self.theta_mean, self.theta_std = standardize(THETA)
        self.Theta_resp, self.Theta_lag, self.Theta_ext = build_lags(THETA_s, self.c)
        self.n = self.Theta_resp.shape[0]
        self.d = self.Theta_ext.shape[1]      # c*K + 1
        self.p = self.Theta_lag.shape[1]      # c*K

    def _init_hyperparams(self) -> None:
        K, c, L, M, n, d, p = self.K, self.c, self.L, self.M, self.n, self.d, self.p
        hp = {
            "mu":      1.0,
            "mumu":    0.0,
            "taumu":   1.0,
            "mu_psi":  0.0,
            "tau_psi": np.ones(p),
            "nu0":     K + 2,
            "Psi0":    np.eye(K),
            "M0":      np.zeros((K, d)),
            "Lambda0": (1.0 / n) * np.eye(d),
        }
        hp.update(self._hp)               # overrides del usuario
        self.hyperparams_ = hp

        # Precomputaciones constantes
        self.Lam0_M0t    = hp["Lambda0"] @ hp["M0"].T     # d x K
        self.M0_Lam0_M0t = hp["M0"] @ hp["Lambda0"] @ hp["M0"].T  # K x K

    def _init_state(self) -> None:
        K, L, M, n, p = self.K, self.L, self.M, self.n, self.p
        hp = self.hyperparams_

        self.Si     = self.rng.integers(1, L + 1, size=n)
        self.alphah = np.zeros(L - 1)
        self.psih   = np.zeros((L - 1, p))
        self.mu     = float(hp["mu"])

        # Grilla Gamma: M x p (una por dimensión del predictor)
        self.Gstar  = np.zeros((M, p))
        for j in range(p):
            xmin = self.Theta_lag[:, j].min()
            xmax = self.Theta_lag[:, j].max()
            self.Gstar[:, j] = xmin + (np.arange(1, M + 1) / M) * (xmax - xmin)

        self.Gammah = np.zeros((L - 1, p))
        for j in range(p):
            idx = self.rng.integers(0, M, size=L - 1)
            self.Gammah[:, j] = self.Gstar[idx, j]

        # Átomos
        self.Bl     = np.zeros((K, self.d, L))
        self.Sigmal = np.zeros((K, K, L))
        for l in range(L):
            self.Bl[:, :, l]     = hp["M0"]
            self.Sigmal[:, :, l] = np.eye(K)

    def _alloc_storage(self, nsim: int) -> None:
        K, L, n, p, d = self.K, self.L, self.n, self.p, self.d
        self._s_alphah = np.zeros((nsim, L - 1),      dtype=np.float32)
        self._s_psih   = np.zeros((nsim, L - 1, p),   dtype=np.float32)
        self._s_Gammah = np.zeros((nsim, L - 1, p),   dtype=np.float32)
        self._s_Bl     = np.zeros((nsim, K, d, L),    dtype=np.float32)
        self._s_Sigmal = np.zeros((nsim, K, K, L),    dtype=np.float32)
        self._s_Si     = np.zeros((nsim, n),           dtype=np.int16)
        self._s_mu     = np.zeros(nsim,                dtype=np.float32)
        self._s_inE    = np.zeros((nsim, n, K),        dtype=np.float32)
        self._s_N1     = np.zeros(nsim,                dtype=np.int16)

    # ------------------------------------------------------------------
    # Pasos del Gibbs
    # ------------------------------------------------------------------

    def _step1_update_Z(self) -> None:
        """Aumentación de datos: Z_{il}^* y W_{il}."""
        n, L = self.n, self.L
        self.Zil = np.zeros((n, L))
        self.Wil = np.zeros((n, L))

        for i in range(n):
            xi  = self.Theta_lag[i]
            s_i = self.Si[i]
            for l in range(min(s_i, L - 1)):
                m_il  = self.alphah[l] - np.sum(self.psih[l] * np.abs(xi - self.Gammah[l]))
                u     = self.rng.uniform()
                Phi_0 = norm.cdf(-m_il)
                if l < s_i - 1:
                    self.Zil[i, l] = m_il + norm.ppf(u * Phi_0)
                else:
                    self.Zil[i, l] = m_il + norm.ppf(u + (1 - u) * Phi_0)
                self.Wil[i, l] = self.Zil[i, l] + np.sum(self.psih[l] * np.abs(xi - self.Gammah[l]))

            if s_i == L:
                for l in range(L - 1):
                    m_il  = self.alphah[l] - np.sum(self.psih[l] * np.abs(xi - self.Gammah[l]))
                    u     = self.rng.uniform()
                    Phi_0 = norm.cdf(-m_il)
                    self.Zil[i, l] = m_il + norm.ppf(u * Phi_0)
                    self.Wil[i, l] = self.Zil[i, l] + np.sum(self.psih[l] * np.abs(xi - self.Gammah[l]))

    def _step2_update_S(self) -> None:
        """Asignaciones S_i ~ Multinomial(phi_xi * lik)."""
        n, L, K = self.n, self.L, self.K
        self.phi_xi = np.zeros((n, L))

        for i in range(n):
            xi  = self.Theta_lag[i]
            vhx = np.array([
                norm.cdf(self.alphah[h] - np.sum(self.psih[h] * np.abs(xi - self.Gammah[h])))
                for h in range(L - 1)
            ])
            phx    = np.empty(L)
            phx[0] = vhx[0]
            for h in range(1, L - 1):
                phx[h] = vhx[h] * np.prod(1 - vhx[:h])
            phx[L - 1] = np.prod(1 - vhx)
            self.phi_xi[i] = phx

            xi_ext  = self.Theta_ext[i]
            log_lik = np.array([
                _log_mvn_pdf(self.Theta_resp[i], self.Bl[:, :, l] @ xi_ext, self.Sigmal[:, :, l])
                for l in range(L)
            ])
            log_post = np.log(phx + 1e-300) + log_lik
            log_post -= log_post.max()
            post     = np.exp(log_post)
            post    /= post.sum()
            self.Si[i] = self.rng.choice(L, p=post) + 1   # 1-indexed

    def _step3_update_atoms(self) -> None:
        """Actualización MNIW conjugada para (B_l, Sigma_l)."""
        hp = self.hyperparams_
        nu0, Psi0, M0, Lambda0 = hp["nu0"], hp["Psi0"], hp["M0"], hp["Lambda0"]

        for l in range(self.L):
            idx = np.where(self.Si == l + 1)[0]
            n_l = len(idx)
            if n_l == 0:
                self.Sigmal[:, :, l] = invwishart.rvs(df=nu0, scale=Psi0)
                self.Bl[:, :, l]     = _sample_MN(M0, self.Sigmal[:, :, l], Lambda0, self.rng)
                continue

            Yl = self.Theta_resp[idx]          # n_l x K
            Xl = self.Theta_ext[idx]           # n_l x d
            SXX = Xl.T @ Xl
            SXY = Xl.T @ Yl
            SYY = Yl.T @ Yl

            Lambda_l = Lambda0 + SXX
            Lambda_l = (Lambda_l + Lambda_l.T) / 2
            M_l_T    = solve(Lambda_l, self.Lam0_M0t + SXY)   # d x K
            M_l      = M_l_T.T                                 # K x d

            nu_l  = nu0 + n_l
            Psi_l = Psi0 + SYY + self.M0_Lam0_M0t - M_l @ Lambda_l @ M_l_T
            Psi_l = (Psi_l + Psi_l.T) / 2

            self.Sigmal[:, :, l] = invwishart.rvs(df=nu_l, scale=Psi_l)
            self.Bl[:, :, l]     = _sample_MN(M_l, self.Sigmal[:, :, l], Lambda_l, self.rng)

    def _step4_update_alpha(self) -> None:
        """alpha_l | rest ~ N(m_l, v_l)."""
        for l in range(self.L - 1):
            idx_ge = np.where(self.Si >= l + 1)[0]
            k_l    = len(idx_ge)
            v_l    = 1.0 / (1 + k_l)
            m_l    = v_l * (self.mu + self.Wil[idx_ge, l].sum())
            self.alphah[l] = self.rng.normal(m_l, np.sqrt(v_l))

    def _step5_update_mu(self) -> None:
        """mu | rest ~ N(mu_mu_hat, 1/tau_mu_hat)."""
        hp          = self.hyperparams_
        tau_mu_hat  = (self.L - 1) + hp["taumu"]
        mu_mu_hat   = (hp["taumu"] * hp["mumu"] + self.alphah.sum()) / tau_mu_hat
        self.mu     = self.rng.normal(mu_mu_hat, 1.0 / np.sqrt(tau_mu_hat))

    def _step6_update_Gamma(self) -> None:
        """Gamma_{lj} | rest ~ Discreta sobre grilla."""
        for l in range(self.L - 1):
            idx_ge = np.where(self.Si >= l + 1)[0]
            if len(idx_ge) == 0:
                continue
            Zil_l  = self.Zil[idx_ge, l]
            Xlag_l = self.Theta_lag[idx_ge]

            for j in range(self.p):
                other = list(range(j)) + list(range(j + 1, self.p))
                base_eta = self.alphah[l] - (
                    self.psih[l, other] * np.abs(Xlag_l[:, other] - self.Gammah[l, other])
                ).sum(axis=1)

                log_pm = np.array([
                    norm.logpdf(Zil_l, base_eta - self.psih[l, j] * np.abs(Xlag_l[:, j] - self.Gstar[m, j])).sum()
                    for m in range(self.M)
                ])
                log_pm -= log_pm.max()
                pm     = np.exp(log_pm);  pm /= pm.sum()
                self.Gammah[l, j] = self.Gstar[self.rng.choice(self.M, p=pm), j]

    def _step7_update_psi(self) -> None:
        """psi_{lj} | rest ~ N^+(m_lj, v_lj)."""
        hp = self.hyperparams_
        for l in range(self.L - 1):
            idx_ge = np.where(self.Si >= l + 1)[0]
            if len(idx_ge) == 0:
                continue
            Zil_l  = self.Zil[idx_ge, l]
            Xlag_l = self.Theta_lag[idx_ge]

            for j in range(self.p):
                other  = list(range(j)) + list(range(j + 1, self.p))
                T_ilj  = self.alphah[l] - Zil_l - (
                    self.psih[l, other] * np.abs(Xlag_l[:, other] - self.Gammah[l, other])
                ).sum(axis=1)
                diff_j = np.abs(Xlag_l[:, j] - self.Gammah[l, j])
                v_lj   = 1.0 / (hp["tau_psi"][j] + (diff_j ** 2).sum())
                m_lj   = v_lj * (hp["tau_psi"][j] * hp["mu_psi"] + (T_ilj * diff_j).sum())
                self.psih[l, j] = _sample_truncated_normal(m_lj, v_lj, self.rng)

    def _step8_predictive(self) -> None:
        """Media predictiva in-sample (escala estandarizada)."""
        inE = np.zeros((self.n, self.K))
        for l in range(self.L):
            mu_l = (self.Bl[:, :, l] @ self.Theta_ext.T).T   # n x K
            inE += self.phi_xi[:, [l]] * mu_l
        self._inE_curr = inE

    # ------------------------------------------------------------------
    # Almacenamiento
    # ------------------------------------------------------------------

    def _store(self, gt: int) -> None:
        self._s_alphah[gt] = self.alphah
        self._s_psih[gt]   = self.psih
        self._s_Gammah[gt] = self.Gammah
        self._s_Bl[gt]     = self.Bl
        self._s_Sigmal[gt] = self.Sigmal
        self._s_Si[gt]     = self.Si
        self._s_mu[gt]     = self.mu
        self._s_inE[gt]    = self._inE_curr
        self._s_N1[gt]     = self.Si.max()

    def _extract_chains(self, burn: int, nsim: int) -> dict[str, np.ndarray]:
        idx = slice(burn, nsim)
        return {
            "alphah": self._s_alphah[idx],
            "psih":   self._s_psih[idx],
            "Gammah": self._s_Gammah[idx],
            "Bl":     self._s_Bl[idx],
            "Sigmal": self._s_Sigmal[idx],
            "Si":     self._s_Si[idx],
            "mu":     self._s_mu[idx],
            "inE":    self._s_inE[idx],
            "N1":     self._s_N1[idx],
            "theta_mean": self.theta_mean,
            "theta_std":  self.theta_std,
        }


# ══════════════════════════════════════════════════════════════════════════
# Funciones puras de muestreo (sin estado, testeables unitariamente)
# ══════════════════════════════════════════════════════════════════════════

def _log_mvn_pdf(x: np.ndarray, mu: np.ndarray, Sigma: np.ndarray) -> float:
    """Log-densidad N_K(x | mu, Sigma) via Cholesky."""
    K = x.shape[0]
    try:
        L_chol = cholesky(Sigma + 1e-10 * np.eye(K))
        z      = solve(L_chol, x - mu)
        return -0.5 * K * np.log(2 * np.pi) - np.log(np.diag(L_chol)).sum() - 0.5 * (z @ z)
    except LinAlgError:
        return -1e300


def _sample_MN(
    M_mean: np.ndarray,
    Sigma: np.ndarray,
    Lambda: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Muestrea B ~ MN(M_mean, Sigma ⊗ Lambda^{-1}).

    B = M_mean + L_Sigma @ Z @ L_Lambda_inv.T
    donde L_Lambda_inv = inv(chol(Lambda)).T
    """
    K_B, d_B = M_mean.shape
    try:
        L_Sig     = cholesky(Sigma + 1e-10 * np.eye(K_B))
        L_Lam     = cholesky(Lambda + 1e-10 * np.eye(d_B))
        L_Lam_inv = solve(L_Lam.T, np.eye(d_B))          # L_Lam^{-T}
        Z = rng.standard_normal((K_B, d_B))
        return M_mean + L_Sig @ Z @ L_Lam_inv.T
    except LinAlgError:
        return M_mean


def _sample_truncated_normal(
    m: float,
    v: float,
    rng: np.random.Generator,
) -> float:
    """Muestrea de N^+(m, v) [truncada en 0] por CDF inversa."""
    u     = rng.uniform()
    Phi_0 = norm.cdf(-m / np.sqrt(v))
    val   = m + np.sqrt(v) * norm.ppf(u + (1 - u) * Phi_0)
    return float(val) if (np.isfinite(val) and val > 0) else 1e-3
