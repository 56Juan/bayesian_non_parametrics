"""
PSBP Sampler v2 — Probit Stick-Breaking Process Mixture
========================================================

Implementación del Gibbs sampler de Chung & Dunson (2009),
"Nonparametric Bayes Conditional Distribution Modeling with Variable
Selection", JASA 104(488).

Cambios respecto a v1 (todos para alinear con el MATLAB de referencia):
  - A2.1 : `apij`, `bpij`, `mupsij`, `taupsij` aceptan ahora arrays de
           longitud p (priors heterogéneas por variable). Si se pasa un
           escalar, se expande como antes.
  - A2.2 : el peso en el update de Si se computa en log-space, idéntico
           a la línea 131 del MATLAB (`exp(log(phx) + log(normpdf))`).
  - A4   : cuando `kh = 0` y `gammajh[h,j] == 1`, se muestrea Gamma_jh
           uniforme de Gstar (replicando MATLAB).
  - A4-b : cuando `kh = 0` y `gammajh[h,j] == 1`, se muestrea psi_jh
           de la prior truncada N(mupsij_j, 1/taupsij_j) I(psi >= 0).

NOTA: la nota D3 (ddof inconsistente) NO se modifica aquí porque
afecta a `psbp_fd_v1.py` (estandarización), no al sampler. El sampler
es agnóstico a la escala.

Author: model_psbp_fd
"""

from __future__ import annotations
from typing import Optional, Dict, Any, Union
import numpy as np
from scipy.stats import norm
from scipy.special import gammaln


# ─────────────────────────────────────────────────────────────────────────────
# Utilidades de muestreo
# ─────────────────────────────────────────────────────────────────────────────

def _truncnorm_lower0_sample(m: float, v: float, rng: np.random.Generator) -> float:
    """
    Muestra de N(m, v) truncada en (-inf, 0].

    Equivalente a la operación de MATLAB:
        m + sqrt(v) * norminv(u * normcdf((0-m)/sqrt(v)))
    donde u ~ U(0, 1).
    """
    sv = np.sqrt(v)
    u = rng.uniform(0.0, 1.0)
    cdf0 = norm.cdf((0.0 - m) / sv)
    return m + sv * norm.ppf(u * cdf0)


def _truncnorm_upper0_sample(m: float, v: float, rng: np.random.Generator) -> float:
    """
    Muestra de N(m, v) truncada en [0, +inf).

    Equivalente a la operación de MATLAB:
        m + sqrt(v) * norminv(u + (1-u) * normcdf((0-m)/sqrt(v)))
    donde u ~ U(0, 1). Si el resultado es inf/nan (cola extrema),
    se devuelve 0.01 como salvaguarda numérica (consistente con MATLAB
    línea 271-273).
    """
    sv = np.sqrt(v)
    u = rng.uniform(0.0, 1.0)
    cdf0 = norm.cdf((0.0 - m) / sv)
    val = m + sv * norm.ppf(u + (1.0 - u) * cdf0)
    return val if np.isfinite(val) else 0.01


def _mvnrnd_safe(mu: np.ndarray, cov: np.ndarray,
                 rng: np.random.Generator) -> np.ndarray:
    """
    Normal multivariada con jitter adaptativo para garantizar
    covarianza simétrica positiva-definida.
    """
    cov = 0.5 * (cov + cov.T)
    eig_min = np.linalg.eigvalsh(cov).min()
    if eig_min < 1e-10:
        cov = cov + np.eye(cov.shape[0]) * (1e-10 - eig_min)
    return rng.multivariate_normal(mu.ravel(), cov)


def _expand_to_p(value: Union[float, np.ndarray, list], p: int,
                 name: str) -> np.ndarray:
    """
    Expande un valor escalar a un array de longitud p, o valida que
    un array tenga longitud p.

    Habilita el uso de priors heterogéneas por variable (A2.1).
    """
    if np.isscalar(value):
        return np.full(p, float(value))
    arr = np.asarray(value, dtype=np.float64).ravel()
    if arr.shape[0] != p:
        raise ValueError(
            f"Hiperparámetro '{name}' tiene longitud {arr.shape[0]}; "
            f"se esperaba escalar o array de longitud p={p}."
        )
    return arr


# ─────────────────────────────────────────────────────────────────────────────
# Clase principal
# ─────────────────────────────────────────────────────────────────────────────

class PSBPSampler:
    """
    Gibbs sampler para el modelo Probit Stick-Breaking Process Mixture
    (v2 — alineado con MATLAB de Chung & Dunson, 2009).

    Parámetros
    ----------
    mcmc_cfg : dict
        Configuración MCMC con claves:
            - 'nsim' : int — número total de iteraciones
            - 'burn' : int — iteraciones de burn-in (informativo)
            - 'N'    : int — truncamiento stick-breaking
            - 'M'    : int — tamaño de la grilla para Gamma_jh

    hp : dict
        Hiperparámetros del modelo. Las siguientes claves admiten **escalar
        o array de longitud p** (priors heterogéneas por variable):
            - 'apij', 'bpij'   : Beta para pi_j
            - 'mupsij', 'taupsij' : Normal-truncada-positiva para psi_jh

        Las siguientes claves siempre son **escalares**:
            - 'atau', 'btau'   : Gamma para tau_h
            - 'ag',   'bg'     : Gamma para g
            - 'mumu', 'taumu'  : Normal para mu
            - 'pwj'            : Bernoulli para w_j

    seed : int, opcional
        Semilla para reproducibilidad.

    verbose_every : int, default=200
        Imprime progreso cada `verbose_every` iteraciones. 0/None = silencio.

    Atributos post-fit
    ------------------
    traces : dict
        Diccionario con todas las trazas MCMC (ver código).

    n_features_, n_samples_ : int
    """

    def __init__(self,
                 mcmc_cfg: Dict[str, int],
                 hp: Dict[str, Any],
                 seed: Optional[int] = None,
                 verbose_every: int = 200):
        self.mcmc_cfg = dict(mcmc_cfg)
        self.hp = dict(hp)
        self.seed = seed
        self.verbose_every = int(verbose_every) if verbose_every else 0

        self.traces: Optional[Dict[str, np.ndarray]] = None
        self.n_features_: Optional[int] = None
        self.n_samples_: Optional[int] = None

    # ─────────────────────────────────────────────────────────────────────
    # Validación
    # ─────────────────────────────────────────────────────────────────────
    def _validate_inputs(self, X: np.ndarray, y: np.ndarray) -> None:
        if X.ndim != 2:
            raise ValueError(f"X debe ser 2D; recibido shape {X.shape}")
        if y.ndim != 1:
            raise ValueError(f"y debe ser 1D; recibido shape {y.shape}")
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X.shape[0]={X.shape[0]} != y.shape[0]={y.shape[0]}"
            )
        if X.shape[1] < 2:
            raise ValueError(
                "X debe incluir columna de intercepto + al menos 1 covariable"
            )
        if not np.allclose(X[:, 0], 1.0):
            raise ValueError(
                "Se espera que X[:, 0] sea la columna de unos (intercepto)"
            )

        required_mcmc = {"nsim", "burn", "N", "M"}
        missing = required_mcmc - set(self.mcmc_cfg.keys())
        if missing:
            raise KeyError(f"mcmc_cfg le faltan claves: {missing}")

        required_hp = {"atau", "btau", "ag", "bg", "apij", "bpij",
                       "mumu", "taumu", "mupsij", "taupsij", "pwj"}
        missing = required_hp - set(self.hp.keys())
        if missing:
            raise KeyError(f"hp le faltan claves: {missing}")

    # ─────────────────────────────────────────────────────────────────────
    # API pública
    # ─────────────────────────────────────────────────────────────────────
    def fit(self, X: np.ndarray, y: np.ndarray) -> "PSBPSampler":
        """
        Ejecuta el Gibbs sampler sobre (X, y).

        Parámetros
        ----------
        X : ndarray, shape (n, p+1)
            Matriz de diseño con primera columna de unos (intercepto).
            Debe estar **estandarizada** salvo por la columna del intercepto.
        y : ndarray, shape (n,)
            Respuesta **estandarizada**.
        """
        self._validate_inputs(X, y)
        self._run_gibbs(X, y)
        return self

    # ─────────────────────────────────────────────────────────────────────
    # Núcleo del sampler
    # ─────────────────────────────────────────────────────────────────────
    def _run_gibbs(self, X: np.ndarray, y: np.ndarray) -> None:
        """Loop principal del Gibbs sampler."""
        rng = np.random.default_rng(self.seed)
        EPS = np.finfo(float).tiny

        # ── Datos derivados ──────────────────────────────────────────────
        n = X.shape[0]
        Xnoint = X[:, 1:]
        p = Xnoint.shape[1]
        self.n_samples_ = n
        self.n_features_ = p

        # ── Configuración MCMC ───────────────────────────────────────────
        nsim = int(self.mcmc_cfg["nsim"])
        burn = int(self.mcmc_cfg["burn"])
        N    = int(self.mcmc_cfg["N"])
        M    = int(self.mcmc_cfg["M"])

        # ── Hiperparámetros (A2.1: aceptan escalares O arrays de long. p) ──
        atau    = float(self.hp["atau"])
        btau    = float(self.hp["btau"])
        ag      = float(self.hp["ag"])
        bg      = float(self.hp["bg"])
        mumu    = float(self.hp["mumu"])
        taumu   = float(self.hp["taumu"])
        pwj     = float(self.hp["pwj"])
        apij    = _expand_to_p(self.hp["apij"],    p, "apij")
        bpij    = _expand_to_p(self.hp["bpij"],    p, "bpij")
        mupsij  = _expand_to_p(self.hp["mupsij"],  p, "mupsij")
        taupsij = _expand_to_p(self.hp["taupsij"], p, "taupsij")

        # ── Inicialización ───────────────────────────────────────────────
        mu  = 1.0
        g   = 1.0
        pij = 0.5 * np.ones(p)
        wj  = np.ones(p)

        xmin, xmax = X.min(), X.max()
        Gstar = xmin + (np.arange(1, M + 1) / M) * (xmax - xmin)

        # bjrange: réplica exacta del MATLAB línea 54. MATLAB usa
        # `unidrnd(7)` que indexa solo los primeros 7 elementos. La
        # traducción `rng.integers(0, 7)` selecciona los mismos 7.
        bjrange = np.array([-4, -3, -2, -1.5, -1, 0, 1, 1.5, 2, 3, 4, 5])
        betajh  = bjrange[rng.integers(0, 7, size=(N, p))].astype(float)
        beta0h  = bjrange[rng.integers(0, 7, size=(N, 1))].astype(float)
        tauh    = rng.gamma(shape=atau, scale=1.0 / btau, size=(N, 1))
        Si      = rng.integers(0, N, size=n)
        alphah  = np.zeros(N - 1)
        psijh   = np.zeros((N - 1, p))
        gammajh = np.ones((N, p))
        Gammajh = Gstar[rng.integers(0, M, size=(N - 1, p))]

        # ── Contenedores de trazas ───────────────────────────────────────
        betajhout  = np.zeros((nsim, N, p),     dtype=np.float32)
        beta0hout  = np.zeros((nsim, N),        dtype=np.float32)
        tauhout    = np.zeros((nsim, N),        dtype=np.float32)
        alphahout  = np.zeros((nsim, N - 1),    dtype=np.float32)
        Gammajhout = np.zeros((nsim, N - 1, p), dtype=np.float32)
        psijhout   = np.zeros((nsim, N - 1, p), dtype=np.float32)
        gammajhout = np.zeros((nsim, N, p),     dtype=np.float32)
        pijout     = np.zeros((nsim, p),        dtype=np.float32)
        wjout      = np.zeros((nsim, p),        dtype=np.float32)
        N1out      = np.zeros(nsim,             dtype=np.float32)
        Nout       = np.zeros(nsim,             dtype=np.float32)
        muout      = np.zeros(nsim,             dtype=np.float32)
        osumout    = np.zeros((nsim, p),        dtype=np.float32)
        inEout     = np.zeros((nsim, n),        dtype=np.float32)

        # ── Loop MCMC ────────────────────────────────────────────────────
        for gt in range(nsim):

            # PASO 1 — Z_il (variables latentes Albert–Chib)
            Zil = np.zeros((n, N))
            Wil = np.zeros((n, N))
            for i in range(n):
                si = int(Si[i])
                n_iter = (si + 1) if si < N - 1 else (N - 1)
                for l in range(n_iter):
                    m_l = alphah[l] - np.sum(
                        psijh[l] * np.abs(Xnoint[i] - Gammajh[l])
                    )
                    if si < N - 1:
                        if l < si:
                            Zil[i, l] = _truncnorm_lower0_sample(m_l, 1.0, rng)
                        else:
                            Zil[i, l] = _truncnorm_upper0_sample(m_l, 1.0, rng)
                    else:
                        Zil[i, l] = _truncnorm_lower0_sample(m_l, 1.0, rng)
                    Wil[i, l] = Zil[i, l] + np.sum(
                        psijh[l] * np.abs(X[i, 1:] - Gammajh[l])
                    )

            # PASO 2 — S_i (asignaciones de cluster)
            # A2.2 — cálculo en log-space, replicando MATLAB línea 131:
            #   phx1 = exp(log(phx + realmin) + log(normpdf(...) + realmin))
            # Se estabiliza restando el máximo log-peso antes de exponenciar.
            phxi = np.zeros((n, N))
            for i in range(n):
                vhx = np.array([
                    norm.cdf(
                        alphah[h]
                        - np.sum(psijh[h] * np.abs(X[i, 1:] - Gammajh[h]))
                    )
                    for h in range(N - 1)
                ])
                phx = np.zeros(N)
                for h in range(N - 1):
                    phx[h] = vhx[h] if h == 0 else vhx[h] * np.prod(1.0 - vhx[:h])
                phx[N - 1] = np.prod(1.0 - vhx)
                phxi[i] = phx

                mu_h  = beta0h[:, 0] * X[i, 0] + betajh @ X[i, 1:]
                sig_h = 1.0 / np.sqrt(tauh[:, 0])
                log_lik = norm.logpdf(y[i], loc=mu_h, scale=sig_h)
                # log-pesos: log(phx + EPS) + log(normpdf + EPS)
                log_w = np.log(phx + EPS) + np.log(np.exp(log_lik) + EPS)
                # estabilización numérica
                log_w -= log_w.max()
                w = np.exp(log_w)
                Si[i] = rng.choice(N, p=w / w.sum())

            # PASO 3 — Predicción in-sample esperada por componente
            inE = np.zeros((n, N))
            for h in range(N):
                inE[:, h] = phxi[:, h] * (
                    X[:, 0] * beta0h[h, 0] + X[:, 1:] @ betajh[h]
                )

            # PASO 4 — beta_h (posterior Normal multivariada)
            betajh = np.zeros((N, p))
            for h in range(N):
                active = np.concatenate([[True], gammajh[h] == 1])
                Xh  = X[:, active]
                pgh = Xh.shape[1]
                idx = (Si == h)
                XhTXh  = Xh.T @ Xh
                Sh     = (n / g) * np.linalg.inv(XhTXh) / tauh[h, 0]
                Sh_inv = np.linalg.inv(Sh)
                if idx.sum() > 0:
                    XhSi   = Xh[idx]
                    Shhat  = np.linalg.inv(Sh_inv + tauh[h, 0] * XhSi.T @ XhSi)
                    muhhat = Shhat @ (tauh[h, 0] * XhSi.T @ y[idx])
                else:
                    Shhat  = Sh
                    muhhat = np.zeros(pgh)
                bt = _mvnrnd_safe(muhhat, Shhat, rng)
                beta0h[h, 0] = bt[0]
                cnt = 1
                for j in range(p):
                    if gammajh[h, j] == 1:
                        betajh[h, j] = bt[cnt]
                        cnt += 1

            # PASO 5 — tau_h (Gamma)
            for h in range(N):
                active = np.concatenate([[True], gammajh[h] == 1])
                Xh     = X[:, active]
                betagh = np.concatenate(
                    [[beta0h[h, 0]], betajh[h, gammajh[h] == 1]]
                )
                idx = (Si == h)
                rss = ((y[idx] - Xh[idx] @ betagh) ** 2).sum() if idx.sum() > 0 else 0.0
                aa = atau + 0.5 * idx.sum() + 0.5 * gammajh[h].sum() + 0.5
                bb = (btau + 0.5 * rss
                      + 0.5 / n * g * betagh @ (Xh.T @ Xh) @ betagh)
                tauh[h, 0] = rng.gamma(shape=aa, scale=1.0 / bb)

            # PASO 6 — g (Zellner)
            aghat = ag + 0.5 * (gammajh.sum() + N)
            temp = 0.0
            for h in range(N):
                active = np.concatenate([[True], gammajh[h] == 1])
                Xh     = X[:, active]
                betagh = np.concatenate(
                    [[beta0h[h, 0]], betajh[h, gammajh[h] == 1]]
                )
                temp += tauh[h, 0] * (betagh @ (Xh.T @ Xh) @ betagh)
            bghat = bg + 0.5 / n * temp
            g = rng.gamma(shape=aghat, scale=1.0 / bghat)

            # PASO 7 — w_j (Bernoulli con marginal Beta-Bernoulli)
            for j in range(p):
                if gammajh[:, j].sum() > 0:
                    wj[j] = 1.0
                else:
                    b = np.exp(
                        gammaln(bpij[j] + N) + gammaln(apij[j] + bpij[j])
                        - gammaln(bpij[j]) - gammaln(apij[j] + bpij[j] + N)
                    )
                    pwjhat = pwj * b / ((1.0 - pwj) + pwj * b)
                    wj[j] = float(rng.binomial(1, pwjhat))

            # PASO 8 — alpha_h (Normal)
            for h in range(N - 1):
                mask = (Si >= h)
                v_ah = 1.0 / (1.0 + mask.sum())
                m_ah = v_ah * (mu + Wil[mask, h].sum())
                alphah[h] = rng.normal(m_ah, np.sqrt(v_ah))

            # PASO 9 — mu (Normal)
            taumuhat = N + taumu
            mumuhat  = (taumu * mumu + alphah.sum()) / taumuhat
            mu = rng.normal(mumuhat, np.sqrt(1.0 / taumuhat))

            # PASO 10 — pi_j (Beta)
            for j in range(p):
                if wj[j] == 0:
                    pij[j] = 0.0
                else:
                    pij[j] = rng.beta(
                        apij[j] + gammajh[:, j].sum(),
                        bpij[j] + N - gammajh[:, j].sum()
                    )

            # PASO 11 — Gamma_jh (grilla discreta)
            # A4 — Cuando kh == 0 y gammajh[h,j] == 1: muestreo uniforme
            #       de Gstar (replica MATLAB líneas 236-254).
            for h in range(N - 1):
                mask_h = (Si >= h)
                kh = mask_h.sum()
                for j in range(p):
                    if gammajh[h, j] != 1:
                        continue
                    if kh > 0:
                        pm = np.zeros(M)
                        for m_idx in range(M):
                            od = (
                                np.sum(psijh[h, :j]
                                       * np.abs(Xnoint[mask_h, :j]
                                                - Gammajh[h, :j]),
                                       axis=1)
                                + np.sum(psijh[h, j + 1:]
                                         * np.abs(Xnoint[mask_h, j + 1:]
                                                  - Gammajh[h, j + 1:]),
                                         axis=1)
                            )
                            mz = (alphah[h] - od
                                  - psijh[h, j]
                                  * np.abs(Xnoint[mask_h, j] - Gstar[m_idx]))
                            pm[m_idx] = (
                                np.exp(1.2 * kh
                                       + norm.logpdf(Zil[mask_h, h], mz, 1.0).sum())
                                + EPS
                            )
                        pm1 = pm / pm.sum()
                        Gammajh[h, j] = Gstar[rng.choice(M, p=pm1)]
                    else:
                        # kh == 0: MATLAB produce uniforme sobre Gstar
                        # (suma vacía → pm[m] = exp(0)+EPS = constante).
                        Gammajh[h, j] = Gstar[rng.integers(0, M)]

            # PASO 12 — psi_jh (Normal truncada positiva)
            # A4-b — Cuando kh == 0 y gammajh[h,j] == 1: muestreo de la
            #         prior truncada N(mupsij_j, 1/taupsij_j) I(psi >= 0)
            #         (replica MATLAB líneas 256-276).
            for h in range(N - 1):
                mask_h = (Si >= h)
                kh = mask_h.sum()
                for j in range(p):
                    if gammajh[h, j] == 0:
                        psijh[h, j] = 0.0
                    elif kh > 0:
                        od = (
                            np.sum(psijh[h, :j]
                                   * np.abs(Xnoint[mask_h, :j]
                                            - Gammajh[h, :j]),
                                   axis=1)
                            + np.sum(psijh[h, j + 1:]
                                     * np.abs(Xnoint[mask_h, j + 1:]
                                              - Gammajh[h, j + 1:]),
                                     axis=1)
                        )
                        Tijh   = alphah[h] - Zil[mask_h, h] - od
                        dist_j = np.abs(Xnoint[mask_h, j] - Gammajh[h, j])
                        v_psi  = 1.0 / (taupsij[j] + dist_j @ dist_j)
                        m_psi  = v_psi * (taupsij[j] * mupsij[j]
                                          + np.sum(Tijh * dist_j))
                        psijh[h, j] = _truncnorm_upper0_sample(m_psi, v_psi, rng)
                    else:
                        # kh == 0: muestreo de la prior truncada positiva.
                        # En MATLAB esto equivale a v = 1/taupsij_j, m = mupsij_j
                        # (las sumas con vector vacío dan 0).
                        v_psi = 1.0 / taupsij[j]
                        m_psi = mupsij[j]
                        psijh[h, j] = _truncnorm_upper0_sample(m_psi, v_psi, rng)

            # PASO 13 — gamma_jh (selección de variables)
            for h in range(N):
                for j in range(p):
                    act_noj = gammajh[h].copy()
                    act_noj[j] = 0
                    full_noj = np.concatenate([[True], act_noj == 1])
                    Xh_noj   = X[:, full_noj]
                    Xh1      = np.hstack([Xnoint[:, j:j + 1], Xh_noj])
                    sb       = (n / g) * np.linalg.inv(Xh1.T @ Xh1) / tauh[h, 0]
                    bvec     = np.concatenate([[beta0h[h, 0]], betajh[h]])
                    betagh2  = bvec[full_noj]

                    if sb.shape[0] > 1:
                        sb11  = sb[0, 0]
                        sb12  = sb[0, 1:]
                        sb22  = sb[1:, 1:]
                        sbj   = sb11 - sb12 @ np.linalg.solve(sb22, sb12)
                        taubj = 1.0 / sbj
                        mubj  = sb12 @ np.linalg.solve(sb22, betagh2)
                    else:
                        taubj = 1.0 / sb[0, 0]
                        mubj  = 0.0

                    idx_h = (Si == h)
                    n_h   = idx_h.sum()
                    ystar = (y[idx_h]
                             - X[idx_h, 0] * beta0h[h, 0]
                             - Xnoint[idx_h, :j]      @ betajh[h, :j]
                             - Xnoint[idx_h, j + 1:]  @ betajh[h, j + 1:])

                    pp = tauh[h, 0] * (Xnoint[idx_h, j] @ Xnoint[idx_h, j]) + taubj
                    if pp > 0:
                        pm_val = ((tauh[h, 0] * Xnoint[idx_h, j] @ ystar
                                   + taubj * mubj) / pp)
                        mb = (norm.logpdf(0, mubj, 1.0 / np.sqrt(taubj))
                              - norm.logpdf(0, pm_val, np.sqrt(1.0 / pp)))
                    else:
                        mb = 0.0

                    if h < N - 1:
                        mask_h = (Si >= h)
                        kh     = mask_h.sum()
                        od = (
                            np.sum(psijh[h, :j]
                                   * np.abs(Xnoint[mask_h, :j]
                                            - Gammajh[h, :j]),
                                   axis=1)
                            + np.sum(psijh[h, j + 1:]
                                     * np.abs(Xnoint[mask_h, j + 1:]
                                              - Gammajh[h, j + 1:]),
                                     axis=1)
                        )
                        Tijh   = alphah[h] - Zil[mask_h, h] - od
                        dist_j = np.abs(Xnoint[mask_h, j] - Gammajh[h, j])
                        v_psi  = 1.0 / (taupsij[j] + dist_j @ dist_j)
                        m_psi  = v_psi * (taupsij[j] * mupsij[j]
                                          + np.sum(Tijh * dist_j))

                        lik0_y = norm.logpdf(
                            y[idx_h],
                            X[idx_h, 0] * beta0h[h, 0]
                            + Xnoint[idx_h, :j]     @ betajh[h, :j]
                            + Xnoint[idx_h, j + 1:] @ betajh[h, j + 1:],
                            1.0 / np.sqrt(tauh[h, 0])
                        ).sum()
                        lik0_Z = norm.logpdf(
                            Zil[mask_h, h], alphah[h] - od, 1.0
                        ).sum()
                        bjhin = np.log(1.0 - pij[j] + EPS) + lik0_y + lik0_Z

                        prior_psi = (
                            norm.logpdf(0, mupsij[j], 1.0 / np.sqrt(taupsij[j]))
                            - np.log(1.0 - norm.cdf(
                                (0.0 - mupsij[j]) * np.sqrt(taupsij[j])
                            ) + EPS)
                        )
                        post_psi = (
                            -norm.logpdf(0, m_psi, np.sqrt(v_psi))
                            + np.log(1.0 - norm.cdf(
                                (0.0 - m_psi) / np.sqrt(v_psi)
                            ) + EPS)
                        )
                        ajhin = (
                            np.log(pij[j] + EPS)
                            + norm.logpdf(ystar, 0, 1.0 / np.sqrt(tauh[h, 0])).sum()
                            + mb
                            + norm.logpdf(Tijh, 0, 1.0).sum()
                            + prior_psi + post_psi
                        )

                        diff = np.clip(bjhin - ajhin, -500.0, 500.0)
                        prob1 = 1.0 / (1.0 + np.exp(diff))
                        gammajh[h, j] = float(rng.binomial(1, prob1))

                    else:  # h == N-1
                        b0_log = (
                            1.2 * n_h + np.log(1.0 - pij[j] + EPS)
                            + norm.logpdf(
                                y[idx_h],
                                X[idx_h, 0] * beta0h[h, 0]
                                + Xnoint[idx_h, :j]     @ betajh[h, :j]
                                + Xnoint[idx_h, j + 1:] @ betajh[h, j + 1:],
                                1.0 / np.sqrt(tauh[h, 0])
                            ).sum()
                        )
                        a0_log = (
                            1.2 * n_h + np.log(pij[j] + EPS)
                            + norm.logpdf(ystar, 0, 1.0 / np.sqrt(tauh[h, 0])).sum()
                            + mb
                        )
                        m_log = max(a0_log, b0_log)
                        prob1 = np.exp(a0_log - m_log) / (
                            np.exp(a0_log - m_log) + np.exp(b0_log - m_log)
                        )
                        gammajh[h, j] = float(rng.binomial(1, np.clip(prob1, 0.0, 1.0)))

            # PASO 14 — Almacenar trazas
            max_si = int(Si.max())
            osumout[gt]    = (gammajh[:max_si + 1].sum(axis=0) == 0).astype(float)
            muout[gt]      = mu
            tauhout[gt]    = tauh[:, 0]
            beta0hout[gt]  = beta0h[:, 0]
            betajhout[gt]  = betajh
            alphahout[gt]  = alphah
            psijhout[gt]   = psijh
            Gammajhout[gt] = Gammajh
            gammajhout[gt] = gammajh
            pijout[gt]     = pij
            wjout[gt]      = wj
            N1out[gt]      = max_si + 1
            Nout[gt]       = N
            inEout[gt]     = inE.sum(axis=1)

            if self.verbose_every and (gt + 1) % self.verbose_every == 0:
                print(f"  iter {gt + 1}/{nsim}")

        # ── Empaquetar trazas ────────────────────────────────────────────
        self.traces = dict(
            betajhout=betajhout,
            beta0hout=beta0hout,
            tauhout=tauhout,
            alphahout=alphahout,
            psijhout=psijhout,
            Gammajhout=Gammajhout,
            gammajhout=gammajhout,
            pijout=pijout,
            wjout=wjout,
            muout=muout,
            osumout=osumout,
            N1out=N1out,
            Nout=Nout,
            inEout=inEout,
        )

    # ─────────────────────────────────────────────────────────────────────
    def get_config(self) -> Dict[str, Any]:
        return {
            "mcmc_cfg":    dict(self.mcmc_cfg),
            "hp":          dict(self.hp),
            "seed":        self.seed,
            "n_samples_":  self.n_samples_,
            "n_features_": self.n_features_,
        }
