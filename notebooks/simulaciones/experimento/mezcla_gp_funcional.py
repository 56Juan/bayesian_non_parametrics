r"""
mezcla_gp_funcional.py — Prototipo: mezcla probit stick-breaking con atomos GP
==============================================================================

EXPERIMENTO, no parte del pipeline canonico (corrida 61). Vive junto a su
notebook en `notebooks/simulaciones/experimento/` y no se importa desde el
paquete. Muestrea en Python, no en MATLAB.

El modelo
---------
Sea X_t la curva del periodo t, representada por sus coeficientes a_t en K
funciones de una base ORTONORMAL en L^2 (Fourier en la familia C; coeficientes
blanqueados de una B-spline en general), y sea x_t = (1, z_t) con z_t los
rezagos estandarizados de las componentes elegidas. Hay UNA asignacion S_t por
curva completa:

    a_t | S_t = h, x_t  ~  N_K( A_h' x_t , C_h )                      (atomo)
    Pr(S_t = h | x_t)   =  pi_h(x_t)                                  (pesos)

    pi_h = v_h prod_{l<h} (1 - v_l),   v_h = Phi(eta_h),   v_N = 1
    eta_h(x) = alpha_h + omega_h' z                    (probit lineal-funcional)

Llevado a la curva, X_t = mu + phi(tau)' a_t, cada atomo es un proceso
gaussiano cuya media es un FAR(q) propio del regimen,

    m_h(tau) = mu(tau) + phi(tau)' A_h' x_t,

y cuya covarianza es

    covarianza = "gp" :  C_h = sigma_h^2 S(ell_h) + s_h^2 I,
                         S(ell) = Phi' W K_ell W Phi,
    covarianza = "iw" :  C_h ~ Inv-Wishart           (control sin estructura).

S(ell) es la covarianza de los coeficientes de un GP de varianza 1 y escala
ell, proyectado sobre la base: dos parametros por atomo en vez de K(K+1)/2.
s_h^2 I es una pepita blanca, que absorbe lo que el nucleo suave no alcanza en
las componentes de alta frecuencia.

Priors
------
    A_h | C_h ~ MN( 0, V0, C_h ),  V0 = (n/g) (X'X)^{-1}         g-prior matricial
    (ell_h, sigma_h, s_h) uniforme sobre una grilla (escalada por la sd total)
    alpha_h ~ N(mu_a, 1),  mu_a ~ N(mu_mu, 1/tau_mu)
    omega_h ~ N(0, s_omega^2 / p · I)                      unidades probit

Por que el Gibbs es exacto (sin Metropolis)
-------------------------------------------
1. S_t: discreta, pi_h(x_t) N_K(a_t; A_h'x_t, C_h).
2. Atomo h, con A_h INTEGRADO: con P_n = V0^{-1} + X_h'X_h, B = P_n^{-1} X_h'Y_h
   y Q = Y_h'Y_h - B' P_n B,

       log p(C | Y_h) = -(n_h/2) log|C| - (1/2) tr(C^{-1} Q) + cte.

   Con C = U diag(d) U', donde U son los vectores propios de S(ell), fijo por
   ell, el determinante y la traza cuestan O(K) por punto de la grilla. Por eso
   (ell, sigma, s) se muestrea EXACTO sobre la grilla completa.
3. A_h | C_h ~ MN(B, P_n^{-1}, C_h).
4. Z*: normales truncadas (Albert-Chib), con la inversa en escala logaritmica
   (`ndtri_exp`) para no romper la truncacion cuando |eta| es grande.
5. (alpha_h, omega_h) | Z*: normal conjugada. mu_a | alpha: normal.

Que NO hace (todavia)
---------------------
- Seleccion de variables (gamma): todas las covariables entran en todos los
  atomos. El PIP no existe en este prototipo.
- Gating de carpa funcional: solo el lineal-funcional.
- Diagnostico sobre etiquetas: los atomos no son identificables. Lo invariante
  es la log-verosimilitud y la media condicional E[a_t | x_t] por extraccion,
  que es lo que exponen `muestrear_cadena` y `media_condicional`.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Sequence

import numpy as np
from scipy.special import log_ndtr, logsumexp, ndtri_exp
from scipy.stats import invwishart

from model_psbp_fd.utils.linalg import safe_chol
from model_psbp_fd.utils.quadrature import pesos_trapezoidales

__all__ = [
    "ARCHIVO_TRAZA",
    "ConfigMezclaGP",
    "Diseno",
    "nucleo",
    "proyectar",
    "covarianzas_proyectadas",
    "disenar",
    "log_pesos",
    "muestrear_cadena",
    "guardar_traza",
    "cargar_trazas",
    "media_condicional",
    "muestras_predictivas",
    "banda_funcional",
]

#: Nombre de la traza de cada cadena dentro de `out_artefact`. Unica definicion.
ARCHIVO_TRAZA = "traza_mezcla_gp_cadena{cadena:02d}.npz"

NUCLEOS = ("matern32", "exponencial", "gaussiano")
COVARIANZAS = ("gp", "iw")

_SIG_REL = tuple(float(v) for v in np.round(np.geomspace(0.02, 2.0, 14), 6))


# ==========================================================================
# CONFIGURACION
# ==========================================================================

@dataclass
class ConfigMezclaGP:
    """
    N          : truncamiento del stick-breaking.
    n_iter     : iteraciones totales; se guardan las de t >= burn cada `thin`.
    covarianza : "gp" (sigma^2 S(ell) + s^2 I) o "iw" (Wishart inversa).
    nucleo     : nucleo del GP del atomo, uno de NUCLEOS.
    ells       : grilla de escalas del GP, en unidades del dominio [0, 1].
    sig_rel    : grilla de sigma, RELATIVA a la sd total de los coeficientes.
    nug_rel    : grilla de la pepita s, relativa a la misma sd total.
    g          : g del g-prior matricial (V0 = (n/g)(X'X)^{-1}).
    s_omega    : sd a priori del gating, en unidades probit; se divide por
                 sqrt(p) para que la dispersion total de eta no crezca con p.
    nu0_extra  : solo "iw": nu0 = K + nu0_extra.
    n_inicial  : atomos ocupados al arrancar.
    """

    N: int = 30
    n_iter: int = 3000
    burn: int = 1000
    thin: int = 4
    covarianza: str = "gp"
    nucleo: str = "matern32"
    ells: tuple = (0.03, 0.05, 0.08, 0.12, 0.18, 0.25, 0.35, 0.5)
    sig_rel: tuple = _SIG_REL
    nug_rel: tuple = (0.002, 0.005, 0.01, 0.02, 0.05)
    g: float = 1.0
    s_omega: float = 0.8
    mu_mu: float = 0.0
    tau_mu: float = 1.0
    nu0_extra: float = 3.0
    n_inicial: int = 5
    seed: int = 41232

    def validar(self) -> None:
        if self.covarianza not in COVARIANZAS:
            raise ValueError(f"covarianza={self.covarianza!r}; use una de {COVARIANZAS}.")
        if self.nucleo not in NUCLEOS:
            raise ValueError(f"nucleo={self.nucleo!r}; use uno de {NUCLEOS}.")
        if not 0 <= self.burn < self.n_iter:
            raise ValueError("burn debe estar en [0, n_iter).")
        if self.N < 2 or self.thin < 1:
            raise ValueError("N >= 2 y thin >= 1.")


@dataclass
class Diseno:
    """Respuesta y covariables alineadas por origen.

    Y      : (n, K) coeficientes de la curva respuesta.
    X      : (n, 1 + p) intercepto + rezagos ESTANDARIZADOS con el bloque train.
    t_idx  : (n,) indice base-0 de la curva respuesta en la serie completa.
    """

    Y: np.ndarray
    X: np.ndarray
    t_idx: np.ndarray
    n_lags: int
    componentes: tuple
    nombres: List[str]
    centro: np.ndarray
    escala: np.ndarray
    T0: int

    @property
    def train(self) -> np.ndarray:
        """Filas cuya respuesta cae en el bloque de entrenamiento."""
        return self.t_idx < self.T0

    @property
    def T0_ev(self) -> int:
        """Corte train/test en el indexado de las filas (para la ventana movil)."""
        return int(self.train.sum())


# ==========================================================================
# BASE Y COVARIANZAS PROYECTADAS
# ==========================================================================

def nucleo(tau: np.ndarray, ell: float, tipo: str) -> np.ndarray:
    """Nucleo estacionario de varianza 1 evaluado en la grilla."""
    D = np.abs(tau[:, None] - tau[None, :])
    if tipo == "matern32":
        r = np.sqrt(3.0) * D / ell
        return (1.0 + r) * np.exp(-r)
    if tipo == "exponencial":
        return np.exp(-D / ell)
    if tipo == "gaussiano":
        return np.exp(-0.5 * (D / ell) ** 2)
    raise ValueError(f"nucleo {tipo!r} desconocido; use uno de {NUCLEOS}.")


def _verificar_ortonormal(Phi: np.ndarray, tau: np.ndarray, tol: float = 1e-6) -> np.ndarray:
    w = pesos_trapezoidales(tau)
    G = Phi.T @ (w[:, None] * Phi)
    err = float(np.abs(G - np.eye(Phi.shape[1])).max())
    if err > tol:
        raise ValueError(
            f"la base no es ortonormal en la cuadratura (error {err:.2e}): el "
            "modelo supone coeficientes en metrica L^2. Blanquee con la Gram.")
    return w


def proyectar(Xc: np.ndarray, Phi: np.ndarray, tau: np.ndarray) -> np.ndarray:
    """Coeficientes <X_t, phi_k>_{L^2} de curvas ya centradas. (T, G) -> (T, K)."""
    w = _verificar_ortonormal(Phi, tau)
    return Xc @ (w[:, None] * Phi)


def covarianzas_proyectadas(Phi: np.ndarray, tau: np.ndarray,
                            ells: Sequence[float], tipo: str) -> list:
    """
    S(ell) = Phi' W K_ell W Phi para cada ell de la grilla, descompuesta como
    S = U diag(lam) U'. Es la covarianza de los K coeficientes de un GP de
    varianza 1 proyectado sobre la base. Se calcula UNA vez por experimento.
    """
    w = _verificar_ortonormal(Phi, tau)
    Pw = Phi * w[:, None]
    salida = []
    for ell in ells:
        S = Pw.T @ nucleo(tau, float(ell), tipo) @ Pw
        lam, U = np.linalg.eigh(0.5 * (S + S.T))
        salida.append((np.clip(lam, 0.0, None), U))
    return salida


# ==========================================================================
# DISENO
# ==========================================================================

def disenar(A: np.ndarray, n_lags: int, componentes: Sequence[int], T0: int) -> Diseno:
    """
    Arma (Y, X) sobre los origenes t = n_lags, ..., T-1.

    componentes : componentes (BASE 1) cuyos rezagos entran como covariables.
    El orden de las columnas es lag-mayor: (a_{c1,t-1}, a_{c2,t-1}, ...,
    a_{c1,t-2}, ...), igual que `Z` en el `_05`. La estandarizacion se ajusta
    SOLO con las filas cuya respuesta cae en train.
    """
    A = np.asarray(A, dtype=float)
    T, K = A.shape
    comp0 = [int(c) - 1 for c in componentes]
    if not comp0 or min(comp0) < 0 or max(comp0) >= K:
        raise ValueError(f"componentes {tuple(componentes)} fuera de 1..{K}.")
    if n_lags < 1:
        raise ValueError("n_lags >= 1.")
    t_idx = np.arange(n_lags, T)
    cols, nombres = [], []
    for l in range(1, n_lags + 1):
        for j in comp0:
            cols.append(A[t_idx - l, j])
            nombres.append(f"a{j + 1}_lag{l}")
    Z = np.column_stack(cols)
    tr = t_idx < T0
    centro, escala = Z[tr].mean(axis=0), Z[tr].std(axis=0)
    X = np.column_stack([np.ones(t_idx.size), (Z - centro) / escala])
    return Diseno(Y=A[t_idx], X=X, t_idx=t_idx, n_lags=int(n_lags),
                  componentes=tuple(int(c) for c in componentes), nombres=nombres,
                  centro=centro, escala=escala, T0=int(T0))


# ==========================================================================
# PESOS
# ==========================================================================

def log_pesos(eta: np.ndarray) -> np.ndarray:
    """log pi_h a partir de eta (n, N-1), estable en las colas. Retorna (n, N)."""
    lv, l1v = log_ndtr(eta), log_ndtr(-eta)
    n = eta.shape[0]
    acum = np.concatenate([np.zeros((n, 1)), np.cumsum(l1v, axis=1)], axis=1)
    return np.concatenate([lv, np.zeros((n, 1))], axis=1) + acum


def _normal_truncada(eta: np.ndarray, positiva: np.ndarray,
                     rng: np.random.Generator) -> np.ndarray:
    """Z ~ N(eta, 1) truncada a (0, inf) donde `positiva`, a (-inf, 0) si no.

    Inversa en escala log: con eta = 10 y truncacion negativa, u * Phi(-eta)
    vale 1e-23 y `ndtri` clipeado devolveria un Z > 0.
    """
    lu = np.log(rng.random(eta.shape))
    z_neg = eta + ndtri_exp(lu + log_ndtr(-eta))          # en (-inf, 0)
    z_pos = eta - ndtri_exp(lu + log_ndtr(eta))           # en (0, inf)
    return np.where(positiva, z_pos, z_neg)


# ==========================================================================
# MUESTREADOR
# ==========================================================================

def _escalas(Y: np.ndarray, cfg: ConfigMezclaGP) -> tuple[np.ndarray, np.ndarray]:
    sd_tot = float(np.sqrt(np.trace(np.atleast_2d(np.cov(Y.T)))))
    return np.asarray(cfg.sig_rel) * sd_tot, np.asarray(cfg.nug_rel) * sd_tot


def muestrear_cadena(Y: np.ndarray, X: np.ndarray, S_eig: list,
                     cfg: ConfigMezclaGP, cadena: int = 1,
                     verbose: bool = True) -> dict:
    """
    Una cadena del Gibbs. Y (n, K) y X (n, 1+p) son SOLO el bloque train.

    Retorna un dict con las trazas guardadas (post-burn, cada `thin`) y los
    diagnosticos por iteracion (todas): `loglik`, `n_ocupados`, `masa_max`.
    """
    cfg.validar()
    rng = np.random.default_rng([int(cfg.seed), int(cadena)])
    Y = np.asarray(Y, dtype=float)
    X = np.asarray(X, dtype=float)
    n, K = Y.shape
    q = X.shape[1]
    p = q - 1
    N = int(cfg.N)
    gp = cfg.covarianza == "gp"

    SIG, NUG = _escalas(Y, cfg)
    nE, nS, nN = len(S_eig), SIG.size, NUG.size
    lam_E = np.stack([le for le, _ in S_eig])              # (nE, K)
    U_E = np.stack([ue for _, ue in S_eig])                # (nE, K, K)

    # g-prior matricial sobre A_h
    V0 = (n / cfg.g) * np.linalg.inv(X.T @ X)
    V0i = np.linalg.inv(V0)
    # Wishart inversa (solo "iw"): E[C] = diag(var(Y)) / 2
    nu0 = K + cfg.nu0_extra
    Lam0 = (nu0 - K - 1.0) * np.diag(Y.var(axis=0)) * 0.5
    # gating
    s_om = cfg.s_omega / np.sqrt(max(p, 1))
    P0 = np.diag(np.r_[1.0, np.full(p, 1.0 / s_om ** 2)])

    # estado inicial
    S = rng.integers(0, min(N, cfg.n_inicial), n)
    A = np.zeros((N, q, K))
    idx = np.tile([nE // 2, nS // 2, nN // 2], (N, 1))
    Csig = np.repeat(np.diag(Y.var(axis=0))[None], N, axis=0)
    alpha = np.zeros(N - 1)
    om = np.zeros((N - 1, p))
    mu_a = 0.0

    guardar = np.arange(cfg.burn, cfg.n_iter, cfg.thin)
    nk = guardar.size
    tr_A = np.zeros((nk, N, q, K), dtype=np.float32)
    tr_idx = np.zeros((nk, N, 3), dtype=np.int16)
    tr_C = np.zeros((nk, N, K, K), dtype=np.float32) if not gp else None
    tr_alpha = np.zeros((nk, N - 1))
    tr_om = np.zeros((nk, N - 1, p))
    tr_mu = np.zeros(nk)
    loglik = np.zeros(cfg.n_iter)
    n_ocup = np.zeros(cfg.n_iter, dtype=np.int16)
    masa_max = np.zeros(cfg.n_iter)
    t_pasos = np.zeros(4)                                  # S, atomos, Z*, gating

    def factor(h):
        if gp:
            a, b, c = idx[h]
            return U_E[a], SIG[b] ** 2 * lam_E[a] + NUG[c] ** 2
        d, U = np.linalg.eigh(Csig[h])
        return U, np.clip(d, 1e-12, None)

    lgrid_d = None
    if gp:
        # d[a, b, c, k] = SIG_b^2 lam_{a,k} + NUG_c^2, fijo: se precalcula.
        lgrid_d = (SIG[None, :, None, None] ** 2 * lam_E[:, None, None, :]
                   + NUG[None, None, :, None] ** 2)
        log_d_sum = np.log(lgrid_d).sum(axis=-1)           # (nE, nS, nN)

    t_ini = time.time()
    k_guardar = 0
    for it in range(cfg.n_iter):
        # ---- 1. asignaciones S_t ------------------------------------------
        t0 = time.time()
        eta = alpha[None, :] + X[:, 1:] @ om.T
        lp = log_pesos(eta)
        for h in range(N):
            U, d = factor(h)
            R = (Y - X @ A[h]) @ U
            lp[:, h] += -0.5 * ((R ** 2) / d).sum(axis=1) - 0.5 * np.log(d).sum()
        lse = logsumexp(lp, axis=1)
        loglik[it] = float(lse.sum() - 0.5 * n * K * np.log(2 * np.pi))
        prob = np.exp(lp - lse[:, None])
        S = (np.cumsum(prob, axis=1) > rng.random((n, 1))).argmax(axis=1)
        cuenta = np.bincount(S, minlength=N)
        n_ocup[it] = int((cuenta > 0).sum())
        masa_max[it] = float(cuenta.max() / n)
        t_pasos[0] += time.time() - t0

        # ---- 2. atomos: covarianza (A integrado) y luego A | C ------------
        t0 = time.time()
        for h in range(N):
            m = S == h
            nh = int(cuenta[h])
            Xh, Yh = X[m], Y[m]
            Pn = V0i + Xh.T @ Xh
            Vn = np.linalg.inv(Pn)
            Vn = 0.5 * (Vn + Vn.T)
            B = Vn @ (Xh.T @ Yh)
            Q = Yh.T @ Yh - B.T @ Pn @ B
            Q = 0.5 * (Q + Q.T)
            if gp:
                if nh == 0:
                    idx[h] = [rng.integers(nE), rng.integers(nS), rng.integers(nN)]
                else:
                    qd = np.einsum("aki,kl,ali->ai", U_E, Q, U_E)     # diag(U'QU)
                    lpg = -0.5 * nh * log_d_sum - 0.5 * (qd[:, None, None, :] / lgrid_d).sum(-1)
                    pr = np.exp(lpg - lpg.max()).ravel()
                    k = rng.choice(pr.size, p=pr / pr.sum())
                    idx[h] = np.unravel_index(k, lpg.shape)
            else:
                Csig[h] = invwishart.rvs(df=nu0 + nh, scale=Lam0 + Q, random_state=rng)
            U, d = factor(h)
            A[h] = B + safe_chol(Vn) @ rng.standard_normal((q, K)) @ (U * np.sqrt(d)).T
        t_pasos[1] += time.time() - t0

        # ---- 3. latentes Z* --------------------------------------------------
        t0 = time.time()
        lidx = np.arange(N - 1)[None, :]
        activo = lidx <= np.minimum(S, N - 2)[:, None]
        positivo = lidx == S[:, None]
        Zs = _normal_truncada(eta, positivo, rng)
        t_pasos[2] += time.time() - t0

        # ---- 4. gating (alpha_h, omega_h) y mu_a -----------------------------
        t0 = time.time()
        m0 = np.r_[mu_a, np.zeros(p)]
        for h in range(N - 1):
            r = activo[:, h]
            D = X[r]
            Qg = P0 + D.T @ D
            Lg = np.linalg.cholesky(Qg)
            media = np.linalg.solve(Qg, D.T @ Zs[r, h] + P0 @ m0)
            b = media + np.linalg.solve(Lg.T, rng.standard_normal(q))
            alpha[h], om[h] = b[0], b[1:]
        prec = cfg.tau_mu + (N - 1)
        mu_a = rng.normal((cfg.tau_mu * cfg.mu_mu + alpha.sum()) / prec, 1.0 / np.sqrt(prec))
        t_pasos[3] += time.time() - t0

        if k_guardar < nk and it == guardar[k_guardar]:
            tr_A[k_guardar] = A
            tr_idx[k_guardar] = idx
            if not gp:
                tr_C[k_guardar] = Csig
            tr_alpha[k_guardar] = alpha
            tr_om[k_guardar] = om
            tr_mu[k_guardar] = mu_a
            k_guardar += 1

        if verbose and ((it + 1) % max(cfg.n_iter // 10, 1) == 0 or it == 0):
            print(f"  [cadena {cadena}] it {it + 1:>5}/{cfg.n_iter}  "
                  f"loglik={loglik[it]:.1f}  ocupados={n_ocup[it]}  "
                  f"masa_max={masa_max[it]:.2f}  "
                  f"({(time.time() - t_ini) / (it + 1) * 1000:.0f} ms/it)")

    seg = time.time() - t_ini
    traza = {
        "A": tr_A, "idx_cov": tr_idx, "alpha": tr_alpha, "omega": tr_om, "mu_a": tr_mu,
        "loglik": loglik, "n_ocupados": n_ocup, "masa_max": masa_max,
        "lam_E": lam_E, "U_E": U_E, "SIG": SIG, "NUG": NUG,
        "ms_por_iter": np.array(seg / cfg.n_iter * 1000.0),
        "ms_por_paso": t_pasos / cfg.n_iter * 1000.0,
        "config": asdict(cfg), "cadena": int(cadena),
    }
    if not gp:
        traza["C"] = tr_C
    return traza


# ==========================================================================
# PERSISTENCIA DE LAS TRAZAS
# ==========================================================================

def guardar_traza(paths: dict, traza: dict, cadena: int) -> Path:
    """Escribe la traza de una cadena en `out_artefact` (npz comprimido)."""
    ruta = Path(paths["out_artefact"]) / ARCHIVO_TRAZA.format(cadena=int(cadena))
    arrays = {k: v for k, v in traza.items() if isinstance(v, np.ndarray)}
    arrays["config_json"] = np.array(json.dumps(traza["config"]))
    arrays["cadena"] = np.array(int(cadena))
    np.savez_compressed(ruta, **arrays)
    return ruta


def cargar_trazas(paths: dict, n_cadenas: int) -> list:
    """Lee las `n_cadenas` trazas escritas por `guardar_traza`."""
    trazas = []
    for c in range(1, int(n_cadenas) + 1):
        ruta = Path(paths["out_artefact"]) / ARCHIVO_TRAZA.format(cadena=c)
        if not ruta.exists():
            raise FileNotFoundError(f"No se encontro {ruta}. Corra el muestreo (§4).")
        with np.load(ruta, allow_pickle=False) as z:
            tr = {k: z[k] for k in z.files if k != "config_json"}
            tr["config"] = json.loads(str(z["config_json"]))
        tr["cadena"] = int(tr["cadena"])
        trazas.append(tr)
    return trazas


# ==========================================================================
# PREDICTIVA
# ==========================================================================

def _factores_draw(traza: dict, s: int) -> tuple[np.ndarray, np.ndarray]:
    """(U, d) de los N atomos en la extraccion s: C_h = U_h diag(d_h) U_h'."""
    if traza["config"]["covarianza"] == "gp":
        a, b, c = (traza["idx_cov"][s, :, i].astype(int) for i in range(3))
        U = traza["U_E"][a]
        d = traza["SIG"][b][:, None] ** 2 * traza["lam_E"][a] + traza["NUG"][c][:, None] ** 2
        return U, d
    d, U = np.linalg.eigh(traza["C"][s].astype(float))
    return U, np.clip(d, 1e-12, None)


def media_condicional(traza: dict, X: np.ndarray) -> np.ndarray:
    """
    E[a_t | x_t] por extraccion: sum_h pi_h(x_t) A_h' x_t. Retorna (S, n, K).
    Es invariante a la permutacion de etiquetas: sirve para R-hat.
    """
    X = np.asarray(X, dtype=float)
    S_, n = traza["A"].shape[0], X.shape[0]
    K = traza["A"].shape[-1]
    out = np.empty((S_, n, K))
    for s in range(S_):
        pi = np.exp(log_pesos(traza["alpha"][s][None, :] + X[:, 1:] @ traza["omega"][s].T))
        medias = np.einsum("nq,hqk->nhk", X, traza["A"][s].astype(float))
        out[s] = np.einsum("nh,nhk->nk", pi, medias)
    return out


def muestras_predictivas(traza: dict, X: np.ndarray,
                         rng: np.random.Generator) -> np.ndarray:
    """Una extraccion de la predictiva por traza guardada: (S, n, K)."""
    X = np.asarray(X, dtype=float)
    S_, n = traza["A"].shape[0], X.shape[0]
    K = traza["A"].shape[-1]
    out = np.empty((S_, n, K))
    for s in range(S_):
        pi = np.exp(log_pesos(traza["alpha"][s][None, :] + X[:, 1:] @ traza["omega"][s].T))
        h = (np.cumsum(pi, axis=1) > rng.random((n, 1))).argmax(axis=1)
        A = traza["A"][s].astype(float)
        media = np.einsum("nq,nqk->nk", X, A[h])
        U, d = _factores_draw(traza, s)
        z = rng.standard_normal((n, K)) * np.sqrt(d[h])
        out[s] = media + np.einsum("nkj,nj->nk", U[h], z)
    return out


def banda_funcional(muestras: np.ndarray, Phi: np.ndarray, mu: np.ndarray,
                    nivel: float = 0.95, bloque: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """
    Banda puntual de nivel `nivel` sobre la curva mu + Phi a, a partir de las
    muestras de coeficientes (S, n, K). Se procesa por bloques de origenes para
    no materializar (S, n, G) completo.
    """
    S_, n, _ = muestras.shape
    G = Phi.shape[0]
    a_ = (1.0 - nivel) / 2.0
    li, ls = np.empty((n, G)), np.empty((n, G))
    for ini in range(0, n, bloque):
        sl = slice(ini, min(ini + bloque, n))
        curvas = mu[None, None, :] + muestras[:, sl, :] @ Phi.T     # (S, b, G)
        li[sl], ls[sl] = np.quantile(curvas, [a_, 1.0 - a_], axis=0)
    return li, ls
