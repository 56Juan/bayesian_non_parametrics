r"""
mezcla_gp_funcional.py — Mezcla probit stick-breaking con atomos GP (experimento)
=================================================================================

EXPERIMENTO sobre los datos de la corrida 107 (C-1), fuera del pipeline
canonico. El ciclo es el mismo Python -> MATLAB -> Python del estudio:

    exp107_01_datos.ipynb     arma el contrato (este modulo lo escribe)
    mezcla_gp_iteracion.m     entrena: una cadena por job, `mezcla_gp_train.m`
    exp107_04_evaluacion.ipynb lee las trazas .mat (este modulo) y evalua

El muestreo ocurre SOLO en MATLAB. Este modulo no muestrea: define la base, el
diseno, las covarianzas proyectadas, el contrato en disco, la lectura de las
trazas y la predictiva a partir de ellas.

El modelo
---------
Sea c_t in R^K el vector de coeficientes de la curva suavizada en una base
ORTONORMAL en L^2 (los coeficientes B-spline blanqueados con la Cholesky de la
Gram, W = L L', c = L'(theta - theta_bar)), y x_t = (1, z_t) con z_t los
rezagos estandarizados de los scores FPCA. Hay UNA asignacion por curva:

    c_t | S_t = h, x_t  ~  N_K( A_h' x_t , C_h )
    Pr(S_t = h | x_t)   =  pi_h(x_t) = v_h prod_{l<h}(1 - v_l),  v_h = Phi(eta_h)
    eta_h(x)            =  alpha_h + omega_h' z

    covarianza "gp":  C_h = sigma_h^2 S(ell_h) + s_h^2 I,   S(ell) = Phi_o' W_q K_ell W_q Phi_o
    covarianza "iw":  C_h ~ Inv-Wishart(nu0, Lam0)          (control sin estructura)
    covarianza "iw_gp_comun":  C_h = C,  C ~ IW(nu0, (nu0-K-1)[sigma^2 S(ell) + s^2 I])
                      El error es COMUN a los regimenes y el GP es el centro del
                      prior, no una restriccion (Yang, Zhu, Choi y Cox 2016). En
                      C-1 la innovacion no depende de Z_t y no es un GP
                      estacionario (KL 7.4 nats/curva contra el mejor Matern):
                      con C_h propia y fijada al GP la mezcla crea atomos para
                      parchar la covarianza (Cai, Campbell y Broderick 2021).

Llevado a la curva, X_t = mu + Phi_o(tau) c_t, cada atomo es un proceso
gaussiano con media FAR propia del regimen y covarianza Phi_o C_h Phi_o'.

Priors: A_h | C_h ~ MN(M0, V0, C_h), con dos opciones (`prior_A`):
    "g"        : M0 = 0, V0 = (n/g)(X'X)^{-1}  (g-prior matricial).
    "encogido" : M0 = B_MCO del bloque train (comun a todos los atomos),
                 V0 = diag(v_intercepto, v_pendiente, ...): intercepto libre,
                 pendientes encogidas hacia el ajuste comun.
Con "g" y covarianza "iw" la mezcla colapsa a un atomo: la penalizacion de
Occam de A_h y C_h libres supera lo que ganan los regimenes (verosimilitud
marginal exacta, 2026-09-28: Z verdadera -353 contra un atomo). Con
"encogido" deja de colapsar. `lam0` fija la escala de la IW: "diag" es
diag(var(Y))/2 y "pooled" centra E[C] en la covarianza residual del MCO comun.
(ell, sigma, s) uniforme sobre una grilla; alpha_h ~ N(mu_a, 1), con mu_a ~ N(mu_mu, 1/tau_mu)
("normal") o con el prior DISPERSO (`prior_mu_alpha="disperso"`): mu_a inducido por
una concentracion DP a ~ G(conc_a, conc_b) igualando E[v | mu_a] = 1/(1+a); con
G(1, 20) es el DPM disperso de Fruhwirth-Schnatter y Malsiner-Walli (2019), que
vacia los atomos sobrantes. `n_mov` > 0 agrega los movimientos de etiqueta de
Papaspiliopoulos y Roberts (2008), como en Hastie, Liverani y Richardson (2015);
el detalle y las razones de aceptacion estan en `mezcla_gp_train.m`.
omega_h ~ N(0, s_omega^2/p I). El Gibbs es exacto: la grilla de (ell, sigma, s)
se muestrea con A_h integrado porque C = U diag(d) U' con U fijo por ell
(ver el encabezado de `mezcla_gp_train.m`).

Convenciones del contrato
-------------------------
`hyperparameters.json` en `out_artefact` es la unica fuente de verdad para
MATLAB, como en el resto del estudio. Los nombres de archivo se deciden en
`ARCHIVOS_MEZCLA` y en ningun otro lado.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Sequence

import numpy as np
from scipy.io import loadmat
from scipy.special import log_ndtr

from model_psbp_fd.utils.quadrature import pesos_trapezoidales

__all__ = [
    "ARCHIVOS_MEZCLA",
    "ConfigMezclaGP",
    "Diseno",
    "nucleo",
    "base_ortonormal",
    "covarianzas_proyectadas",
    "disenar",
    "log_pesos",
    "guardar_contrato",
    "cargar_contrato",
    "cargar_trazas",
    "media_condicional",
    "muestras_predictivas",
    "banda_funcional",
]

#: Nombres de archivo del contrato y de las trazas. Unica definicion; el `.m`
#: los lee desde `hyperparameters.json["archivos"]`, no los escribe a mano.
ARCHIVOS_MEZCLA = {
    "Y_train":  "mezcla_Y_train.csv",      # (n_train, K)  -> MATLAB
    "X_train":  "mezcla_X_train.csv",      # (n_train, q)  -> MATLAB
    "S_lam":    "mezcla_S_lam.csv",        # (nE, K)       -> MATLAB
    "S_U":      "mezcla_S_U.csv",          # (nE*K, K)     -> MATLAB, bloques U_a apilados
    "Y":        "mezcla_Y.csv",            # (n, K)  serie completa, solo Python
    "X":        "mezcla_X.csv",            # (n, q)  serie completa, solo Python
    "t_idx":    "mezcla_t_idx.csv",        # (n,)    indice base-0 de la respuesta
    "Phi_o":    "mezcla_base_ortonormal.csv",  # (G, K)
    "mu":       "mezcla_mu_grilla.csv",    # (G,)
    "M0":       "mezcla_A_M0.csv",         # (q, K)  -> MATLAB, media a priori de A_h
    "Lam0":     "mezcla_Lam0.csv",         # (K, K)  -> MATLAB, escala de la IW
    "traza":    "traza_mezcla_gp_cadena{cadena:02d}.mat",
}

NUCLEOS = ("matern32", "exponencial", "gaussiano")
COVARIANZAS = ("gp", "iw", "iw_gp_comun")
PRIORES_A = ("g", "encogido")
LAM0 = ("diag", "pooled")
PRIORES_MU_ALPHA = ("normal", "disperso")
_SIG_REL = tuple(float(v) for v in np.round(np.geomspace(0.02, 2.0, 14), 6))


# ==========================================================================
# CONFIGURACION
# ==========================================================================

@dataclass
class ConfigMezclaGP:
    """
    N          : truncamiento del stick-breaking.
    nsim, burn, thin : iteraciones; se guardan las de it > burn cada `thin`.
    covarianza : "gp", "iw" o "iw_gp_comun" (C comun, IW centrada en el GP).
    nucleo     : nucleo del GP del atomo, uno de NUCLEOS.
    ells       : grilla de escalas del GP, en unidades del dominio [0, 1].
    sig_rel    : grilla de sigma, RELATIVA a la sd total de la respuesta en train.
    nug_rel    : grilla de la pepita s, relativa a la misma sd total.
    prior_A    : "g" (M0 = 0, g-prior) o "encogido" (M0 = MCO comun, V0 diagonal).
    g          : g del g-prior matricial (V0 = (n/g)(X'X)^{-1}); solo con prior_A="g".
    v_intercepto, v_pendiente : diagonal de V0 con prior_A="encogido".
    s_omega    : sd a priori del gating en unidades probit (se divide por sqrt(p)).
    nu0_extra  : "iw" / "iw_gp_comun": nu0 = K + nu0_extra (> 1 para que E[C] exista).
    lam0       : solo "iw": "diag" -> E[C] = diag(var(Y))/2; "pooled" -> E[C] =
                 covarianza residual del MCO comun en train.
    n_inicial  : atomos ocupados al arrancar.
    prior_mu_alpha : "normal" (mu_a ~ N(mu_mu, 1/tau_mu)) o "disperso" (a ~ G(conc_a, conc_b)).
    conc_a, conc_b : forma y tasa de la concentracion con prior_mu_alpha="disperso".
    n_mov      : intentos por iteracion de cada movimiento de etiqueta (0 = sin movimientos).
    """

    N: int = 30
    nsim: int = 3000
    burn: int = 1000
    thin: int = 4
    covarianza: str = "gp"
    nucleo: str = "matern32"
    ells: tuple = (0.03, 0.05, 0.08, 0.12, 0.18, 0.25, 0.35, 0.5)
    sig_rel: tuple = _SIG_REL
    nug_rel: tuple = (0.002, 0.005, 0.01, 0.02, 0.05)
    prior_A: str = "g"
    g: float = 1.0
    v_intercepto: float = 10.0
    v_pendiente: float = 0.01
    s_omega: float = 0.8
    mu_mu: float = 0.0
    tau_mu: float = 1.0
    nu0_extra: float = 3.0
    lam0: str = "diag"
    n_inicial: int = 5
    prior_mu_alpha: str = "normal"
    conc_a: float = 1.0
    conc_b: float = 20.0
    n_mov: int = 0

    def validar(self) -> None:
        if self.covarianza not in COVARIANZAS:
            raise ValueError(f"covarianza={self.covarianza!r}; use una de {COVARIANZAS}.")
        if self.prior_A not in PRIORES_A:
            raise ValueError(f"prior_A={self.prior_A!r}; use uno de {PRIORES_A}.")
        if self.lam0 not in LAM0:
            raise ValueError(f"lam0={self.lam0!r}; use uno de {LAM0}.")
        if self.covarianza == "iw_gp_comun" and self.nu0_extra <= 1:
            raise ValueError("iw_gp_comun requiere nu0_extra > 1 (E[C] = centro GP).")
        if self.prior_mu_alpha not in PRIORES_MU_ALPHA:
            raise ValueError(f"prior_mu_alpha={self.prior_mu_alpha!r}; use uno de {PRIORES_MU_ALPHA}.")
        if min(self.conc_a, self.conc_b) <= 0 or self.n_mov < 0:
            raise ValueError("conc_a, conc_b > 0 y n_mov >= 0.")
        if min(self.v_intercepto, self.v_pendiente) <= 0:
            raise ValueError("v_intercepto y v_pendiente deben ser > 0.")
        if self.nucleo not in NUCLEOS:
            raise ValueError(f"nucleo={self.nucleo!r}; use uno de {NUCLEOS}.")
        if not 0 <= self.burn < self.nsim:
            raise ValueError("burn debe estar en [0, nsim).")
        if self.N < 2 or self.thin < 1:
            raise ValueError("N >= 2 y thin >= 1.")


@dataclass
class Diseno:
    """Respuesta y covariables alineadas por origen.

    Y     : (n, K) coeficientes de la curva respuesta.
    X     : (n, 1 + p) intercepto + rezagos ESTANDARIZADOS con el bloque train.
    t_idx : (n,) indice base-0 de la curva respuesta en la serie completa.
    """

    Y: np.ndarray
    X: np.ndarray
    t_idx: np.ndarray
    n_lags: int
    nombres: List[str]
    T0: int

    @property
    def train(self) -> np.ndarray:
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


def _verificar_ortonormal(Phi: np.ndarray, tau: np.ndarray, tol: float = 1e-8) -> np.ndarray:
    w = pesos_trapezoidales(tau)
    err = float(np.abs(Phi.T @ (w[:, None] * Phi) - np.eye(Phi.shape[1])).max())
    if err > tol:
        raise ValueError(f"la base no es ortonormal en la cuadratura (error {err:.2e}).")
    return w


def base_ortonormal(Phi: np.ndarray, W: np.ndarray, tau: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Blanquea la base con la Cholesky de su Gram: Phi_o = Phi L^{-T}, W = L L'.

    Retorna (Phi_o, L). Con c = L' theta, Phi theta = Phi_o c y el producto
    euclideo de c ES el producto L^2 exacto (el mismo blanqueo del FAR del _05).
    """
    L = np.linalg.cholesky(W)
    Phi_o = np.linalg.solve(L, Phi.T).T
    _verificar_ortonormal(Phi_o, tau)
    return Phi_o, L


def covarianzas_proyectadas(Phi_o: np.ndarray, tau: np.ndarray,
                            ells: Sequence[float], tipo: str) -> list:
    """
    S(ell) = Phi_o' W_q K_ell W_q Phi_o por cada ell de la grilla, descompuesta
    como S = U diag(lam) U'. Es la covarianza de los coeficientes de un GP de
    varianza 1 proyectado sobre la base. Se calcula aqui, UNA vez, y MATLAB la
    lee del contrato: no hay una segunda definicion en el `.m`.
    """
    w = _verificar_ortonormal(Phi_o, tau)
    Pw = Phi_o * w[:, None]
    salida = []
    for ell in ells:
        S = Pw.T @ nucleo(tau, float(ell), tipo) @ Pw
        lam, U = np.linalg.eigh(0.5 * (S + S.T))
        salida.append((np.clip(lam, 0.0, None), U))
    return salida


# ==========================================================================
# DISENO
# ==========================================================================

def disenar(Y_full: np.ndarray, Z_full: np.ndarray, n_lags: int, T0: int,
            nombres_z: Sequence[str]) -> Diseno:
    """
    Arma (Y, X) sobre los origenes t = n_lags, ..., T-1.

    Y_full : (T, K) respuesta en cada t.
    Z_full : (T, m) serie cuyos rezagos 1..n_lags son las covariables, en orden
             lag-mayor (z_{1,t-1}, ..., z_{m,t-1}, z_{1,t-2}, ...), como `Z` en
             el `_05`. Se estandarizan con las filas cuya respuesta cae en train.
    """
    Y_full = np.asarray(Y_full, dtype=float)
    Z_full = np.atleast_2d(np.asarray(Z_full, dtype=float))
    T = Y_full.shape[0]
    if Z_full.shape[0] != T:
        raise ValueError(f"Y tiene {T} filas y Z {Z_full.shape[0]}.")
    t_idx = np.arange(n_lags, T)
    cols, nombres = [], []
    for l in range(1, n_lags + 1):
        for j in range(Z_full.shape[1]):
            cols.append(Z_full[t_idx - l, j])
            nombres.append(f"{nombres_z[j]}_lag{l}")
    Z = np.column_stack(cols)
    tr = t_idx < T0
    Zs = (Z - Z[tr].mean(axis=0)) / Z[tr].std(axis=0)
    X = np.column_stack([np.ones(t_idx.size), Zs])
    return Diseno(Y=Y_full[t_idx], X=X, t_idx=t_idx, n_lags=int(n_lags),
                  nombres=nombres, T0=int(T0))


# ==========================================================================
# PESOS
# ==========================================================================

def log_pesos(eta: np.ndarray) -> np.ndarray:
    """log pi_h a partir de eta (n, N-1), estable en las colas. Retorna (n, N).
    Gemelo de `log_pesos` en `mezcla_gp_train.m`."""
    lv, l1v = log_ndtr(eta), log_ndtr(-eta)
    n = eta.shape[0]
    acum = np.concatenate([np.zeros((n, 1)), np.cumsum(l1v, axis=1)], axis=1)
    return np.concatenate([lv, np.zeros((n, 1))], axis=1) + acum


# ==========================================================================
# CONTRATO PYTHON -> MATLAB
# ==========================================================================

def _savetxt(path: Path, A) -> None:
    np.savetxt(path, np.atleast_2d(np.asarray(A, dtype=float)), delimiter=",", fmt="%.17g")


def guardar_contrato(paths: dict, dis: Diseno, S_eig: list, Phi_o: np.ndarray,
                     mu: np.ndarray, cfg: ConfigMezclaGP, n_cadenas: int,
                     seed_base: int, extra: dict | None = None) -> dict:
    """
    Escribe todo lo que MATLAB necesita y lo que la evaluacion relee.

    MATLAB recibe SOLO el bloque train (`Y_train`, `X_train`) y las
    covarianzas proyectadas. Las grillas de sigma y s se escriben ya escaladas
    a la respuesta, para que el `.m` no repita ninguna cuenta.
    """
    cfg.validar()
    f, a = Path(paths["functional"]), Path(paths["out_artefact"])
    tr = dis.train
    Ytr = dis.Y[tr]
    n, K = Ytr.shape
    q = dis.X.shape[1]
    lam_E = np.stack([le for le, _ in S_eig])
    U_E = np.stack([ue for _, ue in S_eig])
    sd_tot = float(np.sqrt(np.trace(np.atleast_2d(np.cov(Ytr.T)))))
    nu0 = K + cfg.nu0_extra

    # prior de A_h y escala de la IW: solo con el bloque train
    Xtr = dis.X[tr]
    B_mco = np.linalg.lstsq(Xtr, Ytr, rcond=None)[0]
    if cfg.prior_A == "encogido":
        M0, V0_diag = B_mco, [cfg.v_intercepto] + [cfg.v_pendiente] * (q - 1)
    else:
        M0, V0_diag = np.zeros((q, K)), None
    if cfg.lam0 == "pooled":
        Lam0 = (nu0 - K - 1.0) * np.atleast_2d(np.cov((Ytr - Xtr @ B_mco).T))
    else:
        Lam0 = np.diag((nu0 - K - 1.0) * Ytr.var(axis=0) * 0.5)

    _savetxt(f / ARCHIVOS_MEZCLA["Y_train"], Ytr)
    _savetxt(f / ARCHIVOS_MEZCLA["X_train"], dis.X[tr])
    _savetxt(f / ARCHIVOS_MEZCLA["S_lam"], lam_E)
    _savetxt(f / ARCHIVOS_MEZCLA["S_U"], U_E.reshape(-1, K))
    _savetxt(f / ARCHIVOS_MEZCLA["Y"], dis.Y)
    _savetxt(f / ARCHIVOS_MEZCLA["X"], dis.X)
    _savetxt(f / ARCHIVOS_MEZCLA["t_idx"], dis.t_idx[:, None])
    _savetxt(f / ARCHIVOS_MEZCLA["Phi_o"], Phi_o)
    _savetxt(f / ARCHIVOS_MEZCLA["mu"], np.asarray(mu)[:, None])
    _savetxt(f / ARCHIVOS_MEZCLA["M0"], M0)
    _savetxt(f / ARCHIVOS_MEZCLA["Lam0"], Lam0)

    hp = {
        # claves que `cargar_hiperparametros` exige en todo el estudio
        "n_iter": int(n_cadenas),
        "mcmc_config": {"nsim": cfg.nsim, "burn": cfg.burn, "thin": cfg.thin,
                        "N": cfg.N, "n_inicial": cfg.n_inicial, "n_mov": cfg.n_mov},
        "seed_base": int(seed_base),
        "seed_scheme": "seed_base + chain*9973",
        "scores_scale": "coeficientes_blanqueados_L2",
        "hyperparams_list": [],
        # el modelo
        "modelo": {
            "covarianza": cfg.covarianza, "nucleo": cfg.nucleo,
            "ells": list(map(float, cfg.ells)),
            "SIG": list(map(float, np.asarray(cfg.sig_rel) * sd_tot)),
            "NUG": list(map(float, np.asarray(cfg.nug_rel) * sd_tot)),
            "sd_total": sd_tot,
            "prior_A": cfg.prior_A, "g": cfg.g, "V0_diag": V0_diag,
            "s_omega": cfg.s_omega, "mu_mu": cfg.mu_mu, "tau_mu": cfg.tau_mu,
            "prior_mu_alpha": cfg.prior_mu_alpha, "conc_a": cfg.conc_a, "conc_b": cfg.conc_b,
            "nu0": nu0, "lam0": cfg.lam0,
        },
        "dims": {"n_train": int(n), "K": int(K), "q": int(q), "p": int(q - 1),
                 "nE": int(lam_E.shape[0]), "n_total": int(dis.Y.shape[0])},
        "diseno": {"n_lags": dis.n_lags, "covariables": dis.nombres, "T0": dis.T0,
                   "T0_filas": dis.T0_ev},
        "archivos": ARCHIVOS_MEZCLA,
        "config_python": asdict(cfg),
    }
    if extra:
        hp.update(extra)
    with open(a / "hyperparameters.json", "w", encoding="utf-8") as fh:
        json.dump(hp, fh, indent=2, ensure_ascii=False)
    return hp


def cargar_contrato(paths: dict) -> dict:
    """Relee el contrato: hiperparametros + matrices de la serie completa."""
    f, a = Path(paths["functional"]), Path(paths["out_artefact"])
    with open(a / "hyperparameters.json", encoding="utf-8") as fh:
        hp = json.load(fh)
    K, nE = hp["dims"]["K"], hp["dims"]["nE"]
    leer = lambda k: np.loadtxt(f / ARCHIVOS_MEZCLA[k], delimiter=",", ndmin=2)
    return {
        "hp": hp,
        "Y": leer("Y"), "X": leer("X"),
        "t_idx": leer("t_idx").ravel().astype(int),
        "Phi_o": leer("Phi_o"), "mu": leer("mu").ravel(),
        "lam_E": leer("S_lam"), "U_E": leer("S_U").reshape(nE, K, K),
    }


# ==========================================================================
# TRAZAS MATLAB -> PYTHON
# ==========================================================================

def cargar_trazas(paths: dict, contrato: dict) -> list:
    """
    Lee las trazas `.mat` de todas las cadenas del contrato y las deja en la
    forma que esperan `media_condicional` y `muestras_predictivas`.

    MATLAB elimina las dimensiones singleton FINALES al guardar (CLAUDE.md
    §6.7): con p = 1 `omega` llega 2D. Por eso cada arreglo se reforma con las
    dimensiones del contrato en vez de confiar en `.shape`.
    """
    hp = contrato["hp"]
    d = hp["dims"]
    N = hp["mcmc_config"]["N"]
    K, q, p, n = d["K"], d["q"], d["p"], d["n_train"]
    trazas = []
    for c in range(1, int(hp["n_iter"]) + 1):
        ruta = Path(paths["out_artefact"]) / ARCHIVOS_MEZCLA["traza"].format(cadena=c)
        if not ruta.exists():
            raise FileNotFoundError(f"No se encontro {ruta}. Corra mezcla_gp_iteracion.m.")
        m = loadmat(ruta, squeeze_me=False)
        nk = int(np.asarray(m["alpha"]).shape[0])
        tr = {
            "A": np.asarray(m["A"], dtype=np.float32).reshape(nk, N, q, K),
            "idx_cov": np.asarray(m["idx_cov"]).reshape(nk, N, 3).astype(int) - 1,  # base-0
            "alpha": np.asarray(m["alpha"], dtype=float).reshape(nk, N - 1),
            "omega": np.asarray(m["omega"], dtype=float).reshape(nk, N - 1, p),
            "mu_a": np.asarray(m["mu_a"], dtype=float).ravel(),
            "S": np.asarray(m["S"]).reshape(nk, n).astype(int) - 1,                  # base-0
            "loglik": np.asarray(m["loglik"], dtype=float).ravel(),
            "n_ocupados": np.asarray(m["n_ocupados"]).ravel(),
            "masa_max": np.asarray(m["masa_max"], dtype=float).ravel(),
            "ms_por_iter": float(np.asarray(m["ms_por_iter"]).ravel()[0]),
            "ms_por_paso": np.asarray(m["ms_por_paso"], dtype=float).ravel(),
            "acept_mov": (np.asarray(m["acept_mov"], dtype=float).ravel() if "acept_mov" in m
                          else np.full(3, np.nan)),
            "seed": int(np.asarray(m["seed"]).ravel()[0]),
            "cadena": c,
            "covarianza": hp["modelo"]["covarianza"],
            "lam_E": contrato["lam_E"], "U_E": contrato["U_E"],
            "SIG": np.asarray(hp["modelo"]["SIG"]), "NUG": np.asarray(hp["modelo"]["NUG"]),
        }
        if tr["covarianza"] == "iw":
            tr["C"] = np.asarray(m["C"], dtype=float).reshape(nk, N, K, K)
        elif tr["covarianza"] == "iw_gp_comun":
            tr["C_comun"] = np.asarray(m["C"], dtype=float).reshape(nk, K, K)
            tr["idx_hyp"] = np.asarray(m["idx_hyp"]).reshape(nk, 3).astype(int) - 1  # base-0
        trazas.append(tr)
    return trazas


# ==========================================================================
# PREDICTIVA
# ==========================================================================

def _factores_draw(traza: dict, s: int) -> tuple[np.ndarray, np.ndarray]:
    """(U, d) de los N atomos en la extraccion s: C_h = U_h diag(d_h) U_h'."""
    if traza["covarianza"] == "gp":
        a, b, c = (traza["idx_cov"][s, :, i] for i in range(3))
        U = traza["U_E"][a]
        d = traza["SIG"][b][:, None] ** 2 * traza["lam_E"][a] + traza["NUG"][c][:, None] ** 2
        return U, d
    if traza["covarianza"] == "iw_gp_comun":
        d, U = np.linalg.eigh(traza["C_comun"][s])
        N, K = traza["A"].shape[1], U.shape[0]
        return np.broadcast_to(U, (N, K, K)), np.broadcast_to(np.clip(d, 1e-12, None), (N, K))
    d, U = np.linalg.eigh(traza["C"][s])
    return U, np.clip(d, 1e-12, None)


def _pi(traza: dict, s: int, X: np.ndarray) -> np.ndarray:
    return np.exp(log_pesos(traza["alpha"][s][None, :] + X[:, 1:] @ traza["omega"][s].T))


def media_condicional(traza: dict, X: np.ndarray) -> np.ndarray:
    """
    E[c_t | x_t] por extraccion: sum_h pi_h(x_t) A_h' x_t. Retorna (S, n, K).
    Invariante a la permutacion de etiquetas: sirve para R-hat.
    """
    X = np.asarray(X, dtype=float)
    S_ = traza["A"].shape[0]
    out = np.empty((S_, X.shape[0], traza["A"].shape[-1]))
    for s in range(S_):
        medias = np.einsum("nq,hqk->nhk", X, traza["A"][s].astype(float))
        out[s] = np.einsum("nh,nhk->nk", _pi(traza, s, X), medias)
    return out


def muestras_predictivas(traza: dict, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Una extraccion de la predictiva por traza guardada: (S, n, K)."""
    X = np.asarray(X, dtype=float)
    S_, n = traza["A"].shape[0], X.shape[0]
    K = traza["A"].shape[-1]
    out = np.empty((S_, n, K))
    for s in range(S_):
        pi = _pi(traza, s, X)
        h = (np.cumsum(pi, axis=1) > rng.random((n, 1))).argmax(axis=1)
        media = np.einsum("nq,nqk->nk", X, traza["A"][s].astype(float)[h])
        U, d = _factores_draw(traza, s)
        z = rng.standard_normal((n, K)) * np.sqrt(d[h])
        out[s] = media + np.einsum("nkj,nj->nk", U[h], z)
    return out


def banda_funcional(muestras: np.ndarray, Phi_o: np.ndarray, mu: np.ndarray,
                    nivel: float = 0.95, bloque: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """Banda puntual sobre la curva mu + Phi_o c, a partir de (S, n, K), por
    bloques de origenes para no materializar (S, n, G)."""
    S_, n, _ = muestras.shape
    G = Phi_o.shape[0]
    a_ = (1.0 - nivel) / 2.0
    li, ls = np.empty((n, G)), np.empty((n, G))
    for ini in range(0, n, bloque):
        sl = slice(ini, min(ini + bloque, n))
        curvas = mu[None, None, :] + muestras[:, sl, :] @ Phi_o.T
        li[sl], ls[sl] = np.quantile(curvas, [a_, 1.0 - a_], axis=0)
    return li, ls
