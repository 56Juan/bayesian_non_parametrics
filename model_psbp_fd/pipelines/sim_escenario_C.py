"""
sim_escenario_C.py
==================
Familia C del anexo (`docs/01 Anexo.tex`, `ane_00_03`): metodos basados en los
COEFICIENTES de la representacion. Los tres algoritmos --C-1, C-2 y C-3--
comparten construccion y difieren solo en la configuracion, de modo que este
modulo los aloja a los tres, como `sim_series_clasicas` aloja a los A.

Que se simula
-------------
No se simula una curva y luego se la representa: se simulan DIRECTAMENTE los
coeficientes sobre un sistema ortonormal FIJO y conocido --la base de Fourier--
y la curva es su reconstruccion exacta,

    X_t(tau) = mu(tau) + sum_{j=1}^{J} a_tj phi_j(tau),   a_tj = sqrt(lambda_j) at_tj

con espectro geometrico lambda_j = rho^(j-1) y at_tj (la `a` tilde del anexo)
de media cero y varianza unitaria. Los coeficientes `a` SON los scores: no hay
FPCA que estimar ni base que elegir por GCV, de modo que toda discrepancia
entre prediccion y objetivo proviene de la dinamica y no de la representacion.
El objetivo de evaluacion al truncar en M es la reconstruccion truncada

    X_t^(M)(tau) = mu(tau) + sum_{j<=M} a_tj phi_j(tau),

que es una curva, y por eso el estudio sigue midiendo contra una curva (§6.5 de
CLAUDE.md) aunque la simulacion viva en los coeficientes.

La dinamica: interaccion entre VARIOS rezagos
---------------------------------------------
Sea j* la primera componente del par que interactua y

    x = (at_{t-1,j*}, ..., at_{t-p,j*})     los p rezagos del impulsor.

    at_{t,j*}   = phi' x + sigma_phi e                          (impulsor, AR(p))

    at_{t,j*+1} = c' x  +  b q(x)  +  sigma_c e                 (respuesta)

                          (u'x)^2 - v
                  q(x) = --------------  ,   v = u' R u
                            sqrt(2) v

y at_tj = e_tj para el resto de las componentes.

`q` es el indice cuadratico: el cuadrado de una combinacion lineal del pasado,
centrado y escalado a varianza unitaria. Desarrollado,

    (u'x)^2 = sum_l u_l^2 x_l^2  +  2 sum_{l<m} u_l u_m x_l x_m,

de modo que un solo vector `u` genera a la vez los terminos cuadraticos de cada
rezago y TODOS los productos cruzados entre rezagos distintos. Con u
concentrado en el rezago 1 se recupera la version de un rezago del anexo; con u
repartido sobre los rezagos 2..p la no linealidad depende del pasado remoto y
de la interaccion entre instantes, que es lo que esta familia existe para
medir. Como `q` esta normalizado, `u` solo importa por su DIRECCION.

Contabilidad de varianzas: es cerrada, no simulada
--------------------------------------------------
Con x gaussiano estandar de correlaciones R:

    Var((u'x)^2) = 2 v^2          =>  Var(q) = 1
    Cov(c'x, (u'x)^2) = 0                       (momento impar gaussiano)

    R2_oraculo = c' R c + b^2           varianza que explica la media condicional
    R2_lineal  = c' R c                 techo de CUALQUIER predictor lineal en x
    sigma_c^2  = 1 - R2_oraculo         despejada para que Var(at) = 1

La brecha `R2_oraculo - R2_lineal = b^2` es el diseno del escenario: es
exactamente lo que un FAR(p) correctamente especificado en su parte lineal no
puede alcanzar. Se verifica con `assert` contra la simulacion.

Los tres algoritmos
-------------------
    C-1  no linealidad distribuida en el pasado, dependencia lineal debil.
    C-2  misma no linealidad, dependencia lineal reforzada (impulsor y c').
    C-3  misma construccion trasladada a las componentes subordinadas (j* = 3).

Convenciones de tiempo y reproducibilidad
-----------------------------------------
Por replica se abre UN generador y se consume en este orden: las innovaciones
del par (impulsor y respuesta) sobre burn_in + T periodos, despues las
componentes ruido blanco, despues el ruido de observacion. Las curvas
retenidas son t = 1, ..., T. `sigma_obs = 0` por defecto: esta familia no
aplica esquema de observacion, porque el coeficiente ES el dato.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from scipy.linalg import solve_discrete_lyapunov

from .sim_comun import (
    ConfigObservacion,
    SalidaSimulacion,
    aplicar_ruido_observacion,
    diagnostico_comun,
    evaluar_media,
    grilla_regular,
    pesos_trapezoidales,
    semillas_replicas,
)
# La media es la misma de los algoritmos B y tiene UNA definicion; se importa
# en vez de repetirla para que las dos familias no puedan divergir.
from .sim_escenario_B1 import media_seno

__all__ = [
    "base_fourier",
    "gram_base",
    "ConfigEscenarioC",
    "config_C1",
    "config_C2",
    "config_C3",
    "radio_espectral",
    "correlaciones_impulsor",
    "contabilidad_varianzas",
    "generar_escenario_C",
    "curvas_truncadas",
    "resumen_escenario_C",
]


# ==========================================================================
# BASE DE FOURIER
# ==========================================================================

def base_fourier(tau: np.ndarray, J: int) -> np.ndarray:
    """
    Sistema ortonormal de Fourier sobre [0, 1], SIN el termino constante:

        phi_{2k-1}(tau) = sqrt(2) sin(2 pi k tau)
        phi_{2k}(tau)   = sqrt(2) cos(2 pi k tau),      k = 1, 2, ...

    El constante se omite porque el nivel lo aporta mu; si estuviera, el
    coeficiente a_t1 y la media serian la misma direccion y lambda_1 dejaria de
    ser interpretable como la varianza de una componente.

    Retorna (L, J). J debe ser par para que el sistema quede cerrado por pares
    seno/coseno y el espectro no favorezca arbitrariamente una fase.
    """
    tau = np.asarray(tau, dtype=float)
    if J < 2 or J % 2 != 0:
        raise ValueError(f"J={J}: debe ser un entero par >= 2.")
    Phi = np.empty((tau.size, J))
    for k in range(1, J // 2 + 1):
        Phi[:, 2 * k - 2] = np.sqrt(2.0) * np.sin(2.0 * np.pi * k * tau)
        Phi[:, 2 * k - 1] = np.sqrt(2.0) * np.cos(2.0 * np.pi * k * tau)
    return Phi


def gram_base(Phi: np.ndarray, tau: np.ndarray) -> np.ndarray:
    """Gram de la base en la cuadratura del estudio; debe dar la identidad."""
    w = pesos_trapezoidales(tau)
    return Phi.T @ (w[:, None] * Phi)


# ==========================================================================
# CONFIGURACION
# ==========================================================================

@dataclass
class ConfigEscenarioC(ConfigObservacion):
    """
    Parametros de la familia C.

    J          : terminos de la base de Fourier (par).
    rho        : razon del espectro geometrico lambda_j = rho^(j-1).
    j_estrella : primera componente del par que interactua, EN BASE 1.
    phi        : coeficientes del AR(p) del impulsor; su largo fija p.
    c          : coeficientes lineales de la respuesta sobre los p rezagos.
    u          : perfil de la no linealidad sobre los p rezagos. Solo importa
                 su direccion; u = (1, 0, ..., 0) recupera la version de un
                 rezago del anexo.
    b          : peso del indice cuadratico. b^2 ES la brecha no lineal.
    sigma_obs  : 0 por defecto. El coeficiente es el dato; no hay esquema de
                 observacion en esta familia.
    """

    J: int = 10
    rho: float = 0.5
    j_estrella: int = 1
    phi: tuple = (0.20, 0.12, 0.07, 0.04, 0.02)
    c: tuple = (0.15, 0.0, 0.0, 0.0, 0.0)
    u: tuple = (0.0, 1.0, 1.0, 1.0, 1.0)
    b: float = 0.50
    media_fn: Optional[Callable[[np.ndarray], np.ndarray]] = media_seno
    sigma_obs: float = 0.0
    prop_train_referencia: float = 0.70

    @property
    def p_lags(self) -> int:
        """Orden del pasado que interviene: el largo de phi."""
        return len(self.phi)

    def validar(self) -> None:
        super().validar()
        if self.J < 2 or self.J % 2 != 0:
            raise ValueError(f"J={self.J}: debe ser par y al menos 2.")
        if not 0.0 < self.rho < 1.0:
            raise ValueError("rho debe estar en (0, 1).")
        p = self.p_lags
        if p < 1:
            raise ValueError("phi no puede ser vacio.")
        if len(self.c) != p or len(self.u) != p:
            raise ValueError(
                f"c y u deben tener el largo de phi (p={p}); "
                f"recibidos {len(self.c)} y {len(self.u)}.")
        if not 1 <= self.j_estrella <= self.J - 1:
            raise ValueError(
                f"j_estrella={self.j_estrella}: el par (j*, j*+1) debe caber "
                f"en J={self.J}.")
        if np.allclose(self.u, 0.0):
            raise ValueError("u no puede ser el vector nulo.")
        if self.b < 0:
            raise ValueError("b no puede ser negativo.")
        radio = radio_espectral(self.phi)
        if radio >= 1.0:
            raise ValueError(
                f"radio espectral {radio:.3f} >= 1: el impulsor no es "
                "estacionario.")
        cont = contabilidad_varianzas(self)
        if cont["R2_oraculo"] >= 1.0:
            raise ValueError(
                f"R2_oraculo = {cont['R2_oraculo']:.3f} >= 1: no queda varianza "
                "de innovacion para la respuesta. Baja b o c.")


def config_C1(**kw) -> ConfigEscenarioC:
    """C-1: no linealidad distribuida en los rezagos 2..5, lineal debil."""
    base = dict(j_estrella=1,
                phi=(0.20, 0.12, 0.07, 0.04, 0.02),
                c=(0.15, 0.0, 0.0, 0.0, 0.0),
                u=(0.0, 1.0, 1.0, 1.0, 1.0),
                b=0.50)
    base.update(kw)
    return ConfigEscenarioC(**base)


def config_C2(**kw) -> ConfigEscenarioC:
    """C-2: misma no linealidad, dependencia lineal reforzada."""
    base = dict(j_estrella=1,
                phi=(0.45, 0.20, 0.10, 0.05, 0.03),
                c=(0.35, 0.20, 0.10, 0.0, 0.0),
                u=(0.0, 1.0, 1.0, 1.0, 1.0),
                b=0.50)
    base.update(kw)
    return ConfigEscenarioC(**base)


def config_C3(**kw) -> ConfigEscenarioC:
    """C-3: la construccion de C-1 trasladada a las componentes 3 y 4."""
    base = dict(j_estrella=3,
                phi=(0.20, 0.12, 0.07, 0.04, 0.02),
                c=(0.15, 0.0, 0.0, 0.0, 0.0),
                u=(0.0, 1.0, 1.0, 1.0, 1.0),
                b=0.50)
    base.update(kw)
    return ConfigEscenarioC(**base)


# ==========================================================================
# CONTABILIDAD DE VARIANZAS (CERRADA)
# ==========================================================================

def _companion(phi) -> np.ndarray:
    p = len(phi)
    F = np.zeros((p, p))
    F[0, :] = np.asarray(phi, dtype=float)
    if p > 1:
        F[1:, :-1] = np.eye(p - 1)
    return F


def radio_espectral(phi) -> float:
    """Radio espectral de la companion: < 1 es la estacionariedad del AR(p)."""
    return float(np.max(np.abs(np.linalg.eigvals(_companion(phi)))))


def correlaciones_impulsor(phi) -> tuple[np.ndarray, float]:
    """
    Correlaciones del impulsor y varianza estacionaria con innovacion unitaria.

    Retorna (R, g0): R[l, m] = corr(at_{t-1-l}, at_{t-1-m}) es la matriz de
    correlaciones del vector de rezagos x; g0 es Var(at) cuando la innovacion
    tiene varianza 1, de donde sale sigma_phi = 1/sqrt(g0) para que el impulsor
    quede con varianza unitaria y el espectro marginal sea lambda_j.
    """
    p = len(phi)
    F = _companion(phi)
    Q = np.zeros((p, p))
    Q[0, 0] = 1.0
    G = solve_discrete_lyapunov(F, Q)          # cov del estado, innovacion 1
    g0 = float(G[0, 0])
    r = [1.0] + [float(G[0, k]) / g0 for k in range(1, p)]
    return np.array([[r[abs(i - j)] for j in range(p)] for i in range(p)]), g0


def contabilidad_varianzas(cfg: ConfigEscenarioC) -> dict:
    """
    Las cifras que definen el escenario, en forma cerrada.

    R2_lineal es el techo de cualquier predictor lineal en los p rezagos --o
    sea, del FAR(p) correctamente especificado en su parte lineal--; R2_oraculo
    es el de la media condicional verdadera; la brecha es b^2 por construccion.
    """
    R, g0 = correlaciones_impulsor(cfg.phi)
    c = np.asarray(cfg.c, dtype=float)
    u = np.asarray(cfg.u, dtype=float)
    phi = np.asarray(cfg.phi, dtype=float)
    p = len(phi)
    r2_lin = float(c @ R @ c)
    r2_ora = r2_lin + float(cfg.b) ** 2

    # Correlacion CONTEMPORANEA del par. Cov(at_{t,j*}, at_{t,j*+1}) = c' r,
    # con r = (r_1, ..., r_p) la acf del impulsor: el indice cuadratico no
    # aporta (tercer momento gaussiano). No es cero salvo que c lo sea, de modo
    # que las componentes del par NO son incorreladas y la base de Fourier no
    # es exactamente la base de autofunciones del proceso. Es una cifra que hay
    # que declarar: con C-2 vale 0.43 y la rotacion es sustantiva; el estudio
    # retiene igual los coeficientes de Fourier, que es lo que el anexo define
    # como scores, pero la afirmacion "los scores son incorrelados" no vale.
    acf = list(R[0, :]) + [float(phi @ R[0, ::-1])]      # r_0 ... r_p
    r_vec = np.array(acf[1:p + 1])
    return {
        "R": R,
        "acf": np.array(acf),
        "corr_contemporanea_par": float(c @ r_vec),
        "g0": g0,
        "sigma_phi": float(1.0 / np.sqrt(g0)),
        "v_indice": float(u @ R @ u),
        "R2_lineal": r2_lin,
        "R2_oraculo": r2_ora,
        "brecha_no_lineal": float(cfg.b) ** 2,
        "sigma_c": float(np.sqrt(max(1.0 - r2_ora, 0.0))),
        "radio_espectral": radio_espectral(cfg.phi),
    }


# ==========================================================================
# SIMULACION
# ==========================================================================

def _simular_par(cfg: ConfigEscenarioC, rng: np.random.Generator,
                 cont: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simula el par (impulsor, respuesta) sobre burn_in + T periodos.

    Retorna (impulsor, respuesta, media_condicional), los tres recortados a los
    T periodos retenidos. La media condicional es el ORACULO y existe para todo
    t retenido porque el calentamiento cubre los p rezagos.
    """
    p = cfg.p_lags
    n = cfg.burn_in + cfg.T
    phi = np.asarray(cfg.phi, dtype=float)
    c = np.asarray(cfg.c, dtype=float)
    u = np.asarray(cfg.u, dtype=float)
    v = cont["v_indice"]

    e_d = rng.standard_normal(n)
    e_r = rng.standard_normal(n)

    d = np.zeros(n)                       # impulsor
    r = np.zeros(n)                       # respuesta
    m = np.zeros(n)                       # media condicional de la respuesta

    for t in range(n):
        if t < p:
            # Arranque: el pasado no existe todavia. Se extrae de la marginal
            # N(0,1) --que es la estacionaria-- y el calentamiento se encarga
            # del resto. Consume las mismas normales que un paso normal.
            d[t] = e_d[t]
            r[t] = e_r[t]
            continue
        x = d[t - p:t][::-1]              # (at_{t-1}, ..., at_{t-p})
        d[t] = float(phi @ x) + cont["sigma_phi"] * e_d[t]
        q = (float(u @ x) ** 2 - v) / (np.sqrt(2.0) * v)
        m[t] = float(c @ x) + cfg.b * q
        r[t] = m[t] + cont["sigma_c"] * e_r[t]

    corte = cfg.burn_in
    return d[corte:], r[corte:], m[corte:]


def generar_escenario_C(cfg: ConfigEscenarioC,
                        verificar: bool = True) -> SalidaSimulacion:
    """
    Genera las R replicas de un algoritmo de la familia C.

    `internos` lleva lo que define a esta familia: `A` (R, T, J) los
    coeficientes --que SON los scores--, `A_tilde` los mismos estandarizados,
    `oraculo` la media condicional de la respuesta y `Phi` la base evaluada en
    la grilla.
    """
    cfg.validar()
    cont = contabilidad_varianzas(cfg)

    tau = grilla_regular(cfg.L)
    mu = evaluar_media(cfg.media_fn, tau)
    Phi = base_fourier(tau, cfg.J)
    lam = cfg.rho ** np.arange(cfg.J)
    sqrt_lam = np.sqrt(lam)

    semillas, registro = semillas_replicas(cfg.seed, cfg.R)
    jd = cfg.j_estrella - 1                     # base-0
    jr = jd + 1

    A_t = np.empty((cfg.R, cfg.T, cfg.J))
    oraculo = np.empty((cfg.R, cfg.T))
    curvas = np.empty((cfg.R, cfg.T, cfg.L))
    obs = np.empty_like(curvas)

    for i, ss in enumerate(semillas):
        rng = np.random.default_rng(ss)
        d, r, m = _simular_par(cfg, rng, cont)
        At = rng.standard_normal((cfg.T, cfg.J))    # el resto: ruido blanco
        At[:, jd] = d
        At[:, jr] = r
        A_t[i] = At
        oraculo[i] = m
        curvas[i] = mu[None, :] + (At * sqrt_lam[None, :]) @ Phi.T
        obs[i] = aplicar_ruido_observacion(curvas[i], cfg.sigma_obs, rng)

    A = A_t * sqrt_lam[None, None, :]

    salida = SalidaSimulacion(
        observaciones=obs, curvas=curvas, grilla=tau, media=mu,
        semillas=registro, config=cfg,
        internos={"A": A, "A_tilde": A_t, "oraculo": oraculo, "Phi": Phi,
                  "lambda": lam, "j_estrella": cfg.j_estrella},
    )
    salida.diagnostico = resumen_escenario_C(salida, cont)
    if verificar:
        _verificar(salida)
    return salida


def curvas_truncadas(salida: SalidaSimulacion, M: int) -> np.ndarray:
    """
    Objetivo de evaluacion al retener M componentes:

        X_t^(M) = mu + sum_{j<=M} a_tj phi_j        (R, T, L)

    Es lo que se compara contra la prediccion, y es una curva: la familia C
    simula coeficientes, pero el error se sigue midiendo en L^2 sobre el
    dominio, con la misma definicion que el resto del estudio.
    """
    A = salida.internos["A"][:, :, :M]
    Phi = salida.internos["Phi"][:, :M]
    return salida.media[None, None, :] + A @ Phi.T


# ==========================================================================
# DIAGNOSTICO
# ==========================================================================

def resumen_escenario_C(salida: SalidaSimulacion,
                        cont: Optional[dict] = None) -> dict:
    """
    Diagnostico comun mas lo propio de la familia: la contabilidad cerrada de
    varianzas contra su contraparte empirica, la ortonormalidad de la base y la
    varianza acumulada del espectro, que decide cuanto `M` hace falta para VER
    el par que interactua.
    """
    cfg: ConfigEscenarioC = salida.config
    cont = cont or contabilidad_varianzas(cfg)
    d = diagnostico_comun(salida)

    At = salida.internos["A_tilde"]
    m = salida.internos["oraculo"]
    jd, jr = cfg.j_estrella - 1, cfg.j_estrella
    resp = At[:, :, jr]

    # R^2 empirico del oraculo y del mejor predictor LINEAL en los p rezagos,
    # ajustado sobre la propia replica: es el techo del FAR(p), medido.
    p = cfg.p_lags
    r2_ora_emp, r2_lin_emp = [], []
    for i in range(cfg.R):
        y = resp[i, p:]
        X = np.column_stack([At[i, p - l:cfg.T - l, jd] for l in range(1, p + 1)])
        X1 = np.column_stack([np.ones(len(y)), X])
        r2_ora_emp.append(1.0 - np.var(y - m[i, p:]) / np.var(y))
        beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
        r2_lin_emp.append(1.0 - np.var(y - X1 @ beta) / np.var(y))

    lam = salida.internos["lambda"]
    var_cum = np.cumsum(lam) / lam.sum()
    G = gram_base(salida.internos["Phi"], salida.grilla)
    peso_par = float(lam[jd] + lam[jr]) / float(lam.sum())

    d.update({
        "p_lags": p,
        "j_estrella": cfg.j_estrella,
        "radio_espectral": cont["radio_espectral"],
        "R2_lineal_teorico": cont["R2_lineal"],
        "R2_oraculo_teorico": cont["R2_oraculo"],
        "brecha_no_lineal": cont["brecha_no_lineal"],
        "R2_lineal_empirico": float(np.mean(r2_lin_emp)),
        "R2_oraculo_empirico": float(np.mean(r2_ora_emp)),
        "sigma_phi": cont["sigma_phi"],
        "sigma_c": cont["sigma_c"],
        "acf_impulsor": cont["acf"],
        "corr_contemporanea_par": cont["corr_contemporanea_par"],
        "corr_contemporanea_par_empirica": float(
            np.mean([np.corrcoef(At[i, :, jd], resp[i])[0, 1]
                     for i in range(cfg.R)])),
        "peso_espectral_par": peso_par,
        "var_acum": var_cum,
        "M_minimo": int(cfg.j_estrella + 1),
        "error_ortonormalidad": float(np.abs(G - np.eye(cfg.J)).max()),
        "var_impulsor": float(np.var(At[:, :, jd])),
        "var_respuesta": float(np.var(resp)),
    })
    return d


def _verificar(salida: SalidaSimulacion) -> None:
    """Las igualdades que el diseno promete. Si alguna falla, el escenario no
    es el que declara el anexo y no tiene sentido entrenar contra el.

    Las tolerancias de las cantidades EMPIRICAS son anchas a proposito. El
    indice cuadratico tiene colas de chi-cuadrado y el impulsor es persistente
    (acf 0.70 a rezago 1 en C-2), de modo que el numero efectivo de
    observaciones de q es una fraccion de T: medido sobre cinco semillas con
    T = 4000, la varianza realizada de la respuesta se mueve entre 0.86 y 1.09
    alrededor de su valor exacto 1. Apretar estas tolerancias haria fallar
    semillas correctas; las cifras TEORICAS, que son las que definen el
    escenario, son exactas y no se verifican contra la simulacion sino que la
    gobiernan."""
    d = salida.diagnostico
    assert d["error_ortonormalidad"] < 1e-8, (
        f"la base de Fourier no es ortonormal en la cuadratura: "
        f"{d['error_ortonormalidad']:.2e}")
    for nombre in ("var_impulsor", "var_respuesta"):
        assert abs(d[nombre] - 1.0) < 0.25, (
            f"{nombre} = {d[nombre]:.3f}: la contabilidad de varianzas no "
            "cierra y el espectro marginal no es lambda_j.")
    for teo, emp in (("R2_lineal_teorico", "R2_lineal_empirico"),
                     ("R2_oraculo_teorico", "R2_oraculo_empirico")):
        assert abs(d[teo] - d[emp]) < 0.20, (
            f"{teo} = {d[teo]:.3f} pero {emp} = {d[emp]:.3f}")
    assert d["R2_oraculo_teorico"] > d["R2_lineal_teorico"] + 0.05, (
        "sin brecha no lineal el escenario no distingue al modelo del FAR(p).")
