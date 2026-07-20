"""
sim_escenario_1.py
==================
Escenario 1: proceso autorregresivo funcional lineal, gaussiano y homogeneo en
el tiempo (Algoritmo 1 del anexo).

Modelo generador
----------------
    X_t(tau) = mu(tau) + int_0^1 psi(tau, s) (X_{t-1}(s) - mu(s)) ds + eps_t(tau)

    psi(tau, s) = c exp{ -(tau - s)^2 / (2 gamma^2) }

    Cov(eps_t(tau), eps_t(s)) = sigma_eps^2 exp{ -(tau - s)^2 / (2 ell^2) }

La constante c se calibra de modo que ||Psi||_HS alcance un valor prefijado
menor que uno, condicion suficiente para la existencia de una solucion
estacionaria (Bosq, 2000).

Bajo este generador la ley condicional de la curva dada su historia es
gaussiana, unimodal y homogenea en el tiempo, de modo que los metodos
funcionales lineales estan correctamente especificados. El escenario actua en
consecuencia como control del estudio.

El esquema de observacion, la cuadratura del operador, la innovacion funcional
y el control de calidad transversal provienen de `sim_comun.py`; este modulo
aporta unicamente la dinamica del algoritmo y sus verificaciones especificas.

Uso tipico desde un notebook
----------------------------
    from model_psbp_fd.simulations import (
        ConfigEscenario1, generar_escenario_1, guardar_escenario
    )

    cfg = ConfigEscenario1(L=48, T=200, burn_in=200, R=50, seed=20260719,
                           gamma=0.3, hs_norm=0.7, sigma_eps=1.0, ell=0.2,
                           sigma_obs=0.5)
    salida = generar_escenario_1(cfg)
    salida.diagnostico            # control de calidad del generador
    X = salida.observaciones      # (R, T, L) -> insumo del pipeline
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .sim_comun import (
    ConfigObservacion,
    SalidaSimulacion,
    grilla_regular,
    evaluar_media,
    matriz_operador_ar,
    matriz_covarianza_innovacion,
    factor_cholesky,
    generador_innovacion,
    semillas_replicas,
    aplicar_ruido_observacion,
    diagnostico_comun,
    norma_hilbert_schmidt,
)

__all__ = [
    "ConfigEscenario1",
    "generar_escenario_1",
    "resumen_escenario_1",
    "simular_trayectoria_far1",
]


# ==========================================================================
# CONFIGURACION
# ==========================================================================

@dataclass
class ConfigEscenario1(ConfigObservacion):
    """
    Parametros del Escenario 1.

    Hereda de `ConfigObservacion` los parametros del esquema de observacion
    (L, T, burn_in, sigma_obs, R, seed, media_fn, jitter) y agrega los del
    mecanismo generador.

    Operador autorregresivo
        gamma   : alcance de la dependencia entre puntos del dominio en el
                  nucleo psi. Valores pequenos concentran la influencia de la
                  curva rezagada en un entorno estrecho de cada tau; valores
                  grandes la distribuyen sobre todo el dominio.
        hs_norm : valor objetivo de ||Psi||_HS. Debe pertenecer a (0, 1) para
                  garantizar la existencia de una solucion estacionaria, y
                  determina la persistencia temporal del proceso.

    Innovacion funcional
        sigma_eps : escala de la innovacion.
        ell       : longitud de correlacion de la innovacion; valores pequenos
                    producen trayectorias mas rugosas.
    """

    gamma: float = 0.3
    hs_norm: float = 0.7
    sigma_eps: float = 1.0
    ell: float = 0.2

    def validar(self) -> None:
        super().validar()
        if not (0.0 < self.hs_norm < 1.0):
            raise ValueError(
                f"hs_norm={self.hs_norm}: debe estar en (0, 1) para garantizar "
                "la existencia de una solucion estacionaria."
            )
        if self.gamma <= 0:
            raise ValueError("gamma debe ser positivo.")
        if self.ell <= 0:
            raise ValueError("ell debe ser positivo.")
        if self.sigma_eps <= 0:
            raise ValueError("sigma_eps debe ser positivo.")


# ==========================================================================
# DINAMICA DEL ESCENARIO
# ==========================================================================

def simular_trayectoria_far1(
    Psi: np.ndarray,
    chol_K: np.ndarray,
    mu: np.ndarray,
    T: int,
    burn_in: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Itera la recursion FAR(1) discretizada y devuelve las T curvas retenidas
    SIN ruido de medicion, de dimension (T, L).

    El estado latente es Y_t = X_t - mu, de modo que la recursion opera sobre
    curvas centradas y la media se suma al construir cada curva. La trayectoria
    se inicializa con una realizacion de la innovacion y se descartan los
    primeros `burn_in` periodos.
    """
    L = Psi.shape[0]
    if chol_K.shape[0] != L or mu.shape[0] != L:
        raise ValueError(
            f"Dimensiones inconsistentes: Psi es {Psi.shape}, chol_K es "
            f"{chol_K.shape} y mu es {mu.shape}."
        )

    innovacion = generador_innovacion(chol_K, rng)

    Y = innovacion()
    for _ in range(burn_in):
        Y = Psi @ Y + innovacion()

    curvas = np.empty((T, L))
    for t in range(T):
        Y = Psi @ Y + innovacion()
        curvas[t] = mu + Y
    return curvas


# ==========================================================================
# GENERADOR PRINCIPAL
# ==========================================================================

def generar_escenario_1(cfg: ConfigEscenario1) -> SalidaSimulacion:
    """
    Genera R replicas independientes del Escenario 1.

    El operador y la factorizacion de la covarianza de la innovacion se
    construyen una sola vez y se comparten entre replicas, dado que no dependen
    de la realizacion aleatoria. Cada replica recibe una semilla derivada de la
    semilla maestra, de modo que la replica r es identica con independencia de
    cuantas replicas se generen o del orden de ejecucion.
    """
    cfg.validar()

    tau = grilla_regular(cfg.L)
    mu = evaluar_media(cfg.media_fn, tau)

    Psi = matriz_operador_ar(tau, cfg.gamma, cfg.hs_norm)
    K = matriz_covarianza_innovacion(tau, cfg.sigma_eps, cfg.ell)
    chol_K = factor_cholesky(K, cfg.jitter)

    hijas, registro = semillas_replicas(cfg.seed, cfg.R)

    curvas = np.empty((cfg.R, cfg.T, cfg.L))
    observaciones = np.empty((cfg.R, cfg.T, cfg.L))

    for r, semilla in enumerate(hijas):
        rng = np.random.default_rng(semilla)
        curvas_r = simular_trayectoria_far1(
            Psi, chol_K, mu, cfg.T, cfg.burn_in, rng
        )
        curvas[r] = curvas_r
        observaciones[r] = aplicar_ruido_observacion(curvas_r, cfg.sigma_obs, rng)

    salida = SalidaSimulacion(
        observaciones=observaciones,
        curvas=curvas,
        grilla=tau,
        media=mu,
        semillas=registro,
        config=cfg,
        internos={"operador": Psi, "cov_innovacion": K},
    )
    salida.diagnostico = resumen_escenario_1(salida)
    return salida


# ==========================================================================
# CONTROL DE CALIDAD ESPECIFICO
# ==========================================================================

def resumen_escenario_1(salida: SalidaSimulacion) -> dict:
    """
    Control de calidad del generador, previo a cualquier ajuste.

    Extiende `diagnostico_comun` con las verificaciones propias del Algoritmo 1:
    que la norma de Hilbert-Schmidt efectiva del operador coincida con el valor
    objetivo y sea menor que uno, condicion suficiente de estacionariedad, y el
    radio espectral de la matriz discretizada, que acota inferiormente dicha
    norma y ofrece una lectura complementaria de la persistencia.

    La autocorrelacion puntual a rezago uno, incluida en el diagnostico comun,
    debe ser coherente con la persistencia impuesta por `hs_norm`; una
    discrepancia sistematica entre ambas es sintoma de un error en la cuadratura
    o en la calibracion del nucleo.
    """
    if not isinstance(salida.config, ConfigEscenario1):
        raise TypeError(
            "resumen_escenario_1 requiere una salida generada con "
            f"ConfigEscenario1; se recibio {type(salida.config).__name__}."
        )

    base = diagnostico_comun(salida)

    Psi = salida.internos.get("operador")
    if Psi is None:
        raise KeyError(
            "La salida no contiene el operador discretizado en `internos`; "
            "no puede verificarse la condicion de estacionariedad."
        )

    hs = norma_hilbert_schmidt(Psi)
    radio = float(np.max(np.abs(np.linalg.eigvals(Psi))))

    especifico = {
        "hs_norm_objetivo": float(salida.config.hs_norm),
        "hs_norm_efectiva": hs,
        "hs_norm_error_absoluto": abs(hs - salida.config.hs_norm),
        "radio_espectral_operador": radio,
        "estacionariedad_garantizada": bool(hs < 1.0),
    }
    return {**base, **especifico}
