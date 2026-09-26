"""
sim_escenario_B2_tvfar.py
==========================
HISTORICO. Generador original del Algoritmo B-2 (usado por la corrida 72, que
no se re-ejecuta). El anexo redefinio B-2 como una mezcla de mecanismos con
formas no lineales (ver `pipelines/sim_escenario_B2.py`, vigente); este modulo
ya NO corresponde a lo que `docs/01 Anexo.tex` llama B-2. Se conserva solo para
que 72_01 siga siendo legible como documento historico.

Contenido original: proceso autorregresivo funcional de coeficientes variables
en el tiempo (TV-FAR).

Modelo generador
----------------
    X_t(tau) = mu(tau) + int psi_t(tau, s) (X_{t-1}(s) - mu(s)) ds + eps_t(tau)

    psi_t = (1 - w_t) psi^(0) + w_t psi^(1),      w_t = t / T,

con psi^(0) y psi^(1) de la forma exponencial de B-1, calibrados por separado a
sus normas y alcances. La media y la innovacion son las de B-1 y se REUTILIZAN
de `sim_escenario_B1`; este modulo solo aporta el operador que deriva.

Convencion de w_t
-----------------
`t = 1, ..., T` recorre las curvas RETENIDAS, no el calentamiento: w_1 = 1/T y
w_T = 1, y el operador final se alcanza en la ultima curva. Durante el
calentamiento (t <= 0) se usa w = 0, es decir psi^(0): el proceso parte del
operador inicial y llega a t = 1 en su regimen estacionario, sin que el
calentamiento consuma parte del recorrido de w_t.

Como ambos alcances coinciden (gamma^(0) = gamma^(1) = 0.3) la forma del nucleo
es la misma y ||Psi_t||_HS es lineal en w_t: 0.30 + 0.50 w_t.

Que mide
--------
El operador deriva de 0.30 a 0.80. Un estimador ajustado con el bloque de
entrenamiento (t <= T0, w <= 0.7) queda MAL ESPECIFICADO en el bloque de prueba,
donde el operador verdadero es mas persistente que el promedio del que aprendio.
La ganancia esperada esta fuera de muestra y debe verse como un salto en T0.

Oraculo de un rezago
--------------------
    E[X_t | X_{t-1}] = mu + Psi_t (X_{t-1} - mu),

exacta, con Psi_t el operador VIGENTE en t. La norma cambia con t, de modo que
el diagnostico reporta ||Psi_0||, ||Psi_T0|| y ||Psi_T||; `hs_oraculo_L2` es el
promedio de ||Psi_t|| sobre el bloque de prueba (t > T0), que es contra lo que
se mide el FAR fuera de muestra.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..sim_comun import (
    SalidaSimulacion,
    diagnostico_comun,
    grilla_regular,
    matriz_operador_ar,
    norma_hilbert_schmidt,
    pesos_trapezoidales,
)
from .sim_escenario_B1_far import _ConfigBaseB, generar_far_funcional
from ..sim_series_clasicas import r2_empirico_media_condicional

__all__ = [
    "ConfigEscenarioB2",
    "pesos_deriva_operador",
    "generar_escenario_B2",
    "resumen_escenario_B2",
]


@dataclass
class ConfigEscenarioB2(_ConfigBaseB):
    """
    Algoritmo B-2. Valores del anexo (Cuadro tab:ane_algB2): la media y la
    innovacion son las de B-1; gamma^(0) = gamma^(1) = 0.3,
    ||Psi^(0)||_HS = 0.30, ||Psi^(1)||_HS = 0.80.
    """

    gamma0: float = 0.3
    gamma1: float = 0.3
    hs_norm0: float = 0.30
    hs_norm1: float = 0.80

    def validar(self) -> None:
        super().validar()
        if self.gamma0 <= 0 or self.gamma1 <= 0:
            raise ValueError("gamma0 y gamma1 deben ser positivos.")
        for nombre in ("hs_norm0", "hs_norm1"):
            v = getattr(self, nombre)
            if not 0.0 < v < 1.0:
                raise ValueError(f"{nombre}={v}: debe estar en (0, 1).")


def pesos_deriva_operador(T: int, burn_in: int) -> np.ndarray:
    """
    w_i para los `burn_in + T` pasos: w = t / T con t = 1..T en las curvas
    retenidas y w = 0 en el calentamiento (t <= 0).
    """
    t = np.arange(burn_in + T) - burn_in + 1
    return np.clip(t / float(T), 0.0, 1.0)


def generar_escenario_B2(cfg: ConfigEscenarioB2,
                         diagnosticar: bool = True) -> SalidaSimulacion:
    """Genera R replicas del Algoritmo B-2 sobre el esqueleto de B-1."""
    cfg.validar()
    tau = grilla_regular(int(cfg.L))
    Psi0 = matriz_operador_ar(tau, cfg.gamma0, cfg.hs_norm0, nucleo=cfg.nucleo)
    Psi1 = matriz_operador_ar(tau, cfg.gamma1, cfg.hs_norm1, nucleo=cfg.nucleo)
    w = pesos_deriva_operador(int(cfg.T), int(cfg.burn_in))
    w_quad = pesos_trapezoidales(tau)
    hs_t = np.array([_hs_en_w(Psi0, Psi1, wi, w_quad) for wi in w[int(cfg.burn_in):]])
    return generar_far_funcional(
        cfg, Psi0, Psi1, w, np.ones(w.size),
        internos_extra={"hs_operador_t": hs_t},
        resumen=resumen_escenario_B2, diagnosticar=diagnosticar,
    )


def _hs_en_w(Psi0, Psi1, w, w_quad) -> float:
    return norma_hilbert_schmidt((1.0 - w) * Psi0 + w * Psi1, w_quad)


def _r2_bloque(salida: SalidaSimulacion, desde: int, hasta: int) -> float:
    """R^2 del oraculo sobre las curvas t = desde+1..hasta (indices 0-based [desde, hasta))."""
    X = salida.curvas[:, desde:hasta] - salida.media
    Mc = salida.internos["media_condicional"][:, desde:hasta] - salida.media
    sst = float((X ** 2).sum())
    return float(1.0 - ((X - Mc) ** 2).sum() / sst) if sst > 0 else float("nan")


def resumen_escenario_B2(salida: SalidaSimulacion) -> dict:
    """
    Control de calidad del B-2: diagnostico comun, la norma HS de los dos
    operadores extremos contra sus objetivos, ||Psi_t|| en t = 0, T0 y T, y el
    R^2 del oraculo por bloque (el proceso no es estacionario: la varianza
    crece con la norma del operador, de modo que un R^2 unico no describe a
    ninguno de los dos bloques).
    """
    if not isinstance(salida.config, ConfigEscenarioB2):
        raise TypeError("resumen_escenario_B2 requiere ConfigEscenarioB2; se "
                        f"recibio {type(salida.config).__name__}.")
    cfg = salida.config
    T = int(cfg.T)
    T0 = int(np.floor(cfg.prop_train_referencia * T))
    Psi0, Psi1 = salida.internos["operador"], salida.internos["operador_final"]
    w_quad = salida.internos["pesos_cuadratura"]
    hs_t = salida.internos["hs_operador_t"]                      # (T,) t = 1..T
    hs0 = norma_hilbert_schmidt(Psi0, w_quad)
    hs1 = norma_hilbert_schmidt(Psi1, w_quad)

    return {
        **diagnostico_comun(salida),
        "nucleo": cfg.nucleo,
        "hs_norm0_objetivo": float(cfg.hs_norm0),
        "hs_norm0_efectiva": hs0,
        "hs_norm1_objetivo": float(cfg.hs_norm1),
        "hs_norm1_efectiva": hs1,
        "estacionariedad_garantizada": bool(max(hs0, hs1) < 1.0),
        "hs_operador_inicio": hs0,                                  # ||Psi_0||  (w = 0)
        "hs_operador_T0": _hs_en_w(Psi0, Psi1, T0 / T, w_quad),     # ||Psi_T0||
        "hs_operador_final": float(hs_t[-1]),                       # ||Psi_T||
        "hs_operador_medio_train": float(hs_t[:T0].mean()),
        "hs_operador_medio_test": float(hs_t[T0:].mean()),
        # Norma que el _05 contrasta contra el operador que estima el FAR:
        "hs_oraculo_L2": float(hs_t[T0:].mean()),
        "r2_oraculo_1rezago_empirico": r2_empirico_media_condicional(salida),
        "r2_oraculo_1rezago_train": _r2_bloque(salida, 1, T0),
        "r2_oraculo_1rezago_test": _r2_bloque(salida, T0, T),
        "T0_referencia": T0,
    }
