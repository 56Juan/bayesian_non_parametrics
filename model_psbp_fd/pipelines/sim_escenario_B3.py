"""
sim_escenario_B3.py
===================
Algoritmo B-3 del anexo (`ane_00_02_03_alg_b3`): proceso con cambio estructural
recurrente.

Modelo generador
----------------
    E = { t : ((t - 1) mod P) < D }

    X_t(tau) = mu(tau) + 1(t in E) A g(tau) + Y_t(tau)

    Y_t = int psi(tau, s) Y_{t-1}(s) ds + kappa_t eps_t(tau),
    kappa_t = kappa^{1(t in E)},

con g(tau) = exp{-(tau - 0.5)^2 / (2 * 0.1^2)} y {Y_t} el FAR(1) centrado de B-1.
La media, el operador y la innovacion se REUTILIZAN de `sim_escenario_B1`; este
modulo solo aporta el calendario de episodios, el desplazamiento y la escala
kappa_t.

Calendario en el calentamiento
------------------------------
`E` se extiende periodicamente a t <= 0 (con `mod` de Python, que es no
negativo), de modo que el estado Y llega a t = 1 ya en el regimen ciclico y no
en el estacionario sin episodios. Solo Y usa el calendario en el calentamiento;
el desplazamiento aditivo se suma unicamente a las curvas retenidas.

Ley condicional
---------------
El inicio y el fin de cada episodio no dependen de la dinamica, de modo que un
modelo que solo observa X_{t-1} no puede anticiparlos con certeza: dada la
curva anterior, la ley de X_t es una MEZCLA (episodio / no episodio) con
escalas distintas (kappa = 1.5). Es el escenario donde la mezcla probit y su
banda tienen algo real que capturar.

Oraculo de un rezago
--------------------
El oraculo CONOCE el calendario (es determinista en t), y el modelo no:

    E[X_t | X_{t-1}] = mu + 1(t in E) A g + Psi (X_{t-1} - mu - 1(t-1 in E) A g).

Esa brecha es parte del diseno y hay que declararla al reportar: ningun modelo
que vea solo X_{t-1} alcanza este techo. El diagnostico incluye tambien
`media_condicional_lineal_sin_calendario` (mu + Psi (X_{t-1} - mu)), lo que
hace el operador verdadero SIN conocer el calendario.

`hs_oraculo_L2` es ||Psi||_HS, la norma del operador de Y. El indicador de
episodio va en `internos` y NO se entrega a la estimacion.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .sim_comun import (
    SalidaSimulacion,
    diagnostico_comun,
    grilla_regular,
    matriz_operador_ar,
    norma_hilbert_schmidt,
)
from .sim_escenario_B1 import ConfigEscenarioB1, generar_far_funcional
from .sim_series_clasicas import r2_empirico_media_condicional

__all__ = [
    "ConfigEscenarioB3",
    "indicador_episodio",
    "forma_desplazamiento",
    "generar_escenario_B3",
    "resumen_escenario_B3",
]


@dataclass
class ConfigEscenarioB3(ConfigEscenarioB1):
    """
    Algoritmo B-3. Valores del anexo (Cuadro tab:ane_algB3): la dinamica de Y_t
    es la de B-1 (heredada: gamma, hs_norm, sigma_eps, ell, mu); P = 20, D = 5,
    A = 3.0, kappa = 1.5, g(tau) = exp{-(tau - 0.5)^2 / (2 * 0.1^2)}.

    P : periodos entre inicios de episodios.
    D : duracion de cada episodio, D < P.
    A : amplitud del desplazamiento.
    kappa : factor que amplifica la escala de la innovacion en episodio, >= 1.
    centro_g, ancho_g : centro y desviacion de la forma g.
    """

    P: int = 20
    D: int = 5
    A: float = 3.0
    kappa: float = 1.5
    centro_g: float = 0.5
    ancho_g: float = 0.1

    def validar(self) -> None:
        super().validar()
        if int(self.P) != self.P or int(self.D) != self.D:
            raise ValueError("P y D deben ser enteros.")
        if not 0 < self.D < self.P:
            raise ValueError(f"Se requiere 0 < D < P; recibido D={self.D}, P={self.P}.")
        if self.kappa < 1.0:
            raise ValueError("kappa debe ser al menos 1.")
        if self.ancho_g <= 0:
            raise ValueError("ancho_g debe ser positivo.")


def indicador_episodio(t: np.ndarray, P: int, D: int) -> np.ndarray:
    """1(t in E) con E = {t : ((t-1) mod P) < D}; valido para t <= 0."""
    t = np.asarray(t, dtype=int)
    return ((t - 1) % int(P)) < int(D)


def forma_desplazamiento(tau: np.ndarray, centro: float, ancho: float) -> np.ndarray:
    """g(tau) = exp{-(tau - centro)^2 / (2 ancho^2)}."""
    return np.exp(-((tau - centro) ** 2) / (2.0 * ancho ** 2))


def generar_escenario_B3(cfg: ConfigEscenarioB3,
                         diagnosticar: bool = True) -> SalidaSimulacion:
    """Genera R replicas del Algoritmo B-3 sobre el esqueleto de B-1."""
    cfg.validar()
    T, burn = int(cfg.T), int(cfg.burn_in)
    tau = grilla_regular(int(cfg.L))
    Psi = matriz_operador_ar(tau, cfg.gamma, cfg.hs_norm, nucleo=cfg.nucleo)

    t_pasos = np.arange(burn + T) - burn + 1                    # t = -burn+1 .. T
    en_ep_pasos = indicador_episodio(t_pasos, cfg.P, cfg.D)
    kappa_pasos = np.where(en_ep_pasos, float(cfg.kappa), 1.0)

    forma = cfg.A * forma_desplazamiento(tau, cfg.centro_g, cfg.ancho_g)   # (L,)
    en_ep_d = indicador_episodio(np.arange(T + 1), cfg.P, cfg.D)           # t = 0..T
    desplazamiento = en_ep_d[:, None] * forma[None, :]                     # (T+1, L)

    ind_ret = np.broadcast_to(en_ep_d[1:].astype(np.int8), (int(cfg.R), T)).copy()
    return generar_far_funcional(
        cfg, Psi, None, np.zeros(burn + T), kappa_pasos,
        desplazamiento_ret=desplazamiento,
        internos_extra={"indicador_episodio": ind_ret,
                        "forma_desplazamiento": forma},
        resumen=resumen_escenario_B3, diagnosticar=diagnosticar,
    )


def resumen_escenario_B3(salida: SalidaSimulacion) -> dict:
    """
    Control de calidad del B-3: diagnostico comun, la fraccion de curvas en
    episodio, la parte de la energia de X - mu que es desplazamiento
    determinista, y el R^2 del oraculo de un rezago con y sin calendario.
    """
    if not isinstance(salida.config, ConfigEscenarioB3):
        raise TypeError("resumen_escenario_B3 requiere ConfigEscenarioB3; se "
                        f"recibio {type(salida.config).__name__}.")
    cfg = salida.config
    T = int(cfg.T)
    Psi = salida.internos["operador"]
    w_quad = salida.internos["pesos_cuadratura"]
    ind = salida.internos["indicador_episodio"].astype(bool)       # (R, T)
    forma = salida.internos["forma_desplazamiento"]                # (L,)
    hs = norma_hilbert_schmidt(Psi, w_quad)

    Xc = salida.curvas - salida.media
    energia_ep = float((ind[:, :, None] * forma[None, None, :] ** 2).sum())
    var_dentro = float(Xc[ind].var()) if ind.any() else float("nan")
    var_fuera = float(Xc[~ind].var()) if (~ind).any() else float("nan")

    return {
        **diagnostico_comun(salida),
        "nucleo": cfg.nucleo,
        "hs_norm_objetivo": float(cfg.hs_norm),
        "hs_norm_efectiva": hs,
        "estacionariedad_garantizada": bool(hs < 1.0),
        "hs_oraculo_L2": hs,
        "P": int(cfg.P), "D": int(cfg.D), "A": float(cfg.A), "kappa": float(cfg.kappa),
        "frac_episodio_teorica": float(cfg.D) / float(cfg.P),
        "frac_episodio_empirica": float(ind.mean()),
        "n_episodios_retenidos": int(np.ceil(T / cfg.P)),
        "fraccion_energia_desplazamiento": energia_ep / float((Xc ** 2).sum()),
        "var_puntual_dentro_episodio": var_dentro,
        "var_puntual_fuera_episodio": var_fuera,
        "r2_oraculo_1rezago_empirico": r2_empirico_media_condicional(salida),
        "r2_oraculo_sin_calendario_empirico": r2_empirico_media_condicional(
            salida, clave="media_condicional_lineal_sin_calendario"),
        "T0_referencia": int(np.floor(cfg.prop_train_referencia * T)),
    }
