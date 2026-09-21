"""
sim_escenario_TS.py
===================
Familia T con CAMBIO DE SIMULADOR por umbral de nivel (escenario 53).

    X_t(tau) = mu_{k(t)}(tau) + b(t) + Y_t(tau),      Y_t = componente estable de T.

`b(t)` es una tendencia lineal a trozos que sube con pendiente `pendiente` hasta
tocar `+cota`, entonces CAMBIA DE SENTIDO y baja hasta `-cota`, y asi
sucesivamente (onda triangular acotada en [-cota, cota]). `k(t)` cuenta los
cruces de la cota: cada vez que el nivel la alcanza cambia el simulador, es
decir la media `mu_k` (se cicla sobre `medias_regimen_fn`) y el signo de la
tendencia. En el cruce la media da un SALTO: no se empalma.

Por que se reusa `generar_escenario_T`
--------------------------------------
La tendencia de la familia T es determinista y se suma DESPUES de la recursion
(ver `_simular_replica`), de modo que Y_t no depende del regimen de tendencia.
Aqui se llama al generador con `deriva = 0` y media nula --> sale Y_t, con su
operador, su innovacion y su mezcla probit sin tocar-- y se compone encima la
media y la tendencia por tramos. Asi hay una sola definicion de la dinamica.

Media condicional del oraculo
-----------------------------
`media_condicional = mu_{k(t)} + b(t) + m_cond_Y`. Es exacta: `b` y `k` son
funciones deterministas de t, el oraculo las conoce.

Lo que hay que declarar al reportar
-----------------------------------
* Con `cota` grande frente a `sd(Y) ~ 1` el nivel domina la varianza total: la primera
  FPC es casi puro nivel, y el cambio de `mu_k` (amplitud del orden de unidades)
  queda en las FPC siguientes. Es el diseno, no un defecto.
* El centrado del FPCA y del estandarizador se ajustan con el bloque de
  entrenamiento (que solo ve los primeros tramos) y dejan de ser validos al
  cruzar; ver `resumen_escenario_TS`, que reporta en que instantes ocurre.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Callable, Optional, Sequence

import numpy as np

from ..sim_comun import SalidaSimulacion, diagnostico_comun
from .sim_escenario_T import ConfigEscenarioT, generar_escenario_T

__all__ = [
    "ConfigEscenarioTS",
    "generar_escenario_TS",
    "resumen_escenario_TS",
    "trayectoria_nivel_acotada",
]


def trayectoria_nivel_acotada(T: int, cota: float, pendiente: float,
                              nivel0: float = 0.0, sentido0: int = 1
                              ) -> tuple[np.ndarray, np.ndarray]:
    """Onda triangular b(t) en [-cota, cota] y contador de cruces k(t).

    Devuelve `(b, k)` de largo T. `b[0] = nivel0`; a cada paso suma
    `sentido * pendiente` y, al alcanzar +-cota, se fija en la cota y se invierte
    el sentido. `k[t]` es el numero de inversiones ocurridas hasta t inclusive.
    """
    if cota <= 0 or pendiente <= 0:
        raise ValueError("cota y pendiente deben ser positivas.")
    if abs(nivel0) >= cota:
        raise ValueError("|nivel0| debe ser menor que la cota.")
    b = np.empty(T)
    k = np.zeros(T, dtype=int)
    nivel, sentido, cruces = float(nivel0), int(np.sign(sentido0)) or 1, 0
    for t in range(T):
        if t > 0:
            nivel += sentido * pendiente
            if abs(nivel) >= cota:
                nivel = np.sign(nivel) * cota
                sentido = -sentido
                cruces += 1
        b[t], k[t] = nivel, cruces
    return b, k


@dataclass
class ConfigEscenarioTS(ConfigEscenarioT):
    """`ConfigEscenarioT` + cambio de simulador por umbral de nivel.

    `deriva`, `forma_tendencia` y `media_fn` de la base NO se usan (la tendencia
    y la media las fijan `cota`, `pendiente` y `medias_regimen_fn`).
    """
    cota: float = 20.0
    pendiente: float = 0.4
    nivel0: float = 0.0
    medias_regimen_fn: Sequence[Callable[[np.ndarray], np.ndarray]] = ()

    def validar(self) -> None:
        if len(self.medias_regimen_fn) < 2:
            raise ValueError("medias_regimen_fn necesita al menos dos medias.")
        trayectoria_nivel_acotada(2, self.cota, self.pendiente, self.nivel0)
        self._base().validar()

    def _base(self) -> ConfigEscenarioT:
        propios = {f.name for f in fields(ConfigEscenarioT)}
        kw = {n: getattr(self, n) for n in propios}
        kw.update(deriva=0.0, media_fn=None, forma_tendencia="lineal")
        return ConfigEscenarioT(**kw)

    def to_dict(self) -> dict:
        d = super().to_dict()
        d["medias_regimen_fn"] = [getattr(f, "__name__", "callable")
                                  for f in self.medias_regimen_fn]
        return d


def generar_escenario_TS(cfg: ConfigEscenarioTS) -> SalidaSimulacion:
    cfg.validar()
    salida = generar_escenario_T(cfg._base(), diagnosticar=False)
    tau = salida.grilla
    R, T, L = salida.curvas.shape

    b, k = trayectoria_nivel_acotada(T, cfg.cota, cfg.pendiente, cfg.nivel0)
    mus = np.stack([np.asarray(f(tau), dtype=float) for f in cfg.medias_regimen_fn])
    mu_t = mus[k % len(mus)]                                   # (T, L)
    determinista = mu_t + b[:, None]                           # (T, L)

    Y = salida.curvas                                          # media 0, sin tendencia
    salida.curvas = Y + determinista
    salida.observaciones = salida.observaciones + determinista  # mismo ruido de medicion
    salida.media = mus[0]
    salida.config = cfg
    ints = salida.internos
    ints["media_condicional"] = ints["media_condicional"] + determinista
    ints["tendencia"] = np.broadcast_to(b[:, None], (R, T, L)).copy()
    ints["tendencia_determinista"] = ints["tendencia"]
    ints["nivel_tendencia"] = b
    ints["indice_regimen_nivel"] = k
    ints["medias_regimen"] = mus
    salida.diagnostico = resumen_escenario_TS(salida)
    return salida


def resumen_escenario_TS(salida: SalidaSimulacion) -> dict:
    """Control de calidad: cruces, nivel en T0 y salto de media en cada cruce."""
    cfg = salida.config
    b = salida.internos["nivel_tendencia"]
    k = salida.internos["indice_regimen_nivel"]
    mus = salida.internos["medias_regimen"]
    w = salida.internos["pesos_cuadratura"]
    T = b.size
    T0 = int(np.floor(cfg.prop_train_referencia * T))
    t_cruce = [int(t) for t in np.flatnonzero(np.diff(k) != 0) + 1]   # base-0
    saltos = []
    for t in t_cruce:
        d = mus[k[t] % len(mus)] - mus[k[t - 1] % len(mus)]
        saltos.append(float(np.sqrt(np.sum(w * d ** 2))))
    Y = salida.curvas[0] - salida.internos["tendencia"][0] - np.stack(
        [mus[j % len(mus)] for j in k])
    return {
        **diagnostico_comun(salida),
        "mecanismo": cfg.mecanismo,
        "cota": float(cfg.cota),
        "pendiente": float(cfg.pendiente),
        "n_regimenes_nivel": int(k[-1] + 1),
        "t_cruces_base0": t_cruce,
        "cruces_en_train": [t for t in t_cruce if t < T0],
        "cruces_en_test": [t for t in t_cruce if t >= T0],
        "nivel_en_T0": float(b[T0 - 1]),
        "nivel_min": float(b.min()),
        "nivel_max": float(b.max()),
        "salto_media_L2_por_cruce": saltos,
        "sd_Y": float(Y.std()),
        "razon_cota_sd_Y": float(cfg.cota / max(float(Y.std()), 1e-300)),
        "todo_finito": bool(np.all(np.isfinite(salida.curvas))),
    }
