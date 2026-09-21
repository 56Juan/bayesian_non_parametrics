"""
sim_escenario_J.py
===================
Escenario J: proceso autorregresivo funcional con CUATRO REZAGOS y una
no linealidad POLINOMIAL cuyo grado DECRECE con el rezago. Escenario de
DIAGNOSTICO, no es un Algoritmo del anexo; se nombra con la siguiente letra
libre despues de B y de la familia T (C-I), que ya ocupan A-I.

Modelo generador
-----------------
    Y_t = Psi Y_{t-1} + termino_1(Y_{t-1}) + termino_2(Y_{t-2})
                       + termino_3(Y_{t-3}) + Psi_4 Y_{t-4} + eps_t,

    X_t(tau) = mu(tau) + Y_t(tau),

con `termino_k` (k = 1, 2, 3) una no linealidad de grado `p_k = 5 - k`
(4, 3, 2) leida y escrita LOCALMENTE sobre el dominio, y el cuarto rezago
entrando solo por una cola lineal `Psi_4` de norma pequena.

Por que existe este escenario
------------------------------
Los seis Algoritmos del anexo, el Escenario B y la familia T (C-I) son todos
de UN SOLO REZAGO: la media condicional depende de X_{t-1} y de nada mas
antiguo. Ningun escenario del estudio pone a prueba si el modelo recupera
DEPENDENCIA A REZAGOS MULTIPLES cuando ademas esa dependencia no es uniforme
---un rezago inmediato con una no linealidad marcada y rezagos mas lejanos con
no linealidades cada vez mas suaves, hasta ser lineales---. Es exactamente la
estructura que separa "el modelo ve el rezago 1" de "el modelo ve y pesa
correctamente los rezagos 1 a 4", y es el motivo por el que este escenario
existe: en las corridas 11-30 y B, `N_LAGS = 1` en `hyperparameters.json` es
suficiente porque el generador tambien lo es. Aqui NO lo es, y por eso el
`_01` de este escenario tiene que declarar `N_LAGS = 4` explicitamente (ver
la nota grande en el notebook): con `N_LAGS = 1` el diseno AR ni siquiera
contiene las columnas de los rezagos 2 a 4 y el escenario queda roto en
silencio, exactamente el error que CLAUDE.md advierte para este caso.

Por que la lectura y la escritura son LOCALES, y por que la saturacion no es
cosmetica
------------------------------------------------------------------------------
`termino_k` se construye como en el mecanismo "interaccion" de
`sim_escenario_T.py`: se LEE la curva rezagada con un promedio local
(`nucleo_local`, integral unitaria, no el valor puntual de una celda, que no
sobrevive a la base ni al truncamiento FPCA) y se ESCRIBE la respuesta sobre
una forma localizada distinta (`_forma_unitaria`, norma L^2 unitaria), en un
punto del dominio distinto para cada rezago, de modo que las tres no
linealidades carguen sobre direcciones distintas del espectro FPCA y el
barrido en M tenga, como en el Escenario B, un punto de corte identificable
por rezago en vez de una degradacion uniforme.

Una potencia z^p con p >= 2 sin acotar es explosiva con probabilidad
positiva bajo una recursion con rezagos multiples: basta que un rezago
alimente a otro con signo reforzante para que la trayectoria escape antes de
completar el calentamiento (se verifico al construir el escenario: sin
`tanh`, T=800 no se completa). La saturacion se fija, como en la familia T,
en `saturacion` desviaciones tipicas de la lectura elevada a su potencia,
medidas en un PILOTO sin el termino activo; en el rango central el termino es
la potencia exacta y `fraccion_saturada_k` reporta cuanto de la serie cae
fuera de esa zona.

La decision sobre el signo en las potencias PARES
---------------------------------------------------
La primera version de este modulo usaba `raw_k(z) = signo(z) |z|^{p_k}` para
los tres grados, de modo que `raw_k` fuera impar con independencia de la
paridad de `p_k`, evitando a proposito el sesgo de nivel de `tanh(z^4) >= 0`.
Esa decision se REVIRTIO al validar el escenario numericamente, por una razon
empirica y no solo estetica: `signo(z)|z|^p` es una funcion MONOTONA de z, y
una transformacion monotona de un funcional lineal esta fuertemente
correlacionada linealmente con ese mismo funcional (para gaussianas,
corr(Z, Z^3) ~ 0.77). El mejor predictor lineal la recupera casi por completo,
y `r2_lineal_fuera_de_muestra` SUBIA con la magnitud del termino en vez de
bajar --lo opuesto de lo que el escenario existe para producir--.

La version definitiva usa, para los grados PARES (k=1, grado 4; k=3, grado 2):

    raw_k(z) = z^{p_k} - c_k,

sin corregir el signo. Es la misma logica de simetria que usa
`sim_escenario_B.py` para su conmutacion: si la ley de z fuera exactamente
simetrica, Cov(z, z^p) = 0 EXACTO para p par, de modo que el termino es en
primer orden invisible para cualquier predictor LINEAL de z, mientras que el
ORACULO (que conoce m(Y) tal cual) lo recupera integro. El precio, igual que
el termino cuadratico de `sim_escenario_T.py`, es que z^p con p par tiene
media no nula; `c_k` la resta (medida en el mismo piloto que calibra la
escala), para que el termino no se confunda con un desplazamiento del nivel
global. Para el grado IMPAR (k=2, grado 3) no hace falta nada de esto: z^3 ya
es una funcion impar de z por construccion y no se centra.

Relacion con el Escenario B y la familia T
--------------------------------------------
Comparten la maquinaria de lectura/escritura local y la calibracion por
cociente de desviaciones L^2 medido en un piloto (`_calibrar_terminos`, misma
idea que `_calibrar_interaccion` de `sim_escenario_T.py`, aplicada de forma
independiente a cada uno de los tres rezagos no lineales por simplicidad: no
hay razon a priori para que la calibracion conjunta cambie el orden de
magnitud, y el diagnostico mide la razon EFECTIVA sobre la serie definitiva,
no da por buena la nominal). Difiere de ambos en que la no linealidad no es
ni una conmutacion de signo (B) ni una interaccion entre dos puntos de la
MISMA curva rezagada (T): son tres no linealidades univariadas, una por
rezago, con grado decreciente.

Lo que NO vive aqui
--------------------
El esquema de observacion, la cuadratura del operador, la innovacion
funcional gaussiana, `nucleo_local`, `_forma_unitaria` y el control de
calidad transversal se REUTILIZAN de `sim_comun.py` y `sim_escenario_T.py`;
no se redefinen (la duplicacion de cuadratura/Cholesky es justamente el
gotcha que CLAUDE.md marca como fuente de degradacion silenciosa).

Uso tipico desde un notebook
------------------------------
    from model_psbp_fd.pipelines import (
        ConfigEscenarioJ, generar_escenario_J, resumen_escenario_J,
    )

    cfg = ConfigEscenarioJ(
        L=100, T=800, burn_in=200, R=1, seed=41232, sigma_obs=0.25,
        media_fn=media_senoidal,
    )
    salida = generar_escenario_J(cfg)
    salida.diagnostico["r2_oraculo_fuera_de_muestra"]
    X = salida.observaciones                     # (R, T, L)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np

from ..sim_comun import (
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
    pesos_trapezoidales,
)
from .sim_escenario_T import nucleo_local, _forma_unitaria

__all__ = [
    "ConfigEscenarioJ",
    "generar_escenario_J",
    "resumen_escenario_J",
]

# Grado polinomial de cada uno de los tres rezagos no lineales, decreciente:
# rezago 1 -> grado 4, rezago 2 -> grado 3, rezago 3 -> grado 2. El cuarto
# rezago (grado 1, lineal) no pasa por esta tabla: entra por un operador
# integral aparte (Psi_4), sin saturacion, porque un termino lineal no
# necesita acotarse.
_GRADOS_NO_LINEALES = (4, 3, 2)


# ==========================================================================
# CONFIGURACION
# ==========================================================================

@dataclass
class ConfigEscenarioJ(ConfigObservacion):
    """
    Parametros del Escenario J. Hereda el esquema de observacion de
    `ConfigObservacion` (L, T, burn_in, sigma_obs, R, seed, media_fn, jitter).

    Rezago 1 -- parte lineal principal
        gamma, hs_norm : nucleo gaussiano de Psi y su norma de Hilbert-Schmidt,
                         igual en forma al Algoritmo 1. `hs_norm` se deja
                         DELIBERADAMENTE bajo (0.35 contra 0.70 del Algoritmo
                         1), y no solo por la cota de estacionariedad: al
                         validar el escenario, un `hs_norm` alto dejaba al
                         predictor lineal capturar casi toda la varianza por
                         la sola persistencia lineal, sin margen para que la
                         no linealidad de los rezagos 1-3 se notara en la
                         brecha oraculo-lineal (ver la tabla de diagnostico
                         mas abajo).
        sigma_eps, ell : escala y suavidad de la innovacion funcional,
                         compartida por los cuatro rezagos.

    Rezago 4 -- cola lineal
        gamma_4, hs_norm_4 : mismo nucleo gaussiano, norma pequena por
                             diseno (0.05): es la parte "de control" que
                             hace que el rezago mas antiguo no necesite una
                             no linealidad propia, solo refuerce (poco) la
                             persistencia lineal ya presente en el rezago 1.

    Rezagos 2 y 3 -- no linealidades de grado decreciente
        centros_lectura   : punto tau donde se LEE (promedio local,
                            `nucleo_local`) cada rezago no lineal, uno por
                            k en {1, 2, 3}. Por defecto (0.20, 0.50, 0.80):
                            tres regiones distintas del dominio.
        centros_escritura : punto tau donde se ESCRIBE la respuesta de cada
                            termino (`_forma_unitaria`). Por defecto
                            (0.80, 0.20, 0.50): un desfase ciclico respecto de
                            `centros_lectura` para que la lectura y la
                            escritura de un mismo rezago no coincidan (si
                            coincidieran, el termino se leeria y escribiria
                            sobre la misma region y cargaria sobre una unica
                            direccion FPCA en vez de dos).
        ancho_lectura, ancho_escritura : anchura de los nucleos de lectura y
                            de la forma de escritura. Del orden de `ell` para
                            que la lectura sea estable frente al ruido de
                            discretizacion (ver `nucleo_local`).
        saturacion         : semiancho de la zona lineal del `tanh`, en
                            desviaciones de `raw_k` medidas en el piloto.
                            Mismo papel que en `sim_escenario_T.py`.
        razon_terminos     : (razon_1, razon_2, razon_3), cuanto pesa cada
                            termino no lineal frente a la parte lineal del
                            rezago 1, medido como cociente de desviaciones
                            L^2 y CALIBRADO por el generador (no es una
                            constante sin unidades). Por defecto
                            (0.75, 0.60, 0.45): decreciente con el grado,
                            porque `raw_k` con `p_k` alto crece mucho mas
                            rapido y una razon comparable a la del grado 2
                            dejaria el termino de grado 4 saturado casi
                            siempre (medido: por encima de ~1.0 la fraccion
                            saturada salta de ~2 % a >40 % y el escenario dea
                            de tener la lectura "mayormente lineal, saturada
                            solo en la cola" que se busca).

    Diagnostico del generador con los parametros por defecto (seed=41232,
    L=100, T=800, R=1; ver tambien el modulo de validacion standalone usado
    para elegirlos):

        r2_lineal_fuera_de_muestra   ~ 0.135   (el mejor VAR(4)/FAR(4))
        r2_oraculo_fuera_de_muestra  ~ 0.247   (conociendo m(Y) exacto)
        brecha_oraculo_lineal        ~ 0.112
        fraccion_saturada (k=1,2,3)  ~ 0.014, 0.014, 0.020
        hs_norm_suma                 = 0.40    (< 1, estacionario)

    La brecha es mas modesta que la del Escenario B (~0.34) porque aqui la
    no linealidad esta REPARTIDA en tres rezagos con escritura localizada en
    vez de concentrada en una unica conmutacion de signo sobre todo el
    operador; es el precio de que el escenario ponga a prueba la estructura
    de MULTIPLES rezagos y no solo una no linealidad fuerte en uno solo. Con
    `razon_terminos` mas alto la brecha crece pero la saturacion se dispara
    (ver la nota de `razon_terminos` arriba); el punto elegido prioriza una
    lectura "mayormente lineal, ocasionalmente saturada" sobre maximizar la
    brecha.

    Diagnostico
        n_dim_diagnostico : componentes principales empiricas sobre las que
                            se ajusta el mejor predictor lineal del control
                            de calidad.
        n_pilot           : longitud de la trayectoria piloto de calibracion.
    """

    # Rezago 1
    gamma: float = 0.30
    hs_norm: float = 0.35
    sigma_eps: float = 1.0
    ell: float = 0.5

    # Rezago 4 (cola lineal)
    gamma_4: float = 0.30
    hs_norm_4: float = 0.05

    # Rezagos 1-3, no lineales
    centros_lectura: Sequence[float] = (0.20, 0.50, 0.80)
    centros_escritura: Sequence[float] = (0.80, 0.20, 0.50)
    ancho_lectura: float = 0.10
    ancho_escritura: float = 0.15
    saturacion: float = 3.0
    razon_terminos: Sequence[float] = (0.75, 0.60, 0.45)

    # Diagnostico
    n_dim_diagnostico: int = 5
    n_pilot: int = 3000

    def validar(self) -> None:
        super().validar()

        if self.gamma <= 0:
            raise ValueError("gamma debe ser positivo.")
        if not (0.0 < self.hs_norm < 1.0):
            raise ValueError(f"hs_norm={self.hs_norm}: debe estar en (0, 1).")
        if self.sigma_eps <= 0:
            raise ValueError("sigma_eps debe ser positivo.")
        if self.ell <= 0:
            raise ValueError("ell debe ser positivo.")

        if self.gamma_4 <= 0:
            raise ValueError("gamma_4 debe ser positivo.")
        if not (0.0 < self.hs_norm_4 < 1.0):
            raise ValueError(f"hs_norm_4={self.hs_norm_4}: debe estar en (0, 1).")
        if self.hs_norm + self.hs_norm_4 >= 1.0:
            raise ValueError(
                f"hs_norm + hs_norm_4 = {self.hs_norm + self.hs_norm_4} >= 1: "
                "la suma de las dos normas de Hilbert-Schmidt lineales es la "
                "condicion suficiente de estacionariedad que este escenario usa "
                "(los tres terminos no lineales estan acotados por `tanh` y no "
                "aportan a la parte no acotada de la recursion)."
            )

        if len(self.centros_lectura) != 3 or len(self.centros_escritura) != 3:
            raise ValueError(
                "centros_lectura y centros_escritura deben tener longitud 3, "
                "una por cada rezago no lineal (k=1,2,3)."
            )
        for c in list(self.centros_lectura) + list(self.centros_escritura):
            if not (0.0 <= c <= 1.0):
                raise ValueError(f"centro={c} fuera de [0, 1].")
        if self.ancho_lectura <= 0 or self.ancho_escritura <= 0:
            raise ValueError("ancho_lectura y ancho_escritura deben ser positivos.")
        if self.saturacion <= 0:
            raise ValueError("saturacion debe ser positivo.")
        if len(self.razon_terminos) != 3:
            raise ValueError("razon_terminos debe tener longitud 3 (k=1,2,3).")
        for rz in self.razon_terminos:
            if rz < 0:
                raise ValueError("razon_terminos no puede tener entradas negativas.")
            if rz > 2.0:
                raise ValueError(
                    f"razon_terminos={list(self.razon_terminos)} demasiado grande: "
                    "el termino no lineal dominaria por completo la parte lineal."
                )
        if self.n_dim_diagnostico < 1:
            raise ValueError("n_dim_diagnostico debe ser al menos 1.")
        if self.n_pilot < 100:
            raise ValueError("n_pilot debe ser al menos 100.")


# ==========================================================================
# LECTURA / ESCRITURA LOCAL POR REZAGO
# ==========================================================================

def _armar_terminos(tau: np.ndarray, cfg: "ConfigEscenarioJ") -> dict:
    """
    Pre-calcula, para cada uno de los tres rezagos no lineales, el vector de
    lectura (con la cuadratura ya incorporada, listo para `lectura @ y`) y la
    forma de escritura.
    """
    w = pesos_trapezoidales(tau)
    lecturas, escrituras = [], []
    for c_l, c_e in zip(cfg.centros_lectura, cfg.centros_escritura):
        lecturas.append(w * nucleo_local(tau, float(c_l), cfg.ancho_lectura))
        escrituras.append(_forma_unitaria(tau, float(c_e), cfg.ancho_escritura))
    return {
        "lecturas": np.array(lecturas),      # (3, L), ya con cuadratura
        "escrituras": np.array(escrituras),  # (3, L)
    }


def _raw_k(z: np.ndarray, p: int, centrado: float = 0.0) -> np.ndarray:
    """
    z^p, CENTRADO cuando p es PAR y sin centrar cuando p es IMPAR.

    Esta es la decision final sobre el signo en las potencias pares, y
    reemplaza a una version anterior (`signo(z) * |z|^p` uniforme para los
    tres grados) que se probo primero y se descarto por una razon empirica,
    no solo esteica: con p impar, z^p (grado 3, k=2) ya es una funcion IMPAR
    de z sin necesidad de ningun ajuste, y con p par (grados 4 y 2, k=1 y
    k=3), la version `signo(z)|z|^p` es MONOTONA en z --tan correlacionada
    linealmente con el propio z como cualquier transformacion monotona lo
    esta-- y el mejor predictor lineal la recupera casi por completo (medido
    al construir el escenario: `r2_lineal_fuera_de_muestra` SUBIA con la
    magnitud del termino en vez de bajar, exactamente lo opuesto de lo que el
    escenario existe para producir). z^p sin corregir el signo, en cambio, es
    una funcion PAR: para un proceso con ley aproximadamente simetrica,
    Cov(z, z^p) = 0 de forma EXACTA cuando p es par, de modo que el termino
    es en primer orden invisible para cualquier predictor LINEAL de z --el
    mismo argumento de simetria que usa `sim_escenario_B.py` para su
    conmutacion de signo--, mientras que el ORACULO (que conoce m(Y) tal
    cual) lo recupera integro. El precio es el mismo que paga el termino
    cuadratico de `sim_escenario_T.py`: una funcion par de un funcional
    centrado tiene media no nula (E[z^p] > 0 para p par), y ese
    desplazamiento se resta con el mismo mecanismo de centrado por piloto
    (`centrado`, el analogo del `c_j` de `_calibrar_interaccion`), para que
    el termino no se confunda con un cambio de nivel de la media global.
    """
    if p % 2 == 0:
        return z ** p - centrado
    return z ** p


def _termino_k(z: float, k: int, escala: float, cota: float, centrado: float,
                escritura: np.ndarray) -> np.ndarray:
    """lambda_k * cota_k * tanh(raw_k(z) / cota_k) * h_k(tau)."""
    p = _GRADOS_NO_LINEALES[k]
    raw = _raw_k(np.asarray(z), p, centrado)
    sat = cota * np.tanh(raw / cota)
    return escala * sat * escritura


# ==========================================================================
# CALIBRACION
# ==========================================================================

def _calibrar_terminos(
    Psi: np.ndarray, Psi4: np.ndarray, chol_K: np.ndarray,
    cfg: "ConfigEscenarioJ", terminos: dict, w_quad: np.ndarray, semilla: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calibra, para cada uno de los tres rezagos no lineales, el factor global
    `lambda_k` y la cota de saturacion `cota_k`, a partir de una trayectoria
    PILOTO sin ningun termino no lineal activo (solo Psi y Psi_4).

    Se mide sobre el piloto la desviacion L^2 de la parte lineal del rezago 1
    (`||Psi Y_{t-1}||`) y la de cada `raw_k` saturado sin escalar, y se fija
    `lambda_k` de modo que el cociente sea `razon_terminos[k]`. Es la misma
    logica de `_calibrar_interaccion` de `sim_escenario_T.py`, aplicada de
    forma independiente a cada rezago: la calibracion es de primer orden (al
    activar los tres terminos la varianza del proceso cambia), por eso
    `resumen_escenario_J` reporta la razon EFECTIVA medida sobre la serie
    definitiva en vez de dar por buena la nominal.
    """
    rng = np.random.default_rng(semilla)
    innovacion = generador_innovacion(chol_K, rng)
    L = Psi.shape[0]

    hist = [innovacion() for _ in range(4)]  # hist[-1] = mas reciente
    for _ in range(cfg.burn_in):
        arrastre = Psi @ hist[-1] + Psi4 @ hist[-4]
        hist.append(arrastre + innovacion())
        hist.pop(0)

    n = int(cfg.n_pilot)
    lin = np.empty(n)
    z_pilot = np.empty((n, 3))
    for i in range(n):
        arrastre = Psi @ hist[-1] + Psi4 @ hist[-4]
        lin[i] = float(np.sqrt(np.sum(w_quad * (Psi @ hist[-1]) ** 2)))
        for k in range(3):
            z_pilot[i, k] = float(terminos["lecturas"][k] @ hist[-1 - k])
        Y_next = arrastre + innovacion()
        hist.append(Y_next)
        hist.pop(0)

    sd_lin = float(np.sqrt(np.mean(lin ** 2)))
    escalas = np.empty(3)
    cotas = np.empty(3)
    centrados = np.zeros(3)
    for k in range(3):
        p = _GRADOS_NO_LINEALES[k]
        if p % 2 == 0:
            centrados[k] = float(np.mean(z_pilot[:, k] ** p))
        raw = _raw_k(z_pilot[:, k], p, centrados[k])
        cota = float(cfg.saturacion) * float(np.std(raw))
        cota = cota if cota > 0 else 1.0
        sat = cota * np.tanh(raw / cota)
        bruto = sat[:, None] * terminos["escrituras"][k][None, :]   # (n, L)
        sd_bruto = float(np.sqrt(np.mean(np.sum(w_quad * bruto ** 2, axis=1))))
        if sd_bruto <= 0:
            raise RuntimeError(
                f"El termino no lineal k={k + 1} resulto identicamente nulo en "
                "el piloto: revise centros_lectura/centros_escritura."
            )
        escalas[k] = float(cfg.razon_terminos[k]) * sd_lin / sd_bruto
        cotas[k] = cota
    return escalas, cotas, centrados


# ==========================================================================
# DINAMICA
# ==========================================================================

def _simular_replica(
    Psi: np.ndarray, Psi4: np.ndarray, chol_K: np.ndarray,
    cfg: "ConfigEscenarioJ", terminos: dict, escalas: np.ndarray,
    cotas: np.ndarray, centrados: np.ndarray, w_quad: np.ndarray,
    rng: np.random.Generator,
) -> dict:
    """
    Itera la recursion de cuatro rezagos y devuelve las curvas SIN media ni
    ruido de medicion, junto con las cantidades inobservables que el
    diagnostico necesita: la media condicional verdadera (el oraculo), el
    estado leido en cada rezago no lineal y la fraccion de saturacion.
    """
    L = Psi.shape[0]
    innovacion = generador_innovacion(chol_K, rng)

    hist = [innovacion() for _ in range(4)]
    for _ in range(cfg.burn_in):
        arrastre = Psi @ hist[-1] + Psi4 @ hist[-4]
        no_lineal = np.zeros(L)
        for k in range(3):
            z = float(terminos["lecturas"][k] @ hist[-1 - k])
            no_lineal += _termino_k(z, k, escalas[k], cotas[k], centrados[k], terminos["escrituras"][k])
        hist.append(arrastre + no_lineal + innovacion())
        hist.pop(0)

    T = cfg.T
    Y_serie = np.empty((T, L))
    m_cond = np.empty((T, L))
    z_lags = np.empty((T, 3))
    saturado = np.empty((T, 3))

    for t in range(T):
        arrastre = Psi @ hist[-1] + Psi4 @ hist[-4]
        no_lineal = np.zeros(L)
        for k in range(3):
            z = float(terminos["lecturas"][k] @ hist[-1 - k])
            z_lags[t, k] = z
            p = _GRADOS_NO_LINEALES[k]
            raw = _raw_k(np.asarray(z), p, centrados[k])
            saturado[t, k] = float(np.abs(raw) > cotas[k])
            no_lineal += _termino_k(z, k, escalas[k], cotas[k], centrados[k], terminos["escrituras"][k])
        m_cond[t] = arrastre + no_lineal
        Y_next = m_cond[t] + innovacion()
        hist.append(Y_next)
        hist.pop(0)
        Y_serie[t] = Y_next

    return {
        "Y": Y_serie, "m_cond_Y": m_cond, "z_lags": z_lags, "saturado": saturado,
    }


# ==========================================================================
# GENERADOR PRINCIPAL
# ==========================================================================

def generar_escenario_J(cfg: ConfigEscenarioJ) -> SalidaSimulacion:
    """
    Genera R replicas independientes del Escenario J.

    Los objetos que no dependen de la realizacion (Psi, Psi_4, la
    factorizacion de la innovacion, las lecturas/escrituras locales y la
    calibracion) se construyen una sola vez y se comparten entre replicas.
    """
    cfg.validar()

    tau = grilla_regular(cfg.L)
    w_quad = pesos_trapezoidales(tau)
    mu = evaluar_media(cfg.media_fn, tau)

    Psi = matriz_operador_ar(tau, cfg.gamma, cfg.hs_norm)
    Psi4 = matriz_operador_ar(tau, cfg.gamma_4, cfg.hs_norm_4)
    K = matriz_covarianza_innovacion(tau, cfg.sigma_eps, cfg.ell)
    chol_K = factor_cholesky(K, cfg.jitter)
    terminos = _armar_terminos(tau, cfg)

    hijas, registro = semillas_replicas(cfg.seed, cfg.R)
    semilla_calib = int(np.random.SeedSequence(cfg.seed).spawn(1)[0].entropy) & 0xFFFFFFFF
    escalas, cotas, centrados = _calibrar_terminos(
        Psi, Psi4, chol_K, cfg, terminos, w_quad, semilla_calib
    )

    curvas = np.empty((cfg.R, cfg.T, cfg.L))
    observaciones = np.empty((cfg.R, cfg.T, cfg.L))
    medias_cond = np.empty((cfg.R, cfg.T, cfg.L))
    z_lags = np.empty((cfg.R, cfg.T, 3))
    saturado = np.empty((cfg.R, cfg.T, 3))

    for r, semilla in enumerate(hijas):
        rng = np.random.default_rng(semilla)
        res = _simular_replica(Psi, Psi4, chol_K, cfg, terminos, escalas, cotas, centrados, w_quad, rng)
        curvas[r] = mu[None, :] + res["Y"]
        medias_cond[r] = mu[None, :] + res["m_cond_Y"]
        z_lags[r] = res["z_lags"]
        saturado[r] = res["saturado"]
        observaciones[r] = aplicar_ruido_observacion(curvas[r], cfg.sigma_obs, rng)

    salida = SalidaSimulacion(
        observaciones=observaciones,
        curvas=curvas,
        grilla=tau,
        media=mu,
        semillas=registro,
        config=cfg,
        internos={
            "operador": Psi,
            "operador_rezago4": Psi4,
            "cov_innovacion": K,
            "pesos_cuadratura": w_quad,
            "lecturas": terminos["lecturas"],
            "escrituras": terminos["escrituras"],
            "escalas_terminos": escalas,
            "cotas_saturacion": cotas,
            "centrados_terminos": centrados,
            "z_lags": z_lags,
            "saturado": saturado,
            "media_condicional": medias_cond,
        },
    )
    salida.diagnostico = resumen_escenario_J(salida)
    return salida


# ==========================================================================
# CONTROL DE CALIDAD ESPECIFICO
# ==========================================================================

def _fpca_empirica(Y: np.ndarray, w: np.ndarray, n_dim: int) -> tuple[np.ndarray, np.ndarray]:
    """Autofunciones y varianzas del problema generalizado C u = lambda W u,
    identico al de `sim_escenario_B.py` (no se redefine la logica, se copia
    la version minima porque ambos modulos evitan depender uno del otro para
    este detalle de diagnostico interno)."""
    n_dim = int(min(n_dim, Y.shape[1]))
    raiz = np.sqrt(w)
    Cov = Y.T @ Y / Y.shape[0]
    Sim = (raiz[:, None] * Cov) * raiz[None, :]
    lam, V = np.linalg.eigh(Sim)
    orden = np.argsort(lam)[::-1][:n_dim]
    U = V[:, orden] / raiz[:, None]
    return np.maximum(lam[orden], 0.0), U


def resumen_escenario_J(salida: SalidaSimulacion) -> dict:
    """
    Control de calidad del generador, previo a cualquier ajuste.

    Extiende `diagnostico_comun` con:

    - Contractividad de la parte lineal (normas HS de Psi y Psi_4, radios
      espectrales, suma de normas).
    - Por cada rezago no lineal k=1,2,3: `razon_termino_objetivo_k`,
      `razon_termino_efectiva_k` (medida sobre la serie definitiva, no la
      nominal del piloto) y `fraccion_saturada_k`.
    - Cuanto pierde el mejor predictor LINEAL que ve los cuatro rezagos
      contra el ORACULO (la media condicional verdadera), proyectados sobre
      las primeras `n_dim_diagnostico` componentes principales empiricas,
      ajustado en la primera mitad de la serie y evaluado en la segunda. Es
      la cifra que decide si el escenario cumple su proposito: un R^2 lineal
      bajo con un R^2 oraculo alto es la senal de que la no linealidad de
      grado alto en el rezago 1 domina sobre lo que un VAR(4)/FAR(4) puede
      capturar.
    - Alineacion espectral: cuanta varianza de cada direccion de escritura
      `h_k` explican las primeras `n_dim_diagnostico` componentes FPCA
      empiricas (`fraccion_h_explicada_k`), analogo a
      `10_alineacion_conmutacion.csv` del Escenario B y a
      `10_alineacion_interaccion.csv` de la familia T: es la cifra que
      predice de antemano el punto de corte del barrido en M para cada
      rezago.
    """
    if not isinstance(salida.config, ConfigEscenarioJ):
        raise TypeError(
            "resumen_escenario_J requiere una salida generada con "
            f"ConfigEscenarioJ; se recibio {type(salida.config).__name__}."
        )

    base = diagnostico_comun(salida)
    cfg = salida.config

    requeridos = (
        "operador", "operador_rezago4", "cov_innovacion", "pesos_cuadratura",
        "lecturas", "escrituras", "escalas_terminos", "cotas_saturacion",
        "centrados_terminos", "z_lags", "saturado", "media_condicional",
    )
    for nombre in requeridos:
        if salida.internos.get(nombre) is None:
            raise KeyError(
                f"La salida no contiene '{nombre}' en `internos`; no puede "
                "completarse el control de calidad del Escenario J."
            )

    Psi = salida.internos["operador"]
    Psi4 = salida.internos["operador_rezago4"]
    w = salida.internos["pesos_cuadratura"]
    lecturas = salida.internos["lecturas"]
    escrituras = salida.internos["escrituras"]
    escalas = salida.internos["escalas_terminos"]
    cotas = salida.internos["cotas_saturacion"]
    centrados = salida.internos["centrados_terminos"]
    z_lags = salida.internos["z_lags"]
    saturado = salida.internos["saturado"]
    medias = salida.internos["media_condicional"]

    R, T, L = salida.curvas.shape

    # ── Parte lineal ────────────────────────────────────────────────────────
    hs1 = norma_hilbert_schmidt(Psi, w)
    hs4 = norma_hilbert_schmidt(Psi4, w)
    radio1 = float(np.max(np.abs(np.linalg.eigvals(Psi))))
    radio4 = float(np.max(np.abs(np.linalg.eigvals(Psi4))))

    # ── Razon efectiva y saturacion por rezago no lineal ───────────────────
    razon_efectiva = np.empty(3)
    for k in range(3):
        p = _GRADOS_NO_LINEALES[k]
        raw = _raw_k(z_lags[:, :, k].ravel(), p, centrados[k])
        sat = cotas[k] * np.tanh(raw / cotas[k])
        bruto = sat[:, None] * escrituras[k][None, :]
        sd_termino = float(np.sqrt(np.mean(np.sum(w * bruto ** 2, axis=1))))
        arrastre_lin = (salida.curvas - salida.media[None, None, :]).reshape(-1, L) @ Psi.T
        sd_lin = float(np.sqrt(np.mean(np.sum(w * arrastre_lin ** 2, axis=1))))
        razon_efectiva[k] = sd_termino / max(sd_lin, 1e-300)

    frac_saturada = saturado.reshape(-1, 3).mean(axis=0)

    # ── Mejor predictor lineal (4 rezagos) contra el oraculo ───────────────
    r2_lin, r2_orc, r2_lin_in = [], [], []
    for r in range(R):
        Y = salida.curvas[r] - salida.curvas[r].mean(axis=0, keepdims=True)
        Mc = medias[r] - medias[r].mean(axis=0, keepdims=True)
        _, U = _fpca_empirica(Y, w, cfg.n_dim_diagnostico)
        proy = w[:, None] * U
        S = Y @ proy               # (T, n_dim)
        Sm = Mc @ proy

        n_lags = 4
        n = T - n_lags
        D = np.column_stack(
            [np.ones(n)] + [S[n_lags - j - 1: T - j - 1] for j in range(n_lags)]
        )
        objetivo = S[n_lags:]
        oraculo = Sm[n_lags:]
        corte = n // 2

        coef, *_ = np.linalg.lstsq(D[:corte], objetivo[:corte], rcond=None)
        sce = float(np.sum((objetivo[corte:] - D[corte:] @ coef) ** 2))
        sct = float(np.sum((objetivo[corte:] - objetivo[:corte].mean(axis=0)) ** 2))
        r2_lin.append(1.0 - sce / max(sct, 1e-300))
        sco = float(np.sum((objetivo[corte:] - oraculo[corte:]) ** 2))
        r2_orc.append(1.0 - sco / max(sct, 1e-300))

        coef_in, *_ = np.linalg.lstsq(D, objetivo, rcond=None)
        r2_lin_in.append(
            1.0 - float(np.sum((objetivo - D @ coef_in) ** 2))
            / max(float(np.sum((objetivo - objetivo.mean(axis=0)) ** 2)), 1e-300)
        )
    r2_lineal = float(np.mean(r2_lin))
    r2_oraculo = float(np.mean(r2_orc))

    # ── Alineacion espectral de cada direccion de escritura ────────────────
    Y0 = salida.curvas[0] - salida.curvas[0].mean(axis=0, keepdims=True)
    lam0, U0 = _fpca_empirica(Y0, w, cfg.n_dim_diagnostico)
    frac_h_explicada = []
    for k in range(3):
        h = escrituras[k]
        coefs = (w * h) @ U0                # <h_k, u_j>_{L^2} para cada j retenida
        frac_h_explicada.append([float(c ** 2) for c in coefs])

    especifico = {
        # Parte lineal
        "hs_norm_rezago1_objetivo": float(cfg.hs_norm),
        "hs_norm_rezago1_efectiva": hs1,
        "hs_norm_rezago4_objetivo": float(cfg.hs_norm_4),
        "hs_norm_rezago4_efectiva": hs4,
        "hs_norm_suma": hs1 + hs4,
        "radio_espectral_rezago1": radio1,
        "radio_espectral_rezago4": radio4,
        "estacionariedad_garantizada": bool(hs1 + hs4 < 1.0),
        # Por rezago no lineal
        "grados_no_lineales": list(_GRADOS_NO_LINEALES),
        "razon_termino_objetivo": [float(x) for x in cfg.razon_terminos],
        "razon_termino_efectiva": [float(x) for x in razon_efectiva],
        "fraccion_saturada": [float(x) for x in frac_saturada],
        "cotas_saturacion": [float(x) for x in cotas],
        "centrados_terminos": [float(x) for x in centrados],
        "centros_lectura": [float(x) for x in cfg.centros_lectura],
        "centros_escritura": [float(x) for x in cfg.centros_escritura],
        # Lo que el escenario existe para medir
        "n_dim_diagnostico": int(cfg.n_dim_diagnostico),
        "n_lags_diagnostico": 4,
        "r2_lineal_fuera_de_muestra": r2_lineal,
        "r2_lineal_dentro_de_muestra": float(np.mean(r2_lin_in)),
        "r2_oraculo_fuera_de_muestra": r2_oraculo,
        "brecha_oraculo_lineal": float(r2_oraculo - r2_lineal),
        "razon_oraculo_lineal": (
            float(r2_oraculo / r2_lineal) if r2_lineal > 1e-6 else None
        ),
        # Alineacion espectral
        "varianza_fpca_diagnostico": [float(x) for x in lam0],
        "fraccion_h_explicada_por_componente": frac_h_explicada,
    }
    return {**base, **especifico}
