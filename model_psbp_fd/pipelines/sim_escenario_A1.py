"""
sim_escenario_A1.py
===================
Algoritmo A-1 del anexo (`docs/01 Anexo.tex`, seccion `ane_00_01_01_alg_a1`):
proceso ARFIMA(0, d, 0) de MEMORIA LARGA, segmentado en curvas.

    (1 - B)^d Z_s = zeta_s,      zeta_s ~ iid N(0, sigma_zeta^2),

y la curva t es el bloque contiguo Z_{(t-1)L+1}, ..., Z_{tL} leido sobre la
grilla {tau_l}. Es el primero de los "metodos basados en series de tiempo
clasicas": la dinamica NO se especifica sobre la curva sino sobre una serie
escalar, y la estructura funcional aparece al cortar el registro en sus
unidades naturales de observacion, igual que cuando un registro de alta
frecuencia se parte por dia.

Por que ARFIMA y no un ARMA
---------------------------
Con d en (0, 1/2) el proceso es estacionario y su autocorrelacion decae de
forma HIPERBOLICA,

    rho(k) ~ [Gamma(1-d)/Gamma(d)] k^{2d-1},

no geometrica. Lo que gobierna la dependencia entre puntos homologos de curvas
consecutivas es rho(L), que con decaimiento geometrico seria practicamente
cero para cualquier L razonable y aqui no lo es: con d = 0.4 y L = 75 vale
~0.28. Esa es toda la apuesta del escenario --dependencia entre curvas que no
se agota en el rezago 1-- y por eso `resumen_escenario_A1` la reporta.

La autocorrelacion se calcula con la recursion exacta de la ecuacion
`eq:ane_algA1_acf`,

    rho(0) = 1,   rho(k) = rho(k-1) (k-1+d)/(k-d),

que evita evaluar cocientes de Gamma con argumentos grandes, y la varianza de
la innovacion se despeja como sigma_zeta^2 = sigma_Z^2 Gamma(1-d)^2/Gamma(1-2d)
de modo que Var(Z_s) = sigma_Z^2 para todo d. Aqui esa cifra NO se usa para
simular --la simulacion va por las autocovarianzas-- y se reporta solo como
diagnostico: 1 - sigma_zeta^2/sigma_Z^2 es el R^2 del predictor optimo ESCALAR
con pasado infinito, el techo teorico de lo que el nivel escalar deja predecir.

Simulacion exacta (Davies--Harte)
---------------------------------
`simular_davies_harte` incrusta la matriz de covarianza de Toeplitz en una
circulante de orden m = 2(n-1) y la diagonaliza con la FFT. La trayectoria que
sale tiene EXACTAMENTE la autocovarianza pedida --no aproximadamente, como
truncar la representacion MA(inf)-- lo que importa aqui porque el escenario
vive de la cola de rho y truncarla seria borrar justo lo que se quiere medir.
El precio es que los autovalores de la circulante deben ser no negativos; para
ARFIMA(0,d,0) con 0 < d < 1/2 lo son, y la funcion falla si no.

Corolario practico: `burn_in` es COSMETICO en este escenario. La trayectoria ya
sale de la distribucion estacionaria, asi que descartar los primeros bloques no
corrige nada; se respeta porque es parte del contrato de `ConfigObservacion` y
porque deja la interfaz igual que en los demas generadores.

El oraculo de un rezago
-----------------------
Como el proceso es gaussiano, la esperanza del bloque t dado el bloque t-1 es
LINEAL y se calcula en forma cerrada:

    m_t = Sigma_21 Sigma_11^{-1} z_{t-1},

con Sigma_11 = Toeplitz(gamma(0..L-1)) y Sigma_21[i,j] = gamma(L + i - j). Eso
es `internos["media_condicional"]`, y es el oraculo EXACTO del problema que
resuelven los metodos del estudio, que predicen con `n_lags = 1`.

Ojo con leerlo como el techo absoluto: NO lo es. Con memoria larga los bloques
t-2, t-3, ... siguen aportando, de modo que el predictor de pasado completo es
estrictamente mejor. La distancia entre `r2_oraculo_1rezago` y
`r2_oraculo_pasado_infinito_escalar` mide cuanta informacion queda fuera por
mirar una sola curva atras; en este escenario es grande a proposito.

A-1 es un CONTROL NEGATIVO, y hay que decirlo
---------------------------------------------
El proceso es gaussiano y estacionario. Por lo tanto la ley del bloque t dado
el bloque t-1 es NORMAL, con media lineal en z_{t-1} y covarianza CONSTANTE:
no hay multimodalidad, ni asimetria, ni heterocedasticidad condicional. Todo
aquello que la mezcla probit del PSBPM-FD existe para capturar esta ausente
por construccion, y el FAR(1) sobre la curva esta CORRECTAMENTE ESPECIFICADO
para el problema de un rezago.

Lo que A-1 si mide es otra cosa: cuanto cuesta que el diseno del estudio mire
una sola curva atras cuando la dependencia es de largo alcance. Con d = 0.4 el
predictor de pasado completo explica bastante mas que el de un rezago, y esa
brecha --`r2_oraculo_pasado_infinito_escalar` contra `r2_oraculo_1rezago`-- es
comun a TODOS los metodos del estudio, que corren con `n_lags = 1`. Esperar
que el PSBPM-FD gane aqui es esperar que gane donde la clase lineal es exacta;
el resultado esperado es el empate, y como tal hay que presentarlo.

Lo que hay que declarar al reportar
-----------------------------------
* La curva NO es suave: es un tramo de una trayectoria escalar, y su rugosidad
  la fija d, no un nucleo. La base B-spline elegida por GCV suaviza parte de
  esa rugosidad y la trata como error de representacion.
* Con sigma_Z = 0.5 (valor del anexo) y sigma_obs = 0.25 la razon senal-ruido
  en varianza es 4. Es baja comparada con los escenarios funcionales, donde la
  media o la tendencia aportan amplitud; aqui no hay ninguna de las dos.
* No hay funcion media en el algoritmo del anexo. `media_fn` se admite y se
  suma a todas las curvas por igual, pero el valor por defecto --y el que usa
  la corrida 61-- es la media nula.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_factor, cho_solve, toeplitz

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

__all__ = [
    "ConfigEscenarioA1",
    "autocorrelacion_arfima",
    "varianza_innovacion_arfima",
    "simular_davies_harte",
    "generar_escenario_A1",
    "resumen_escenario_A1",
]


# ==========================================================================
# ESTRUCTURA DE COVARIANZA DEL ARFIMA(0, d, 0)
# ==========================================================================

def autocorrelacion_arfima(d: float, k_max: int) -> np.ndarray:
    """
    rho(0..k_max) por la recursion rho(k) = rho(k-1) (k-1+d)/(k-d).

    Es la ecuacion `eq:ane_algA1_acf` del anexo. Se itera en vez de evaluar la
    forma cerrada con Gammas porque para k grande esta ultima es un cociente de
    numeros enormes y pierde digitos justo en la cola, que es la parte del
    escenario que interesa.
    """
    if not 0.0 < d < 0.5:
        raise ValueError(f"d debe estar en (0, 1/2); recibido {d}.")
    if k_max < 0:
        raise ValueError("k_max no puede ser negativo.")
    k = np.arange(1, k_max + 1, dtype=float)
    rho = np.empty(k_max + 1)
    rho[0] = 1.0
    if k_max >= 1:
        rho[1:] = np.cumprod((k - 1.0 + d) / (k - d))
    return rho


def varianza_innovacion_arfima(d: float, sigma_Z: float) -> float:
    """sigma_zeta^2 = sigma_Z^2 Gamma(1-d)^2 / Gamma(1-2d), para Var(Z) = sigma_Z^2."""
    log_v = 2.0 * math.lgamma(1.0 - d) - math.lgamma(1.0 - 2.0 * d)
    return float(sigma_Z ** 2 * math.exp(log_v))


def simular_davies_harte(gamma: np.ndarray, n: int,
                         rng: np.random.Generator) -> np.ndarray:
    """
    Trayectoria gaussiana estacionaria EXACTA con autocovarianza `gamma`.

    gamma : autocovarianzas gamma(0), ..., gamma(n-1) (se ignora lo que sobre).
    n     : largo de la trayectoria pedida.

    Incrusta la Toeplitz en la circulante de primera fila
    [gamma(0), ..., gamma(n-1), gamma(n-2), ..., gamma(1)], de orden
    m = 2(n-1), y la diagonaliza con la FFT. Falla si algun autovalor sale
    negativo: eso significa que la incrustacion no es definida no negativa y el
    metodo no aplica a esa secuencia de covarianzas.
    """
    gamma = np.asarray(gamma, dtype=float)
    if gamma.size < n:
        raise ValueError(f"gamma necesita al menos {n} entradas; tiene {gamma.size}.")
    if n < 3:
        raise ValueError("n debe ser al menos 3.")
    g = gamma[:n]
    fila = np.concatenate([g, g[-2:0:-1]])        # (m,) con m = 2(n-1)
    m = fila.size
    lam = np.fft.fft(fila).real

    escala = float(np.abs(lam).max())
    lam_min = float(lam.min())
    if lam_min < -1e-8 * escala:
        raise ValueError(
            f"La incrustacion circulante tiene autovalores negativos "
            f"(min = {lam_min:.3e}): Davies-Harte no aplica a estas "
            "autocovarianzas.")
    lam = np.clip(lam, 0.0, None)

    mitad = m // 2
    Y = np.empty(m, dtype=complex)
    Y[0] = math.sqrt(lam[0]) * rng.standard_normal()
    Y[mitad] = math.sqrt(lam[mitad]) * rng.standard_normal()
    u = rng.standard_normal(mitad - 1)
    v = rng.standard_normal(mitad - 1)
    Y[1:mitad] = np.sqrt(lam[1:mitad] / 2.0) * (u + 1j * v)
    Y[mitad + 1:] = np.conj(Y[1:mitad][::-1])

    return (np.fft.fft(Y).real / math.sqrt(m))[:n]


# ==========================================================================
# CONFIGURACION
# ==========================================================================

@dataclass
class ConfigEscenarioA1(ConfigObservacion):
    """
    Algoritmo A-1: ARFIMA(0, d, 0) segmentado en curvas de largo L.

    d : parametro de memoria larga, en (0, 1/2). Valor del anexo: 0.4.
    sigma_Z : desviacion estandar MARGINAL de Z_s. Valor del anexo: 0.5. La
        varianza de la innovacion se despeja de ella, de modo que cambiar d no
        cambia la escala de las curvas.
    prop_train_referencia : solo para el diagnostico; el corte T0 real lo fija
        el notebook.
    """

    d: float = 0.4
    sigma_Z: float = 0.5
    prop_train_referencia: float = 0.70

    def validar(self) -> None:
        super().validar()
        if not 0.0 < self.d < 0.5:
            raise ValueError(f"d debe estar en (0, 1/2); recibido {self.d}.")
        if self.sigma_Z <= 0:
            raise ValueError("sigma_Z debe ser positivo.")
        if not 0.0 < self.prop_train_referencia < 1.0:
            raise ValueError("prop_train_referencia debe estar en (0, 1).")


# ==========================================================================
# GENERADOR
# ==========================================================================

def _oraculo_un_rezago(gamma: np.ndarray, L: int, jitter: float
                       ) -> tuple[np.ndarray, np.ndarray]:
    """
    Operador del predictor optimo del bloque t dado el bloque t-1, y la
    covarianza de su residuo.

    Retorna (A, Sigma_resid) con A = Sigma_21 Sigma_11^{-1}, de forma (L, L).
    """
    S22 = toeplitz(gamma[:L])                                  # (L, L)
    S11 = S22 + jitter * max(gamma[0], 1.0) * np.eye(L)
    idx = np.arange(L)
    S21 = gamma[np.abs(L + idx[:, None] - idx[None, :])]       # (L, L)
    c, low = cho_factor(S11, lower=True)
    A = cho_solve((c, low), S21.T).T                           # Sigma_21 S11^{-1}
    return A, S22 - A @ S21.T


def generar_escenario_A1(cfg: ConfigEscenarioA1,
                         diagnosticar: bool = True) -> SalidaSimulacion:
    """
    Genera R replicas del Algoritmo A-1.

    Cada replica simula UNA trayectoria escalar de largo (burn_in + T) * L por
    Davies-Harte, descarta los primeros `burn_in` bloques y corta el resto en T
    bloques contiguos sin traslape.
    """
    cfg.validar()
    L, T, R = int(cfg.L), int(cfg.T), int(cfg.R)
    burn = int(cfg.burn_in)
    n_bloques = burn + T
    n_total = n_bloques * L

    tau = grilla_regular(L)
    media = evaluar_media(cfg.media_fn, tau)

    # gamma(k) para k = 0..n_total-1: la cola completa, sin truncar.
    gamma = cfg.sigma_Z ** 2 * autocorrelacion_arfima(cfg.d, n_total - 1)
    A_orac, S_resid = _oraculo_un_rezago(gamma, L, cfg.jitter)

    hijas, registro = semillas_replicas(cfg.seed, R)
    curvas = np.empty((R, T, L))
    observaciones = np.empty((R, T, L))
    medias_cond = np.empty((R, T, L))
    series = np.empty((R, n_total))

    for r, hija in enumerate(hijas):
        rng = np.random.default_rng(hija)
        Z = simular_davies_harte(gamma, n_total, rng)
        todos = Z.reshape(n_bloques, L)
        bloques = todos[burn:]                                  # (T, L)
        # Predictor del bloque t a partir del t-1. Con burn_in > 0 el bloque
        # anterior al primero retenido EXISTE y se usa; sin calentamiento el
        # primer origen no tiene predictor y va en cero.
        prev = (todos[burn - 1:-1] if burn > 0
                else np.vstack([np.zeros((1, L)), bloques[:-1]]))

        curvas[r] = bloques + media
        medias_cond[r] = prev @ A_orac.T + media
        observaciones[r] = aplicar_ruido_observacion(curvas[r], cfg.sigma_obs, rng)
        series[r] = Z

    salida = SalidaSimulacion(
        observaciones=observaciones,
        curvas=curvas,
        grilla=tau,
        media=media,
        semillas=registro,
        config=cfg,
        internos={
            "serie_escalar": series,
            "autocovarianzas": gamma[:max(2 * L + 1, 3)],
            "operador_oraculo": A_orac,
            "cov_residuo_oraculo": S_resid,
            "media_condicional": medias_cond,
        },
    )
    if diagnosticar:
        salida.diagnostico = resumen_escenario_A1(salida)
    return salida


# ==========================================================================
# CONTROL DE CALIDAD
# ==========================================================================

def resumen_escenario_A1(salida: SalidaSimulacion) -> dict:
    """
    Control de calidad propio del Algoritmo A-1.

    Ademas del diagnostico comun verifica lo que DEFINE al escenario: la
    autocorrelacion empirica de la serie escalar contra la teorica en los
    rezagos que importan (1, L y 2L, es decir dentro de la curva, entre curvas
    consecutivas y a dos curvas de distancia), el R^2 del oraculo de un rezago
    y su distancia contra el techo escalar de pasado infinito.
    """
    cfg = salida.config
    L, T = int(cfg.L), int(cfg.T)
    gamma = salida.internos["autocovarianzas"]
    rho = gamma / gamma[0]
    S_resid = salida.internos["cov_residuo_oraculo"]
    Z = salida.internos["serie_escalar"]

    def acf_emp(k: int) -> float:
        vals = []
        for r in range(Z.shape[0]):
            z = Z[r] - Z[r].mean()
            den = float((z * z).sum())
            vals.append(float((z[:-k] * z[k:]).sum() / den) if den > 0 else np.nan)
        return float(np.nanmean(vals))

    sigma_zeta2 = varianza_innovacion_arfima(cfg.d, cfg.sigma_Z)
    r2_1rezago = float(1.0 - np.trace(S_resid) / (L * gamma[0]))

    # Norma HS del oraculo LEIDO COMO OPERADOR INTEGRAL en L^2. La matriz A
    # actua punto a punto, m(tau_i) = sum_j A_ij z(tau_j); el nucleo que le
    # corresponde bajo la cuadratura es kappa_ij = A_ij / w_j, y de ahi
    #     ||K||_HS^2 = sum_ij w_i w_j kappa_ij^2 = sum_ij w_i A_ij^2 / w_j.
    # Es la unica cifra de este escenario comparable con el ||Psi||_HS que el
    # _05 contrasta contra el operador estimado por el FAR, y por eso viaja en
    # el diagnostico hasta simulation_config.json.
    w_cuad = pesos_trapezoidales(salida.grilla)
    A = salida.internos["operador_oraculo"]
    hs2 = float(np.sum((w_cuad[:, None] / w_cuad[None, :]) * A ** 2))

    # El mismo R^2, medido sobre las curvas generadas: es la cifra que un
    # metodo con n_lags = 1 puede aspirar a igualar, y no mas.
    X = salida.curvas - salida.media
    Mc = salida.internos["media_condicional"] - salida.media
    sse = float(((X[:, 1:] - Mc[:, 1:]) ** 2).sum())
    sst = float((X[:, 1:] ** 2).sum())

    return {
        **diagnostico_comun(salida),
        "d": float(cfg.d),
        "sigma_Z": float(cfg.sigma_Z),
        "sigma_zeta2_teorica": float(sigma_zeta2),
        "rho_teorica_lag1": float(rho[1]),
        "rho_teorica_lagL": float(rho[L]),
        "rho_teorica_lag2L": float(rho[min(2 * L, rho.size - 1)]),
        "rho_empirica_lag1": acf_emp(1),
        "rho_empirica_lagL": acf_emp(L),
        "rho_empirica_lag2L": acf_emp(2 * L),
        "r2_oraculo_1rezago": r2_1rezago,
        "r2_oraculo_1rezago_empirico": (float(1.0 - sse / sst) if sst > 0
                                        else float("nan")),
        "r2_oraculo_pasado_infinito_escalar": float(
            1.0 - sigma_zeta2 / cfg.sigma_Z ** 2),
        "hs_oraculo_L2": float(np.sqrt(hs2)),
        "sd_Z_empirica": float(Z.std()),
        "T0_referencia": int(np.floor(cfg.prop_train_referencia * T)),
        "burn_in_bloques": int(cfg.burn_in),
        "nota_burn_in": ("Davies-Harte es exacto: burn_in no corrige nada, se "
                         "respeta por contrato de ConfigObservacion."),
        "todo_finito": bool(np.all(np.isfinite(salida.curvas))),
    }
