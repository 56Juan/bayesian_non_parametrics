"""
sim_escenario_A2.py
===================
Algoritmo A-2 del anexo (`docs/01 Anexo.tex`, seccion `ane_00_01_02_alg_a2`):
proceso AUTORREGRESIVO CON CAMBIO DE REGIMEN MARKOVIANO, segmentado en curvas.

    Z_s = m_{S_s} + u_s,      u_s = phi u_{s-1} + sigma_{S_s} eps_s,
    eps_s ~ iid N(0, 1),      u_0 = 0,

con {S_s} una cadena de Markov de dos estados, matriz simetrica
P = [[p, 1-p], [1-p, p]], S_1 extraido de la estacionaria (1/2, 1/2) y
m_1 = -m, m_2 = +m, sigma_1 < sigma_2. La curva t es el bloque contiguo
Z_{(t-1)L+1}, ..., Z_{tL}. El mecanismo de segmentacion es comun a A-1, A-2 y
A-3 y vive en `sim_series_clasicas`; este modulo aporta solo el proceso escalar.

Que hace interesante al escenario
---------------------------------
Con p = 0.998 cada regimen dura en promedio 1/(1-p) = 500 observaciones, es
decir 500/L ~ 6.7 curvas con L = 75. Dos consecuencias:

* La dependencia entre curvas consecutivas es el NIVEL: dentro de un regimen la
  curva oscila alrededor de +-m, y esa persistencia es lo que el rezago 1
  puede aprovechar.
* La ley condicional de la curva t dado la t-1 es una MEZCLA. Con probabilidad
  ~ p^(L-1) = 0.86 el regimen no cambia a lo largo de la curva; con la restante
  aparece un salto de nivel de 2m dentro de ella (o la curva anterior ya termino
  cerca de un cambio). Ahi la predictiva es bimodal y su media --la unica cifra
  que un metodo puntual reporta-- cae en medio de dos modos. Es el caso en que la
  mezcla probit del PSBPM-FD tiene algo que un FAR gaussiano no puede
  representar.

La cadena latente {S_s} NO se entrega a la estimacion: vive en
`internos["regimen"]` y solo sirve de control de calidad.

Simulacion
----------
Vectorizada y exacta. La cadena es S_s = S_1 xor (paridad de los cambios hasta
s), con los cambios iid Bernoulli(1-p): equivale a la simulacion fila a fila de
P y evita un bucle de largo (burn_in + T) L. u es la salida del filtro
1/(1 - phi B) aplicado a sigma_{S_s} eps_s con estado inicial cero. Orden de
consumo del generador, que fija las trayectorias con una semilla dada:
S_1, los cambios (n-1 uniformes) y los eps (n normales).

Los oraculos de un rezago
-------------------------
La media condicional de la curva t dado la t-1 NO es lineal. Se guardan dos
cantidades, y la distancia entre ellas es lo que el escenario permite medir:

(1) `media_condicional` --el oraculo de BAYES de un rezago--. Se filtra el
    regimen con las 75 observaciones del bloque t-1 (filtro de Hamilton, exacto
    para este modelo: dado S_{s-1} = i la media de Z_s es
    m_k + phi (Z_{s-1} - m_i), de modo que basta propagar P(S_{s-1} | Z_{<=s-1})
    por el par de estados). Con pi = P(S_last | bloque t-1), la esperanza a j
    pasos es en forma CERRADA:

        E[Z_{last+j}] = lam^j m_bar + phi^j (Z_last - m_bar),
        m_bar = m (pi_2 - pi_1),   lam = 2p - 1,

    porque E[u_{s+j} | S_s = i, u_s] = phi^j u_s y sum_k P^j(i,k) m_k = lam^j m_i.
    Es el predictor optimo cuando lo unico observado es UNA curva atras, el
    problema que resuelven los metodos del estudio con n_lags = 1. Aproximacion:
    la primera observacion del bloque se trata con la varianza estacionaria del
    AR dentro del regimen, sigma_k^2/(1-phi^2), porque el estado u previo al
    bloque no se observa. Es un detalle del primer paso; el resto del filtro es
    exacto. Verificado numericamente con los parametros del anexo: filtrar toda
    la serie desde s = 1 (todo el pasado) da EXACTAMENTE la misma media, porque
    con niveles +-1.5 y escalas <= 1 el regimen queda identificado con
    probabilidad 1 (a precision de maquina) dentro de una sola curva. Por eso no
    se guarda un oraculo "de pasado completo": es identico.

(2) `media_condicional_lineal` --el mejor predictor LINEAL por momentos--. Como
    la serie es debilmente estacionaria con
        gamma(k) = m^2 (2p-1)^k + phi^k * sigma_bar^2/(1-phi^2),
    sigma_bar^2 = (sigma_1^2 + sigma_2^2)/2 (los terminos cruzados se anulan
    porque eps es independiente de la cadena y de media cero), la proyeccion
    Sigma_21 Sigma_11^{-1} es cerrada. Es el techo de lo que puede el FAR(1).

`internos["media_condicional"]` es el (1). No hay `hs_oraculo_L2`: la media
condicional es no lineal y no es un operador, y el mejor operador lineal (2) es
casi una evaluacion puntual en el ultimo instante (su nucleo integral no existe
en el continuo), de modo que su norma de Hilbert-Schmidt depende de la grilla y
no es comparable con la del FAR estimado sobre una base B-spline.

Lo que hay que declarar al reportar
-----------------------------------
* La curva NO es suave: es un tramo de un AR(1) con phi = 0.5 mas el nivel del
  regimen. La base B-spline por GCV trata parte de esa rugosidad como error de
  representacion.
* La razon senal-ruido es alta (Var(Z) = m^2 + sigma_bar^2/(1-phi^2) ~ 3.1 con
  sigma_obs = 0.25) y casi toda la varianza es NIVEL de regimen, no dinamica
  dentro de la curva: una sola componente FPCA ya la captura.
* Con R = 1 el numero de cambios de regimen en el bloque de prueba (~ 240/6.7)
  es del orden de 35: los episodios de mezcla son pocos y la comparacion entre
  modelos descansa en ellos.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import lfilter

from .sim_comun import ConfigObservacion, SalidaSimulacion
from .sim_series_clasicas import (
    diagnostico_serie_escalar,
    generar_serie_segmentada,
    oraculo_lineal_un_rezago,
    r2_empirico_media_condicional,
)

__all__ = [
    "ConfigEscenarioA2",
    "simular_cambio_regimen",
    "autocovarianza_cambio_regimen",
    "filtrar_regimen",
    "media_condicional_regimen",
    "generar_escenario_A2",
    "resumen_escenario_A2",
]


# ==========================================================================
# CONFIGURACION
# ==========================================================================

@dataclass
class ConfigEscenarioA2(ConfigObservacion):
    """
    Algoritmo A-2: AR con cambio de regimen markoviano de dos estados.

    p : probabilidad de permanencia en el regimen (matriz P simetrica). 0.998.
    m : desplazamiento de nivel, m_1 = -m y m_2 = +m. 1.5.
    sigma1, sigma2 : escala de la innovacion en cada regimen. 0.5 y 1.0.
    phi : coeficiente autorregresivo comun, |phi| < 1. 0.5.
    prop_train_referencia : solo para el diagnostico; el corte T0 real lo fija
        el notebook.

    Los valores por defecto son los del Cuadro `tab:ane_algA2` del anexo.
    """

    p: float = 0.998
    m: float = 1.5
    sigma1: float = 0.5
    sigma2: float = 1.0
    phi: float = 0.5
    prop_train_referencia: float = 0.70

    def validar(self) -> None:
        super().validar()
        if not 0.5 < self.p < 1.0:
            raise ValueError(f"p debe estar en (1/2, 1); recibido {self.p}.")
        if self.m < 0:
            raise ValueError("m no puede ser negativo.")
        if self.sigma1 <= 0 or self.sigma2 <= 0:
            raise ValueError("sigma1 y sigma2 deben ser positivos.")
        if not abs(self.phi) < 1.0:
            raise ValueError(f"|phi| debe ser menor que 1; recibido {self.phi}.")
        if not 0.0 < self.prop_train_referencia < 1.0:
            raise ValueError("prop_train_referencia debe estar en (0, 1).")

    @property
    def niveles(self) -> np.ndarray:
        """(m_1, m_2) = (-m, +m)."""
        return np.array([-self.m, self.m])

    @property
    def escalas(self) -> np.ndarray:
        """(sigma_1, sigma_2)."""
        return np.array([self.sigma1, self.sigma2])


# ==========================================================================
# PROCESO ESCALAR
# ==========================================================================

def simular_cambio_regimen(cfg: ConfigEscenarioA2, n: int,
                           rng: np.random.Generator
                           ) -> tuple[np.ndarray, np.ndarray]:
    """
    Trayectoria de A-2 de largo n. Retorna (Z, S) con S en {1, 2}.

    Consumo del generador: S_1 (1 uniforme), los n-1 cambios (uniformes) y los
    n eps (normales), en ese orden.
    """
    s1 = int(rng.random() >= 0.5)                       # estado 0 o 1
    cambios = rng.random(n - 1) < (1.0 - cfg.p)
    S = (s1 + np.concatenate([[0], np.cumsum(cambios)])) % 2    # (n,) en {0, 1}
    eps = rng.standard_normal(n)
    u = lfilter([1.0], [1.0, -cfg.phi], cfg.escalas[S] * eps)   # u_0 = 0
    Z = cfg.niveles[S] + u
    return Z, (S + 1).astype(float)


def autocovarianza_cambio_regimen(cfg: ConfigEscenarioA2, k_max: int
                                  ) -> np.ndarray:
    """
    gamma(0..k_max) estacionaria de Z:

        gamma(k) = m^2 (2p-1)^k + phi^k sigma_bar^2 / (1 - phi^2).

    El termino de nivel usa el segundo autovalor de P, lam = 2p-1, y el del AR
    el hecho de que eps sea independiente de la cadena y de media cero: Cov(u_s,
    u_{s+k}) = phi^k E[u_s^2] con E[u_s^2] = sigma_bar^2/(1-phi^2), y el nivel
    no correlaciona con u.
    """
    k = np.arange(k_max + 1, dtype=float)
    lam = 2.0 * cfg.p - 1.0
    sig2 = 0.5 * (cfg.sigma1 ** 2 + cfg.sigma2 ** 2)
    return (cfg.m ** 2 * lam ** k
            + cfg.phi ** k * sig2 / (1.0 - cfg.phi ** 2))


# ==========================================================================
# FILTRO DE REGIMEN (HAMILTON) Y ORACULOS DE BAYES
# ==========================================================================

def _log_normal(z, media, sd):
    return -0.5 * ((z - media) / sd) ** 2 - np.log(sd) - 0.5 * np.log(2 * np.pi)


def filtrar_regimen(Z: np.ndarray, cfg: ConfigEscenarioA2,
                    inicio_exacto: bool) -> np.ndarray:
    """
    Filtro hacia adelante de P(S_s = k | Z_{<=s}) sobre trayectorias en filas.

    Z : (B, n). Cada fila se filtra por separado y en paralelo.
    inicio_exacto : True si la fila arranca en s = 1 de la serie (u_0 = 0, la
        primera observacion es N(m_k, sigma_k^2)); False si arranca a mitad de
        serie, donde se usa la varianza estacionaria del AR dentro del regimen,
        N(m_k, sigma_k^2/(1-phi^2)).

    Retorna (B, n, 2). La recursion es exacta: dado S_{s-1} = i, la media de Z_s
    es m_k + phi (Z_{s-1} - m_i), de modo que P(S_{s-1} | Z_{<=s-1}) es
    estadistico suficiente para el paso siguiente.
    """
    Z = np.atleast_2d(Z)
    B, n = Z.shape
    m, sg, phi, p = cfg.niveles, cfg.escalas, cfg.phi, cfg.p
    P = np.array([[p, 1.0 - p], [1.0 - p, p]])
    sd0 = sg if inicio_exacto else sg / np.sqrt(1.0 - phi ** 2)

    out = np.empty((B, n, 2))
    lw = _log_normal(Z[:, [0]], m[None, :], sd0[None, :]) + np.log(0.5)
    lw -= lw.max(axis=1, keepdims=True)
    f = np.exp(lw)
    f /= f.sum(axis=1, keepdims=True)
    out[:, 0] = f

    for s in range(1, n):
        z, zp = Z[:, s], Z[:, s - 1]
        # ll[b, i, k] = log N(z; m_k + phi (zp - m_i), sigma_k^2)
        mu = m[None, None, :] + phi * (zp[:, None, None] - m[None, :, None])
        ll = _log_normal(z[:, None, None], mu, sg[None, None, :])
        w = np.log(np.clip(f[:, :, None] * P[None], 1e-300, None)) + ll
        w -= w.max(axis=(1, 2), keepdims=True)
        w = np.exp(w)
        w /= w.sum(axis=(1, 2), keepdims=True)
        f = w.sum(axis=1)                                    # marginal en k
        out[:, s] = f
    return out


def media_condicional_regimen(pi_last: np.ndarray, z_last: np.ndarray,
                              cfg: ConfigEscenarioA2, L: int) -> np.ndarray:
    """
    E[Z_{last+j}] para j = 1..L dados pi = P(S_last | pasado) y Z_last:

        lam^j m_bar + phi^j (Z_last - m_bar),   m_bar = m (pi_2 - pi_1).

    pi_last : (N, 2) ; z_last : (N,). Retorna (N, L).
    """
    lam = 2.0 * cfg.p - 1.0
    m_bar = cfg.m * (pi_last[:, 1] - pi_last[:, 0])            # (N,)
    j = np.arange(1, L + 1, dtype=float)[None, :]
    return lam ** j * m_bar[:, None] + cfg.phi ** j * (z_last - m_bar)[:, None]


# ==========================================================================
# GENERADOR
# ==========================================================================

def generar_escenario_A2(cfg: ConfigEscenarioA2,
                         diagnosticar: bool = True) -> SalidaSimulacion:
    """
    Genera R replicas del Algoritmo A-2.

    Cada replica simula la cadena y el AR sobre (burn_in + T) L pasos, descarta
    los primeros `burn_in` bloques y corta el resto en T bloques contiguos. Los
    dos oraculos de un rezago se describen en el encabezado del modulo.
    """
    cfg.validar()
    L = int(cfg.L)
    n_total = (int(cfg.burn_in) + int(cfg.T)) * L

    gamma = autocovarianza_cambio_regimen(cfg, max(2 * L, L))
    A_lin, S_resid = oraculo_lineal_un_rezago(gamma, L, cfg.jitter)

    def simulador(rng, n):
        Z, S = simular_cambio_regimen(cfg, n, rng)
        return Z, {"regimen": S}

    def oraculos(prev, Z, lat, ctx):
        # Bayes de un rezago: se filtra solo el bloque anterior.
        f1 = filtrar_regimen(prev, cfg, inicio_exacto=False)[:, -1, :]
        return {
            "media_condicional": media_condicional_regimen(f1, prev[:, -1], cfg, L),
            "media_condicional_lineal": prev @ A_lin.T,
        }

    return generar_serie_segmentada(
        cfg, simulador,
        oraculos=oraculos,
        internos_extra={
            "autocovarianzas": gamma,
            "operador_oraculo_lineal": A_lin,
            "cov_residuo_oraculo_lineal": S_resid,
        },
        resumen=resumen_escenario_A2 if diagnosticar else None,
        diagnosticar=diagnosticar,
    )


# ==========================================================================
# CONTROL DE CALIDAD
# ==========================================================================

def resumen_escenario_A2(salida: SalidaSimulacion) -> dict:
    """
    Control de calidad propio del Algoritmo A-2.

    Ademas del diagnostico comun verifica lo que DEFINE al escenario: la
    autocorrelacion empirica contra la teorica, la persistencia y el numero de
    los cambios de regimen en el tramo retenido, la fraccion de curvas que
    contienen un cambio (las de ley condicional en mezcla), y los R^2 de los dos
    oraculos de un rezago.
    """
    cfg = salida.config
    L, T = int(cfg.L), int(cfg.T)
    burn = int(cfg.burn_in)
    gamma = salida.internos["autocovarianzas"]
    rho = gamma / gamma[0]
    S_resid = salida.internos["cov_residuo_oraculo_lineal"]
    S = salida.internos["regimen"][:, burn * L:]              # (R, T L)
    diag = diagnostico_serie_escalar(salida)

    n_cambios = (np.diff(S, axis=1) != 0).sum(axis=1)          # (R,)
    # Curvas con salto de nivel INTERNO (los cambios en el limite entre bloques
    # no cuentan): son las de ley condicional en mezcla.
    Sb = S.reshape(S.shape[0], T, L)
    con_cambio = (Sb != Sb[:, :, :1]).any(axis=2)

    r2_lin_teorico = float(1.0 - np.trace(S_resid) / (L * gamma[0]))
    return {
        **diag,
        "p": float(cfg.p), "m": float(cfg.m),
        "sigma1": float(cfg.sigma1), "sigma2": float(cfg.sigma2),
        "phi": float(cfg.phi),
        "duracion_media_regimen_obs": float(1.0 / (1.0 - cfg.p)),
        "duracion_media_regimen_curvas": float(1.0 / ((1.0 - cfg.p) * L)),
        "prob_curva_sin_cambio_teorica": float(cfg.p ** (L - 1)),
        "frac_curvas_con_cambio_empirica": float(con_cambio.mean()),
        "n_cambios_regimen_retenidos": float(n_cambios.mean()),
        "frac_tiempo_regimen_2": float((S == 2).mean()),
        "rho_teorica_lag1": float(rho[1]),
        "rho_teorica_lagL": float(rho[L]),
        "rho_teorica_lag2L": float(rho[min(2 * L, rho.size - 1)]),
        "r2_oraculo_1rezago_lineal": r2_lin_teorico,
        "r2_oraculo_1rezago_lineal_empirico": r2_empirico_media_condicional(
            salida, "media_condicional_lineal"),
        "r2_oraculo_1rezago": r2_empirico_media_condicional(salida),
        "r2_oraculo_1rezago_empirico": r2_empirico_media_condicional(salida),
        "nota_media_condicional": (
            "media_condicional = Bayes de un rezago (filtro de regimen sobre el "
            "bloque t-1); NO es lineal. Sin hs_oraculo_L2: no hay operador."),
    }
