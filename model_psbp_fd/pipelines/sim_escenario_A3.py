"""
sim_escenario_A3.py
===================
Algoritmo A-3 del anexo (`docs/01 Anexo.tex`, seccion `ane_00_01_03_alg_a3`):
proceso AR(1) con innovaciones GARCH(1,1), segmentado en curvas.

    Z_s = phi Z_{s-1} + a_s,      a_s = sqrt(h_s) z_s,      z_s ~ iid N(0, 1),
    h_s = alpha_0 + alpha_1 a_{s-1}^2 + beta_1 h_{s-1},
    alpha_0 = sigma_a^2 (1 - alpha_1 - beta_1),    h_1 = sigma_a^2,   Z_0 = 0.

La curva t es el bloque contiguo Z_{(t-1)L+1}, ..., Z_{tL}. El mecanismo de
segmentacion es comun a A-1, A-2 y A-3 y vive en `sim_series_clasicas`; este
modulo aporta solo el proceso escalar. Es el analogo escalar del Algoritmo 2 del
anexo: la dependencia entre curvas opera sobre la ESCALA, no sobre el nivel. La
serie {h_s} NO se entrega a la estimacion: vive en `internos["h"]`.

Que hace interesante al escenario, y su limite
----------------------------------------------
Como a_s es una diferencia de martingala, la media condicional de Z es
LINEAL y no depende de h:

    E[Z_{last+j} | pasado] = phi^j Z_last.

Con phi = 0.5 el centro de la curva t dado la t-1 es entonces una cola
geometrica que se apaga en pocos puntos (phi^5 ~ 0.03) y la curva completa es,
salvo sus primeros instantes, ruido. El R^2 de ese oraculo es

    (1/L) sum_{j=1}^{L} phi^{2j} ~ phi^2 / ((1 - phi^2) L) = 0.0044 con L = 75,

es decir, el CENTRO no tiene practicamente nada que predecir a nivel de curva.
Lo que si es predecible es la ESCALA: con alpha_1 + beta_1 = 0.98 la varianza
condicional persiste del orden de 1/(1-0.98) = 50 pasos y la ley condicional de
la curva t dado el pasado es una normal de media phi^j Z_last y varianza
V_j = sum_{i=1}^{j} phi^{2(j-i)} E[h_{last+i}] que VARIA entre curvas
(`internos["sd_condicional_pasado_completo"]`). Lo unico que este escenario
permite ganar es la banda (cobertura, MPIW, Winkler), no el centro; un
resultado puntual empatado es lo esperado, y como tal hay que presentarlo.

Ese diagnostico se agrava con la representacion: la dependencia entre curvas
vive en los primeros puntos de la curva y la base B-spline por GCV, que suaviza
ruido de alta frecuencia, tiende a eliminarla. Es un limite del esquema
"curva = bloque" con phi pequeno, no del modelo, y hay que declararlo.

Simulacion
----------
Los z_s se extraen de una vez (n normales, unico consumo del generador), h y a se
calculan con un bucle escalar (la recursion GARCH es intrinsecamente secuencial)
y Z es la salida del filtro 1/(1 - phi B) aplicado a a con Z_0 = 0. A diferencia
de A-1 y A-2, aqui `burn_in` no es cosmetico: h_1 = sigma_a^2 y Z_0 = 0 son
condiciones iniciales, y la memoria de h (~50 pasos) queda muy por debajo de
`burn_in * L`.

El oraculo
----------
`internos["media_condicional"]` = phi^j Z_last, j = 1..L: la media condicional
EXACTA, dado el bloque t-1 y tambien dado todo el pasado, y la del mejor
predictor lineal. No hay que elegir entre oraculos: coinciden. No hay
`hs_oraculo_L2`: el operador es una evaluacion puntual en tau = 1 (nucleo delta),
que no es de Hilbert-Schmidt en el continuo, y su norma sobre la grilla crece
con L; no es comparable con la del FAR sobre una base B-spline.

Lo que hay que declarar al reportar
-----------------------------------
* R^2 del oraculo ~ 0.4 %: el centro es casi impredecible por construccion.
* La curva NO es suave: es un tramo de un AR(1) con phi = 0.5 y varianza
  variable; la razon senal-ruido efectiva es Var(Z)/sigma_obs^2 = 21 en promedio
  pero es heterogenea entre curvas.
* `sd_condicional_pasado_completo` usa h_{last+1}, que el bloque t-1 solo no
  revela; es la banda del oraculo con TODO el pasado, no alcanzable con
  n_lags = 1.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.signal import lfilter

from .sim_comun import ConfigObservacion, SalidaSimulacion
from .sim_series_clasicas import (
    acf_empirica_serie,
    diagnostico_serie_escalar,
    generar_serie_segmentada,
    r2_empirico_media_condicional,
)

__all__ = [
    "ConfigEscenarioA3",
    "simular_ar_garch",
    "sd_condicional_bloque",
    "generar_escenario_A3",
    "resumen_escenario_A3",
]


# ==========================================================================
# CONFIGURACION
# ==========================================================================

@dataclass
class ConfigEscenarioA3(ConfigObservacion):
    """
    Algoritmo A-3: AR(1) con innovaciones GARCH(1,1).

    phi : coeficiente autorregresivo, |phi| < 1. 0.5.
    alpha1 : sensibilidad de la varianza futura a la innovacion reciente. 0.08.
    beta1 : persistencia de la varianza condicional pasada. 0.90.
    sigma_a2 : varianza INCONDICIONAL de la innovacion. 1.0. El nivel base se
        despeja de ella, alpha_0 = sigma_a2 (1 - alpha1 - beta1).
    prop_train_referencia : solo para el diagnostico; el corte T0 real lo fija
        el notebook.

    Los valores por defecto son los del Cuadro `tab:ane_algA3` del anexo.
    """

    phi: float = 0.5
    alpha1: float = 0.08
    beta1: float = 0.90
    sigma_a2: float = 1.0
    prop_train_referencia: float = 0.70

    def validar(self) -> None:
        super().validar()
        if not abs(self.phi) < 1.0:
            raise ValueError(f"|phi| debe ser menor que 1; recibido {self.phi}.")
        if self.alpha1 < 0 or self.beta1 < 0:
            raise ValueError("alpha1 y beta1 no pueden ser negativos.")
        if self.alpha1 + self.beta1 >= 1.0:
            raise ValueError("Estacionariedad de segundo orden: se requiere "
                             f"alpha1 + beta1 < 1; recibido "
                             f"{self.alpha1 + self.beta1}.")
        if self.sigma_a2 <= 0:
            raise ValueError("sigma_a2 debe ser positivo.")
        if not 0.0 < self.prop_train_referencia < 1.0:
            raise ValueError("prop_train_referencia debe estar en (0, 1).")

    @property
    def alpha0(self) -> float:
        return float(self.sigma_a2 * (1.0 - self.alpha1 - self.beta1))

    @property
    def persistencia(self) -> float:
        return float(self.alpha1 + self.beta1)


# ==========================================================================
# PROCESO ESCALAR
# ==========================================================================

def simular_ar_garch(cfg: ConfigEscenarioA3, n: int,
                     rng: np.random.Generator
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Trayectoria de A-3 de largo n. Retorna (Z, h, a).

    Consumo del generador: los n z_s (normales) de una vez.
    """
    z = rng.standard_normal(n)
    a0, a1, b1 = cfg.alpha0, cfg.alpha1, cfg.beta1
    h = np.empty(n)
    a = np.empty(n)
    h_s = cfg.sigma_a2                                  # h_1 = sigma_a^2
    a_prev = 0.0
    for s in range(n):
        if s > 0:
            h_s = a0 + a1 * a_prev * a_prev + b1 * h_s
        h[s] = h_s
        a_prev = math.sqrt(h_s) * z[s]
        a[s] = a_prev
    Z = lfilter([1.0], [1.0, -cfg.phi], a)              # Z_0 = 0
    return Z, h, a


def sd_condicional_bloque(h_siguiente: np.ndarray, cfg: ConfigEscenarioA3,
                          L: int) -> np.ndarray:
    """
    Desviacion condicional de Z_{last+j}, j = 1..L, dado el pasado hasta `last`.

    h_siguiente : (N,) h_{last+1}, medible respecto del pasado. Como
        E[h_{last+i}] = sigma_a^2 + (alpha_1+beta_1)^{i-1} (h_{last+1} - sigma_a^2),
    la varianza sale de la recursion V_j = phi^2 V_{j-1} + E[h_{last+j}].
    Retorna (N, L).
    """
    rho = cfg.persistencia
    V = np.zeros_like(h_siguiente, dtype=float)
    out = np.empty((h_siguiente.size, L))
    exceso = h_siguiente - cfg.sigma_a2
    for j in range(L):
        V = cfg.phi ** 2 * V + cfg.sigma_a2 + rho ** j * exceso
        out[:, j] = np.sqrt(V)
    return out


# ==========================================================================
# GENERADOR
# ==========================================================================

def generar_escenario_A3(cfg: ConfigEscenarioA3,
                         diagnosticar: bool = True) -> SalidaSimulacion:
    """
    Genera R replicas del Algoritmo A-3.

    Cada replica simula el AR-GARCH sobre (burn_in + T) L pasos, descarta los
    primeros `burn_in` bloques y corta el resto en T bloques contiguos.
    """
    cfg.validar()
    L = int(cfg.L)
    burn = int(cfg.burn_in)
    j = np.arange(1, L + 1, dtype=float)[None, :]

    def simulador(rng, n):
        Z, h, a = simular_ar_garch(cfg, n, rng)
        return Z, {"h": h, "innovacion": a}

    def oraculos(prev, Z, lat, ctx):
        T = prev.shape[0]
        # Indice (base 0) del ultimo instante del bloque anterior a cada origen.
        ult = np.clip(np.arange(burn, burn + T) * L - 1, 0, None)
        h_sig = lat["h"][np.minimum(ult + 1, lat["h"].size - 1)]
        return {
            "media_condicional": cfg.phi ** j * prev[:, [-1]],
            "sd_condicional_pasado_completo": sd_condicional_bloque(h_sig, cfg, L),
        }

    return generar_serie_segmentada(
        cfg, simulador,
        oraculos=oraculos,
        resumen=resumen_escenario_A3 if diagnosticar else None,
        diagnosticar=diagnosticar,
    )


# ==========================================================================
# CONTROL DE CALIDAD
# ==========================================================================

def resumen_escenario_A3(salida: SalidaSimulacion) -> dict:
    """
    Control de calidad propio del Algoritmo A-3.

    Ademas del diagnostico comun verifica lo que DEFINE al escenario: la
    autocorrelacion de Z contra phi^k, la agrupacion de volatilidad
    (autocorrelacion de a_s^2 y curtosis de a_s contra su valor teorico), la
    dispersion de la varianza entre curvas y el R^2 del oraculo, que es
    minimo.
    """
    cfg = salida.config
    L, T = int(cfg.L), int(cfg.T)
    burn = int(cfg.burn_in)
    a1, b1, phi = cfg.alpha1, cfg.beta1, cfg.phi
    diag = diagnostico_serie_escalar(salida)

    a = salida.internos["innovacion"][:, burn * L:]           # (R, T L)
    h = salida.internos["h"][:, burn * L:]
    sd_c = salida.internos["sd_condicional_pasado_completo"]  # (R, T, L)

    # Curtosis teorica de a_s (finita si 3 a1^2 + 2 a1 b1 + b1^2 < 1).
    den = 1.0 - b1 ** 2 - 2.0 * a1 * b1 - 3.0 * a1 ** 2
    kurt_teo = float(3.0 + 6.0 * a1 ** 2 / den) if den > 0 else float("inf")
    kurt_emp = float(np.mean(((a - a.mean(axis=1, keepdims=True)) ** 4).mean(axis=1)
                             / a.var(axis=1) ** 2))
    # Autocorrelacion teorica de a^2 al rezago 1 (GARCH(1,1)).
    rho1_a2_teo = (a1 * (1.0 - a1 * b1 - b1 ** 2)
                   / (1.0 - 2.0 * a1 * b1 - b1 ** 2)) if den > 0 else float("nan")

    var_curva = salida.curvas.var(axis=2)                     # (R, T)
    r2_teo = float(np.mean(phi ** (2.0 * np.arange(1, L + 1))))
    return {
        **diag,
        "phi": float(phi), "alpha1": float(a1), "beta1": float(b1),
        "sigma_a2": float(cfg.sigma_a2), "alpha0": cfg.alpha0,
        "persistencia_alpha1_beta1": cfg.persistencia,
        "memoria_h_pasos": float(1.0 / (1.0 - cfg.persistencia)),
        "var_Z_teorica": float(cfg.sigma_a2 / (1.0 - phi ** 2)),
        "var_Z_empirica": float(salida.internos["serie_escalar"].var()),
        "rho_teorica_lag1": float(phi),
        "rho_teorica_lagL": float(phi ** L),
        "kurtosis_a_teorica": kurt_teo,
        "kurtosis_a_empirica": kurt_emp,
        "rho1_a2_teorica": float(rho1_a2_teo),
        "rho1_a2_empirica": acf_empirica_serie(a ** 2, 1),
        "rhoL_a2_empirica": acf_empirica_serie(a ** 2, L),
        "h_min": float(h.min()), "h_max": float(h.max()),
        "h_media": float(h.mean()),
        "cv_var_por_curva": float(var_curva.std() / var_curva.mean()),
        "sd_cond_j1_p05": float(np.quantile(sd_c[..., 0], 0.05)),
        "sd_cond_j1_p95": float(np.quantile(sd_c[..., 0], 0.95)),
        "r2_oraculo_1rezago": r2_teo,
        "r2_oraculo_1rezago_empirico": r2_empirico_media_condicional(salida),
        "nota_media_condicional": (
            "media_condicional = phi^j Z_last: exacta dado el bloque t-1 y dado "
            "todo el pasado, y coincide con el mejor predictor lineal. Sin "
            "hs_oraculo_L2: el operador es una evaluacion puntual."),
    }
