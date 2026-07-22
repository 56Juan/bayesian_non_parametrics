"""
sim_escenario_2.py
==================
Escenario 2: proceso funcional GARCH(1,1) con varianza condicional dependiente
del estado (Algoritmo 2 del anexo).

Modelo generador
----------------
    X_t(tau) = mu(tau) + sigma_t(tau) eta_t(tau)

    sigma_t^2(tau) = delta(tau)
                   + int_0^1 beta(tau, s)  (X_{t-1}(s) - mu(s))^2 ds
                   + int_0^1 gamma(tau, s) sigma_{t-1}^2(s) ds

con delta(tau) > 0 y nucleos no negativos beta, gamma >= 0, que garantizan
sigma_t^2(tau) > 0 para toda trayectoria. El nucleo beta gobierna la
sensibilidad de la varianza futura a la magnitud de la curva mas reciente
(analogo al parametro ARCH escalar) y gamma gobierna la persistencia de la
varianza condicional pasada (analogo al parametro GARCH escalar).

La existencia de una solucion estrictamente estacionaria requiere que el
operador conjunto con nucleo beta + gamma tenga radio espectral menor que uno,
condicion analoga a alpha + beta < 1 en el caso escalar, que aqui se verifica
sobre la matriz discretizada.

Bajo este generador la ley condicional de la curva dada su historia es
gaussiana en forma pero de escala variable en el tiempo: la media condicional
es constante e igual a mu, y toda la dependencia temporal reside en el segundo
momento. Los metodos funcionales lineales, cuya innovacion es homocedastica por
construccion, no disponen de mecanismo alguno para representar esta estructura.

Parametrizacion
---------------
En lugar de especificar delta, beta y gamma en bruto, la configuracion emplea
cantidades interpretables de las que aquellos se derivan:

    persistencia  -> radio espectral objetivo de B + Gamma (analogo a alpha+beta)
    prop_arch     -> fraccion de la persistencia atribuible a beta (ARCH)
    var_objetivo  -> nivel estacionario de la varianza puntual, del cual se
                     despeja delta = (I - B - Gamma) v

Esta reparametrizacion evita el problema practico de fijar delta a ciegas: en
un GARCH la varianza incondicional es delta / (1 - alpha - beta), de modo que
un mismo delta produce escalas muy distintas segun la persistencia elegida, y
los escenarios dejarian de ser comparables entre configuraciones.

El esquema de observacion, la cuadratura, la innovacion funcional y el control
de calidad transversal provienen de `sim_comun.py`; este modulo aporta la
dinamica del algoritmo y sus verificaciones especificas.

Uso tipico desde un notebook
----------------------------
    from model_psbp_fd.simulations import (
        ConfigEscenario2, generar_escenario_2, guardar_escenario
    )

    cfg = ConfigEscenario2(L=48, T=200, burn_in=200, R=50, seed=20260719,
                           persistencia=0.85, prop_arch=0.25,
                           alcance_beta=0.15, alcance_gamma=0.30,
                           var_objetivo=1.0, ell=0.2, sigma_obs=0.5)
    salida = generar_escenario_2(cfg)
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
    pesos_trapezoidales,
    evaluar_media,
    matriz_covarianza_innovacion,
    factor_cholesky,
    generador_innovacion,
    semillas_replicas,
    aplicar_ruido_observacion,
    diagnostico_comun,
)

__all__ = [
    "ConfigEscenario2",
    "generar_escenario_2",
    "resumen_escenario_2",
    "simular_trayectoria_fgarch",
    "matriz_kernel_no_negativo",
    "radio_espectral",
    "construir_operadores_garch",
]


# ==========================================================================
# CONFIGURACION
# ==========================================================================

@dataclass
class ConfigEscenario2(ConfigObservacion):
    """
    Parametros del Escenario 2.

    Hereda de `ConfigObservacion` los parametros del esquema de observacion
    (L, T, burn_in, sigma_obs, R, seed, media_fn, jitter) y agrega los del
    mecanismo generador.

    Recursion de varianza condicional
        persistencia : radio espectral objetivo del operador conjunto
                       B + Gamma. Debe pertenecer a (0, 1) para garantizar la
                       existencia de una solucion estrictamente estacionaria.
                       Valores cercanos a uno producen agrupamiento de
                       volatilidad prolongado; valores pequenos producen una
                       varianza condicional que revierte rapido a su nivel.
        prop_arch    : fraccion de la persistencia atribuible al nucleo beta,
                       en (0, 1). Valores altos hacen que la varianza reaccione
                       de forma abrupta a la ultima curva observada; valores
                       bajos trasladan el peso a gamma y producen una varianza
                       mas suave y persistente.
        alcance_beta : alcance de la dependencia entre puntos del dominio en el
                       nucleo beta. Valores pequenos hacen que la varianza en
                       tau responda solo a la magnitud de la curva anterior en
                       un entorno estrecho de tau.
        alcance_gamma: idem para el nucleo gamma.
        var_objetivo : nivel estacionario de la varianza puntual del proceso,
                       del cual se despeja delta. Fija la escala del proceso con
                       independencia de la persistencia elegida.

    Innovacion funcional
        ell : longitud de correlacion de la innovacion eta; determina la
              suavidad de las trayectorias. La varianza puntual de eta es
              unitaria por construccion (se emplea la matriz de correlacion,
              no de covarianza), de modo que sigma_t^2 es exactamente la
              varianza condicional del proceso.
    """

    persistencia: float = 0.85
    prop_arch: float = 0.25
    alcance_beta: float = 0.15
    alcance_gamma: float = 0.30
    var_objetivo: float = 1.0
    ell: float = 0.2

    def validar(self) -> None:
        super().validar()
        if not (0.0 < self.persistencia < 1.0):
            raise ValueError(
                f"persistencia={self.persistencia}: debe estar en (0, 1) para "
                "garantizar la existencia de una solucion estacionaria."
            )
        if not (0.0 < self.prop_arch < 1.0):
            raise ValueError(
                f"prop_arch={self.prop_arch}: debe estar en (0, 1). Los valores "
                "extremos degeneran el modelo en un ARCH puro (1) o en una "
                "varianza sin realimentacion del proceso (0)."
            )
        if self.alcance_beta <= 0 or self.alcance_gamma <= 0:
            raise ValueError("alcance_beta y alcance_gamma deben ser positivos.")
        if self.var_objetivo <= 0:
            raise ValueError("var_objetivo debe ser positivo.")
        if self.ell <= 0:
            raise ValueError("ell debe ser positivo.")


# ==========================================================================
# OPERADORES DE LA RECURSION DE VARIANZA
# ==========================================================================

def radio_espectral(M: np.ndarray) -> float:
    """Radio espectral de una matriz cuadrada."""
    return float(np.max(np.abs(np.linalg.eigvals(M))))


def matriz_kernel_no_negativo(tau: np.ndarray, alcance: float) -> np.ndarray:
    """
    Matriz discretizada de un operador integral con nucleo gaussiano no negativo

        k(tau, s) = exp{ -(tau - s)^2 / (2 alcance^2) },

    con la misma cuadratura trapezoidal empleada en el Algoritmo 1:

        (int k(tau, s) f(s) ds)|_{tau_l} ~= sum_{l'} k(tau_l, tau_l') w_{l'} f(tau_l').

    La matriz retornada NO esta escalada: la calibracion de su magnitud se
    realiza en `construir_operadores_garch`, que fija el radio espectral del
    operador conjunto. La no negatividad del nucleo garantiza que la recursion
    de varianza preserve la positividad de sigma^2.
    """
    if alcance <= 0:
        raise ValueError("alcance debe ser positivo.")
    w = pesos_trapezoidales(tau)
    dist2 = (tau[:, None] - tau[None, :]) ** 2
    nucleo = np.exp(-dist2 / (2.0 * alcance ** 2))
    return nucleo * w[None, :]


def construir_operadores_garch(
    tau: np.ndarray,
    persistencia: float,
    prop_arch: float,
    alcance_beta: float,
    alcance_gamma: float,
    var_objetivo: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Construye (B, Gamma, delta, var_estacionaria) a partir de los parametros
    interpretables de la configuracion.

    Calibracion de la persistencia
    ------------------------------
    Se normaliza cada nucleo a norma infinito unitaria (maxima suma por fila),
    se combinan segun `prop_arch` y se escala el conjunto de modo que

        ||B + Gamma||_inf = persistencia

    de forma exacta y sin busqueda numerica, aprovechando que la norma es
    homogenea de grado uno para escalares positivos y que los nucleos son no
    negativos, de modo que la norma de la suma es la suma de las normas cuando
    ambos alcanzan su maximo en la misma fila.

    La eleccion de la norma infinito y no del radio espectral como cantidad
    calibrada es deliberada. El anexo exige rho(B + Gamma) < 1, condicion que
    esta calibracion implica siempre, dado que rho(M) <= ||M||_inf para toda
    matriz. La implicacion inversa no vale: calibrar directamente el radio
    espectral admite configuraciones cuyas sumas por fila superan la unidad, en
    las cuales el despeje de delta produce valores negativos y la recursion de
    varianza deja de estar bien definida. La norma infinito es ademas el
    analogo exacto de la condicion alpha + beta < 1 del GARCH escalar, al que
    ambos criterios se reducen cuando L = 1.

    Nivel estacionario
    ------------------
    En estado estacionario la varianza puntual esperada v satisface

        v = delta + (B + Gamma) v,

    puesto que E[(X_{t-1} - mu)^2] = E[sigma_{t-1}^2] = v cuando la innovacion
    tiene varianza puntual unitaria. De ahi se despeja

        delta = (I - B - Gamma) v,

    con v = var_objetivo * 1. Bajo la calibracion anterior las sumas por fila
    de B + Gamma no superan `persistencia`, de modo que

        delta >= var_objetivo * (1 - persistencia) > 0

    queda garantizado por construccion. La verificacion explicita se mantiene
    como red de seguridad ante cambios futuros en la forma de los nucleos.
    """
    B0 = matriz_kernel_no_negativo(tau, alcance_beta)
    G0 = matriz_kernel_no_negativo(tau, alcance_gamma)

    norma_B0 = float(np.max(B0.sum(axis=1)))
    norma_G0 = float(np.max(G0.sum(axis=1)))
    if norma_B0 <= 0 or norma_G0 <= 0:
        raise RuntimeError("Algun nucleo discretizado resulto identicamente nulo.")

    # Normalizacion a norma infinito unitaria y combinacion segun prop_arch
    Bn = (prop_arch / norma_B0) * B0
    Gn = ((1.0 - prop_arch) / norma_G0) * G0
    escala = persistencia / float(np.max((Bn + Gn).sum(axis=1)))

    B = escala * Bn
    Gamma = escala * Gn

    # Nivel estacionario y despeje de delta
    L = tau.size
    v = np.full(L, float(var_objetivo))
    delta = v - (B + Gamma) @ v

    if np.any(delta <= 0):
        n_mal = int(np.sum(delta <= 0))
        raise ValueError(
            f"delta resulto no positivo en {n_mal} de {L} puntos de la grilla "
            f"(min={delta.min():.4g}), pese a la calibracion por norma infinito. "
            "Revise la forma de los nucleos: la construccion supone nucleos no "
            "negativos con sumas por fila acotadas por `persistencia`."
        )

    return B, Gamma, delta, v


# ==========================================================================
# DINAMICA DEL ESCENARIO
# ==========================================================================

def simular_trayectoria_fgarch(
    B: np.ndarray,
    Gamma: np.ndarray,
    delta: np.ndarray,
    var_estacionaria: np.ndarray,
    chol_R: np.ndarray,
    mu: np.ndarray,
    T: int,
    burn_in: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Itera la recursion FGARCH(1,1) discretizada y devuelve

        curvas : (T, L) curvas retenidas SIN ruido de medicion,
        sigma2 : (T, L) varianzas condicionales de esos mismos periodos.

    El estado se inicializa en el nivel estacionario: sigma_0^2 = v y
    (X_0 - mu)^2 se toma tambien en v mediante una realizacion de la innovacion
    escalada, de modo que el periodo de calentamiento parta desde una condicion
    ya coherente con la distribucion estacionaria y no desde un transitorio
    artificial.

    La innovacion `chol_R` corresponde a la factorizacion de la matriz de
    CORRELACION de eta (diagonal unitaria), no de una covarianza general: con
    ello E[eta_t(tau)^2] = 1 exactamente para todo tau, que es la condicion de
    identificabilidad que separa sigma_t^2 de la escala de la innovacion.
    """
    L = B.shape[0]
    if (Gamma.shape[0] != L or delta.shape[0] != L
            or chol_R.shape[0] != L or mu.shape[0] != L):
        raise ValueError(
            f"Dimensiones inconsistentes: B es {B.shape}, Gamma es "
            f"{Gamma.shape}, delta es {delta.shape}, chol_R es {chol_R.shape} "
            f"y mu es {mu.shape}."
        )

    innovacion = generador_innovacion(chol_R, rng)

    sigma2 = var_estacionaria.copy()
    resid2 = var_estacionaria * innovacion() ** 2      # (X_0 - mu)^2 inicial

    def paso() -> tuple[np.ndarray, np.ndarray]:
        """Un periodo de la recursion; devuelve (residuo centrado, sigma2)."""
        nonlocal sigma2, resid2
        sigma2 = delta + B @ resid2 + Gamma @ sigma2
        y = np.sqrt(sigma2) * innovacion()             # X_t - mu
        resid2 = y ** 2
        return y, sigma2

    for _ in range(burn_in):
        paso()

    curvas = np.empty((T, L))
    sigma2_out = np.empty((T, L))
    for t in range(T):
        y, s2 = paso()
        curvas[t] = mu + y
        sigma2_out[t] = s2
    return curvas, sigma2_out


# ==========================================================================
# GENERADOR PRINCIPAL
# ==========================================================================

def generar_escenario_2(cfg: ConfigEscenario2) -> SalidaSimulacion:
    """
    Genera R replicas independientes del Escenario 2.

    Los operadores de la recursion de varianza y la factorizacion de la
    correlacion de la innovacion se construyen una sola vez y se comparten
    entre replicas, dado que no dependen de la realizacion aleatoria. Cada
    replica recibe una semilla derivada de la semilla maestra, de modo que la
    replica r es identica con independencia de cuantas replicas se generen o
    del orden de ejecucion.
    """
    cfg.validar()

    tau = grilla_regular(cfg.L)
    w_quad = pesos_trapezoidales(tau)
    mu = evaluar_media(cfg.media_fn, tau)

    B, Gamma, delta, v = construir_operadores_garch(
        tau, cfg.persistencia, cfg.prop_arch,
        cfg.alcance_beta, cfg.alcance_gamma, cfg.var_objetivo,
    )

    # Matriz de CORRELACION de la innovacion: varianza puntual unitaria exacta
    R_eta = matriz_covarianza_innovacion(tau, 1.0, cfg.ell)
    chol_R = factor_cholesky(R_eta, cfg.jitter)

    hijas, registro = semillas_replicas(cfg.seed, cfg.R)

    curvas = np.empty((cfg.R, cfg.T, cfg.L))
    observaciones = np.empty((cfg.R, cfg.T, cfg.L))
    sigma2 = np.empty((cfg.R, cfg.T, cfg.L))

    for r, semilla in enumerate(hijas):
        rng = np.random.default_rng(semilla)
        curvas_r, sigma2_r = simular_trayectoria_fgarch(
            B, Gamma, delta, v, chol_R, mu, cfg.T, cfg.burn_in, rng
        )
        curvas[r] = curvas_r
        sigma2[r] = sigma2_r
        observaciones[r] = aplicar_ruido_observacion(curvas_r, cfg.sigma_obs, rng)

    salida = SalidaSimulacion(
        observaciones=observaciones,
        curvas=curvas,
        grilla=tau,
        media=mu,
        semillas=registro,
        config=cfg,
        internos={
            "operador_beta": B,
            "operador_gamma": Gamma,
            "delta": delta,
            "var_estacionaria": v,
            "corr_innovacion": R_eta,
            "pesos_cuadratura": w_quad,
            "sigma2": sigma2,          # (R, T, L) varianza condicional latente
        },
    )
    salida.diagnostico = resumen_escenario_2(salida)
    return salida


# ==========================================================================
# CONTROL DE CALIDAD ESPECIFICO
# ==========================================================================

def resumen_escenario_2(salida: SalidaSimulacion) -> dict:
    """
    Control de calidad del generador, previo a cualquier ajuste.

    Extiende `diagnostico_comun` con tres bloques de verificacion propios del
    Algoritmo 2.

    1. Condicion de estacionariedad. Se comprueba que el radio espectral
       efectivo de B + Gamma coincida con el valor objetivo y sea menor que
       uno, y que la varianza condicional simulada permanezca acotada, que es
       la verificacion empirica de estabilidad que el anexo exige.

    2. Calibracion de escala. La varianza puntual empirica debe aproximar el
       nivel estacionario teorico `var_objetivo`; una discrepancia sistematica
       indica un error en el despeje de delta o un calentamiento insuficiente.

    3. Presencia efectiva del rasgo estructural. Un generador GARCH cuya
       varianza condicional apenas varie seria indistinguible del Escenario 1 y
       dejaria sin contenido al escenario. Se reportan, en consecuencia, el
       coeficiente de variacion temporal de sigma^2, la curtosis marginal del
       proceso (el mecanismo GARCH induce exceso de curtosis sobre la
       gaussiana, cuyo valor de referencia es 3) y la autocorrelacion a rezago
       uno de los residuos al cuadrado, que es el estadistico clasico de
       deteccion de efectos ARCH. Valores proximos a cero en las tres
       cantidades senalarian un escenario degenerado.
    """
    if not isinstance(salida.config, ConfigEscenario2):
        raise TypeError(
            "resumen_escenario_2 requiere una salida generada con "
            f"ConfigEscenario2; se recibio {type(salida.config).__name__}."
        )

    base = diagnostico_comun(salida)

    for clave in ("operador_beta", "operador_gamma", "sigma2", "var_estacionaria"):
        if salida.internos.get(clave) is None:
            raise KeyError(
                f"La salida no contiene '{clave}' en `internos`; no puede "
                "verificarse la condicion de estacionariedad del Algoritmo 2."
            )

    B = salida.internos["operador_beta"]
    Gamma = salida.internos["operador_gamma"]
    sigma2 = salida.internos["sigma2"]            # (R, T, L)
    v = salida.internos["var_estacionaria"]

    rho = radio_espectral(B + Gamma)
    norma_inf = float(np.max((B + Gamma).sum(axis=1)))

    # ── Bloque 3: evidencia de que el mecanismo esta activo ────────────────
    # Coeficiente de variacion temporal de sigma^2, promediado sobre el dominio
    cv_sigma2 = float(np.mean(sigma2.std(axis=1) / sigma2.mean(axis=1)))

    # Curtosis marginal del proceso centrado (referencia gaussiana: 3)
    Y = salida.curvas - salida.media[None, None, :]
    m2 = np.mean(Y ** 2, axis=1)
    m4 = np.mean(Y ** 4, axis=1)
    curtosis = float(np.mean(m4 / np.where(m2 > 0, m2 ** 2, np.nan)))

    # ACF a rezago 1 de los residuos al cuadrado (estadistico de efecto ARCH)
    R_rep = Y.shape[0]
    acf1_cuad = np.empty(R_rep)
    for r in range(R_rep):
        Z = Y[r] ** 2
        Z = Z - Z.mean(axis=0, keepdims=True)
        num = np.sum(Z[:-1] * Z[1:], axis=0)
        den = np.sum(Z * Z, axis=0)
        acf1_cuad[r] = float(np.nanmean(num / np.where(den > 0, den, np.nan)))

    def ee(x: np.ndarray) -> float:
        return float(x.std(ddof=1) / np.sqrt(x.size)) if x.size > 1 else float("nan")

    especifico = {
        # 1. Estacionariedad y estabilidad
        "persistencia_objetivo": float(salida.config.persistencia),
        "norma_inf_BG": norma_inf,                  # cantidad calibrada
        "persistencia_error_absoluto": abs(norma_inf - salida.config.persistencia),
        "radio_espectral_BG": rho,                  # condicion del anexo
        "estacionariedad_garantizada": bool(rho < 1.0),
        "sigma2_finito": bool(np.all(np.isfinite(sigma2))),
        "sigma2_positivo": bool(np.all(sigma2 > 0)),
        "sigma2_min": float(sigma2.min()),
        "sigma2_max": float(sigma2.max()),
        # 2. Calibracion de escala
        "var_objetivo": float(salida.config.var_objetivo),
        "var_estacionaria_teorica": float(np.mean(v)),
        "var_empirica_media": float(base["var_puntual_media"]),
        "var_error_relativo": float(
            abs(base["var_puntual_media"] - np.mean(v)) / np.mean(v)
        ),
        # 3. Presencia efectiva de heterocedasticidad condicional
        "cv_temporal_sigma2": cv_sigma2,
        "curtosis_marginal": curtosis,
        "exceso_curtosis": curtosis - 3.0,
        "acf1_residuos_cuadrado": float(acf1_cuad.mean()),
        "acf1_cuadrado_ee_montecarlo": ee(acf1_cuad),
    }
    return {**base, **especifico}
