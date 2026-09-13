"""
sim_escenario_K.py
===================
Escenario K: FAR(1) con VARIANZA CONDICIONAL cuya FORMA en tau se conmuta
segun el signo del estado rezagado. Escenario de DIAGNOSTICO, no es un
Algoritmo del anexo; letra siguiente libre despues de J.

Modelo generador
-----------------
    X_t(tau) = mu(tau) + Y_t(tau),

    Y_t(tau) = (Psi Y_{t-1})(tau) + s(tau; z_{t-1}) * xi_t(tau),

    z_{t-1} = <Y_{t-1}, e>_{L^2},           e = direccion_constante (nivel),

    s(tau; z) = sqrt( 1 + kappa * tanh(nitidez * z / sd_z) * v(tau) ),

    v(tau) = cos(2 pi tau),   (+1 en los extremos, -1 en el centro),

con `xi_t(tau) = (chol_K @ N(0, I))(tau)` la MISMA innovacion gaussiana
correlacionada de los Algoritmos 1-3 (covarianza K, longitud `ell`), y `s`
un reescalamiento PUNTUAL en tau: como es una funcion positiva del dominio,
multiplicar `xi_t` por ella preserva la ley gaussiana condicional y no exige
ninguna factorizacion nueva --el resultado tiene covarianza condicional
Diag(s) K Diag(s), siempre definida positiva porque K lo es y `s > 0`--.

Por que existe este escenario
------------------------------
Ninguna simulacion del estudio hace que la varianza puntual DENTRO de la
curva dependa del estado rezagado. El Algoritmo 2 (FGARCH) modula un factor
de volatilidad ESCALAR sigma_t^2 que multiplica toda la curva por igual; los
demas son homocedasticos en tau. El patron real que motiva este escenario ---
una region del dominio (p. ej. los extremos de tau) con mas varianza que otra
(el centro) en un periodo, y la region ancha invirtiendose en el periodo
siguiente segun el estado anterior--- no lo produce ningun generador
existente. `v(tau)` fija una FORMA (mas varianza en los extremos, menos en el
centro, como el boceto que origina el escenario) y `tanh(z/sd_z)` CONMUTA su
signo con el estado rezagado: cuando z_{t-1} > 0 la varianza se concentra en
los extremos, cuando z_{t-1} < 0 se concentra en el centro. Es el analogo,
sobre el canal de VARIANZA, de lo que el Escenario B hace sobre el canal de
MEDIA: una conmutacion antisimetrica gobernada por el estado rezagado.

Por que la varianza MARGINAL en tau es (casi) constante y eso no es un defecto
-------------------------------------------------------------------------------
`tanh` es una funcion IMPAR y, bajo una ley aproximadamente simetrica de z, el
promedio de `tanh(z/sd_z)` sobre la estacionaria es practicamente nulo. Por lo
tanto la varianza puntual PROMEDIADA sobre el tiempo es aproximadamente
uniforme en tau --el generador no rompe la propiedad de que la varianza
marginal en tau sea aproximadamente constante en ninguno de los escenarios
existentes--, pero la varianza CONDICIONAL al signo del rezago anterior tiene
dos formas antisimetricas y bien diferenciadas. Un metodo cuya predictiva solo
puede escalar globalmente el ancho de la banda (una desviacion posterior por
score, propagada con las MISMAS autofunciones fijas de la representacion) no
puede reproducir un cambio de FORMA en tau condicionado al signo del rezago,
aun si acierta el ancho promedio: es exactamente la limitacion de
`PropagadorFuncional` que el heurístico de CLAUDE.md documenta ---la forma de
la heterocedasticidad en tau que el modelo puede producir esta acotada por el
subespacio generado por psi_m(tau)^2 de las M autofunciones fijas--- puesta a
prueba de forma deliberada.

Por que la direccion de conmutacion es el NIVEL y no una direccion oscilatoria
-------------------------------------------------------------------------------
A diferencia del Escenario B, aqui NO hay riesgo de que la conmutacion
cancele una correlacion lineal en la MEDIA: el termino conmutado multiplica a
la innovacion, no al arrastre, de modo que la media condicional del proceso
sigue siendo exactamente la del Algoritmo 1 (`Psi Y_{t-1}`) sin importar el
signo de z. Por eso se usa `direccion_constante` (el nivel de la curva
rezagada), la eleccion mas simple e interpretable, sin la restriccion de
ortogonalidad que el Escenario B necesita para su propio mecanismo.

Relacion con el Algoritmo 2 (FGARCH) y con el Escenario B
------------------------------------------------------------
Comparte con el Algoritmo 2 el objetivo de producir heterocedasticidad
condicional, pero en un eje ortogonal: el Algoritmo 2 la pone en el tiempo t
(escalar), este la pone en tau (forma). Comparte con el Escenario B el
mecanismo de conmutacion antisimetrica gobernada por un funcional lineal del
estado rezagado, pero aplicado al canal de varianza y no al de media, de modo
que --a diferencia del Escenario B-- el MISE de la media condicional NO
discrimina aqui (la media es la del Algoritmo 1): el eje que este escenario
existe para poner a prueba es exclusivamente el de CRPS, energia y cobertura
CONDICIONAL en el signo del rezago, no el de error puntual.

Lo que NO vive aqui
--------------------
El esquema de observacion, la cuadratura del operador, la innovacion
funcional gaussiana y el control de calidad transversal provienen de
`sim_comun.py`; `direccion_constante` se reutiliza de `sim_escenario_3.py`.

Uso tipico desde un notebook
------------------------------
    from model_psbp_fd.pipelines import (
        ConfigEscenarioK, generar_escenario_K, resumen_escenario_K,
    )

    cfg = ConfigEscenarioK(
        L=75, T=400, burn_in=200, R=1, seed=41232, sigma_obs=0.25,
        media_fn=media_senoidal,
    )
    salida = generar_escenario_K(cfg)
    salida.diagnostico["razon_varianza_extremos_centro_zpos"]
    salida.diagnostico["razon_varianza_extremos_centro_zneg"]
    X = salida.observaciones                     # (R, T, L)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from .sim_comun import (
    ConfigObservacion,
    SalidaSimulacion,
    grilla_regular,
    evaluar_media,
    matriz_operador_ar,
    matriz_covarianza_innovacion,
    factor_cholesky,
    semillas_replicas,
    aplicar_ruido_observacion,
    diagnostico_comun,
    norma_hilbert_schmidt,
    pesos_trapezoidales,
)
from .sim_escenario_3 import direccion_constante

__all__ = [
    "ConfigEscenarioK",
    "generar_escenario_K",
    "resumen_escenario_K",
    "envolvente_extremos_centro",
    "simular_trayectoria_far_heterocedastico",
]


# ==========================================================================
# ENVOLVENTE FIJA DE LA HETEROCEDASTICIDAD EN TAU
# ==========================================================================

def envolvente_extremos_centro(tau: np.ndarray) -> np.ndarray:
    """
    Forma v(tau) = cos(2 pi tau): +1 en tau=0 y tau=1 (los extremos), -1 en
    tau=0.5 (el centro), norma SUPREMO exactamente unitaria por construccion.

    Positiva en los extremos del dominio, negativa en el centro: cuando el
    factor que la multiplica es positivo, la varianza puntual crece en los
    extremos y decrece en el centro; cuando es negativo, se invierte. Es la
    forma mas simple que reproduce el patron del boceto que origina el
    escenario (mas varianza en los bordes, menos en medio, y la region ancha
    conmutando segun el estado).

    La primera version de esta funcion usaba (tau-0.5)^2 centrada y
    normalizada. Se DESCARTO: una parabola centrada por su media es
    ASIMETRICA respecto de cero --el maximo en los extremos (+0.1667 tras
    restar la media 1/12) es el DOBLE del minimo en el centro (-0.0833)--, de
    modo que al normalizar a norma supremo los extremos llegan a +1 pero el
    centro solo a -0.5: el contraste nunca puede ser simetrico y kappa=0.90
    solo entregaba una razon de varianza extremos/centro de ~2.7:1 en vez del
    ~19:1 que kappa permite en principio. `cos(2 pi tau)` es EXACTAMENTE
    simetrica (+1 en los extremos, -1 en el centro, sin necesidad de centrar
    ni de normalizar aparte) y con ella el contraste alcanzable con kappa=0.90
    se acerca al ~19:1 completo en ambos lados.
    """
    tau = np.asarray(tau, dtype=float)
    return np.cos(2.0 * np.pi * tau)


# ==========================================================================
# CONFIGURACION
# ==========================================================================

@dataclass
class ConfigEscenarioK(ConfigObservacion):
    """
    Parametros del Escenario K. Hereda el esquema de observacion de
    `ConfigObservacion` (L, T, burn_in, sigma_obs, R, seed, media_fn, jitter).

    Media condicional --- identica al Algoritmo 1
        gamma, hs_norm : nucleo gaussiano de Psi y su norma de Hilbert-Schmidt.
                         Los valores por defecto (0.30, 0.70) son los del
                         Algoritmo 1 a proposito: la media condicional de este
                         escenario debe ser indistinguible de la de la
                         corrida 11/20, de modo que cualquier diferencia en
                         CRPS/energia/cobertura condicional sea atribuible al
                         canal de varianza y no a una diferencia de dinamica.
        sigma_eps, ell : escala y suavidad de la CORRELACION espacial de la
                         innovacion (antes de aplicar `s(tau; z)`).

    Heterocedasticidad condicional en tau
        kappa         : amplitud del reescalamiento. Debe cumplir
                         kappa * max|v(tau)| < 1 para que
                         1 + kappa tanh(.) v(tau) sea estrictamente positivo
                         en todo tau y todo z; se verifica con un assert
                         numerico en `validar()` contra el maximo de
                         `envolvente_extremos_centro` en la grilla resultante.
                         Con la envolvente por defecto normalizada a norma
                         SUPREMO unitaria (max|v| = 1), la cota es
                         simplemente kappa < 1. El valor por defecto 0.90 deja
                         s(tau;z)^2 recorriendo [0.10, 1.90] en el extremo del
                         rango de z observado --una razon de varianza
                         extremos/centro de hasta ~19:1--, sin arriesgar la
                         positividad.
        nitidez       : pendiente del argumento del tanh, en las unidades de
                         z/sd_z. Con nitidez=1.0, tanh(z/sd_z) solo se acerca
                         a +-1 cuando |z| es varias veces sd_z, un evento
                         raro; la MAYORIA de los instantes tienen |z| ~ sd_z y
                         el factor efectivo tanh(z/sd_z) ronda apenas ~0.6-0.8,
                         diluyendo el contraste de kappa a una fraccion chica
                         de su valor teorico. El valor por defecto 5.0 hace
                         que tanh(nitidez * z/sd_z) sature para |z| bastante
                         menor que sd_z, de modo que la mayoria de los
                         instantes --no solo la cola-- alcanzan un contraste
                         cercano al maximo permitido por kappa. Medido con los
                         defaults (kappa=0.90, seed=41232, L=75, T=400, R=1):
                         `razon_escala_cuadrado_extremos_centro_zpos/zneg` ~
                         7.1 / 0.14 (contraste INSTANTANEO, no diluido por la
                         persistencia), y `razon_varianza_extremos_centro_
                         zpos/zneg` ~ 3.6 / 0.22 sobre la curva REALIZADA
                         --menor por la memoria del operador Psi, ver la nota
                         grande en `resumen_escenario_K`--. Subir `nitidez`
                         acerca ambas razones a los extremos teoricos (0.10 y
                         1.90) pero vuelve la conmutacion casi un umbral duro,
                         perdiendo los instantes "ambiguos" (z cerca de 0) que
                         son los que tendrian una mezcla genuina de escalas;
                         bajarla las diluye de vuelta hacia 1. El valor 5.0 es
                         el punto elegido entre ambos extremos.
        direccion_fn  : direccion e(tau) sobre la que se proyecta el estado
                         rezagado para obtener z_{t-1}. Por defecto
                         `direccion_constante` (el nivel de la curva).
        envolvente_fn : forma v(tau) de la heterocedasticidad. Por defecto
                         `envolvente_extremos_centro`.

    Diagnostico
        n_pilot : longitud de la trayectoria piloto usada para estimar la
                  desviacion estandar `sd_z` del estado rezagado con la que
                  se normaliza el argumento del `tanh`.
    """

    # Media condicional (Algoritmo 1)
    gamma: float = 0.30
    hs_norm: float = 0.70
    sigma_eps: float = 1.0
    ell: float = 0.5

    # Heterocedasticidad condicional
    kappa: float = 0.90
    nitidez: float = 5.0
    direccion_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None
    envolvente_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None

    # Diagnostico
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

        if self.kappa <= 0:
            raise ValueError(
                "kappa debe ser positivo. Con kappa -> 0 la varianza deja de "
                "depender del rezago y el escenario se reduce al Algoritmo 1."
            )
        if self.nitidez <= 0:
            raise ValueError(
                "nitidez debe ser positivo. Con nitidez -> 0 el tanh se "
                "aplana en torno a 0 y el contraste se diluye para todo z, "
                "con independencia de kappa."
            )

        tau = grilla_regular(self.L)
        env_fn = self.envolvente_fn or envolvente_extremos_centro
        v = np.asarray(env_fn(tau), dtype=float)
        if v.shape != tau.shape:
            raise ValueError(
                f"envolvente_fn retorno forma {v.shape}; se esperaba {tau.shape}."
            )
        cota = float(self.kappa * np.max(np.abs(v)))
        if cota >= 1.0:
            raise ValueError(
                f"kappa * max|v(tau)| = {cota:.4f} >= 1: `s(tau; z)^2` podria "
                "volverse no positivo en el limite |tanh| -> 1. Reduzca kappa "
                "o la amplitud de la envolvente."
            )
        if self.n_pilot < 100:
            raise ValueError("n_pilot debe ser al menos 100.")


# ==========================================================================
# DINAMICA
# ==========================================================================

def _sd_estado_piloto(
    Psi: np.ndarray, chol_K: np.ndarray, direccion: np.ndarray,
    w_quad: np.ndarray, cfg: ConfigEscenarioK, rng: np.random.Generator,
) -> float:
    """
    Estima, sobre una trayectoria piloto SIN heterocedasticidad activa, la
    desviacion estandar de z_t = <Y_t, e>. Es la escala con la que se
    normaliza el argumento del `tanh`, de modo que `kappa` tenga una lectura
    estable frente a cambios en `hs_norm` o `sigma_eps`.
    """
    L = Psi.shape[0]
    Y = chol_K @ rng.standard_normal(L)
    for _ in range(cfg.burn_in):
        Y = Psi @ Y + chol_K @ rng.standard_normal(L)
    z = np.empty(cfg.n_pilot)
    for i in range(cfg.n_pilot):
        z[i] = float(np.sum(w_quad * direccion * Y))
        Y = Psi @ Y + chol_K @ rng.standard_normal(L)
    sd = float(np.std(z))
    return sd if sd > 0 else 1.0


def simular_trayectoria_far_heterocedastico(
    Psi: np.ndarray, chol_K: np.ndarray, mu: np.ndarray, cfg: ConfigEscenarioK,
    direccion: np.ndarray, envolvente: np.ndarray, sd_z: float,
    w_quad: np.ndarray, rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Itera la recursion y devuelve las T curvas retenidas SIN ruido de
    medicion, junto con el estado rezagado y el factor de escala efectivo.

    Retorna
    -------
    curvas   : (T, L) las curvas X_t = mu + Y_t.
    z_lag    : (T,)   z_{t-1}, el estado que gobierna la escala en t.
    escala_t : (T, L) s(tau; z_{t-1}) efectivamente aplicada en cada instante,
                      guardada porque es la cantidad INOBSERVABLE que el
                      diagnostico contrasta contra la envolvente teorica.
    """
    L = Psi.shape[0]

    Y = chol_K @ rng.standard_normal(L)
    for _ in range(cfg.burn_in):
        z = float(np.sum(w_quad * direccion * Y))
        s = np.sqrt(1.0 + cfg.kappa * np.tanh(cfg.nitidez * z / sd_z) * envolvente)
        Y = Psi @ Y + s * (chol_K @ rng.standard_normal(L))

    curvas = np.empty((cfg.T, L))
    z_lag = np.empty(cfg.T)
    escala_t = np.empty((cfg.T, L))

    for t in range(cfg.T):
        z = float(np.sum(w_quad * direccion * Y))
        s = np.sqrt(1.0 + cfg.kappa * np.tanh(cfg.nitidez * z / sd_z) * envolvente)
        z_lag[t] = z
        escala_t[t] = s
        Y = Psi @ Y + s * (chol_K @ rng.standard_normal(L))
        curvas[t] = mu + Y

    return curvas, z_lag, escala_t


# ==========================================================================
# GENERADOR PRINCIPAL
# ==========================================================================

def generar_escenario_K(cfg: ConfigEscenarioK) -> SalidaSimulacion:
    """
    Genera R replicas independientes del Escenario K.

    El operador, la factorizacion de la covarianza de innovacion, la
    direccion de conmutacion y la envolvente se construyen una sola vez y se
    comparten entre replicas. `sd_z` --la escala del `tanh`-- se estima con
    una trayectoria piloto independiente de las semillas de las replicas, de
    modo que no consume su generador aleatorio.
    """
    cfg.validar()

    tau = grilla_regular(cfg.L)
    w_quad = pesos_trapezoidales(tau)
    mu = evaluar_media(cfg.media_fn, tau)
    direccion = direccion_constante(tau)
    envolvente = np.asarray(
        (cfg.envolvente_fn or envolvente_extremos_centro)(tau), dtype=float
    )

    Psi = matriz_operador_ar(tau, cfg.gamma, cfg.hs_norm)
    K = matriz_covarianza_innovacion(tau, cfg.sigma_eps, cfg.ell)
    chol_K = factor_cholesky(K, cfg.jitter)

    semilla_piloto = int(np.random.SeedSequence(cfg.seed).spawn(1)[0].entropy) & 0xFFFFFFFF
    sd_z = _sd_estado_piloto(Psi, chol_K, direccion, w_quad, cfg,
                              np.random.default_rng(semilla_piloto))

    hijas, registro = semillas_replicas(cfg.seed, cfg.R)

    curvas = np.empty((cfg.R, cfg.T, cfg.L))
    observaciones = np.empty((cfg.R, cfg.T, cfg.L))
    z_lag = np.empty((cfg.R, cfg.T))
    escala_t = np.empty((cfg.R, cfg.T, cfg.L))

    for r, semilla in enumerate(hijas):
        rng = np.random.default_rng(semilla)
        c_r, z_r, s_r = simular_trayectoria_far_heterocedastico(
            Psi, chol_K, mu, cfg, direccion, envolvente, sd_z, w_quad, rng
        )
        curvas[r] = c_r
        z_lag[r] = z_r
        escala_t[r] = s_r
        observaciones[r] = aplicar_ruido_observacion(c_r, cfg.sigma_obs, rng)

    salida = SalidaSimulacion(
        observaciones=observaciones,
        curvas=curvas,
        grilla=tau,
        media=mu,
        semillas=registro,
        config=cfg,
        internos={
            "operador": Psi,
            "cov_innovacion": K,
            "direccion_estado": direccion,
            "envolvente": envolvente,
            "sd_estado": sd_z,
            "proyeccion_estado": z_lag,
            "escala_innovacion": escala_t,
            "pesos_cuadratura": w_quad,
        },
    )
    salida.diagnostico = resumen_escenario_K(salida)
    return salida


# ==========================================================================
# CONTROL DE CALIDAD ESPECIFICO
# ==========================================================================

def resumen_escenario_K(salida: SalidaSimulacion) -> dict:
    """
    Control de calidad del generador, previo a cualquier ajuste.

    Tres bloques.

    Contractividad de la media condicional: identica al Algoritmo 1, se
    reporta para verificar que la comparacion 1 vs K aisla el canal de
    varianza (misma `hs_norm` efectiva, mismo radio espectral).

    Positividad y magnitud del reescalamiento: rango efectivo de `s(tau; z)`
    sobre la serie generada, y verificacion de que nunca fue no positivo
    (`escala_cuadrado_min > 0`).

    Heterocedasticidad condicional en tau, la razon de ser del escenario: se
    separan los instantes con z_{t-1} > 0 (extremos amplificados, segun la
    envolvente) de los z_{t-1} < 0 (centro amplificado) y se calcula, para
    cada submuestra, la razon de varianza EMPIRICA de la curva realizada
    entre el 20% de puntos con mayor |envolvente| (los extremos) y el 20% con
    menor |envolvente| (el centro). Si el mecanismo opera, esa razon deberia
    ser sistematicamente mayor a 1 en la submuestra z>0 y menor a 1 en z<0;
    `contraste_confirmado` es el chequeo booleano de esa asimetria. Se
    reporta ademas la razon sobre la varianza MARGINAL (promediada sobre
    todos los instantes), que por el argumento de simetria del encabezado
    del modulo deberia ser proxima a 1: es la cifra que distingue este
    escenario del Algoritmo 2, donde la varianza puntual SI cambia en
    promedio.
    """
    if not isinstance(salida.config, ConfigEscenarioK):
        raise TypeError(
            "resumen_escenario_K requiere una salida generada con "
            f"ConfigEscenarioK; se recibio {type(salida.config).__name__}."
        )

    base = diagnostico_comun(salida)
    cfg = salida.config

    requeridos = (
        "operador", "cov_innovacion", "direccion_estado", "envolvente",
        "sd_estado", "proyeccion_estado", "escala_innovacion",
        "pesos_cuadratura",
    )
    for nombre in requeridos:
        if salida.internos.get(nombre) is None:
            raise KeyError(
                f"La salida no contiene '{nombre}' en `internos`; no puede "
                "completarse el control de calidad del Escenario K."
            )

    Psi = salida.internos["operador"]
    w = salida.internos["pesos_cuadratura"]
    envolvente = salida.internos["envolvente"]
    z_lag = salida.internos["proyeccion_estado"]
    escala_t = salida.internos["escala_innovacion"]

    R, T, L = salida.curvas.shape

    # ── Media condicional ────────────────────────────────────────────────
    hs = norma_hilbert_schmidt(Psi, w)
    radio = float(np.max(np.abs(np.linalg.eigvals(Psi))))

    # ── Positividad y magnitud del reescalamiento ──────────────────────────
    s2 = escala_t ** 2
    escala_cuadrado_min = float(s2.min())
    escala_cuadrado_max = float(s2.max())

    # ── Heterocedasticidad condicional en tau ──────────────────────────────
    orden = np.argsort(envolvente)
    n_extremo = max(1, int(0.20 * L))
    idx_centro = orden[:n_extremo]        # envolvente mas negativa: el centro
    idx_extremos = orden[-n_extremo:]     # envolvente mas positiva: los bordes

    X = salida.curvas  # (R, T, L)

    def _razon_ext_centro(mascara: np.ndarray) -> float:
        sel = X.reshape(-1, L)[mascara.ravel()]
        var_extremos = float(np.var(sel[:, idx_extremos]))
        var_centro = float(np.var(sel[:, idx_centro]))
        return var_extremos / max(var_centro, 1e-300)

    z_pos = z_lag > 0
    z_neg = z_lag < 0
    razon_zpos = _razon_ext_centro(z_pos)
    razon_zneg = _razon_ext_centro(z_neg)
    razon_marginal = _razon_ext_centro(np.ones_like(z_lag, dtype=bool))

    contraste_confirmado = bool(razon_zpos > 1.0 and razon_zneg < 1.0)

    # Contraste INSTANTANEO (no diluido por la memoria AR): razon de s(tau;z)^2
    # entre extremos y centro, condicionada al signo de z, calculada
    # directamente sobre la escala aplicada en cada instante (no sobre la
    # curva realizada). Es la cantidad que el escenario disena --y que kappa y
    # nitidez controlan directamente-- y por construccion es mucho mayor que
    # `razon_varianza_extremos_centro_*`: esta ultima mide la varianza de la
    # curva REALIZADA, que acumula memoria de muchos instantes pasados via el
    # operador Psi (persistencia hs_norm), de modo que condicionar solo en el
    # z del rezago MAS RECIENTE diluye el efecto de un solo shock entre varios.
    # La brecha entre ambas cifras --grande cuando hs_norm es alto-- es
    # information util: dice cuanto de la heterocedasticidad instantanea
    # sobrevive a la persistencia y llega a ser observable en la curva.
    def _razon_escala_ext_centro(mascara: np.ndarray) -> float:
        sel = s2.reshape(-1, L)[mascara.ravel()]
        return float(sel[:, idx_extremos].mean() / max(sel[:, idx_centro].mean(), 1e-300))

    razon_escala_zpos = _razon_escala_ext_centro(z_pos)
    razon_escala_zneg = _razon_escala_ext_centro(z_neg)

    # Correlacion punto a punto entre tanh(z_{t-1}/sd_z) y la razon
    # (escala_extremos / escala_centro - 1) realmente aplicada: verifica que
    # el generador aplico la envolvente como se pretendia, con independencia
    # de la contaminacion por ruido de medicion.
    factor_tanh = np.tanh(cfg.nitidez * z_lag / salida.internos["sd_estado"])
    razon_escala_t = (
        escala_t[..., idx_extremos].mean(axis=-1)
        / np.maximum(escala_t[..., idx_centro].mean(axis=-1), 1e-300)
    ) - 1.0
    corr_mecanismo = float(np.corrcoef(factor_tanh.ravel(), razon_escala_t.ravel())[0, 1])

    especifico = {
        # Media condicional
        "hs_norm_objetivo": float(cfg.hs_norm),
        "hs_norm_efectiva": float(hs),
        "radio_espectral": radio,
        "contractividad": bool(hs < 1.0),
        # Positividad y magnitud
        "kappa": float(cfg.kappa),
        "nitidez": float(cfg.nitidez),
        "sd_estado_rezagado": float(salida.internos["sd_estado"]),
        "escala_cuadrado_min": escala_cuadrado_min,
        "escala_cuadrado_max": escala_cuadrado_max,
        "positividad_garantizada": bool(escala_cuadrado_min > 0.0),
        # Contraste INSTANTANEO (no diluido por la memoria AR): la cifra que
        # kappa y nitidez controlan de forma directa.
        "razon_escala_cuadrado_extremos_centro_zpos": razon_escala_zpos,
        "razon_escala_cuadrado_extremos_centro_zneg": razon_escala_zneg,
        # Heterocedasticidad condicional en la CURVA REALIZADA: lo que un
        # metodo predictivo observaria. Menor que la anterior por la memoria
        # AR (ver nota arriba); la razon entre ambas mide cuanto sobrevive.
        "razon_varianza_extremos_centro_zpos": razon_zpos,
        "razon_varianza_extremos_centro_zneg": razon_zneg,
        "razon_varianza_extremos_centro_marginal": razon_marginal,
        "contraste_confirmado": contraste_confirmado,
        "correlacion_mecanismo": corr_mecanismo,
        "fraccion_z_positivo": float(np.mean(z_pos)),
    }
    return {**base, **especifico}
