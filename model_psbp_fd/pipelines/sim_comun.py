"""
sim_comun.py
============
Componentes transversales a los cinco algoritmos de generacion de datos del
estudio de simulacion.

Aloja lo que el anexo define una sola vez y todos los algoritmos reutilizan:

    - el esquema de observacion comun (grilla, ruido de medicion, replicas),
    - la cuadratura de los operadores integrales,
    - la construccion y factorizacion de la covarianza de la innovacion
      funcional gaussiana,
    - el registro reproducible de semillas,
    - el diagnostico de control de calidad que no depende del mecanismo
      generador particular.

Lo que NO vive aqui: la dinamica de cada escenario y sus verificaciones
especificas (norma de Hilbert-Schmidt para el Algoritmo 1, radio espectral de
B + Gamma para el Algoritmo 2, acotamiento de la varianza condicional para el
Algoritmo 5), que corresponden a cada modulo `sim_escenario_k.py`.

Convencion de la salida
-----------------------
El unico objeto que debe entregarse a los metodos de estimacion es
`SalidaSimulacion.observaciones`, de dimension (R, T, L). El campo `curvas`
contiene las trayectorias sin ruido de medicion y es de uso exclusivo del
control de calidad del generador; el campo `internos` guarda objetos propios
del mecanismo generador (operadores, regimenes, varianzas condicionales), que
son inobservables por construccion.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Callable, Optional

import numpy as np

__all__ = [
    "ConfigObservacion",
    "SalidaSimulacion",
    "grilla_regular",
    "media_nula",
    "norma_hilbert_schmidt",
    "pesos_trapezoidales",
    "matriz_operador_ar",
    "matriz_covarianza_innovacion",
    "factor_cholesky",
    "generador_innovacion",
    "semillas_replicas",
    "evaluar_media",
    "aplicar_ruido_observacion",
    "diagnostico_comun",
    "guardar_escenario",
    "cargar_escenario",
]


# ==========================================================================
# CONFIGURACION BASE
# ==========================================================================

@dataclass
class ConfigObservacion:
    """
    Parametros del esquema de observacion, comunes a todos los escenarios.

    Las configuraciones especificas de cada algoritmo heredan de esta clase y
    agregan sus propios parametros dinamicos.

    Atributos
    ---------
    L : int
        Numero de puntos de la grilla regular sobre [0, 1].
    T : int
        Numero de curvas retenidas por replica.
    burn_in : int
        Periodos de calentamiento descartados, de modo que la serie retenida
        provenga de la distribucion estacionaria cuando esta existe.
    sigma_obs : float
        Desviacion estandar del ruido de medicion.
    R : int
        Numero de replicas independientes.
    seed : int
        Semilla maestra. Cada replica recibe una semilla derivada mediante
        SeedSequence, de modo que la replica r es identica con independencia
        de cuantas replicas se generen o del orden de ejecucion.
    media_fn : callable, opcional
        Funcion tau -> mu(tau), vectorizada sobre la grilla. Por defecto nula.
    jitter : float
        Jitter base para la factorizacion de matrices de covarianza.
    """

    L: int = 48
    T: int = 200
    burn_in: int = 200
    sigma_obs: float = 0.5
    R: int = 50
    seed: int = 20260719
    media_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None
    jitter: float = 1e-10

    def validar(self) -> None:
        """
        Verifica los parametros del esquema de observacion.

        Las subclases deben llamar a `super().validar()` y anadir sus propias
        verificaciones especificas.
        """
        if self.L < 2:
            raise ValueError("L debe ser al menos 2.")
        if self.T < 2:
            raise ValueError("T debe ser al menos 2.")
        if self.burn_in < 0:
            raise ValueError("burn_in no puede ser negativo.")
        if self.R < 1:
            raise ValueError("R debe ser al menos 1.")
        if self.sigma_obs < 0:
            raise ValueError("sigma_obs no puede ser negativo.")
        if self.jitter <= 0:
            raise ValueError("jitter debe ser positivo.")

    def to_dict(self) -> dict:
        """Version serializable de la configuracion; omite los callables."""
        d = asdict(self)
        for clave, valor in list(d.items()):
            if callable(getattr(self, clave, None)):
                d[clave] = getattr(getattr(self, clave), "__name__", "callable")
            elif valor is None and clave.endswith("_fn"):
                d[clave] = "media_nula" if clave == "media_fn" else None
        return d


@dataclass
class SalidaSimulacion:
    """
    Producto de cualquiera de los generadores.

    observaciones : (R, T, L)
        Matriz de datos con ruido de medicion. UNICO objeto que debe
        entregarse a los metodos de estimacion.
    curvas : (R, T, L)
        Curvas sin ruido de medicion. Uso exclusivo de control de calidad.
    grilla : (L,)
        Puntos tau_l del esquema de observacion.
    media : (L,)
        Evaluacion de mu sobre la grilla.
    semillas : (R,)
        Semilla efectiva empleada por cada replica.
    config : ConfigObservacion
        Configuracion con la que se genero la salida.
    internos : dict
        Objetos propios del mecanismo generador (operadores discretizados,
        secuencias de regimen, varianzas condicionales). Inobservables por
        construccion; nunca deben alimentar la etapa de estimacion.
    diagnostico : dict
        Metricas de control de calidad del generador.
    """

    observaciones: np.ndarray
    curvas: np.ndarray
    grilla: np.ndarray
    media: np.ndarray
    semillas: np.ndarray
    config: ConfigObservacion
    internos: dict = field(default_factory=dict)
    diagnostico: dict = field(default_factory=dict)

    @property
    def forma(self) -> tuple:
        """Dimensiones (R, T, L) de la salida."""
        return self.observaciones.shape


# ==========================================================================
# GRILLA Y FUNCION MEDIA
# ==========================================================================

def grilla_regular(L: int) -> np.ndarray:
    """
    Grilla regular tau_l = (l - 1) / (L - 1), l = 1, ..., L, que incluye ambos
    extremos del dominio: tau_1 = 0 y tau_L = 1.

    La grilla es comun a todas las curvas y replicas, de modo que la columna l
    de la matriz de datos corresponde siempre al mismo punto del dominio; no
    existe desfase de medicion entre periodos consecutivos por construccion.
    """
    if L < 2:
        raise ValueError("L debe ser al menos 2.")
    return np.linspace(0.0, 1.0, L)


def pesos_trapezoidales(tau: np.ndarray) -> np.ndarray:
    """
    Pesos de la cuadratura trapezoidal asociada a la grilla:

        int_0^1 f(s) ds ~= sum_l w_l f(tau_l),

    con w_1 = w_L = h/2 y w_l = h en los puntos interiores para una grilla
    equiespaciada de paso h = 1/(L-1); la formula general por diferencias
    admite tambien grillas no equiespaciadas. Los pesos suman la longitud del
    dominio. Esta regla reemplaza a la rectangular de punto medio, que era la
    natural cuando la grilla no incluia los extremos.
    """
    tau = np.asarray(tau, dtype=float)
    if tau.size < 2:
        raise ValueError("La grilla debe tener al menos 2 puntos.")
    w = np.empty_like(tau)
    w[1:-1] = (tau[2:] - tau[:-2]) / 2.0
    w[0] = (tau[1] - tau[0]) / 2.0
    w[-1] = (tau[-1] - tau[-2]) / 2.0
    return w


def media_nula(tau: np.ndarray) -> np.ndarray:
    """Funcion media por defecto: mu(tau) = 0."""
    return np.zeros_like(tau)


def evaluar_media(
    media_fn: Optional[Callable[[np.ndarray], np.ndarray]],
    tau: np.ndarray,
) -> np.ndarray:
    """
    Evalua la funcion media sobre la grilla, verificando que este vectorizada.

    Un error frecuente es entregar una funcion escalar; en ese caso el
    resultado no tiene la forma de la grilla y la falla se manifestaria mucho
    despues, dentro de la recursion. La verificacion se hace aqui.
    """
    fn = media_fn if media_fn is not None else media_nula
    mu = np.asarray(fn(tau), dtype=float)
    if mu.shape != tau.shape:
        raise ValueError(
            f"media_fn debe retornar un arreglo de forma {tau.shape}, se obtuvo "
            f"{mu.shape}. Verifique que la funcion este vectorizada sobre tau."
        )
    return mu


# ==========================================================================
# OPERADORES INTEGRALES Y CUADRATURA
# ==========================================================================

def norma_hilbert_schmidt(Psi: np.ndarray, pesos: np.ndarray) -> float:
    """
    Norma de Hilbert-Schmidt del operador continuo asociado a la matriz Psi
    discretizada con cuadratura trapezoidal.

    Si Psi[l, l'] = psi(tau_l, tau_l') w_{l'}, con w los pesos de cuadratura,
    entonces el nucleo se recupera como psi(tau_l, tau_l') = Psi[l, l'] / w_{l'}
    y la norma se aproxima por

        ||Psi||_HS^2 = int int psi^2(tau, s) dtau ds
                    ~= sum_{l, l'} psi(tau_l, tau_l')^2 w_l w_{l'}.
    """
    pesos = np.asarray(pesos, dtype=float)
    psi_mat = Psi / pesos[None, :]
    return float(np.sqrt(np.sum((pesos[:, None] * pesos[None, :]) * psi_mat ** 2)))


def matriz_operador_ar(
    tau: np.ndarray,
    gamma: float,
    hs_norm: float,
) -> np.ndarray:
    """
    Matriz Psi (L, L) del operador integral autorregresivo con nucleo gaussiano

        psi(tau, s) = c exp{ -(tau - s)^2 / (2 gamma^2) },

    discretizado por cuadratura trapezoidal sobre la grilla con extremos:

        (int psi(tau, s) f(s) ds)|_{tau_l} ~= sum_{l'} psi(tau_l, tau_l') w_{l'} f(tau_l'),

    con w los pesos trapezoidales de `pesos_trapezoidales`.
    La constante c se calibra de modo que ||Psi||_HS coincida con `hs_norm`.
    Un valor menor que uno garantiza la existencia de una solucion
    estacionaria para el proceso autorregresivo funcional de orden uno.

    Parametros
    ----------
    tau : (L,)
        Grilla de discretizacion.
    gamma : float
        Alcance de la dependencia entre puntos del dominio. Valores pequenos
        concentran la influencia de la curva rezagada en un entorno estrecho
        de cada tau; valores grandes la distribuyen sobre todo el dominio.
    hs_norm : float
        Valor objetivo de la norma de Hilbert-Schmidt.
    """
    if gamma <= 0:
        raise ValueError("gamma debe ser positivo.")
    if hs_norm <= 0:
        raise ValueError("hs_norm debe ser positivo.")
    w = pesos_trapezoidales(tau)
    dist2 = (tau[:, None] - tau[None, :]) ** 2
    nucleo = np.exp(-dist2 / (2.0 * gamma ** 2))       # psi sin la constante c
    hs_nucleo = float(np.sqrt(np.sum((w[:, None] * w[None, :]) * nucleo ** 2)))
    if hs_nucleo <= 0:
        raise RuntimeError("El nucleo discretizado resulto identicamente nulo.")
    c = hs_norm / hs_nucleo                            # calibracion de la constante
    return (c * nucleo) * w[None, :]                   # cuadratura en la variable s


# ==========================================================================
# INNOVACION FUNCIONAL GAUSSIANA
# ==========================================================================

def matriz_covarianza_innovacion(
    tau: np.ndarray,
    sigma_eps: float,
    ell: float,
) -> np.ndarray:
    """
    Matriz de covarianza K (L, L) de la innovacion funcional gaussiana, con
    nucleo exponencial cuadratico

        Cov(eps(tau), eps(s)) = sigma_eps^2 exp{ -(tau - s)^2 / (2 ell^2) }.

    La longitud de correlacion `ell` determina la suavidad de las trayectorias:
    valores pequenos producen innovaciones mas rugosas, valores grandes
    producen innovaciones mas suaves.
    """
    if sigma_eps <= 0:
        raise ValueError("sigma_eps debe ser positivo.")
    if ell <= 0:
        raise ValueError("ell debe ser positivo.")
    dist2 = (tau[:, None] - tau[None, :]) ** 2
    return (sigma_eps ** 2) * np.exp(-dist2 / (2.0 * ell ** 2))


def factor_cholesky(K: np.ndarray, jitter: float = 1e-10) -> np.ndarray:
    """
    Factor triangular de K, con jitter incremental sobre la diagonal.

    El nucleo exponencial cuadratico es notoriamente mal condicionado para
    grillas finas o longitudes de correlacion grandes, de modo que la
    factorizacion directa puede fallar. Se escala el jitter progresivamente
    hasta lograr la factorizacion y, si aun asi falla, se recurre a la
    descomposicion espectral truncando los valores propios negativos
    atribuibles a error numerico.

    El factor retornado A satisface A A^T ~= K, que es lo unico que se requiere
    para generar realizaciones mediante A z con z ~ N(0, I).
    """
    L = K.shape[0]
    escala = float(np.mean(np.diag(K)))
    if escala <= 0:
        raise ValueError("K tiene diagonal no positiva; revise la especificacion.")
    for k in range(12):
        try:
            return np.linalg.cholesky(K + (jitter * (10.0 ** k) * escala) * np.eye(L))
        except np.linalg.LinAlgError:
            continue
    valores, vectores = np.linalg.eigh(K)
    valores = np.clip(valores, 0.0, None)
    return vectores @ np.diag(np.sqrt(valores))


def generador_innovacion(
    chol_K: np.ndarray,
    rng: np.random.Generator,
) -> Callable[[], np.ndarray]:
    """
    Retorna una funcion sin argumentos que extrae una realizacion de la
    innovacion funcional gaussiana con covarianza K.

    Se entrega como clausura para que la recursion de cada escenario invoque
    `innovacion()` sin arrastrar el factor y el generador aleatorio en cada
    llamada, manteniendo legible el cuerpo del bucle temporal.
    """
    L = chol_K.shape[0]

    def innovacion() -> np.ndarray:
        return chol_K @ rng.standard_normal(L)

    return innovacion


# ==========================================================================
# REPRODUCIBILIDAD Y ESQUEMA DE OBSERVACION
# ==========================================================================

def semillas_replicas(seed: int, R: int) -> tuple[list, np.ndarray]:
    """
    Deriva R semillas independientes a partir de una semilla maestra.

    Retorna la lista de objetos SeedSequence, aptos para construir los
    generadores de cada replica, y un arreglo de enteros con el registro
    de las semillas efectivas para su documentacion en los resultados.

    El uso de SeedSequence.spawn garantiza dos propiedades: independencia
    estadistica entre las secuencias de cada replica, y estabilidad de la
    replica r frente a cambios en R o en el orden de ejecucion.
    """
    if R < 1:
        raise ValueError("R debe ser al menos 1.")
    hijas = np.random.SeedSequence(seed).spawn(R)
    registro = np.array(
        [int(s.spawn_key[0]) if s.spawn_key else int(s.entropy) for s in hijas],
        dtype=np.int64,
    )
    return hijas, registro


def aplicar_ruido_observacion(
    curvas: np.ndarray,
    sigma_obs: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Aplica el esquema de observacion comun

        x_{tl} = X_t(tau_l) + epsilon_{tl},  epsilon_{tl} ~ N(0, sigma_obs^2),

    sobre un arreglo de curvas de dimension (T, L). Con sigma_obs = 0 retorna
    una copia sin contaminar, caso util para aislar el efecto del suavizado.
    """
    if sigma_obs == 0:
        return curvas.copy()
    return curvas + rng.normal(0.0, sigma_obs, size=curvas.shape)


# ==========================================================================
# CONTROL DE CALIDAD COMUN
# ==========================================================================

def diagnostico_comun(salida: SalidaSimulacion) -> dict:
    """
    Metricas de control de calidad independientes del mecanismo generador.

    Cada modulo `sim_escenario_k.py` construye su resumen especifico partiendo
    de este diccionario y anadiendo las verificaciones propias de su algoritmo.

    Se reportan: finitud de las trayectorias, media y varianza puntual
    empiricas con su dispersion entre replicas, autocorrelacion puntual a
    rezago uno promediada sobre el dominio con su error estandar de Monte
    Carlo, y la razon senal-ruido efectiva del esquema de observacion.
    """
    X = salida.curvas
    R, T, L = X.shape

    # Autocorrelacion puntual a rezago 1, promediada sobre el dominio.
    acf1 = np.empty(R)
    for r in range(R):
        Z = X[r] - X[r].mean(axis=0, keepdims=True)
        num = np.sum(Z[:-1] * Z[1:], axis=0)
        den = np.sum(Z * Z, axis=0)
        acf1[r] = float(np.nanmean(num / np.where(den > 0, den, np.nan)))

    var_puntual = X.var(axis=1)          # (R, L)
    media_replica = X.mean(axis=(1, 2))  # (R,)
    sigma_obs = salida.config.sigma_obs

    def ee(v: np.ndarray) -> float:
        return float(v.std(ddof=1) / np.sqrt(v.size)) if v.size > 1 else float("nan")

    return {
        "n_replicas": int(R),
        "n_curvas": int(T),
        "n_puntos_grilla": int(L),
        "todo_finito": bool(np.all(np.isfinite(X))),
        "media_empirica_global": float(X.mean()),
        "media_ee_montecarlo": ee(media_replica),
        "var_puntual_media": float(var_puntual.mean()),
        "var_puntual_min": float(var_puntual.min()),
        "var_puntual_max": float(var_puntual.max()),
        "acf1_media": float(acf1.mean()),
        "acf1_ee_montecarlo": ee(acf1),
        "razon_senal_ruido": (
            float(var_puntual.mean() / sigma_obs ** 2)
            if sigma_obs > 0 else float("inf")
        ),
    }


# ==========================================================================
# PERSISTENCIA
# ==========================================================================

def guardar_escenario(
    salida: SalidaSimulacion,
    ruta: str,
    incluir_curvas: bool = True,
    incluir_internos: bool = False,
) -> str:
    """
    Serializa la salida en un archivo .npz comprimido.

    Parametros
    ----------
    incluir_curvas : bool
        Si es True se guardan las curvas sin ruido de medicion. Para el archivo
        que consume la etapa de estimacion conviene False, de modo que ninguna
        cantidad inobservable pueda filtrarse al pipeline por accidente.
    incluir_internos : bool
        Si es True se guardan los objetos internos del generador que sean
        arreglos de NumPy. Solo tiene sentido para inspeccion del generador.
    """
    campos = {
        "observaciones": salida.observaciones,
        "grilla": salida.grilla,
        "media": salida.media,
        "semillas": salida.semillas,
        "config_json": np.array(json.dumps(salida.config.to_dict())),
        "diagnostico_json": np.array(json.dumps(salida.diagnostico)),
    }
    if incluir_curvas:
        campos["curvas"] = salida.curvas
    if incluir_internos:
        for clave, valor in salida.internos.items():
            if isinstance(valor, np.ndarray):
                campos[f"interno_{clave}"] = valor

    if not ruta.endswith(".npz"):
        ruta = ruta + ".npz"
    np.savez_compressed(ruta, **campos)
    return ruta


def cargar_escenario(ruta: str) -> dict:
    """
    Lee un archivo generado por `guardar_escenario` y devuelve un diccionario
    con los arreglos y los metadatos ya deserializados.

    No reconstruye la dataclass de configuracion, dado que esta depende del
    escenario; la configuracion se devuelve como diccionario bajo la clave
    "config", suficiente para documentar y reproducir la corrida.
    """
    datos = np.load(ruta, allow_pickle=False)
    salida = {clave: datos[clave] for clave in datos.files
              if not clave.endswith("_json")}
    if "config_json" in datos.files:
        salida["config"] = json.loads(str(datos["config_json"]))
    if "diagnostico_json" in datos.files:
        salida["diagnostico"] = json.loads(str(datos["diagnostico_json"]))
    return salida
