"""
sim_escenario_T.py
==================
Familia de escenarios con TENDENCIA en el tiempo y no linealidad dentro de la
curva. Escenarios de DIAGNOSTICO: no son Algoritmos del anexo y se nombran con
letra, como el A de la corrida 17 y el B de la 18.

Un solo modulo genera los cuatro, porque comparten la tendencia, el esquema de
observacion, el oraculo y el control de calidad, y difieren en dos ejes
cruzados. Tenerlos en cuatro modulos habria significado cuatro copias de la
tendencia y del diagnostico, que es exactamente la duplicacion que este
proyecto ya pago cara en otros lugares.

    Y_t(tau) = m(Y_{t-1}, S_t)(tau) + eps_t(tau),
    X_t(tau) = mu(tau) + b_{S_t}(t) g(tau) + Y_t(tau),
    x_{tl}   = X_t(tau_l) + e_{tl},        e_{tl} ~ N(0, sigma_obs^2).

Los dos ejes
------------
`forma_tendencia`   "lineal" | "cuadratica"
    b(t) crece hasta acumular `deriva` entre t = 1 y t = T. La lineal la
    reparte de forma uniforme; la cuadratica la concentra al final, de modo que
    el bloque de entrenamiento apenas la ve y el desajuste en T0 es mas
    violento. Es la misma parametrizacion del Escenario A (corrida 17), para
    que las tres corridas sean comparables entre si.

`mecanismo`         "interaccion" | "mezcla"
    "interaccion" -- unimodal. La media condicional suma al operador lineal un
        termino CUADRATICO que hace interactuar puntos del dominio DENTRO de la
        misma curva rezagada:

            m(Y) = Psi Y + lambda * sum_j peso_j s_j tanh((Y(a_j) Y(b_j) - c_j)/s_j) h_j,

        donde Y(a) se lee como el promedio local de la curva en torno a tau = a
        --un producto interno con un nucleo normalizado, no el valor de una
        celda de la grilla-- y h_j es la forma sobre la que se deposita la
        interaccion. Con a = b el termino es el CUADRADO del nivel local, que
        es el caso mas simple de interaccion consigo mismo.

        La saturacion `tanh` NO es cosmetica y hay que declararla al reportar.
        Una recursion bilineal SIN acotar es explosiva con probabilidad
        positiva: el termino crece como ||Y||^2 y, por contractivo que sea Psi,
        basta una excursion grande para que la trayectoria escape --- medido,
        con lambda calibrada a 0.6 de la parte lineal, la serie desborda antes
        de los 600 periodos. La saturacion se fija en `saturacion` desviaciones
        del producto, de modo que en el rango central --- donde ocurre la
        practica totalidad de los origenes --- el termino ES el producto y la
        interaccion es exactamente cuadratica; solo las colas quedan acotadas.
        Con el valor por defecto la correccion afecta a menos del 1 % de los
        instantes, y `fraccion_saturada` lo reporta para que no sea una
        suposicion.

        Ninguna clase lineal puede representarlo: el FAR, el VAR sobre scores y
        el FAR(1) de Bosq son lineales en Y_{t-1}, y un termino bilineal en Y
        es ortogonal a esa clase en la medida en que la ley de Y sea simetrica.
        A diferencia del Escenario B, aqui la no linealidad NO cancela la parte
        lineal: el operador Psi sigue estando y sigue siendo estimable, de modo
        que la referencia lineal no muere, se queda corta. Es el caso realista.

    "mezcla" -- multimodal. Dos ramas con su propio factor sobre el operador Y
        SU PROPIA TENDENCIA:

            m(Y, S) = factor_S * Psi Y,     b_S(t) = deriva_S * perfil(t),

        con S sorteado por un probit sobre la proyeccion del estado rezagado,
        igual que en el Escenario B. La consecuencia es la que el escenario
        existe para producir: la separacion entre las dos modas de la ley
        condicional CRECE CON t, porque las tendencias de las dos ramas
        divergen. En t = 1 las modas coinciden y hacia el final del bloque de
        prueba estan separadas por |deriva_1 - deriva_2| en unidades de X. Un
        modelo unimodal responde con una sola moda en el medio, y el error de
        esa respuesta crece con el tiempo.

De donde sale cada corrida
--------------------------
    corrida 19  Escenario C  mecanismo="interaccion"  forma_tendencia="lineal"
    corrida 22  Escenario D  mecanismo="interaccion"  forma_tendencia="cuadratica"
    corrida 23  Escenario E  mecanismo="mezcla"       forma_tendencia="lineal"
    corrida 24  Escenario F  mecanismo="mezcla"       forma_tendencia="cuadratica"

Es un diseno factorial 2x2: cualquier diferencia entre 19 y 22 (o entre 23 y
24) es atribuible a la FORMA de la tendencia, y cualquier diferencia entre la
fila de arriba y la de abajo, al tipo de no linealidad.

Estacionariedad
---------------
NINGUNO de los cuatro es estacionario: la tendencia lo impide por construccion,
y el diagnostico lo reporta como tal en vez de simular que lo verifica. La
componente Y_t si es estable ---||Psi||_HS < 1, |factor| <= 1 y el termino de
interaccion se calibra a una fraccion de la parte lineal--- y el control de
calidad comprueba que su varianza no crezca entre la primera y la segunda mitad
de la serie, que es la forma operativa de decir que la no linealidad no
desestabilizo la recursion.

La consecuencia practica hay que tenerla presente al leer los resultados, y es
la leccion de la corrida 17: **el centrado del FPCA y del estandarizador se
ajustan con el bloque de entrenamiento y dejan de ser validos en el de prueba**.
Parte del error de test de estos cuatro escenarios es de VIGENCIA DEL CENTRADO
y no de capacidad predictiva, y `resumen_escenario_T` entrega las dos cifras
que permiten separarlas: `r2_oraculo` y `r2_oraculo_destendenciado`.

Por que el R^2 lineal aqui NO es cero
-------------------------------------
En el Escenario B el mejor predictor lineal empataba con la media
incondicional. Aqui no, y no es un defecto: una tendencia hace que el score
rezagado prediga muy bien el contemporaneo ---en el limite es una raiz
unitaria--- de modo que un VAR sobre scores parecera excelente. Ese es
justamente el fenomeno que la corrida 21 encontro en los datos reales
(`acf1 = 0.989`, la persistencia como piso a batir). Por eso el diagnostico
reporta TODO por duplicado, en bruto y sobre el proceso DESTENDENCIADO: la
primera cifra dice lo que se vera en las tablas, la segunda dice cuanto de eso
es la tendencia y cuanto es dinamica.

Uso tipico
----------
    from model_psbp_fd.pipelines import ConfigEscenarioT, generar_escenario_T

    cfg = ConfigEscenarioT(
        L=75, T=400, burn_in=200, R=1, seed=41232, sigma_obs=0.25,
        media_fn=media_senoidal,
        mecanismo="interaccion", forma_tendencia="lineal", deriva=3.0,
    )
    salida = generar_escenario_T(cfg)
    salida.diagnostico["r2_oraculo_destendenciado"]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np
from scipy.stats import norm

from .sim_comun import (
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
from .sim_escenario_B import (
    direccion_oscilatoria,
    coeficiente_sarle_mezcla_simetrica,
)

__all__ = [
    "ConfigEscenarioT",
    "generar_escenario_T",
    "resumen_escenario_T",
    "perfil_tendencia",
    "forma_tendencia_lineal_en_tau",
    "nucleo_local",
]

_MECANISMOS = ("interaccion", "mezcla")
_FORMAS = ("lineal", "cuadratica")


# ==========================================================================
# TENDENCIA
# ==========================================================================

def perfil_tendencia(t_idx: np.ndarray, T: int, forma: str) -> np.ndarray:
    """
    Perfil b(t) / deriva en [0, 1]: fraccion de la deriva total acumulada en t.

    Normalizado de modo que b(1) = 0 y b(T) = deriva, para que `deriva` se lea
    directamente como el desplazamiento total y no dependa de T ni de la forma.
    Es la misma normalizacion del Escenario A (corrida 17), y por eso las tres
    corridas con tendencia son comparables entre si a igual `deriva`.

    La diferencia entre las dos formas esta en DONDE se acumula. Con T0 = 0.7 T:

        lineal      : b(T0) = 0.70 deriva   ->  el entrenamiento ya la vio casi toda
        cuadratica  : b(T0) = 0.49 deriva   ->  la mitad de la deriva ocurre en el test

    y el desfase medio entre bloques ---que es lo que invalida el centrado--- es
    0.35 deriva en la lineal contra 0.44 en la cuadratica. La cuadratica es por
    tanto el caso adverso, no una variante cosmetica.
    """
    if forma not in _FORMAS:
        raise ValueError(f"forma_tendencia no reconocida: {forma!r}. Opciones: {_FORMAS}.")
    u = (np.asarray(t_idx, dtype=float) - 1.0) / (T - 1.0)
    return u if forma == "lineal" else u ** 2


def forma_tendencia_lineal_en_tau(inclinacion: float) -> Callable[[np.ndarray], np.ndarray]:
    """
    Devuelve g(tau) = 1 + inclinacion (tau - 1/2).

    Con inclinacion = 0 la deriva es de NIVEL PURO: identica en todo el dominio,
    y por tanto de rango uno sobre la direccion constante. Ese es el caso base
    porque hace interpretable la descomposicion nivel/forma del error. Con
    inclinacion distinta de cero la deriva ademas inclina la curva y contamina
    la forma, lo cual es mas realista pero mezcla los dos efectos.
    """
    def g(tau: np.ndarray) -> np.ndarray:
        return 1.0 + float(inclinacion) * (np.asarray(tau, dtype=float) - 0.5)
    return g


# ==========================================================================
# LECTURA LOCAL DE LA CURVA
# ==========================================================================

def nucleo_local(tau: np.ndarray, centro: float, ancho: float) -> np.ndarray:
    """
    Nucleo gaussiano centrado en `centro`, normalizado a INTEGRAL UNITARIA con
    la cuadratura del proyecto, de modo que <Y, nucleo> sea el promedio local de
    Y en torno a ese punto y no dependa de la resolucion de la grilla.

    Se lee la curva asi y no por su valor en la celda mas cercana por dos
    razones. La primera es numerica: el valor puntual de una celda es una
    variable con ruido de discretizacion, y elevarlo al cuadrado amplifica ese
    ruido. La segunda es conceptual: el objeto del estudio es una funcion, y una
    interaccion definida sobre valores puntuales no sobrevive a la
    representacion en base ni al truncamiento FPCA. Con `ancho` del orden de la
    escala de suavidad de la innovacion, la lectura local es estable y sigue
    siendo interpretable como "el valor de la curva alrededor de tau = centro".
    """
    tau = np.asarray(tau, dtype=float)
    if not (0.0 <= centro <= 1.0):
        raise ValueError(f"centro={centro} fuera de [0, 1].")
    if ancho <= 0:
        raise ValueError("ancho debe ser positivo.")
    w = pesos_trapezoidales(tau)
    k = np.exp(-((tau - centro) ** 2) / (2.0 * ancho ** 2))
    masa = float(np.sum(w * k))
    if masa <= 0:
        raise RuntimeError("El nucleo local resulto de masa nula.")
    return k / masa


def _forma_unitaria(tau: np.ndarray, centro: float, ancho: float) -> np.ndarray:
    """Bump gaussiano normalizado a norma L^2 unitaria: la forma sobre la que se
    deposita una interaccion. Norma unitaria y no integral unitaria, porque lo
    que debe fijar la escala del termino es su peso y no su anchura."""
    tau = np.asarray(tau, dtype=float)
    w = pesos_trapezoidales(tau)
    h = np.exp(-((tau - centro) ** 2) / (2.0 * ancho ** 2))
    return h / np.sqrt(float(np.sum(w * h * h)))


# ==========================================================================
# CONFIGURACION
# ==========================================================================

@dataclass
class ConfigEscenarioT(ConfigObservacion):
    """
    Parametros de la familia. Hereda el esquema de observacion de
    `ConfigObservacion` y agrega la dinamica, la tendencia y el mecanismo.

    Parte lineal (comun a los cuatro)
        gamma, hs_norm : nucleo del operador y su norma de Hilbert-Schmidt.
        sigma_eps, ell : escala y suavidad de la innovacion funcional.

    Tendencia (comun a los cuatro)
        deriva          : desplazamiento TOTAL acumulado entre t = 1 y t = T, en
                          unidades de X. Con sd(X) ~ 1.2 en el Algoritmo 1, un
                          valor de 3.0 son ~2.5 desviaciones. Es grande a
                          proposito: el escenario debe producir un efecto
                          inequivoco con R = 1.
        forma_tendencia : "lineal" | "cuadratica" (ver `perfil_tendencia`).
        inclinacion     : c de g(tau) = 1 + c(tau - 1/2). Cero = nivel puro.

    Mecanismo "interaccion"
        interacciones      : pares (a, b) de puntos del dominio que interactuan.
                             Por defecto ((0.25, 0.75), (0.50, 0.50)): una
                             interaccion entre dos regiones distintas de la
                             curva y un termino cuadratico consigo misma.
        pesos_interaccion  : peso relativo de cada par. Se normalizan.
        ancho_lectura      : anchura del nucleo con que se lee la curva en cada
                             punto. Del orden de `ell` para que la lectura sea
                             estable.
        ancho_respuesta    : anchura de la forma sobre la que se deposita cada
                             interaccion, centrada en el punto medio del par.
        saturacion         : semiancho de la zona lineal del tanh, en
                             desviaciones del producto medidas en el piloto.
                             Acota la recursion sin alterar el termino en el
                             rango central (ver el encabezado del modulo).
        razon_interaccion  : cuanto pesa el termino cuadratico frente al lineal,
                             medido como cociente de desviaciones L^2. El
                             generador CALIBRA el factor global para alcanzarlo,
                             de modo que este numero ---y no una constante sin
                             unidades--- sea el parametro con significado. 0.6
                             deja una parte lineal todavia dominante, que es el
                             caso interesante: la referencia lineal no muere, se
                             queda corta.

    Mecanismo "mezcla"
        factores_operador  : multiplicador del operador en cada rama. Por
                             defecto (1.0, -1.0), antisimetrico como en el
                             Escenario B.
        derivas_regimen    : multiplicador de la deriva en cada rama. Por
                             defecto (1.0, -1.0): las dos ramas se separan a lo
                             largo del tiempo y la distancia entre las modas
                             CRECE. Es el rasgo propio de estos escenarios.
        nitidez, umbral    : probit sobre z = <Y_{t-1}, e>. Mismo significado
                             que en el Escenario B; `nitidez_en_sd_z` es la
                             cifra interpretable.
        direccion_fn       : direccion e(tau). Por defecto la oscilatoria del
                             Escenario B, que carga sobre la segunda componente
                             principal y produce el punto de corte del barrido
                             en M.

    Diagnostico
        n_dim_diagnostico     : componentes sobre las que se ajusta el mejor
                                predictor lineal del control de calidad.
        prop_train_referencia : solo para el diagnostico, para poder reportar
                                b(T0) y el desfase entre bloques sin que el
                                generador dependa de la particion.
        n_pilot               : longitud de la trayectoria piloto con que se
                                calibra `razon_interaccion`.
    """

    # Parte lineal
    gamma: float = 0.30
    hs_norm: float = 0.70
    sigma_eps: float = 1.0
    ell: float = 0.5

    # Tendencia
    deriva: float = 3.0
    forma_tendencia: str = "lineal"
    inclinacion: float = 0.0

    # Mecanismo
    mecanismo: str = "interaccion"

    # -- "interaccion"
    interacciones: Sequence[tuple] = ((0.25, 0.75), (0.50, 0.50))
    pesos_interaccion: Sequence[float] = (1.0, 1.0)
    ancho_lectura: float = 0.10
    ancho_respuesta: float = 0.15
    razon_interaccion: float = 0.60
    saturacion: float = 3.0

    # -- "mezcla"
    factores_operador: Sequence[float] = (1.0, -1.0)
    derivas_regimen: Sequence[float] = (1.0, -1.0)
    nitidez: float = 4.0
    umbral: float = 0.0
    direccion_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None

    # Diagnostico
    n_dim_diagnostico: int = 5
    prop_train_referencia: float = 0.70
    n_pilot: int = 2000

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
        if self.forma_tendencia not in _FORMAS:
            raise ValueError(
                f"forma_tendencia='{self.forma_tendencia}' invalida. Opciones: {list(_FORMAS)}.")
        if self.mecanismo not in _MECANISMOS:
            raise ValueError(
                f"mecanismo='{self.mecanismo}' invalido. Opciones: {list(_MECANISMOS)}.")

        if self.mecanismo == "interaccion":
            if len(self.interacciones) == 0:
                raise ValueError(
                    "Sin pares en `interacciones` el escenario se reduce al "
                    "Algoritmo 1 con tendencia, que es el Escenario A.")
            if len(self.pesos_interaccion) != len(self.interacciones):
                raise ValueError(
                    f"pesos_interaccion tiene {len(self.pesos_interaccion)} entradas "
                    f"y interacciones {len(self.interacciones)}.")
            for par in self.interacciones:
                if len(par) != 2:
                    raise ValueError(f"El par {par!r} no tiene dos puntos.")
                for x in par:
                    if not (0.0 <= float(x) <= 1.0):
                        raise ValueError(f"El punto {x} del par {par!r} cae fuera de [0, 1].")
            if self.ancho_lectura <= 0 or self.ancho_respuesta <= 0:
                raise ValueError("ancho_lectura y ancho_respuesta deben ser positivos.")
            if self.razon_interaccion <= 0:
                raise ValueError(
                    "razon_interaccion debe ser positivo. Con cero no hay termino "
                    "cuadratico y el escenario coincide con el A.")
            if self.razon_interaccion > 2.0:
                raise ValueError(
                    f"razon_interaccion={self.razon_interaccion} es demasiado grande: el "
                    "termino cuadratico domina al lineal y la recursion pierde "
                    "estabilidad. El control de calidad lo detectaria, pero es mas "
                    "barato no llegar ahi.")
            if self.saturacion <= 0:
                raise ValueError("saturacion debe ser positivo.")
            if self.saturacion > 10.0:
                raise ValueError(
                    f"saturacion={self.saturacion} deja la recursion practicamente sin "
                    "cota y la trayectoria puede desbordar. Ver el encabezado del modulo.")
            if self.n_pilot < 200:
                raise ValueError("n_pilot debe ser al menos 200 para calibrar la escala.")

        else:  # "mezcla"
            for nombre, sec in (("factores_operador", self.factores_operador),
                                ("derivas_regimen", self.derivas_regimen)):
                if len(sec) != 2:
                    raise ValueError(
                        f"{nombre} tiene {len(sec)} entradas; el probit define dos ramas.")
            if np.max(np.abs(np.asarray(self.factores_operador, dtype=float))) > 1.0 + 1e-12:
                raise ValueError(
                    "Ningun factor del operador puede exceder 1 en valor absoluto: la "
                    "cota ||factor Psi||_HS <= ||Psi||_HS es lo que da la estabilidad "
                    "de la componente Y.")
            if np.allclose(self.factores_operador[0], self.factores_operador[1]) and \
               np.allclose(self.derivas_regimen[0], self.derivas_regimen[1]):
                raise ValueError(
                    "Las dos ramas son identicas en operador Y en tendencia: no hay "
                    "mezcla que estimar.")
            if self.nitidez <= 0:
                raise ValueError("nitidez debe ser positivo.")

        if self.n_dim_diagnostico < 1:
            raise ValueError("n_dim_diagnostico debe ser al menos 1.")
        if not (0.0 < self.prop_train_referencia < 1.0):
            raise ValueError("prop_train_referencia debe estar en (0, 1).")


# ==========================================================================
# TERMINO DE INTERACCION
# ==========================================================================

def _armar_interaccion(cfg: ConfigEscenarioT, tau: np.ndarray) -> dict:
    """
    Construye los objetos del termino cuadratico: los nucleos de lectura de cada
    punto, la forma sobre la que se deposita cada par y los pesos normalizados.
    Todo depende solo de la grilla, de modo que se construye una vez y se
    comparte entre replicas.
    """
    w = pesos_trapezoidales(tau)
    pesos = np.asarray(cfg.pesos_interaccion, dtype=float)
    pesos = pesos / float(np.sum(np.abs(pesos)))

    lecturas_a, lecturas_b, formas = [], [], []
    for (a, b) in cfg.interacciones:
        lecturas_a.append(w * nucleo_local(tau, float(a), cfg.ancho_lectura))
        lecturas_b.append(w * nucleo_local(tau, float(b), cfg.ancho_lectura))
        formas.append(_forma_unitaria(tau, 0.5 * (float(a) + float(b)),
                                      cfg.ancho_respuesta))
    return {
        "lecturas_a": np.array(lecturas_a),     # (J, L) ya con la cuadratura
        "lecturas_b": np.array(lecturas_b),     # (J, L)
        "formas": np.array(formas),             # (J, L)
        "pesos": pesos,                         # (J,)
    }


def _producto_interaccion(inter: dict, y: np.ndarray) -> np.ndarray:
    """Productos Y(a_j) Y(b_j) de cada par, para una curva centrada y (L,)."""
    return (inter["lecturas_a"] @ y) * (inter["lecturas_b"] @ y)


def _termino_interaccion(inter: dict, y: np.ndarray, escala: float,
                         centrado: np.ndarray, cota: np.ndarray) -> np.ndarray:
    """
    lambda * sum_j peso_j (Y(a_j) Y(b_j) - c_j) h_j(tau), la contribucion
    cuadratica a la media condicional.

    `cota` es el semiancho de la zona lineal de la saturacion: con |u| << cota
    el tanh es la identidad y el termino es exactamente el producto; con |u|
    grande queda acotado por cota, que es lo que impide que la recursion
    bilineal escape (ver el encabezado del modulo).

    El centrado c_j se resta para que el termino tenga media aproximadamente
    nula: sin el, un termino cuadratico agrega un desplazamiento de nivel
    constante que el FPCA absorberia con la media empirica y que se confundiria
    con la tendencia, que es justo el efecto que estos escenarios quieren medir
    por separado.
    """
    u = _producto_interaccion(inter, y) - centrado             # (J,)
    prod = cota * np.tanh(u / cota)                            # saturacion suave
    return escala * ((inter["pesos"] * prod) @ inter["formas"])


# ==========================================================================
# DINAMICA
# ==========================================================================

def _simular_replica(
    Psi: np.ndarray,
    chol_K: np.ndarray,
    cfg: ConfigEscenarioT,
    inter: Optional[dict],
    escala_inter: float,
    centrado_inter: np.ndarray,
    cota_inter: np.ndarray,
    direccion: Optional[np.ndarray],
    w_quad: np.ndarray,
    rng: np.random.Generator,
) -> dict:
    """
    Itera la recursion de la componente Y y devuelve, ademas de las curvas SIN
    tendencia ni ruido de medicion, las cantidades inobservables que el
    diagnostico y la evaluacion necesitan.

    La tendencia NO entra en la recursion: se suma despues, en
    `generar_escenario_T`. Es una decision de diseno y no un atajo. Si la
    tendencia realimentara la dinamica, el proceso dejaria de tener una
    descomposicion en componente estable mas deriva determinista, y no se
    podria separar el error de vigencia del centrado del error de prediccion
    ---que es la unica lectura que hace interpretables a estos escenarios.
    """
    L = Psi.shape[0]
    innovacion = generador_innovacion(chol_K, rng)

    Y = innovacion()
    # Calentamiento
    for _ in range(cfg.burn_in):
        if cfg.mecanismo == "interaccion":
            Y = Psi @ Y + _termino_interaccion(
                inter, Y, escala_inter, centrado_inter, cota_inter) + innovacion()
        else:
            z = float(np.sum(w_quad * direccion * Y))
            p = float(norm.cdf(cfg.nitidez * (z - cfg.umbral)))
            s = 0 if rng.random() < p else 1
            Y = float(cfg.factores_operador[s]) * (Psi @ Y) + innovacion()

    T = cfg.T
    Y_serie = np.empty((T, L))
    m_cond = np.empty((T, L))          # media condicional de Y (sin tendencia)
    regimen = np.zeros(T, dtype=int)
    prob_r0 = np.full(T, np.nan)
    z_lag = np.full(T, np.nan)
    contrib_inter = np.full(T, np.nan)  # norma L2 del termino cuadratico
    contrib_lineal = np.full(T, np.nan)
    saturado = np.full(T, np.nan)      # fraccion de pares en zona de saturacion

    for t in range(T):
        arrastre = Psi @ Y
        contrib_lineal[t] = float(np.sqrt(np.sum(w_quad * arrastre ** 2)))

        if cfg.mecanismo == "interaccion":
            q = _termino_interaccion(inter, Y, escala_inter, centrado_inter, cota_inter)
            saturado[t] = float(np.mean(
                np.abs(_producto_interaccion(inter, Y) - centrado_inter) > cota_inter))
            contrib_inter[t] = float(np.sqrt(np.sum(w_quad * q ** 2)))
            m_cond[t] = arrastre + q
            Y = m_cond[t] + innovacion()
        else:
            z = float(np.sum(w_quad * direccion * Y))
            p = float(norm.cdf(cfg.nitidez * (z - cfg.umbral)))
            s = 0 if rng.random() < p else 1
            f0, f1 = float(cfg.factores_operador[0]), float(cfg.factores_operador[1])
            m_cond[t] = (p * f0 + (1.0 - p) * f1) * arrastre
            z_lag[t], prob_r0[t], regimen[t] = z, p, s
            Y = (f0 if s == 0 else f1) * arrastre + innovacion()

        Y_serie[t] = Y

    return {
        "Y": Y_serie, "m_cond_Y": m_cond, "regimen": regimen,
        "prob_r0": prob_r0, "z_lag": z_lag,
        "contrib_inter": contrib_inter, "contrib_lineal": contrib_lineal,
        "saturado": saturado,
    }


def _calibrar_interaccion(
    Psi: np.ndarray, chol_K: np.ndarray, cfg: ConfigEscenarioT,
    inter: dict, w_quad: np.ndarray, semilla: int,
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Calibra el factor global del termino cuadratico, el centrado de cada par y
    la cota de saturacion.

    Se simula una trayectoria PILOTO con el termino desactivado, se miden sobre
    ella la desviacion L^2 de la parte lineal y la de la parte cuadratica sin
    escalar, y se fija el factor de modo que su cociente sea
    `razon_interaccion`. El centrado c_j es la media piloto de cada producto.

    El piloto usa el proceso SIN interaccion, de modo que la calibracion es una
    aproximacion de primer orden: al activar el termino la varianza del proceso
    cambia y la razon efectiva se desvia algo de la pedida. Por eso
    `resumen_escenario_T` reporta `razon_interaccion_efectiva`, medida sobre la
    serie definitiva, y no da por buena la nominal. La alternativa ---iterar la
    calibracion hasta punto fijo--- multiplicaria el costo por poco: con
    `razon_interaccion` en torno a 0.6 la desviacion medida es de pocos puntos
    porcentuales.
    """
    rng = np.random.default_rng(semilla)
    innovacion = generador_innovacion(chol_K, rng)
    Y = innovacion()
    for _ in range(cfg.burn_in):
        Y = Psi @ Y + innovacion()

    n = int(cfg.n_pilot)
    prods = np.empty((n, len(inter["pesos"])))
    lin = np.empty(n)
    for i in range(n):
        arrastre = Psi @ Y
        lin[i] = float(np.sqrt(np.sum(w_quad * arrastre ** 2)))
        prods[i] = _producto_interaccion(inter, Y)
        Y = arrastre + innovacion()

    centrado = prods.mean(axis=0)
    cota = float(cfg.saturacion) * prods.std(axis=0)
    cota = np.where(cota > 0, cota, 1.0)
    u = prods - centrado
    bruto = (inter["pesos"] * (cota * np.tanh(u / cota))) @ inter["formas"]   # (n, L)
    sd_bruto = float(np.sqrt(np.mean(np.sum(w_quad * bruto ** 2, axis=1))))
    sd_lin = float(np.sqrt(np.mean(lin ** 2)))
    if sd_bruto <= 0:
        raise RuntimeError(
            "El termino de interaccion resulto identicamente nulo en el piloto: "
            "revise `interacciones` y `ancho_lectura`.")
    return float(cfg.razon_interaccion * sd_lin / sd_bruto), centrado, cota


# ==========================================================================
# GENERADOR PRINCIPAL
# ==========================================================================

def generar_escenario_T(cfg: ConfigEscenarioT) -> SalidaSimulacion:
    """
    Genera R replicas independientes. Los objetos que no dependen de la
    realizacion ---operador, factorizacion de la innovacion, nucleos de
    interaccion, calibracion, perfil de tendencia--- se construyen una sola vez.
    """
    cfg.validar()

    tau = grilla_regular(cfg.L)
    w_quad = pesos_trapezoidales(tau)
    mu = evaluar_media(cfg.media_fn, tau)

    Psi = matriz_operador_ar(tau, cfg.gamma, cfg.hs_norm)
    K = matriz_covarianza_innovacion(tau, cfg.sigma_eps, cfg.ell)
    chol_K = factor_cholesky(K, cfg.jitter)

    inter = escala_inter = centrado_inter = cota_inter = None
    direccion = None
    if cfg.mecanismo == "interaccion":
        inter = _armar_interaccion(cfg, tau)
        escala_inter, centrado_inter, cota_inter = _calibrar_interaccion(
            Psi, chol_K, cfg, inter, w_quad, semilla=cfg.seed + 7919)
    else:
        direccion = (direccion_oscilatoria(tau) if cfg.direccion_fn is None
                     else _normalizar(cfg.direccion_fn(tau), tau))
        escala_inter, centrado_inter, cota_inter = 0.0, np.zeros(1), np.ones(1)

    # Tendencia: perfil temporal y forma sobre el dominio
    t_idx = np.arange(1, cfg.T + 1)
    perfil = perfil_tendencia(t_idx, cfg.T, cfg.forma_tendencia)          # (T,)
    g_tau = forma_tendencia_lineal_en_tau(cfg.inclinacion)(tau)           # (L,)

    hijas, registro = semillas_replicas(cfg.seed, cfg.R)

    curvas = np.empty((cfg.R, cfg.T, cfg.L))
    observaciones = np.empty((cfg.R, cfg.T, cfg.L))
    medias_cond = np.empty((cfg.R, cfg.T, cfg.L))
    tendencias = np.empty((cfg.R, cfg.T, cfg.L))
    tendencias_det = np.empty((cfg.R, cfg.T, cfg.L))
    regimenes = np.zeros((cfg.R, cfg.T), dtype=int)
    probs_r0 = np.full((cfg.R, cfg.T), np.nan)
    z_lags = np.full((cfg.R, cfg.T), np.nan)
    contrib_inter = np.full((cfg.R, cfg.T), np.nan)
    contrib_lineal = np.full((cfg.R, cfg.T), np.nan)
    saturado = np.full((cfg.R, cfg.T), np.nan)

    for r, semilla in enumerate(hijas):
        rng = np.random.default_rng(semilla)
        out = _simular_replica(Psi, chol_K, cfg, inter, escala_inter,
                               centrado_inter, cota_inter, direccion, w_quad, rng)

        if not np.all(np.isfinite(out["Y"])):
            raise RuntimeError(
                "La recursion desbordo: la componente Y dejo de ser finita. Con "
                "mecanismo='interaccion' significa que el termino cuadratico domina; "
                "bajar `razon_interaccion` o `saturacion`.")

        b_t = cfg.deriva * perfil                                    # (T,)
        if cfg.mecanismo == "mezcla":
            # La tendencia depende del regimen VIGENTE, y por eso la separacion
            # entre las dos modas crece con t: es el rasgo del escenario.
            f = np.asarray(cfg.derivas_regimen, dtype=float)[out["regimen"]]
            tend = np.outer(b_t * f, g_tau)
            # Componente determinista de la tendencia: la esperada bajo la
            # ocupacion de las ramas. Es funcion de t solamente, de modo que
            # restarla NO altera la relacion entre la media condicional y el
            # pasado --- que es la condicion para que el diagnostico
            # destendenciado siga midiendo lo que dice medir. Restar la
            # tendencia REALIZADA rompe esa propiedad, porque depende del
            # regimen y por tanto de la propia variable a predecir.
            f_med = float(np.mean(np.asarray(cfg.derivas_regimen, dtype=float)[out["regimen"]]))
            tend_det = np.outer(b_t * f_med, g_tau)
            # La media condicional incorpora la tendencia ESPERADA bajo el
            # probit, que es la mezcla de las dos ramas, no la realizada.
            p0 = out["prob_r0"]
            f_esp = p0 * float(cfg.derivas_regimen[0]) + (1 - p0) * float(cfg.derivas_regimen[1])
            tend_esp = np.outer(b_t * f_esp, g_tau)
        else:
            tend = np.outer(b_t, g_tau)
            tend_esp = tend
            tend_det = tend

        X_true = mu + tend + out["Y"]
        curvas[r] = X_true
        medias_cond[r] = mu + tend_esp + out["m_cond_Y"]
        tendencias[r] = tend
        tendencias_det[r] = tend_det
        regimenes[r] = out["regimen"]
        probs_r0[r] = out["prob_r0"]
        z_lags[r] = out["z_lag"]
        contrib_inter[r] = out["contrib_inter"]
        contrib_lineal[r] = out["contrib_lineal"]
        saturado[r] = out["saturado"]
        observaciones[r] = aplicar_ruido_observacion(X_true, cfg.sigma_obs, rng)

    internos = {
        "operador": Psi,
        "cov_innovacion": K,
        "media_condicional": medias_cond,
        "tendencia": tendencias,
        "tendencia_determinista": tendencias_det,
        "perfil_tendencia": perfil,
        "forma_tendencia_tau": g_tau,
        "contribucion_interaccion": contrib_inter,
        "fraccion_saturada": saturado,
        "contribucion_lineal": contrib_lineal,
        "pesos_cuadratura": w_quad,
    }
    if cfg.mecanismo == "mezcla":
        internos.update({
            "direccion_estado": direccion,
            "regimenes": regimenes,
            "prob_regimen_0": probs_r0,
            "proyeccion_estado": z_lags,
        })
    else:
        internos.update({
            "escala_interaccion": np.array(escala_inter),
            "centrado_interaccion": np.asarray(centrado_inter),
            "cota_saturacion": np.asarray(cota_inter),
            "lecturas_a": inter["lecturas_a"],
            "lecturas_b": inter["lecturas_b"],
            "formas_interaccion": inter["formas"],
        })

    salida = SalidaSimulacion(
        observaciones=observaciones, curvas=curvas, grilla=tau, media=mu,
        semillas=registro, config=cfg, internos=internos,
    )
    salida.diagnostico = resumen_escenario_T(salida)
    return salida


def _normalizar(e: np.ndarray, tau: np.ndarray) -> np.ndarray:
    """Normaliza una direccion a norma unitaria en L^2 con la cuadratura."""
    e = np.asarray(e, dtype=float).ravel()
    if e.shape != tau.shape:
        raise ValueError(f"direccion_fn retorno forma {e.shape}; se esperaba {tau.shape}.")
    w = pesos_trapezoidales(tau)
    norma = float(np.sqrt(np.sum(w * e * e)))
    if norma <= 0:
        raise ValueError("direccion_fn produjo una funcion de norma nula.")
    return e / norma


# ==========================================================================
# CONTROL DE CALIDAD
# ==========================================================================

def _fpca_empirica(Y: np.ndarray, w: np.ndarray, n_dim: int) -> tuple[np.ndarray, np.ndarray]:
    """Autofunciones del problema generalizado C u = lambda W u sobre la muestra
    centrada Y (T, L), con la cuadratura del proyecto. Diagnostico interno del
    generador: no es la FPCA del pipeline, que se ajusta solo con train."""
    n_dim = int(min(n_dim, Y.shape[1]))
    raiz = np.sqrt(w)
    Cov = Y.T @ Y / Y.shape[0]
    Sim = (raiz[:, None] * Cov) * raiz[None, :]
    lam, V = np.linalg.eigh(Sim)
    orden = np.argsort(lam)[::-1][:n_dim]
    return np.maximum(lam[orden], 0.0), V[:, orden] / raiz[:, None]


def _r2_lineal_y_oraculo(Y: np.ndarray, M: np.ndarray, w: np.ndarray,
                         n_dim: int) -> tuple[float, float, float]:
    """
    R^2 del mejor predictor lineal (ajustado en la primera mitad, evaluado en la
    segunda) y de la media condicional verdadera, ambos sobre los scores de las
    primeras `n_dim` componentes principales empiricas. Devuelve tambien el R^2
    lineal DENTRO de muestra, que se reporta aparte porque esta inflado por
    sobreajuste y es el sesgo que haria parecer favorable a un escenario sin
    serlo.
    """
    Yc = Y - Y.mean(axis=0, keepdims=True)
    Mc = M - M.mean(axis=0, keepdims=True)
    _, U = _fpca_empirica(Yc, w, n_dim)
    proy = w[:, None] * U
    S, Sm = Yc @ proy, Mc @ proy
    X0, X1, Mo = S[:-1], S[1:], Sm[1:]
    n = X1.shape[0]
    corte = n // 2
    D = np.column_stack([np.ones(n), X0])
    coef, *_ = np.linalg.lstsq(D[:corte], X1[:corte], rcond=None)
    sct = float(np.sum((X1[corte:] - X1[:corte].mean(axis=0)) ** 2))
    r2_lin = 1.0 - float(np.sum((X1[corte:] - D[corte:] @ coef) ** 2)) / max(sct, 1e-300)
    r2_orc = 1.0 - float(np.sum((X1[corte:] - Mo[corte:]) ** 2)) / max(sct, 1e-300)
    coef_in, *_ = np.linalg.lstsq(D, X1, rcond=None)
    r2_in = 1.0 - float(np.sum((X1 - D @ coef_in) ** 2)) / max(
        float(np.sum((X1 - X1.mean(axis=0)) ** 2)), 1e-300)
    return r2_lin, r2_orc, r2_in


def resumen_escenario_T(salida: SalidaSimulacion) -> dict:
    """
    Control de calidad del generador, previo a cualquier ajuste. Cuatro bloques.

    Tendencia. Se regresa el nivel puntual de cada curva sobre b(t) y se
    comprueba que la pendiente recupere el promedio de g(tau): es la
    verificacion de que se inyecto lo que se pidio, analoga a la del Escenario A.
    Se reportan ademas `b_en_T0` y `desfase_train_test`, que es la cantidad que
    invalida el centrado del FPCA, y su magnitud en desviaciones del proceso.

    Estabilidad de la componente sin tendencia. El escenario NO es estacionario
    ---la tendencia lo impide--- pero Y_t debe serlo. Se compara su varianza
    entre la primera y la segunda mitad: `razon_varianza_Y_mitades` lejos de 1
    significa que la no linealidad desestabilizo la recursion, y entonces las
    conclusiones no serian sobre la no linealidad sino sobre una explosion.

    Cuanto pierde el mejor predictor lineal, POR DUPLICADO. En bruto y sobre el
    proceso destendenciado. La primera cifra es la que se vera en las tablas del
    `_05`; la segunda dice cuanto de la aparente habilidad del lineal es la
    tendencia. Con una deriva grande, un VAR sobre scores puede alcanzar un R^2
    alto sin haber aprendido nada de la dinamica: es el fenomeno que la corrida
    21 encontro en los datos reales, y aqui esta puesto a proposito.

    Lo propio del mecanismo. Para "interaccion", la razon efectiva entre el
    termino cuadratico y el lineal, que es el parametro con significado del
    escenario. Para "mezcla", la ocupacion de las ramas, la ambiguedad y ---lo
    que ningun otro escenario tiene--- la separacion entre modas al PRINCIPIO y
    al FINAL de la serie, junto con el coeficiente de Sarle exacto de la ley
    condicional verdadera en ambos extremos.
    """
    if not isinstance(salida.config, ConfigEscenarioT):
        raise TypeError(
            "resumen_escenario_T requiere una salida generada con ConfigEscenarioT; "
            f"se recibio {type(salida.config).__name__}.")

    base = diagnostico_comun(salida)
    cfg = salida.config
    w = salida.internos["pesos_cuadratura"]
    Psi = salida.internos["operador"]
    K = salida.internos["cov_innovacion"]
    medias = salida.internos["media_condicional"]
    tend = salida.internos["tendencia"]
    perfil = salida.internos["perfil_tendencia"]
    g_tau = salida.internos["forma_tendencia_tau"]

    R, T, L = salida.curvas.shape
    T0 = int(np.floor(cfg.prop_train_referencia * T))
    b_t = cfg.deriva * perfil
    g_medio = float(np.mean(g_tau))

    # ── Tendencia ───────────────────────────────────────────────────────────
    nivel = salida.curvas[0].mean(axis=1)
    Z = np.column_stack([np.ones(T), b_t])
    coef = np.linalg.lstsq(Z, nivel, rcond=None)[0]
    resid = nivel - Z @ coef
    s2 = float(resid @ resid) / (T - 2)
    ee_pend = float(np.sqrt(s2 * np.linalg.inv(Z.T @ Z)[1, 1]))
    # El residuo de esa regresion es la componente Y promediada sobre tau, que
    # es fuertemente autocorrelacionada: el error estandar de MCO la subestima y
    # con el la pendiente parece significativamente distinta de la esperada
    # cuando no lo es. Se corrige por tamano muestral efectivo con la
    # autocorrelacion de primer orden del residuo, que es la correccion minima
    # honesta; el `assert` del notebook usa ESTA cifra y no la de MCO.
    r1 = float(np.corrcoef(resid[:-1], resid[1:])[0, 1]) if T > 3 else 0.0
    r1 = min(max(r1, 0.0), 0.99)
    ee_pend_ac = float(ee_pend * np.sqrt((1.0 + r1) / (1.0 - r1)))

    # Con "mezcla" la tendencia realizada alterna de signo con el regimen, de
    # modo que la pendiente esperada NO es g_medio sino su promedio ponderado
    # por la ocupacion de las ramas. Se reporta la referencia correcta para
    # cada mecanismo en vez de una sola que solo valdria en un caso.
    if cfg.mecanismo == "mezcla":
        f = np.asarray(cfg.derivas_regimen, dtype=float)
        ocup = np.array([float(np.mean(salida.internos["regimenes"] == j)) for j in (0, 1)])
        pend_esperada = g_medio * float(ocup @ f)
    else:
        pend_esperada = g_medio

    sd_sin_tend = float(np.std(salida.curvas[0] - tend[0]))
    desfase = float(b_t[T0:].mean() - b_t[:T0].mean())

    # ── Estabilidad de Y ────────────────────────────────────────────────────
    Y_sin = salida.curvas[0] - tend[0] - salida.media
    v1 = float(np.mean(np.var(Y_sin[: T // 2], axis=0)))
    v2 = float(np.mean(np.var(Y_sin[T // 2:], axis=0)))

    # ── Lineal vs oraculo, en bruto y destendenciado ───────────────────────
    r2_lin, r2_orc, r2_in = _r2_lineal_y_oraculo(
        salida.curvas[0], medias[0], w, cfg.n_dim_diagnostico)
    tend_det = salida.internos["tendencia_determinista"]
    r2_lin_d, r2_orc_d, _ = _r2_lineal_y_oraculo(
        salida.curvas[0] - tend_det[0], medias[0] - tend_det[0], w, cfg.n_dim_diagnostico)

    especifico = {
        # Operador
        "hs_norm_objetivo": float(cfg.hs_norm),
        "hs_norm_efectiva": float(norma_hilbert_schmidt(Psi, w)),
        "radio_espectral": float(np.max(np.abs(np.linalg.eigvals(Psi)))),
        "estacionariedad_garantizada": False,
        "motivo_no_estacionario": "tendencia determinista en t (por construccion)",
        # Tendencia
        "mecanismo": cfg.mecanismo,
        "forma_tendencia": cfg.forma_tendencia,
        "deriva_total": float(cfg.deriva),
        "inclinacion": float(cfg.inclinacion),
        "g_medio": g_medio,
        "pendiente_recuperada": float(coef[1]),
        "pendiente_esperada": float(pend_esperada),
        "pendiente_ee": ee_pend,
        "pendiente_ee_autocorr": ee_pend_ac,
        "acf1_residuo_tendencia": r1,
        "desvio_pendiente_en_ee": float(abs(coef[1] - pend_esperada) / max(ee_pend_ac, 1e-300)),
        "b_en_T0": float(b_t[T0 - 1]),
        "b_en_T": float(b_t[-1]),
        "desfase_train_test": desfase,
        "sd_proceso_sin_tendencia": sd_sin_tend,
        "razon_deriva_sd": float(cfg.deriva / max(sd_sin_tend, 1e-300)),
        "desfase_en_sd": float(desfase / max(sd_sin_tend, 1e-300)),
        # Estabilidad
        "var_Y_primera_mitad": v1,
        "var_Y_segunda_mitad": v2,
        "razon_varianza_Y_mitades": float(v2 / max(v1, 1e-300)),
        # Lineal vs oraculo
        "n_dim_diagnostico": int(cfg.n_dim_diagnostico),
        "r2_lineal_fuera_de_muestra": r2_lin,
        "r2_lineal_dentro_de_muestra": r2_in,
        "r2_oraculo_fuera_de_muestra": r2_orc,
        "brecha_oraculo_lineal": float(r2_orc - r2_lin),
        "r2_lineal_destendenciado": r2_lin_d,
        "r2_oraculo_destendenciado": r2_orc_d,
        "brecha_oraculo_lineal_destendenciada": float(r2_orc_d - r2_lin_d),
    }

    if cfg.mecanismo == "interaccion":
        ci = salida.internos["contribucion_interaccion"][0]
        cl = salida.internos["contribucion_lineal"][0]
        especifico.update({
            "interacciones": [[float(a), float(b)] for (a, b) in cfg.interacciones],
            "razon_interaccion_objetivo": float(cfg.razon_interaccion),
            "razon_interaccion_efectiva": float(
                np.sqrt(np.mean(ci ** 2)) / max(float(np.sqrt(np.mean(cl ** 2))), 1e-300)),
            "escala_interaccion_calibrada": float(salida.internos["escala_interaccion"]),
            "contribucion_interaccion_media_L2": float(np.mean(ci)),
            "contribucion_lineal_media_L2": float(np.mean(cl)),
            "saturacion": float(cfg.saturacion),
            "cota_saturacion": [float(x) for x in np.atleast_1d(
                salida.internos["cota_saturacion"])],
            "fraccion_saturada": float(np.nanmean(salida.internos["fraccion_saturada"])),
        })
    else:
        reg = salida.internos["regimenes"]
        p0 = salida.internos["prob_regimen_0"]
        z = salida.internos["proyeccion_estado"]
        ambiguo = (p0 > 0.25) & (p0 < 0.75)
        transiciones = float(np.mean([int(np.sum(np.diff(reg[r]) != 0)) for r in range(R)]))

        # Separacion entre las dos modas: la parte dinamica no depende de t, la
        # de tendencia si, y esa es la novedad de estos escenarios.
        Y0 = salida.curvas[0] - tend[0] - salida.media
        arr = Y0[:-1] @ Psi.T
        df = abs(float(cfg.factores_operador[0]) - float(cfg.factores_operador[1]))
        sep_din = float(np.mean(np.sqrt(np.sum(w * (df * arr) ** 2, axis=1))))
        dd = abs(float(cfg.derivas_regimen[0]) - float(cfg.derivas_regimen[1]))
        norma_g = float(np.sqrt(np.sum(w * g_tau ** 2)))
        sep_tend_T0 = dd * float(b_t[T0 - 1]) * norma_g
        sep_tend_T = dd * float(b_t[-1]) * norma_g
        sd_innov_L2 = float(np.sqrt(max(float(np.sum(w * np.diag(K))), 0.0)))

        # Sarle exacto de la ley condicional verdadera sobre la primera
        # componente, al principio y al final de la serie: la referencia oracle.
        _, U0 = _fpca_empirica(Y0 - Y0.mean(axis=0, keepdims=True), w, 1)
        u1 = U0[:, 0]
        a_din = (arr @ (w * u1)) * (df / 2.0)
        a_tend = (dd / 2.0) * b_t[1:] * float(np.sum(w * g_tau * u1))
        sd_u1 = float(np.sqrt(max(float(u1 @ ((w[:, None] * K) * w[None, :]) @ u1), 0.0)))
        b_sarle = coeficiente_sarle_mezcla_simetrica(p0[0][1:], a_din + a_tend, sd_u1)
        amb1 = ambiguo[0][1:]
        primera = np.arange(len(b_sarle)) < (T0 - 1)

        especifico.update({
            "factores_operador": [float(x) for x in cfg.factores_operador],
            "derivas_regimen": [float(x) for x in cfg.derivas_regimen],
            "proporcion_regimen_0": float(np.mean(reg == 0)),
            "n_transiciones_media": transiciones,
            "duracion_media_racha": float(T / (transiciones + 1.0)),
            "sd_proyeccion_estado": float(np.std(z)),
            "nitidez": float(cfg.nitidez),
            "nitidez_en_sd_z": float(cfg.nitidez * float(np.std(z))),
            "fraccion_origenes_ambiguos": float(np.mean(ambiguo)),
            "n_origenes_ambiguos": int(np.sum(ambiguo) / R),
            "separacion_modas_dinamica_L2": sep_din,
            "separacion_modas_tendencia_en_T0": sep_tend_T0,
            "separacion_modas_tendencia_en_T": sep_tend_T,
            "sd_innovacion_L2": sd_innov_L2,
            "separacion_total_en_sd_T0": float(
                (sep_din + sep_tend_T0) / max(sd_innov_L2, 1e-300)),
            "separacion_total_en_sd_T": float(
                (sep_din + sep_tend_T) / max(sd_innov_L2, 1e-300)),
            "sarle_oraculo_ambiguos_train": float(np.nanmean(b_sarle[amb1 & primera]))
                if (amb1 & primera).any() else float("nan"),
            "sarle_oraculo_ambiguos_test": float(np.nanmean(b_sarle[amb1 & ~primera]))
                if (amb1 & ~primera).any() else float("nan"),
            "sarle_oraculo_deterministas_test": float(np.nanmean(b_sarle[~amb1 & ~primera]))
                if (~amb1 & ~primera).any() else float("nan"),
            "sarle_referencia_uniforme": 5.0 / 9.0,
            "sarle_referencia_gaussiana": 1.0 / 3.0,
        })

    return {**base, **especifico}
