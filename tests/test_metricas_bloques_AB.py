"""
test_metricas_bloques_AB.py
===========================
Pruebas de las metricas del Bloque A (normas L^p del error) y del Bloque B
(intervalos de prediccion), y de su enganche con la ventana movil.

Como se corren
--------------
No hay suite previa en el repositorio ni configuracion de pytest, de modo que
estas pruebas se ejecutan invocando pytest directamente sobre el archivo, desde
la raiz del proyecto:

    python -m pytest tests/test_metricas_bloques_AB.py -v

El archivo tambien es ejecutable sin pytest (`python tests/test_metricas_bloques_AB.py`),
que corre las mismas comprobaciones e imprime las cifras. Se escribio asi
siguiendo `test_far_operador.py`, que es la unica prueba previa del repositorio.

Que se comprueba, y por que estas cosas y no otras
--------------------------------------------------
1. La cadena  ||e||_1 <= ||e||_2 <= ||e||_inf  se cumple sobre datos reales y
   FALLA con assert cuando los pesos del dominio no estan normalizados a masa
   uno. Es la unica forma de que la verificacion sirva: un assert que nunca se
   ha visto fallar no prueba que detecte nada.
2. Los valores del Bloque B contra cuentas hechas a mano: el Winkler de un
   punto dentro del intervalo es exactamente el ancho, y el de uno fuera suma
   (2/alpha) veces la distancia.
3. EL CASO M = 1, que es un punto del barrido y donde el repositorio tiene tres
   trampas de forma documentadas (colapso del eje singleton en los `.mat`,
   `np.cov` devolviendo un escalar 0-d y `np.loadtxt` colapsando a 1D). Aqui se
   ejercita la consecuencia que afecta a estas metricas: que las entradas
   lleguen como (n,) en vez de (n, 1) y que la tabla de la ventana movil salga
   igual de bien en ambos casos.
4. Que la ventana movil sigue emitiendo sus columnas de marco --t_ini, t_fin,
   t_centro, bloque, n_ventana, cruza_T0-- despues de la extension: es lo que
   consumen `30_04` y el `tabla_ventana` de `30_05` §6, y romperlo seria
   romperlos a ellos.
5. Que `rmse_f` de la ventana es la RAIZ DEL MSE AGREGADO y no el promedio de
   los RMSE por origen. Las dos cifras se emiten (`rmse_f` y `l2_medio`) y la
   prueba fija cual es cual, que es justo la ambiguedad que hay que cerrar.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

try:
    import pytest
except ModuleNotFoundError:                      # ejecucion directa sin pytest
    class _Pytest:                               # stub minimo: solo lo usado
        @staticmethod
        def raises(exc):
            class _Ctx:
                def __enter__(self_inner):
                    return self_inner

                def __exit__(self_inner, tipo, val, tb):
                    if tipo is None:
                        raise AssertionError(f"No se lanzo {exc.__name__}.")
                    return issubclass(tipo, exc)
            return _Ctx()
    pytest = _Pytest()                           # type: ignore

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model_psbp_fd.fit import (                                    # noqa: E402
    bootstrap_bloques, cuantil_error_absoluto, error_maximo,
    largo_bloque_sugerido, mae, mise, normas_error_por_origen,
    pesos_normalizados, picp, resumen_error_funcional, resumen_intervalo,
    ventana_movil_funcional, ventana_movil_scores, winkler,
)
from model_psbp_fd.fit.metrics_puntual import verificar_cadena_lp   # noqa: E402
from model_psbp_fd.utils.quadrature import pesos_trapezoidales      # noqa: E402


# ==========================================================================
# DATOS SINTETICOS
# ==========================================================================

def _datos(n=60, G=25, M=2, seed=7, dominio=(0.0, 1.0)):
    """Curvas, prediccion y banda. `dominio` permite salirse de [0, 1]."""
    rng = np.random.default_rng(seed)
    tau = np.linspace(dominio[0], dominio[1], G)
    X = np.sin(2 * np.pi * tau)[None, :] + rng.normal(0, 0.4, size=(n, G))
    Xhat = X + rng.normal(0, 0.3, size=(n, G))
    sd = 0.3
    li, ls = Xhat - 1.96 * sd, Xhat + 1.96 * sd
    Y = rng.normal(size=(n, M))
    Yhat = Y + rng.normal(0, 0.5, size=(n, M))
    return tau, X, Xhat, li, ls, Y, Yhat


# ==========================================================================
# BLOQUE A
# ==========================================================================

def test_cadena_lp_se_cumple():
    """||e||_1 <= ||e||_2 <= ||e||_inf, origen a origen, sobre [0, 1]."""
    tau, X, Xhat, *_ = _datos()
    d = normas_error_por_origen(X, Xhat, tau)
    assert np.all(d["l1"] <= d["l2"] + 1e-12)
    assert np.all(d["l2"] <= d["linf"] + 1e-12)
    assert np.all(d["razon_linf_l1"] >= 1.0 - 1e-12)


def test_cadena_lp_se_cumple_fuera_de_0_1():
    """
    La cadena vale tambien con dominio [0, 4].

    Es la comprobacion que justifica `pesos_normalizados`: con los pesos
    trapezoidales CRUDOS --que suman la longitud del dominio-- la desigualdad
    L^1 <= L^2 se invierte en cuanto esa longitud pasa de uno, porque deja de
    ser una medida de probabilidad y Jensen no aplica.
    """
    tau, X, Xhat, *_ = _datos(dominio=(0.0, 4.0))
    d = normas_error_por_origen(X, Xhat, tau)
    assert np.all(d["l1"] <= d["l2"] + 1e-12)

    # Y sin normalizar, la cadena se rompe: el assert tiene que detectarlo.
    w_crudo = pesos_trapezoidales(tau)
    E = np.abs(Xhat - X)
    l1_mal = E @ w_crudo
    l2_mal = np.sqrt((E ** 2) @ w_crudo)
    with pytest.raises(AssertionError):
        verificar_cadena_lp(l1_mal, l2_mal, E.max(axis=1))


def test_pesos_normalizados_suman_uno():
    tau, *_ = _datos()
    w = pesos_normalizados(tau)
    assert abs(float(w.sum()) - 1.0) < 1e-12
    # Con una ponderacion no uniforme del dominio siguen sumando uno.
    v = np.linspace(0.5, 2.0, tau.size)
    assert abs(float(pesos_normalizados(tau, v).sum()) - 1.0) < 1e-12


def test_orden_de_agregacion_del_rmse():
    """
    `rmse_f` es la RAIZ DEL MSE AGREGADO; `l2_medio` el promedio de RMSE.

    Por Jensen la segunda es menor o igual, con igualdad solo si el error es
    constante entre origenes. La prueba fija cual es cual para que nadie las
    intercambie mas adelante.
    """
    tau, X, Xhat, *_ = _datos()
    r = resumen_error_funcional(X, Xhat, tau)
    assert r["l2_medio"] <= r["rmse_f"] + 1e-12
    assert 0.0 < r["razon_agregacion"] <= 1.0 + 1e-12
    # rmse_f coincide con sqrt(mise), que es la convencion del resto del
    # capitulo: la cuadratura normalizada de un dominio de longitud uno es la
    # misma integral.
    assert abs(r["rmse_f"] - np.sqrt(mise(X, Xhat, tau))) < 1e-10


def test_mae_y_extremos_escalares():
    tau, X, Xhat, *_ = _datos()
    assert mae(X, Xhat) > 0
    q95 = cuantil_error_absoluto(X, Xhat, 0.95)
    ext = error_maximo(X, Xhat)
    assert q95 <= ext["linf"] + 1e-12
    assert len(ext["argmax"]) == 2                # (origen, punto de la grilla)


# ==========================================================================
# BLOQUE B
# ==========================================================================

def test_winkler_a_mano():
    """Dentro del intervalo el puntaje es el ancho; fuera suma la distancia."""
    nivel = 0.95
    alpha = 1.0 - nivel
    li = np.array([0.0, 0.0, 0.0])
    ls = np.array([2.0, 2.0, 2.0])
    y = np.array([1.0, -0.5, 3.0])                # dentro, por debajo, arriba
    w = winkler(y, li, ls, nivel=nivel)
    assert abs(w[0] - 2.0) < 1e-12
    assert abs(w[1] - (2.0 + (2.0 / alpha) * 0.5)) < 1e-12
    assert abs(w[2] - (2.0 + (2.0 / alpha) * 1.0)) < 1e-12


def test_winkler_penaliza_el_intervalo_manipulado():
    """
    Es una regla PROPIA: ni ensanchar ni estrechar la mejora.

    Es la razon por la que es la primaria del bloque, y por la que PICP y MPIW
    --que si se pueden manipular, cada una en un sentido-- son diagnosticas.
    """
    tau, X, Xhat, li, ls, *_ = _datos()
    base = resumen_intervalo(X, li, ls, nivel=0.95, tau=tau)["winkler"]
    centro = 0.5 * (li + ls)
    ancho = 0.5 * (ls - li)
    for factor in (0.25, 0.5, 2.0, 4.0):
        w = resumen_intervalo(X, centro - factor * ancho, centro + factor * ancho,
                              nivel=0.95, tau=tau)["winkler"]
        assert w >= base - 1e-9, (
            f"factor={factor} mejoro el Winkler: no seria una regla propia.")


def test_picp_y_mpiw_acompanan():
    tau, X, Xhat, li, ls, *_ = _datos()
    r = resumen_intervalo(X, li, ls, nivel=0.95, tau=tau)
    assert 0.0 <= r["picp"] <= 1.0
    assert r["mpiw"] > 0
    assert abs(r["ace"] - (r["picp"] - 0.95)) < 1e-12
    assert r["primaria"] == "winkler"
    # El error estandar viene marcado como optimista: supone independencia.
    assert picp(X, li, ls)["ee_supone_independencia"] is True


# ==========================================================================
# VENTANA MOVIL: COLUMNAS DE MARCO Y ENGANCHE
# ==========================================================================

_MARCO = ["t_ini", "t_fin", "t_centro", "bloque", "n_ventana", "cruza_T0"]


def test_ventana_conserva_columnas_de_marco():
    """Las columnas que consumen 30_04 y `tabla_ventana` de 30_05 §6."""
    tau, X, Xhat, li, ls, *_ = _datos(n=60)
    t = ventana_movil_funcional(X, Xhat, tau, T0=40, w=10, li=li, ls=ls)
    for c in _MARCO + ["mise", "rmse_f", "mise_rel",
                       "cobertura_puntual", "ancho_medio"]:
        assert c in t.columns, f"falta la columna {c}"
    for c in ["mae_f", "l2_medio", "razon_agregacion", "linf_max",
              "linf_medio", "q95_abs", "razon_linf_l1",
              "winkler", "picp", "mpiw"]:
        assert c in t.columns, f"falta la columna nueva {c}"
    # picp y mpiw son los mismos numeros que cobertura_puntual y ancho_medio.
    assert np.allclose(t["picp"], t["cobertura_puntual"])
    assert np.allclose(t["mpiw"], t["ancho_medio"])


def test_ventana_sin_bloque_A_es_la_tabla_de_antes():
    tau, X, Xhat, li, ls, *_ = _datos(n=60)
    t = ventana_movil_funcional(X, Xhat, tau, T0=40, w=10, li=li, ls=ls,
                                bloque_A=False)
    assert list(t.columns) == _MARCO[:5] + ["cruza_T0", "mise", "rmse_f",
                                            "mise_rel", "cobertura_puntual",
                                            "ancho_medio"]


def test_ventana_rmse_f_es_raiz_del_mse_agregado():
    tau, X, Xhat, *_ = _datos(n=60)
    t = ventana_movil_funcional(X, Xhat, tau, T0=40, w=10)
    assert np.allclose(t["rmse_f"], np.sqrt(t["mise"]))
    assert np.all(t["l2_medio"] <= t["rmse_f"] + 1e-12)


# ==========================================================================
# EL CASO M = 1
# ==========================================================================

def test_M1_scores_como_vector_plano():
    """
    Con M = 1 las entradas pueden llegar como (n,) en vez de (n, 1).

    Es la forma en que el repositorio se ha equivocado antes: `np.loadtxt`
    colapsa a 1D con una sola columna y MATLAB elimina el eje singleton final
    al guardar los `.mat`. La tabla tiene que salir identica en las dos formas.
    """
    tau, X, Xhat, li_f, ls_f, Y, Yhat = _datos(n=50, M=1)
    n = Y.shape[0]
    li_s, ls_s = Yhat - 1.0, Yhat + 1.0

    t_2d = ventana_movil_scores(Y, Yhat, T0=30, w=10, li=li_s, ls=ls_s)
    t_1d = ventana_movil_scores(Y.ravel(), Yhat.ravel(), T0=30, w=10,
                                li=li_s.ravel(), ls=ls_s.ravel())
    assert len(t_2d) == len(t_1d)
    for c in ("rmse", "mae", "cobertura", "winkler", "picp", "mpiw"):
        assert c in t_2d.columns
        assert np.allclose(t_2d[c].to_numpy(), t_1d[c].to_numpy())

    # Y con muestras de la predictiva, que es como lo llama 30_04.
    rng = np.random.default_rng(0)
    Z = Yhat[None, :, :] + rng.normal(0, 0.5, size=(40, n, 1))
    t_crps = ventana_movil_scores(Y, Yhat, T0=30, w=10, muestras=Z)
    assert "crps" in t_crps.columns and np.all(np.isfinite(t_crps["crps"]))


def test_M1_curva_con_una_sola_componente():
    """La parte funcional con M = 1: la curva sigue siendo (n, G)."""
    tau, X, Xhat, li, ls, *_ = _datos(n=50, M=1)
    t = ventana_movil_funcional(X, Xhat, tau, T0=30, w=10, li=li, ls=ls)
    assert np.all(np.isfinite(t[["mae_f", "rmse_f", "linf_max", "winkler"]]))
    r = resumen_error_funcional(X, Xhat, tau)
    assert r["n"] == 50


def test_M1_curva_de_un_solo_origen():
    """Caso degenerado (1, G): ni las normas ni el Bloque B deben romperse."""
    tau, X, Xhat, li, ls, *_ = _datos(n=1, M=1)
    d = normas_error_por_origen(X, Xhat, tau)
    assert d["l1"].shape == (1,)
    r = resumen_intervalo(X, li, ls, nivel=0.95, tau=tau)
    assert np.isfinite(r["winkler"])


# ==========================================================================
# BOOTSTRAP DE BLOQUES
# ==========================================================================

def test_bootstrap_bloques_cubre_y_declara_su_bloque():
    rng = np.random.default_rng(3)
    # Serie con dependencia: AR(1) fuerte, que es el caso que el bootstrap
    # ordinario trataria mal.
    n = 300
    e = rng.normal(size=n)
    v = np.empty(n)
    v[0] = e[0]
    for i in range(1, n):
        v[i] = 0.8 * v[i - 1] + e[i]

    d = bootstrap_bloques(v, largo_bloque=20, B=500, seed=1)
    assert d["li"] < d["observado"] < d["ls"]
    assert d["largo_bloque"] == 20
    # Un bloque mas largo captura mas dependencia y ensancha el intervalo: es
    # la razon por la que el largo se declara siempre junto al intervalo.
    ancho = lambda b: (lambda r: r["ls"] - r["li"])(
        bootstrap_bloques(v, largo_bloque=b, B=500, seed=1))
    assert ancho(30) > ancho(1)


def test_largo_bloque_sugerido_manda_el_solape():
    """Con ventanas solapadas la dependencia mecanica domina a la del proceso."""
    d = largo_bloque_sugerido(n=200, w=40, paso=1)
    assert d["largo_bloque"] == 40
    assert d["manda"].startswith("solape")
    # Sin ventana, manda la memoria de la serie.
    d2 = largo_bloque_sugerido(n=200)
    assert d2["manda"].startswith("memoria")


# ==========================================================================
# EJECUCION DIRECTA
# ==========================================================================

if __name__ == "__main__":
    np.set_printoptions(precision=6, suppress=True)
    pruebas = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for f in pruebas:
        f()
        print(f"OK  {f.__name__}")

    tau, X, Xhat, li, ls, _, _ = _datos()
    print("\n== Bloque A (curva, dominio [0, 1]) ==")
    for k, v in resumen_error_funcional(X, Xhat, tau).items():
        print(f"  {k:22s} = {v}")
    print("\n== Bloque B (banda 95 %) ==")
    for k, v in resumen_intervalo(X, li, ls, nivel=0.95, tau=tau).items():
        print(f"  {k:22s} = {v}")
    print(f"\n{len(pruebas)} pruebas OK")
