"""
test_resumenes_predictiva.py
============================
Predictores puntuales de la curva (`fit/resumenes_predictiva.py`, docs 03_05_04),
su enganche con `evaluacion_barrido` / `comparacion_barrido` y la compatibilidad
con los notebooks de la 200.

    python -m pytest tests/test_resumenes_predictiva.py -v

1. Mezcla bimodal 60/40 (separacion 1.8, sd 0.17) a traves del `PSBPPredictor`
   real: mediana, medoide, MBD y modal caen en el modo mayoritario; la esperanza,
   en el valle.
2. Unimodal gaussiana: los cinco coinciden salvo error Monte Carlo.
3. MBD por rangos, suma de desvios y medoide contra su definicion directa.
4. Persistencia: el npz con los predictores, un npz antiguo sin ellos, y
   `cargar_draws_scores` -> None si no hay draws.
5. La ventana movil con el predictor por defecto es la de siempre (sin columna
   `predictor`) y las llamadas de 200_04 / 200_05 siguen ligando con las firmas.
"""

from __future__ import annotations

import ast
import inspect
import json
import sys
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from model_psbp_fd.fit.resumenes_predictiva import (  # noqa: E402
    mediana_puntual, medoide_l1, mediana_mbd, profundidad_mbd, suma_desvios_absolutos,
    atomo_modal, distancia_a_muestra_mas_cercana,
)
from model_psbp_fd.models.pspb_fd_v3 import ModeloTraza  # noqa: E402
from model_psbp_fd.utils.quadrature import pesos_trapezoidales  # noqa: E402

TAU = np.linspace(0, 1, 41)
PHI = np.sqrt(2) * np.sin(2 * np.pi * TAU)          # norma L^2 uno


def _trazas_mezcla(p1, mu1, mu2, sd, T=400, ruido_alpha=0.0, seed=0):
    """Trazas de un PSBPM con N = 2 atomos y gating constante (psi = 0):
    pi_1 = Phi(alpha) = p1, medias mu1 / mu2, sd comun."""
    rng = np.random.default_rng(seed)
    alpha = norm.ppf(p1) + ruido_alpha * rng.standard_normal(T)
    return {
        "betajhout": np.zeros((T, 2, 1)), "beta0hout": np.tile([mu1, mu2], (T, 1)),
        "tauhout": np.full((T, 2), 1.0 / sd ** 2), "alphahout": alpha[:, None],
        "psijhout": np.zeros((T, 1, 1)), "Gammajhout": np.zeros((T, 1, 1)),
    }


def _predictores_desde_trazas(trazas, n=3, S_por_iter=15, seed=1):
    mt = ModeloTraza(trazas, burn=0, feature_names=["fpc_1_lag1"])
    df = pd.DataFrame({"fpc_1": np.zeros(n), "fpc_1_lag1": np.zeros(n)})
    esperanza = mt.momentos(df)["media"]                                  # (n,)
    Z = mt.muestrear(df, S_por_iter, seed=seed)                           # (S, n)
    X = Z[:, :, None] * PHI[None, None, :]                                # (S, n, G)
    modal = atomo_modal({0: {0: mt}}, {0: df})[:, 0]                      # (n,)
    return {"esperanza": esperanza[:, None] * PHI, "mediana": mediana_puntual(X),
            "medoide": medoide_l1(X, TAU), "mbd": mediana_mbd(X),
            "modal": modal[:, None] * PHI}, X


def _nivel(curva):
    """Coeficiente de una curva c * PHI (proyeccion L^2)."""
    w = pesos_trapezoidales(TAU)
    return float((curva * PHI) @ w / ((PHI ** 2) @ w))


def test_bimodal_60_40():
    P, _ = _predictores_desde_trazas(_trazas_mezcla(0.6, 0.0, 1.8, 0.17))
    niveles = {k: _nivel(v[0]) for k, v in P.items()}
    assert abs(niveles["esperanza"] - 0.72) < 0.02, niveles               # en el valle
    for k in ("mediana", "medoide", "mbd", "modal"):
        assert abs(niveles[k]) < 0.3, (k, niveles)                        # modo mayoritario
        assert abs(niveles[k] - 1.8) > 1.2, (k, niveles)


def test_modal_usa_la_mediana_sobre_iteraciones():
    # Gating que en ~30 % de las iteraciones pone mas peso en el atomo 2: el
    # promedio de mu_{h*} se iria hacia el valle, la mediana no.
    tr = _trazas_mezcla(0.6, 0.0, 1.8, 0.17, ruido_alpha=0.5, seed=3)
    mt = ModeloTraza(tr, burn=0, feature_names=["fpc_1_lag1"])
    df = pd.DataFrame({"fpc_1": [0.0], "fpc_1_lag1": [0.0]})
    v = mt.atomo_modal(df)[:, 0]
    assert 0.1 < (v > 0.9).mean() < 0.5
    assert abs(atomo_modal({0: {0: mt}}, {0: df})[0, 0]) < 1e-12
    assert v.mean() > 0.2


def test_unimodal_coinciden():
    P, X = _predictores_desde_trazas(_trazas_mezcla(0.999999, 0.5, 0.5, 0.3), S_por_iter=25)
    ref = _nivel(P["esperanza"][0])
    for k, v in P.items():
        assert abs(_nivel(v[0]) - ref) < 0.05, (k, _nivel(v[0]), ref)


def test_mbd_por_rangos_contra_definicion():
    rng = np.random.default_rng(0)
    S, n, G = 7, 2, 5
    X = rng.standard_normal((S, n, G))
    D = profundidad_mbd(X)
    for i in range(n):
        for s in range(S):
            tot = 0.0
            for a in range(S):
                for b in range(a + 1, S):
                    lo, hi = np.minimum(X[a, i], X[b, i]), np.maximum(X[a, i], X[b, i])
                    tot += np.mean((X[s, i] >= lo) & (X[s, i] <= hi))
            assert abs(D[s, i] - tot / comb(S, 2)) < 1e-12


def test_suma_desvios_y_medoide_contra_fuerza_bruta():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((50, 3, 41))
    sad = suma_desvios_absolutos(X)
    directo = np.abs(X[:, None] - X[None, :]).sum(axis=1)
    assert np.allclose(sad, directo)
    wt = pesos_trapezoidales(TAU)
    med, idx = medoide_l1(X, TAU, devolver_indice=True)
    for i in range(3):
        d = (np.abs(X[:, None, i] - X[None, :, i]) @ wt).sum(axis=1)
        assert idx[i] == int(np.argmin(d))
        assert np.array_equal(med[i], X[idx[i], i])


def test_distancia_a_muestra():
    rng = np.random.default_rng(2)
    X = rng.standard_normal((200, 4, 41))
    d = distancia_a_muestra_mas_cercana(X, X[17], TAU, n_ref=50)
    assert np.allclose(d["d_curva"], 0.0)
    assert (d["d_tipica"] > 0).all()
    lejos = distancia_a_muestra_mas_cercana(X, X[17] + 10.0, TAU, n_ref=50)
    assert (lejos["razon"] > 5).all()


# --------------------------------------------------------------------------
# Persistencia y enganche con comparacion_barrido
# --------------------------------------------------------------------------

def _est_falso(tmp: Path, n=12, G=6):
    from model_psbp_fd.fit.evaluacion_barrido import guardar_bandas
    rng = np.random.default_rng(4)
    paths = {"predict": tmp}
    X = {p: rng.standard_normal((n, G)) for p in ("esperanza", "mediana", "modal")}
    e = {"paths": paths, "li_f": -np.ones((n, G)), "ls_f": np.ones((n, G)),
         "X_pred": X["esperanza"], "X_PRED": X, "SC_draws": rng.standard_normal((30, n, 2))}
    DIS = {"NIVEL": 0.95, "N_LAGS": 2, "T0": 8, "MODO_RESIDUO": "ninguno", "OBJETIVO": "curva_suavizada"}
    ORIG = {"t_orig": np.arange(3, 3 + n), "X_obj_ev": np.zeros((n, G))}
    guardar_bandas({1: e}, [1], DIS, ORIG, guardar_draws=True)
    return e, DIS, ORIG


def test_npz_con_predictores_y_draws(tmp_path):
    from model_psbp_fd.fit.comparacion_barrido import cargar_psbp, nombre_psbp
    from model_psbp_fd.pipelines import cargar_draws_scores
    e, DIS, ORIG = _est_falso(tmp_path)
    z = np.load(tmp_path / "banda_funcional_psbp.npz")
    assert {"X_pred", "X_pred_mediana", "X_pred_modal"} <= set(z.files)
    assert np.allclose(cargar_draws_scores({"predict": tmp_path}), e["SC_draws"].astype(np.float32))

    EST = {1: {"paths": e["paths"], "CURVAS": {}}}
    cargar_psbp(EST, DIS, ORIG, predictores=("esperanza", "mediana", "modal"))
    assert set(EST[1]["CURVAS"]) == {nombre_psbp(p) for p in ("esperanza", "mediana", "modal")}
    assert np.allclose(EST[1]["X_PSBP"], e["X_pred"], atol=1e-6)
    assert np.allclose(EST[1]["CURVAS"][nombre_psbp("mediana")], e["X_PRED"]["mediana"], atol=1e-6)


def test_npz_antiguo_sigue_leyendo(tmp_path):
    from model_psbp_fd.fit.comparacion_barrido import cargar_psbp
    from model_psbp_fd.pipelines import cargar_draws_scores
    n, G = 12, 6
    np.savez_compressed(tmp_path / "banda_funcional_psbp.npz", li=np.zeros((n, G)), ls=np.ones((n, G)),
                        X_pred=np.full((n, G), 0.5), t_orig=np.arange(n), nivel=np.array([0.95]),
                        n_lags=np.array([2]), T0=np.array([8]), modo_residuo=np.array(["ninguno"]),
                        objetivo=np.array(["curva_suavizada"]))
    DIS = {"NIVEL": 0.95, "N_LAGS": 2, "T0": 8, "OBJETIVO": "curva_suavizada"}
    ORIG = {"X_obj_ev": np.zeros((n, G))}
    EST = {1: {"paths": {"predict": tmp_path}, "CURVAS": {}}}
    cargar_psbp(EST, DIS, ORIG)
    assert list(EST[1]["CURVAS"]) == ["PSBPM-FD"]
    assert cargar_draws_scores({"predict": tmp_path}) is None
    EST = {1: {"paths": {"predict": tmp_path}, "CURVAS": {}}}
    try:
        cargar_psbp(EST, DIS, ORIG, predictores=("esperanza", "mediana"))
        raise RuntimeError("debio fallar: el npz antiguo no trae X_pred_mediana")
    except AssertionError:
        pass


def test_tablas_ventana_por_defecto_sin_columna_predictor():
    from model_psbp_fd.fit.evaluacion_barrido import tablas_ventana, _por_predictor
    from model_psbp_fd.fit.rolling import ventana_movil_funcional
    import tempfile
    rng = np.random.default_rng(5)
    n, G = 60, 41
    with tempfile.TemporaryDirectory() as d:
        e = {"X_pred": rng.standard_normal((n, G)), "li_f": -2 * np.ones((n, G)), "ls_f": 2 * np.ones((n, G)),
             "X_proj": rng.standard_normal((n, G)), "paths": {"out_report": Path(d)}}
        e["X_PRED"] = {"esperanza": e["X_pred"], "mediana": e["X_pred"] + 0.1}
        DIS = {"grilla": TAU, "N_LAGS": 2, "NIVEL": 0.95, "VENTANAS_W": [10, 20], "W_REF": 10}
        ORIG = {"X_obj_ev": rng.standard_normal((n, G)), "T0_orig": 40}
        tablas_ventana(e, DIS, ORIG)
        directo = ventana_movil_funcional(ORIG["X_obj_ev"], e["X_pred"], TAU, 40, w=10, li=e["li_f"],
                                          ls=e["ls_f"], t_offset=2, nivel=0.95, bloque_A=True)
        assert "predictor" not in e["tablas_fun"][10].columns
        pd.testing.assert_frame_equal(e["tablas_fun"][10], directo)
        tablas_ventana(e, DIS, ORIG, predictores=("esperanza", "mediana"))
        t = e["tablas_fun"][10]
        assert set(t["predictor"]) == {"esperanza", "mediana"}
        pd.testing.assert_frame_equal(_por_predictor(t, "esperanza").drop(columns="predictor"), directo)


def _llamadas_notebook(nb: Path):
    """(nombre, n_posicionales, claves) de cada llamada a una funcion importada
    de model_psbp_fd en las celdas de codigo."""
    celdas = [ "".join(c["source"]) for c in json.loads(nb.read_text(encoding="utf-8"))["cells"]
               if c["cell_type"] == "code"]
    codigo = "\n".join("\n".join(l for l in c.splitlines() if not l.lstrip().startswith(("%", "!")))
                       for c in celdas)
    arbol = ast.parse(codigo)
    importados = {}
    for nodo in ast.walk(arbol):
        if isinstance(nodo, ast.ImportFrom) and nodo.module and nodo.module.startswith("model_psbp_fd"):
            for a in nodo.names:
                importados[a.asname or a.name] = (nodo.module, a.name)
    llamadas = [(n.func.id, len(n.args), [k.arg for k in n.keywords if k.arg])
                for n in ast.walk(arbol) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                and n.func.id in importados]
    return importados, llamadas


def test_notebooks_200_siguen_ligando():
    import importlib
    for nombre in ("200_04_evaluacion.ipynb", "200_05_comparacion.ipynb"):
        importados, llamadas = _llamadas_notebook(RAIZ / "notebooks/simulaciones/200_sim_TAR" / nombre)
        objetos = {}
        for alias, (mod, attr) in importados.items():
            objetos[alias] = getattr(importlib.import_module(mod), attr)       # importa
        for fn, n_pos, claves in llamadas:
            obj = objetos[fn]
            if callable(obj) and not isinstance(obj, type):
                inspect.signature(obj).bind(*range(n_pos), **{k: None for k in claves})
        assert llamadas, nombre


if __name__ == "__main__":
    import tempfile
    for nombre, f in list(globals().items()):
        if nombre.startswith("test_") and callable(f):
            if "tmp_path" in inspect.signature(f).parameters:
                with tempfile.TemporaryDirectory() as d:
                    f(Path(d))
            else:
                f()
            print("OK", nombre)
