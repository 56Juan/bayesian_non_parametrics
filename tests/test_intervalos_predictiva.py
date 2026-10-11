"""
test_intervalos_predictiva.py
=============================
Las tres construcciones de intervalo de docs 03_05_05 (`fit/intervalos_predictiva.py`)
y su enganche con la ventana movil, `evaluacion_barrido` y `comparacion_barrido`.

    python -m pytest tests/test_intervalos_predictiva.py -v

1. Metricas por tramos con K = 1 = las de `metrics_distribucional`, al bit; la ventana
   movil con `region=` de un tramo = la de `li`/`ls`.
2. Bimodal (dos normales a 6 sd, 0.6/0.4): la HDR da dos tramos, mas corta que la
   equi-colas y con cobertura nominal; la simultanea contiene enteras a `nivel` +- 1/S
   de las extracciones.
3. Unimodal gaussiana: HDR = equi-colas salvo el ancho de celda, sin fusiones.
4. Colas tipo atau = 0.5: simultanea con IQR y HDR con recorte del orden de la
   equi-colas; con la sd como escala la simultanea se dispara.
5. Con los valores por defecto la tabla de ventana no gana columnas (las mismas que el
   `57_ventana_funcional_w30.csv` de la 201); con intervalos lleva `predictor` e
   `intervalo` y las filas (esperanza, equicolas) son las de siempre.
6. El .npz con los intervalos y su lectura en `_05`: `cargar_psbp`, `construir_bandas`,
   Bloque B por tramos y `bloqueB_contra_far`.
7. `banda_contra_varianza_condicional` con el valor por defecto da la tabla de siempre.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

import matplotlib  # noqa: E402
matplotlib.use("Agg")

from model_psbp_fd.fit.intervalos_predictiva import (  # noqa: E402
    INTERVALOS, RECORTE_HDR, banda_equicolas, banda_simultanea, hdr_puntual, winkler_tramos,
    ancho_tramos, mpiw_tramos, cobertura_tramos, cobertura_simultanea_tramos, resumen_intervalo_tramos,
)
from model_psbp_fd.fit.metrics_distribucional import (  # noqa: E402
    winkler, mpiw, indicador_cobertura, indicador_cobertura_simultanea, resumen_intervalo,
)
from model_psbp_fd.fit.rolling import ventana_movil_funcional  # noqa: E402

NIVEL = 0.95
TAU = np.linspace(0, 1, 41)


# --------------------------------------------------------------------------
# 1. Un tramo = las metricas de siempre
# --------------------------------------------------------------------------

def _banda(n=50, G=41, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.standard_normal((n, G))
    c = 0.3 * rng.standard_normal((n, G))
    a = np.abs(rng.standard_normal((n, G))) + 0.5
    return y, c - a, c + 1.2 * a


def test_un_tramo_reproduce_al_bit():
    y, li, ls = _banda()
    L, U = li[..., None], ls[..., None]
    assert np.array_equal(winkler_tramos(y, L, U, NIVEL), winkler(y, li, ls, NIVEL))
    assert np.array_equal(winkler_tramos(y, li, ls, NIVEL), winkler(y, li, ls, NIVEL))
    assert mpiw_tramos(L, U) == mpiw(li, ls)
    assert np.array_equal(cobertura_tramos(y, L, U), indicador_cobertura(y, li, ls))
    assert np.array_equal(cobertura_simultanea_tramos(y, L, U), indicador_cobertura_simultanea(y, li, ls))
    r0, r1 = resumen_intervalo(y, li, ls, NIVEL, TAU), resumen_intervalo_tramos(y, L, U, NIVEL, TAU)
    assert all(r0[k] == r1[k] for k in ("winkler", "picp", "ee_picp", "ace", "mpiw"))


def test_ventana_region_un_tramo_igual_a_li_ls():
    y, li, ls = _banda(n=80)
    X = y + 0.2
    a = ventana_movil_funcional(y, X, TAU, 50, w=20, li=li, ls=ls, nivel=NIVEL, bloque_A=True)
    b = ventana_movil_funcional(y, X, TAU, 50, w=20, region=(li[..., None], ls[..., None]),
                                nivel=NIVEL, bloque_A=True)
    pd.testing.assert_frame_equal(a, b, check_exact=True)


def test_tramos_suman_y_penalizan_al_mas_cercano():
    y = np.array([[0.0, 5.0, 2.5]])
    L = np.array([[[-1.0, 4.0], [-1.0, 4.0], [-1.0, 4.0]]])
    U = np.array([[[1.0, 6.0], [1.0, 6.0], [1.0, np.nan]]])
    L[0, 2, 1] = np.nan
    assert np.allclose(ancho_tramos(L, U), [[4.0, 4.0, 2.0]])
    assert cobertura_tramos(y, L, U).tolist() == [[True, True, False]]
    w = winkler_tramos(y, L, U, NIVEL)
    assert np.allclose(w, [[4.0, 4.0, 2.0 + (2 / 0.05) * 1.5]])


# --------------------------------------------------------------------------
# 2-4. Las construcciones sobre muestras conocidas
# --------------------------------------------------------------------------

def _bimodal(S, n, G, rng):
    modo = rng.random((S, n, G)) < 0.4
    return np.where(modo, 6.0, 0.0) + rng.standard_normal((S, n, G))


def test_bimodal_hdr_dos_tramos_y_simultanea():
    rng = np.random.default_rng(10)
    S, n, G = 4000, 2, 4
    Xd = _bimodal(S, n, G, rng)
    L, U, fus = hdr_puntual(Xd, NIVEL, devolver_fusiones=True)
    assert (np.isfinite(L).sum(axis=-1) == 2).all()
    assert fus.sum() == 0
    li, ls = banda_equicolas(Xd, NIVEL)
    assert (ancho_tramos(L, U) < 0.9 * (ls - li)).all()
    nuevas = _bimodal(20000, n, G, rng)
    cob = cobertura_tramos(nuevas, np.broadcast_to(L, (20000,) + L.shape), np.broadcast_to(U, (20000,) + U.shape))
    assert abs(cob.mean() - NIVEL) < 0.02, cob.mean()
    a, b = banda_simultanea(Xd, NIVEL)
    dentro = ((Xd >= a[None]) & (Xd <= b[None])).all(axis=-1).mean(axis=0)
    assert (np.abs(dentro - NIVEL) <= 1.0 / S + 1e-12).all(), dentro


def test_unimodal_hdr_igual_a_equicolas():
    rng = np.random.default_rng(11)
    S, n, G = 4000, 3, 5
    Xd = 2.0 + 1.5 * rng.standard_normal((S, n, G))
    L, U, fus = hdr_puntual(Xd, NIVEL, devolver_fusiones=True)
    assert (np.isfinite(L).sum(axis=-1) == 1).all() and fus.sum() == 0
    li, ls = banda_equicolas(Xd, NIVEL)
    celda = (Xd.max(axis=0) - Xd.min(axis=0)) / 256
    assert (np.abs(L[..., 0] - li) <= 4 * celda).all()
    assert (np.abs(U[..., 0] - ls) <= 4 * celda).all()
    assert np.abs(ancho_tramos(L, U) / (ls - li) - 1).max() < 0.02


def test_colas_atau_robustas():
    rng = np.random.default_rng(12)
    S, n, G = 3000, 2, 20
    Xd = rng.standard_normal((S, n, G))
    cola = rng.random((S, n, G)) < 0.01
    Xd[cola] = rng.standard_cauchy(int(cola.sum()))
    li, ls = banda_equicolas(Xd, NIVEL)
    base = ls - li
    # Sin colas, una banda simultanea sobre 20 puntos ya es ~1.6 veces la puntual.
    a, b = banda_simultanea(Xd, NIVEL)
    assert ((b - a) / base).max() < 2.5
    L, U = hdr_puntual(Xd, NIVEL, recorte=RECORTE_HDR)
    r = ancho_tramos(L, U) / base
    assert 0.8 < r.min() and r.max() < 1.3, (r.min(), r.max())
    a_sd, b_sd = banda_simultanea(Xd, NIVEL, centro=Xd.mean(axis=0), escala=Xd.std(axis=0))
    assert ((b_sd - a_sd) / base).max() > 5.0


def test_fusiona_mas_de_kmax_tramos():
    rng = np.random.default_rng(13)
    S = 6000
    modos = rng.integers(0, 4, size=(S, 1, 1)) * 10.0
    Xd = modos + 0.5 * rng.standard_normal((S, 1, 1))
    L, U, fus = hdr_puntual(Xd, NIVEL, k_max=3, devolver_fusiones=True)
    assert np.isfinite(L).sum() == 3 and fus.sum() == 1


# --------------------------------------------------------------------------
# 5. Tabla de ventana
# --------------------------------------------------------------------------

def _e_falso(d, n=60, G=41, seed=5):
    from model_psbp_fd.fit.intervalos_predictiva import hdr_puntual as hdr
    rng = np.random.default_rng(seed)
    Xd = rng.standard_normal((400, n, G))
    li, ls = banda_equicolas(Xd, NIVEL)
    sim = banda_simultanea(Xd, NIVEL)
    e = {"X_pred": rng.standard_normal((n, G)) * 0.1, "li_f": li, "ls_f": ls,
         "X_proj": rng.standard_normal((n, G)), "paths": {"out_report": Path(d), "predict": Path(d)}}
    e["X_PRED"] = {"esperanza": e["X_pred"], "mediana": e["X_pred"] + 0.05}
    e["INTERVALOS"] = {"equicolas": (li[..., None], ls[..., None]),
                       "simultanea": (sim[0][..., None], sim[1][..., None]),
                       "hdr": hdr(Xd, NIVEL)}
    DIS = {"grilla": TAU, "N_LAGS": 2, "NIVEL": NIVEL, "VENTANAS_W": [10, 20], "W_REF": 10,
           "T0": 42, "MODO_RESIDUO": "ninguno", "OBJETIVO": "curva_suavizada"}
    ORIG = {"X_obj_ev": rng.standard_normal((n, G)), "T0_orig": 40, "t_orig": np.arange(3, 3 + n),
            "es_train": np.arange(3, 3 + n) <= 42}
    return e, DIS, ORIG


def test_tablas_ventana_columnas():
    from model_psbp_fd.fit.evaluacion_barrido import (
        tablas_ventana, filtrar_ventana, _por_predictor, verificar_relaciones_metricas,
        resumen_metricas,
    )
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        e, DIS, ORIG = _e_falso(d)
        tablas_ventana(e, DIS, ORIG)
        base = e["tablas_fun"][10]
        ref = RAIZ / "reports/simulaciones/escenario_201_chungEscG_1_r01_m05/57_ventana_funcional_w30.csv"
        if ref.exists():
            assert list(base.columns) == list(pd.read_csv(ref, nrows=1).columns)
        assert "predictor" not in base.columns and "intervalo" not in base.columns
        tablas_ventana(e, DIS, ORIG, predictores=("esperanza", "mediana"), intervalos=INTERVALOS)
        t = e["tablas_fun"][10]
        pares = set(map(tuple, t[["predictor", "intervalo"]].drop_duplicates().to_numpy()))
        assert pares == {("esperanza", "equicolas"), ("mediana", "equicolas"),
                         ("esperanza", "simultanea"), ("esperanza", "hdr")}
        pd.testing.assert_frame_equal(_por_predictor(t).drop(columns=["predictor", "intervalo"]), base,
                                      check_exact=True)
        assert len(filtrar_ventana(t, predictor="esperanza")) == 3 * len(base)
        verificar_relaciones_metricas({5: e}, [5], [10, 20])
        r = resumen_metricas(e, 5, DIS, ORIG).reset_index()
        assert {"predictor", "intervalo"} <= set(r.columns)
        assert set(r.loc[r.columna == "winkler", "intervalo"]) == set(INTERVALOS)


# --------------------------------------------------------------------------
# 6. .npz y _05
# --------------------------------------------------------------------------

def test_npz_intervalos_y_bloque_B():
    from model_psbp_fd.fit.evaluacion_barrido import guardar_bandas
    from model_psbp_fd.fit.comparacion_barrido import (
        cargar_psbp, construir_bandas, tabla_bloque_B, ventana_bloque_B, bloqueB_contra_far,
        G_FAR, G_PSB, G_PSS, G_PSH,
    )
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        e, DIS, ORIG = _e_falso(d)
        guardar_bandas({5: e}, [5], DIS, ORIG)
        with np.load(Path(d) / "banda_funcional_psbp.npz") as z:
            assert {"li_simultanea", "ls_simultanea", "hdr_L", "hdr_U", "hdr_kmax"} <= set(z.files)
        EST = {5: {"paths": e["paths"], "CURVAS": {"FAR": e["X_pred"] + 0.1}, "far_kn": 3,
                   "n_components": 5, "X_proj": e["X_proj"]}}
        cargar_psbp(EST, DIS, ORIG, intervalos=INTERVALOS)
        f = EST[5]
        assert f["INTERVALOS_PSBP"]["hdr"][0].shape == e["INTERVALOS"]["hdr"][0].shape
        assert np.array_equal(f["INTERVALOS_PSBP"]["equicolas"][0][..., 0], f["BANDA_PSBP"][0])
        construir_bandas(f, DIS, ORIG, psbp_gaussiana=False, intervalos_psbp=INTERVALOS)
        assert list(f["BANDAS"]) == [G_FAR, G_PSB, G_PSS, G_PSH]
        B = tabla_bloque_B(f, DIS, ORIG)
        assert set(B.index.get_level_values("banda")) == {G_FAR, G_PSB, G_PSS, G_PSH}
        vb = ventana_bloque_B(f, DIS, ORIG)
        assert set(vb["banda"]) == {G_FAR, G_PSB, G_PSS, G_PSH}
        assert (Path(d) / "81b_indicador_cobertura_PSBPM-FD-hdr.csv").exists()
        c = bloqueB_contra_far(EST, Path(d))
        assert set(c["banda"]) == {G_PSB, G_PSS, G_PSH} and (Path(d) / "111_bloqueB_intervalos_vs_FAR.csv").exists()
        # Sin intervalos: el .npz y las bandas de siempre.
        e2 = {k: v for k, v in e.items() if k != "INTERVALOS"}
        guardar_bandas({5: e2}, [5], DIS, ORIG)
        with np.load(Path(d) / "banda_funcional_psbp.npz") as z:
            assert "hdr_L" not in z.files
        with pytest.raises(AssertionError):
            cargar_psbp({5: {"paths": e["paths"], "CURVAS": {}}}, DIS, ORIG, intervalos=INTERVALOS)


def test_estilos_bandas_coinciden_con_las_constantes():
    from model_psbp_fd.fit.comparacion_barrido import G_FAR, G_PSG, G_PSB, G_PSS, G_PSH
    from model_psbp_fd.graphics.viz_comparacion import ESTILOS_BANDAS
    assert set(ESTILOS_BANDAS) == {G_FAR, G_PSG, G_PSB, G_PSS, G_PSH}


def test_figuras_con_tramos():
    from model_psbp_fd.graphics.viz_comparacion import plot_bandas_contraste
    from model_psbp_fd.graphics.viz_comportamientos import plot_comportamientos
    rng = np.random.default_rng(6)
    n, G = 6, 41
    Xd = _bimodal(800, n, G, rng)
    L, U = hdr_puntual(Xd, NIVEL)
    li, ls = banda_equicolas(Xd, NIVEL)
    X = rng.standard_normal((n, G))
    bandas = {"FAR (x)": ((li, ls), X), "PSBPM-FD (hdr)": ((L, U), X)}
    plot_bandas_contraste(bandas, {"peores": [0, 1]}, X, TAU, np.arange(n), 3, "mae_f",
                          ("FAR", "PSBPM-FD"), "peores")
    C = {"K": np.ones(n, int), "li_c": li[:, None], "ls_c": ls[:, None], "curvas": X[:, None],
         "masas": np.ones((n, 1)), "nivel": NIVEL}
    plot_comportamientos(C, [0, 1], X, li, ls, TAU, np.arange(n), 3, hdr=(L, U))


# --------------------------------------------------------------------------
# 7. Banda contra h_t
# --------------------------------------------------------------------------

def test_banda_contra_varianza_por_defecto():
    from model_psbp_fd.fit.heterocedasticidad import banda_contra_varianza_condicional
    rng = np.random.default_rng(7)
    T, J, G, n_lags = 80, 4, 41, 2
    Phi = np.linalg.qr(rng.standard_normal((G, J)))[0] / np.sqrt(np.diff(TAU)[0])
    esc = {"interno_var_condicional": np.abs(rng.standard_normal((1, T, 2))) + 0.5,
           "interno_lambda": np.array([2.0, 1.0, 0.5, 0.25]), "interno_Phi": Phi, "grilla": TAU}
    t_orig = np.arange(n_lags + 1, T + 1)
    n = len(t_orig)
    Xd = rng.standard_normal((300, n, G))
    li, ls = banda_equicolas(Xd, NIVEL)
    sim = banda_simultanea(Xd, NIVEL)
    e = {"component_idx": [0, 1, 2], "Psi_grid": Phi[:, :3], "li_f": li, "ls_f": ls,
         "li_s": -np.ones((n, 3)), "ls_s": 1 + rng.random((n, 3)),
         "INTERVALOS": {"equicolas": (li[..., None], ls[..., None]),
                        "simultanea": (sim[0][..., None], sim[1][..., None]), "hdr": hdr_puntual(Xd, NIVEL)}}
    ORIG = {"t_orig": t_orig, "es_train": t_orig <= 50}
    DIS = {"grilla": TAU, "NIVEL": NIVEL}
    base = banda_contra_varianza_condicional({3: e}, [3], DIS, ORIG, esc, cos_min=0.9)
    assert "intervalo" not in base.columns
    todo = banda_contra_varianza_condicional({3: e}, [3], DIS, ORIG, esc, cos_min=0.9, intervalos=INTERVALOS)
    eq = todo[todo.intervalo == "equicolas"].drop(columns="intervalo").reset_index(drop=True)
    pd.testing.assert_frame_equal(eq, base, check_exact=True)
    assert set(todo.loc[todo.objeto == "curva", "intervalo"]) == set(INTERVALOS)
