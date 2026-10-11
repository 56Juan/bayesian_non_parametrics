"""
test_comportamientos.py
=======================
Reglas para contar comportamientos predictivos (`fit/comportamientos.py`, docs
03_06_02) sobre muestras sinteticas con respuesta conocida:

    gaussiana, asimetrica (skewnorm a = 10) y lognormal   -> K = 1
    bimodal 60/40 y 90/10 (separacion 1.8, sd 0.17)        -> K = 2
    tres modos en triangulo (R^2)                          -> K = 3
    tres modos COLINEALES                                  -> K < 3: limitacion de la
        regla tal como la define el docs (eq. valle), que evalua la densidad en los
        centroides; con K = 2 el centroide del grupo que junta dos modos cae en el
        valle entre ellos y la razon no baja de 0.5
    dos modos pegados (separacion 1.0, sd 0.3)             -> K = 1 (valle)
    cola rala de masa 0.6 % a 8 sd                         -> K = 1 (masa)
    colas no acotadas (atau <= 1)                          -> K = 2 solo con recorte

y el catalogo global, la validacion contra lo observado, la del regimen y la
persistencia.

    python -m pytest tests/test_comportamientos.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.stats import skewnorm

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from model_psbp_fd.fit.comportamientos import (  # noqa: E402
    numero_comportamientos, recortar, debe_recortar, comportamientos_origenes,
    catalogo_global, validar_observado, validar_regimen, guardar_comportamientos,
    cargar_comportamientos,
)

ALPHA = 0.05
S = 6000


def _mezcla(rng, pesos, medias, sd, M=3):
    """S extracciones en R^M: el primer score es la mezcla, los demas N(0, sd)."""
    comp = rng.choice(len(pesos), size=S, p=pesos)
    Z = rng.standard_normal((S, M)) * sd
    Z[:, 0] += np.asarray(medias)[comp]
    return Z


def _K(Z, recorte=True):
    return numero_comportamientos(recortar(Z) if recorte else Z, ALPHA)


def test_unimodales_dan_K1():
    rng = np.random.default_rng(0)
    casos = {"gaussiana": rng.standard_normal((S, 3)),
             "asimetrica": np.column_stack([skewnorm.rvs(10, size=S, random_state=1),
                                            rng.standard_normal((S, 2))]),
             "lognormal": np.column_stack([rng.lognormal(0.0, 0.6, S), rng.standard_normal((S, 2)) * 0.3])}
    for nombre, Z in casos.items():
        assert _K(Z)["K"] == 1, nombre


def test_bimodales_dan_K2():
    rng = np.random.default_rng(1)
    for pesos in ((0.6, 0.4), (0.9, 0.1)):
        r = _K(_mezcla(rng, pesos, (0.0, 1.8), 0.17))
        assert r["K"] == 2, (pesos, r["K"], r["valle"])
        assert np.allclose(r["masas"], pesos, atol=0.02), r["masas"]
        assert abs(r["centroides"][0, 0] - 0.0) < 0.05 and abs(r["centroides"][1, 0] - 1.8) < 0.05


def test_tres_modos():
    rng = np.random.default_rng(2)
    comp = rng.choice(3, size=S, p=(0.4, 0.3, 0.3))
    medias = np.array([[0.0, 0.0], [1.8, 0.0], [0.9, 1.56]])
    r = _K(medias[comp] + 0.17 * rng.standard_normal((S, 2)))
    assert r["K"] == 3, (r["K"], r["r2"])
    assert np.allclose(np.sort(r["masas"]), (0.3, 0.3, 0.4), atol=0.02)
    # Colineales: la regla del docs no los ve (documentado arriba). Con pesos
    # iguales la razon de valle es ~1; con 0.5/0.3/0.2 queda en 0.5-0.66 y en
    # algun sorteo cae justo bajo 0.5 (K = 2). Nunca 3.
    for pesos in ((1 / 3, 1 / 3, 1 / 3), (0.5, 0.3, 0.2)):
        assert _K(_mezcla(rng, pesos, (0.0, 1.8, 3.6), 0.17))["K"] < 3


def test_modos_pegados_dan_K1():
    rng = np.random.default_rng(3)
    r = _K(_mezcla(rng, (0.5, 0.5), (0.0, 1.0), 0.3))
    assert r["K"] == 1 and r["valle"] >= 0.5, r


def test_cola_rala_K1_por_masa():
    rng = np.random.default_rng(4)
    Z = _mezcla(rng, (0.994, 0.006), (0.0, 8.0), 1.0)
    for recorte in (False, True):
        r = _K(Z, recorte)
        assert r["K"] == 1, (recorte, r)


def test_recorte_con_colas_no_acotadas():
    rng = np.random.default_rng(5)
    Z = _mezcla(rng, (0.6, 0.4), (0.0, 1.8), 0.17)
    malos = rng.choice(S, size=int(0.003 * S), replace=False)
    Z[malos] = rng.standard_cauchy((len(malos), Z.shape[1])) * 1e4          # atomos con precision ~0
    assert _K(Z, recorte=False)["K"] == 1          # k-medias gasta el grupo en los extremos
    r = _K(Z, recorte=True)
    assert r["K"] == 2 and np.allclose(r["masas"], (0.6, 0.4), atol=0.02), r
    Zr = recortar(Z)
    assert Zr.shape == Z.shape and np.isfinite(Zr).all() and np.abs(Zr).max() < 5
    hp = {"hyperparams_list": [{"hyperparams": {"atau": 0.5}}, {"hyperparams": {"atau": 2.0}}]}
    assert debe_recortar(hp)
    hp["hyperparams_list"][0]["hyperparams"]["atau"] = 3.0
    assert not debe_recortar(hp)


# --------------------------------------------------------------------------
# Por origen, catalogo, validacion y persistencia
# --------------------------------------------------------------------------

G = 21
PSI = np.column_stack([np.sqrt(2) * np.sin(2 * np.pi * (k + 1) * np.linspace(0, 1, G)) for k in range(2)])


def _a_curva(Y):
    return np.atleast_2d(Y) @ PSI.T


def _panel(rng, n=40):
    """n origenes: los pares bimodales (60/40 entre (0, 0) y (1.8, 0)), los
    impares unimodales en (0.9, 1.5)."""
    SC = 0.17 * rng.standard_normal((2000, n, 2))
    for i in range(n):
        if i % 2 == 0:
            SC[:, i, 0] += np.where(rng.random(2000) < 0.4, 1.8, 0.0)
        else:
            SC[:, i] += (0.9, 1.5)
    return SC


def test_por_origen_catalogo_y_persistencia(tmp_path):
    rng = np.random.default_rng(6)
    SC = _panel(rng)
    C = comportamientos_origenes(SC, _a_curva, 0.95, recortar_colas=True, verbose=False)
    assert (C["K"][::2] == 2).all() and (C["K"][1::2] == 1).all()
    assert np.allclose(C["masas"][::2, :2], [0.6, 0.4], atol=0.05)
    i = 0
    X0 = _a_curva(SC[:, i])
    lab = np.argmin(((SC[:, i, None, :] - C["centroides"][i, None, :2]) ** 2).sum(-1), axis=1)
    assert np.allclose(C["li_c"][i, 0], np.quantile(X0[lab == 0], 0.025, axis=0), atol=0.05)
    assert np.allclose(C["X_pred_cluster"][i], C["curvas"][i, 0])

    cat = catalogo_global(C, _a_curva)
    assert cat["K"] == 3, cat["masas"]               # (0,0) y (1.8,0) de los pares, (0.9,1.5)
    assert np.allclose(np.sort(cat["masas"]), (0.2, 0.3, 0.5), atol=0.03)
    assert np.allclose(np.nansum(cat["prob"], axis=1), 1.0)

    guardar_comportamientos({"predict": tmp_path}, C, np.arange(SC.shape[1]))
    C2 = cargar_comportamientos({"predict": tmp_path})
    assert np.array_equal(C2["K"], C["K"]) and np.allclose(C2["masas"], C["masas"], equal_nan=True)
    assert cargar_comportamientos({"predict": tmp_path / "no"}) is None


def test_validar_observado_y_regimen():
    rng = np.random.default_rng(7)
    SC = _panel(rng)
    n = SC.shape[1]
    C = comportamientos_origenes(SC, _a_curva, 0.95, recortar_colas=True, verbose=False)
    # Observado = una extraccion del grupo mayoritario en los pares.
    Y = np.where(np.arange(n)[:, None] % 2 == 0, [0.02, 0.0], [0.9, 1.5])
    X = _a_curva(Y)
    li, ls = np.quantile(_a_curva(SC.reshape(-1, 2)).reshape(2000, n, G), [0.025, 0.975], axis=0)
    v = validar_observado(C, Y, X, li, ls, np.linspace(0, 1, G))
    pares = v[v.i % 2 == 0]
    assert (pares["acierto"] == 1).all() and np.allclose(pares["p_obs"], 0.6, atol=0.05)
    assert (pares["mpiw_real"] < pares["mpiw_marg"]).all()
    assert (v[v.K == 1]["mpiw_real"] - v[v.K == 1]["mpiw_marg"]).abs().max() < 0.05

    # Oraculo = la misma ley, con el regimen de cada extraccion conocido.
    idx = np.arange(n)
    R_or = (SC[:, :, :1] > 0.9) & (idx[None, :, None] % 2 == 0)
    R_real = np.zeros((n, 1), bool)
    df = validar_regimen(C, idx, SC, R_or, R_real)
    assert (df["K"] == df["K_oraculo"]).all()
    assert df["tv"].max() < 0.02 and df["pureza"].min() > 0.99
    assert (df["acierto_modelo"] == 1).all() and (df["acierto_oraculo"] == 1).all()


if __name__ == "__main__":
    import inspect
    import tempfile
    for nombre, f in list(globals().items()):
        if nombre.startswith("test_") and callable(f):
            if "tmp_path" in inspect.signature(f).parameters:
                with tempfile.TemporaryDirectory() as d:
                    f(Path(d))
            else:
                f()
            print("OK", nombre)
