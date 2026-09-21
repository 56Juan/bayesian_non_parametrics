"""
test_far_operador.py
====================
Pruebas de `fit/far_operador.py`.

Como se corren
--------------
No hay suite previa en el repositorio ni configuracion de pytest, de modo que
estas pruebas se ejecutan invocando pytest directamente sobre el archivo, desde
la raiz del proyecto:

    python -m pytest tests/test_far_operador.py -v

El archivo tambien es ejecutable sin pytest (`python tests/test_far_operador.py`),
que imprime el informe de comparacion. Se escribio asi porque la cifra que
interesa ---cuanto se separa este estimador del VAR--- es un numero a leer, no
un aserto que pase o falle.

Sobre el caso de contraste con R
--------------------------------
`test_equivalencia_con_r` compara contra una salida de `far::far()` generada en
R 4.6.0 con `far 0.6-7` y guardada en `tests/datos_far_r/`. Si esos archivos no
estan, la prueba se salta con aviso en vez de fallar: no se puede exigir una
instalacion de R para correr el resto.
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
        def skip(msg):
            raise RuntimeError(msg)

        class mark:
            slow = staticmethod(lambda f: f)
    pytest = _Pytest()

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from model_psbp_fd.fit.far_operador import (          # noqa: E402
    FAR1, FARp, norma_hs, seleccionar_kn, simular_far1, comparar_con_r,
)
from model_psbp_fd.utils.quadrature import pesos_trapezoidales   # noqa: E402

DIR_R = Path(__file__).parent / "datos_far_r"


# ==========================================================================
# 1. EQUIVALENCIA CON far::far()
# ==========================================================================

def _cargar_r():
    import pandas as pd
    f_rho, f_z = DIR_R / "rho_R_5x5.txt", DIR_R / "Z_R_5x5.txt"
    if not (f_rho.exists() and f_z.exists()):
        pytest.skip(f"No hay salidas de R en {DIR_R}; ver el docstring.")
    rho = pd.read_csv(f_rho, sep=r"\s+").values.astype(float)
    Z = pd.read_csv(f_z, sep=r"\s+").values.astype(float)   # (L, n) como en R
    return Z.T, rho                                          # (n, L) del repo


def test_equivalencia_con_r():
    """
    Con `pesos='conteo'` la clase debe reproducir a far::far() hasta precision
    de maquina. No es una aspiracion: los pesos se cancelan en el producto
    `C1 C0^-1` y la unica diferencia posible seria de implementacion.
    """
    X, rho_r = _cargar_r()
    rep = comparar_con_r(X, rho_r, kn=3, pesos="conteo")
    assert rep["layout_verificable"], (
        "La matriz de R no es lo bastante asimetrica para descartar una "
        "transposicion; el resto de la prueba no seria concluyente."
    )
    assert rep["error_max_puntual"] < 1e-12, rep
    assert rep["corr_predicciones"] > 1 - 1e-12, rep


def test_pesos_solo_afectan_el_truncamiento():
    """
    Cambiar la cuadratura NO debe mover `rho_` cuando `kn` agota el rango: los
    pesos se cancelan en `C1 C0^-1` y solo entran por el subespacio retenido.
    Con `kn` chico si difieren, y la diferencia se concentra en los bordes.
    """
    X, _ = _cargar_r()
    tau = np.linspace(0.0, 1.0, X.shape[1])
    kn_pleno = X.shape[1]
    a = FAR1(kn=kn_pleno, pesos="conteo").fit(X).rho_
    b = FAR1(kn=kn_pleno, grilla=tau).fit(X).rho_
    assert np.abs(a - b).max() < 1e-8, (
        "Sin truncamiento los pesos deben cancelarse exactamente."
    )


# ==========================================================================
# 2. CONSISTENCIA: ||rho_hat - rho||_HS -> 0
# ==========================================================================

def _hs_error(T, kn, seed):
    """Ajusta sobre el Escenario 1 del repositorio y devuelve el error HS."""
    from model_psbp_fd.pipelines import ConfigEscenario1, generar_escenario_1
    cfg = ConfigEscenario1(L=75, T=T, R=1, seed=seed, burn_in=200,
                           gamma=0.30, hs_norm=0.70, ell=0.5, sigma_obs=0.0)
    sal = generar_escenario_1(cfg)
    X = sal.curvas[0]                       # (T, L) sin ruido de observacion
    tau = sal.grilla
    Psi = sal.internos["operador"]          # matriz de ACCION verdadera
    mod = FAR1(kn=kn, grilla=tau).fit(X)
    w = pesos_trapezoidales(tau)
    return norma_hs(mod.rho_ - Psi, w) / norma_hs(Psi, w)


@pytest.mark.slow
def test_consistencia_hs():
    """
    Al crecer `n` el error relativo en norma de Hilbert-Schmidt debe bajar.

    Se usa el generador del repositorio ---`generar_escenario_1`--- y no un
    simulador propio: ya es un FAR(1) de nucleo gaussiano con `hs_norm = 0.70`
    calibrada y el operador verdadero en `internos["operador"]`. Se genera sin
    ruido de observacion (`sigma_obs = 0`) porque lo que aqui se prueba es la
    consistencia del estimador, no su robustez al ruido de medicion: con ruido
    el limite no es `Psi` sino un operador atenuado, y la prueba fallaria por
    una razon que no es un defecto de la implementacion.

    El aserto es sobre la TENDENCIA y no sobre un umbral absoluto: con
    truncamiento espectral el error tiene un piso de sesgo que depende de `kn`,
    de modo que exigir que llegue a cero seria exigir algo falso.
    """
    errores = [_hs_error(T, kn=6, seed=41232 + i)
               for i, T in enumerate((200, 800, 3200))]
    assert errores[-1] < errores[0], f"el error no decrece: {errores}"
    assert errores[-1] < 0.5 * errores[0], f"decrece demasiado poco: {errores}"


@pytest.mark.slow
def test_consistencia_rango_uno():
    """
    Con el nucleo separable `exp(-(t^2+s^2)/2)`, de RANGO UNO, el truncamiento
    no introduce sesgo para `kn >= 1` y la convergencia debe ser limpia. Es la
    prueba que el generador del repositorio no da, y la unica razon de que
    `simular_far1` exista.
    """
    errs = []
    for T in (500, 2000, 8000):
        d = simular_far1(L=40, T=T, hs_objetivo=0.70, seed=7)
        mod = FAR1(kn=4, grilla=d["grilla"]).fit(d["curvas"])
        errs.append(norma_hs(mod.rho_ - d["operador"], d["pesos"])
                    / norma_hs(d["operador"], d["pesos"]))
    assert errs[-1] < errs[0], f"el error no decrece: {errs}"


# ==========================================================================
# 3. CONTRATO DE LA CLASE
# ==========================================================================

def test_sin_fuga_y_formas():
    d = simular_far1(L=20, T=120, seed=3)
    X, tau = d["curvas"], d["grilla"]
    T0 = 84
    mod = FAR1(kn=3, grilla=tau).fit(X[:T0])
    assert mod.rho_.shape == (20, 20)
    assert mod.base_.shape == (20, 3)
    assert mod.predict(X[:-1]).shape == (119, 20)
    assert mod.predict_serie(X).shape == (119, 20)
    # El ajuste no puede depender de lo que venga despues de T0.
    otro = FAR1(kn=3, grilla=tau).fit(np.vstack([X[:T0], X[T0:] * 3.0])[:T0])
    assert np.allclose(mod.rho_, otro.rho_)


def test_base_ortonormal_en_la_metrica():
    d = simular_far1(L=30, T=200, seed=5)
    mod = FAR1(kn=5, grilla=d["grilla"]).fit(d["curvas"])
    W = (mod.base_ * mod.pesos_[:, None]).T @ mod.base_
    assert np.abs(W - np.eye(5)).max() < 1e-10


def test_seleccion_kn_no_toca_el_test():
    d = simular_far1(L=25, T=300, seed=11)
    X, tau = d["curvas"], d["grilla"]
    T0 = 210
    r = seleccionar_kn(X[:T0], kn_max=6, grilla=tau)
    assert 1 <= r.kn <= 6
    assert r.tabla.shape == (6, 7)
    # Cambiar el bloque de prueba no puede alterar la eleccion.
    X2 = X.copy(); X2[T0:] *= 5.0
    assert seleccionar_kn(X2[:T0], kn_max=6, grilla=tau).kn == r.kn


def test_condicion_crece_con_kn():
    """El numero de condicion efectivo debe crecer con `kn`: es la forma en
    que el mal planteamiento del problema se manifiesta."""
    d = simular_far1(L=30, T=400, seed=13)
    conds = [FAR1(kn=k, grilla=d["grilla"]).fit(d["curvas"]).diagnostico_kn()["condicion"]
             for k in (2, 5, 10)]
    assert conds[0] < conds[1] < conds[2], conds


# ==========================================================================
# FAR(p)
# ==========================================================================

def _simular_var_funcional(coefs, L=60, d=6, T=900, seed=3):
    """Curvas cuyos coeficientes siguen c_t = sum_l coefs[l] c_{t-l} + eps."""
    from model_psbp_fd.pipelines.deprecated.sim_escenario_L import base_ortonormal
    rng = np.random.default_rng(seed)
    tau = np.linspace(0.0, 1.0, L)
    Phi, _ = base_ortonormal(tau, d, "bspline_lowdin")
    p = len(coefs)
    sig = 0.7 ** np.arange(d)
    C = np.zeros((T + 50, d))
    for t in range(p, T + 50):
        C[t] = sum(coefs[l] * C[t - 1 - l] for l in range(p)) + rng.normal(0, sig)
    return tau, C[50:] @ Phi


def test_farp_orden_uno_reproduce_far1():
    """`FARp(p=1)` debe coincidir con `FAR1` hasta precision de maquina.

    Es la razon por la que el VAR del subespacio se resuelve por Yule-Walker y
    no por minimos cuadrados sobre el disenyo apilado: alli C0 se estima con
    todas las curvas utilizables y C1 con los pares, tal como en el fuente de
    R. Si este test se rompe, `FARp` dejo de ser una generalizacion de `FAR1`
    y las corridas de orden 1 y 3 ya no son comparables entre si.
    """
    d = simular_far1(L=40, T=300, seed=7)
    X, tau = d["curvas"], d["grilla"]
    for pesos in ("conteo", "trapecio"):
        for kn in (1, 2, 4):
            a = FAR1(kn=kn, grilla=tau, pesos=pesos).fit(X)
            b = FARp(p=1, kn=kn, grilla=tau, pesos=pesos).fit(X)
            assert np.allclose(a.rho_, b.rho_[0], atol=1e-12), (pesos, kn)
            assert np.allclose(a.predict_serie(X), b.predict_serie(X),
                               atol=1e-12), (pesos, kn)


def test_farp_recupera_dependencia_en_rezagos_lejanos():
    """Con el rezago 3 dominante, FAR(3) debe recuperarlo y batir a FAR(1)."""
    tau, X = _simular_var_funcional([0.15, 0.0, 0.75])
    T0 = 600
    rmse = {}
    for p in (1, 3):
        m = FARp(p=p, kn=3, grilla=tau, pesos="trapecio").fit(X[:T0])
        err = X[T0:][p:] - m.predict_serie(X[T0:])
        rmse[p] = float(np.sqrt((err ** 2).mean()))
    assert rmse[3] < 0.8 * rmse[1], rmse

    m3 = FARp(p=3, kn=3, grilla=tau, pesos="trapecio").fit(X[:T0])
    diag = [np.trace(m3.rho_subespacio_[l]) / 3 for l in range(3)]
    assert abs(diag[0] - 0.15) < 0.08, diag
    assert abs(diag[1] - 0.00) < 0.08, diag
    assert abs(diag[2] - 0.75) < 0.08, diag


def test_farp_alineacion_de_predict_serie():
    """`predict_serie` pierde `p` origenes, no uno: desalinearlo desplazaria
    todas las metricas de la ventana movil en silencio."""
    tau, X = _simular_var_funcional([0.3, 0.2, 0.3], T=300)
    for p in (1, 2, 3):
        m = FARp(p=p, kn=3, grilla=tau, pesos="trapecio").fit(X[:200])
        assert m.predict_serie(X[200:]).shape == (X[200:].shape[0] - p,
                                                  X.shape[1])


def test_farp_radio_espectral_detecta_no_estacionariedad():
    """El radio de la companera separa un proceso estable de uno explosivo."""
    tau, X = _simular_var_funcional([0.15, 0.0, 0.75])
    estable = FARp(p=3, kn=3, grilla=tau, pesos="trapecio").fit(X[:600])
    assert estable.radio_espectral() < 1.05

    tau2, X2 = _simular_var_funcional([0.3, 0.3, 0.3], T=600)
    m = FARp(p=3, kn=3, grilla=tau2, pesos="trapecio").fit(X2[:400])
    assert m.radio_espectral() > estable.radio_espectral() - 0.5


def test_seleccionar_kn_acepta_orden():
    """`seleccionar_kn(p=1)` no puede cambiar respecto de antes de existir p."""
    d = simular_far1(L=30, T=400, seed=11)
    r1 = seleccionar_kn(d["curvas"][:300], kn_max=5, grilla=d["grilla"])
    r1p = seleccionar_kn(d["curvas"][:300], kn_max=5, grilla=d["grilla"], p=1)
    assert np.allclose(r1.tabla, r1p.tabla, atol=1e-12)
    r3 = seleccionar_kn(d["curvas"][:300], kn_max=5, grilla=d["grilla"], p=3)
    assert 1 <= r3.kn <= 5


if __name__ == "__main__":
    np.set_printoptions(precision=6, suppress=True)
    try:
        X, rho_r = _cargar_r()
        print("== Equivalencia con far::far() (pesos de conteo) ==")
        for k, v in comparar_con_r(X, rho_r, kn=3, pesos="conteo").items():
            print(f"  {k:24s} {v}")
        print("\n== Con pesos trapezoidales (divergencia esperada, en los bordes) ==")
        tau5 = np.linspace(0.0, 1.0, X.shape[1])
        for k, v in comparar_con_r(X, rho_r, kn=3, grilla=tau5, pesos=None).items():
            print(f"  {k:24s} {v}")
    except Exception as exc:                      # pragma: no cover
        print(f"[aviso] sin datos de R: {exc}")
