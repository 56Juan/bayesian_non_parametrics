"""
test_sim_escenario_B_mezcla.py
================================
Pruebas de `pipelines/sim_escenario_B1.py`, `B2.py` y `B3.py`: la mezcla de
mecanismos funcionales del anexo actualizado (`docs/01 Anexo.tex`,
`ane_00_02_01_alg_b1` a `_03_alg_b3`), NO el generador historico (FAR
exponencial / TV-FAR / cambio estructural) que sigue en
`pipelines/deprecated/`.

Como se corren
---------------
    python -m pytest tests/test_sim_escenario_B_mezcla.py -v

El archivo tambien es ejecutable sin pytest (`python tests/test_sim_escenario_B_mezcla.py`).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

try:
    import pytest
except ModuleNotFoundError:                      # ejecucion directa sin pytest
    class _Pytest:
        @staticmethod
        def raises(exc):
            class _Ctx:
                def __enter__(self_):
                    return self_
                def __exit__(self_, exc_type, exc_val, tb):
                    if exc_type is None:
                        raise AssertionError(f"se esperaba {exc}")
                    return issubclass(exc_type, exc)
            return _Ctx()
    pytest = _Pytest()

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from model_psbp_fd.pipelines.sim_escenario_B1 import (    # noqa: E402
    ConfigEscenarioB1,
    calcular_rasgos,
    funciones_rasgo,
    generar_escenario_B1,
    grilla_regular,
    pesos_trapezoidales,
)
from model_psbp_fd.pipelines.sim_escenario_B2 import (    # noqa: E402
    ConfigEscenarioB2,
    generar_escenario_B2,
)
from model_psbp_fd.pipelines.sim_escenario_B3 import (    # noqa: E402
    ConfigEscenarioB3,
    generar_escenario_B3,
    probabilidades_softmax,
)

# Config chica y comun a las pruebas: no hace falta T=1000 para verificar
# formas, finitud o acotamiento.
_KW = dict(L=75, T=200, burn_in=100, R=3, sigma_obs=0.25, seed=41232)


# ==========================================================================
# 1. RASGOS FUNCIONALES: ORTONORMALIDAD Y PROYECCION
# ==========================================================================

def test_funciones_rasgo_ortonormales():
    tau = grilla_regular(75)
    pesos = pesos_trapezoidales(tau)
    Phi = funciones_rasgo(tau)
    Gram = calcular_rasgos(Phi, pesos, Phi)   # (3, 3): <phi_i, phi_j>
    assert np.allclose(Gram, np.eye(3), atol=5e-3)


def test_calcular_rasgos_forma():
    tau = grilla_regular(75)
    pesos = pesos_trapezoidales(tau)
    Phi = funciones_rasgo(tau)
    X = np.zeros((10, 75))
    R = calcular_rasgos(X, pesos, Phi)
    assert R.shape == (10, 3)
    assert np.allclose(R, 0.0)


# ==========================================================================
# 2. VALIDACION DE CONFIGURACION
# ==========================================================================

def test_pi_debe_sumar_uno():
    with pytest.raises(ValueError):
        ConfigEscenarioB1(pi=(0.5, 0.5, 0.5)).validar()


def test_pi_no_negativa():
    with pytest.raises(ValueError):
        ConfigEscenarioB1(pi=(-0.1, 0.6, 0.5)).validar()


def test_sigma_eps_positivo():
    with pytest.raises(ValueError):
        ConfigEscenarioB1(sigma_eps=0.0).validar()


def test_ell_positivo():
    with pytest.raises(ValueError):
        ConfigEscenarioB1(ell=-0.1).validar()


# ==========================================================================
# 3. FORMA, FINITUD Y ACOTAMIENTO DE LOS TRES GENERADORES
# ==========================================================================

def _revisar_salida_basica(salida, T, L, R):
    assert salida.observaciones.shape == (R, T, L)
    assert salida.curvas.shape == (R, T, L)
    assert np.all(np.isfinite(salida.curvas))
    assert np.all(np.isfinite(salida.observaciones))
    # Acotamiento: con los parametros calibrados, |X| no deberia acercarse a
    # ordenes de magnitud propios de una recursion divergente.
    assert np.max(np.abs(salida.curvas)) < 20.0


def test_generar_escenario_B1_forma_y_acotamiento():
    salida = generar_escenario_B1(ConfigEscenarioB1(**_KW))
    _revisar_salida_basica(salida, _KW["T"], _KW["L"], _KW["R"])


def test_generar_escenario_B2_forma_y_acotamiento():
    salida = generar_escenario_B2(ConfigEscenarioB2(**_KW))
    _revisar_salida_basica(salida, _KW["T"], _KW["L"], _KW["R"])


def test_generar_escenario_B3_forma_y_acotamiento():
    salida = generar_escenario_B3(ConfigEscenarioB3(**_KW))
    _revisar_salida_basica(salida, _KW["T"], _KW["L"], _KW["R"])


# ==========================================================================
# 4. REPRODUCIBILIDAD
# ==========================================================================

def test_misma_semilla_misma_salida():
    s1 = generar_escenario_B1(ConfigEscenarioB1(**_KW))
    s2 = generar_escenario_B1(ConfigEscenarioB1(**_KW))
    assert np.array_equal(s1.curvas, s2.curvas)
    assert np.array_equal(s1.observaciones, s2.observaciones)


# ==========================================================================
# 5. FRECUENCIA DE MECANISMOS: B-1/B-2 (PI FIJA) VS B-3 (DEPENDIENTE)
# ==========================================================================

def test_B1_frecuencia_mecanismos_cerca_de_pi():
    kw = dict(_KW, T=2000, burn_in=200, R=5)
    salida = generar_escenario_B1(ConfigEscenarioB1(**kw))
    frac = np.array(salida.diagnostico["pi_empirica"])
    # T*R = 10000 sorteos; error de Monte Carlo de una Bernoulli(1/3) con ese
    # tamano de muestra es << 0.05.
    assert np.max(np.abs(frac - 1.0 / 3.0)) < 0.05


def test_B3_asignacion_depende_de_los_rasgos():
    salida = generar_escenario_B3(ConfigEscenarioB3(**_KW))
    P = salida.internos["probabilidades_mecanismo"].reshape(-1, 3)
    # Si la asignacion fuera efectivamente independiente de R, P seria
    # practicamente constante en (1/3, 1/3, 1/3); con la calibracion vigente
    # se aleja apreciablemente en promedio.
    assert P.std(axis=0).min() > 0.02
    assert salida.diagnostico["prob_mecanismo_desviacion_del_centro"] > 0.02


def test_probabilidades_softmax_suman_uno():
    rng = np.random.default_rng(0)
    for _ in range(20):
        R = rng.normal(size=3)
        p = probabilidades_softmax(R)
        assert p.shape == (3,)
        assert np.all(p >= 0.0)
        assert abs(p.sum() - 1.0) < 1e-10


# ==========================================================================
# 6. ORACULO DE UN REZAGO
# ==========================================================================

def test_oraculo_R2_en_rango_valido():
    for Cfg, gen in (
        (ConfigEscenarioB1, generar_escenario_B1),
        (ConfigEscenarioB2, generar_escenario_B2),
        (ConfigEscenarioB3, generar_escenario_B3),
    ):
        salida = gen(Cfg(**_KW))
        r2 = salida.diagnostico["r2_oraculo_1rezago"]
        assert np.isfinite(r2)
        # El oraculo es la esperanza condicional exacta: nunca puede ser peor
        # que la media incondicional en la MUESTRA que el genero (R^2 >= 0
        # salvo ruido de muestra chico), ni mayor que 1.
        assert -0.05 <= r2 <= 1.0


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    for nombre, Cfg, gen in (
        ("B1", ConfigEscenarioB1, generar_escenario_B1),
        ("B2", ConfigEscenarioB2, generar_escenario_B2),
        ("B3", ConfigEscenarioB3, generar_escenario_B3),
    ):
        salida = gen(Cfg(L=75, T=1000, burn_in=300, R=5, sigma_obs=0.25, seed=41232))
        print(f"\n== {nombre} ==")
        for k, v in salida.diagnostico.items():
            print(f"  {k:32s} {v}")
