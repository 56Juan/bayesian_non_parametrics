"""
preparacion_barrido.py
======================
Preparacion de cada punto del barrido en M, SIN graficos (las figuras viven en
`graphics/`). Es la logica que antes era `procesar_punto` dentro de `_01`: toma los
scores del punto, arma los datasets AR, los hiperparametros y el contrato, y verifica
que todo cruce.

    covariables_ar          nombres de covariables por componente (rezago propio o cruzado)
    datasets_ar             respuesta + predictores en t-1 .. t-N_LAGS, por componente y bloque
    diagnostico_rezagos     correlaciones (Pearson y Spearman) respuesta vs rezagos, solo train
    hiperparametros_chung   Chung y Dunson 4.2 en escala cruda, con las sd de TRAIN
    preparar_punto          orquesta lo anterior para un M y escribe los artefactos

`preparar_punto(M, ctx)` recibe un diccionario `ctx` con lo comun a todos los puntos
(ver su docstring): ese es el mecanismo que garantiza que los puntos comparten
realizacion y base, sin depender de variables globales del notebook.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from ..functions_models import DataStandardizer
from ..pipelines import (
    guardar_datasets_ar, guardar_estandarizador, guardar_fpca,
    guardar_hiperparametros, guardar_config_evaluacion, verificar_contrato,
)
from ..utils.rutas import experiment_id
from .baselines import tabla_baselines

__all__ = ["covariables_ar", "datasets_ar", "diagnostico_rezagos", "hiperparametros_chung",
           "preparar_punto"]

_CLAVES_CTX = (
    "PATHS_POR_M", "BASENAME", "ESCENARIO_ID", "REPLICA_ID", "SEED", "fpca", "THETA",
    "idx_train", "idx_test", "T", "T0", "PROP_TRAIN", "N_LAGS", "CRUZADOS", "M_ENTRENO",
    "M_SUGERIDO", "HP_GLOBAL", "HP_BY_TYPE", "HP_CHUNG_ESC", "VARIANTE", "VAR_CFG",
    "MCMC_CONFIG", "N_CHAINS", "X_SUAV", "grilla",
)


def covariables_ar(n_components: int, n_lags: int, cruzados: bool):
    """
    `(cov_names, cov_por_comp, covariables)`. `cov_names` es la union en orden lag-mayor
    (todas las componentes del lag 1, luego del lag 2, ...). Con rezago PROPIO la
    componente k solo ve `fpc_<k+1>_lag1..N_LAGS`; con CRUZADOS ve todas las de `cov_names`
    (p = M x N_LAGS).
    """
    cov_names = [f"fpc_{j + 1}_lag{lag}" for lag in range(1, n_lags + 1) for j in range(n_components)]
    if cruzados:
        cov_por_comp = {k: list(cov_names) for k in range(n_components)}
    else:
        cov_por_comp = {k: [f"fpc_{k + 1}_lag{lag}" for lag in range(1, n_lags + 1)]
                        for k in range(n_components)}
    return cov_names, cov_por_comp, "todos_los_rezagos" if cruzados else "rezago_propio"


def datasets_ar(SCORES: np.ndarray, T: int, T0: int, n_lags: int, cov_por_comp: Dict,
                cruzados: bool) -> Tuple[Dict, Dict]:
    """`(dfs_train, dfs_test)`: por componente, la respuesta en t y los predictores en
    t-1 .. t-N_LAGS (la primera columna es la respuesta)."""
    n_components = SCORES.shape[1]

    def bloque(k, t_ini, t_fin):
        t_idx = np.arange(t_ini, t_fin)
        X = (np.hstack([SCORES[t_idx - lag, :] for lag in range(1, n_lags + 1)]) if cruzados else
             np.column_stack([SCORES[t_idx - lag, k] for lag in range(1, n_lags + 1)]))
        return pd.DataFrame(np.column_stack([SCORES[t_idx, k], X]),
                            columns=[f"fpc_{k + 1}"] + cov_por_comp[k])

    return ({k: bloque(k, n_lags, T0) for k in range(n_components)},
            {k: bloque(k, T0, T) for k in range(n_components)})


def _spearman(Y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Spearman columna a columna via rangos (pandas, sin scipy)."""
    Yc = pd.DataFrame(Y).rank().to_numpy(); Yc = Yc - Yc.mean(0)
    Xc = pd.DataFrame(X).rank().to_numpy(); Xc = Xc - Xc.mean(0)
    return (Yc.T @ Xc) / np.outer(np.sqrt((Yc ** 2).sum(0)), np.sqrt((Xc ** 2).sum(0)))


def diagnostico_rezagos(S_train: np.ndarray, n_lags_max: int = 3) -> Dict:
    """
    Correlacion de cada score en t con cada score rezagado 1..n_lags_max, sobre el bloque
    de entrenamiento: matrices `corr_pearson` y `corr_spearman` (K, K * n_lags_max), las
    etiquetas de filas y columnas (LaTeX) y la banda +-1.96/sqrt(n).
    """
    T_, K = S_train.shape
    y = S_train[n_lags_max:, :]
    cp, cs, cols = np.zeros((K, K * n_lags_max)), np.zeros((K, K * n_lags_max)), []
    for lag in range(1, n_lags_max + 1):
        x = S_train[n_lags_max - lag:T_ - lag, :]
        sp = _spearman(y, x)
        for j in range(K):
            c = (lag - 1) * K + j
            for k in range(K):
                cp[k, c] = np.corrcoef(y[:, k], x[:, j])[0, 1]
            cs[:, c] = sp[:, j]
            cols.append(rf"$\xi_{{t-{lag},{j + 1}}}$")
    return {"corr_pearson": cp, "corr_spearman": cs, "col_labels": cols,
            "row_labels": [rf"$\xi_{{t,{k + 1}}}$" for k in range(K)],
            "band": 1.96 / np.sqrt(len(y)), "n_lags_max": n_lags_max, "K_total": K}


def _clasificar(nombre: str, k: int, cov_por_comp: Dict) -> str:
    """`own_lag<l>` o `cross_lag<l>` a partir de `fpc_<idx>_lag<l>`."""
    assert nombre in cov_por_comp[k], f"{nombre} no es covariable de la componente {k}."
    idx, lag = nombre[len("fpc_"):].split("_lag")
    return f"{'own' if int(idx) == k + 1 else 'cross'}_lag{int(lag)}"


def hiperparametros_chung(SCORES_train: np.ndarray, dfs_train: Dict, cov_por_comp: Dict,
                          HP_GLOBAL: Dict, HP_BY_TYPE: Dict, HP_CHUNG_ESC: Dict) -> List[Dict]:
    """
    Chung y Dunson (2009) 4.2 llevado a la escala cruda con las sd de TRAIN: si x = sd z, el
    prior del paper sobre z equivale a `btau = btau_z sd_y^2` y `taupsij = taupsij_z sd_x^2`,
    de modo que el prior escala con la varianza de cada componente. `mupsij = 0`. La
    inclusion (`apij`, `bpij`) sale del tipo y lag de cada covariable.
    """
    lista = []
    for k in range(SCORES_train.shape[1]):
        sd_y = float(SCORES_train[:, k].std(ddof=0))
        sd_x = dfs_train[k][cov_por_comp[k]].to_numpy().std(axis=0, ddof=0)
        ab = [(float(HP_BY_TYPE[_clasificar(nm, k, cov_por_comp)]["apij"]),
               float(HP_BY_TYPE[_clasificar(nm, k, cov_por_comp)]["bpij"])) for nm in cov_por_comp[k]]
        lista.append({**HP_GLOBAL,
                      "btau": HP_CHUNG_ESC["btau_z"] * sd_y ** 2,
                      "apij": np.array([a for a, _ in ab], dtype=float),
                      "bpij": np.array([b for _, b in ab], dtype=float),
                      "mupsij": np.zeros(len(cov_por_comp[k])),
                      "taupsij": HP_CHUNG_ESC["taupsij_z"] * sd_x ** 2})
    return lista


def _eval_config(M: int, n_lags: int, T: int, T0: int, prop_train: float) -> Dict:
    """`eval_config.json`. Las ventanas, el nivel y el objetivo se LEEN de aqui en `_04` y
    `_05`; no se redeclaran. Objetivo: curva suavizada (docs 03_05_00), nunca la grilla
    cruda; `modo_residuo = "ninguno"` esta forzado por el modelo."""
    return {
        "scheme": "holdout_temporal", "T": int(T), "T0": int(T0), "prop_train": float(prop_train),
        "horizons": [1], "n_lags": int(n_lags), "m_fpca": int(M),
        "scores_scale": "raw_fpca_scores",
        "objetivo_evaluacion": "curva_suavizada",
        "objetivo_secundario": "representacion_fpca",
        "modo_residuo": "ninguno",
        "nivel_credibilidad": 0.95,
        "ventana_movil": {"w": [20, 30, 40], "paso": 1, "solapadas": True},
        "metricas_bloque_A": ["mae_f", "rmse_f", "linf_medio", "linf_max"],
        "metricas_bloque_B": ["mpiw", "picp", "picp_simultaneo", "winkler",
                              "winkler_max_medio", "winkler_max_glob"],
    }


def _hp_artifact(M: int, ctx: Dict, hp_list: List[Dict], n_components: int, covariables: str,
                 n_train_eff: int, n_test_eff: int) -> Dict:
    """`hyperparameters.json`: la unica fuente de verdad del contrato Python <-> MATLAB."""
    c = ctx
    m_ent = M if c["CRUZADOS"] else c["M_ENTRENO"]
    return {
        "global": c["HP_GLOBAL"], "by_type": c["HP_BY_TYPE"],
        "variante": {"nombre": c["VARIANTE"], **c["VAR_CFG"],
                     "grid_gamma": "por_predictor_rango_train_con_extremos",
                     **({"chung_escala_cruda": {**c["HP_CHUNG_ESC"],
                         "ab_por_lag": {str(l): v for l, v in c["HP_CHUNG_ESC"]["ab_por_lag"].items()}}}
                        if c["VAR_CFG"]["hiperparametros"] == "chung_escala_cruda" else {})},
        "mcmc_config": c["MCMC_CONFIG"], "n_iter": c["N_CHAINS"],
        "seed_scheme": "seed_base + chain*9973 + k*31",
        "escenario_id": int(c["ESCENARIO_ID"]), "replica_id": int(c["REPLICA_ID"]),
        "m_fpca": int(M), "m_entrenamiento": int(m_ent),
        "trazas_en": experiment_id(c["BASENAME"], c["ESCENARIO_ID"], c["REPLICA_ID"], m_ent),
        "covariables": covariables, "seed_base": int(c["SEED"]),
        "scores_scale": "raw_fpca_scores",
        "partition": {
            "T": int(c["T"]), "T0": int(c["T0"]), "prop_train": float(c["PROP_TRAIN"]),
            "n_train_eff": int(n_train_eff), "n_test_eff": int(n_test_eff),
            "train_files": [f"dataset_fpc_{k + 1}_train.csv" for k in range(n_components)],
            "test_files": [f"dataset_fpc_{k + 1}_test.csv" for k in range(n_components)],
        },
        "hyperparams_list": [
            {"component_k": k, "fpc_idx": int(k + 1),
             "hyperparams": {key: (v.tolist() if isinstance(v, np.ndarray) else v)
                             for key, v in hp_list[k].items()}}
            for k in range(n_components)],
    }


def preparar_punto(M: int, ctx: Dict) -> Dict:
    """
    Produce TODOS los artefactos del punto M del barrido y verifica su contrato.

    `ctx` trae lo comun a todos los puntos y NO se recalcula (garantia de que comparten
    realizacion y base): `PATHS_POR_M`, `BASENAME`, `ESCENARIO_ID`, `REPLICA_ID`, `SEED`,
    `fpca` (de oraculo), `THETA`, `idx_train`, `idx_test`, `T`, `T0`, `PROP_TRAIN`,
    `N_LAGS`, `CRUZADOS`, `M_ENTRENO`, `M_SUGERIDO`, `HP_GLOBAL`, `HP_BY_TYPE`,
    `HP_CHUNG_ESC`, `VARIANTE`, `VAR_CFG`, `MCMC_CONFIG`, `N_CHAINS`, `X_SUAV` y `grilla`.

    Los scores van CRUDOS (el estandarizador se ajusta con train para registrar
    `n_ajuste`, que `verificar_contrato` exige, y se deja en la identidad). Retorna el
    resumen del punto, mas `diagnostico_rezagos` y `baselines` para que el notebook los
    dibuje y los muestre.
    """
    faltan = [k for k in _CLAVES_CTX if k not in ctx]
    assert not faltan, f"ctx sin: {faltan}"
    c = ctx
    fpca, THETA, T, T0, n_lags = c["fpca"], c["THETA"], c["T"], c["T0"], c["N_LAGS"]
    PATHS = c["PATHS_POR_M"][M]
    eid = experiment_id(c["BASENAME"], c["ESCENARIO_ID"], c["REPLICA_ID"], M)
    print(f"\n{'=' * 70}\nM = {M}   ·   {eid}\n{'=' * 70}")

    # -- componentes retenidas y scores ---------------------------------------
    assert 1 <= M <= fpca.evals.size, f"M fuera de [1, {fpca.evals.size}]."
    coincide = (M == c["M_SUGERIDO"])
    print(f"M(95 %) segun la regla : {c['M_SUGERIDO']}   (K disponible = {fpca.evals.size})")
    print(f"M de este punto        : {M}"
          + ("   (coincide con la regla)" if coincide else "   <- DIFIERE de la regla: es un punto de sensibilidad"))
    fpca.set_M(int(M))
    SCORES = fpca.transform(THETA)                      # (T, M): xi del generador, primeras M
    S_tr, S_te = SCORES[c["idx_train"]], SCORES[c["idx_test"]]
    print(f"M = {fpca.M}   var. explicada = {fpca.var_cum[fpca.M - 1]:.4%}")
    print(f"[train] max|media xi| = {np.abs(S_tr.mean(0)).max():.2e}   (~ 0)")
    print(f"[test]  max|media xi| = {np.abs(S_te.mean(0)).max():.3f}")
    print(f"[test]  var xi / lambda = {np.array2string(S_te.var(0, ddof=1) / fpca.lambdas, precision=3)}")

    # -- escala: SIN estandarizar ---------------------------------------------
    std = DataStandardizer(method="zscore_column", ddof=0)
    std.fit(S_tr, etiqueta=f"train[1:{T0}]")
    chk = std.verificar_ajuste(T0)
    print(f"[holdout] ajustado con {chk['n_ajuste']} filas = T0 ({chk['etiqueta_ajuste']}) -> ok={chk['ajuste_ok']}")
    std.mean, std.std = np.zeros_like(std.mean), np.ones_like(std.std)     # identidad
    SCORES_STD = std.transform(SCORES)
    assert np.allclose(SCORES_STD, SCORES), "la transformacion deberia ser la identidad."
    guardar_estandarizador(PATHS, std)
    res = guardar_fpca(PATHS, fpca, SCORES, SCORES_STD=SCORES_STD, meta_extra={"T0": int(T0)})
    print(f"\n[functional] artefactos FPCA · cond_W = {res['meta']['cond_W']:.3e}")

    # -- diagnostico de rezagos (solo train) ----------------------------------
    diag = diagnostico_rezagos(SCORES_STD[c["idx_train"]])

    # -- orden AR y datasets --------------------------------------------------
    n_components = SCORES_STD.shape[1]
    n_train_eff, n_test_eff = T0 - n_lags, T - T0
    assert T0 > n_lags
    cov_names, cov_por_comp, covariables = covariables_ar(n_components, n_lags, c["CRUZADOS"])
    print(f"componentes : {n_components}")
    print(f"N_LAGS      : {n_lags}   ·   p = {len(cov_por_comp[0])} covariable(s) por componente ({covariables})")
    print(f"n_train_eff : {n_train_eff}   n_test_eff : {n_test_eff}")
    print(f"cov_names   : {cov_names}")
    dfs_train, dfs_test = datasets_ar(SCORES_STD, T, T0, n_lags, cov_por_comp, c["CRUZADOS"])
    manifest = {
        "scores_scale": "raw_fpca_scores", "n_components": n_components, "n_lags": int(n_lags),
        "component_idx": list(range(n_components)), "cov_names": cov_names,
        "cov_por_componente": {str(k): cov_por_comp[k] for k in range(n_components)},
        "covariables": covariables, "T": int(T), "T0": int(T0), "prop_train": float(c["PROP_TRAIN"]),
        "n_train_eff": int(n_train_eff), "n_test_eff": int(n_test_eff), "ajuste_en": "train",
    }
    guardar_datasets_ar(PATHS, dfs_train, dfs_test, manifest)
    print(f"[functional] {2 * n_components} datasets + datasets_manifest.json")
    for k in range(n_components):
        print(f"  fpc_{k + 1}: train {dfs_train[k].shape} · test {dfs_test[k].shape}")

    # -- hiperparametros y MCMC ----------------------------------------------
    hp_list = hiperparametros_chung(S_tr, dfs_train, cov_por_comp, c["HP_GLOBAL"], c["HP_BY_TYPE"],
                                    c["HP_CHUNG_ESC"])
    print(f"\nN_CHAINS = {c['N_CHAINS']} cadenas por componente -> {c['N_CHAINS'] * n_components} jobs en MATLAB para este M")
    print(f"  {'k':>2} {'fpc':>4} {'var(y)':>9} {'E[tau]':>8} {'btau':>9}  taupsij")
    for k in range(n_components):
        hp = hp_list[k]
        print(f"  {k:>2} {k + 1:>4} {S_tr[:, k].var(ddof=0):>9.5f} {hp['atau'] / hp['btau']:>8.2f} "
              f"{hp['btau']:>9.5f}  {np.array2string(hp['taupsij'], precision=3)}")
    guardar_hiperparametros(PATHS, _hp_artifact(M, c, hp_list, n_components, covariables,
                                                n_train_eff, n_test_eff))
    print(f"[out_artefact] hyperparameters.json -> {PATHS['out_artefact']}")

    # -- evaluacion, lineas base y contrato -----------------------------------
    guardar_config_evaluacion(PATHS, _eval_config(M, n_lags, T, T0, c["PROP_TRAIN"]))
    print("[out_artefact] eval_config.json")
    baselines = tabla_baselines(SCORES_STD, T0, estandarizador=std, fpca=fpca,
                                X_obs=c["X_SUAV"], tau=c["grilla"], h=1)
    baselines.to_csv(PATHS["out_report"] / "30_baselines_test.csv")
    informe = verificar_contrato(PATHS)
    print(f"contrato_ok = {informe['contrato_ok']}   (M={informe['M']}, K={informe['K']}, "
          f"T0={informe['T0']}, n_components={informe['n_components']})")
    print(f"estandarizador ajustado con {informe.get('estandarizador_n_ajuste')} filas (T0={informe['T0']})")
    print(f"verificacion FPCA: todo_ok = {informe['verificacion_fpca']['todo_ok']}")
    if not informe["contrato_ok"]:
        print("\nPROBLEMAS:")
        for p in informe.get("problemas", []):
            print(f"  - {p}")
    return {"M": int(M), "experiment_id": eid, "var_acum": float(fpca.var_cum[fpca.M - 1]),
            "K": int(fpca.evals.size), "coincide_regla_95": bool(coincide),
            "n_components": int(n_components), "p_covariables": int(len(cov_por_comp[0])),
            "contrato_ok": bool(informe["contrato_ok"]), "problemas": informe.get("problemas", []),
            "diagnostico_rezagos": diag, "baselines": baselines}
