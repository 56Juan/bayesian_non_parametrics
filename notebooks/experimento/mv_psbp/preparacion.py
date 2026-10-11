"""
Preparacion del contrato conjunto (paso 1 del ciclo Python -> MATLAB -> Python).

Respuesta: los K coeficientes BLANQUEADOS theta_w = theta L, con W = L L^T la
Gram de la base (`safe_chol`, la misma que usa el FAR del _05). Asi la norma
euclidea de theta_w es la norma L^2 de la curva y la mezcla, Sigma_h y la
asignacion de atomos se miden en la metrica correcta. Con base ortonormal
(Fourier de 200-202) W = I y theta_w = theta.

Predictores: theta_w rezagada L veces (p = K*L), los mismos para el gating y
la regresion de cada atomo; la inclusion apaga lo que sobra.

Hiperparametros (Chung y Dunson en escala cruda, como en las corridas 200-202):
  nu0 = q + 2, S0 = 0.5 * diag(var_train(theta_w))     <- analogo de btau = 0.5 sd^2
  taupsij_j = taupsij_z * var_train(x_j), mupsij = 0    <- gating equivariante a escala
  apij/bpij = 0.5/0.5 en el rezago 1, 0.5/5 en adelante
  ag = bg = 0.5, mumu = 0, taumu = 1, pwj = 0.5
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from model_psbp_fd.functions_models import FunctionalRepresentation
from model_psbp_fd.utils.linalg import safe_chol

ARCHIVOS = {
    "curvas": "X_curves.npy", "grilla": "domain_grid.npy",
    "theta": "theta.csv", "theta_w": "theta_w.csv", "gram_W": "gram_W.csv", "chol_L": "chol_L.csv",
    "fr_pickle": "functional_representation.pkl",
    "train": "dataset_mv_train.csv", "test": "dataset_mv_test.csv",
    "manifest": "datasets_manifest.json", "hiperparametros": "hyperparameters.json",
    "eval_config": "eval_config.json",
}


# ----------------------------------------------------------------- representacion
def representacion_conocida(Phi: np.ndarray, grilla: np.ndarray) -> FunctionalRepresentation:
    """Base conocida del generador (200-202): la representacion ES la base."""
    return FunctionalRepresentation.desde_base(np.asarray(Phi, float), np.asarray(grilla, float))


def representacion_bspline_gcv(X_train: np.ndarray, grilla: np.ndarray, nb_max: int = 12,
                               ordenes=(2, 3, 4)) -> tuple:
    """B-spline por GCV sobre train, como el 27_01, con tope `nb_max` en el numero de
    funciones: el tope existe porque p = K * L entra al gating por distancia."""
    L_ = X_train.shape[1]
    ss_tot = np.sum((X_train - X_train.mean(0, keepdims=True)) ** 2)
    reg = []
    for od in ordenes:
        for nb in range(max(4, od), nb_max + 1):
            fr = FunctionalRepresentation(method="bspline", n_basis=nb, order=od)
            TH = fr.fit_transform(X_train, grilla)
            sse = np.sum((X_train - fr.reconstruct(TH)) ** 2, axis=1)
            reg.append(dict(n_basis=nb, order=od, var_retained=1 - sse.sum() / ss_tot,
                            gcv_mean=float(np.mean(L_ * sse / (L_ - nb) ** 2))))
    df = pd.DataFrame(reg)
    best = df.nsmallest(1, "gcv_mean").iloc[0]
    fr = FunctionalRepresentation(method="bspline", n_basis=int(best.n_basis), order=int(best.order))
    fr.fit(X_train, grilla)
    return fr, df


def blanquear(fr: FunctionalRepresentation, THETA: np.ndarray) -> tuple:
    """theta_w = theta L con W = L L^T. Verifica que preserve la norma L^2."""
    W = np.asarray(fr.gram_, float)
    L = safe_chol(W)
    TW = THETA @ L
    n_g = np.einsum("tk,kl,tl->t", THETA, W, THETA)
    err = float(np.max(np.abs(n_g - (TW ** 2).sum(1))) / max(np.max(n_g), 1e-300))
    assert err < 1e-8, f"el blanqueo no preserva la norma L^2 (error relativo {err:.2e})"
    return TW, W, L


def desblanquear(TW: np.ndarray, L: np.ndarray) -> np.ndarray:
    return np.linalg.solve(L.T, np.asarray(TW, float).T).T


# ----------------------------------------------------------------------- datasets
def nombres_covariables(K: int, n_lags: int):
    return [f"coef_{k + 1}_lag{l}" for l in range(1, n_lags + 1) for k in range(K)]


def dataset_mv(TW: np.ndarray, n_lags: int) -> pd.DataFrame:
    """Fila t (t >= n_lags): respuesta theta_w[t] (K columnas) y predictores
    theta_w[t-1], ..., theta_w[t-n_lags]. Columna `t` con el indice del objetivo."""
    T, K = TW.shape
    Y = TW[n_lags:]
    Z = np.column_stack([TW[n_lags - l:T - l] for l in range(1, n_lags + 1)])
    df = pd.DataFrame(np.column_stack([np.arange(n_lags, T), Y, Z]),
                      columns=["t"] + [f"coef_{k + 1}" for k in range(K)] + nombres_covariables(K, n_lags))
    df["t"] = df["t"].astype(int)
    return df


# ---------------------------------------------------------------- hiperparametros
def hiperparametros_sub(df_train: pd.DataFrame, resp: list, covs: list, mcmc_config: Dict, n_chains: int,
                        seed_base: int, taupsij_z: float = 10.0, ab_por_lag: Optional[Dict] = None,
                        nombre: str = "bloque") -> Dict:
    """Mismo prior que `hiperparametros_mv` para un submodelo con respuesta `resp` y
    predictores `covs` (subconjuntos de columnas del dataset completo)."""
    ab_por_lag = ab_por_lag or {1: (0.5, 0.5), 2: (0.5, 5.0)}
    Y = df_train[resp].to_numpy(float); X = df_train[covs].to_numpy(float)
    var_y = Y.var(axis=0, ddof=1); var_x = X.var(axis=0, ddof=1)
    lags = [int(c.split("lag")[-1]) for c in covs]
    ab = [ab_por_lag.get(l, (0.5, 5.0)) for l in lags]
    q = len(resp)
    return {"modelo": "psbp_mv", "submodelo": nombre, "respuesta": list(resp),
            "global": {"ag": 0.5, "bg": 0.5, "mumu": 0.0, "taumu": 1.0, "pwj": 0.5,
                       "nu0": int(q + 2), "S0_diag": (0.5 * var_y).tolist(),
                       "S0_regla": "0.5 * var_train en la diagonal, nu0 = q + 2"},
            "por_predictor": {"nombres": list(covs), "apij": [a for a, _ in ab], "bpij": [b for _, b in ab],
                              "mupsij": [0.0] * len(covs), "taupsij": (taupsij_z * var_x).tolist(), "taupsij_z": taupsij_z},
            "mcmc_config": dict(mcmc_config), "n_iter": int(n_chains), "seed_base": int(seed_base),
            "seed_scheme": "seed_base + chain*9973", "q": int(q), "p": int(len(covs))}


def hiperparametros_mv(df_train: pd.DataFrame, K: int, n_lags: int, mcmc_config: Dict, n_chains: int,
                       seed_base: int, taupsij_z: float = 10.0, ab_por_lag: Optional[Dict] = None) -> Dict:
    ab_por_lag = ab_por_lag or {1: (0.5, 0.5), 2: (0.5, 5.0)}
    covs = nombres_covariables(K, n_lags)
    Y = df_train[[f"coef_{k + 1}" for k in range(K)]].to_numpy(float)
    X = df_train[covs].to_numpy(float)
    var_y = Y.var(axis=0, ddof=1); var_x = X.var(axis=0, ddof=1)
    lags = [int(c.split("lag")[-1]) for c in covs]
    ab = [ab_por_lag.get(l, (0.5, 5.0)) for l in lags]
    return {
        "modelo": "psbp_mv",
        "global": {"ag": 0.5, "bg": 0.5, "mumu": 0.0, "taumu": 1.0, "pwj": 0.5,
                   "nu0": int(K + 2), "S0_diag": (0.5 * var_y).tolist(),
                   "S0_regla": "0.5 * var_train(theta_w) en la diagonal, nu0 = q + 2"},
        "por_predictor": {"nombres": covs,
                          "apij": [a for a, _ in ab], "bpij": [b for _, b in ab],
                          "mupsij": [0.0] * len(covs),
                          "taupsij": (taupsij_z * var_x).tolist(),
                          "taupsij_z": taupsij_z},
        "mcmc_config": dict(mcmc_config), "n_iter": int(n_chains), "seed_base": int(seed_base),
        "seed_scheme": "seed_base + chain*9973",
        "q": int(K), "p": int(len(covs)), "n_lags": int(n_lags),
    }


# ------------------------------------------------------------------- orquestacion
def preparar_punto_mv(paths: Dict[str, Path], X: np.ndarray, grilla: np.ndarray, fr: FunctionalRepresentation,
                      T0: int, n_lags: int, mcmc_config: Dict, n_chains: int, seed_base: int,
                      meta: Optional[Dict] = None, taupsij_z: float = 10.0, nivel: float = 0.95,
                      ventanas_w=(20, 30, 40), objetivo: str = "curva_suavizada") -> Dict:
    """Escribe curvas, representacion, blanqueo, datasets, contrato y eval_config."""
    import pickle
    X = np.asarray(X, float); grilla = np.asarray(grilla, float); T = len(X)
    THETA = fr.transform(X, grilla); K = THETA.shape[1]
    TW, W, L = blanquear(fr, THETA)
    df = dataset_mv(TW, n_lags)
    tr, te = df[df.t < T0], df[df.t >= T0]
    f = paths["functional"]
    np.save(f / ARCHIVOS["curvas"], X); np.save(f / ARCHIVOS["grilla"], grilla)
    np.savetxt(f / ARCHIVOS["theta"], THETA, delimiter=","); np.savetxt(f / ARCHIVOS["theta_w"], TW, delimiter=",")
    np.savetxt(f / ARCHIVOS["gram_W"], W, delimiter=","); np.savetxt(f / ARCHIVOS["chol_L"], L, delimiter=",")
    with open(f / ARCHIVOS["fr_pickle"], "wb") as fh:
        pickle.dump(fr, fh)
    tr.to_csv(f / ARCHIVOS["train"], index=False); te.to_csv(f / ARCHIVOS["test"], index=False)
    hp = hiperparametros_mv(tr, K, n_lags, mcmc_config, n_chains, seed_base, taupsij_z)
    hp["partition"] = {"T": int(T), "T0": int(T0), "prop_train": float(T0 / T),
                       "n_train_eff": int(len(tr)), "n_test_eff": int(len(te))}
    hp["experiment_id"] = paths["eid"]
    hp["rutas"] = {"functional": str(f), "out_artefact": str(paths["out_artefact"])}
    manifest = {"modelo": "psbp_mv", "K": int(K), "n_lags": int(n_lags), "cov_names": nombres_covariables(K, n_lags),
                "respuesta": [f"coef_{k + 1}" for k in range(K)], "T": int(T), "T0": int(T0),
                "blanqueo": "theta_w = theta L, W = L L^T (safe_chol)",
                "base": fr.get_config() if hasattr(fr, "get_config") else {}, "meta": meta or {}}
    json.dump(manifest, open(f / ARCHIVOS["manifest"], "w"), indent=2)
    json.dump(hp, open(paths["out_artefact"] / ARCHIVOS["hiperparametros"], "w"), indent=2)
    ev = {"T": int(T), "T0": int(T0), "n_lags": int(n_lags), "K": int(K), "nivel_credibilidad": nivel,
          "objetivo_evaluacion": objetivo, "modo_residuo": "ninguno",
          "ventana_movil": {"w": list(ventanas_w), "paso": 1, "solapadas": True},
          "metricas_A": ["mae_f", "rmse_f", "linf_medio", "linf_max"],
          "metricas_B": ["mpiw", "picp", "picp_simultaneo", "winkler", "winkler_max_medio", "winkler_max_glob"]}
    json.dump(ev, open(paths["out_artefact"] / ARCHIVOS["eval_config"], "w"), indent=2)
    return {"THETA": THETA, "TW": TW, "W": W, "L": L, "K": K, "df": df, "hp": hp, "eval_config": ev}


def preparar_punto_bloque(paths: Dict[str, Path], X: np.ndarray, grilla: np.ndarray, fr: FunctionalRepresentation,
                          T0: int, n_lags: int, mcmc_config: Dict, n_chains: int, seed_base: int,
                          bloque=(0, 1, 2, 3), meta: Optional[Dict] = None, taupsij_z: float = 10.0,
                          nivel: float = 0.95, ventanas_w=(20, 30, 40), objetivo: str = "curva_suavizada") -> Dict:
    """Bloque conjunto sobre las coordenadas `bloque` (predictores: los n_lags rezagos de
    TODAS las coordenadas) y un univariado por coordenada restante con sus propios
    rezagos. Un dataset y un hyperparameters_<sub>.json por submodelo."""
    import pickle
    X = np.asarray(X, float); grilla = np.asarray(grilla, float); T = len(X)
    THETA = fr.transform(X, grilla); K = THETA.shape[1]
    TW, W, L = blanquear(fr, THETA)
    df = dataset_mv(TW, n_lags)
    tr, te = df[df.t < T0], df[df.t >= T0]
    f = paths["functional"]; a = paths["out_artefact"]
    np.save(f / ARCHIVOS["curvas"], X); np.save(f / ARCHIVOS["grilla"], grilla)
    np.savetxt(f / ARCHIVOS["theta"], THETA, delimiter=","); np.savetxt(f / ARCHIVOS["theta_w"], TW, delimiter=",")
    np.savetxt(f / ARCHIVOS["gram_W"], W, delimiter=","); np.savetxt(f / ARCHIVOS["chol_L"], L, delimiter=",")
    with open(f / ARCHIVOS["fr_pickle"], "wb") as fh:
        pickle.dump(fr, fh)
    tr.to_csv(f / ARCHIVOS["train"], index=False); te.to_csv(f / ARCHIVOS["test"], index=False)
    covs_todas = nombres_covariables(K, n_lags)
    subs = [{"nombre": "bloque", "coords": [int(k) for k in bloque],
             "respuesta": [f"coef_{k + 1}" for k in bloque], "predictores": covs_todas}]
    for k in range(K):
        if k in bloque:
            continue
        subs.append({"nombre": f"uni_{k + 1}", "coords": [k], "respuesta": [f"coef_{k + 1}"],
                     "predictores": [f"coef_{k + 1}_lag{l}" for l in range(1, n_lags + 1)]})
    hps = {}
    for i, sm in enumerate(subs):
        cols = ["t"] + sm["respuesta"] + sm["predictores"]
        tr[cols].to_csv(f / f"dataset_{sm['nombre']}_train.csv", index=False)
        te[cols].to_csv(f / f"dataset_{sm['nombre']}_test.csv", index=False)
        hp = hiperparametros_sub(tr, sm["respuesta"], sm["predictores"], mcmc_config, n_chains,
                                 seed_base + 1000 * i, taupsij_z, nombre=sm["nombre"])
        hp["partition"] = {"T": int(T), "T0": int(T0), "prop_train": float(T0 / T),
                           "n_train_eff": int(len(tr)), "n_test_eff": int(len(te))}
        hp["experiment_id"] = paths["eid"]
        json.dump(hp, open(a / f"hyperparameters_{sm['nombre']}.json", "w"), indent=2)
        sm["hyperparameters"] = str(a / f"hyperparameters_{sm['nombre']}.json")
        sm["dataset_train"] = str(f / f"dataset_{sm['nombre']}_train.csv")
        sm["out_prefix"] = f"chain_{sm['nombre']}"
        hps[sm["nombre"]] = hp
    manifest = {"modelo": "psbp_bloque", "K": int(K), "n_lags": int(n_lags), "cov_names": covs_todas,
                "respuesta": [f"coef_{k + 1}" for k in range(K)], "T": int(T), "T0": int(T0),
                "bloque": [int(k) for k in bloque], "submodelos": subs,
                "blanqueo": "theta_w = theta L, W = L L^T (safe_chol)",
                "base": fr.get_config() if hasattr(fr, "get_config") else {}, "meta": meta or {}}
    json.dump(manifest, open(f / ARCHIVOS["manifest"], "w"), indent=2)
    json.dump({"modelo": "psbp_bloque", "submodelos": [sm["nombre"] for sm in subs], "mcmc_config": dict(mcmc_config),
               "n_iter": int(n_chains), "seed_base": int(seed_base), "q": int(K), "p": int(len(covs_todas)),
               "n_lags": int(n_lags), "partition": hps["bloque"]["partition"], "experiment_id": paths["eid"]},
              open(a / ARCHIVOS["hiperparametros"], "w"), indent=2)
    ev = {"T": int(T), "T0": int(T0), "n_lags": int(n_lags), "K": int(K), "nivel_credibilidad": nivel,
          "objetivo_evaluacion": objetivo, "modo_residuo": "ninguno",
          "ventana_movil": {"w": list(ventanas_w), "paso": 1, "solapadas": True},
          "metricas_A": ["mae_f", "rmse_f", "linf_medio", "linf_max"],
          "metricas_B": ["mpiw", "picp", "picp_simultaneo", "winkler", "winkler_max_medio", "winkler_max_glob"]}
    json.dump(ev, open(a / ARCHIVOS["eval_config"], "w"), indent=2)
    return {"THETA": THETA, "TW": TW, "W": W, "L": L, "K": K, "df": df, "hp": hps, "eval_config": ev,
            "submodelos": subs}


def cargar_punto_mv(paths: Dict[str, Path]) -> Dict:
    import pickle
    f = paths["functional"]
    with open(f / ARCHIVOS["fr_pickle"], "rb") as fh:
        fr = pickle.load(fh)
    d = {"fr": fr,
         "X": np.load(f / ARCHIVOS["curvas"]), "grilla": np.load(f / ARCHIVOS["grilla"]),
         "THETA": np.loadtxt(f / ARCHIVOS["theta"], delimiter=","),
         "TW": np.loadtxt(f / ARCHIVOS["theta_w"], delimiter=","),
         "L": np.atleast_2d(np.loadtxt(f / ARCHIVOS["chol_L"], delimiter=",")),
         "train": pd.read_csv(f / ARCHIVOS["train"]), "test": pd.read_csv(f / ARCHIVOS["test"]),
         "manifest": json.load(open(f / ARCHIVOS["manifest"])),
         "hp": json.load(open(paths["out_artefact"] / ARCHIVOS["hiperparametros"])),
         "eval_config": json.load(open(paths["out_artefact"] / ARCHIVOS["eval_config"]))}
    d["df"] = pd.concat([d["train"], d["test"]], ignore_index=True)
    return d


def escribir_jobs(ruta_json: Path, lista_paths) -> None:
    """Lista de contratos para el driver MATLAB (rutas absolutas)."""
    jobs = []
    for p in lista_paths:
        man = json.load(open(p["functional"] / ARCHIVOS["manifest"]))
        if man.get("modelo") == "psbp_bloque":
            for sm in man["submodelos"]:
                jobs.append({"experiment_id": f"{p['eid']}/{sm['nombre']}", "hyperparameters": sm["hyperparameters"],
                             "dataset_train": sm["dataset_train"], "out_artefact": str(p["out_artefact"]),
                             "out_prefix": sm["out_prefix"]})
        else:
            jobs.append({"experiment_id": p["eid"], "hyperparameters": str(p["out_artefact"] / ARCHIVOS["hiperparametros"]),
                         "dataset_train": str(p["functional"] / ARCHIVOS["train"]), "out_artefact": str(p["out_artefact"]),
                         "out_prefix": "chain_mv"})
    json.dump(jobs, open(ruta_json, "w"), indent=2)
