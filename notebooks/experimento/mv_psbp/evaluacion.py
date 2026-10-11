"""
Paso 04: prediccion a h=1 del PSBPM-FD conjunto sobre toda la serie (train y
test, origenes t >= n_lags), curvas por `fr.reconstruct(theta)`, banda por
cuantiles de las muestras de curva, y las diez metricas sobre la ventana movil
(`ventana_movil_funcional`, bloque_A=True), contra la curva suavizada.

Persiste `banda_funcional_psbp_mv.npz` con la misma convencion que el _04
(`t_orig` base-1 del objetivo) para que el paso 05 la lea sin recalcular.
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model_psbp_fd.fit.rolling import ventana_movil_funcional
from model_psbp_fd.fit.metrics_puntual import resumen_error_funcional
from model_psbp_fd.fit.metrics_distribucional import resumen_intervalo, indicador_cobertura_simultanea
from model_psbp_fd.graphics.viz_evaluacion import plot_ventana_movil

from .preparacion import cargar_punto_mv, desblanquear
from .trazas import ModeloTrazaMV, ModeloBloque

METRICAS_A = ["mae_f", "rmse_f", "linf_medio", "linf_max"]
METRICAS_B = ["mpiw", "picp", "picp_simultaneo", "winkler", "winkler_max_medio", "winkler_max_glob"]


def predecir(paths: Dict, n_chains: int, thin: int = 4, por_iteracion: int = 1, seed: int = 0) -> Dict:
    d = cargar_punto_mv(paths)
    fr, L, df, ev = d["fr"], d["L"], d["df"], d["eval_config"]
    K, n_lags, T0, nivel = ev["K"], ev["n_lags"], ev["T0"], ev["nivel_credibilidad"]
    covs = d["manifest"]["cov_names"]
    Xp = df[covs].to_numpy(float); t_obj = df["t"].to_numpy(int)
    if d["manifest"].get("modelo") == "psbp_bloque":
        modelo = ModeloBloque.desde_paths(paths, d["manifest"], n_chains, thin=thin)
        media_w = modelo.media(df)
        S_w = modelo.muestrear(df, por_iteracion=por_iteracion, seed=seed)
        Xp_gating = df[modelo.subs[0]["cols"]].to_numpy(float)       # predictores del bloque (ocupacion / entropia)
    else:
        modelo = ModeloTrazaMV.desde_paths(paths, n_chains, thin=thin)
        media_w = modelo.media(Xp)                                   # (n, K) en theta_w
        S_w = modelo.muestrear(Xp, por_iteracion=por_iteracion, seed=seed)   # (S, n, K)
        Xp_gating = Xp
    X_pred = fr.reconstruct(desblanquear(media_w, L))            # (n, G)
    S, n, _ = S_w.shape
    a = (1 - nivel) / 2; G = len(d["grilla"])
    li, ls = np.empty((n, G)), np.empty((n, G))
    for i0 in range(0, n, 200):                                  # por bloques de origenes: (S, n, G) no cabe
        i1 = min(i0 + 200, n); m = i1 - i0
        cur = fr.reconstruct(desblanquear(S_w[:, i0:i1].reshape(S * m, K), L)).reshape(S, m, G)
        li[i0:i1], ls[i0:i1] = np.quantile(cur, a, axis=0), np.quantile(cur, 1 - a, axis=0)
    curvas = None
    X_suav = fr.reconstruct(d["THETA"])                          # objetivo: curva suavizada
    return {"d": d, "modelo": modelo, "Xp": Xp_gating, "t_obj": t_obj, "X_pred": X_pred, "li": li, "ls": ls,
            "curvas_muestras": curvas, "X_obj": X_suav[t_obj], "X_suav": X_suav, "T0": T0, "n_lags": n_lags,
            "nivel": nivel, "grilla": d["grilla"], "K": K, "S": S}


def guardar_banda(paths: Dict, P: Dict) -> Path:
    out = Path(paths["predict"]) / "banda_funcional_psbp_mv.npz"
    np.savez(out, li=P["li"], ls=P["ls"], X_pred=P["X_pred"], t_orig=P["t_obj"] + 1,
             nivel=np.array([P["nivel"]]), n_lags=np.array([P["n_lags"]]), T0=np.array([P["T0"]]),
             modo_residuo=np.array(["ninguno"]), objetivo=np.array(["curva_suavizada"]))
    return out


def tablas_ventana(paths: Dict, P: Dict, ventanas_w=(20, 30, 40), w_ref: int = 30) -> Dict[int, pd.DataFrame]:
    rep = Path(paths["out_report"]); tablas = {}
    for w in ventanas_w:
        t = ventana_movil_funcional(P["X_obj"], P["X_pred"], P["grilla"], P["T0"], w=w, t_offset=P["n_lags"],
                                    li=P["li"], ls=P["ls"], nivel=P["nivel"], bloque_A=True, verbose=False)
        t.attrs["w"] = w; tablas[w] = t
        t.to_csv(rep / f"54_ventana_movil_w{w}.csv", index=False)
    # relaciones que no deben romperse
    t = tablas[w_ref]
    assert (t["mae_f"] <= t["l2_medio"] + 1e-9).all() and (t["l2_medio"] <= t["linf_medio"] + 1e-9).all()
    assert (t["picp_simultaneo"] <= t["picp"] + 1e-9).all()
    assert (t["winkler"] <= t["winkler_max_medio"] + 1e-9).all() and (t["winkler_max_medio"] <= t["winkler_max_glob"] + 1e-9).all()
    fig = plot_ventana_movil(t, P["T0"], METRICAS_A + ["winkler", "picp", "mpiw"], tablas_por_w=tablas,
                             title="PSBPM-FD conjunto: ventana movil (curva suavizada)",
                             save_path=str(rep / "55_ventana_movil.png"))
    plt.close("all")
    return tablas


def resumen(paths: Dict, P: Dict, tablas: Dict[int, pd.DataFrame], w_ref: int = 30) -> pd.DataFrame:
    """Minimo / maximo / promedio por bloque de las diez metricas, ventanas que no cruzan T0,
    mas el agregado sobre todo el bloque de test (56_resumen.csv)."""
    t = tablas[w_ref]; t = t[~t["cruza_T0"]]
    filas = []
    for m in METRICAS_A + METRICAS_B:
        for b in ("train", "test"):
            s = t[t.bloque == b][m]
            filas.append(dict(metrica=m, bloque=b, minimo=s.min(), maximo=s.max(), promedio=s.mean()))
    res = pd.DataFrame(filas)
    te = P["t_obj"] >= P["T0"]
    agg = resumen_error_funcional(P["X_obj"][te], P["X_pred"][te], P["grilla"])
    ri = resumen_intervalo(P["X_obj"][te], P["li"][te], P["ls"][te], nivel=P["nivel"], tau=P["grilla"])
    agg.update(winkler=ri["winkler"], picp=ri["picp"], mpiw=ri["mpiw"],
               picp_simultaneo=float(indicador_cobertura_simultanea(P["X_obj"][te], P["li"][te], P["ls"][te]).mean()))
    rep = Path(paths["out_report"])
    res.to_csv(rep / "56_resumen_min_max_prom.csv", index=False)
    pd.Series(agg).to_csv(rep / "56b_agregado_test.csv")
    return res


def figuras(paths: Dict, P: Dict, n_curvas: int = 6) -> None:
    rep = Path(paths["out_report"]); g = P["grilla"]
    te = np.where(P["t_obj"] >= P["T0"])[0]
    idx = te[np.linspace(0, len(te) - 1, n_curvas).astype(int)]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7)); axes = axes.ravel()
    for a, i in zip(axes, idx):
        a.fill_between(g, P["li"][i], P["ls"][i], color="tab:orange", alpha=0.25, label="banda")
        a.plot(g, P["X_obj"][i], "k", lw=1.2, label="observada (suavizada)")
        a.plot(g, P["X_pred"][i], "tab:red", lw=1.2, label="PSBPM-FD conjunto")
        a.set_title(f"t = {P['t_obj'][i]}", fontsize=9)
    axes[0].legend(fontsize=7); fig.suptitle("Muestra de predicciones en test"); fig.tight_layout()
    fig.savefig(rep / "57_muestra_predicciones.png", dpi=130); plt.close(fig)
    # ocupacion de la mezcla y entropia
    gat = P["modelo"].bloque if hasattr(P["modelo"], "bloque") else P["modelo"]
    occ = gat.ocupacion(P["Xp"][P["t_obj"] < P["T0"]])
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.5))
    ax[0].bar(np.arange(1, len(occ) + 1), occ); ax[0].set_title("peso medio por atomo (train)"); ax[0].set_xlabel("atomo h")
    ent = gat.entropia_media(P["Xp"])
    ax[1].plot(P["t_obj"], ent, lw=0.6); ax[1].axvline(P["T0"], color="r", ls="--")
    ax[1].set_title("entropia de los pesos w_h(x_t)"); ax[1].set_xlabel("t")
    fig.tight_layout(); fig.savefig(rep / "58_ocupacion_mezcla.png", dpi=130); plt.close(fig)
    pd.DataFrame({"atomo": np.arange(1, len(occ) + 1), "peso_medio_train": occ}).to_csv(rep / "58_ocupacion_mezcla.csv", index=False)


def paso04(paths: Dict, n_chains: int, thin: int = 4, seed: int = 0) -> Dict:
    P = predecir(paths, n_chains, thin=thin, seed=seed)
    guardar_banda(paths, P)
    tablas = tablas_ventana(paths, P)
    res = resumen(paths, P, tablas)
    figuras(paths, P)
    P["tablas"] = tablas; P["resumen"] = res
    return P
