"""
Paso 05: PSBPM-FD conjunto contra el FAR(p) y, donde existe, el PSBPM-FD
univariado de la corrida viva (su banda_funcional_psbp.npz, leida de fuera sin
modificarla). Mismas tablas que el _05: ventana por modelo (72), razones y
ganadores del Bloque A (96), Bloque B (97, 99), resumen (98) y figuras 76/77.

El FAR se ajusta EXACTAMENTE como en fit/comparacion_barrido.ajustar_far: sobre
theta blanqueada, pesos="conteo", p = n_lags, kn por hold-out (far.cv, L2) con
tope kn_max; se evalua con los K coeficientes. Con N = 1 atomo el conjunto es
ese mismo VAR(p) con kn = K: la comparacion es entre un modelo y su caso
particular lineal.
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model_psbp_fd.fit.far_operador import FARp, seleccionar_kn
from model_psbp_fd.fit.rolling import ventana_movil_funcional
from model_psbp_fd.fit.intervalos import residuos_para_banda, banda_predictiva_modelo
from model_psbp_fd.graphics.viz_comparacion import plot_ganador_modelo

from .preparacion import cargar_punto_mv, desblanquear
from .evaluacion import METRICAS_A, METRICAS_B

ETIQ_A = [("1. MAE  (L^1)", "mae_f"), ("2. RMSE (L^2)", "rmse_f"),
          ("3. E_max promedio por curva", "linf_medio"), ("4. E_max peor de la ventana", "linf_max")]
ETIQ_B = [("5. MPIW", "mpiw"), ("6. PICP puntual", "picp"), ("7. PICPB curva completa", "picp_simultaneo"),
          ("8. Winkler", "winkler"), ("9. Winkler max promedio", "winkler_max_medio"), ("10. Winkler max peor", "winkler_max_glob")]
MV = "PSBPM-FD conjunto"; UNI = "PSBPM-FD univariado"; FAR = "FAR"


def ajustar_far(d: Dict, kn_max: int = 12, criterio: str = "L2") -> Dict:
    TW, L, fr = d["TW"], d["L"], d["fr"]; ev = d["eval_config"]; T0, p = ev["T0"], ev["n_lags"]
    cv = seleccionar_kn(TW[:T0], kn_max=min(kn_max, TW.shape[1]), pesos="conteo", p=p)
    kn = {"L1": cv.kn_L1, "L2": cv.kn_L2, "Linf": cv.kn_Linf}[criterio]
    far = FARp(p=p, kn=kn, pesos="conteo").fit(TW[:T0])
    pred_w = far.predict_serie(TW)                       # (T-p, K), alinea con t >= p
    X_far = fr.reconstruct(desblanquear(pred_w, L))
    return {"X_pred": X_far, "kn": int(kn), "cv": cv, "diag": far.diagnostico_kn()}


def alinear_univariado(uni: Optional[Dict], t_obj: np.ndarray) -> Optional[Dict]:
    """La corrida viva puede tener otro n_lags: se cruza por el indice del objetivo."""
    if uni is None:
        return None
    pos = {t: i for i, t in enumerate(uni["t_obj"])}
    comunes = np.array([t in pos for t in t_obj])
    idx = np.array([pos[t] for t in t_obj[comunes]])
    return {"X_pred": uni["X_pred"][idx], "li": uni["li"][idx], "ls": uni["ls"][idx], "mask": comunes, "M": uni["M"]}


def cargar_banda_experimento(paths_otro: Dict) -> Dict:
    """Prediccion y banda de OTRO punto de este experimento (p. ej. el conjunto completo)."""
    b = np.load(Path(paths_otro["predict"]) / "banda_funcional_psbp_mv.npz", allow_pickle=True)
    return {"X_pred": b["X_pred"].astype(float), "li": b["li"].astype(float), "ls": b["ls"].astype(float),
            "t_obj": b["t_orig"].astype(int) - 1, "M": "K", "T0": int(b["T0"][0])}


def paso05(paths: Dict, uni: Optional[Dict] = None, kn_max: int = 12, ventanas_w=(20, 30, 40), w_ref: int = 30,
           extras: Optional[Dict[str, Dict]] = None, etiqueta_mv: str = MV) -> Dict:
    d = cargar_punto_mv(paths); ev = d["eval_config"]; T0, p, nivel, g = ev["T0"], ev["n_lags"], ev["nivel_credibilidad"], d["grilla"]
    rep = Path(paths["out_report"])
    b = np.load(Path(paths["predict"]) / "banda_funcional_psbp_mv.npz", allow_pickle=True)
    t_obj = b["t_orig"].astype(int) - 1
    X_suav = d["fr"].reconstruct(d["THETA"]); X_obj = X_suav[t_obj]
    es_train = t_obj < T0
    far = ajustar_far(d, kn_max=kn_max)
    assert far["X_pred"].shape == X_obj.shape, (far["X_pred"].shape, X_obj.shape)
    modelos = {etiqueta_mv: b["X_pred"].astype(float), FAR: far["X_pred"]}
    r_far = residuos_para_banda(X_obj, far["X_pred"], es_train)
    bandas = {f"{FAR} (IC gaussiano)": banda_predictiva_modelo(far["X_pred"], r_far, nivel=nivel, por_tau=True),
              f"{etiqueta_mv} (predictiva nativa)": (b["li"].astype(float), b["ls"].astype(float))}
    mask_u = np.ones(len(t_obj), bool)
    otros = dict(extras or {})
    if uni is not None:
        otros[UNI] = uni
    for nombre, o in otros.items():
        u = alinear_univariado(o, t_obj); mask_u &= u["mask"]
        X_u = np.full_like(X_obj, np.nan); X_u[u["mask"]] = u["X_pred"]; modelos[nombre] = X_u
        li_u = np.full_like(X_obj, np.nan); ls_u = np.full_like(X_obj, np.nan)
        li_u[u["mask"]], ls_u[u["mask"]] = u["li"], u["ls"]
        bandas[f"{nombre} (predictiva nativa)"] = (li_u, ls_u)
    u = None if uni is None else alinear_univariado(uni, t_obj)
    # ---- ventana movil por modelo y por banda; los origenes sin univariado se recortan para TODOS
    sel = mask_u; Xo = X_obj[sel]; t_sel = t_obj[sel]; off = int(t_sel[0])
    tablas_A, tablas_B = {}, {}
    for w in ventanas_w:
        partes = []
        for nombre, Xp in modelos.items():
            t = ventana_movil_funcional(Xo, Xp[sel], g, T0, w=w, t_offset=off, bloque_A=True, verbose=False)
            t["modelo"] = nombre; partes.append(t)
        tablas_A[w] = pd.concat(partes, ignore_index=True); tablas_A[w].to_csv(rep / f"72_ventana_modelos_w{w}.csv", index=False)
        partes = []
        for nombre, (li, ls) in bandas.items():
            t = ventana_movil_funcional(Xo, Xo, g, T0, w=w, t_offset=off, li=li[sel], ls=ls[sel], nivel=nivel, bloque_A=True, verbose=False)
            t["banda"] = nombre; partes.append(t)
        tablas_B[w] = pd.concat(partes, ignore_index=True); tablas_B[w].to_csv(rep / f"73_ventana_bandas_w{w}.csv", index=False)
    # ---- tablas 96-99
    gA, rA = _ganadores(tablas_A[w_ref], "modelo", ETIQ_A, nivel); gA.to_csv(rep / "96_ganador_ventana_bloqueA.csv", index=False); rA.to_csv(rep / "96_razones_bloqueA.csv", index=False)
    gB, rB = _ganadores(tablas_B[w_ref], "banda", ETIQ_B, nivel); gB.to_csv(rep / "97_ganador_ventana_bloqueB.csv", index=False); rB.to_csv(rep / "97_razones_bloqueB.csv", index=False)
    _resumen(tablas_A[w_ref], "modelo", ETIQ_A).to_csv(rep / "98_resumen_bloqueA_min_max_prom.csv", index=False)
    _resumen(tablas_B[w_ref], "banda", ETIQ_B).to_csv(rep / "99_resumen_bloqueB_min_max_prom.csv", index=False)
    # ---- figuras 76 / 77
    gA["M"] = d["eval_config"]["K"]; gB["M"] = d["eval_config"]["K"]
    plot_ganador_modelo(gA, d["eval_config"]["K"], ETIQ_A, w_ref, save_path=str(rep / "76_ganador_bloqueA.png")); plt.close("all")
    gB2 = gB.rename(columns={"banda": "modelo"})
    plot_ganador_modelo(gB2, d["eval_config"]["K"], ETIQ_B, w_ref, save_path=str(rep / "77_ganador_bloqueB.png")); plt.close("all")
    _fig_series(tablas_A[w_ref], tablas_B[w_ref], T0, rep)
    info = {"far_kn": far["kn"], "far_p": p, "far_radio_espectral": float(far["diag"]["radio_espectral"]),
            "n_origenes": int(sel.sum()), "univariado": None if u is None else f"M={u['M']} de la corrida viva"}
    pd.Series(info).to_csv(rep / "70_info_modelos.csv")
    return {"tablas_A": tablas_A, "tablas_B": tablas_B, "ganadores_A": gA, "razones_A": rA, "ganadores_B": gB,
            "razones_B": rB, "info": info, "modelos": modelos, "bandas": bandas, "X_obj": X_obj, "t_obj": t_obj}


def _ganadores(t: pd.DataFrame, col: str, metricas, nivel: float):
    limpio = t[~t["cruza_T0"]]; grupos = list(limpio[col].unique())
    gana, rel = [], []
    for etq, c in metricas:
        piv = limpio.pivot_table(index=["t_centro", "bloque"], columns=col, values=c)[grupos]
        obj = (piv - nivel).abs() if c.startswith("picp") else piv
        ganador = obj.idxmin(axis=1)
        for bl in ("train", "test"):
            s = ganador.index.get_level_values("bloque") == bl
            sub, o = ganador[s], obj[s]; mejor = o.min(axis=1).replace(0, np.nan)
            for gname in grupos:
                gana.append({"metrica": etq, "columna": c, "bloque": bl, col: gname, "n_ventanas": len(sub),
                             "pct_ventanas_ganadas": float((sub == gname).mean()) if len(sub) else np.nan,
                             "criterio": "|valor - nominal|" if c.startswith("picp") else "menor es mejor"})
                fila = {"metrica": etq, "bloque": bl, col: gname, "razon_vs_mejor": float((o[gname] / mejor).mean())}
                ref = [x for x in o.columns if x.startswith(FAR)]
                if ref:
                    fila["razon_vs_FAR"] = float((o[gname] / o[ref[0]].replace(0, np.nan)).mean())
                rel.append(fila)
    return pd.DataFrame(gana), pd.DataFrame(rel)


def _resumen(t: pd.DataFrame, col: str, metricas) -> pd.DataFrame:
    limpio = t[~t["cruza_T0"]]; filas = []
    for etq, c in metricas:
        for (gname, bl), r in limpio.groupby([col, "bloque"])[c].agg(["min", "max", "mean"]).iterrows():
            filas.append({"metrica": etq, "columna": c, col: gname, "bloque": bl, "minimo": r["min"], "maximo": r["max"], "promedio": r["mean"]})
    return pd.DataFrame(filas)


def _fig_series(tA: pd.DataFrame, tB: pd.DataFrame, T0: int, rep: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    for a, c in zip(axes, ("mae_f", "linf_medio")):
        for m, s in tA.groupby("modelo"):
            a.plot(s.t_centro, s[c], lw=0.9, label=m)
        a.axvline(T0, color="k", ls="--"); a.set_ylabel(c)
    for m, s in tB.groupby("banda"):
        axes[2].plot(s.t_centro, s["winkler"], lw=0.9, label=m)
    axes[2].axvline(T0, color="k", ls="--"); axes[2].set_ylabel("winkler"); axes[2].set_xlabel("t (centro de la ventana)")
    for a in axes: a.legend(fontsize=7)
    fig.suptitle("Ventana movil w=30: modelos (Bloque A) y bandas (Bloque B)"); fig.tight_layout()
    fig.savefig(rep / "75_ventana_modelos_bandas.png", dpi=120); plt.close(fig)
