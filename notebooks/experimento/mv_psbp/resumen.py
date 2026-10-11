"""
Resumen transversal: una tabla por bloque con los cuatro casos (200, 201, 202,
real), leida de los reportes de cada punto. Es lo que mira primero quien abre
la carpeta (`00_RESUMEN.ipynb`).
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict

import pandas as pd

from .rutas import RAIZ_EXPERIMENTO

CASOS = {"200 TAR": ("simulaciones", "mv_escenario_200_chungEscG_1_r01_K10"),
         "201 GARCH": ("simulaciones", "mv_escenario_201_chungEscG_1_r01_K10"),
         "202 MULT": ("simulaciones", "mv_escenario_202_chungEscX_1_r01_K10"),
         "real eolica DE": ("reales", "mv_real_eolica_onshore_DE_v01_r01_K12"),
         "200 TAR bloque": ("simulaciones", "mvb_escenario_200_chungEscG_1_r01_K10"),
         "201 GARCH bloque": ("simulaciones", "mvb_escenario_201_chungEscG_1_r01_K10"),
         "202 MULT bloque": ("simulaciones", "mvb_escenario_202_chungEscX_1_r01_K10")}


def _rep(caso: str) -> Path:
    dom, eid = CASOS[caso]
    return RAIZ_EXPERIMENTO / "reports" / dom / eid


def bloque_A(bloque: str = "test") -> pd.DataFrame:
    filas = []
    for caso in CASOS:
        f = _rep(caso) / "96_razones_bloqueA.csv"
        if not f.exists():
            continue
        r = pd.read_csv(f); r = r[r.bloque == bloque]
        g = pd.read_csv(_rep(caso) / "96_ganador_ventana_bloqueA.csv"); g = g[(g.bloque == bloque) & (g.columna == "mae_f")]
        for m in r.modelo.unique():
            filas.append({"caso": caso, "modelo": m,
                          **{f"MAE/FAR": float(r[(r.modelo == m) & r.metrica.str.startswith("1.")].razon_vs_FAR.iloc[0]),
                             "RMSE/FAR": float(r[(r.modelo == m) & r.metrica.str.startswith("2.")].razon_vs_FAR.iloc[0]),
                             "Emax_prom/FAR": float(r[(r.modelo == m) & r.metrica.str.startswith("3.")].razon_vs_FAR.iloc[0]),
                             "pct_ventanas_MAE": float(g[g.modelo == m].pct_ventanas_ganadas.iloc[0])}})
    return pd.DataFrame(filas)


def bloque_B(bloque: str = "test") -> pd.DataFrame:
    filas = []
    for caso in CASOS:
        f = _rep(caso) / "99_resumen_bloqueB_min_max_prom.csv"
        if not f.exists():
            continue
        b = pd.read_csv(f); b = b[b.bloque == bloque]
        g = pd.read_csv(_rep(caso) / "97_ganador_ventana_bloqueB.csv"); g = g[(g.bloque == bloque) & (g.columna == "winkler")]
        for banda in b.banda.unique():
            s = b[b.banda == banda].set_index("columna").promedio
            filas.append({"caso": caso, "banda": banda, "winkler": s["winkler"], "picp": s["picp"],
                          "picp_simultaneo": s["picp_simultaneo"], "mpiw": s["mpiw"],
                          "pct_ventanas_winkler": float(g[g.banda == banda].pct_ventanas_ganadas.iloc[0])})
    return pd.DataFrame(filas)


def convergencia() -> pd.DataFrame:
    filas = []
    for caso in CASOS:
        f = _rep(caso) / "43_diagnosticos.csv"
        if not f.exists():
            continue
        c = pd.read_csv(f)
        if "submodelo" in c.columns:
            c = c[c.submodelo == "bloque"]
        c = c.set_index("variable")
        filas.append({"caso": caso, **{f"rhat_{v}": c.loc[v, "rhat"] for v in ("loglik", "mse_in", "N_activos", "entropia")},
                      "ess_min_mse_in": c.loc["mse_in", "ess_min"], "N_activos": c.loc["N_activos", "media"],
                      "entropia": c.loc["entropia", "media"],
                      "pi_j_no_converge": int((~c[c.index.str.startswith("pi[")].converge).sum()),
                      "pi_j_total": int(c.index.str.startswith("pi[").sum())})
    return pd.DataFrame(filas)


def info() -> pd.DataFrame:
    filas = []
    for caso in CASOS:
        f = _rep(caso) / "70_info_modelos.csv"
        if f.exists():
            s = pd.read_csv(f, index_col=0).iloc[:, 0]; filas.append({"caso": caso, **s.to_dict()})
    return pd.DataFrame(filas)
