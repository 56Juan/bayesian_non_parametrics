"""
Insumos de cada caso. Las simulaciones se LEEN de las corridas vivas
(data/simulaciones/raw/escenario_20X_*/escenario_1.npz), sin regenerar: asi
las curvas, la base y la particion son exactamente las de los reportes 200-202
y la comparacion con el univariado es punto a punto. El caso real entra como
npz (X, grilla, fechas) preparado fuera (eolica onshore DE, SMARD, CC BY 4.0).
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict

import numpy as np

from .rutas import RAIZ_PROYECTO

ESCENARIOS = {
    "200": {"basename": "escenario_200_chungEscG", "carpeta": "200_sim_TAR", "nombre": "TAR",
            "univariado_M": 10, "univariado_n_lags": 4},
    "201": {"basename": "escenario_201_chungEscG", "carpeta": "201_sim_GARCH", "nombre": "GARCH",
            "univariado_M": 10, "univariado_n_lags": 4},
    "202": {"basename": "escenario_202_chungEscX", "carpeta": "202_sim_MULT", "nombre": "MULT",
            "univariado_M": 6, "univariado_n_lags": 2},
}


def ruta_corrida_viva(esc: str, M: int, que: str = "raw") -> Path:
    e = ESCENARIOS[esc]; eid = f"{e['basename']}_1_r01_m{M:02d}"
    base = RAIZ_PROYECTO / "data" / "simulaciones"
    return {"raw": base / "raw" / eid, "predict": base / "processed" / "predict" / eid,
            "functional": base / "processed" / "functional" / eid,
            "artefact": RAIZ_PROYECTO / "artefact" / "simulaciones" / eid,
            "report": RAIZ_PROYECTO / "reports" / "simulaciones" / eid}[que]


def cargar_simulacion(esc: str) -> Dict:
    """Curvas verdaderas, grilla, base del generador y T0 de la corrida viva."""
    raw = ruta_corrida_viva(esc, 5, "raw")
    d = np.load(raw / "escenario_1.npz", allow_pickle=True)
    cfg = json.load(open(raw / "simulation_config.json"))
    ev = json.load(open(ruta_corrida_viva(esc, 5, "artefact") / "eval_config.json"))
    X = d["curvas"][0]
    return {"X": X, "grilla": d["grilla"], "Phi": d["interno_Phi"], "T0": int(ev["T0"]),
            "scores": d["interno_scores"][0], "oraculo": d["interno_media_condicional_curva"][0],
            "config": cfg, "eval_config_vivo": ev, "origen": str(raw)}


def cargar_univariado_vivo(esc: str) -> Dict:
    """Prediccion y banda del PSBPM-FD univariado (banda_funcional_psbp.npz del _04 de la
    corrida viva), en su M de referencia. t_orig es base-1 del objetivo."""
    e = ESCENARIOS[esc]
    npz = ruta_corrida_viva(esc, e["univariado_M"], "predict") / "banda_funcional_psbp.npz"
    b = np.load(npz, allow_pickle=True)
    return {"X_pred": b["X_pred"].astype(float), "li": b["li"].astype(float), "ls": b["ls"].astype(float),
            "t_obj": b["t_orig"].astype(int) - 1, "T0": int(b["T0"][0]), "n_lags": int(b["n_lags"][0]),
            "nivel": float(b["nivel"][0]), "M": e["univariado_M"], "origen": str(npz)}


def cargar_real_npz(ruta: Path) -> Dict:
    d = np.load(ruta, allow_pickle=True)
    return {"X": d["X"].astype(float), "grilla": d["grilla"].astype(float), "fechas": d["fechas"].astype(str)}
