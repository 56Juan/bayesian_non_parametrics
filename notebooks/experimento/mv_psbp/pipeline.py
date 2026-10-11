"""
Orquestacion de los pasos del experimento: lo que los notebooks E_01..E_05
llaman. Cada paso lee y escribe solo dentro de notebooks/experimento/.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from .rutas import RAIZ_EXPERIMENTO, construir_paths_mv, experiment_id_mv
from .datos import ESCENARIOS, cargar_simulacion, cargar_real_npz
from .preparacion import (representacion_conocida, representacion_bspline_gcv, preparar_punto_mv,
                          preparar_punto_bloque, escribir_jobs, cargar_punto_mv)

MCMC_CONFIG = {"nsim": 6000, "burn": 1500, "N": 30, "M": 50}
N_CHAINS = 3
SEED_BASE = 41232
N_LAGS = 2            # dos rezagos funcionales
TAUPSIJ_Z = 10.0      # gating libre (chungEscG / chungEscX)
JOBS_JSON = RAIZ_EXPERIMENTO / "artefact" / "jobs_mv.json"


def paso01_simulacion(esc: str, mcmc_config: Dict = MCMC_CONFIG, n_chains: int = N_CHAINS,
                      seed_base: int = SEED_BASE, n_lags: int = N_LAGS, limpiar: bool = True) -> Dict:
    """Datos de la corrida viva -> representacion conocida -> contrato conjunto."""
    sim = cargar_simulacion(esc)
    K = sim["Phi"].shape[1]
    eid = experiment_id_mv(f"mv_{ESCENARIOS[esc]['basename']}", 1, 1, K)
    paths = construir_paths_mv(eid, "simulaciones", limpiar=limpiar)
    np.savez(paths["raw"] / "insumo.npz", X=sim["X"], grilla=sim["grilla"], Phi=sim["Phi"],
             scores=sim["scores"], oraculo=sim["oraculo"], origen=sim["origen"])
    fr = representacion_conocida(sim["Phi"], sim["grilla"])
    out = preparar_punto_mv(paths, sim["X"], sim["grilla"], fr, sim["T0"], n_lags, mcmc_config, n_chains,
                            seed_base, meta={"origen": sim["origen"], "escenario": esc,
                                             "base": "fourier_generador_ortonormal"}, taupsij_z=TAUPSIJ_Z)
    out.update({"paths": paths, "eid": eid, "sim": sim, "fr": fr})
    return out


def paso01_real(nombre: str, npz: Path, nb_max: int = 12, prop_train: float = 0.70,
                mcmc_config: Dict = MCMC_CONFIG, n_chains: int = N_CHAINS, seed_base: int = SEED_BASE,
                n_lags: int = N_LAGS, limpiar: bool = True, meta: Optional[Dict] = None) -> Dict:
    """Curvas reales -> B-spline por GCV (tope nb_max) -> blanqueo -> contrato conjunto."""
    d = cargar_real_npz(npz)
    X, grilla = d["X"], d["grilla"]
    T = len(X); T0 = int(round(prop_train * T))
    fr, df_gcv = representacion_bspline_gcv(X[:T0], grilla, nb_max=nb_max)
    K = fr.K_
    eid = experiment_id_mv(f"mv_real_{nombre}", "v01", 1, K)
    paths = construir_paths_mv(eid, "reales", limpiar=limpiar)
    np.savez(paths["raw"] / "insumo.npz", X=X, grilla=grilla, fechas=d["fechas"])
    df_gcv.to_csv(paths["out_report"] / "05_seleccion_basis.csv", index=False)
    out = preparar_punto_mv(paths, X, grilla, fr, T0, n_lags, mcmc_config, n_chains, seed_base,
                            meta={**(meta or {}), "n_basis": int(fr.n_basis), "order": int(fr.order),
                                  "nb_max": nb_max, "fechas": [str(d["fechas"][0]), str(d["fechas"][-1])]},
                            taupsij_z=TAUPSIJ_Z)
    out.update({"paths": paths, "eid": eid, "fr": fr, "df_gcv": df_gcv, "fechas": d["fechas"]})
    return out


def registrar_jobs(lista_paths, ruta: Path = JOBS_JSON) -> Path:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    escribir_jobs(ruta, lista_paths)
    return ruta


def paths_de(eid: str, dominio: str) -> Dict:
    return construir_paths_mv(eid, dominio, limpiar=False)


def paso01_bloque(esc: str, bloque=(0, 1, 2, 3), mcmc_config: Dict = MCMC_CONFIG, n_chains: int = N_CHAINS,
                  seed_base: int = SEED_BASE, n_lags: int = N_LAGS, limpiar: bool = True) -> Dict:
    """Como paso01_simulacion pero con bloque conjunto sobre `bloque` + univariados."""
    sim = cargar_simulacion(esc)
    K = sim["Phi"].shape[1]
    eid = experiment_id_mv(f"mvb_{ESCENARIOS[esc]['basename']}", 1, 1, K)
    paths = construir_paths_mv(eid, "simulaciones", limpiar=limpiar)
    np.savez(paths["raw"] / "insumo.npz", X=sim["X"], grilla=sim["grilla"], Phi=sim["Phi"],
             scores=sim["scores"], oraculo=sim["oraculo"], origen=sim["origen"])
    fr = representacion_conocida(sim["Phi"], sim["grilla"])
    out = preparar_punto_bloque(paths, sim["X"], sim["grilla"], fr, sim["T0"], n_lags, mcmc_config, n_chains,
                                seed_base, bloque=bloque, taupsij_z=TAUPSIJ_Z,
                                meta={"origen": sim["origen"], "escenario": esc, "base": "fourier_generador_ortonormal"})
    out.update({"paths": paths, "eid": eid, "sim": sim, "fr": fr})
    return out
