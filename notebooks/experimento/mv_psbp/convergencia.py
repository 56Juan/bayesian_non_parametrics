"""
Paso 03: convergencia de las cadenas conjuntas sobre cantidades invariantes a
la permutacion de etiquetas: log-verosimilitud, error in-sample de E[y|x],
numero de atomos activos, entropia de los pesos, mu, g y las probabilidades de
inclusion pi_j. R-hat, ESS y Geweke con `fit.diagnostics_mcmc`. No toca el
bloque de prueba.
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model_psbp_fd.fit.diagnostics_mcmc import gelman_rubin, ess_geyer, geweke_z

import json
from .trazas import leer_traza_mv, ruta_traza_mv, ruta_traza_sub


def _series(t: Dict) -> Dict[str, np.ndarray]:
    s = {"loglik": t["loglikout"].ravel(), "mse_in": t["mse_inout"].ravel(), "N_activos": t["N1out"].ravel(),
         "entropia": t["entropout"].ravel(), "mu": t["muout"].ravel(), "g": t["gout"].ravel()}
    for j, nombre in enumerate(t["feature_names"]):
        s[f"pi[{nombre}]"] = t["pijout"][:, j].astype(float)
    return s


def tabla_diagnosticos(trazas: List[Dict]) -> pd.DataFrame:
    burn = trazas[0]["burn"]
    series = [_series(t) for t in trazas]
    filas = []
    for k in series[0]:
        mat = np.vstack([s[k][burn:] for s in series])
        if np.allclose(mat, mat[0, 0]):
            filas.append(dict(variable=k, rhat=np.nan, ess_min=np.nan, geweke_max=np.nan, media=float(mat.mean()), sd=0.0))
            continue
        rhat = gelman_rubin(mat) if len(trazas) > 1 else np.nan
        ess = min(ess_geyer(r) for r in mat)
        gw = max(abs(geweke_z(r)) for r in mat)
        filas.append(dict(variable=k, rhat=rhat, ess_min=ess, geweke_max=gw, media=float(mat.mean()), sd=float(mat.std())))
    df = pd.DataFrame(filas)
    df["converge"] = (df.rhat.fillna(1) < 1.1) & (df.ess_min.fillna(1e9) > 100) & (df.geweke_max.fillna(0) < 2)
    return df


def pip(trazas: List[Dict]) -> pd.DataFrame:
    """PIP por predictor: media post-burn de pi_j (prior-posterior de inclusion) y
    fraccion de atomos ocupados que incluyen el predictor."""
    burn = trazas[0]["burn"]; nombres = trazas[0]["feature_names"]
    pi = np.mean([t["pijout"][burn:].mean(0) for t in trazas], axis=0)
    inc = []
    for t in trazas:
        S = t["Sout"][burn:]; gam = t["gammajhout"][burn:]
        occ = np.array([[np.any(S[i] == h + 1) for h in range(gam.shape[1])] for i in range(len(S))])
        inc.append(np.array([(gam[:, :, j] * occ).sum() / occ.sum() for j in range(gam.shape[2])]))
    return pd.DataFrame({"predictor": nombres, "pi_j": pi, "frac_atomos_ocupados_con_j": np.mean(inc, axis=0)})


def figuras(trazas: List[Dict], rep: Path) -> None:
    burn = trazas[0]["burn"]
    claves = ["loglik", "mse_in", "N_activos", "entropia", "mu", "g"]
    fig, axes = plt.subplots(len(claves), 1, figsize=(12, 2.2 * len(claves)), sharex=True)
    for a, k in zip(axes, claves):
        for c, t in enumerate(trazas):
            a.plot(_series(t)[k], lw=0.6, label=f"cadena {c + 1}")
        a.axvline(burn, color="k", ls=":"); a.set_ylabel(k, fontsize=8)
    axes[0].legend(fontsize=7, ncol=3); axes[-1].set_xlabel("iteracion")
    fig.suptitle("Trazas de cantidades invariantes a la permutacion"); fig.tight_layout()
    fig.savefig(rep / "40_trazas.png", dpi=120); plt.close(fig)
    # ocupacion: cuantos atomos tienen al menos una observacion por iteracion
    fig, ax = plt.subplots(figsize=(10, 3))
    for c, t in enumerate(trazas):
        S = t["Sout"]; ax.plot([len(np.unique(S[i])) for i in range(len(S))], lw=0.6, label=f"cadena {c + 1}")
    ax.set_title("atomos ocupados por iteracion"); ax.legend(fontsize=7); fig.tight_layout()
    fig.savefig(rep / "44_ocupacion.png", dpi=120); plt.close(fig)


def paso03(paths: Dict, n_chains: int) -> Dict:
    rep = Path(paths["out_report"])
    man = json.load(open(Path(paths["functional"]) / "datasets_manifest.json"))
    if man.get("modelo") == "psbp_bloque":
        # el bloque conjunto lleva las figuras de siempre; los univariados van a sub_<k>/
        diags, pips, trazas = [], [], None
        for sm in man["submodelos"]:
            tr = [leer_traza_mv(ruta_traza_sub(paths, sm["out_prefix"], c)) for c in range(1, n_chains + 1)]
            dg = tabla_diagnosticos(tr); dg.insert(0, "submodelo", sm["nombre"]); diags.append(dg)
            pq = pip(tr); pq.insert(0, "submodelo", sm["nombre"]); pips.append(pq)
            if sm["nombre"] == "bloque":
                trazas = tr; figuras(tr, rep)
            else:
                sub = rep / f"sub_{sm['nombre']}"; sub.mkdir(exist_ok=True); figuras(tr, sub)
        diag = pd.concat(diags, ignore_index=True); pp = pd.concat(pips, ignore_index=True)
        diag.to_csv(rep / "43_diagnosticos.csv", index=False); pp.to_csv(rep / "46_pip.csv", index=False)
    else:
        trazas = [leer_traza_mv(ruta_traza_mv(paths, c)) for c in range(1, n_chains + 1)]
        diag = tabla_diagnosticos(trazas); diag.to_csv(rep / "43_diagnosticos.csv", index=False)
        pp = pip(trazas); pp.to_csv(rep / "46_pip.csv", index=False)
        figuras(trazas, rep)
    gd = np.vstack([t["gamdiagout"] for t in trazas])
    extra = {"gamma_rango_loglik_medio": float(gd[:, 3].mean()), "gamma_no_finito": int(gd[:, 1].sum()),
             "min_por_iter": float(np.mean([t["mse_inout"].ravel()[-1] for t in trazas]))}
    return {"trazas": trazas, "diagnosticos": diag, "pip": pp, "extra": extra}
