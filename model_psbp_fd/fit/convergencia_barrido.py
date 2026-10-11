"""
convergencia_barrido.py
=======================
Analisis de convergencia de las cadenas MCMC sobre el barrido en M, SIN
graficos (las figuras viven en `graphics/viz_convergencia.py`).

Es la logica que antes estaba repartida en el notebook `_03`: carga de
artefactos y trazas, diagnosticos por punto, ocupacion de la mezcla, gating,
contraste de PIP con el generador, verificacion muestreador <-> predictor, paso
de Gamma y las tablas que cruzan los M. Cada funcion recibe el estado `EST[M]`
que arma `cargar_artefactos` / `cargar_trazas` y escribe los CSV con los mismos
nombres de siempre (40-49, 80-83, 85); el notebook solo orquesta y dibuja.

`EST[M]` es el estado completo de un punto del barrido; todo lo demas lo consume
por indice, nunca por variable global.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.io import loadmat

from ..models.pspb_fd_v3 import ModeloTraza, leer_traza, ruta_traza, pesos_probit
from ..pipelines import (
    cargar_datasets_ar, cargar_hiperparametros, cargar_config_evaluacion,
    cargar_escenario, cargar_fpca, cargar_curvas, verificar_contrato,
)
from ..utils.quadrature import pesos_trapezoidales
from ..utils.rutas import rutas_por_M, experiment_id
from .diagnostics_mcmc import (
    tabla_diagnosticos, resumen_convergencia, diagnostico_variable,
)
from .inclusion import matriz_pip, contraste_con_verdad

__all__ = [
    "cargar_artefactos", "cargar_trazas",
    "diagnosticar_punto", "imprimir_convergencia",
    "ocupacion_mezcla", "diagnostico_gating", "pip_punto", "contraste_generador",
    "verificar_muestreador_predictor", "diagnostico_paso_gamma",
    "tabla_barrido", "ocupacion_barrido", "pip_barrido", "veredicto_barrido",
    "comparar_variantes",
]


# ==========================================================================
# 1. ARTEFACTOS Y 2. TRAZAS
# ==========================================================================

_TRAZAS_EXTRA = ("muout", "N1out", "Nout")   # que `leer_traza` no carga


def _leer_traza_completa(ruta):
    """`leer_traza` mas `N1out` (atomos ocupados), `Nout` y `muout`."""
    traces, burn, feat = leer_traza(ruta)
    crudo = loadmat(str(ruta), variable_names=list(_TRAZAS_EXTRA))
    for k in _TRAZAS_EXTRA:
        if k in crudo:
            traces[k] = np.asarray(crudo[k], dtype=float).ravel()
    return traces, burn, feat


def cargar_artefactos(basename: str, escenario_id, replica_id,
                      M_list: Sequence[int], project_root: Path,
                      verbose: bool = True, con_test: bool = False,
                      dominio: str = "simulaciones"):
    """
    Carga, por cada M del barrido, los artefactos que dejo `_01` y cruza el
    contrato (`verificar_contrato`: manifest, hiperparametros y FPCA).

    Retorna `(EST, saltados)`: `EST[M]` con paths, datasets de train (y de test si
    `con_test`), manifest, hiperparametros, config de evaluacion y los parametros
    de evaluacion que LEE del artefacto (T, T0, nivel, modo_residuo, objetivo,
    ventanas); `saltados` lista `(M, motivo)` de los puntos sin datos. El ID
    declara M y el manifest lo determina: si discrepan, el directorio esta mal
    nombrado y se falla (la comparacion entre M mezclaria el efecto de M con el
    de otra cosa).

    `dominio="reales"` lee de `data/reales/...`; con datos reales `escenario_id` es la
    ventana ya formateada (``"v01"``) y `replica_id` es None (ver `experiment_id`).
    """
    paths = rutas_por_M(basename, escenario_id, replica_id, M_list,
                        dominio=dominio, project_root=project_root)
    EST, saltados = {}, []
    for M, P in paths.items():
        eid = experiment_id(basename, escenario_id, replica_id, M)
        if not (P["functional"] / "datasets_manifest.json").exists():
            saltados.append((M, "sin datasets_manifest.json - falta correr _01"))
            continue
        verificar_contrato(P)
        dfs_train, manifest = cargar_datasets_ar(P, bloque="train")
        hp_json = cargar_hiperparametros(P)
        P["trazas"] = P["out_artefact"].parent / hp_json.get("trazas_en", P["out_artefact"].name)
        n_comp = len(manifest["component_idx"])
        assert n_comp == M, (
            f"{eid} declara M={M} pero su manifest tiene {n_comp} componentes. "
            "El directorio no corresponde a este punto del barrido.")
        assert len(hp_json["hyperparams_list"]) == n_comp, \
            f"[M={M}] n_components discrepa entre manifest e hyperparameters.json."
        EST[M] = {
            "paths": P, "eid": eid, "dfs_train": dfs_train, "manifest": manifest,
            "hp_json": hp_json, "eval_config": cargar_config_evaluacion(P),
            "component_idx": manifest["component_idx"], "n_components": n_comp,
            "cov_names": manifest["cov_names"],
            "cov_por_componente": [list(manifest["cov_por_componente"][str(k)])
                                   for k in range(n_comp)],
            "n_lags": int(manifest["n_lags"]), "n_iter": int(hp_json["n_iter"]),
            "mcmc_cfg": hp_json["mcmc_config"],
            "burn": int(hp_json["mcmc_config"]["burn"]),
        }
        ev = EST[M]["eval_config"]
        EST[M].update({
            "T": int(manifest["T"]), "T0": int(manifest["T0"]),
            "nivel": float(ev.get("nivel_credibilidad", 0.95)),
            "modo_residuo": ev.get("modo_residuo", "ninguno"),
            "objetivo": ev.get("objetivo_evaluacion", "curva_suavizada"),
            "ventanas_w": list(ev.get("ventana_movil", {}).get("w", [20, 30, 40])),
            "n_post": hp_json["mcmc_config"]["nsim"] - int(hp_json["mcmc_config"]["burn"]),
        })
        if con_test:
            EST[M]["dfs_test"] = cargar_datasets_ar(P, bloque="test")[0]
    assert EST, "Ningun punto del barrido tiene datos. Ejecuta _01 primero."

    if verbose:
        print(f"{'M':>3}  {'comp':>4}  {'cadenas':>7}  {'covariables':>11}  seed_base")
        for M, e in EST.items():
            print(f"{M:>3}  {e['n_components']:>4}  {e['n_iter']:>7}  "
                  f"{len(e['cov_names']):>11}  {e['hp_json']['seed_base']}")
        for M, motivo in saltados:
            print(f"  ! M={M} saltado: {motivo}")
        cfgs = {M: (e["mcmc_cfg"]["nsim"], e["mcmc_cfg"]["N"], e["burn"], e["n_iter"])
                for M, e in EST.items()}
        if len(set(cfgs.values())) > 1:
            print("\n! La configuracion MCMC NO es la misma en todos los M "
                  "(nsim/N/burn/cadenas):")
            for M, c in cfgs.items():
                print(f"    M={M}: {c}")
        else:
            print(f"\nOK misma configuracion MCMC en todos los M: "
                  f"nsim/N/burn/cadenas = {next(iter(cfgs.values()))}")
    return EST, saltados


def cargar_trazas(EST: Dict, saltar_sin_trazas: bool = True,
                  burn_eval: Optional[int] = None, verbose: bool = True):
    """
    Lee las trazas de cada (componente, cadena) y arma `e["models_chains"]`.

    Modifica `EST` (quita los M sin trazas si `saltar_sin_trazas`). Verifica que
    `feature_names` del .mat coincida con las columnas del dataset y que
    (nsim, N, burn) coincidan con `hyperparameters.json`. Retorna `M_OK`.
    """
    sin_trazas = []
    for M in list(EST):
        e, P = EST[M], EST[M]["paths"]
        ci, n_comp, n_iter = e["component_idx"], e["n_components"], e["n_iter"]
        faltan = [ruta_traza(P, ci[k] + 1, c + 1).name for k in range(n_comp)
                  for c in range(n_iter) if not ruta_traza(P, ci[k] + 1, c + 1).exists()]
        if faltan:
            msg = (f"faltan {len(faltan)} trazas en {P['out_artefact']}: "
                   f"{faltan[:3]}{'...' if len(faltan) > 3 else ''}")
            if not saltar_sin_trazas:
                raise AssertionError(f"[M={M}] {msg}\nEjecuta psbp_fd_iteracion.m.")
            sin_trazas.append((M, msg))
            del EST[M]
            continue

        models_chains, meta = {k: {} for k in range(n_comp)}, set()
        for k in range(n_comp):
            esperado = list(e["dfs_train"][k].columns[1:])
            for c in range(n_iter):
                traces, burn, feat = _leer_traza_completa(ruta_traza(P, ci[k] + 1, c + 1))
                assert feat == esperado, (
                    f"[M={M} k={k} chain={c+1}] feature_names del .mat != columnas "
                    f"del dataset:\n  mat  = {feat}\n  train= {esperado}")
                models_chains[k][c] = ModeloTraza(
                    traces, burn if burn_eval is None else burn_eval, feat)
                meta.add((traces["betajhout"].shape[0], traces["beta0hout"].shape[1], burn))
        assert len(meta) == 1, f"[M={M}] configuraciones MCMC heterogeneas: {meta}."
        nsim, n_atomos, burn = next(iter(meta))
        cfg = e["mcmc_cfg"]
        assert (nsim, n_atomos, burn) == (cfg["nsim"], cfg["N"], cfg["burn"]), (
            f"[M={M}] trazas (nsim={nsim}, N={n_atomos}, burn={burn}) != "
            f"hyperparameters.json ({cfg}).")
        burn_ev = burn if burn_eval is None else burn_eval
        assert burn_ev < nsim, f"burn_eval={burn_ev} >= nsim={nsim}."
        e.update({"models_chains": models_chains, "n_atomos": n_atomos,
                  "burn": burn_ev, "n_post": nsim - burn_ev})

    assert EST, ("Ningun punto del barrido tiene trazas. Ejecuta "
                 "psbp_fd_iteracion.m con M_FPCA_LIST antes de este notebook.")
    if verbose:
        for M, motivo in sin_trazas:
            print(f"! M={M} saltado: {motivo}")
        for M, e in EST.items():
            print(f"OK M={M}: {e['n_components']} componente(s) x {e['n_iter']} cadena(s) "
                  f"· {e['n_post']} draws posteriores c/u -> {e['n_post'] * e['n_iter']} "
                  "por componente")
    return tuple(sorted(EST))


# ==========================================================================
# 3. DIAGNOSTICOS
# ==========================================================================

def diagnosticar_punto(e: Dict) -> pd.DataFrame:
    """ESS, Geweke y R-hat de un punto (`40_diagnosticos_mcmc.csv`) + resumen."""
    diag = tabla_diagnosticos(e["models_chains"], e["burn"],
                              component_idx=e["component_idx"],
                              claves=("betajhout", "pijout"), verbose=True)
    diag.to_csv(e["paths"]["out_report"] / "40_diagnosticos_mcmc.csv", index=False)
    e["diag_df"] = diag
    e["res_conv"] = resumen_convergencia(diag)
    return diag


def imprimir_convergencia(EST: Dict, M_OK: Sequence[int]) -> None:
    """Veredicto por punto. Se reporta el PEOR caso, no el promedio: una sola
    variable sin converger invalida la posterior conjunta."""
    for M in M_OK:
        e, res = EST[M], EST[M]["res_conv"]
        print("=" * 62)
        print(f"  M = {M}   ({e['eid']})")
        print(f"  Rhat max {res['rhat_max']:.4f} (umbral {res['umbrales']['rhat']})  ·  "
              f"ESS min {res['ess_min']:.1f} (umbral {res['umbrales']['ess']})  ·  "
              f"|Geweke| max {res['geweke_max']:.2f} (umbral {res['umbrales']['geweke']})")
        print(f"  variables sin converger: {res['n_no_converge']} de {res['n_variables']}   "
              f"VEREDICTO: {'OK todas convergen' if res['todo_converge'] else 'X REVISAR'}")
        for v in res["variables_malas"]:
            d = e["diag_df"]
            f = d[(d.componente == v["componente"]) & (d.param == v["param"])
                  & (d.variable == v["variable"])].iloc[0]
            print(f"    · FPC {v['componente']} {v['param']} {v['variable']}: "
                  f"Rhat={f['rhat']:.3f} ESS={f['ess_min']:.0f} G={f['geweke_max']:+.2f}")
    print("=" * 62)
    malos = [M for M in M_OK if not EST[M]["res_conv"]["todo_converge"]]
    print(f"\nX NO convergen los puntos M = {malos}." if malos
          else "\nOK todos los puntos del barrido convergen.")


# ==========================================================================
# 4. OCUPACION DE LA MEZCLA Y GATING
# ==========================================================================

def ocupacion_mezcla(e: Dict, M: int, verbose: bool = True) -> pd.DataFrame:
    """
    Atomos ocupados post-burn por (componente, cadena). `N1out` es max_i S_i;
    si toca N, el truncamiento esta mordiendo y hay que subirlo.
    """
    N, burn, filas = e["n_atomos"], e["burn"], []
    for k in range(e["n_components"]):
        for c in sorted(e["models_chains"][k]):
            post = e["models_chains"][k][c].traces["N1out"][burn:]
            filas.append({"M": M, "FPC": e["component_idx"][k] + 1, "cadena": c + 1,
                          "media": post.mean(), "max": int(post.max()),
                          "p99": float(np.quantile(post, 0.99)), "N_trunc": N,
                          "toca_truncamiento": bool(post.max() >= N),
                          "frac_en_N": float(np.mean(post >= N))})
    df = pd.DataFrame(filas)
    e["ocup_df"] = df
    if verbose:
        print(f"! [M={M}] alguna cadena alcanza N: subir mcmc_config['N']."
              if df["toca_truncamiento"].any()
              else f"OK [M={M}] ninguna cadena alcanza N={N}: el truncamiento no restringe.")
    return df


def _series_gating(mt, X) -> Dict[str, np.ndarray]:
    """Series post-burn del gating, promediadas con el peso medio de cada atomo
    (invariantes a la permutacion de etiquetas)."""
    tr, b = mt.traces, mt.burn
    al, ps, gm = tr["alphahout"][b:], tr["psijhout"][b:], tr["Gammajhout"][b:]
    T, p = al.shape[0], X.shape[1]
    nombres = (["atomos_global", "atomos_punto", "senal_gating", "sd_argumento", "alpha"]
               + [f"psi_{j}" for j in range(p)] + [f"Gamma_{j}" for j in range(p)])
    S = {n: np.empty(T) for n in nombres}
    for t in range(T):
        w = pesos_probit(X, al[t], ps[t], gm[t])
        wbar = w.mean(0)
        s = wbar[:-1].sum()
        u = wbar[:-1] / s if s > 1e-12 else np.full(len(wbar) - 1, 1.0 / (len(wbar) - 1))
        eta = al[t][:, None] - np.sum(
            ps[t][:, None, :] * np.abs(X[None] - gm[t][:, None, :]), axis=2)
        S["atomos_global"][t] = 1.0 / np.sum(wbar ** 2)
        S["atomos_punto"][t] = np.mean(1.0 / np.sum(w ** 2, axis=1))
        S["sd_argumento"][t] = np.sum(u * eta.std(axis=1))
        S["alpha"][t] = np.sum(u * al[t])
        for j in range(p):
            S[f"psi_{j}"][t] = np.sum(u * ps[t][:, j])
            S[f"Gamma_{j}"][t] = np.sum(u * gm[t][:, j])
    S["senal_gating"] = S["atomos_global"] - S["atomos_punto"]
    return S


def diagnostico_gating(e: Dict, M: int, cache: Dict):
    """
    Diagnostico de psi, alpha, Gamma. Retorna `(gat_df, series)` con
    `series[k][c]` para graficar; escribe `48_gating_diagnosticos.csv`. `cache`
    (con clave (M, componente, cadena)) evita recalcular las series.
    """
    ci, filas, series = e["component_idx"], [], {}
    for k in range(e["n_components"]):
        X = e["dfs_train"][k].iloc[:, 1:].to_numpy()
        cams = sorted(e["models_chains"][k])
        for c in cams:
            if (M, ci[k], c) not in cache:
                cache[(M, ci[k], c)] = _series_gating(e["models_chains"][k][c], X)
        series[k] = {c: cache[(M, ci[k], c)] for c in cams}
        for v in series[k][cams[0]]:
            C = np.vstack([series[k][c][v] for c in cams])
            fila = {"M": M, "FPC": ci[k] + 1, "variable": v}
            fila.update({f"media_c{c+1}": series[k][c][v].mean() for c in cams})
            fila.update(diagnostico_variable(C))
            filas.append(fila)
    gat = pd.DataFrame(filas)
    gat.to_csv(e["paths"]["out_report"] / "48_gating_diagnosticos.csv", index=False)
    e["gating_df"] = gat
    return gat, series


# ==========================================================================
# 5. PIP
# ==========================================================================

def pip_punto(e: Dict, sd_max_aviso: float = 0.15) -> pd.DataFrame:
    """PIP global por covariable (`46_pip.csv`) y dispersion entre cadenas."""
    pip = matriz_pip(e["models_chains"], e["burn"],
                     component_idx=e["component_idx"], verbose=True)
    pip.to_csv(e["paths"]["out_report"] / "46_pip.csv")
    e["pip_df"] = pip
    sd_cols = [c for c in pip.columns if c.endswith("_sd")]
    e["pip_sd_max"] = float(np.nanmax(pip[sd_cols].to_numpy())) if sd_cols else float("nan")
    print(f"[M={e['n_components']}] dispersion maxima entre cadenas: {e['pip_sd_max']:.3f}"
          + ("   ! las cadenas discrepan sobre la seleccion"
             if e["pip_sd_max"] > sd_max_aviso else "   OK las cadenas coinciden"))
    return pip


def contraste_generador(e: Dict, escenario_id: int, umbral: float = 0.5,
                        cos_min: float = 0.95) -> pd.DataFrame:
    """
    PIP contra el soporte verdadero del generador (`47_contraste_pip.csv`).

    `soporte[j, l-1, m]` es True si xi_(m, t-l) entra en E[xi_(j,t) | pasado].
    El soporte cubre solo los rezagos del generador: los lags del pipeline que
    lo exceden cuentan como NO verdaderos. El PIP es sobre scores ESTIMADOS: el
    contraste es directo solo si psi_k ~ phi_k (se imprime |cos|).
    """
    M = e["n_components"]
    esc = cargar_escenario(str(e["paths"]["raw"] / f"escenario_{escenario_id}.npz"))
    sop, ci = esc["interno_soporte"], e["component_idx"]
    L_gen = sop.shape[1]
    verdad = {f"FPC {ci[k] + 1}": [
                  f"fpc_{m + 1}_lag{l + 1}" for l in range(min(e["n_lags"], L_gen))
                  for m in range(M) if sop[ci[k], l, m]]
              for k in range(M)}
    w = pesos_trapezoidales(esc["grilla"])
    cos = np.abs(np.diag(cargar_fpca(e["paths"]).Psi_grid.T
                         @ (w[:, None] * esc["interno_Phi"][:, :M])))
    e["cos_psi_phi"] = cos
    print(f"[M={M}] |cos(psi_k, phi_k)| = {np.round(cos, 3)}"
          + ("   <- rotadas: contraste aproximado" if cos.min() < cos_min else ""))
    contraste = contraste_con_verdad(e["pip_df"], verdad, umbral=umbral)
    e["contraste"] = contraste
    if not contraste.empty:
        contraste.to_csv(e["paths"]["out_report"] / "47_contraste_pip.csv")
    return contraste


# ==========================================================================
# 6. MUESTREADOR <-> PREDICTOR Y PASO DE GAMMA
# ==========================================================================

def verificar_muestreador_predictor(e: Dict, corr_min: float = 0.999) -> bool:
    """
    `inEout` (E[y|x] dentro de muestra, calculado por MATLAB) debe coincidir con
    lo que el predictor de Python reconstruye de las mismas trazas. Es la unica
    prueba directa de que ambos lados interpretan las trazas igual.
    """
    ok, M = True, e["n_components"]
    print(f"-- M={M} --")
    for k in e["models_chains"]:
        for c in sorted(e["models_chains"][k]):
            m = e["models_chains"][k][c]
            mu_tr = m.predictor_.predict(m._diseno(e["dfs_train"][k]))
            inE = m.traces["inEout"][m.burn:].mean(axis=0)
            corr = float(np.corrcoef(inE, mu_tr)[0, 1])
            ok &= corr > corr_min
            print(f"  FPC {e['component_idx'][k]+1} cadena{c+1}: corr={corr:.6f}  "
                  f"|dif|max={float(np.abs(inE - mu_tr).max()):.4f}"
                  + ("" if corr > corr_min else "   ! REVISAR"))
    e["contrato_ok"] = bool(ok)
    return bool(ok)


def diagnostico_paso_gamma(EST: Dict, M_OK: Sequence[int]) -> pd.DataFrame:
    """
    Dos chequeos del paso de Gamma, leidos de las trazas: cada Gamma_hj cae en la
    grilla de su predictor, y pm1 es finita y suma 1 en cada sorteo
    (`gamdiagout`). Escribe `49b_paso_gamma.csv` por M.
    """
    cache, filas = {}, []
    for M in M_OK:
        e = EST[M]
        for k in range(e["n_components"]):
            for c in range(e["n_iter"]):
                ruta = ruta_traza(e["paths"], e["component_idx"][k] + 1, c + 1)
                if ruta not in cache:
                    m = loadmat(str(ruta), variable_names=["Gstar", "Gammajhout", "gamdiagout"])
                    assert "Gstar" in m and "gamdiagout" in m, (
                        f"{ruta.name} no trae Gstar/gamdiagout: ¿psbp_train.m desactualizado?")
                    G = np.asarray(m["Gammajhout"], float)
                    Gs = np.atleast_2d(np.asarray(m["Gstar"], float))
                    if G.ndim == 2:
                        G = G[:, :, None]
                    if Gs.shape[1] != G.shape[2]:
                        Gs = Gs.T
                    dist = max(float(np.abs(G[:, :, j, None] - Gs[None, None, :, j]).min(-1).max())
                               for j in range(G.shape[2]))
                    gd = np.asarray(m["gamdiagout"], float)[e["burn"]:]
                    cache[ruta] = {
                        "fuera_de_grilla": dist > 1e-5 * max(1.0, float(np.abs(Gs).max())),
                        "dist_max": dist, "sorteos": int(gd[:, 0].sum()),
                        "no_finitos": int(gd[:, 1].sum()),
                        "err_suma_max": float(gd[:, 2].max()),
                        "rango_log_mediana": float(np.median(gd[:, 3])),
                        "rango_log_max": float(gd[:, 3].max())}
                filas.append({"M": M, "FPC": e["component_idx"][k] + 1, "cadena": c + 1,
                              **cache[ruta]})
    df = pd.DataFrame(filas)
    for M in M_OK:
        df[df.M == M].to_csv(EST[M]["paths"]["out_report"] / "49b_paso_gamma.csv", index=False)
    ok = ((~df["fuera_de_grilla"]).all() and df["no_finitos"].sum() == 0
          and df["err_suma_max"].max() < 1e-8)
    print(("OK" if ok else "X") + " Gamma siempre en su grilla y pm1 finita y normalizada"
          f"   ·   rango_log mediano {df['rango_log_mediana'].median():.0f}")
    return df


# ==========================================================================
# 7. TABLAS QUE CRUZAN LOS M
# ==========================================================================

def tabla_barrido(EST: Dict, M_OK: Sequence[int], path_barrido: Path) -> pd.DataFrame:
    """
    Una fila por M (`80_convergencia_por_M.csv`). Las cifras son comparables
    entre M: R-hat/ESS/Geweke son el PEOR caso del punto, no un promedio sobre un
    numero de variables que cambia con M.
    """
    filas = []
    for M in M_OK:
        e, res, ocu, con = EST[M], EST[M]["res_conv"], EST[M]["ocup_df"], EST[M].get("contraste")
        fila = {"M": M, "experiment_id": e["eid"],
                "p_covariables": len(e["cov_por_componente"][0]),
                "n_variables_diag": int(res["n_variables"]),
                "rhat_max": float(res["rhat_max"]), "ess_min": float(res["ess_min"]),
                "geweke_max": float(res["geweke_max"]),
                "n_no_converge": int(res["n_no_converge"]),
                "converge": bool(res["todo_converge"]),
                "ocupacion_media": float(ocu["media"].mean()),
                "ocupacion_max": int(ocu["max"].max()),
                "N_trunc": int(ocu["N_trunc"].iloc[0]),
                "toca_truncamiento": bool(ocu["toca_truncamiento"].any()),
                "pip_sd_max": e["pip_sd_max"], "contrato_ok": bool(e["contrato_ok"])}
        if con is not None and not con.empty and "pip_media_activas" in con.columns:
            fila["pip_media_activas"] = float(con["pip_media_activas"].mean())
        filas.append(fila)
    df = pd.DataFrame(filas).set_index("M")
    df.to_csv(path_barrido / "80_convergencia_por_M.csv")
    return df


def ocupacion_barrido(EST: Dict, M_OK: Sequence[int], path_barrido: Path):
    """Ocupacion por (M, FPC) (`82_ocupacion_por_M.csv`). Retorna (largo, pivot)."""
    todo = pd.concat([EST[M]["ocup_df"] for M in M_OK], ignore_index=True)
    todo.to_csv(path_barrido / "82_ocupacion_por_M.csv", index=False)
    return todo, todo.groupby(["M", "FPC"])["media"].mean().unstack("FPC")


def pip_barrido(EST: Dict, M_OK: Sequence[int], path_barrido: Path) -> pd.DataFrame:
    """PIP en formato largo (M, covariable, componente) (`83_pip_por_M.csv`): las
    matrices de distintos M no comparten filas ni columnas."""
    partes = []
    for M in M_OK:
        pip = EST[M]["pip_df"]
        largo = pip[[c for c in pip.columns if not c.endswith("_sd")]].stack() \
            .rename("pip").reset_index()
        largo.columns = ["covariable", "componente", "pip"]
        largo.insert(0, "M", M)
        partes.append(largo)
    df = pd.concat(partes, ignore_index=True)
    df.to_csv(path_barrido / "83_pip_por_M.csv", index=False)
    return df


def veredicto_barrido(barrido: pd.DataFrame, nombre: str, M_FPCA_LIST: Sequence[int],
                      M_OK: Sequence[int], path_barrido: Path) -> None:
    """Veredicto del barrido y tendencia del muestreo con M (>= 3 puntos)."""
    print("=" * 66)
    print(f"  BARRIDO {nombre}   ·   M evaluados: {list(M_OK)}")
    print("=" * 66)
    for M in M_OK:
        f = barrido.loc[M]
        print(f"  M={M}  p={int(f['p_covariables'])}  Rhatmax={f['rhat_max']:.3f}  "
              f"ESSmin={f['ess_min']:.0f}  ocup={f['ocupacion_media']:.1f}/{int(f['N_trunc'])}  "
              f"{'OK' if f['converge'] else 'X NO CONVERGE'}")
    no = [int(M) for M in barrido.index if not barrido.loc[M, "converge"]]
    falt = sorted(set(M_FPCA_LIST) - set(M_OK))
    print("-" * 66)
    if falt:
        print(f"! sin trazas todavia: M = {falt}  -> correr psbp_fd_iteracion.m")
    if no:
        print(f"X NO convergen: M = {no}. Excluirlos de la comparacion de _04/_05,")
        print("  o subir nsim/burn antes de leer su error como efecto de M.")
    else:
        print("OK todo el barrido converge: las diferencias entre los M son atribuibles")
        print("   al modelo y no al muestreo.")
    if len(M_OK) >= 3:
        r = np.corrcoef(list(M_OK), barrido["rhat_max"].to_numpy())[0, 1]
        s = np.corrcoef(list(M_OK), barrido["ess_min"].to_numpy())[0, 1]
        print(f"\ncorr(M, Rhatmax) = {r:+.3f}   ·   corr(M, ESSmin) = {s:+.3f}")
        if r > 0.8 or s < -0.8:
            print("  <- el muestreo se degrada monotonamente con M: parte de lo que _04")
            print("     atribuya a M seria esfuerzo de muestreo insuficiente.")
    print("=" * 66)
    print(f"Tablas y figuras cruzadas en {path_barrido}  (archivos 80-83)")


# ==========================================================================
# 8. COMPARACION ENTRE VARIANTES
# ==========================================================================

def comparar_variantes(variantes: Dict[str, str], M_ref: int, escenario_id: int,
                       replica_id: int, project_root: Path, path_var: Path):
    """
    Lee lo que `_03` dejo en `M = M_ref` para cada variante `{etiqueta: basename}`.
    Verifica que compartan realizacion y retorna `(hp_var, comp_var, largo)`:
    hiperparametros efectivos, convergencia/ocupacion/paso de Gamma por variante y
    los diagnosticos por variable para graficar. Escribe los CSV 85.
    """
    path_var.mkdir(parents=True, exist_ok=True)
    VAR = {}
    for v, basename in variantes.items():
        P = rutas_por_M(basename, escenario_id, replica_id, [M_ref],
                        dominio="simulaciones", project_root=project_root)[M_ref]
        base = f"{basename}_{escenario_id}_r{replica_id:02d}"
        f40, f49 = P["out_report"] / "40_diagnosticos_mcmc.csv", P["out_report"] / "49b_paso_gamma.csv"
        if not f40.exists():
            print(f"! {v}: sin {f40.name}; correr el _03 de {basename}")
            continue
        VAR[v] = {"hp": cargar_hiperparametros(P), "diag": pd.read_csv(f40),
                  "curvas": cargar_curvas(P)[0],
                  "ocup": pd.read_csv(project_root / "reports" / "simulaciones"
                                      / f"{base}_barrido_M" / "82_ocupacion_por_M.csv"),
                  "gamma": pd.read_csv(f49) if f49.exists() else None}
    assert VAR, "Ninguna variante tiene diagnosticos todavia."
    v0 = next(iter(VAR))
    for v in VAR:
        assert np.array_equal(VAR[v]["curvas"], VAR[v0]["curvas"]), \
            f"{v} y {v0} no comparten realizacion: la comparacion no es valida."
    print(f"OK {len(VAR)} variantes sobre la misma realizacion: {list(VAR)}")

    filas = []
    for v, d in VAR.items():
        hp = d["hp"]
        for it in hp["hyperparams_list"]:
            h = it["hyperparams"]
            filas.append({"variante": v, "FPC": it["fpc_idx"], "scores_scale": hp["scores_scale"],
                          "N": hp["mcmc_config"]["N"], "nsim": hp["mcmc_config"]["nsim"],
                          "burn": hp["mcmc_config"]["burn"], "atau": h["atau"], "btau": h["btau"],
                          "ag": h["ag"], "bg": h["bg"], "apij": np.mean(h["apij"]),
                          "bpij": np.mean(h["bpij"]), "mupsij": np.mean(h["mupsij"]),
                          "taupsij": np.mean(h["taupsij"])})
    hp_var = pd.DataFrame(filas).set_index(["variante", "FPC"])
    hp_var.to_csv(path_var / "85_hiperparametros_por_variante.csv")
    cols = ["atau", "btau", "ag", "bg", "apij", "bpij", "mupsij", "taupsij"]
    for v in VAR:
        dif = float(np.abs(hp_var.loc[v, cols].to_numpy() - hp_var.loc[v0, cols].to_numpy()).max())
        print(f"max |hp({v}) - hp({v0})| = {dif:.2e}"
              + ("   (mismos hiperparametros)" if dif < 1e-10 else "   (hiperparametros distintos)"))

    filas = []
    for v, d in VAR.items():
        dg, oc, gm = d["diag"], d["ocup"], d["gamma"]
        oc = oc[oc.M == M_ref]
        fila = {"variante": v, "n_no_converge": int((~dg["converge"]).sum()),
                "n_variables": len(dg), "rhat_max": dg["rhat"].max(),
                "geweke_max": dg["geweke_max"].max(), "n_post": int(dg["n_post"].iloc[0])}
        for par in ("beta_j", "p_j"):
            s = dg[dg.param == par]
            fila[f"ess_min_{par}"] = s["ess_min"].min()
            fila[f"ess_por_draw_{par}"] = s["ess_min"].min() / fila["n_post"]
        fila.update({"ocupacion_media": oc["media"].mean(), "ocupacion_max": int(oc["max"].max()),
                     "N": int(oc["N_trunc"].iloc[0]),
                     "frac_en_N": oc["frac_en_N"].mean() if "frac_en_N" in oc else np.nan,
                     "gamma_fuera_grilla": bool(gm["fuera_de_grilla"].any()) if gm is not None else np.nan,
                     "gamma_no_finitos": int(gm["no_finitos"].sum()) if gm is not None else np.nan})
        filas.append(fila)
    comp_var = pd.DataFrame(filas).set_index("variante")
    comp_var.to_csv(path_var / "85_convergencia_por_variante.csv")

    largo = pd.concat([d["diag"].assign(variante=v) for v, d in VAR.items()], ignore_index=True)
    largo["etiqueta"] = ("FPC" + largo["componente"].astype(str) + " "
                         + largo["variable"].str.split("_").str[-1])
    return hp_var, comp_var, largo
