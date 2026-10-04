"""
comparacion_barrido.py
======================
Comparacion del PSBPM-FD contra el FAR(p) y el Random Forest sobre el barrido en
M, SIN graficos (las figuras viven en `graphics/viz_comparacion.py`). Es la
logica que antes estaba en el notebook `_05` (y, resumida, en `_06` y `_07`).

Los tres modelos, contra la misma curva suavizada y con las mismas diez metricas
que `evaluacion_barrido`:

    FAR(p)    sobre los COEFICIENTES de la base, en la metrica L^2, p = N_LAGS
              fijo (no se barre: seria una ventaja de especificacion).
    RF        un bosque por componente con SUS PROPIOS rezagos; el orden 1..N_LAGS
              se elige por ventanas de TRAIN ganadas en `mae_f`, nunca con test.
    PSBPM-FD  su prediccion y su banda, leidas de `banda_funcional_psbp.npz` que
              persiste `_04`: no se recalcula nada desde las trazas.

Bloque A (error puntual) aplica a los tres; el Bloque B (intervalos) solo al FAR y
al PSBPM-FD, los unicos con mecanismo de intervalo propio. `EST[M]["CURVAS"]` guarda
la curva predicha de cada modelo; el resto se deriva de ahi.
"""

from __future__ import annotations

import json
from typing import Dict, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from ..models.pspb_fd_v3 import curva_media_desde_scores
from ..pipelines import cargar_representacion
from ..utils.linalg import safe_chol
from .far_operador import FARp, seleccionar_kn
from .intervalos import residuos_para_banda, banda_predictiva_modelo, sigma_residual
from .metrics_distribucional import (
    indicador_cobertura, indicador_cobertura_simultanea, resumen_intervalo,
)
from .metrics_puntual import mise, normas_error_por_origen
from .rolling import ventana_movil_funcional
from .evaluacion_barrido import METRICAS_B

__all__ = [
    "G_FAR", "G_PSG", "G_PSB", "PISO",
    "preparar_diseno", "info_generador", "ajustar_far", "ajustar_rf", "cargar_psbp",
    "tablas_modelos", "verificar_cadena_modelos",
    "construir_bandas", "tabla_bloque_B", "ventana_bloque_B",
    "ganadores_por_ventana", "resumen_min_max", "promedio_vs_M",
    "origenes_extremos", "contraste_origenes",
]

G_FAR = "FAR (IC del modelo AR)"
G_PSG = "PSBPM-FD (IC gaussiano, mismo mecanismo)"
G_PSB = "PSBPM-FD (predictiva bayesiana)"
PISO = "(truncamiento FPCA)"       # referencia, no es un competidor


# ==========================================================================
# 1. DISENO COMUN Y MODELOS
# ==========================================================================

def preparar_diseno(EST: Dict, DIS: Dict, ORIG: Dict) -> None:
    """
    Por cada M: datasets completos (train + test), objetivo `Y_obs` (n_orig, M),
    matriz de diseno `Z` con todos los rezagos (en orden de `cov_names`), el mapa
    scores->curva determinista y el piso de truncamiento FPCA (MISE de la curva
    suavizada contra su propia proyeccion en M, cota inferior de cualquier modelo
    sobre esa representacion).
    """
    n_orig, es_train, grilla = ORIG["n_orig"], ORIG["es_train"], DIS["grilla"]
    print(f"origenes evaluados: {n_orig}  (train {es_train.sum()} · test {(~es_train).sum()})")
    for M, e in EST.items():
        n_comp = e["n_components"]
        dfs = {k: pd.concat([e["dfs_train"][k], e["dfs_test"][k]], ignore_index=True)
               for k in range(n_comp)}
        assert len(dfs[0]) == n_orig, f"[M={M}] {len(dfs[0])} != {n_orig}"
        col2k = {c: k for k in range(n_comp) for c in dfs[k].columns[1:]}
        assert set(col2k) == set(e["cov_names"]), (
            f"[M={M}] covariables {sorted(col2k)} != cov_names {e['cov_names']}.")
        Y_obs = np.column_stack([dfs[k].iloc[:, 0].to_numpy(dtype=float) for k in range(n_comp)])
        Z = np.column_stack([dfs[col2k[c]][c].to_numpy(dtype=float) for c in e["cov_names"]])
        e.update({"dfs_full": dfs, "Y_obs": Y_obs, "Z": Z, "CURVAS": {}, "PRED": {}, "INFO": {}})
        e["curvas_desde_scores_std"] = lambda Y, e=e: curva_media_desde_scores(Y, e["fpca"], e["std"])
        e["X_proj"] = e["fpca"].reconstruct(e["SCORES"])[DIS["N_LAGS"]:]
        assert np.allclose(e["curvas_desde_scores_std"](Y_obs), e["X_proj"], atol=1e-8), (
            f"[M={M}] el mapa scores->curva no reproduce la proyeccion FPCA.")
        e["MISE_TRUNC"] = {b: mise(ORIG["X_obj_ev"][m], e["X_proj"][m], grilla)
                           for b, m in (("train", es_train), ("test", ~es_train))}
    print("MISE del TRUNCAMIENTO FPCA (piso comun a todos los modelos de cada M):")
    for M, e in EST.items():
        print(f"  M={M:>2}   train {e['MISE_TRUNC']['train']:.6f}   test {e['MISE_TRUNC']['test']:.6f}")


def info_generador(e0: Dict, n_lags: int) -> None:
    """R^2 del oraculo de Bayes y del mejor lineal en los mismos rezagos, leidos del
    diagnostico del generador: la brecha es lo que un FAR(p) no puede alcanzar."""
    p = e0["paths"]["raw"] / "simulation_config.json"
    d = json.loads(p.read_text(encoding="utf-8")).get("diagnostico", {}) if p.exists() else {}
    if "r2_oraculo" in d and "r2_lineal" in d:
        print(f"R^2 L2 del oraculo de Bayes ({n_lags} rezagos) = {float(d['r2_oraculo']):.4f}   ·   "
              f"mejor lineal en los mismos rezagos = {float(d['r2_lineal']):.4f}")


def ajustar_far(EST: Dict, DIS: Dict, ORIG: Dict, kn_max: int = 12, criterio: str = "L2",
                verbose: bool = True) -> Dict:
    """
    FAR(p), p = N_LAGS, sobre los coeficientes de la representacion, en
    coordenadas blanqueadas con la Cholesky de la Gram (`W = L L^T`, `theta_w =
    theta L`), de modo que el producto escalar euclideo ES el producto L^2 (con
    base ortonormal W = I y es identidad; se verifica igual). Por eso se ajusta
    con `pesos="conteo"`: pesar de nuevo con la trapezoidal aplicaria la metrica
    L^2 dos veces. kn se elige por `far.cv` sobre train. Se EVALUA con los K
    coeficientes completos, sin truncar a M. Llena `e["CURVAS"]["FAR"]` y
    `e["PRED"]["FAR"]` en cada punto; retorna el resumen del ajuste.
    """
    T0, N_LAGS, n_orig = DIS["T0"], DIS["N_LAGS"], ORIG["n_orig"]
    e0 = EST[next(iter(EST))]
    fr, THETA, _ = cargar_representacion(e0["paths"])
    THETA = np.asarray(THETA, dtype=float)
    W = np.asarray(fr.gram_, dtype=float)
    L = safe_chol(W)
    THETA_W = THETA @ L
    n_g = np.einsum("tk,kl,tl->t", THETA, W, THETA)
    err = float(np.max(np.abs(n_g - (THETA_W ** 2).sum(axis=1))) / max(np.max(n_g), 1e-300))
    assert err < 1e-8, f"El blanqueo por la Gram no preserva la norma L^2 (error relativo {err:.2e})."

    kn_tope = int(min(kn_max, THETA_W.shape[1]))
    cv = seleccionar_kn(THETA_W[:T0], kn_max=kn_tope, pesos="conteo", p=N_LAGS)
    kn = {"L1": cv.kn_L1, "L2": cv.kn_L2, "Linf": cv.kn_Linf}[criterio]
    far = FARp(p=N_LAGS, kn=kn, pesos="conteo").fit(THETA_W[:T0])
    diag = far.diagnostico_kn()
    pred_w = far.predict_serie(THETA_W)
    assert pred_w.shape[0] == n_orig, (pred_w.shape, n_orig)
    THETA_pred = np.linalg.solve(L.T, pred_w.T).T
    X_far = fr.reconstruct(THETA_pred)
    hs = float(np.sqrt(np.sum(np.asarray(diag["norma_hs_por_rezago"]) ** 2)))
    for e in EST.values():
        e["CURVAS"]["FAR"] = X_far
        e["PRED"]["FAR"] = e["std"].transform(e["fpca"].transform(THETA_pred))
        e["INFO"]["FAR"] = (f"coeficientes K={THETA.shape[1]} COMPLETO (no truncado a M), p={N_LAGS}, "
                            f"kn={kn} (far.cv/{criterio}), ||rho||_HS={hs:.4f}")
        e.update({"far_kn": int(kn), "far_p": int(N_LAGS), "far_hs": hs,
                  "far_hs_por_rezago": list(diag["norma_hs_por_rezago"]),
                  "far_cond": float(diag["condicion"])})
    if verbose:
        print(f"base: K = {THETA.shape[1]}   ·   cond(W) = {np.linalg.cond(W):.2f}   ·   "
              f"||x||_L2 == ||theta_w||_2 (error relativo {err:.2e})")
        print(f"far.cv: hold-out de las ultimas {cv.n_validacion} curvas de train · kn_max = {kn_tope}")
        print(f"{'kn':>4}  " + "  ".join(f"{c:>9}" for c in cv.columnas[1:]))
        for fila in cv.tabla:
            print(f"{int(fila[0]):>4}  " + "  ".join(f"{v:>9.4f}" for v in fila[1:])
                  + ("  <-" if int(fila[0]) == kn else ""))
        l2 = cv.tabla[:, 2]
        plano = float((l2.max() - l2.min()) / l2.min())
        print(f"kn elegido por {criterio}: {kn}   (L1 -> {cv.kn_L1}, Linf -> {cv.kn_Linf})   "
              f"amplitud relativa {plano:.2%}" + ("   <- PLANO: la eleccion de kn no es informativa"
                                                  if plano < 0.05 else ""))
        print(f"radio espectral VAR({N_LAGS}) = {diag['radio_espectral']:.4f}"
              + ("   <- >= 1: no estacionario" if diag["radio_espectral"] >= 1 else "")
              + f"   ·   condicion lambda_1/lambda_kn = {diag['condicion']:.2f}")
    return {"kn": kn, "cv": cv, "diag": diag, "hs": hs}


def _cols_propias(e: Dict, k: int, r: int):
    """Columnas de Z con los rezagos 1..r de la componente k."""
    nombres = e["cov_por_componente"][k][:r]
    for l, nombre in enumerate(nombres, start=1):
        assert nombre.endswith(f"_lag{l}"), (
            f"cov_por_componente[{k}] no esta en orden de rezago: {nombre!r} en lag {l}.")
    return [e["cov_names"].index(nombre) for nombre in nombres]


def ajustar_rf(EST: Dict, DIS: Dict, ORIG: Dict, params: Dict, pesos_tau=None) -> None:
    """
    Random Forest: un bosque por componente con SUS PROPIOS rezagos 1..r (los mismos
    insumos del PSBPM-FD y el FAR). El orden r en 1..N_LAGS se elige por el % de
    ventanas de TRAIN ganadas en `mae_f`, nunca con test. Llena `e["CURVAS"]["RF"]`.
    """
    N_LAGS, W_REF, es_train = DIS["N_LAGS"], DIS["W_REF"], ORIG["es_train"]
    ordenes = list(range(1, N_LAGS + 1))
    for M, e in EST.items():
        def _pred(r):
            P = np.empty_like(e["Y_obs"])
            for k in range(e["n_components"]):
                Zk = e["Z"][:, _cols_propias(e, k, r)]
                rf = RandomForestRegressor(**params).fit(Zk[es_train], e["Y_obs"][es_train, k])
                P[:, k] = rf.predict(Zk)
            return P
        preds = {r: _pred(r) for r in ordenes}
        if len(ordenes) == 1:
            ganador, info = ordenes[0], None
        else:
            filas = []
            for r in ordenes:
                t_ = ventana_movil_funcional(
                    ORIG["X_obj_ev"], e["curvas_desde_scores_std"](preds[r]), DIS["grilla"],
                    ORIG["T0_orig"], w=W_REF, t_offset=N_LAGS, pesos_tau=pesos_tau,
                    bloque_A=True, verbose=False)
                t_ = t_[(~t_["cruza_T0"]) & (t_["bloque"] == "train")]
                filas.append(t_[["t_centro", "mae_f"]].assign(orden=r))
            piv = pd.concat(filas, ignore_index=True).pivot(index="t_centro", columns="orden", values="mae_f")
            pct = piv.idxmin(axis=1).value_counts(normalize=True).reindex(ordenes, fill_value=0.0)
            ganador = int(pct.idxmax())     # empate -> el primer orden (mas parco)
            info = f"orden elegido por ventanas de train: {ganador} ({pct[ganador]:.1%} de {len(piv)}) de {ordenes}"
        e["PRED"]["RF"] = preds[ganador]
        e["CURVAS"]["RF"] = e["curvas_desde_scores_std"](preds[ganador])
        e["INFO"]["RF"] = f"n_estimators={params['n_estimators']}, leaf>={params['min_samples_leaf']}, orden={ganador}"
        e["ml_orden_rf"] = ganador
        assert e["PRED"]["RF"].shape == e["Y_obs"].shape
        print(f"M={M}: RF orden={ganador}" + (f"   ({info})" if info else "")
              + f"   ·   {e['n_components']} modelos univariantes (rezagos propios)")


def cargar_psbp(EST: Dict, DIS: Dict, ORIG: Dict) -> tuple:
    """
    Prediccion y banda del PSBPM-FD, leidas de `banda_funcional_psbp.npz` (la
    persiste `_04`: es la misma prediccion puntual, media analitica, y la banda por
    cuantiles). Verifica que compartan particion, nivel y objetivo. Los M sin
    artefacto se quitan de `EST` con aviso. Retorna `M_OK`.
    """
    for M in list(EST):
        e = EST[M]
        npz = e["paths"]["predict"] / "banda_funcional_psbp.npz"
        if not npz.exists():
            print(f"! M={M} saltado: falta {npz.name}. Ejecuta 200_04 para este M.")
            del EST[M]
            continue
        z = np.load(npz, allow_pickle=False)
        assert z["li"].shape == ORIG["X_obj_ev"].shape, (
            f"[M={M}] la banda del _04 es {z['li'].shape} y aqui {ORIG['X_obj_ev'].shape}: "
            "¿distinta particion?")
        assert abs(float(z["nivel"][0]) - DIS["NIVEL"]) < 1e-9, (
            f"[M={M}] el _04 uso nivel {float(z['nivel'][0])} y aqui {DIS['NIVEL']}.")
        assert int(z["T0"][0]) == DIS["T0"] and int(z["n_lags"][0]) == DIS["N_LAGS"]
        assert str(z["objetivo"][0]) == DIS["OBJETIVO"], (
            f"[M={M}] el _04 evaluo contra {z['objetivo'][0]!r}, aqui {DIS['OBJETIVO']!r}.")
        e["CURVAS"]["PSBPM-FD"] = z["X_pred"].astype(float)
        e["BANDA_PSBP"] = (z["li"].astype(float), z["ls"].astype(float))
    assert EST, "Ningun punto del barrido tiene la banda del PSBPM-FD: ejecuta 200_04."
    M_OK = tuple(sorted(EST))
    print(f"OK PSBPM-FD leido del _04 en M = {list(M_OK)}")
    return M_OK


# ==========================================================================
# 2. BLOQUE A
# ==========================================================================

def tablas_modelos(e: Dict, DIS: Dict, ORIG: Dict, pesos_tau=None, prefijo: int = 70) -> None:
    """
    Ventana movil de cada modelo (y del piso de truncamiento) por ancho, contra la
    curva suavizada: `e["tablas_w"][w]` con la columna `modelo`
    (`<prefijo+2>_ventana_modelos_w*.csv`). El FAR se evalua con sus K coeficientes;
    RF y PSBPM-FD vienen de `CURVAS`.
    """
    e["tablas_w"] = {}
    series = {**e["CURVAS"], PISO: e["X_proj"]}
    for w in DIS["VENTANAS_W"]:
        partes = []
        for nombre, X in series.items():
            t_ = ventana_movil_funcional(ORIG["X_obj_ev"], X, DIS["grilla"], ORIG["T0_orig"], w=w,
                                         t_offset=DIS["N_LAGS"], pesos_tau=pesos_tau,
                                         bloque_A=True, verbose=False)
            t_["modelo"] = nombre
            partes.append(t_)
        out = pd.concat(partes, ignore_index=True)
        out.attrs["w"] = w
        e["tablas_w"][w] = out
        out.to_csv(e["paths"]["out_report"] / f"{prefijo + 2}_ventana_modelos_w{w}.csv", index=False)
    e["ventana_df"] = e["tablas_w"][DIS["W_REF"]]


def verificar_cadena_modelos(EST: Dict, W_REF: int, tol: float = 1e-9) -> None:
    """
    mae_f <= l2_medio <= linf_medio y l2_medio <= rmse_f (Jensen). `rmse_f` NO
    pertenece a la cadena L1<=L2<=Linf: es media cuadratica entre origenes, asi que
    puede superar a `linf_medio` si el error es desigual (se avisa, no se rechaza).
    """
    for M, e in EST.items():
        t = e["tablas_w"][W_REF]
        s = t[~t["cruza_T0"]].dropna(subset=["mae_f", "l2_medio", "linf_medio", "rmse_f"])
        assert (s["mae_f"] <= s["l2_medio"] + tol).all(), f"[M={M}] mae_f > l2_medio"
        assert (s["l2_medio"] <= s["linf_medio"] + tol).all(), f"[M={M}] l2_medio > linf_medio"
        assert (s["l2_medio"] <= s["rmse_f"] + tol).all(), f"[M={M}] l2_medio > rmse_f (Jensen)"
        jen = s[s["rmse_f"] > s["linf_medio"] + tol]
        if len(jen):
            print(f"[M={M}] Jensen: {len(jen)}/{len(s)} ventanas con rmse_f > linf_medio "
                  f"(exceso relativo max {((jen['rmse_f'] / jen['linf_medio']) - 1).max():.2%})")
    print(f"OK cadena L1 <= L2 <= Linf verificada en {len(EST)} puntos (w={W_REF})")


# ==========================================================================
# 3. BLOQUE B
# ==========================================================================

def construir_bandas(e: Dict, DIS: Dict, ORIG: Dict, por_tau: bool = True,
                     correccion_estimacion: bool = True, psbp_gaussiana: bool = True) -> None:
    """
    `e["BANDAS"] = {nombre: ((li, ls), X_pred)}` con tres filas que separan "gana por
    el modelo" de "gana por el mecanismo de intervalo": la banda gaussiana del FAR
    (residuos de TRAIN contra el objetivo), la MISMA banda gaussiana sobre la media
    del PSBPM-FD, y la banda bayesiana nativa del PSBPM-FD (cuantiles, del _04).
    """
    NIVEL, es_train, X_obj = DIS["NIVEL"], ORIG["es_train"], ORIG["X_obj_ev"]
    corr = np.sqrt(1.0 + e["far_kn"] / int(es_train.sum())) if correccion_estimacion else None
    X_far, X_ps = e["CURVAS"]["FAR"], e["CURVAS"]["PSBPM-FD"]
    r_far = residuos_para_banda(X_obj, X_far, es_train)
    bandas = {G_FAR: (banda_predictiva_modelo(X_far, r_far, nivel=NIVEL, por_tau=por_tau,
                                              correccion_estimacion=corr), X_far)}
    if psbp_gaussiana:
        r_ps = residuos_para_banda(X_obj, X_ps, es_train)
        bandas[G_PSG] = (banda_predictiva_modelo(X_ps, r_ps, nivel=NIVEL, por_tau=por_tau,
                                                 correccion_estimacion=corr), X_ps)
    bandas[G_PSB] = (e["BANDA_PSBP"], X_ps)
    e["BANDAS"] = bandas
    sd = sigma_residual(r_far, por_tau=por_tau)
    print(f"M={e['n_components']}  kn={e['far_kn']}  correccion={corr if corr else 1.0:.4f}  "
          f"sigma_eps(tau) del FAR: min={sd.min():.4f} max={sd.max():.4f} (razon {sd.max()/sd.min():.2f})")


def tabla_bloque_B(e: Dict, DIS: Dict, ORIG: Dict) -> pd.DataFrame:
    """
    Winkler (primaria), PICP, PICPB, MPIW, ACE por banda, bloque y objetivo
    (`80_bloqueB_modelos.csv`). Contra los DOS objetivos de docs 03_05_00: la banda
    del FAR vive en los K coeficientes y la del PSBPM-FD en las M autofunciones, asi
    que contra un objetivo unico uno de los dos pagaria un truncamiento que el otro no.
    """
    NIVEL, grilla, es_train = DIS["NIVEL"], DIS["grilla"], ORIG["es_train"]
    filas = []
    for objetivo, Y in (("curva_suavizada", ORIG["X_obj_ev"]), ("representacion_fpca", e["X_proj"])):
        for nombre, ((li, ls), _Xp) in e["BANDAS"].items():
            for etq, mask in (("train", es_train), ("test", ~es_train)):
                r = resumen_intervalo(Y[mask], li[mask], ls[mask], nivel=NIVEL, tau=grilla)
                filas.append({"objetivo": objetivo, "banda": nombre, "bloque": etq,
                              **{k: r[k] for k in ("winkler", "picp", "ee_picp", "ace", "mpiw")},
                              "picp_simultaneo": float(indicador_cobertura_simultanea(
                                  Y[mask], li[mask], ls[mask]).mean())})
    B = pd.DataFrame(filas).set_index(["bloque", "objetivo", "banda"]).sort_index()
    B.to_csv(e["paths"]["out_report"] / "80_bloqueB_modelos.csv")
    e["B_modelos"] = B
    M = e["n_components"]
    for objetivo in ("curva_suavizada", "representacion_fpca"):
        te = B.loc[("test", objetivo)]
        if {G_FAR, G_PSG} <= set(te.index):
            print(f"[M={M}] {objetivo:<20} mismo MECANISMO, distinto MODELO: "
                  f"PSBPM-FD - FAR = {te.loc[G_PSG, 'winkler'] - te.loc[G_FAR, 'winkler']:+.4f} de Winkler")
        if {G_PSG, G_PSB} <= set(te.index):
            print(f"        {objetivo:<20} mismo MODELO, distinto MECANISMO: "
                  f"bayesiana - gaussiana = {te.loc[G_PSB, 'winkler'] - te.loc[G_PSG, 'winkler']:+.4f}")
    return B


def ventana_bloque_B(e: Dict, DIS: Dict, ORIG: Dict, pesos_tau=None, prefijo: int = 70) -> pd.DataFrame:
    """
    Ventana movil de cada banda en el ancho de referencia (`81_ventana_bloqueB_w*.csv`)
    e indicador I_t por origen, puntual y simultaneo (`81b_...`): el promedio esconde
    DONDE falla la banda. Verifica picp_simultaneo <= picp y winkler <= winkler_max_medio
    <= winkler_max_glob.
    """
    W_REF, NIVEL, N_LAGS = DIS["W_REF"], DIS["NIVEL"], DIS["N_LAGS"]
    partes = []
    for nombre, ((li, ls), Xp) in e["BANDAS"].items():
        t_ = ventana_movil_funcional(ORIG["X_obj_ev"], Xp, DIS["grilla"], ORIG["T0_orig"], w=W_REF,
                                     li=li, ls=ls, t_offset=N_LAGS, pesos_tau=pesos_tau,
                                     nivel=NIVEL, bloque_A=True, verbose=False)
        t_["banda"] = nombre
        partes.append(t_)
    vb = pd.concat(partes, ignore_index=True)
    vb.attrs["w"] = W_REF
    vb.to_csv(e["paths"]["out_report"] / f"81_ventana_bloqueB_w{W_REF}.csv", index=False)
    e["ventana_B"] = vb
    s = vb[~vb["cruza_T0"]].dropna(subset=[c for _, c in METRICAS_B])
    tol, M = 1e-9, e["n_components"]
    assert (s["picp_simultaneo"] <= s["picp"] + tol).all(), f"[M={M}] picp_simultaneo > picp"
    assert (s["winkler"] <= s["winkler_max_medio"] + tol).all(), f"[M={M}] winkler > winkler_max_medio"
    assert (s["winkler_max_medio"] <= s["winkler_max_glob"] + tol).all(), f"[M={M}] winkler_max_medio > winkler_max_glob"
    for nombre, ((li, ls), _Xp) in e["BANDAS"].items():
        pd.DataFrame({"t": ORIG["t_orig"], "bloque": np.where(ORIG["es_train"], "train", "test"),
                      "banda": nombre,
                      "I_t_puntual": indicador_cobertura(ORIG["X_obj_ev"], li, ls).mean(axis=1),
                      "I_t_simultaneo": indicador_cobertura_simultanea(ORIG["X_obj_ev"], li, ls).astype(float)}
                     ).to_csv(e["paths"]["out_report"] / f"81b_indicador_cobertura_{nombre.split(' ')[0]}.csv",
                              index=False)
    return vb


# ==========================================================================
# 4. GANADORES, RESUMEN Y COMPARACION ENTRE M
# ==========================================================================

def ganadores_por_ventana(EST: Dict, fuente: str, col_grupo: str, metricas, nivel: float,
                          excluir: Sequence[str] = ()):
    """
    % de ventanas ganadas y razon contra el mejor (y contra el FAR en el Bloque A) por
    grupo, metrica, bloque y M, sobre las ventanas que NO cruzan T0. `fuente` es la
    clave de EST con la tabla: `"ventana_df"` (modelos) o `"ventana_B"` (bandas). En
    picp / picp_simultaneo gana el mas cercano al nominal. La razon se promedia por
    ventana (no se divide el promedio) para que cada ventana pese igual.
    """
    gana, rel = [], []
    for M, e in EST.items():
        t = e[fuente]
        limpio = t[~t["cruza_T0"]]
        grupos = [g for g in limpio[col_grupo].unique() if g not in excluir]
        for etiqueta, col in metricas:
            piv = limpio.pivot_table(index=["t_centro", "bloque"], columns=col_grupo, values=col)[grupos]
            es_picp = col.startswith("picp")
            obj = (piv - nivel).abs() if es_picp else piv
            ganador = obj.idxmin(axis=1)
            for bloque in ("train", "test"):
                sel = ganador.index.get_level_values("bloque") == bloque
                sub, o = ganador[sel], obj[sel]
                mejor = o.min(axis=1).replace(0, np.nan)
                for g in grupos:
                    gana.append({"M": M, "metrica": etiqueta, "columna": col, "bloque": bloque,
                                 col_grupo: g, "n_ventanas": len(sub),
                                 "pct_ventanas_ganadas": float((sub == g).mean()) if len(sub) else np.nan,
                                 "criterio": "|valor - nominal|" if es_picp else "menor es mejor"})
                    fila = {"M": M, "metrica": etiqueta, "bloque": bloque, col_grupo: g,
                            "razon_vs_mejor": float((o[g] / mejor).mean())}
                    if col_grupo == "modelo" and "FAR" in o.columns:
                        fila["razon_vs_FAR"] = float((o[g] / o["FAR"].replace(0, np.nan)).mean())
                    rel.append(fila)
    return pd.DataFrame(gana), pd.DataFrame(rel)


def resumen_min_max(EST: Dict, fuente: str, col_grupo: str, metricas,
                    excluir: Sequence[str] = ()) -> pd.DataFrame:
    """Minimo, maximo y promedio de cada metrica sobre las ventanas, por grupo, bloque y M."""
    filas = []
    for M, e in EST.items():
        t = e[fuente]
        limpio = t[(~t["cruza_T0"]) & (~t[col_grupo].isin(excluir))]
        for etiqueta, col in metricas:
            g = limpio.groupby([col_grupo, "bloque"])[col].agg(["min", "max", "mean"])
            for (grupo, bloque), r in g.iterrows():
                filas.append({"M": M, "metrica": etiqueta, "columna": col, col_grupo: grupo,
                              "bloque": bloque, "minimo": float(r["min"]), "maximo": float(r["max"]),
                              "promedio": float(r["mean"])})
    return pd.DataFrame(filas)


def promedio_vs_M(resumen: pd.DataFrame, col_grupo: str) -> pd.DataFrame:
    """Promedio en TEST por M (filas) y (columna, grupo) (columnas), para graficar la
    comparacion contra M."""
    return (resumen[resumen.bloque == "test"]
            .pivot_table(index="M", columns=["columna", col_grupo], values="promedio"))


# ==========================================================================
# 5. CURVA A CURVA
# ==========================================================================

def origenes_extremos(e: Dict, DIS: Dict, ORIG: Dict, metrica: str, n: int, pesos_tau=None) -> Dict:
    """Los `n` origenes de TEST mejor y peor predichos por el PSBPM-FD segun `metrica`
    (una del Bloque A por origen: mae_f, rmse_f o linf_medio)."""
    clave = {"mae_f": "l1", "rmse_f": "l2", "linf_medio": "linf"}
    assert metrica in clave, f"{metrica!r} no es una del Bloque A: {list(clave)}"
    serie = normas_error_por_origen(ORIG["X_obj_ev"], e["CURVAS"]["PSBPM-FD"], DIS["grilla"],
                                    pesos_tau=pesos_tau, verificar=False)[clave[metrica]]
    idx = np.where(~ORIG["es_train"])[0]
    orden = idx[np.argsort(serie[idx])]
    ext = {"mejores": orden[:n], "peores": orden[-n:][::-1], "serie": serie}
    t = ORIG["t_orig"]
    print(f"[M={e['n_components']}] por {metrica} (PSBPM-FD, test): mejores t = "
          f"{[int(t[i]) for i in ext['mejores']]}   ·   peores t = {[int(t[i]) for i in ext['peores']]}")
    return ext


def contraste_origenes(EST: Dict, extremos: Dict, DIS: Dict, ORIG: Dict, metrica: str,
                       prefijo: int = 70) -> pd.DataFrame:
    """Ancho y cobertura por origen de cada banda en los origenes extremos
    (`<prefijo+9>_ic_contraste_origenes.csv` por M)."""
    X, t = ORIG["X_obj_ev"], ORIG["t_orig"]
    filas = []
    for M, ext in extremos.items():
        for grupo in ("mejores", "peores"):
            for i in ext[grupo]:
                for nombre, ((li, ls), _Xp) in EST[M]["BANDAS"].items():
                    dentro = (X[i] >= li[i]) & (X[i] <= ls[i])
                    filas.append({"M": M, "grupo": grupo, "t": int(t[i]), "banda": nombre,
                                  f"{metrica}_PSBPM": float(ext["serie"][i]),
                                  "mpiw": float(np.mean(ls[i] - li[i])),
                                  "picp_puntual": float(dentro.mean()),
                                  "I_t_simultaneo": float(bool(dentro.all()))})
    df = pd.DataFrame(filas)
    for M in extremos:
        df[df.M == M].drop(columns="M").to_csv(
            EST[M]["paths"]["out_report"] / f"{prefijo + 9}_ic_contraste_origenes.csv", index=False)
    return df
