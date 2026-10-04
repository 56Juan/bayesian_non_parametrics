"""
evaluacion_barrido.py
=====================
Evaluacion predictiva sobre el barrido en M, SIN graficos (las figuras viven en
`graphics/`). Es la logica que antes estaba repartida en el notebook `_04`:
representaciones y diseno comun, prediccion a h=1 con su banda, tablas de
ventana movil (Bloque A y B), quien gana cada ventana, resumen, y la evaluacion
puntual de los scores.

Carga de artefactos y de trazas: `convergencia_barrido.cargar_artefactos` (con
`con_test=True`, que ya cruza el contrato) y `cargar_trazas`. Aqui se agrega lo
propio de la evaluacion. `EST[M]` es el estado de cada punto; el diseno comun
(`DIS`) y la serie de origenes (`ORIG`) son diccionarios que el notebook
desempaqueta una vez.

Las diez metricas son las del marco teorico `02_02_03` y no se agregan otras.
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import pandas as pd

from ..functions_models import DataStandardizer
from ..models.pspb_fd_v3 import PropagadorFuncional, curva_media_desde_scores
from ..pipelines import (
    cargar_curvas, cargar_curvas_true, cargar_representacion,
    cargar_fpca, cargar_estandarizador,
)
from .metrics_puntual import mise
from .metrics_distribucional import (
    intervalo_muestral, winkler as winkler_scores,
    indicador_cobertura, indicador_cobertura_simultanea,
)
from .pooling import agrupar_momentos
from .rolling import ventana_movil_scores, ventana_movil_funcional

__all__ = [
    "METRICAS_A", "METRICAS_B",
    "cargar_representaciones", "serie_origenes", "predecir_barrido", "guardar_bandas",
    "tablas_ventana", "apilar_tablas", "verificar_relaciones_metricas",
    "ganancias_por_ventana", "tabla_ganadores_test", "saltos_T0", "peores_ventanas",
    "resumen_metricas", "resumen_barrido", "monitoreo_scores", "intervalos_scores",
]

# Las 10 metricas del marco teorico 02_02_03, en su orden. Declaradas en la
# celda [CONFIG] del notebook con el mismo contenido.
METRICAS_A = [("1. MAE  (L^1)", "mae_f"), ("2. RMSE (L^2)", "rmse_f"),
              ("3. E_max promedio por curva", "linf_medio"),
              ("4. E_max peor de la ventana", "linf_max")]
METRICAS_B = [("5. MPIW", "mpiw"), ("6. PICP puntual", "picp"),
              ("7. PICPB curva completa", "picp_simultaneo"), ("8. Winkler", "winkler"),
              ("9. Winkler max promedio", "winkler_max_medio"),
              ("10. Winkler max peor ventana", "winkler_max_glob")]


# ==========================================================================
# 1. REPRESENTACIONES Y DISENO COMUN
# ==========================================================================

def cargar_representaciones(EST: Dict, verbose: bool = True) -> Dict:
    """
    FPCA, estandarizador, curvas y representacion de cada punto, y el diseno
    COMUN del barrido. Exige que T, T0, n_lags, nivel, modo_residuo, objetivo,
    ventanas y configuracion MCMC coincidan, y que las curvas sean las mismas:
    si no, las diferencias entre M mezclarian el efecto de M con el de otra cosa.
    Retorna `DIS` con T, T0, N_LAGS, NIVEL, MODO_RESIDUO, OBJETIVO, VENTANAS_W,
    W_REF, grilla, X_true, X_obs y X_suav (la curva suavizada, el objetivo).
    """
    for M, e in EST.items():
        P = e["paths"]
        assert e["manifest"]["scores_scale"] == "raw_fpca_scores"
        assert e["objetivo"] == "curva_suavizada", (
            f"[M={M}] objetivo_evaluacion={e['objetivo']!r}: se evalua contra la curva "
            "suavizada (docs 03_05_00). Regenera con el _01.")
        fpca = cargar_fpca(P)
        X_obs, grilla = cargar_curvas(P)
        fr, THETA, _ = cargar_representacion(P)
        assert e["component_idx"] == list(range(fpca.M)), (
            "La propagacion funcional necesita el vector completo de scores en "
            f"orden; [M={M}] COMPONENT_IDX={e['component_idx']} y M={fpca.M}.")
        e.update({"fpca": fpca, "std": cargar_estandarizador(P, DataStandardizer),
                  "X_obs": X_obs, "X_true": cargar_curvas_true(P),
                  "X_suav": fr.reconstruct(THETA), "fr": fr, "grilla": grilla,
                  "Psi_grid": fpca.Psi_grid, "mu_grid": fpca.mu_grid,
                  "SCORES": fpca.SCORES})
        if verbose:
            print(f"M={M}  FPCA M={fpca.M} K={fpca.K}  var. explicada="
                  f"{float(fpca.meta['var_explained']):.4%}  ·  curvas {e['X_true'].shape}")

    comun = {M: (e["T"], e["T0"], e["n_lags"], e["nivel"], e["modo_residuo"], e["objetivo"],
                 tuple(e["ventanas_w"]), e["n_iter"], e["mcmc_cfg"]["nsim"],
                 e["mcmc_cfg"]["N"], e["burn"]) for M, e in EST.items()}
    assert len(set(comun.values())) == 1, (
        "El barrido en M exige T, T0, n_lags, nivel, modo_residuo, objetivo, ventanas y "
        f"configuracion MCMC identicos. Difieren: {comun}")
    M0 = next(iter(EST))
    for M, e in EST.items():
        assert np.allclose(e["X_true"], EST[M0]["X_true"]) and np.allclose(e["grilla"], EST[M0]["grilla"]), (
            f"[M={M}] las curvas verdaderas difieren de las de M={M0}: los puntos no "
            "salen de la misma simulacion.")
    e0 = EST[M0]
    DIS = {"T": e0["T"], "T0": e0["T0"], "N_LAGS": e0["n_lags"], "NIVEL": e0["nivel"],
           "MODO_RESIDUO": e0["modo_residuo"], "OBJETIVO": e0["objetivo"],
           "VENTANAS_W": e0["ventanas_w"], "W_REF": e0["ventanas_w"][len(e0["ventanas_w"]) // 2],
           "grilla": e0["grilla"], "X_true": e0["X_true"], "X_obs": e0["X_obs"],
           "X_suav": e0["X_suav"]}
    if verbose:
        print(f"\nOK {len(EST)} puntos con el mismo diseno, objetivo, MCMC y simulacion.")
        print(f"T={DIS['T']} T0={DIS['T0']} n_lags={DIS['N_LAGS']}   ventanas w={DIS['VENTANAS_W']}"
              f"   referencia w={DIS['W_REF']}   nivel={DIS['NIVEL']:.2f}")
    return DIS


def serie_origenes(DIS: Dict) -> Dict:
    """Serie de origenes (identica en todos los M) y los objetivos alineados."""
    n_lags, T, T0 = DIS["N_LAGS"], DIS["T"], DIS["T0"]
    t_orig = np.arange(n_lags + 1, T + 1)          # tiempo del experimento, base-1
    return {"t_orig": t_orig, "n_orig": len(t_orig), "T0_orig": T0 - n_lags,
            "es_train": t_orig <= T0,
            "X_obj_ev": DIS["X_suav"][n_lags:],     # OBJETIVO: curva suavizada
            "X_obs_ev": DIS["X_obs"][n_lags:]}      # observada, solo para las figuras


# ==========================================================================
# 2. PREDICCION A h=1 Y SU BANDA
# ==========================================================================

def _predecir_punto(e: Dict, M: int, DIS: Dict, ORIG: Dict, cache: Dict, S_POR_ITER: int,
                    S_FUNC: int, BLOQUE_ORIG: int, SEED_PRED: int) -> None:
    n_comp, n_iter, n_post, ci = e["n_components"], e["n_iter"], e["n_post"], e["component_idx"]
    n_orig, grilla, NIVEL = ORIG["n_orig"], DIS["grilla"], DIS["NIVEL"]
    dfs_full = {k: pd.concat([e["dfs_train"][k], e["dfs_test"][k]], ignore_index=True)
                for k in range(n_comp)}
    assert len(dfs_full[0]) == n_orig, f"[M={M}] {len(dfs_full[0])} != {n_orig}"

    # Momentos: ley de varianza total entre cadenas. Muestras: concatenacion (mezcla
    # de igual peso).
    Y_obs = np.column_stack([dfs_full[k].iloc[:, 0].to_numpy() for k in range(n_comp)])
    Y_hat, Y_sd = np.empty_like(Y_obs), np.empty_like(Y_obs)
    S_por_cadena = n_post * S_POR_ITER
    S_total = S_por_cadena * n_iter
    SC_draws = np.empty((S_total, n_orig, n_comp), dtype=np.float32)
    for k in range(n_comp):
        medias, sds = [], []
        for j, c in enumerate(sorted(e["models_chains"][k])):
            if (M, ci[k], c) not in cache:
                mc = e["models_chains"][k][c]
                mom = mc.momentos(dfs_full[k])
                mue = mc.muestrear(dfs_full[k], S_POR_ITER, seed=SEED_PRED + 1000 * ci[k] + c)
                cache[(M, ci[k], c)] = (mom["media"], mom["sd"], np.asarray(mue, dtype=np.float32))
            m_, s_, muestras = cache[(M, ci[k], c)]
            assert muestras.shape == (S_por_cadena, n_orig), (
                f"[M={M} k={k} c={c}] muestrear devolvio {muestras.shape}; se esperaba "
                f"({S_por_cadena}, {n_orig}).")
            medias.append(m_)
            sds.append(s_)           # PREDICTIVA (v3), no la del centro
            SC_draws[j * S_por_cadena:(j + 1) * S_por_cadena, :, k] = muestras
        Y_hat[:, k], Y_sd[:, k] = agrupar_momentos(np.column_stack(medias), np.column_stack(sds))
    li_s, ls_s = intervalo_muestral(SC_draws, nivel=NIVEL)            # (n_orig, M)

    # Propagacion a la curva por tramos de origenes. Sin residuo es determinista.
    assert DIS["MODO_RESIDUO"] == "ninguno", "la propagacion por tramos supone modo_residuo='ninguno'."
    paso_thin = max(1, S_total // S_FUNC)
    prop = PropagadorFuncional(e["Psi_grid"], e["mu_grid"], estandarizador=e["std"],
                               modo_residuo=DIS["MODO_RESIDUO"])
    SC_thin = SC_draws[::paso_thin]
    li_f, ls_f = np.empty((n_orig, len(grilla))), np.empty((n_orig, len(grilla)))
    for i0 in range(0, n_orig, BLOQUE_ORIG):
        sl = slice(i0, i0 + BLOQUE_ORIG)
        Xd = prop.curvas_desde_scores(SC_thin[:, sl], seed=SEED_PRED)
        li_f[sl] = np.quantile(Xd, (1 - NIVEL) / 2, axis=0)
        ls_f[sl] = np.quantile(Xd, 1 - (1 - NIVEL) / 2, axis=0)
    # Prediccion puntual = media ANALITICA (lineal en los scores): con atau <= 1 la
    # predictiva tiene varianza infinita y la media muestral la arrastran unos draws.
    X_pred = curva_media_desde_scores(Y_hat, e["fpca"], e["std"])
    X_proj = e["fpca"].reconstruct(e["SCORES"])[DIS["N_LAGS"]:]       # piso de la representacion
    e.update({"dfs_full": dfs_full, "Y_obs": Y_obs, "Y_hat": Y_hat, "Y_sd": Y_sd,
              "SC_draws": SC_draws, "li_s": li_s, "ls_s": ls_s, "X_pred": X_pred,
              "li_f": li_f, "ls_f": ls_f, "X_proj": X_proj, "S_total": int(S_total),
              "S_funcional": int(SC_thin.shape[0]), "paso_thin": paso_thin})
    print(f"M={M}: S={S_total} extracciones por score · funcionales {SC_thin.shape[0]} "
          f"(1 de cada {paso_thin}) · SC_draws {SC_draws.nbytes / 1e6:.0f} MB")
    print(f"       MISE del TRUNCAMIENTO FPCA (test) = "
          f"{mise(ORIG['X_obj_ev'][~ORIG['es_train']], X_proj[~ORIG['es_train']], grilla):.6f}"
          "   <- distancia del objetivo a su propia proyeccion en M")


def predecir_barrido(EST: Dict, M_OK: Sequence[int], DIS: Dict, ORIG: Dict,
                     S_POR_ITER: int = 1, S_FUNC: int = 1000, BLOQUE_ORIG: int = 200,
                     SEED_PRED: int = 20260823, max_gb: float = 2.0) -> None:
    """
    Prediccion a h=1 de cada M: momentos y extracciones por score (`Y_hat`, `Y_sd`,
    `SC_draws`, banda `li_s`/`ls_s`), curva predicha y su banda por cuantiles
    (`X_pred`, `li_f`, `ls_f`). Modifica `EST`. Solo se conserva en cache el M en
    curso (cada punto tiene sus propias trazas).
    """
    n_orig = ORIG["n_orig"]
    print(f"origenes evaluados: {n_orig}  (train {ORIG['es_train'].sum()} · "
          f"test {(~ORIG['es_train']).sum()})")
    gb = sum(EST[M]["n_post"] * S_POR_ITER * EST[M]["n_iter"] * n_orig
             * EST[M]["n_components"] * 4 for M in M_OK) / 1e9
    assert gb < max_gb, (f"SC_draws pediria {gb:.1f} GB sumando los M. Baja S_POR_ITER o "
                         "acorta M_FPCA_LIST antes de continuar.")
    print(f"SC_draws de TODO el barrido: {gb * 1e3:.0f} MB en float32")
    cache = {}
    for M in M_OK:
        for clave in [c for c in cache if c[0] != M]:
            del cache[clave]
        _predecir_punto(EST[M], M, DIS, ORIG, cache, S_POR_ITER, S_FUNC, BLOQUE_ORIG, SEED_PRED)


def guardar_bandas(EST: Dict, M_OK: Sequence[int], DIS: Dict, ORIG: Dict) -> None:
    """Banda y prediccion puntual en `predict/` (objeto de datos, no de lectura):
    el `_05` las necesita para el Winkler del PSBPM-FD sin re-muestrear."""
    for M in M_OK:
        e = EST[M]
        destino = e["paths"]["predict"] / "banda_funcional_psbp.npz"
        np.savez_compressed(
            destino, li=e["li_f"].astype(np.float32), ls=e["ls_f"].astype(np.float32),
            X_pred=e["X_pred"].astype(np.float32), t_orig=ORIG["t_orig"],
            nivel=np.array([DIS["NIVEL"]]), n_lags=np.array([DIS["N_LAGS"]]),
            T0=np.array([DIS["T0"]]), modo_residuo=np.array([DIS["MODO_RESIDUO"]]),
            objetivo=np.array([DIS["OBJETIVO"]]))
        print(f"[M={M}] banda funcional -> {destino.name}  "
              f"({destino.stat().st_size / 1e6:.1f} MB, {e['li_f'].shape})")
    print("Banda por CUANTILES de la predictiva muestral: no supone forma alguna.")


# ==========================================================================
# 3. BLOQUES A Y B SOBRE LA VENTANA MOVIL
# ==========================================================================

def tablas_ventana(e: Dict, DIS: Dict, ORIG: Dict, pesos_tau=None) -> None:
    """
    Tabla de ventana movil por ancho, contra los DOS objetivos de docs 03_05_00:
    la curva suavizada (`e["tablas_fun"]`) y X_t^(M) (`e["tablas_rep"]`). Es la
    unica vez que se calculan las metricas; todo lo demas se deriva de ellas.
    """
    def _tablas(objetivo, verboso):
        return {w: ventana_movil_funcional(
                    objetivo, e["X_pred"], DIS["grilla"], ORIG["T0_orig"], w=w,
                    li=e["li_f"], ls=e["ls_f"], t_offset=DIS["N_LAGS"], pesos_tau=pesos_tau,
                    nivel=DIS["NIVEL"], bloque_A=True, verbose=(verboso and w == DIS["W_REF"]))
                for w in DIS["VENTANAS_W"]}
    e["tablas_fun"] = _tablas(ORIG["X_obj_ev"], True)
    e["tablas_rep"] = _tablas(e["X_proj"], False)
    e["tablas_fun"][DIS["W_REF"]].to_csv(
        e["paths"]["out_report"] / f"57_ventana_funcional_w{DIS['W_REF']}.csv", index=False)


def apilar_tablas(EST: Dict, M_OK: Sequence[int], VENTANAS_W: Sequence[int],
                  columnas: Sequence[str]) -> Dict[int, pd.DataFrame]:
    """`{w: tabla}` con los M apilados en formato largo (columna `punto`), para
    superponer los M en una sola figura por ancho."""
    out = {}
    for w in VENTANAS_W:
        partes = []
        for M in M_OK:
            t = EST[M]["tablas_fun"][w].copy()
            faltan = [c for c in columnas if c not in t.columns]
            assert not faltan, f"[M={M}, w={w}] faltan columnas: {faltan}"
            t.insert(0, "punto", f"M={M}")
            partes.append(t)
        out[w] = pd.concat(partes, ignore_index=True)
        out[w].attrs["w"] = int(w)
    return out


def verificar_relaciones_metricas(EST: Dict, M_OK: Sequence[int], VENTANAS_W: Sequence[int]) -> None:
    """
    Relaciones del conjunto de metricas que no deben romperse: mae_f <= l2_medio <=
    linf_medio <= linf_max (la cadena va sobre l2_medio, no rmse_f: es media
    CUADRATICA entre origenes), picp_simultaneo <= picp y winkler <= winkler_max_medio
    <= winkler_max_glob.
    """
    for M in M_OK:
        for w in VENTANAS_W:
            t = EST[M]["tablas_fun"][w]
            tol = 1e-9
            assert (t["mae_f"] <= t["l2_medio"] + tol).all(), f"[M={M}, w={w}] mae_f > l2_medio"
            assert (t["l2_medio"] <= t["linf_medio"] + tol).all(), f"[M={M}, w={w}] l2_medio > linf_medio"
            assert (t["linf_medio"] <= t["linf_max"] + tol).all(), f"[M={M}, w={w}] linf_medio > linf_max"
            assert (t["picp_simultaneo"] <= t["picp"] + tol).all(), (
                f"[M={M}, w={w}] la cobertura simultanea supero a la puntual")
            assert (t["winkler"] <= t["winkler_max_medio"] + tol).all(), f"[M={M}, w={w}] winkler > winkler_max_medio"
            assert (t["winkler_max_medio"] <= t["winkler_max_glob"] + tol).all(), \
                f"[M={M}, w={w}] winkler_max_medio > winkler_max_glob"
    print(f"OK relaciones entre metricas verificadas en {len(M_OK)} puntos x "
          f"{len(VENTANAS_W)} anchos (L1<=L2<=Linf, PICPB<=PICP, W<=Wmax_medio<=Wmax_glob)")


def ganancias_por_ventana(EST: Dict, M_OK: Sequence[int], VENTANAS_W: Sequence[int],
                          metricas, NIVEL: float) -> pd.DataFrame:
    """
    % de ventanas ganadas por cada M, por metrica, bloque y ancho. En picp el
    objetivo es el NOMINAL, no el minimo. Exige que las ventanas de todos los M
    coincidan posicion a posicion.
    """
    filas = []
    for w in VENTANAS_W:
        ref = EST[M_OK[0]]["tablas_fun"][w]
        for M in M_OK[1:]:
            assert list(EST[M]["tablas_fun"][w]["t_centro"]) == list(ref["t_centro"]), (
                f"[M={M}, w={w}] las ventanas no estan alineadas con M={M_OK[0]}.")
        for etiqueta, col in metricas:
            piv = pd.DataFrame({M: EST[M]["tablas_fun"][w][col].to_numpy() for M in M_OK})
            es_picp = col.startswith("picp")
            obj = (piv - NIVEL).abs() if es_picp else piv
            g = pd.DataFrame({"bloque": ref["bloque"], "cruza_T0": ref["cruza_T0"],
                              "ganador": obj.idxmin(axis=1)})
            g = g[~g["cruza_T0"]]
            for bloque in ("train", "test"):
                sub = g[g.bloque == bloque]
                for M in M_OK:
                    filas.append({"w": int(w), "metrica": etiqueta, "columna": col,
                                  "criterio": "|valor - nominal|" if es_picp else "menor es mejor",
                                  "bloque": bloque, "M": M, "n_ventanas": len(sub),
                                  "pct_ventanas_ganadas": (float((sub["ganador"] == M).mean())
                                                           if len(sub) else float("nan"))})
    return pd.DataFrame(filas)


def tabla_ganadores_test(gana: pd.DataFrame) -> pd.DataFrame:
    """% de ventanas TEST ganadas por M, por metrica y ancho."""
    return (gana[gana.bloque == "test"]
            .pivot_table(index=["metrica", "w"], columns="M", values="pct_ventanas_ganadas"))


def saltos_T0(EST: Dict, M_OK: Sequence[int], VENTANAS_W: Sequence[int]) -> pd.DataFrame:
    """Salto del MISE en T0 (test / train), sin las ventanas que cruzan T0."""
    filas = []
    for M in M_OK:
        for w in VENTANAS_W:
            t = EST[M]["tablas_fun"][w]
            t = t[~t["cruza_T0"]]
            a = t.loc[t.bloque == "train", "mise"].mean()
            b = t.loc[t.bloque == "test", "mise"].mean()
            filas.append({"M": M, "w": w, "mise_train": float(a), "mise_test": float(b),
                          "salto": float(b / a)})
    return pd.DataFrame(filas)


def peores_ventanas(e: Dict, w: int, metrica: str, n: int) -> pd.DataFrame:
    """Las `n` ventanas de test (sin cruzar T0) con peor valor de `metrica`."""
    t = e["tablas_fun"][w]
    cand = t[(t["bloque"] == "test") & (~t["cruza_T0"])]
    assert metrica in cand.columns, f"{metrica!r} no esta en la tabla de ventana."
    return cand.nlargest(n, metrica)


# ==========================================================================
# 4. RESUMEN
# ==========================================================================

def resumen_metricas(e: Dict, M: int, DIS: Dict, ORIG: Dict) -> pd.DataFrame:
    """
    Minimo, maximo y promedio de cada metrica sobre la ventana de referencia,
    por bloque y por objetivo (`66_metricas_resumen.csv`), sin las ventanas que
    cruzan T0. Persiste tambien el indicador I_t por origen (`67_...csv`): el
    promedio esconde DONDE falla la banda.
    """
    W_REF, NIVEL = DIS["W_REF"], DIS["NIVEL"]
    filas = []
    for objetivo, clave in (("curva_suavizada", "tablas_fun"), ("representacion_fpca", "tablas_rep")):
        tab = e[clave][W_REF]
        lim = tab[~tab["cruza_T0"]]
        for etiqueta, col in METRICAS_A + METRICAS_B:
            for bloque in ("train", "test"):
                v = lim.loc[lim.bloque == bloque, col]
                filas.append({"objetivo": objetivo, "metrica": etiqueta, "columna": col,
                              "bloque": bloque, "minimo": float(v.min()),
                              "maximo": float(v.max()), "promedio": float(v.mean())})
        t_ = lim[lim.bloque == "test"]
        print(f"[M={M}] {objetivo:<20} PICP puntual test = {t_['picp'].mean():.4f} "
              f"(nominal {NIVEL:.2f})  ·  PICP simultaneo test = {t_['picp_simultaneo'].mean():.4f}")
    resumen = pd.DataFrame(filas).set_index(["objetivo", "metrica", "bloque"])
    resumen.to_csv(e["paths"]["out_report"] / "66_metricas_resumen.csv")
    e["resumen_df"] = resumen

    partes = []
    for objetivo, Y in (("curva_suavizada", ORIG["X_obj_ev"]), ("representacion_fpca", e["X_proj"])):
        partes.append(pd.DataFrame({
            "objetivo": objetivo, "t": ORIG["t_orig"],
            "bloque": np.where(ORIG["es_train"], "train", "test"),
            "I_t_puntual": indicador_cobertura(Y, e["li_f"], e["ls_f"]).mean(axis=1),
            "I_t_simultaneo": indicador_cobertura_simultanea(Y, e["li_f"], e["ls_f"]).astype(float)}))
    pd.concat(partes, ignore_index=True).to_csv(
        e["paths"]["out_report"] / "67_indicador_cobertura.csv", index=False)
    return resumen


def resumen_barrido(EST: Dict, M_OK: Sequence[int], path_barrido) -> pd.DataFrame:
    """Los resumenes de todos los M en una tabla (`89_resumen_por_M.csv`)."""
    partes = [EST[M]["resumen_df"].reset_index().assign(M=M) for M in M_OK]
    df = pd.concat(partes, ignore_index=True)
    df.to_csv(path_barrido / "89_resumen_por_M.csv", index=False)
    return df


# ==========================================================================
# 5. SCORES: MONITOREO UNIVARIADO E INTERVALOS
# ==========================================================================

def monitoreo_scores(e: Dict, DIS: Dict, ORIG: Dict) -> None:
    """Ventana movil de cada score (MAE, RMSE, intervalo), por ancho
    (`90_ventana_scores_w*.csv`). `verbose` solo en el ancho de referencia."""
    etiquetas = [f"FPC {i + 1}" for i in e["component_idx"]]
    e["tablas_score"] = {
        w: ventana_movil_scores(e["Y_obs"], e["Y_hat"], ORIG["T0_orig"], w=w,
                                muestras=e["SC_draws"], li=e["li_s"], ls=e["ls_s"],
                                t_offset=DIS["N_LAGS"], etiquetas=etiquetas,
                                verbose=(w == DIS["W_REF"]))
        for w in DIS["VENTANAS_W"]}
    for w, t in e["tablas_score"].items():
        t.to_csv(e["paths"]["out_report"] / f"90_ventana_scores_w{w}.csv", index=False)


def intervalos_scores(EST: Dict, M_OK: Sequence[int], DIS: Dict, ORIG: Dict,
                      path_barrido, umbral_ace: float = 0.10) -> pd.DataFrame:
    """
    PICP, ACE (= PICP - nominal), MPIW y Winkler de la banda de cada score, por
    bloque (`88_scores_intervalo_por_M.csv`, `93_...` por M). No usa
    `momentos()["sd"]`: con atau <= 1 esa cifra diverge y los cuantiles no.
    """
    NIVEL, es_train = DIS["NIVEL"], ORIG["es_train"]
    filas = []
    for M in M_OK:
        e = EST[M]
        for k in range(e["n_components"]):
            y, lo, hi = e["Y_obs"][:, k], e["li_s"][:, k], e["ls_s"][:, k]
            w_t = winkler_scores(y, lo, hi, nivel=NIVEL)
            dentro = (y >= lo) & (y <= hi)
            for bloque, m in (("train", es_train), ("test", ~es_train)):
                rmse = float(np.sqrt(((y[m] - e["Y_hat"][m, k]) ** 2).mean()))
                ancho = float((hi[m] - lo[m]).mean())
                filas.append({"M": M, "componente": f"FPC {e['component_idx'][k] + 1}",
                              "bloque": bloque, "picp": float(dentro[m].mean()),
                              "ace": float(dentro[m].mean() - NIVEL), "mpiw": ancho,
                              "winkler": float(w_t[m].mean()), "sd_banda": ancho / (2 * 1.959964),
                              "rmse": rmse,
                              # Una banda gaussiana calibrada daria ~3.92.
                              "mpiw_sobre_rmse": ancho / max(rmse, 1e-12)})
    df = pd.DataFrame(filas)
    df.to_csv(path_barrido / "88_scores_intervalo_por_M.csv", index=False)
    for M in M_OK:
        df[df.M == M].to_csv(EST[M]["paths"]["out_report"] / "93_scores_intervalo_resumen.csv",
                             index=False)
    mala = df[(df.bloque == "test") & (df.ace.abs() > umbral_ace)]
    if len(mala):
        print(f"! componentes con |ACE| > {umbral_ace} en test (miscalibracion que el "
              "promedio funcional del Bloque B esconde):")
        for _, r in mala.iterrows():
            print(f"    M={r['M']}  {r['componente']}  PICP={r['picp']:.3f}  ACE={r['ace']:+.3f}")
    else:
        print(f"OK ninguna componente se aleja mas de {umbral_ace} del nominal en test.")
    return df
