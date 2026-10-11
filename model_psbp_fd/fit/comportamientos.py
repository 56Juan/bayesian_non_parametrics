"""
comportamientos.py
==================
Comportamientos predictivos (docs/03 Modelo.tex §03_06): por origen, las S
extracciones de scores Z_t (S, M) se agrupan por k-medias y el numero de grupos
se decide con tres reglas,

    (0) valle        con K = 2, la densidad (KDE gaussiana, Scott) de la proyeccion
                     sobre el eje entre centroides baja de VALLE x el menor de los
                     dos valores en los centroides; si no, K = 1.
    (1) crecimiento  se acepta K + 1 si R^2_{K+1} >= R^2_K (K + 1) / K, y el barrido
                     se detiene en el primer K que no lo cumple.
    (2) masa         todo grupo reune al menos alpha = 1 - nivel de las extracciones;
                     la particion que produce un grupo menor se descarta.

La distancia euclidea entre scores es la L^2 entre curvas (autofunciones
ortonormales), asi que no se estandariza. Con atau <= 1 la predictiva tiene colas
no acotadas: antes de agrupar, cada score se recorta a sus cuantiles
CUANTILES_RECORTE (winsorizado, conserva S). Las curvas de cada grupo y sus bandas
condicionales se calculan con las extracciones SIN recortar. Nada usa el indice de
atomo h: las extracciones ya mezclan los atomos dentro de cada iteracion.

Las dos constantes de las reglas (VALLE y alpha) viven aqui; alpha sale del nivel
de la banda.
"""

from __future__ import annotations

import time
from typing import Callable, Dict, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from sklearn.cluster import KMeans

from ..models.pspb_fd_v3 import curva_media_desde_scores
from .metrics_distribucional import winkler, indicador_cobertura_simultanea
from .metrics_puntual import pesos_normalizados

__all__ = [
    "VALLE", "K_MAX", "N_INIT", "SEMILLA", "CUANTILES_RECORTE",
    "debe_recortar", "recortar", "razon_valle", "numero_comportamientos",
    "comportamientos_origenes", "catalogo_global", "validar_observado",
    "resumen_validacion", "validar_regimen", "resumen_regimen",
    "guardar_comportamientos", "cargar_comportamientos",
    "comportamientos_barrido", "catalogo_barrido", "regimen_barrido",
]

VALLE = 0.5                       # cociente de valle (docs eq. valle)
K_MAX = 6                         # tope del barrido en K
N_INIT = 10                       # inicios de k-medias
SEMILLA = 20261010
CUANTILES_RECORTE = (0.005, 0.995)


# ==========================================================================
# 1. REGLAS SOBRE UNA MUESTRA
# ==========================================================================

def debe_recortar(hp_json: Dict) -> bool:
    """True si alguna componente tiene atau <= 1 (`hyperparameters.json`)."""
    return min(float(it["hyperparams"]["atau"]) for it in hp_json["hyperparams_list"]) <= 1.0


def recortar(Z: np.ndarray, q: Sequence[float] = CUANTILES_RECORTE) -> np.ndarray:
    """Winsoriza cada score (ultimo eje) a sus cuantiles `q` sobre el eje 0."""
    lo, hi = np.quantile(Z, q, axis=0)
    return np.clip(Z, lo, hi)


def razon_valle(Z: np.ndarray, c1: np.ndarray, c2: np.ndarray, w=None, n_grid: int = 101) -> float:
    """min f / min(f(a), f(b)) sobre [a, b], con f la KDE (Scott) de la proyeccion
    de Z sobre el eje que une los centroides. 1 si los centroides coinciden."""
    d = np.asarray(c2, float) - np.asarray(c1, float)
    nd = float(np.linalg.norm(d))
    if nd < 1e-12:
        return 1.0
    u = d / nd
    v = Z @ u
    if float(np.std(v)) < 1e-12:
        return 1.0
    f = gaussian_kde(v, bw_method="scott", weights=w)(np.linspace(c1 @ u, c2 @ u, n_grid))
    return float(f.min() / max(min(f[0], f[-1]), 1e-300))


def _kmedias(Z, K, w, seed, n_init):
    km = KMeans(n_clusters=K, n_init=n_init, random_state=seed).fit(Z, sample_weight=w)
    return km.labels_, km.cluster_centers_, float(km.inertia_)


def numero_comportamientos(Z: np.ndarray, alpha: float, valle: float = VALLE, k_max: int = K_MAX,
                           w=None, seed: int = SEMILLA, n_init: int = N_INIT) -> Dict:
    """
    Las tres reglas sobre una muestra Z (S, M), con pesos opcionales `w` (S,).
    Retorna K, etiquetas (S,), masas (K,), centroides (K, M), r2 (K,) y la razon
    de valle de K = 2; los grupos van ordenados por masa decreciente.
    """
    Z = np.asarray(Z, dtype=float)
    S = Z.shape[0]
    wn = np.full(S, 1.0 / S) if w is None else np.asarray(w, float) / float(np.sum(w))
    centro = wn @ Z
    W1 = float(wn @ ((Z - centro) ** 2).sum(axis=1)) * (S if w is None else float(np.sum(w)))
    lab, cen, r2, rv = np.zeros(S, dtype=int), centro[None, :], [0.0], np.nan
    if S >= 4 and W1 > 1e-12:
        l2, c2, W2 = _kmedias(Z, 2, w, seed, n_init)
        rv = razon_valle(Z, c2[0], c2[1], w)
        if rv < valle and np.bincount(l2, weights=wn, minlength=2).min() >= alpha:
            lab, cen, r2 = l2, c2, [0.0, 1.0 - W2 / W1]
            for K in range(2, k_max):
                lk, ck, Wk = _kmedias(Z, K + 1, w, seed, n_init)
                r2k = 1.0 - Wk / W1
                if r2k >= r2[-1] * (K + 1) / K and np.bincount(lk, weights=wn, minlength=K + 1).min() >= alpha:
                    lab, cen, r2 = lk, ck, r2 + [r2k]
                else:
                    break
    K = cen.shape[0]
    masas = np.bincount(lab, weights=wn, minlength=K)
    orden = np.argsort(-masas, kind="stable")
    rango = np.empty(K, dtype=int)
    rango[orden] = np.arange(K)
    return {"K": K, "etiquetas": rango[lab], "masas": masas[orden], "centroides": cen[orden],
            "r2": np.asarray(r2), "valle": rv}


# ==========================================================================
# 2. POR ORIGEN, SOBRE LA PREDICTIVA DE UN PUNTO DEL BARRIDO
# ==========================================================================

def comportamientos_origenes(SC: np.ndarray, a_curva: Callable[[np.ndarray], np.ndarray],
                             nivel: float, recortar_colas: bool, mascara=None,
                             k_max: int = K_MAX, valle: float = VALLE, seed: int = SEMILLA,
                             n_init: int = N_INIT, verbose: bool = True) -> Dict[str, np.ndarray]:
    """
    Comportamientos de cada origen. `SC` (S, n, M) son las extracciones de scores;
    `a_curva` el mapa determinista scores (k, M) -> curvas (k, G). Retorna arreglos
    con relleno NaN hasta `k_max`: K (n,), masas (n, k_max), centroides
    (n, k_max, M), curvas tipicas, li_c y ls_c (n, k_max, G) (banda equi-colas de
    nivel `nivel` dentro de cada grupo), r2 (n, k_max), valle (n,) y
    `X_pred_cluster` (n, G), la curva tipica del mas probable. Los origenes fuera
    de `mascara` quedan en NaN (K = 0).
    """
    S, n, M = SC.shape
    alpha = 1.0 - float(nivel)
    G = a_curva(np.zeros((1, M))).shape[1]
    sel = np.arange(n) if mascara is None else np.flatnonzero(mascara)
    out = {"K": np.zeros(n, dtype=int), "masas": np.full((n, k_max), np.nan),
           "centroides": np.full((n, k_max, M), np.nan), "curvas": np.full((n, k_max, G), np.nan),
           "li_c": np.full((n, k_max, G), np.nan), "ls_c": np.full((n, k_max, G), np.nan),
           "r2": np.full((n, k_max), np.nan), "valle": np.full(n, np.nan),
           "X_pred_cluster": np.full((n, G), np.nan)}
    q = ((1.0 - nivel) / 2.0, 1.0 - (1.0 - nivel) / 2.0)
    for j, i in enumerate(sel):
        Z = np.asarray(SC[:, i, :], dtype=float)
        r = numero_comportamientos(recortar(Z) if recortar_colas else Z, alpha, valle, k_max,
                                   seed=seed, n_init=n_init)
        K = r["K"]
        X = a_curva(Z)
        out["K"][i], out["valle"][i] = K, r["valle"]
        out["masas"][i, :K], out["centroides"][i, :K] = r["masas"], r["centroides"]
        out["r2"][i, :K] = r["r2"]
        out["curvas"][i, :K] = a_curva(r["centroides"])
        for c in range(K):
            out["li_c"][i, c], out["ls_c"][i, c] = np.quantile(X[r["etiquetas"] == c], q, axis=0)
        out["X_pred_cluster"][i] = out["curvas"][i, 0]
        if verbose and (j + 1) % 250 == 0:
            print(f"    comportamientos: {j + 1}/{len(sel)} origenes")
    out.update({"nivel": float(nivel), "alpha": alpha, "valle_umbral": float(valle),
                "recortado": bool(recortar_colas), "mascara": np.isin(np.arange(n), sel)})
    return out


def catalogo_global(C: Dict, a_curva: Callable, mascara=None, k_max: int = K_MAX,
                    valle: float = VALLE, seed: int = SEMILLA, n_init: int = N_INIT) -> Dict:
    """
    Catalogo de comportamientos del barrido de origenes: las mismas reglas sobre
    los centroides de todos los origenes de `mascara`, ponderados por su masa
    (cada origen pesa 1 en total). Cada comportamiento local se asigna al global
    mas cercano y sus masas se suman: `prob` (n, L) es el vector de
    probabilidades de los L comportamientos globales por origen (NaN fuera).
    """
    n = C["K"].shape[0]
    sel = C["mascara"] if mascara is None else (np.asarray(mascara, bool) & C["mascara"])
    i_pt, c_pt = np.nonzero(np.isfinite(C["masas"]) & sel[:, None])
    P, w = C["centroides"][i_pt, c_pt], C["masas"][i_pt, c_pt]
    r = numero_comportamientos(P, C["alpha"], valle, k_max, w=w, seed=seed, n_init=n_init)
    L = r["K"]
    asig = np.argmin(((P[:, None, :] - r["centroides"][None]) ** 2).sum(axis=2), axis=1)
    prob = np.full((n, L), np.nan)
    prob[np.flatnonzero(sel)] = 0.0
    np.add.at(prob, (i_pt, asig), w)
    return {"K": L, "centroides": r["centroides"], "masas": r["masas"], "curvas": a_curva(r["centroides"]),
            "r2": r["r2"], "valle": r["valle"], "prob": prob}


# ==========================================================================
# 3. VALIDACION CONTRA LO OBSERVADO
# ==========================================================================

def validar_observado(C: Dict, Y_obs: np.ndarray, X_obj: np.ndarray, li_f: np.ndarray,
                      ls_f: np.ndarray, tau: np.ndarray, pesos_tau=None) -> pd.DataFrame:
    """
    Por origen de `C["mascara"]`: el comportamiento en que cae lo observado
    (`c_obs`, centroide mas cercano a los scores observados `Y_obs` (n, M): L^2
    entre curvas), su probabilidad `p_obs` contra la esperada bajo calibracion
    (`p_esperada` = sum_c p_c^2), si es el de mayor masa (`acierto`), y PICP,
    PICPB, MPIW y Winkler contra `X_obj` (n, G) de tres bandas: la condicional del
    REALIZADO (`_real`, elegida mirando lo observado: cota de lo que daria saber el
    comportamiento), la condicional del MAS PROBABLE (`_mp`, un pronostico) y la
    marginal (`_marg`, la del Bloque B).
    """
    nivel, wn = C["nivel"], pesos_normalizados(tau, pesos_tau)
    filas = []
    for i in np.flatnonzero(C["mascara"]):
        K = int(C["K"][i])
        cen, p = C["centroides"][i, :K], C["masas"][i, :K]
        c_obs = int(np.argmin(((cen - Y_obs[i, :cen.shape[1]][None]) ** 2).sum(axis=1)))
        fila = {"i": int(i), "K": K, "c_obs": c_obs, "p_obs": float(p[c_obs]),
                "p_esperada": float((p ** 2).sum()), "acierto": float(c_obs == 0)}
        for sufijo, (lo, hi) in (("real", (C["li_c"][i, c_obs], C["ls_c"][i, c_obs])),
                                 ("mp", (C["li_c"][i, 0], C["ls_c"][i, 0])),
                                 ("marg", (li_f[i], ls_f[i]))):
            y = X_obj[i]
            fila.update({f"picp_{sufijo}": float(np.mean((y >= lo) & (y <= hi))),
                         f"picpb_{sufijo}": float(indicador_cobertura_simultanea(y[None], lo[None], hi[None])[0]),
                         f"mpiw_{sufijo}": float(np.mean(hi - lo)),
                         f"winkler_{sufijo}": float(winkler(y, lo, hi, nivel=nivel) @ wn)})
        filas.append(fila)
    return pd.DataFrame(filas)


def resumen_validacion(df: pd.DataFrame, es_train: np.ndarray) -> pd.DataFrame:
    """Promedios de `validar_observado` por bloque, en todos los origenes y en los
    de K >= 2 (en K = 1 la banda condicional ES la marginal)."""
    d = df.assign(bloque=np.where(es_train[df["i"].to_numpy()], "train", "test"))
    cols = [c for c in d.columns if c not in ("i", "K", "c_obs", "bloque")]
    partes = []
    for sub, m in (("todos", np.ones(len(d), bool)), ("K>=2", d["K"].to_numpy() >= 2)):
        g = d[m].groupby("bloque")[cols].mean()
        g.insert(0, "n_origenes", d[m].groupby("bloque").size())
        partes.append(g.assign(subconjunto=sub).reset_index())
    return pd.concat(partes, ignore_index=True).set_index(["subconjunto", "bloque"])


# ==========================================================================
# 4. VALIDACION CONTRA EL REGIMEN VERDADERO (SIMULACION)
# ==========================================================================

def validar_regimen(C: Dict, idx: np.ndarray, Z_or: np.ndarray, R_or: np.ndarray,
                    R_real: np.ndarray, valle: float = VALLE, k_max: int = K_MAX,
                    seed: int = SEMILLA, n_init: int = N_INIT) -> pd.DataFrame:
    """
    Cruce con la ley VERDADERA del generador, en los origenes `idx` (n',):

    Z_or   (S', n', M) extracciones de la predictiva oraculo (ley condicional
           verdadera dado el pasado), truncadas a las M primeras componentes.
    R_or   (S', n', A) regimen de cada extraccion oraculo (scores activos visibles).
    R_real (n', A) regimen realizado.

    Columnas: `K_oraculo` (las mismas reglas sobre Z_or), `tv` (distancia de
    variacion total entre las masas del modelo y las del oraculo sobre la MISMA
    particion: cada extraccion oraculo va al centroide del modelo mas cercano),
    `pureza` (fraccion de las extracciones oraculo de cada comportamiento que
    comparten su configuracion de regimen modal, promedio ponderado),
    `c_regimen` (comportamiento al que van las extracciones oraculo con el regimen
    REALIZADO), `acierto_modelo` (c_regimen es el mas probable del modelo) y
    `acierto_oraculo` (c_regimen es el mas probable bajo el oraculo: la cota con
    la misma particion), `p_regimen_modelo` y `p_regimen_oraculo`.
    """
    A = R_or.shape[2]
    pot = 2 ** np.arange(A)
    filas = []
    for j, i in enumerate(np.asarray(idx, dtype=int)):
        assert C["mascara"][i], f"el origen {i} no tiene comportamientos calculados."
        K = int(C["K"][i])
        cen, p = C["centroides"][i, :K], C["masas"][i, :K]
        Zi = np.asarray(Z_or[:, j], float)
        k_or = numero_comportamientos(Zi, C["alpha"], valle, k_max, seed=seed, n_init=n_init)["K"]
        a = np.argmin(((Zi[:, None, :] - cen[None]) ** 2).sum(axis=2), axis=1)
        q = np.bincount(a, minlength=K) / len(a)
        cod = R_or[:, j].astype(int) @ pot
        cod_real = int(R_real[j].astype(int) @ pot)
        pureza = 0.0
        for c in range(K):
            if (a == c).any():
                pureza += q[c] * np.bincount(cod[a == c]).max() / (a == c).sum()
        m_real = cod == cod_real
        c_reg = int(np.bincount(a[m_real], minlength=K).argmax()) if m_real.any() else -1
        filas.append({"i": int(i), "K": K, "K_oraculo": int(k_or),
                      "tv": float(0.5 * np.abs(p - q).sum()), "pureza": float(pureza),
                      "c_regimen": c_reg,
                      "acierto_modelo": float(c_reg == 0) if c_reg >= 0 else np.nan,
                      "acierto_oraculo": float(int(np.argmax(q)) == c_reg) if c_reg >= 0 else np.nan,
                      "p_regimen_modelo": float(p[c_reg]) if c_reg >= 0 else np.nan,
                      "p_regimen_oraculo": float(q[c_reg]) if c_reg >= 0 else np.nan,
                      "n_B_real": int(R_real[j].sum())})
    return pd.DataFrame(filas)


def resumen_regimen(df: pd.DataFrame) -> Dict:
    """Acuerdo en K, matriz de confusion K_modelo x K_oraculo y promedios."""
    conf = pd.crosstab(df["K"], df["K_oraculo"], rownames=["K_modelo"], colnames=["K_oraculo"])
    return {"acuerdo_K": float((df["K"] == df["K_oraculo"]).mean()),
            "confusion": conf,
            "medias": df[["tv", "pureza", "acierto_modelo", "acierto_oraculo",
                          "p_regimen_modelo", "p_regimen_oraculo"]].mean()}


# ==========================================================================
# 5. PERSISTENCIA
# ==========================================================================

_CLAVES_NPZ = ("K", "masas", "centroides", "curvas", "li_c", "ls_c", "r2", "valle",
               "X_pred_cluster", "mascara")


def guardar_comportamientos(paths: Dict, C: Dict, t_orig: np.ndarray):
    """`comportamientos_psbp.npz` en `predict/`: lo que `_05` necesita sin re-agrupar."""
    from ..pipelines.artifacts import ARCHIVOS
    destino = paths["predict"] / ARCHIVOS["comportamientos"]
    np.savez_compressed(destino, t_orig=t_orig, nivel=np.array([C["nivel"]]),
                        alpha=np.array([C["alpha"]]), valle_umbral=np.array([C["valle_umbral"]]),
                        recortado=np.array([C["recortado"]]),
                        **{k: (C[k].astype(np.float32) if C[k].dtype.kind == "f" else C[k])
                           for k in _CLAVES_NPZ})
    return destino


def cargar_comportamientos(paths: Dict) -> Optional[Dict]:
    """Lee `comportamientos_psbp.npz`; None si no existe."""
    from ..pipelines.artifacts import ARCHIVOS
    p = paths["predict"] / ARCHIVOS["comportamientos"]
    if not p.exists():
        return None
    z = np.load(p, allow_pickle=False)
    C = {k: (z[k].astype(float) if z[k].dtype.kind == "f" else z[k]) for k in _CLAVES_NPZ}
    C.update({"t_orig": z["t_orig"], "nivel": float(z["nivel"][0]), "alpha": float(z["alpha"][0]),
              "valle_umbral": float(z["valle_umbral"][0]), "recortado": bool(z["recortado"][0])})
    return C


# ==========================================================================
# 6. SOBRE EL BARRIDO EN M (lo que orquesta el _04 §12)
# ==========================================================================

def _a_curva(e: Dict) -> Callable[[np.ndarray], np.ndarray]:
    """Mapa determinista scores -> curva del punto (el de los competidores)."""
    return lambda Y: curva_media_desde_scores(np.atleast_2d(Y), e["fpca"], e["std"])


def comportamientos_barrido(EST: Dict, M_OK: Sequence[int], DIS: Dict, ORIG: Dict, path_barrido,
                            solo_test: bool = False, k_max: int = K_MAX,
                            verbose: bool = True) -> pd.DataFrame:
    """
    Por M, comportamientos de cada origen sobre `e["SC_draws"]` (`e["COMP"]`), con
    recorte de colas si algun atau <= 1 (`hyperparameters.json`). Persiste
    `comportamientos_psbp.npz` (predict/) y `62_comportamientos_por_origen.csv`;
    en el barrido, `102_K_por_M.csv` (fraccion de origenes con cada K, por bloque).
    """
    filas = []
    for M in M_OK:
        e, t0 = EST[M], time.time()
        rec = debe_recortar(e["hp_json"])
        C = comportamientos_origenes(e["SC_draws"], _a_curva(e), DIS["NIVEL"], rec,
                                     ~ORIG["es_train"] if solo_test else None, k_max=k_max,
                                     verbose=verbose)
        e["COMP"] = C
        guardar_comportamientos(e["paths"], C, ORIG["t_orig"])
        sel = C["mascara"]
        K = C["K"]
        r2_K = np.array([C["r2"][i, K[i] - 1] if K[i] > 0 else np.nan for i in range(len(K))])
        tabla = pd.DataFrame({"t": ORIG["t_orig"], "bloque": np.where(ORIG["es_train"], "train", "test"),
                              "K": K, "valle": C["valle"], "r2_K": r2_K,
                              **{f"masa_{c + 1}": C["masas"][:, c] for c in range(k_max)}})[sel]
        tabla.to_csv(e["paths"]["out_report"] / "62_comportamientos_por_origen.csv", index=False)
        for bloque, m in (("train", ORIG["es_train"]), ("test", ~ORIG["es_train"])):
            Kb = K[sel & m]
            if len(Kb):
                for k in range(1, k_max + 1):
                    filas.append({"M": M, "bloque": bloque, "K": k, "n_origenes": len(Kb),
                                  "frac": float((Kb == k).mean())})
        Kt = K[sel & ~ORIG["es_train"]]
        print(f"M={M:>2}: K en test  " + "  ".join(f"K={k}: {(Kt == k).mean():.1%}"
                                                   for k in range(1, int(Kt.max()) + 1))
              + f"   (recorte {'si' if rec else 'no'}, {time.time() - t0:.0f} s)")
    out = pd.DataFrame(filas)
    out.to_csv(path_barrido / "102_K_por_M.csv", index=False)
    return out


def catalogo_barrido(EST: Dict, M_OK: Sequence[int], ORIG: Dict, path_barrido,
                     solo_test: bool = True) -> pd.DataFrame:
    """
    Por M, el catalogo global (`catalogo_global`) sobre los origenes de test (o
    todos), en `e["CATALOGO"]`; `64_catalogo_global.csv` por M (masa y centroide
    de cada comportamiento global) y `103_catalogo_por_M.csv` en el barrido.
    """
    partes = []
    for M in M_OK:
        e = EST[M]
        cat = catalogo_global(e["COMP"], _a_curva(e), ~ORIG["es_train"] if solo_test else None)
        e["CATALOGO"] = cat
        df = pd.DataFrame({"comportamiento": np.arange(1, cat["K"] + 1), "masa": cat["masas"],
                           **{f"xi_{k + 1}": cat["centroides"][:, k] for k in range(cat["centroides"].shape[1])}})
        df.to_csv(e["paths"]["out_report"] / "64_catalogo_global.csv", index=False)
        partes.append(df.assign(M=M, L=cat["K"], valle=cat["valle"]))
        print(f"M={M:>2}: catalogo global con L={cat['K']} comportamiento(s), masas "
              f"{np.round(cat['masas'], 3).tolist()}")
    out = pd.concat(partes, ignore_index=True)
    out.to_csv(path_barrido / "103_catalogo_por_M.csv", index=False)
    return out


def regimen_barrido(EST: Dict, M_OK: Sequence[int], ORIG: Dict, idx: np.ndarray,
                    Z_or: np.ndarray, R_or: np.ndarray, R_real: np.ndarray,
                    Y_real: np.ndarray, path_barrido) -> pd.DataFrame:
    """
    Por M, `validar_regimen` en los origenes `idx` con la predictiva oraculo
    `Z_or` (S', n', J), sus regimenes `R_or` (S', n', A) y los realizados `R_real`
    (n', A), truncados a las M primeras componentes y a los min(A, M) activos
    visibles. `Y_real` (n', J) son los scores verdaderos: se verifica que coincidan
    con los del modelo (mismo orden y escala). Persiste `65_validacion_regimen.csv`
    por M y `104_validacion_regimen_por_M.csv` (resumen) en el barrido.
    """
    filas = []
    for M in M_OK:
        e = EST[M]
        assert np.allclose(e["Y_obs"][idx], Y_real[:, :M], atol=1e-8), (
            f"[M={M}] los scores del modelo no son los del generador: no se puede cruzar.")
        a = min(R_or.shape[2], M)
        df = validar_regimen(e["COMP"], idx, Z_or[:, :, :M], R_or[:, :, :a], R_real[:, :a])
        df.insert(0, "t", ORIG["t_orig"][idx])
        df.to_csv(e["paths"]["out_report"] / "65_validacion_regimen.csv", index=False)
        e["val_regimen"] = df
        r = resumen_regimen(df)
        filas.append({"M": M, "activos_visibles": a, "acuerdo_K": r["acuerdo_K"],
                      "frac_K2mas_modelo": float((df["K"] >= 2).mean()),
                      "frac_K2mas_oraculo": float((df["K_oraculo"] >= 2).mean()),
                      **r["medias"].to_dict()})
        print(f"M={M:>2}: acuerdo en K {r['acuerdo_K']:.1%} · TV {r['medias']['tv']:.3f} · "
              f"pureza {r['medias']['pureza']:.3f} · acierto de regimen modelo "
              f"{r['medias']['acierto_modelo']:.1%} / oraculo {r['medias']['acierto_oraculo']:.1%}")
    out = pd.DataFrame(filas)
    out.to_csv(path_barrido / "104_validacion_regimen_por_M.csv", index=False)
    return out
