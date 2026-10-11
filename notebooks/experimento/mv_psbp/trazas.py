"""
Lectura de las trazas conjuntas (`chain_mv_iter<NN>.mat`) y predictor.

Para un predictor x (p,) la predictiva a h=1 del vector theta_w es la mezcla
    sum_h w_h(x) N_q( B_h [1; x], Sigma_h ),   w_h(x) = v_h prod_{l<h} (1 - v_l),
    v_h = Phi(alpha_h - sum_j psi_hj |x_j - Gamma_hj|).
`media` es la media analitica (sin muestreo), `muestrear` sortea (atomo, y) por
iteracion retenida. Se promedia sobre las iteraciones post-burn (con `thin`) y
sobre las cadenas: el predictor es la mezcla de las cadenas, como en v3.

Todo se calcula por BLOQUES de iteraciones (`chunk`): los pesos de S
iteraciones x n origenes x N atomos no caben en memoria con n ~ 4000.
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import numpy as np
import scipy.io as sio
from scipy.stats import norm


def ruta_traza_mv(paths: Dict, chain: int) -> Path:
    return Path(paths["out_artefact"]) / f"chain_mv_iter{chain:02d}.mat"


def leer_traza_mv(ruta) -> Dict[str, np.ndarray]:
    m = sio.loadmat(ruta)
    out = {k: np.asarray(v) for k, v in m.items() if not k.startswith("__")}
    for k in ("nsim", "burn", "N", "M", "p", "q", "n", "seed"):
        out[k] = int(np.asarray(m[k]).ravel()[0])
    out["feature_names"] = str(np.asarray(m["feature_names"]).ravel()[0]).split(",")
    # MATLAB colapsa las dimensiones singleton finales: con q = 1 se restauran
    nsim, N, q, p = out["nsim"], out["N"], out["q"], out["p"]
    out["Bout"] = np.asarray(out["Bout"]).reshape(nsim, N, q, p + 1)
    out["Sigout"] = np.asarray(out["Sigout"]).reshape(nsim, N, q, q)
    for k in ("Gammajhout", "psijhout"):
        out[k] = np.asarray(out[k]).reshape(nsim, N - 1, p)
    out["gammajhout"] = np.asarray(out["gammajhout"]).reshape(nsim, N, p)
    out["pijout"] = np.asarray(out["pijout"]).reshape(nsim, p)
    return out


def ruta_traza_sub(paths: Dict, prefijo: str, chain: int) -> Path:
    return Path(paths["out_artefact"]) / f"{prefijo}_iter{chain:02d}.mat"


class ModeloTrazaMV:
    """Predictor conjunto a partir de una o varias cadenas."""

    def __init__(self, trazas: List[Dict], thin: int = 4, chunk: int = 40):
        t0 = trazas[0]
        self.q, self.p, self.N, self.burn, self.nsim = t0["q"], t0["p"], t0["N"], t0["burn"], t0["nsim"]
        self.feature_names = t0["feature_names"]
        self.thin, self.chunk = thin, chunk
        self.iters = np.arange(self.burn, self.nsim, thin)
        self.B = np.concatenate([t["Bout"][self.iters].astype(float) for t in trazas])
        self.Sig = np.concatenate([t["Sigout"][self.iters].astype(float) for t in trazas])
        self.alpha = np.concatenate([t["alphahout"][self.iters].astype(float) for t in trazas])
        self.Gam = np.concatenate([t["Gammajhout"][self.iters].astype(float) for t in trazas])
        self.psi = np.concatenate([t["psijhout"][self.iters].astype(float) for t in trazas])
        self.n_it = self.B.shape[0]

    @classmethod
    def desde_paths(cls, paths: Dict, n_chains: int, thin: int = 4) -> "ModeloTrazaMV":
        return cls([leer_traza_mv(ruta_traza_mv(paths, c)) for c in range(1, n_chains + 1)], thin=thin)

    # ------------------------------------------------------------ por bloques
    def _bloques(self) -> Iterator[slice]:
        for a in range(0, self.n_it, self.chunk):
            yield slice(a, min(a + self.chunk, self.n_it))

    def _pesos_bloque(self, X: np.ndarray, s: slice) -> np.ndarray:
        """w (S_b, n, N) para las iteraciones del bloque, sin materializar (S, n, N, p)."""
        psi, Gam, alpha = self.psi[s], self.Gam[s], self.alpha[s]
        D = np.zeros((psi.shape[0], len(X), self.N - 1))
        for j in range(self.p):
            D += psi[:, None, :, j] * np.abs(X[None, :, None, j] - Gam[:, None, :, j])
        v = np.clip(norm.cdf(alpha[:, None, :] - D), 1e-12, 1 - 1e-12)
        resto = np.cumprod(1 - v, axis=2)
        return np.concatenate([v[:, :, :1], v[:, :, 1:] * resto[:, :, :-1], resto[:, :, -1:]], axis=2)

    def _media_atomos_bloque(self, X1: np.ndarray, s: slice) -> np.ndarray:
        return np.einsum("shqk,nk->snhq", self.B[s], X1)                 # (S_b, n, N, q)

    # ----------------------------------------------------------------- API
    def pesos_medios(self, X: np.ndarray) -> np.ndarray:
        """Pesos promediados sobre iteraciones y cadenas, (n, N)."""
        X = np.atleast_2d(np.asarray(X, float)); acc = np.zeros((len(X), self.N))
        for s in self._bloques():
            acc += self._pesos_bloque(X, s).sum(0)
        return acc / self.n_it

    def entropia_media(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, float)); acc = np.zeros(len(X))
        for s in self._bloques():
            w = self._pesos_bloque(X, s); acc += (-(w * np.log(w + 1e-300)).sum(2)).sum(0)
        return acc / self.n_it

    def media(self, X: np.ndarray) -> np.ndarray:
        """Media analitica de la predictiva (n, q)."""
        X = np.atleast_2d(np.asarray(X, float)); X1 = np.column_stack([np.ones(len(X)), X])
        acc = np.zeros((len(X), self.q))
        for s in self._bloques():
            acc += np.einsum("snh,snhq->nq", self._pesos_bloque(X, s), self._media_atomos_bloque(X1, s))
        return acc / self.n_it

    def muestrear(self, X: np.ndarray, por_iteracion: int = 1, seed: Optional[int] = None) -> np.ndarray:
        """Muestras de la predictiva (S, n, q): por iteracion retenida sortea el atomo
        con w_h(x) y luego y ~ N(B_h x, Sigma_h)."""
        rng = np.random.default_rng(seed)
        X = np.atleast_2d(np.asarray(X, float)); n = len(X); X1 = np.column_stack([np.ones(n), X])
        chol = np.linalg.cholesky(self.Sig)                                  # (S, N, q, q)
        out = np.empty((self.n_it * por_iteracion, n, self.q)); r = 0
        for s in self._bloques():
            w = self._pesos_bloque(X, s); mu = self._media_atomos_bloque(X1, s); cw = np.cumsum(w, axis=2)
            for b in range(w.shape[0]):
                it = s.start + b
                for _ in range(por_iteracion):
                    h = (cw[b] < rng.random(n)[:, None]).sum(axis=1).clip(0, self.N - 1)
                    z = rng.standard_normal((n, self.q))
                    out[r] = mu[b, np.arange(n), h] + np.einsum("nqk,nk->nq", chol[it, h], z); r += 1
        return out

    def ocupacion(self, X: np.ndarray) -> np.ndarray:
        """Peso medio de cada atomo sobre los x dados, (N,)."""
        return self.pesos_medios(X).mean(0)


class ModeloBloque:
    """Predictor compuesto: bloque conjunto sobre `coords` + univariados en el resto.
    Cada submodelo recibe sus propias columnas de predictores del dataset completo; la
    predictiva de las K coordenadas se arma por concatenacion (independencia condicional
    entre submodelos dado el pasado, como en el univariado)."""

    def __init__(self, submodelos: List[Dict], K: int):
        self.subs = submodelos; self.K = K      # cada uno: {"modelo", "coords", "cols"}

    @classmethod
    def desde_paths(cls, paths: Dict, manifest: Dict, n_chains: int, thin: int = 4) -> "ModeloBloque":
        subs = []
        for sm in manifest["submodelos"]:
            trazas = [leer_traza_mv(ruta_traza_sub(paths, sm["out_prefix"], c)) for c in range(1, n_chains + 1)]
            subs.append({"nombre": sm["nombre"], "modelo": ModeloTrazaMV(trazas, thin=thin),
                         "coords": list(sm["coords"]), "cols": list(sm["predictores"])})
        return cls(subs, manifest["K"])

    def media(self, df) -> np.ndarray:
        out = np.zeros((len(df), self.K))
        for sm in self.subs:
            out[:, sm["coords"]] = sm["modelo"].media(df[sm["cols"]].to_numpy(float))
        return out

    def muestrear(self, df, por_iteracion: int = 1, seed: Optional[int] = None) -> np.ndarray:
        out = None
        for i, sm in enumerate(self.subs):
            S = sm["modelo"].muestrear(df[sm["cols"]].to_numpy(float), por_iteracion, None if seed is None else seed + i)
            if out is None:
                out = np.zeros((S.shape[0], len(df), self.K))
            out[:, :, sm["coords"]] = S[:out.shape[0]]
        return out

    @property
    def bloque(self) -> "ModeloTrazaMV":
        return self.subs[0]["modelo"]
