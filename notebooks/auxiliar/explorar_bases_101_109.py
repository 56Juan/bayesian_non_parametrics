"""
Exploracion (diagnostico, fuera del pipeline) de la representacion de las
curvas de las corridas 101-109: otras bases ademas de la B-spline y FPCA
directo sobre la grilla.

Para cada escenario se toman las curvas OBSERVADAS (T, G) de data/simulaciones/raw
y se comparan cuatro representaciones:
    bspline  : B-spline cubica, K por GCV
    fourier  : base de Fourier (K impar), K por GCV
    legendre : polinomios de Legendre en [0,1], K por GCV
    grilla   : FPCA directo sobre los G valores (metrica L2 trapezoidal)
En las tres primeras la FPCA se hace sobre los coeficientes con la metrica de la
Gram de la base (C W u = lambda u). Todo se ajusta SOLO con el bloque de
entrenamiento (primeras PROP_TRAIN*T curvas), como en el pipeline.

Salidas: reports/simulaciones/exploracion_bases/<id>_*.png y resumen_*.csv
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline
from scipy.linalg import eigh
from numpy.polynomial import legendre as npleg

RAIZ = Path(__file__).resolve().parents[2]
RAW = RAIZ / "data" / "simulaciones" / "raw"
OUT = RAIZ / "reports" / "simulaciones" / "exploracion_bases"
OUT.mkdir(parents=True, exist_ok=True)

PROP_TRAIN = 0.70
M_MAX = 8
K_MAX = 40
NLAGS_AR = 3
ESCENARIOS = {  # id -> (carpeta, archivo)
    101: ("escenario_101_1_r01_m01", "escenario_1"),
    102: ("escenario_102_2_r01_m01", "escenario_2"),
    103: ("escenario_103_3_r01_m01", "escenario_3"),
    104: ("escenario_104_1_r01_m02", "escenario_1"),
    105: ("escenario_105_2_r01_m02", "escenario_2"),
    106: ("escenario_106_3_r01_m02", "escenario_3"),
    107: ("escenario_107_1_r01_m03", "escenario_1"),
    108: ("escenario_108_2_r01_m03", "escenario_2"),
    109: ("escenario_109_3_r01_m03", "escenario_3"),
}
COLORES = {"bspline": "C0", "fourier": "C1", "legendre": "C2", "grilla": "k"}


# ---------------------------------------------------------------- bases
def pesos_trap(tau):
    w = np.zeros_like(tau)
    d = np.diff(tau)
    w[:-1] += d / 2
    w[1:] += d / 2
    return w


def disenio(base, tau, K):
    """Matriz (G, K) de la base evaluada en la grilla (tau en [0,1])."""
    if base == "bspline":
        grado = 3
        nint = K - grado - 1
        t = np.r_[[0] * (grado + 1), np.linspace(0, 1, nint + 2)[1:-1], [1] * (grado + 1)]
        return BSpline.design_matrix(tau, t, grado).toarray()
    if base == "fourier":
        cols = [np.ones_like(tau)]
        j = 1
        while len(cols) < K:
            cols.append(np.sqrt(2) * np.cos(2 * np.pi * j * tau))
            if len(cols) < K:
                cols.append(np.sqrt(2) * np.sin(2 * np.pi * j * tau))
            j += 1
        return np.column_stack(cols)
    if base == "legendre":
        x = 2 * tau - 1
        return np.column_stack([np.sqrt(2 * k + 1) * npleg.legval(x, [0] * k + [1]) for k in range(K)])
    raise ValueError(base)


def gcv(base, X, tau, Ks):
    """GCV medio sobre curvas, ajuste por minimos cuadrados sobre la grilla."""
    G = len(tau)
    out = []
    for K in Ks:
        B = disenio(base, tau, K)
        theta = np.linalg.lstsq(B, X.T, rcond=None)[0]
        rss = ((X.T - B @ theta) ** 2).sum(0)
        out.append(np.mean((rss / G) / (1 - K / G) ** 2))
    return np.array(out)


# ---------------------------------------------------------------- FPCA
def fpca_coef(theta, B, w, ntr, M):
    """FPCA sobre coeficientes con metrica de la Gram. Retorna scores (T, M), lambda, psi (G, M)."""
    W = B.T @ (w[:, None] * B)
    mu = theta[:ntr].mean(0)
    Tc = theta - mu
    C = np.cov(theta[:ntr].T, ddof=0)
    C = np.atleast_2d(C)
    L = np.linalg.cholesky(W)  # W = L L^T; theta_w = theta L tiene producto euclideo = L2
    lam, V = eigh(L.T @ C @ L)
    o = np.argsort(lam)[::-1]
    lam, V = lam[o], V[:, o]
    U = np.linalg.solve(L.T, V)
    M = min(M, U.shape[1])
    sc = Tc @ W @ U[:, :M]
    psi = B @ U[:, :M]
    falta = M_MAX - sc.shape[1]  # K < M_MAX: se rellena con NaN
    if falta > 0:
        sc = np.pad(sc, ((0, 0), (0, falta)), constant_values=np.nan)
        psi = np.pad(psi, ((0, 0), (0, falta)), constant_values=np.nan)
        lam = np.r_[lam, np.full(falta, np.nan)]
    return sc, lam, psi


def fpca_grilla(X, w, ntr, M):
    mu = X[:ntr].mean(0)
    Xc = X - mu
    sw = np.sqrt(w)
    _, s, Vt = np.linalg.svd(Xc[:ntr] * sw, full_matrices=False)
    lam = s ** 2 / ntr
    psi = Vt.T / sw[:, None]
    sc = (Xc * w) @ psi[:, :M]
    return sc, lam, psi[:, :M]


def alinear_signo(r, ref):
    """Invierte scores y autofuncion juntos para que X_hat no cambie."""
    for k in range(min(r["sc"].shape[1], ref.shape[1])):
        if np.isfinite(r["sc"][:, k]).all() and np.corrcoef(r["sc"][:, k], ref[:, k])[0, 1] < 0:
            r["sc"][:, k] *= -1
            r["psi"][:, k] *= -1


def acf(x, nl):
    x = x - x.mean()
    d = (x * x).sum()
    return np.array([(x[l:] * x[:-l]).sum() / d for l in range(1, nl + 1)])


def r2_ar(x, p, ntr):
    """R2 fuera de muestra de un AR(p) ajustado en train (rezago propio)."""
    Z = np.column_stack([x[p - l - 1:len(x) - l - 1] for l in range(p)])
    y = x[p:]
    Z1 = np.column_stack([np.ones(len(y)), Z])
    n = ntr - p
    b = np.linalg.lstsq(Z1[:n], y[:n], rcond=None)[0]
    e = y[n:] - Z1[n:] @ b
    return 1 - (e ** 2).sum() / ((y[n:] - y[:n].mean()) ** 2).sum()


# ---------------------------------------------------------------- escenario
def analizar(eid):
    carpeta, arch = ESCENARIOS[eid]
    z = np.load(RAW / carpeta / f"{arch}.npz", allow_pickle=True)
    X = z["observaciones"][0]
    tau = z["grilla"]
    tau = (tau - tau.min()) / (tau.max() - tau.min())
    T, G = X.shape
    ntr = int(PROP_TRAIN * T)
    w = pesos_trap(tau)

    Ks = {"bspline": np.arange(4, K_MAX + 1), "fourier": np.arange(3, K_MAX + 1, 2),
          "legendre": np.arange(2, K_MAX + 1)}
    curvas_gcv, Ksel = {}, {}
    for b, ks in Ks.items():
        g = gcv(b, X[:ntr], tau, ks)
        curvas_gcv[b] = (ks, g)
        Ksel[b] = int(ks[np.argmin(g)])

    rep = {}
    for b in Ks:
        B = disenio(b, tau, Ksel[b])
        theta = np.linalg.lstsq(B, X.T, rcond=None)[0].T
        sc, lam, psi = fpca_coef(theta, B, w, ntr, M_MAX)
        rep[b] = dict(theta=theta, B=B, sc=sc, lam=lam, psi=psi, K=Ksel[b])
    sc, lam, psi = fpca_grilla(X, w, ntr, M_MAX)
    rep["grilla"] = dict(theta=X, B=None, sc=sc, lam=lam, psi=psi, K=G)
    for b in ("bspline", "fourier", "legendre"):
        alinear_signo(rep[b], rep["grilla"]["sc"])

    # varianza de referencia: total de la curva observada en train (metrica L2)
    mu = X[:ntr].mean(0)
    vtot = (((X[:ntr] - mu) ** 2) @ w).mean()
    for r in rep.values():
        r["cum"] = np.cumsum(r["lam"][:M_MAX]) / vtot

    # ---------- figura 1: GCV y scree
    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    for b, (ks, g) in curvas_gcv.items():
        ax[0].plot(ks, g, color=COLORES[b], label=f"{b} (K*={Ksel[b]})")
        ax[0].plot(Ksel[b], g.min(), "o", color=COLORES[b])
    ax[0].set(yscale="log", xlabel="K", ylabel="GCV", title="GCV por base (train)")
    ax[0].legend()
    for b, r in rep.items():
        ax[1].plot(range(1, M_MAX + 1), r["cum"], "o-", color=COLORES[b], label=b)
    ax[1].axhline(0.95, ls=":", c="gray")
    ax[1].set(xlabel="M", ylabel="var. acumulada / var. total", title="Varianza acumulada FPCA")
    ax[1].legend()
    for b, r in rep.items():
        ax[2].semilogy(range(1, M_MAX + 1), r["lam"][:M_MAX], "o-", color=COLORES[b], label=b)
    ax[2].set(xlabel="k", ylabel="autovalor", title="Autovalores")
    fig.suptitle(f"Escenario {eid}")
    fig.tight_layout()
    fig.savefig(OUT / f"{eid}_01_gcv_varianza.png", dpi=110)
    plt.close(fig)

    # ---------- figura 2: scores xi_1..4 por representacion
    nk = 4
    fig, ax = plt.subplots(4, nk, figsize=(17, 9), sharex=True)
    for i, b in enumerate(rep):
        for k in range(nk):
            ax[i, k].plot(rep[b]["sc"][:, k], lw=.5, color=COLORES[b])
            ax[i, k].axvline(ntr, c="r", lw=.6)
            if i == 0:
                ax[i, k].set_title(f"xi_{k+1}")
            if k == 0:
                ax[i, k].set_ylabel(b)
    fig.suptitle(f"Escenario {eid} - scores por representacion (rojo: T0)")
    fig.tight_layout()
    fig.savefig(OUT / f"{eid}_02_scores.png", dpi=110)
    plt.close(fig)

    # ---------- figura 3: coeficientes (3 bases) y ACF(1) de cada coeficiente
    nc = 6
    fig, ax = plt.subplots(3, nc + 1, figsize=(20, 7))
    for i, b in enumerate(("bspline", "fourier", "legendre")):
        th = rep[b]["theta"]
        for j in range(nc):
            ax[i, j].plot(th[:, j], lw=.5, color=COLORES[b])
            ax[i, j].axvline(ntr, c="r", lw=.6)
            ax[i, j].set_title(f"{b} c_{j+1}", fontsize=8)
        a1 = [acf(th[:ntr, j], 1)[0] for j in range(th.shape[1])]
        ax[i, nc].bar(range(1, len(a1) + 1), a1, color=COLORES[b])
        ax[i, nc].set(title=f"ACF(1) por coef. (K={rep[b]['K']})", ylim=(-1, 1))
    fig.suptitle(f"Escenario {eid} - coeficientes")
    fig.tight_layout()
    fig.savefig(OUT / f"{eid}_03_coeficientes.png", dpi=110)
    plt.close(fig)

    # ---------- figura 4: |corr| de scores de cada base contra FPCA directo
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    ref = rep["grilla"]["sc"]
    for a, b in zip(ax, ("bspline", "fourier", "legendre")):
        Cm = np.abs(np.corrcoef(rep[b]["sc"].T, ref.T)[:M_MAX, M_MAX:])
        im = a.imshow(Cm, vmin=0, vmax=1, cmap="viridis")
        a.set(title=f"|corr| scores {b} (filas) vs grilla (cols)", xlabel="k grilla", ylabel=f"k {b}")
        a.set_xticks(range(M_MAX)); a.set_xticklabels(range(1, M_MAX + 1))
        a.set_yticks(range(M_MAX)); a.set_yticklabels(range(1, M_MAX + 1))
    fig.colorbar(im, ax=ax, shrink=.8)
    fig.suptitle(f"Escenario {eid}")
    fig.savefig(OUT / f"{eid}_04_corr_scores.png", dpi=110)
    plt.close(fig)

    # ---------- tabla resumen por (base, k)
    filas = []
    for b, r in rep.items():
        for k in range(M_MAX):
            s = r["sc"][:, k]
            if not np.isfinite(s).all():
                continue
            a = acf(s[:ntr], 3)
            filas.append(dict(
                escenario=eid, base=b, K=r["K"], k=k + 1, lambda_k=r["lam"][k],
                var_acum=r["cum"][k],
                sd_train=s[:ntr].std(), sd_test=s[ntr:].std(),
                acf1=a[0], acf2=a[1], acf3=a[2],
                r2_ar=r2_ar(s, NLAGS_AR, ntr),
                corr_con_grilla=abs(np.corrcoef(s, ref[:, k])[0, 1]),
            ))
    # error de representacion contra la curva observada y contra la verdadera (M=4)
    Xv = z["curvas"][0]
    err = []
    for b, r in rep.items():
        for M in (2, 4, 8):
            mu_ = X[:ntr].mean(0)
            Xr = mu_ + r["sc"][:, :M] @ r["psi"][:, :M].T
            err.append(dict(escenario=eid, base=b, M=M,
                            mise_vs_observada=float((((X - Xr) ** 2) @ w).mean()),
                            mise_vs_verdadera=float((((Xv - Xr) ** 2) @ w).mean())))
    return pd.DataFrame(filas), pd.DataFrame(err), Ksel


if __name__ == "__main__":
    tabs, errs, kk = [], [], []
    for eid in ESCENARIOS:
        print("escenario", eid, flush=True)
        t, e, ks = analizar(eid)
        tabs.append(t); errs.append(e); kk.append(dict(escenario=eid, **ks))
    pd.concat(tabs).to_csv(OUT / "resumen_scores.csv", index=False)
    pd.concat(errs).to_csv(OUT / "resumen_error_representacion.csv", index=False)
    pd.DataFrame(kk).to_csv(OUT / "resumen_K_gcv.csv", index=False)
    print(pd.DataFrame(kk).to_string(index=False))
