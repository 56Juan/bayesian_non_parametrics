"""Exploracion de los datasets de la tesis de Katerin (Cuadro 4.1) donde el FAR(1) rindio mal.

Por dataset: curva diaria de 24 h -> B-spline (10, 3) -> FPCA_L2 sobre train -> FAR(p)
con la receta de `comparacion_barrido.ajustar_far` (blanqueo por la Gram, kn por far.cv,
pesos="conteo") + diagnostico sobre los scores (solo train) de lo que un FAR no ve.
Nada de esto toca el pipeline.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

from model_psbp_fd.functions_models import FunctionalRepresentation, FPCA_L2, base_en_grilla
from model_psbp_fd.fit.far_operador import FARp, seleccionar_kn
from model_psbp_fd.fit.metrics_puntual import normas_error_por_origen
from model_psbp_fd.graphics import plot_series_componentes
from model_psbp_fd.utils import pesos_trapezoidales, safe_chol

ROOT = Path(r"C:\Users\56jua\Desktop\git_tesis\bayesian_non_parametrics-1")
RAW = ROOT / "data" / "reales" / "raw"
OUT = ROOT / "reports" / "reales" / "candidatos_katerin"
G, PROP_TRAIN, MAX_AUSENTES, NB, ORD = 24, 0.70, 12, 10, 3
P_FAR = (1, 2, 7)


# ---------------------------------------------------------------- lectores (dia x hora)
def _matriz(s: pd.Series, ini, fin) -> pd.DataFrame:
    s = s.groupby(s.index.floor("h")).mean()
    df = pd.DataFrame({"v": s.values, "dia": s.index.floor("D"), "h": s.index.hour})
    dias = pd.date_range(ini, fin, freq="D")
    return df.pivot_table(index="dia", columns="h", values="v").reindex(index=dias, columns=range(G))


def leer_traffic():
    d = pd.read_csv(RAW / "katerin_traffic" / "Metro_Interstate_Traffic_Volume.csv.gz",
                    parse_dates=["date_time"])
    s = d.set_index("date_time")["traffic_volume"] / 1000.0     # miles de vehiculos / h
    # hueco de ~10 meses en 2014-08..2015-06: se usa el tramo continuo posterior
    return _matriz(s, "2015-07-01", "2018-09-29"), "volumen (miles veh/h)"


def leer_beijing():
    fs = sorted((RAW / "katerin_beijing").rglob("PRSA_Data_*_20130301-20170228.csv"))
    est = []
    for f in fs:
        d = pd.read_csv(f)
        t = pd.to_datetime(d[["year", "month", "day", "hour"]])
        est.append(pd.Series(d["PM2.5"].values, index=t))
    s = np.log1p(pd.concat(est, axis=1).mean(axis=1))          # promedio de 12 estaciones
    return _matriz(s, "2013-03-01", "2017-02-28"), "log(1 + PM2.5) promedio 12 estaciones"


def leer_household():
    d = pd.read_csv(RAW / "katerin_household" / "household_power_consumption.txt", sep=";",
                    usecols=["Date", "Time", "Global_active_power"], na_values="?",
                    low_memory=False)
    t = pd.to_datetime(d["Date"] + " " + d["Time"], format="%d/%m/%Y %H:%M:%S")
    s = pd.Series(d["Global_active_power"].astype(float).values, index=t)
    h = s.groupby(s.index.floor("h"))
    s = h.mean().where(h.count() >= 30)                        # >= 30 min observados
    return _matriz(s.dropna(), "2006-12-17", "2010-11-25"), "potencia activa (kW)"


def leer_ett():
    d = pd.read_csv(RAW / "katerin_ett" / "ETTh1.csv", parse_dates=["date"])
    return _matriz(d.set_index("date")["OT"], "2016-07-01", "2018-06-25"), "OT (grados C)"


DATASETS = {"traffic": leer_traffic, "beijing_pm25": leer_beijing,
            "household": leer_household, "ett_oT": leer_ett}


# ---------------------------------------------------------------- diagnostico de scores
def _r2(y, yhat, ybar):
    return 1.0 - np.sum((y - yhat) ** 2) / np.sum((y - ybar) ** 2)


def diagnostico_scores(XI, T0, dow, n_comp, seed=0):
    """Solo train: ajuste en el 70 % inicial de train, validacion en el 30 % final."""
    p = 2
    Xtr = XI[:T0, :n_comp]
    Z = np.hstack([Xtr[p - l - 1:T0 - l - 1] for l in range(p)])   # rezagos 1..p de todas
    D = pd.get_dummies(dow[p:T0]).to_numpy(float)[:, 1:]
    n = Z.shape[0]
    cut = int(0.7 * n)
    filas = []
    for k in range(n_comp):
        y = Xtr[p:, k]
        ytr, yva, ybar = y[:cut], y[cut:], y[:cut].mean()

        def lin(A):
            A1 = np.column_stack([np.ones(n), A])
            b = np.linalg.lstsq(A1[:cut], ytr, rcond=None)[0]
            return A1 @ b

        f_lin = lin(Z)
        f_cal = lin(np.hstack([Z, D]))
        rf = RandomForestRegressor(300, min_samples_leaf=10, random_state=seed, n_jobs=-1)
        f_rf = rf.fit(Z[:cut], ytr).predict(Z)
        e = y - f_lin                                              # residuo lineal
        ae = np.abs(e)
        # volatilidad: |e_t| contra |e_{t-1}|, |e_{t-2}| y |rezagos|
        V = np.column_stack([np.ones(n - 2), ae[1:-1], ae[:-2], np.abs(Z[2:])])
        bv = np.linalg.lstsq(V[:cut - 2], ae[2:cut], rcond=None)[0]
        er = e[:cut]
        sk = pd.Series(er).skew()
        ku = pd.Series(er).kurt()
        filas.append({
            "k": k + 1,
            "R2_lineal": _r2(yva, f_lin[cut:], ybar),
            "R2_lineal+dia": _r2(yva, f_cal[cut:], ybar),
            "R2_RF": _r2(yva, f_rf[cut:], ybar),
            "R2_vol": _r2(ae[cut:], (V @ bv)[cut - 2:], ae[2:cut].mean()),
            "acf1_e2": pd.Series(e ** 2).autocorr(1),
            "asim_e": sk, "curt_e": ku,
            "bimodal_e": (sk ** 2 + 1) / (ku + 3),                # Sarle: > 0.555 sugiere bimodal
        })
    return pd.DataFrame(filas).set_index("k")


# ---------------------------------------------------------------- un dataset
def procesar(nombre, lector):
    out = OUT / nombre
    out.mkdir(parents=True, exist_ok=True)
    PM, variable = lector()
    X = PM.to_numpy(float)
    dias = PM.index
    T = X.shape[0]
    T0 = int(np.floor(PROP_TRAIN * T))
    grilla = np.linspace(0, 1, G)
    na = np.isnan(X).sum(1)
    interp = na > MAX_AUSENTES

    fr = FunctionalRepresentation(method="bspline", n_basis=NB, order=ORD, center=False)
    fr.fit(X[:T0], grilla)
    Xf = X.copy()
    Xf[interp] = np.nan
    TH = fr.transform(Xf, grilla)
    TH = pd.DataFrame(TH, index=dias).interpolate(method="time", limit_direction="both").to_numpy()
    XS = fr.reconstruct(TH)                                         # objetivo: curva suavizada

    Phi = base_en_grilla(fr, TH.shape[1])
    fpca = FPCA_L2().fit(TH[:T0], Phi, grilla)
    K = fpca.evals.size
    XI = (TH - fpca.mu_theta) @ (fpca.W @ fpca.B_full)
    M95 = fpca.seleccionar_M(0.95)
    ar1 = [pd.Series(XI[:T0, k]).autocorr(1) for k in range(K)]
    ar7 = [pd.Series(XI[:T0, k]).autocorr(7) for k in range(K)]

    # -- FAR(p): misma receta que ajustar_far
    L = safe_chol(np.asarray(fr.gram_, float))
    THW = TH @ L
    te = np.arange(T0, T)
    w = pesos_trapezoidales(grilla)
    preds, far_info = {}, {}
    for p in P_FAR:
        cv = seleccionar_kn(THW[:T0], kn_max=min(12, K), pesos="conteo", p=p)
        far = FARp(p=p, kn=cv.kn_L2, pesos="conteo").fit(THW[:T0])
        pw = far.predict_serie(THW)
        Xp = fr.reconstruct(np.linalg.solve(L.T, pw.T).T)           # alinea con X[p:]
        preds[f"FAR({p})"] = np.vstack([np.full((p, G), np.nan), Xp])
        d = far.diagnostico_kn()
        far_info[f"FAR({p})"] = {"kn": int(cv.kn_L2), "radio_espectral": float(d["radio_espectral"])}
    # referencias de diagnostico (no son competidores del pipeline)
    preds["persistencia"] = np.vstack([np.full(G, np.nan), XS[:-1]])
    preds["semanal (t-7)"] = np.vstack([np.full((7, G), np.nan), XS[:-7]])
    preds["media train"] = np.tile(XS[:T0].mean(0), (T, 1))

    filas = []
    for mod, P in preds.items():
        nm = normas_error_por_origen(XS[te], P[te], grilla, verificar=True)
        ise = ((XS[te] - P[te]) ** 2) @ w
        filas.append({"modelo": mod, "mae_f": nm["l1"].mean(), "rmse_f": np.sqrt(ise.mean()),
                      "linf_medio": nm["linf"].mean(), "linf_max": nm["linf"].max()})
    tabla = pd.DataFrame(filas).set_index("modelo")
    tabla["mae_rel_FAR1"] = tabla["mae_f"] / tabla.loc["FAR(1)", "mae_f"]

    n_diag = int(min(max(M95, 2), 4))
    diag = diagnostico_scores(XI, T0, dias.dayofweek.to_numpy(), n_diag)

    # -- persistencia de salidas
    pd.DataFrame(TH, index=dias, columns=[f"theta_{j+1}" for j in range(K)]).to_csv(out / "theta.csv")
    pd.DataFrame(XI, index=dias, columns=[f"xi_{j+1}" for j in range(K)]).to_csv(out / "scores.csv")
    tabla.to_csv(out / "far_test.csv")
    diag.to_csv(out / "diagnostico_scores_train.csv")
    fpca_df = pd.DataFrame({"autovalor": fpca.evals, "var_ratio": fpca.var_ratio,
                            "var_acum": fpca.var_cum, "acf1": ar1, "acf7": ar7},
                           index=pd.RangeIndex(1, K + 1, name="k"))
    fpca_df.to_csv(out / "fpca.csv")

    fig = plot_series_componentes(TH, T0=T0, simbolo=r"\theta", ncols=2,
                                  titulo=f"{nombre} · coeficientes B-spline ({NB}, {ORD})")
    fig.savefig(out / "01_coeficientes.png", dpi=110, bbox_inches="tight"); plt.close(fig)
    nm_ = max(M95, 4)
    fig = plot_series_componentes(XI[:, :nm_], T0=T0, simbolo=r"\xi", color="C1", ncols=2,
                                  titulo=f"{nombre} · scores FPCA 1-{nm_} (M95 = {M95})")
    fig.savefig(out / "02_scores.png", dpi=110, bbox_inches="tight"); plt.close(fig)

    # curvas: media por dia de semana y una semana de test con FAR(1)/FAR(7)
    fig, ax = plt.subplots(1, 2, figsize=(13, 3.8))
    for d_ in range(7):
        ax[0].plot(grilla * 23, XS[:T0][dias[:T0].dayofweek == d_].mean(0),
                   label="LMXJVSD"[d_], lw=1.4)
    ax[0].set_title("media de train por dia de semana"); ax[0].legend(ncol=7, fontsize=7)
    ax[0].set_xlabel("hora"); ax[0].set_ylabel(variable)
    i0 = T0 + 14
    hh = np.arange(7 * G)
    ax[1].plot(hh, XS[i0:i0 + 7].ravel(), "k", lw=1.6, label="curva suavizada")
    for mod, c in (("FAR(1)", "C3"), ("FAR(7)", "C2")):
        ax[1].plot(hh, preds[mod][i0:i0 + 7].ravel(), c, lw=1.1, label=mod)
    for j in range(1, 7):
        ax[1].axvline(j * G, color="0.7", lw=0.6)
    ax[1].set_title(f"7 dias de test desde {dias[i0].date()} ({dias[i0].day_name()})")
    ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "03_curvas_far.png", dpi=110); plt.close(fig)

    meta = {"dataset": nombre, "variable": variable, "T": int(T), "T0": int(T0),
            "desde": str(dias[0].date()), "hasta": str(dias[-1].date()),
            "horas_ausentes": int(np.isnan(X).sum()), "dias_interpolados": int(interp.sum()),
            "K": int(K), "M95": int(M95), "far": far_info}
    (out / "resumen.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return meta, fpca_df, tabla, diag


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    sel = sys.argv[1:] or list(DATASETS)
    for nombre in sel:
        meta, fpca_df, tabla, diag = procesar(nombre, DATASETS[nombre])
        print("\n" + "=" * 90 + f"\n{nombre}: {json.dumps(meta, ensure_ascii=False)}")
        print(fpca_df.head(6).round(4).to_string())
        print(tabla.round(4).to_string())
        print(diag.round(3).to_string())
