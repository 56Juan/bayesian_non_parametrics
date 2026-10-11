"""
sim_escenario_MARKOV.py
=======================
Cambio de regimen MARKOVIANO en los scores: la idea del Algoritmo A-2 (corridas 62, 65,
102) llevada al esquema de scores de la 200. En la A-2 el regimen cambiaba dentro de la
curva (la curva era un tramo de una serie escalar) y la informacion quedaba en el borde
entre curvas, repartida en muchas componentes; aqui el regimen es CONSTANTE dentro de
cada curva y vive en los J = 10 coeficientes de la base de Fourier.

    S_t in {0, 1},   P(S_t = S_{t-1}) = p,
    u_tj = m_{S_t} + phi_{S_t} (u_{t-1,j} - m_{S_{t-1}}) + sigma_{S_t} e_tj,   j = 1..J_m,

con m_0 = -m, m_1 = +m y sigma_0 < sigma_1 (regimen bajo tranquilo, alto volatil). El
regimen afecta a los J_m = 5 primeros scores (la mitad); los otros cinco son AR(1) pasivos
que no lo ven. Con `regimen_comun = True` hay un solo S_t para los cinco (los scores quedan
correlacionados en el mismo t, ~0.8); con False cada score tiene su propio S_tj
independiente, como los regimenes del TAR de la 200. El cambio no se puede anticipar (p fijo), pero el regimen vigente
se reconoce en el rezago de cualquiera de los scores afectados, de modo que con rezago
propio el gating tiene que aprender "en que regimen estoy", no "cuando cambia" (eso es la
200). La ley de xi_t dado el pasado es una mezcla de dos componentes con pesos p y 1 - p.

Score: z_tj = u_tj / sd(u), con sd(u) poblacional medida en una trayectoria larga (200k pasos,
semilla fija) porque con phi distinto por regimen no hay forma cerrada; xi_tj = sqrt(lambda_j) z_tj.

Oraculo: E[z_t | todo el pasado] por el filtro de Hamilton exacto (conoce p, m, phi, sigma),
uno por regimen. Con regimen comun usa la evidencia conjunta de los cinco scores, asi que es
una cota superior tambien para un modelo de rezago propio.
"""

from __future__ import annotations

import numpy as np

from .sim_scores_comun import AR_PASIVO, J_SCORES, generar_desde_scores, resumen_scores
from .sim_escenario_TAR import ConfigEscenarioTAR

__all__ = ["PARAMETROS_MARKOV", "ConfigEscenarioMARKOV", "generar_escenario_MARKOV",
           "resumen_escenario_MARKOV"]

# Calibracion (2026-10-10, 5 replicas, ganancia en MSE test sobre el AR(4) propio de los scores
# afectados): con phi = 0.5 (el de la A-2) el oraculo gana solo 4.5 % y un RF propio nada, porque
# la persistencia dentro del regimen ya la captura la recta. Con phi = 0 la persistencia viene solo
# del regimen y la media condicional es un escalon: oraculo +15.8 %, RF propio +7.0 %, lineal
# cruzado +3.1 %. p = 0.90 deja ~70 cambios en train y ~33 en test (duracion media ~10 curvas).
PARAMETROS_MARKOV = {"p": 0.90, "m": 1.5, "phi0": 0.0, "phi1": 0.0, "sigma0": 0.5, "sigma1": 1.0,
                     "n_afectados": 5, "regimen_comun": True}


class ConfigEscenarioMARKOV(ConfigEscenarioTAR):
    """Mismos defaults de observacion que el TAR (sigma_obs = 0, n_rezagos = 2)."""


def _sd_poblacional(p, mS, phi, sig, n=200_000, seed=7) -> float:
    rng = np.random.default_rng(seed)
    cambia = rng.random(n) >= p
    e = rng.standard_normal(n)
    s, u, x = 0, mS[0], np.empty(n)
    for i in range(n):
        s_new = 1 - s if cambia[i] else s
        u = mS[s_new] + phi[s_new] * (u - mS[s]) + sig[s_new] * e[i]
        x[i], s = u, s_new
    return float(x[1000:].std())


def _simulador(par: dict):
    p, m = par["p"], par["m"]
    phi = np.array([par["phi0"], par["phi1"]])
    sig = np.array([par["sigma0"], par["sigma1"]])
    mS = np.array([-m, m])
    Jm = int(par["n_afectados"])
    P = np.array([[p, 1 - p], [1 - p, p]])
    assert np.all(np.abs(phi) < 1), "phi fuera de (-1, 1)."
    sd_u = _sd_poblacional(p, mS, phi, sig)

    grupos = [list(range(Jm))] if par["regimen_comun"] else [[j] for j in range(Jm)]

    def simular(T: int, burn_in: int, rng: np.random.Generator) -> dict:
        J, G = J_SCORES, len(grupos)
        sig_p = np.sqrt(1.0 - AR_PASIVO ** 2)
        n = int(burn_in) + int(T)
        z = np.zeros((n, J)); ora = np.zeros((n, J))
        S = np.zeros((n, G), dtype=int); prob = np.zeros((n, G))
        s = rng.integers(2, size=G)
        u = np.empty(Jm)
        for g, idx in enumerate(grupos):
            u[idx] = mS[s[g]] + sig[s[g]] / np.sqrt(1 - phi[s[g]] ** 2) * rng.standard_normal(len(idx))
        zp = np.zeros(J - Jm)
        pi = np.full((G, 2), 0.5)                       # P(S_{t-1} | pasado), filtro de Hamilton por grupo
        for i in range(n):
            e = rng.standard_normal(J)
            cambia = rng.random(G) >= p
            u_new = np.empty(Jm)
            for g, idx in enumerate(grupos):
                ug = u[idx]
                # prediccion del oraculo antes de ver u_t
                w = pi[g][:, None] * P                  # w[a, b] = P(S_{t-1}=a, S_t=b | pasado)
                media = mS[None, :, None] + phi[None, :, None] * (ug[None, None, :] - mS[:, None, None])
                ora[i, idx] = (w[:, :, None] * media).sum((0, 1)) / sd_u
                prob[i, g] = w[:, 1].sum()              # P(S_t = alto | pasado)
                # paso del proceso
                sn = 1 - s[g] if cambia[g] else s[g]
                u_new[idx] = mS[sn] + phi[sn] * (ug - mS[s[g]]) + sig[sn] * e[idx]
                # actualizacion del filtro con u_t
                res = u_new[idx][None, None, :] - media
                ll = -0.5 * (res ** 2).sum(-1) / sig[None, :] ** 2 - len(idx) * np.log(sig)[None, :]
                ll = np.log(w + 1e-300) + ll
                post = np.exp(ll - ll.max())
                pi[g] = post.sum(0) / post.sum()
                s[g] = sn
            u = u_new
            z[i, :Jm] = u / sd_u; S[i] = s
            ora[i, Jm:] = AR_PASIVO * zp
            zp = AR_PASIVO * zp + sig_p * e[Jm:]
            z[i, Jm:] = zp
        c = int(burn_in)
        return {"z": z[c:], "oraculo": ora[c:], "regimen": S[c:], "prob_regimen_alto": prob[c:]}
    return simular


def generar_escenario_MARKOV(cfg: ConfigEscenarioMARKOV, parametros: dict | None = None):
    """R replicas del escenario de regimen markoviano en los scores."""
    par = {**PARAMETROS_MARKOV, **(parametros or {})}
    salida = generar_desde_scores(cfg, _simulador(par), "MARKOV", parametros=par)
    salida.diagnostico = resumen_escenario_MARKOV(salida)
    return salida


def resumen_escenario_MARKOV(salida) -> dict:
    """`resumen_scores` mas duracion y numero de cambios de regimen en train y en test."""
    d = resumen_scores(salida)
    S = salida.internos["regimen"]
    T0 = d["T0_referencia"]
    cambios = np.abs(np.diff(S, axis=1))                    # (R, T-1, grupos)
    d["frac_regimen_alto"] = float(S.mean())
    d["cambios_train"] = float(cambios[:, :T0 - 1].sum(1).mean())
    d["cambios_test"] = float(cambios[:, T0 - 1:].sum(1).mean())
    d["duracion_media"] = float(S.shape[1] / (cambios.sum(1).mean() + 1))
    xi = salida.internos["scores"][..., :int(salida.internos["parametros"]["n_afectados"])]
    c = np.corrcoef(xi.reshape(-1, xi.shape[-1]).T)
    d["corr_contemporanea_afectados"] = float(c[np.triu_indices_from(c, 1)].mean())
    return d
