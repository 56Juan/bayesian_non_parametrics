"""
Genera los notebooks E_01..E_05 de cada caso (orquestacion fina sobre
mv_psbp) y los ejecuta con nbconvert. Uso:

    python -m mv_psbp.notebooks_gen generar          # escribe los .ipynb
    python -m mv_psbp.notebooks_gen ejecutar 200     # ejecuta los de un caso
"""
from __future__ import annotations
import subprocess, sys
from pathlib import Path

import nbformat as nbf

from .rutas import RAIZ_EXPERIMENTO

CASOS = {
    "200": dict(carpeta="200_TAR", tipo="sim", esc="200", eid="mv_escenario_200_chungEscG_1_r01_K10", titulo="200 — TAR"),
    "201": dict(carpeta="201_GARCH", tipo="sim", esc="201", eid="mv_escenario_201_chungEscG_1_r01_K10", titulo="201 — GARCH"),
    "202": dict(carpeta="202_MULT", tipo="sim", esc="202", eid="mv_escenario_202_chungEscX_1_r01_K10", titulo="202 — multimodalidad"),
    "real": dict(carpeta="real_eolica_DE", tipo="real", esc=None, eid="mv_real_eolica_onshore_DE_v01_r01_K12",
                 titulo="Real — eólica onshore Alemania (SMARD)"),
    "200b": dict(carpeta="200_TAR_bloque", tipo="bloque", esc="200", eid="mvb_escenario_200_chungEscG_1_r01_K10",
                 eid_mv="mv_escenario_200_chungEscG_1_r01_K10", titulo="200 — TAR · bloque conjunto ξ₁..ξ₄"),
    "201b": dict(carpeta="201_GARCH_bloque", tipo="bloque", esc="201", eid="mvb_escenario_201_chungEscG_1_r01_K10",
                 eid_mv="mv_escenario_201_chungEscG_1_r01_K10", titulo="201 — GARCH · bloque conjunto ξ₁..ξ₄"),
    "202b": dict(carpeta="202_MULT_bloque", tipo="bloque", esc="202", eid="mvb_escenario_202_chungEscX_1_r01_K10",
                 eid_mv="mv_escenario_202_chungEscX_1_r01_K10", titulo="202 — multimodalidad · bloque conjunto ξ₁..ξ₄"),
}

CABECERA = """import sys, json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image, display
RAIZ = Path.cwd().resolve().parent
sys.path.insert(0, str(RAIZ))
from mv_psbp.pipeline import *
from mv_psbp import datos, convergencia, evaluacion, comparacion
pd.set_option("display.width", 200)
"""


def _config(c):
    if c["tipo"] == "bloque":
        return f"""# [CONFIG]
ESC       = "{c['esc']}"           # escenario de la corrida viva (se lee de fuera, no se modifica)
EID       = "{c['eid']}"
EID_MV    = "{c['eid_mv']}"       # conjunto completo (K coordenadas), tercer competidor en E_05
DOMINIO   = "simulaciones"
BLOQUE    = (0, 1, 2, 3)         # coordenadas del bloque conjunto (scores activos del generador)
N_CHAINS  = {3}
LIMPIAR   = False
"""
    if c["tipo"] == "sim":
        return f"""# [CONFIG]
ESC       = "{c['esc']}"           # escenario de la corrida viva (se lee de fuera, no se modifica)
EID       = "{c['eid']}"
DOMINIO   = "simulaciones"
N_CHAINS  = {3}
LIMPIAR   = False   # True vacia data/artefact del punto (borra las trazas): solo para regenerar desde cero
"""
    return f"""# [CONFIG]
NOMBRE    = "eolica_onshore_DE"
INSUMO    = RAIZ / "data" / "reales" / "insumos" / "eolica_onshore_DE_smard.npz"
EID       = "{c['eid']}"
DOMINIO   = "reales"
N_CHAINS  = {3}
NB_MAX    = 12      # tope de funciones B-spline: p = K * L entra al gating por distancia
MCMC      = {{"nsim": 2500, "burn": 500, "N": 30, "M": 50}}   # con n = 3007 las cadenas convergen con 2500 (ver E_03)
LIMPIAR   = False
"""


def _nb(celdas):
    nb = nbf.v4.new_notebook(); nb["cells"] = []
    for tipo, src in celdas:
        nb["cells"].append(nbf.v4.new_markdown_cell(src) if tipo == "md" else nbf.v4.new_code_cell(src))
    nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    return nb


def generar() -> None:
    for k, c in CASOS.items():
        d = RAIZ_EXPERIMENTO / c["carpeta"]; d.mkdir(exist_ok=True)
        t = c["titulo"]
        # ---- E_01
        if c["tipo"] == "bloque":
            c01 = """o = paso01_bloque(ESC, bloque=BLOQUE, limpiar=LIMPIAR)
paths = o["paths"]
print(o["eid"]); print("K =", o["K"], " T0 =", o["sim"]["T0"])
for sm in o["submodelos"]:
    print(f"  {sm['nombre']:8s} respuesta {sm['respuesta']}  predictores {len(sm['predictores'])}")
"""
        elif c["tipo"] == "sim":
            c01 = """o = paso01_simulacion(ESC, limpiar=LIMPIAR)
paths = o["paths"]
print(o["eid"]); print("K =", o["K"], " T0 =", o["sim"]["T0"], " p =", o["hp"]["p"])
print("insumo:", o["sim"]["origen"])
print("blanqueo: max|L - I| =", np.abs(o["L"] - np.eye(o["K"])).max())
"""
        else:
            c01 = """o = paso01_real(NOMBRE, INSUMO, nb_max=NB_MAX, limpiar=LIMPIAR, mcmc_config=MCMC,
                meta={"fuente": "SMARD CC BY 4.0", "variable": "factor de planta eolica onshore"})
paths = o["paths"]
print(o["eid"]); print("K =", o["K"], " base =", o["fr"].n_basis, o["fr"].order, " T0 =", o["hp"]["partition"]["T0"], " p =", o["hp"]["p"])
display(o["df_gcv"].nsmallest(5, "gcv_mean"))
fig, ax = plt.subplots(1, 2, figsize=(13, 3.5))
ax[0].plot(o["df"]["t"], o["TW"][o["df"]["t"], 0], lw=0.5); ax[0].axvline(o["hp"]["partition"]["T0"], color="r", ls="--"); ax[0].set_title("theta_w[:, 1]")
ax[1].plot(o["grilla"] if "grilla" in o else np.linspace(0, 1, 96), o["fr"].reconstruct(o["THETA"][:40]).T, lw=0.6); ax[1].set_title("40 curvas suavizadas")
plt.show()
"""
        nb = _nb([("md", f"# {t} · E_01 datos y contrato conjunto\n\nRepresentación, blanqueo por la Gram, datasets de dos rezagos funcionales e hiperparámetros. Escribe el contrato que lee `matlab/psbp_fd_iteracion_mv.m`."),
                  ("code", CABECERA), ("code", _config(c)), ("code", c01),
                  ("code", """hp = o["hp"]["bloque"] if "bloque" in o["hp"] else o["hp"]   # en la variante bloque, el contrato del bloque conjunto
print(json.dumps({k: hp[k] for k in ("global", "mcmc_config", "n_iter", "seed_base", "q", "p", "partition")}, indent=1))
jobs = registrar_jobs([paths], RAIZ / "artefact" / f"jobs_{EID}.json"); print("jobs ->", jobs)""")])
        nbf.write(nb, d / "E_01_datos.ipynb")
        # ---- E_02 entrenamiento (MATLAB)
        nb = _nb([("md", f"# {t} · E_02 entrenamiento (MATLAB)\n\nLlama a `psbp_fd_iteracion_mv` con el contrato del E_01. Si las trazas ya existen no vuelve a entrenar (`FORZAR = True` para repetir)."),
                  ("code", CABECERA), ("code", _config(c)),
                  ("code", """import subprocess, shutil
FORZAR = False
paths = paths_de(EID, DOMINIO)
trazas = sorted(Path(paths["out_artefact"]).glob("chain_*_iter*.mat"))
if trazas and not FORZAR:
    print("trazas existentes, no se entrena:"); [print("  ", p.name) for p in trazas]
else:
    matlab = shutil.which("matlab") or r"C:/Program Files/MATLAB/R2026a/bin/matlab.exe"
    jobs = RAIZ / "artefact" / f"jobs_{EID}.json"
    log = RAIZ / "artefact" / f"log_{EID}.txt"
    cmd = [matlab, "-batch", f"psbp_fd_iteracion_mv('{jobs.as_posix()}', {N_CHAINS})", "-wait", "-logfile", str(log)]
    print(" ".join(cmd)); r = subprocess.run(cmd, cwd=RAIZ / "matlab", capture_output=True, text=True); print(log.read_text()[-3000:] if log.exists() else r.stdout[-3000:]); print(r.stderr[-2000:])""")])
        nbf.write(nb, d / "E_02_entrenamiento.ipynb")
        # ---- E_03
        nb = _nb([("md", f"# {t} · E_03 convergencia\n\nDiagnósticos sobre cantidades invariantes a la permutación de etiquetas (log-verosimilitud, error in-sample, átomos activos, entropía, μ, g, π_j). No toca el bloque de prueba."),
                  ("code", CABECERA), ("code", _config(c)),
                  ("code", """paths = paths_de(EID, DOMINIO)
c = convergencia.paso03(paths, N_CHAINS)
display(c["diagnosticos"].round(3)); print(c["extra"])
display(Image(filename=str(paths["out_report"] / "40_trazas.png")))
display(Image(filename=str(paths["out_report"] / "44_ocupacion.png")))"""),
                  ("code", """display(c["pip"].round(3))
pp = c["pip"].set_index("predictor")
ax = pp[["pi_j", "frac_atomos_ocupados_con_j"]].plot.bar(figsize=(12, 3.5)); ax.set_title("inclusión por predictor"); plt.show()""")])
        nbf.write(nb, d / "E_03_convergencia.ipynb")
        # ---- E_04
        nb = _nb([("md", f"# {t} · E_04 evaluación\n\nPredicción a h=1 sobre train y test, banda por cuantiles y las diez métricas sobre la ventana móvil contra la curva suavizada. Persiste `banda_funcional_psbp_mv.npz`."),
                  ("code", CABECERA), ("code", _config(c)),
                  ("code", """paths = paths_de(EID, DOMINIO)
P = evaluacion.paso04(paths, N_CHAINS, thin=4)
print("muestras de curva por origen:", P["S"])
display(P["resumen"].round(4))
display(pd.read_csv(paths["out_report"] / "56b_agregado_test.csv", index_col=0).T.round(4))"""),
                  ("code", """for f in ("55_ventana_movil.png", "57_muestra_predicciones.png", "58_ocupacion_mezcla.png"):
    display(Image(filename=str(paths["out_report"] / f)))""")])
        nbf.write(nb, d / "E_04_evaluacion.ipynb")
        # ---- E_05
        uni = ("uni = datos.cargar_univariado_vivo(ESC)\nprint('PSBPM-FD univariado de la corrida viva:', uni['origen'], ' M =', uni['M'], ' n_lags =', uni['n_lags'])"
               if c["tipo"] in ("sim", "bloque") else "uni = None   # no hay corrida univariada de esta serie")
        if c["tipo"] == "bloque":
            uni += "\n" + "extras = {'PSBPM-FD conjunto K': comparacion.cargar_banda_experimento(paths_de(EID_MV, DOMINIO))}"
            llamada = 'R = comparacion.paso05(paths, uni, extras=extras, etiqueta_mv="PSBPM-FD bloque")'
        else:
            llamada = "R = comparacion.paso05(paths, uni)"
        nb = _nb([("md", f"# {t} · E_05 comparación\n\nPSBPM-FD " + ("bloque" if c["tipo"] == "bloque" else "conjunto") + " contra el FAR(2) (kn por hold-out, coeficientes blanqueados, igual que el `_05`)" + (" y contra el PSBPM-FD univariado de la corrida viva (su `banda_funcional_psbp.npz`, leída de fuera sin modificarla)" if c["tipo"] in ("sim", "bloque") else "") + (" y contra el conjunto completo de este experimento." if c["tipo"] == "bloque" else ".") + " Bloque A con los modelos, Bloque B con las bandas."),
                  ("code", CABECERA), ("code", _config(c)),
                  ("code", f"""paths = paths_de(EID, DOMINIO)
{uni}
{llamada}
print(R["info"])"""),
                  ("code", """print("Bloque A (test): razón contra el FAR y contra el mejor, por métrica")
r = R["razones_A"]; display(r[r.bloque == "test"].round(3))
g = R["ganadores_A"]; display(g[g.bloque == "test"][["metrica", "modelo", "n_ventanas", "pct_ventanas_ganadas"]].round(3))
display(Image(filename=str(paths["out_report"] / "76_ganador_bloqueA.png")))"""),
                  ("code", """print("Bloque B (test): bandas")
rb = R["razones_B"]; display(rb[rb.bloque == "test"].round(3))
gb = R["ganadores_B"]; display(gb[gb.bloque == "test"][["metrica", "banda", "pct_ventanas_ganadas"]].round(3))
display(pd.read_csv(paths["out_report"] / "99_resumen_bloqueB_min_max_prom.csv").query("bloque == 'test'").round(4))
display(Image(filename=str(paths["out_report"] / "77_ganador_bloqueB.png")))
display(Image(filename=str(paths["out_report"] / "75_ventana_modelos_bandas.png")))""")])
        nbf.write(nb, d / "E_05_comparacion.ipynb")
        print("notebooks ->", d)


def ejecutar(caso: str, pasos=("E_01_datos", "E_02_entrenamiento", "E_03_convergencia", "E_04_evaluacion", "E_05_comparacion")) -> None:
    d = RAIZ_EXPERIMENTO / CASOS[caso]["carpeta"]
    for p in pasos:
        nb = d / f"{p}.ipynb"
        print("ejecutando", nb.relative_to(RAIZ_EXPERIMENTO))
        r = subprocess.run([sys.executable, "-m", "nbconvert", "--to", "notebook", "--execute", "--inplace",
                            "--ExecutePreprocessor.timeout=7200", str(nb)], cwd=str(d), capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stderr[-4000:]); raise SystemExit(f"fallo {nb.name}")


if __name__ == "__main__":
    if sys.argv[1] == "generar":
        generar()
    else:
        ejecutar(sys.argv[2], tuple(sys.argv[3:]) or ("E_01_datos", "E_02_entrenamiento", "E_03_convergencia", "E_04_evaluacion", "E_05_comparacion"))
