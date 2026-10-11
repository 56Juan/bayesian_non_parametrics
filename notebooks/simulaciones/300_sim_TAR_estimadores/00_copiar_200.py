"""
00_copiar_200.py
================
Paso 0 de la corrida 300: construye sus artefactos POR COPIA de una corrida ya
entrenada, sin ejecutar `_01`, MATLAB ni `_03`.

Por cada M copia `raw/`, `processed/functional/` y `artefact/` (trazas .mat
incluidas) del id de origen al de destino y reescribe en todos los JSON copiados
toda aparicion del id de origen por el de destino (`experiment_id` de
`simulation_config.json`, `trazas_en` de `hyperparameters.json`, ...). Con rezago
propio `trazas_en` apunta al M entrenado: al reemplazar la raiz del id se conserva
esa relacion en el destino. NO copia `reports/` ni `predict/`: los genera la 300.

El origen no se toca. Aborta si algun directorio de destino ya existe con
contenido, salvo `--sobrescribir`. Termina con `verificar_contrato` por M.

Uso (desde la raiz del proyecto)
--------------------------------
    python notebooks/simulaciones/300_sim_TAR_estimadores/00_copiar_200.py
    python .../00_copiar_200.py --origen escenario_201_chungEscG_1_r01 \
        --destino escenario_301_chungEscG_1_r01 --M 1 2 3 4 5 6 7 8 9 10
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[3]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from model_psbp_fd.pipelines.artifacts import verificar_contrato  # noqa: E402
from model_psbp_fd.utils.rutas import construir_paths  # noqa: E402

CLAVES_COPIA = ("raw", "functional", "out_artefact")


def _reemplazar(obj, viejo: str, nuevo: str):
    """Reemplaza `viejo` por `nuevo` en todos los str (y claves) de un JSON."""
    if isinstance(obj, str):
        return obj.replace(viejo, nuevo)
    if isinstance(obj, list):
        return [_reemplazar(v, viejo, nuevo) for v in obj]
    if isinstance(obj, dict):
        return {_reemplazar(k, viejo, nuevo): _reemplazar(v, viejo, nuevo) for k, v in obj.items()}
    return obj


def copiar(base_origen: str, base_destino: str, M_list, dominio: str = "simulaciones",
           sobrescribir: bool = False) -> dict:
    """
    `base_*` es el id sin el sufijo `_mNN` (p. ej. `escenario_200_chungEscG_1_r01`).
    Retorna `{M: informe de verificar_contrato}`.
    """
    assert base_origen != base_destino, "origen y destino coinciden."
    M_list = sorted({int(m) for m in M_list})
    pares = {M: (f"{base_origen}_m{M:02d}", f"{base_destino}_m{M:02d}") for M in M_list}

    # 1. Todo se verifica antes de escribir nada.
    for M, (e_o, e_d) in pares.items():
        P_o = construir_paths(e_o, dominio=dominio, project_root=RAIZ, crear=False)
        P_d = construir_paths(e_d, dominio=dominio, project_root=RAIZ, crear=False)
        for c in CLAVES_COPIA:
            assert P_o[c].is_dir(), f"[M={M}] no existe el origen {P_o[c]}"
            if P_d[c].exists() and any(P_d[c].iterdir()) and not sobrescribir:
                raise FileExistsError(f"[M={M}] el destino {P_d[c]} ya tiene contenido "
                                      "(usar --sobrescribir).")

    # 2. Copia y reescritura de los JSON.
    for M, (e_o, e_d) in pares.items():
        P_o = construir_paths(e_o, dominio=dominio, project_root=RAIZ, crear=False)
        P_d = construir_paths(e_d, dominio=dominio, project_root=RAIZ, limpiar=False)
        n_arch, n_json = 0, 0
        for c in CLAVES_COPIA:
            if sobrescribir and P_d[c].exists():
                shutil.rmtree(P_d[c])
            shutil.copytree(P_o[c], P_d[c], dirs_exist_ok=True)
            n_arch += sum(1 for p in P_d[c].rglob("*") if p.is_file())
            for js in P_d[c].rglob("*.json"):
                texto = js.read_text(encoding="utf-8")
                if base_origen not in texto:
                    continue
                d = _reemplazar(json.loads(texto), base_origen, base_destino)
                js.write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")
                n_json += 1
            restos = [p.name for p in P_d[c].rglob("*.json")
                      if base_origen in p.read_text(encoding="utf-8")]
            assert not restos, f"[M={M}] quedan referencias a {base_origen} en {restos}"
        print(f"M={M:>2}  {e_o} -> {e_d}   {n_arch} archivos, {n_json} JSON reescritos")

    # 3. Contrato de cada punto de destino.
    informes = {}
    for M, (_, e_d) in pares.items():
        P_d = construir_paths(e_d, dominio=dominio, project_root=RAIZ, crear=False)
        informes[M] = verificar_contrato(P_d)
        hp = json.loads((P_d["out_artefact"] / "hyperparameters.json").read_text(encoding="utf-8"))
        trazas = P_d["out_artefact"].parent / hp.get("trazas_en", e_d)
        n_mat = len(list(trazas.glob("chain_fpc_*_iter*.mat")))
        print(f"M={M:>2}  contrato OK   trazas_en={hp.get('trazas_en', e_d)} ({n_mat} .mat)")
    return informes


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--origen", default="escenario_200_chungEscG_1_r01")
    ap.add_argument("--destino", default="escenario_300_chungEscG_1_r01")
    ap.add_argument("--M", nargs="+", type=int, default=list(range(1, 11)))
    ap.add_argument("--dominio", default="simulaciones")
    ap.add_argument("--sobrescribir", action="store_true")
    a = ap.parse_args(argv)
    copiar(a.origen, a.destino, a.M, a.dominio, a.sobrescribir)


if __name__ == "__main__":
    main()
