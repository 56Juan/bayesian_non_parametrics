"""
00_copiar_28.py
==================
Paso 0 de la corrida 104: construye sus artefactos POR COPIA de la corrida 28 (ya
entrenada), sin ejecutar `_01`, MATLAB ni `_03`.

Por cada M copia `raw/`, `processed/functional/` y `artefact/` del id de origen al de
destino y reescribe en todos los JSON copiados toda aparicion del id de origen por el
de destino (`experiment_id` de `dataset_config.json`, `trazas_en` de
`hyperparameters.json`, ...). NO copia `reports/` ni `predict/`: los genera la 104.

Trazas `.mat`: por defecto se copian (como en 101/102 y las 30x). Con
`--trazas-en-origen` NO se copian y `hyperparameters.json["trazas_en"]` queda
apuntando al directorio de artefactos de la corrida 28, que es el mecanismo con el
que los M menores reutilizan trazas con rezago propio (`cargar_trazas` lee
`out_artefact.parent / trazas_en`). Es lo que usa la 104: las trazas de la 28
pesan 545 MB (M = 1..6, 1 cadena) y ya estan versionadas; duplicarlas haria pesar demasiado el repo
(CLAUDE.md §8.2). El origen no se toca en ningun caso.

Aborta si algun directorio de destino ya existe con contenido, salvo `--sobrescribir`.
Termina con `verificar_contrato` por M.

Uso (desde la raiz del proyecto)
--------------------------------
    python notebooks/reales/104_real_pm25/00_copiar_28.py --trazas-en-origen
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


def copiar(base_origen: str, base_destino: str, M_list, dominio: str = "reales",
           sobrescribir: bool = False, trazas_en_origen: bool = False) -> dict:
    """
    `base_*` es el id sin el sufijo `_mNN` (p. ej. `real_pm25_poh_v02`).
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
    ignorar = shutil.ignore_patterns("*.mat") if trazas_en_origen else None
    for M, (e_o, e_d) in pares.items():
        P_o = construir_paths(e_o, dominio=dominio, project_root=RAIZ, crear=False)
        P_d = construir_paths(e_d, dominio=dominio, project_root=RAIZ, limpiar=False)
        n_arch, n_json = 0, 0
        for c in CLAVES_COPIA:
            if sobrescribir and P_d[c].exists():
                shutil.rmtree(P_d[c])
            shutil.copytree(P_o[c], P_d[c], dirs_exist_ok=True, ignore=ignorar)
            n_arch += sum(1 for p in P_d[c].rglob("*") if p.is_file())
            for js in P_d[c].rglob("*.json"):
                texto = js.read_text(encoding="utf-8")
                if base_origen not in texto:
                    continue
                d_viejo = json.loads(texto)
                d = _reemplazar(d_viejo, base_origen, base_destino)
                if trazas_en_origen and js.name == "hyperparameters.json":
                    d["trazas_en"] = d_viejo.get("trazas_en", e_o)      # las trazas se leen del origen
                js.write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")
                n_json += 1
            for p in P_d[c].rglob("*.json"):
                d = json.loads(p.read_text(encoding="utf-8"))
                if trazas_en_origen and p.name == "hyperparameters.json":
                    d = {k: v for k, v in d.items() if k != "trazas_en"}
                assert base_origen not in json.dumps(d), f"[M={M}] quedan referencias a {base_origen} en {p.name}"
        print(f"M={M:>2}  {e_o} -> {e_d}   {n_arch} archivos, {n_json} JSON reescritos")

    # 3. Contrato de cada punto de destino.
    informes = {}
    for M, (_, e_d) in pares.items():
        P_d = construir_paths(e_d, dominio=dominio, project_root=RAIZ, crear=False)
        informes[M] = verificar_contrato(P_d)
        hp = json.loads((P_d["out_artefact"] / "hyperparameters.json").read_text(encoding="utf-8"))
        trazas = P_d["out_artefact"].parent / hp.get("trazas_en", e_d)
        n_mat = len(list(trazas.glob("chain_fpc_*_iter*.mat")))
        assert n_mat > 0, f"[M={M}] sin trazas en {trazas}"
        print(f"M={M:>2}  contrato OK   trazas_en={hp.get('trazas_en', e_d)} ({n_mat} .mat)")
    return informes


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--origen", default="real_pm25_poh_v02")
    ap.add_argument("--destino", default="real_pm25_104_v02")
    ap.add_argument("--M", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6])
    ap.add_argument("--dominio", default="reales")
    ap.add_argument("--sobrescribir", action="store_true")
    ap.add_argument("--trazas-en-origen", action="store_true",
                    help="no copiar los .mat; trazas_en apunta al origen")
    a = ap.parse_args(argv)
    copiar(a.origen, a.destino, a.M, a.dominio, a.sobrescribir, a.trazas_en_origen)


if __name__ == "__main__":
    main()
