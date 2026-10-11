"""
Rutas del experimento: misma convencion que utils/rutas.py (raw / functional /
predict / out_report / out_artefact) pero con raiz en notebooks/experimento/.
`config_paths` no hace falta en MATLAB: el driver lee `jobs_mv.json`, que
trae rutas absolutas.
"""
from __future__ import annotations
import shutil
from pathlib import Path
from typing import Dict

RAIZ_EXPERIMENTO = Path(__file__).resolve().parents[1]
RAIZ_PROYECTO = RAIZ_EXPERIMENTO.parents[1]


def experiment_id_mv(basename: str, escenario_id, replica_id: int, K: int) -> str:
    """`<basename>_<escenario>_r<NN>_K<NN>`: K (coeficientes) ocupa el lugar de M."""
    return f"{basename}_{escenario_id}_r{replica_id:02d}_K{K:02d}"


def construir_paths_mv(eid: str, dominio: str = "simulaciones", limpiar: bool = False) -> Dict[str, Path]:
    r = RAIZ_EXPERIMENTO
    paths = {
        "raw":          r / "data" / dominio / "raw" / eid,
        "functional":   r / "data" / dominio / "processed" / "functional" / eid,
        "predict":      r / "data" / dominio / "processed" / "predict" / eid,
        "out_report":   r / "reports" / dominio / eid,
        "out_artefact": r / "artefact" / dominio / eid,
    }
    if limpiar:
        for k in ("raw", "functional", "predict", "out_artefact"):
            shutil.rmtree(paths[k], ignore_errors=True)
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    paths["eid"] = eid
    return paths
