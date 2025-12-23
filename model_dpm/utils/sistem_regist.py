from pathlib import Path
from datetime import datetime


def register_experiment(
    config: dict,
    run_name: str,
    experiment_type: str,
    description: str
) -> str:
    """
    Registra un experimento en experiment_registry.md y retorna el experiment_id.
    """
    # -----------------------
    # Extraer configuración
    # -----------------------
    project_cfg = config["project"]
    exp_cfg = config["experiments"]

    project_name = project_cfg["name"]
    project_version = project_cfg["version"]

    # -----------------------
    # Determinar ruta del registry
    # -----------------------
    project_root = Path(__file__).resolve().parents[1]
    registry_path = project_root / exp_cfg["registry_file"]
    
    # Asegurar que el directorio existe
    registry_path.parent.mkdir(parents=True, exist_ok=True)

    # -----------------------
    # Generar ID experimento
    # -----------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_id = f"v{project_version}_{run_name}_{timestamp}"

    # -----------------------
    # Crear archivo si no existe
    # -----------------------
    if not registry_path.exists():
        registry_path.write_text(
            "| ID | Fecha | Tipo | Proyecto | Versión | Ejecución | Descripción |\n"
            "|----|-------|------|----------|---------|-----------|-------------|\n",
            encoding="utf-8"
        )

    # -----------------------
    # Escribir nueva entrada
    # -----------------------
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    row = (
        f"| {experiment_id} | {date_str} | {experiment_type} | "
        f"{project_name} | {project_version} | {run_name} | {description} |\n"
    )

    with registry_path.open("a", encoding="utf-8") as f:
        f.write(row)

    print(f"✅ Experimento registrado: {experiment_id}")
    print(f"📁 Registry: {registry_path}")

    return experiment_id