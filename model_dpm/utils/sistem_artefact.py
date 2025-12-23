from pathlib import Path
import pickle
from typing import Any, Dict


def get_project_root():
    """Encuentra la raíz del proyecto buscando pyproject.toml"""
    current = Path(__file__).resolve()
    for parent in [current] + list(current.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    return current.parent.parent





def load_artifact(artifact_path: Path) -> Any:
    """
    Carga un artefacto desde disco.
    
    Parameters
    ----------
    artifact_path : Path
        Ruta al archivo del artefacto
    
    Returns
    -------
    Any
        Objeto cargado
    """
    if not artifact_path.exists():
        raise FileNotFoundError(f"No se encontró el artefacto: {artifact_path}")
    
    with open(artifact_path, 'rb') as f:
        artifact = pickle.load(f)
    
    print(f"✅ Artefacto cargado desde: {artifact_path}")
    
    return artifact



def save_artifact(
    config: dict,
    experiment_id: str,
    artifact: Any,
    artifact_name: str,
    data_type: str = "simulation"
) -> Path:
    """
    Guarda un artefacto (trazas, modelo, etc.) asociado a un experimento.
    
    Parameters
    ----------
    config : dict
        Diccionario de configuración
    experiment_id : str
        ID del experimento
    artifact : Any
        Objeto a guardar (trace, modelo, etc.)
    artifact_name : str
        Nombre del artefacto (ej: 'trace', 'model')
    data_type : str
        "simulation" o "real"
    
    Returns
    -------
    Path
        Ruta donde se guardó el artefacto
    """
    project_root = get_project_root()
    
    # Mapear singular a plural para el config
    data_key = "simulations" if data_type == "simulation" else "real"
    
    # Obtener directorio de artefactos
    artifacts_dir = project_root / config['artifacts'][data_key].lstrip('../')
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    # Crear nombre de archivo
    filename = f"{experiment_id}_{artifact_name}.pkl"
    artifact_path = artifacts_dir / filename
    
    # Guardar
    with open(artifact_path, 'wb') as f:
        pickle.dump(artifact, f)
    
    print(f"✅ Artefacto '{artifact_name}' guardado")
    print(f"📁 Ruta: {artifact_path}")
    
    return artifact_path


def list_artifacts(
    config: dict,
    experiment_id: str = None,
    data_type: str = "simulation"
) -> list[Path]:
    """
    Lista todos los artefactos de un experimento o todos los disponibles.
    
    Parameters
    ----------
    config : dict
        Diccionario de configuración
    experiment_id : str, optional
        ID del experimento. Si es None, lista todos.
    data_type : str
        "simulation" o "real"
    
    Returns
    -------
    list[Path]
        Lista de rutas de artefactos
    """
    project_root = get_project_root()
    
    # Mapear singular a plural para el config
    data_key = "simulations" if data_type == "simulation" else "real"
    
    artifacts_dir = project_root / config['artifacts'][data_key].lstrip('../')
    
    if not artifacts_dir.exists():
        return []
    
    # Filtrar por experiment_id si se especifica
    if experiment_id:
        pattern = f"{experiment_id}_*.pkl"
    else:
        pattern = "*.pkl"
    
    artifacts = sorted(artifacts_dir.glob(pattern))
    
    if artifacts:
        print(f"📦 Encontrados {len(artifacts)} artefacto(s):")
        for art in artifacts:
            print(f"  • {art.name}")
    else:
        print("⚠️  No se encontraron artefactos")
    
    return artifacts