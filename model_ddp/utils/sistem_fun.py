import yaml
from pathlib import Path

def get_project_root():
    """
    Encuentra la raíz del proyecto buscando el archivo pyproject.toml
    """
    current = Path(__file__).resolve()
    
    # Buscar hacia arriba hasta encontrar pyproject.toml
    for parent in [current] + list(current.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    
    # Si no lo encuentra, asumir que está 2 niveles arriba de este archivo
    # (model_dpm/utils/sistem_fun.py -> model_dpm/)
    return current.parent.parent

def load_config(config_path=None):
    """
    Carga la configuración del proyecto.
    
    Args:
        config_path: Ruta al archivo de configuración. Si es None, busca en versioning/config.yaml
    """
    if config_path is None:
        project_root = get_project_root()
        config_path = project_root / "versioning" / "config.yaml"
    else:
        config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo de configuración en: {config_path}\n"
            f"Asegúrate de que existe versioning/config.yaml en la raíz del proyecto."
        )
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config

def get_data_path(config, data_type="simulation", io_type="output"):
    """
    Obtiene la ruta de datos según el tipo.
    
    Args:
        config: Diccionario de configuración
        data_type: "simulation" o "real"
        io_type: "input" o "output"
    
    Returns:
        Path: Ruta absoluta al directorio de datos
    """
    project_root = get_project_root()
    
    if data_type == "simulation":
        relative_path = config['data']['simulations'][io_type]
    else:
        relative_path = config['data']['real'][io_type]
    
    # Convertir ruta relativa a absoluta
    return project_root / relative_path.lstrip('../')

def get_report_path(config, data_type="simulation", report_type="graphics"):
    """
    Obtiene la ruta de reportes según el tipo.
    
    Args:
        config: Diccionario de configuración
        data_type: "simulation" o "real"
        report_type: "graphics" o "tables"
    
    Returns:
        Path: Ruta absoluta al directorio de reportes
    """
    project_root = get_project_root()
    
    if data_type == "simulation":
        relative_path = config['reports']['simulations'][report_type]
    else:
        relative_path = config['reports']['real'][report_type]
    
    # Convertir ruta relativa a absoluta
    return project_root / relative_path.lstrip('../')

def get_model_path(config):
    """
    Obtiene la ruta para guardar modelos.
    
    Returns:
        Path: Ruta absoluta al directorio de modelos
    """
    project_root = get_project_root()
    relative_path = config['models']['save_dir']
    return project_root / relative_path.lstrip('../')