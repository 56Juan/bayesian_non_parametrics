"""
Utilidades del sistema para gestión de rutas y configuración del proyecto.

Este módulo proporciona funciones para:
- Localizar la raíz del proyecto
- Cargar configuración desde config.yaml
- Obtener rutas absolutas para datos, reportes y artefactos
- Gestionar experimentos y metadatos
"""

import yaml
from pathlib import Path
from typing import Dict, Optional, Literal
from datetime import datetime


def get_project_root() -> Path:
    """
    Encuentra la raíz del proyecto buscando el archivo pyproject.toml.
    
    Busca hacia arriba en la jerarquía de directorios desde la ubicación
    del archivo actual hasta encontrar pyproject.toml.
    
    Returns
    -------
    Path
        Ruta absoluta a la raíz del proyecto
        
    Raises
    ------
    FileNotFoundError
        Si no se encuentra pyproject.toml en ningún nivel superior
    """
    current = Path(__file__).resolve()
    
    # Buscar hacia arriba hasta encontrar pyproject.toml
    for parent in [current] + list(current.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    
    # Si no lo encuentra, intentar con el directorio dos niveles arriba
    # (model_ddp/utils/system_utils.py -> raíz del proyecto)
    fallback = current.parent.parent
    if (fallback / "pyproject.toml").exists():
        return fallback
    
    raise FileNotFoundError(
        "No se encontró pyproject.toml. "
        "Asegúrate de estar ejecutando desde dentro del proyecto."
    )


def load_config(config_path: Optional[Path] = None) -> Dict:
    """
    Carga la configuración del proyecto desde config.yaml.
    
    Parameters
    ----------
    config_path : Path, optional
        Ruta al archivo de configuración. Si es None, busca en 
        versioning/config.yaml desde la raíz del proyecto.
    
    Returns
    -------
    Dict
        Diccionario con la configuración del proyecto
        
    Raises
    ------
    FileNotFoundError
        Si no se encuentra el archivo de configuración
    yaml.YAMLError
        Si hay errores al parsear el archivo YAML
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
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error al parsear config.yaml: {e}")
    
    return config


def resolve_path(relative_path: str) -> Path:
    """
    Convierte una ruta relativa desde config.yaml a ruta absoluta.
    
    Parameters
    ----------
    relative_path : str
        Ruta relativa que puede comenzar con '../'
        
    Returns
    -------
    Path
        Ruta absoluta desde la raíz del proyecto
    """
    project_root = get_project_root()
    
    # Remover '../' del inicio y construir ruta absoluta
    clean_path = relative_path.lstrip('../')
    return project_root / clean_path


def get_data_path(
    config: Dict, 
    data_type: Literal["simulation", "real"] = "simulation",
    io_type: Literal["input", "output"] = "output"
) -> Path:
    """
    Obtiene la ruta de datos según el tipo y operación.
    
    Parameters
    ----------
    config : Dict
        Diccionario de configuración cargado
    data_type : {'simulation', 'real'}
        Tipo de datos: simulaciones o datos reales
    io_type : {'input', 'output'}
        Tipo de operación: lectura o escritura
    
    Returns
    -------
    Path
        Ruta absoluta al directorio de datos
        
    Raises
    ------
    ValueError
        Si data_type o io_type no son válidos
    KeyError
        Si la configuración no contiene las claves esperadas
        
    Examples
    --------
    >>> config = load_config()
    >>> path = get_data_path(config, "simulation", "output")
    >>> print(path)
    /ruta/proyecto/data/simulaciones
    """
    if data_type not in ["simulation", "real"]:
        raise ValueError(f"data_type debe ser 'simulation' o 'real', no '{data_type}'")
    
    if io_type not in ["input", "output"]:
        raise ValueError(f"io_type debe ser 'input' o 'output', no '{io_type}'")
    
    try:
        if data_type == "simulation":
            relative_path = config['data']['simulations'][io_type]
        else:
            relative_path = config['data']['real'][io_type]
    except KeyError as e:
        raise KeyError(
            f"Configuración incompleta: no se encontró data.{data_type}s.{io_type} "
            f"en config.yaml"
        ) from e
    
    path = resolve_path(relative_path)
    
    # Crear el directorio si no existe
    path.mkdir(parents=True, exist_ok=True)
    
    return path


def get_report_path(
    config: Dict,
    data_type: Literal["simulation", "real"] = "simulation",
    report_type: Literal["graphics", "tables"] = "graphics"
) -> Path:
    """
    Obtiene la ruta de reportes según el tipo.
    
    Parameters
    ----------
    config : Dict
        Diccionario de configuración cargado
    data_type : {'simulation', 'real'}
        Tipo de datos: simulaciones o datos reales
    report_type : {'graphics', 'tables'}
        Tipo de reporte: gráficas o tablas
    
    Returns
    -------
    Path
        Ruta absoluta al directorio de reportes
        
    Raises
    ------
    ValueError
        Si data_type o report_type no son válidos
    KeyError
        Si la configuración no contiene las claves esperadas
        
    Examples
    --------
    >>> config = load_config()
    >>> path = get_report_path(config, "simulation", "graphics")
    >>> print(path)
    /ruta/proyecto/reports/simulaciones
    """
    if data_type not in ["simulation", "real"]:
        raise ValueError(f"data_type debe ser 'simulation' o 'real', no '{data_type}'")
    
    if report_type not in ["graphics", "tables"]:
        raise ValueError(f"report_type debe ser 'graphics' o 'tables', no '{report_type}'")
    
    try:
        if data_type == "simulation":
            relative_path = config['reports']['simulations'][report_type]
        else:
            relative_path = config['reports']['real'][report_type]
    except KeyError as e:
        raise KeyError(
            f"Configuración incompleta: no se encontró reports.{data_type}s.{report_type} "
            f"en config.yaml"
        ) from e
    
    path = resolve_path(relative_path)
    
    # Crear el directorio si no existe
    path.mkdir(parents=True, exist_ok=True)
    
    return path


def get_artifact_path(
    config: Dict,
    data_type: Literal["simulation", "real"] = "simulation"
) -> Path:
    """
    Obtiene la ruta para guardar artefactos (modelos entrenados).
    
    Parameters
    ----------
    config : Dict
        Diccionario de configuración cargado
    data_type : {'simulation', 'real'}
        Tipo de datos: simulaciones o datos reales
    
    Returns
    -------
    Path
        Ruta absoluta al directorio de artefactos
        
    Raises
    ------
    ValueError
        Si data_type no es válido
    KeyError
        Si la configuración no contiene las claves esperadas
        
    Examples
    --------
    >>> config = load_config()
    >>> path = get_artifact_path(config, "simulation")
    >>> print(path)
    /ruta/proyecto/artefact/simulaciones/models
    """
    if data_type not in ["simulation", "real"]:
        raise ValueError(f"data_type debe ser 'simulation' o 'real', no '{data_type}'")
    
    try:
        if data_type == "simulation":
            relative_path = config['artifacts']['simulations']
        else:
            relative_path = config['artifacts']['real']
    except KeyError as e:
        raise KeyError(
            f"Configuración incompleta: no se encontró artifacts.{data_type}s "
            f"en config.yaml"
        ) from e
    
    path = resolve_path(relative_path)
    
    # Crear el directorio si no existe
    path.mkdir(parents=True, exist_ok=True)
    
    return path


def get_experiment_registry_path(config: Dict) -> Path:
    """
    Obtiene la ruta al archivo de registro de experimentos.
    
    Parameters
    ----------
    config : Dict
        Diccionario de configuración cargado
    
    Returns
    -------
    Path
        Ruta absoluta al archivo experiment_registry.md
    """
    try:
        relative_path = config['experiments']['registry_file']
    except KeyError as e:
        raise KeyError(
            "Configuración incompleta: no se encontró experiments.registry_file "
            "en config.yaml"
        ) from e
    
    return resolve_path(relative_path)


def get_log_dir(config: Dict) -> Path:
    """
    Obtiene el directorio de logs de experimentos.
    
    Parameters
    ----------
    config : Dict
        Diccionario de configuración cargado
    
    Returns
    -------
    Path
        Ruta absoluta al directorio de logs
    """
    try:
        relative_path = config['experiments']['log_dir']
    except KeyError as e:
        raise KeyError(
            "Configuración incompleta: no se encontró experiments.log_dir "
            "en config.yaml"
        ) from e
    
    path = resolve_path(relative_path)
    path.mkdir(parents=True, exist_ok=True)
    
    return path


def create_experiment_id(prefix: str = "exp") -> str:
    """
    Crea un ID único para un experimento basado en timestamp.
    
    Parameters
    ----------
    prefix : str
        Prefijo para el ID del experimento
        
    Returns
    -------
    str
        ID del experimento en formato: prefix_YYYYMMDD_HHMMSS
        
    Examples
    --------
    >>> exp_id = create_experiment_id("sim")
    >>> print(exp_id)
    sim_20241223_143052
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}"


def ensure_directories(config: Dict) -> None:
    """
    Asegura que todos los directorios necesarios existan.
    
    Crea todos los directorios especificados en la configuración
    si no existen ya.
    
    Parameters
    ----------
    config : Dict
        Diccionario de configuración cargado
    """
    # Crear directorios de datos
    for data_type in ["simulation", "real"]:
        for io_type in ["input", "output"]:
            get_data_path(config, data_type, io_type)
    
    # Crear directorios de reportes
    for data_type in ["simulation", "real"]:
        for report_type in ["graphics", "tables"]:
            get_report_path(config, data_type, report_type)
    
    # Crear directorios de artefactos
    for data_type in ["simulation", "real"]:
        get_artifact_path(config, data_type)
    
    # Crear directorio de logs
    get_log_dir(config)
    
    print("✓ Todos los directorios han sido verificados/creados")


