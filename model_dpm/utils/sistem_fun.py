import yaml
from pathlib import Path

#Funciones del sistema de configuración


def load_config(config_path="../versioning/config.yaml"):
    """Carga la configuración del proyecto."""
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
    """
    if data_type == "simulation":
        path = config['data']['simulations'][io_type]
    else:
        path = config['data']['real'][io_type]
    
    return Path(path)

def get_report_path(config, data_type="simulation", report_type="graphics"):
    """
    Obtiene la ruta de reportes según el tipo.
    
    Args:
        config: Diccionario de configuración
        data_type: "simulation" o "real"
        report_type: "graphics" o "tables"
    """
    if data_type == "simulation":
        path = config['reports']['simulations'][report_type]
    else:
        path = config['reports']['real'][report_type]
    
    return Path(path)