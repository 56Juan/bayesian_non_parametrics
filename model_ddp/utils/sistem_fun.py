# sistem_fun.py
"""
Módulo de utilidades del sistema para el proyecto model_ddp.
Proporciona funciones para manejo de rutas, configuración y organización de experimentos.
"""

import yaml
from pathlib import Path
from datetime import datetime
import os


def get_project_root():
    """
    Encuentra la raíz del proyecto buscando el archivo pyproject.toml.
    
    Returns:
        Path: Ruta absoluta a la raíz del proyecto
    """
    current = Path(__file__).resolve()
    
    # Buscar hacia arriba hasta encontrar pyproject.toml
    for parent in [current] + list(current.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    
    # Si no lo encuentra, asumir que está 2 niveles arriba de este archivo
    # (model_ddp/utils/sistem_fun.py -> project_root/)
    return current.parent.parent


def load_config(config_path=None):
    """
    Carga la configuración del proyecto desde config.yaml.
    
    Args:
        config_path (str|Path, optional): Ruta al archivo de configuración. 
                                         Si es None, busca en versioning/config.yaml
    
    Returns:
        dict: Diccionario con la configuración del proyecto
        
    Raises:
        FileNotFoundError: Si no se encuentra el archivo de configuración
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
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def get_data_path(config, data_type="simulation", io_type="output"):
    """
    Obtiene la ruta de datos según el tipo especificado.
    
    Args:
        config (dict): Diccionario de configuración
        data_type (str): "simulation" o "real"
        io_type (str): "input" o "output"
    
    Returns:
        Path: Ruta absoluta al directorio de datos
        
    Example:
        >>> config = load_config()
        >>> path = get_data_path(config, "simulation", "output")
        >>> print(path)  # /path/to/project/data/simulaciones
    """
    project_root = get_project_root()
    
    if data_type == "simulation":
        relative_path = config['data']['simulations'][io_type]
    else:
        relative_path = config['data']['real'][io_type]
    
    # Convertir ruta relativa a absoluta y crear si no existe
    data_path = project_root / relative_path.lstrip('../')
    data_path.mkdir(parents=True, exist_ok=True)
    
    return data_path


def get_report_path(config, data_type="simulation", report_type="graphics"):
    """
    Obtiene la ruta de reportes según el tipo especificado.
    
    Args:
        config (dict): Diccionario de configuración
        data_type (str): "simulation" o "real"
        report_type (str): "graphics" o "tables"
    
    Returns:
        Path: Ruta absoluta al directorio de reportes
        
    Example:
        >>> config = load_config()
        >>> path = get_report_path(config, "simulation", "graphics")
        >>> print(path)  # /path/to/project/reports/simulaciones
    """
    project_root = get_project_root()
    
    if data_type == "simulation":
        relative_path = config['reports']['simulations'][report_type]
    else:
        relative_path = config['reports']['real'][report_type]
    
    # Convertir ruta relativa a absoluta y crear si no existe
    report_path = project_root / relative_path.lstrip('../')
    report_path.mkdir(parents=True, exist_ok=True)
    
    return report_path


def get_artifact_path(config, data_type="simulation"):
    """
    Obtiene la ruta de artefactos (modelos entrenados) según el tipo.
    
    Args:
        config (dict): Diccionario de configuración
        data_type (str): "simulation" o "real"
    
    Returns:
        Path: Ruta absoluta al directorio de artefactos
        
    Example:
        >>> config = load_config()
        >>> path = get_artifact_path(config, "simulation")
        >>> print(path)  # /path/to/project/artefact/simulaciones/models
    """
    project_root = get_project_root()
    
    if data_type == "simulation":
        relative_path = config['artifacts']['simulations']
    else:
        relative_path = config['artifacts']['real']
    
    # Convertir ruta relativa a absoluta y crear si no existe
    artifact_path = project_root / relative_path.lstrip('../')
    artifact_path.mkdir(parents=True, exist_ok=True)
    
    return artifact_path


def get_model_path(config):
    """
    Obtiene la ruta para guardar modelos.
    NOTA: Esta función usa get_artifact_path como alias para compatibilidad.
    
    Args:
        config (dict): Diccionario de configuración
    
    Returns:
        Path: Ruta absoluta al directorio de modelos
    """
    # Por defecto retorna la ruta de artefactos de simulaciones
    return get_artifact_path(config, data_type="simulation")


def get_versioning_path(config):
    """
    Obtiene la ruta del directorio de versionado.
    
    Args:
        config (dict): Diccionario de configuración
    
    Returns:
        Path: Ruta absoluta al directorio de versioning
    """
    project_root = get_project_root()
    versioning_path = project_root / "versioning"
    versioning_path.mkdir(parents=True, exist_ok=True)
    
    return versioning_path


def create_experiment_id(prefix="exp"):
    """
    Crea un ID único para el experimento basado en timestamp.
    
    Args:
        prefix (str): Prefijo para el ID (ej: "lsbp", "exp", "test", "sim")
    
    Returns:
        str: ID único en formato prefix_YYYYMMDD_HHMMSS
        
    Example:
        >>> exp_id = create_experiment_id("lsbp")
        >>> print(exp_id)  # lsbp_20231215_143022
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}"


def ensure_directories(config):
    """
    Asegura que todos los directorios necesarios del proyecto existan.
    Crea todos los directorios definidos en la configuración si no existen.
    
    Args:
        config (dict): Diccionario de configuración
        
    Returns:
        dict: Diccionario con las rutas creadas
    """
    project_root = get_project_root()
    created_paths = {}
    
    # Crear directorios de datos
    for data_type in ['simulations', 'real']:
        for io_type in ['input', 'output']:
            path = config['data'][data_type][io_type]
            full_path = project_root / path.lstrip('../')
            full_path.mkdir(parents=True, exist_ok=True)
            created_paths[f'data_{data_type}_{io_type}'] = full_path
    
    # Crear directorios de reportes
    for data_type in ['simulations', 'real']:
        for report_type in ['graphics', 'tables']:
            path = config['reports'][data_type][report_type]
            full_path = project_root / path.lstrip('../')
            full_path.mkdir(parents=True, exist_ok=True)
            created_paths[f'report_{data_type}_{report_type}'] = full_path
    
    # Crear directorios de artefactos
    for data_type in ['simulations', 'real']:
        path = config['artifacts'][data_type]
        full_path = project_root / path.lstrip('../')
        full_path.mkdir(parents=True, exist_ok=True)
        created_paths[f'artifact_{data_type}'] = full_path
    
    # Crear directorio de versioning
    versioning_path = project_root / config['experiments']['log_dir'].lstrip('../')
    versioning_path.mkdir(parents=True, exist_ok=True)
    created_paths['versioning'] = versioning_path
    
    # Crear directorio de referencias si no existe
    references_path = project_root / "references"
    references_path.mkdir(parents=True, exist_ok=True)
    created_paths['references'] = references_path
    
    print("✓ Todos los directorios han sido creados/verificados")
    
    return created_paths


def get_all_paths(config):
    """
    Obtiene un diccionario con todas las rutas importantes del proyecto.
    
    Args:
        config (dict): Diccionario de configuración
    
    Returns:
        dict: Diccionario jerárquico con todas las rutas del proyecto
        
    Example:
        >>> config = load_config()
        >>> paths = get_all_paths(config)
        >>> print(paths['data']['simulation']['output'])
    """
    project_root = get_project_root()
    
    return {
        'root': project_root,
        'data': {
            'simulation': {
                'input': get_data_path(config, 'simulation', 'input'),
                'output': get_data_path(config, 'simulation', 'output')
            },
            'real': {
                'input': get_data_path(config, 'real', 'input'),
                'output': get_data_path(config, 'real', 'output')
            }
        },
        'reports': {
            'simulation': {
                'graphics': get_report_path(config, 'simulation', 'graphics'),
                'tables': get_report_path(config, 'simulation', 'tables')
            },
            'real': {
                'graphics': get_report_path(config, 'real', 'graphics'),
                'tables': get_report_path(config, 'real', 'tables')
            }
        },
        'artifacts': {
            'simulation': get_artifact_path(config, 'simulation'),
            'real': get_artifact_path(config, 'real')
        },
        'versioning': get_versioning_path(config),
        'references': project_root / "references",
        'notebooks': {
            'simulation': project_root / "notebooks" / "simulaciones",
            'real': project_root / "notebooks" / "reales"
        }
    }


def save_experiment_metadata(config, experiment_data):
    """
    Guarda metadatos básicos de un experimento en el registro.

    Args:
        config (dict): Diccionario de configuración
        experiment_data (dict): Información básica del experimento

    Returns:
        Path: Ruta al archivo de registro
    """
    project_root = get_project_root()
    registry_path = Path(config['experiments']['registry_file'].replace('../', ''))
    registry_file = project_root / registry_path

    # Asegurar que el directorio existe
    registry_file.parent.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    registry_entry = f"""
## {experiment_data.get('experiment_id', 'unknown')}
- **Nombre**: {experiment_data.get('nombre', 'N/A')}
- **Fecha**: {timestamp}
- **Tipo**: {experiment_data.get('tipo', 'N/A')}

**Descripción**
{experiment_data.get('descripcion', 'N/A')}

---
"""

    with open(registry_file, 'a', encoding='utf-8') as f:
        f.write(registry_entry)

    return registry_file



def print_project_structure(config):
    """
    Imprime la estructura de directorios del proyecto.
    
    Args:
        config (dict): Diccionario de configuración
    """
    paths = get_all_paths(config)
    
    print("\n" + "=" * 70)
    print("📁 ESTRUCTURA DEL PROYECTO")
    print("=" * 70)
    print(f"\n📂 Raíz: {paths['root']}")
    print("\n📊 Datos:")
    print(f"  Simulaciones:")
    print(f"    Input:  {paths['data']['simulation']['input']}")
    print(f"    Output: {paths['data']['simulation']['output']}")
    print(f"  Reales:")
    print(f"    Input:  {paths['data']['real']['input']}")
    print(f"    Output: {paths['data']['real']['output']}")
    print("\n📈 Reportes:")
    print(f"  Simulaciones: {paths['reports']['simulation']['graphics']}")
    print(f"  Reales:       {paths['reports']['real']['graphics']}")
    print("\n🔧 Artefactos:")
    print(f"  Simulaciones: {paths['artifacts']['simulation']}")
    print(f"  Reales:       {paths['artifacts']['real']}")
    print("\n📝 Versionado:")
    print(f"  {paths['versioning']}")
    print("\n" + "=" * 70)


# Función de test para verificar el módulo
def test_system_utils():
    """
    Función de prueba para verificar que todas las utilidades funcionan correctamente.
    """
    print("\n🧪 TESTING SYSTEM UTILITIES")
    print("=" * 70)
    
    try:
        # Test 1: Cargar configuración
        print("\n1. Cargando configuración...")
        config = load_config()
        print(f"   ✓ Configuración cargada: {config['project']['name']} v{config['project']['version']}")
        
        # Test 2: Crear directorios
        print("\n2. Creando/verificando directorios...")
        ensure_directories(config)
        
        # Test 3: Crear ID de experimento
        print("\n3. Creando ID de experimento...")
        exp_id = create_experiment_id("test")
        print(f"   ✓ ID creado: {exp_id}")
        
        # Test 4: Obtener rutas
        print("\n4. Obteniendo rutas...")
        data_path = get_data_path(config, "simulation", "output")
        report_path = get_report_path(config, "simulation", "graphics")
        artifact_path = get_artifact_path(config, "simulation")
        print(f"   ✓ Ruta de datos: {data_path}")
        print(f"   ✓ Ruta de reportes: {report_path}")
        print(f"   ✓ Ruta de artefactos: {artifact_path}")
        
        # Test 5: Mostrar estructura
        print("\n5. Estructura del proyecto:")
        print_project_structure(config)
        
        print("\n" + "=" * 70)
        print("✅ TODOS LOS TESTS PASARON EXITOSAMENTE")
        print("=" * 70 + "\n")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Ejecutar tests si se corre el módulo directamente
    test_system_utils()