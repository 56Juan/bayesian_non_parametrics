import os
import sys
from pathlib import Path


def get_project_root(marker: str = "README.md") -> Path:
    """
    Encuentra la raíz del proyecto buscando un archivo marcador.
    
    Args:
        marker: Nombre del archivo marcador (por defecto "README.md")
    
    Returns:
        Path: Ruta absoluta a la raíz del proyecto
    
    Raises:
        FileNotFoundError: Si no encuentra el marcador
    """
    current = Path(os.getcwd()).resolve()
    
    for parent in [current] + list(current.parents):
        if (parent / marker).exists():
            return parent
    
    # Opción 1: Lanzar excepción (RECOMENDADO)
    raise FileNotFoundError(
        f"No se encontró '{marker}' en la jerarquía de directorios. "
        f"Directorio actual: {current}"
    )