r"""
rutas.py — Las cinco rutas del contrato, en un solo lugar
=========================================================

Hasta ahora **no existia un `config_paths` del lado de Python**: cada notebook
armaba a mano el dict `PATHS` con las claves `raw`, `functional`, `predict`,
`out_report` y `out_artefact`, y `config_paths.m` replicaba esas mismas rutas
del lado MATLAB. Como hay una copia del `.m` por carpeta de experimento y una
celda de rutas por notebook, cualquier cambio de convencion habia que
propagarlo a mano a todas: fue la fuente de error mas persistente del proyecto,
la misma que motivo `pipelines/artifacts.py` para los *nombres de archivo*.
Este modulo hace con los *directorios* lo que aquel hizo con los archivos.

Lo que queda fuera a proposito
------------------------------
- **Los nombres de archivo dentro de cada directorio** siguen siendo cosa del
  dict `ARCHIVOS` de `pipelines/artifacts.py`. Aqui solo se decide *donde*.
- **`config_paths.m` no se toca.** Sigue siendo la definicion del lado MATLAB;
  este modulo es su gemelo, no su reemplazo. Si una cambia, la otra tambien.

La convencion del identificador
-------------------------------
Hay tres vivas en el repo y `experiment_id` cubre las dos que se usan hoy::

    <basename>_<escenario>                         corridas 03-10  (heredada)
    <basename>_<escenario>_r<replica:02d>          corridas 11-17
    <basename>_<escenario>_r<replica:02d>_m<M:02d> corridas 20 en adelante

`M` viaja en el identificador para que cada punto del barrido escriba sus
propios datos, trazas y reportes sin pisar los demas. Los datos reales usan
otra convencion (`real_<serie>_v<NN>_m<NN>`), que no depende de escenario ni de
replica: para esos, se pasa el id ya armado a `construir_paths`.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional, Sequence

from .raiz import get_project_root

__all__ = [
    "CLAVES_PATHS",
    "experiment_id",
    "construir_paths",
    "normalizar_lista_M",
    "rutas_por_M",
    "ruta_barrido_M",
    "guardar_en_todos",
    "replicar_figura",
]

#: Las cinco claves del contrato, en el orden en que se documentan. Cualquier
#: consumidor (`verificar_contrato`, los `guardar_*`/`cargar_*`) espera este
#: juego exacto: ni mas ni menos.
CLAVES_PATHS = ("raw", "functional", "predict", "out_report", "out_artefact")

#: Directorios que `construir_paths(limpiar=True)` vacia. `out_report` NO esta:
#: sus figuras se sobrescriben por nombre y sirven de registro visual de la
#: corrida anterior, mientras que un dataset o una traza rancios se leerian como
#: si fueran de esta.
_CLAVES_LIMPIABLES = ("raw", "functional", "predict", "out_artefact")

_DOMINIOS = ("simulaciones", "reales")


def experiment_id(
    basename: str,
    escenario_id: int,
    replica_id: Optional[int] = None,
    M: Optional[int] = None,
) -> str:
    """
    Arma el `EXPERIMENT_ID` segun la convencion vigente.

    Parametros
    ----------
    basename : str
        Prefijo del experimento (p. ej. ``"escenario"``).
    escenario_id : int
        Algoritmo `k` del anexo.
    replica_id : int, opcional
        Replica Monte Carlo. Si es None se omite el sufijo `_rNN`, que es la
        convencion heredada de las corridas 03-10.
    M : int, opcional
        Componentes FPCA retenidas. Si es None se omite el sufijo `_mNN`, que
        es la convencion de las corridas 11-17.

    Ejemplos
    --------
    >>> experiment_id("escenario", 1, 1, 2)
    'escenario_1_r01_m02'
    >>> experiment_id("escenario", 3)
    'escenario_3'
    """
    eid = f"{basename}_{int(escenario_id)}"
    if replica_id is not None:
        eid += f"_r{int(replica_id):02d}"
    if M is not None:
        eid += f"_m{int(M):02d}"
    return eid


def construir_paths(
    eid: str,
    dominio: str = "simulaciones",
    project_root: Optional[Path] = None,
    limpiar: bool = False,
    crear: bool = True,
) -> Dict[str, Path]:
    """
    Las cinco rutas del contrato para un `EXPERIMENT_ID`, con los directorios
    creados.

    Parametros
    ----------
    eid : str
        `EXPERIMENT_ID` ya armado (ver `experiment_id`). Debe coincidir
        **exactamente** con el que usan el `.m` y los notebooks de evaluacion.
    dominio : {"simulaciones", "reales"}
        Rama del arbol de datos. `data/<dominio>/...`, `reports/<dominio>/...`,
        `artefact/<dominio>/...`.
    project_root : Path, opcional
        Raiz del proyecto. Por defecto se localiza con `get_project_root()`.
    limpiar : bool
        Si es True, **borra** el contenido de `raw`, `functional`, `predict` y
        `out_artefact` antes de recrearlos. Es lo que evita que una reejecucion
        deje artefactos rancios de una corrida anterior mezclados con los
        nuevos: un `dataset_fpc_4_*.csv` que sobra tras bajar `M`, o unas
        trazas `.mat` de un ajuste que ya no existe. Ojo: borra las trazas, y
        eso es lo correcto --si los datasets se regeneran, las trazas viejas
        son de otro ajuste--, pero implica volver a correr MATLAB.
    crear : bool
        Si es False no crea nada; sirve para *leer* rutas sin tocar el disco.

    Retorna
    -------
    dict
        Las claves de `CLAVES_PATHS`, cada una un `Path` absoluto.
    """
    if dominio not in _DOMINIOS:
        raise ValueError(f"dominio debe ser uno de {_DOMINIOS}; se recibio {dominio!r}.")
    if not eid:
        raise ValueError("El EXPERIMENT_ID esta vacio.")

    root = Path(project_root) if project_root is not None else get_project_root()
    datos = root / "data" / dominio

    paths = {
        "raw":          datos / "raw" / eid,
        "functional":   datos / "processed" / "functional" / eid,
        "predict":      datos / "processed" / "predict" / eid,
        "out_report":   root / "reports" / dominio / eid,
        "out_artefact": root / "artefact" / dominio / eid,
    }
    assert tuple(paths) == CLAVES_PATHS, "El juego de claves cambio; revise CLAVES_PATHS."

    if limpiar:
        for clave in _CLAVES_LIMPIABLES:
            if paths[clave].exists():
                shutil.rmtree(paths[clave])
    if crear:
        for ruta in paths.values():
            ruta.mkdir(parents=True, exist_ok=True)
    return paths


def normalizar_lista_M(M_list: Iterable[int]) -> tuple:
    """
    Deja la lista del barrido en forma canonica: enteros unicos, ordenados y no
    vacia. Es **la misma normalizacion que hace `psbp_fd_iteracion.m`**
    (`unique(M_FPCA_LIST(:))'`), y esta aqui para que las dos mitades del ciclo
    recorran el barrido en el mismo orden y con los mismos puntos aunque el
    notebook lo declare desordenado o con repetidos.
    """
    M_norm = tuple(sorted({int(m) for m in M_list}))
    if not M_norm:
        raise ValueError("La lista del barrido en M esta vacia.")
    if any(m < 1 for m in M_norm):
        raise ValueError(f"M debe ser >= 1; se recibio {M_norm}.")
    return M_norm


def rutas_por_M(
    basename: str,
    escenario_id: int,
    replica_id: int,
    M_list: Sequence[int],
    dominio: str = "simulaciones",
    project_root: Optional[Path] = None,
    limpiar: bool = False,
) -> Dict[int, Dict[str, Path]]:
    """
    Un juego de rutas **por punto** del barrido en `M`.

    Cada directorio tiene que quedar **autocontenido**: MATLAB lee el suyo, los
    notebooks de evaluacion leen el suyo, y `verificar_contrato()` cruza
    manifest, hiperparametros y artefactos FPCA dentro de un solo
    `EXPERIMENT_ID`. Por eso los artefactos que **no** dependen de `M` --la
    realizacion, las curvas, la representacion-- se replican en todos ellos con
    `guardar_en_todos` en vez de vivir en uno solo.

    Retorna
    -------
    dict
        `{M: paths}`, con `M` recorriendo `normalizar_lista_M(M_list)`.
    """
    return {
        M: construir_paths(
            experiment_id(basename, escenario_id, replica_id, M),
            dominio=dominio, project_root=project_root, limpiar=limpiar,
        )
        for M in normalizar_lista_M(M_list)
    }


def ruta_barrido_M(
    basename: str,
    escenario_id: int,
    replica_id: int,
    dominio: str = "simulaciones",
    project_root: Optional[Path] = None,
    crear: bool = True,
) -> Path:
    """
    Directorio de las salidas que **cruzan** los puntos del barrido.

    Lo que compara varios `M` no cabe en el directorio de ninguno de ellos: va
    a `reports/<dominio>/<basename>_<escenario>_r<NN>_barrido_M/`, hermano de
    los de cada punto. Es donde escriben los prefijos 80-93 de los notebooks
    `_03`, `_04` y `_05`.
    """
    root = Path(project_root) if project_root is not None else get_project_root()
    base = experiment_id(basename, escenario_id, replica_id)
    ruta = root / "reports" / dominio / f"{base}_barrido_M"
    if crear:
        ruta.mkdir(parents=True, exist_ok=True)
    return ruta


def guardar_en_todos(
    fn: Callable,
    paths_por_M: Dict[int, Dict[str, Path]],
    *args,
    **kwargs,
):
    """
    Aplica un `guardar_*` de `pipelines.artifacts` al directorio de **cada** M.

    El escenario, las curvas y la representacion no dependen de `M` --salen de
    la misma realizacion y de la misma base-- pero cada punto del barrido tiene
    que poder leerse solo. Se replican con el propio modulo de artefactos, en
    el punto del notebook donde se producen, en vez de copiar archivos a mano
    despues: asi los nombres los sigue definiendo el dict `ARCHIVOS` y no hay
    una segunda convencion que mantener sincronizada.

    Retorna lo que devolvio la ultima llamada, que es lo mismo que devolvieron
    las demas salvo por el directorio.
    """
    salida = None
    for paths in paths_por_M.values():
        salida = fn(paths, *args, **kwargs)
    return salida


def replicar_figura(
    nombre: str,
    paths_por_M: Dict[int, Dict[str, Path]],
    fig=None,
    dpi: int = 150,
) -> None:
    """
    Guarda una figura en el `out_report` de cada `M`, con el mismo nombre.

    Las figuras de la simulacion, del GCV y del espectro describen la
    realizacion y la base, que son **comunes a todo el barrido**; se replican
    para que el directorio de cada punto se lea solo, sin obligar a mirar el de
    otro `M`. `matplotlib` se importa aqui dentro a proposito: `utils` no
    depende de el, y este es el unico punto que lo necesita.
    """
    if fig is None:
        import matplotlib.pyplot as plt
        fig = plt.gcf()
    for paths in paths_por_M.values():
        fig.savefig(paths["out_report"] / nombre, dpi=dpi, bbox_inches="tight")
