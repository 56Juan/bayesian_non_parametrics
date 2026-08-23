# Modelo PSBP extendido a prediccion de Series de tiempo Funcionales

> **PSBP extendido a series de tiempo funcionales**  
> Basado en Chung & Dunson (2009).

Este paquete implementa una extensión funcional del modelo **Product of Stick-Breaking Process (PSBP)** para series de tiempo, permitiendo trabajar con curvas funcionales discretas mediante representación en bases (B-splines, Fourier), FPCA y estandarización de datos.

## ⚙️ Instalación del entorno (`.venv`)

Requiere **Python ≥ 3.11** ya instalado. Todos los comandos se ejecutan desde la raíz del proyecto.

**1. Crear el entorno virtual**

```bash
python -m venv .venv
```

**2. Activarlo**

```bash
# Windows — PowerShell
.venv\Scripts\Activate.ps1

# Windows — CMD
.venv\Scripts\activate.bat

# Windows — Git Bash
source .venv/Scripts/activate

# Linux / macOS
source .venv/bin/activate
```

Si PowerShell bloquea el script de activación, habilitar la ejecución solo para la sesión actual:

```bash
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

**3. Instalar el paquete en modo editable**

```bash
python -m pip install --upgrade pip
pip install -e .
```

`-e` (editable) instala `model_psbp_fd` enlazado al código fuente: los cambios en los módulos se reflejan sin reinstalar. El comando arrastra además todas las dependencias declaradas en `pyproject.toml` (`numpy`, `scipy`, `pandas`, `matplotlib`, `seaborn`, `scikit-fda`, `pyyaml`, `jinja2`, `openpyxl`).

Para las dependencias de desarrollo (`pytest`, `pytest-cov`):

```bash
pip install -e ".[dev]"
```

**4. Verificar la instalación**

```bash
python -c "import model_psbp_fd; from model_psbp_fd.models import estado_versiones; print(model_psbp_fd.__version__); print(estado_versiones())"
```

`estado_versiones()` reporta qué versiones del modelo quedaron disponibles. `psbp_fd_v3` —la que usa el estudio— debe aparecer como `disponible`; `psbp_fd_v1` puede figurar como no disponible si la extensión compilada (`.pyd`) no corresponde a la plataforma actual, lo que **no** afecta el flujo de trabajo vigente.

**5. Registrar el kernel para los notebooks** (opcional)

```bash
pip install ipykernel
python -m ipykernel install --user --name psbp_fd --display-name "Python (psbp_fd)"
```

> **Nota.** El muestreo MCMC ocurre en MATLAB, no en Python. El entorno `.venv` cubre la generación de datos, el preprocesamiento y la evaluación; para ejecutar los `.m` se requiere MATLAB por separado.
>
> Existe también una alternativa con conda en `environment.yml` (entorno `psbp_fd`). Usar una u otra, no ambas.

## 🏗️ Estructura del proyecto

```
model_psbp_fd/                           # Raíz del proyecto
│
├── model_psbp_fd/                       # Paquete principal (Python)
│   ├── __init__.py
│   │
│   ├── functions_models/                # Clases auxiliares de preprocesamiento
│   │   ├── __init__.py
│   │   ├── functions_standarize.py      # Estandarización de coeficientes
│   │   └── functions_repre_functional.py # Representación funcional (bases)
│   │
│   ├── models/                          # Versiones del modelo PSBP-FD
│   │   ├── __init__.py
│   │   ├── CHANGELOG.md
│   │   │
│   │   ├── psbp_fd_v1/                  # Primera versión funcional
│   │   │   ├── __init__.py
│   │   │   ├── psbp_fd_v1.py            # Clase principal: __init__, fit(), predict()
│   │   │   ├── functions/
│   │   │   │   ├── sampler.py           # Muestreador MCMC
│   │   │   │   └── predict.py           # Funciones de predicción
│   │   │   ├── psbp_fd_v1.cp312-win_amd64.pyd
│   │   │   └── cpp/                     # Extensiones C++ (pybind11)
│   │   │       ├── psbp_fd_v1.cpp
│   │   │       ├── psbp_fd_v1.hpp
│   │   │       ├── bindings.cpp
│   │   │       └── CMakeLists.txt
│   │   │
│   │   └── psbp_fd_vx/                  # Versión experimental/futura
│   │       └── ...
│   │
│   ├── pipelines/                       # Composición data → modelo → evaluación
│   │   ├── __init__.py
│   │   ├── simulation_pipeline.py
│   │   └── real_data_pipeline.py
│   │
│   ├── fit/                             # Diagnósticos y métricas
│   │   ├── __init__.py
│   │   └── ...
│   │
│   ├── graphics/                        # Visualización
│   │   ├── __init__.py
│   │   ├── viz_functional_data.py
│   │   ├── viz_global_components.py
│   │   ├── viz_prediction.py
│   │   ├── viz_time_series.py
│   │   └── viz_traces.py
│   │
│   └── utils/                           # Utilidades transversales
│       ├── __init__.py
│       └── ...
│
├── artefact/                                # Objetos serializados 
│   ├── reales/
│   │   └── models/
│   └── simulaciones/
│       └── models/
│
├── data/                                     # Datos brutos y procesados 
│   ├── reales/
│   │   └── models/
│   └── simulaciones/
│       ├── processed/
│       │   ├── functional/
│       │   └── .gitkeep
│       └── predict/
│           ├── .gitkeep
│
├── notebooks/                                 # Análisis y experimentación
│   ├── simulacion/                      
│   │   └─── model_xxx/                        # Script principal MATLAB - Python
│   │         ├── config_paths.m               # Configuración de rutas
│   │         ├── run_pipeline.m               # Generacion de Trazas 
│   │         └── *.ipynb                      # Nootbook completo 
│   └── reales/
│       └── *.ipynb
│
├── reports/                             # Outputs finales (figuras, tablas)
│   ├── reales/
│   └── simulaciones/
│
├── references/                          # Documentación de referencia
│   └── ...
│
├── versioning/                          # Registro experimental manual
│   └── ...
│
├── pyproject.toml
├── environment.yml
├── README.md
└── .gitignore
```
