# model_psbp_fd

PSBP extendido a series de tiempo funcionales.
Basado en Chung & Dunson (2009), JASA 104(488).

## Instalación
```bash
conda env create -f environment.yml
conda activate psbp_fd
pip install -e .
```

## MODELO
Aqui describire que contendra el modelo, funciones previas que debo tener .

La clase models contendra un clase del modelo que englobara a las clase de estandarizacion y de representacion funcional 



representacion funcional mediante bases

estandarizacion de los datos 



## Idea del repo

model_psbp_fd/
│
├── model_psbp_fd/                        # Paquete principal (src-layout interno)
│   ├── __init__.py
│   │
│   ├── functions_models/                 # Clases de funciones auxiliares para tratar la data
│   │   ├── __init__.py
│   │   ├── functions_standarize.py       # Contiene la clase que permite estandarizar
│   │   └── functions_repre_functional.py # Contiene la clase para representación funcional
│   │
│   ├── models/                           # Versiones del modelo
│   │   ├── __init__.py
│   │   ├── CHANGELOG.md                  # Documentación de versiones
│   │   │
│   │   ├── psbp_fd_v1/                   # Primera versión funcional
│   │   │   ├── __init__.py
│   │   │   ├── psbp_fd_v1.py             # Clase del modelo completo: __init__, fit(), predict()
│   │   │   ├── psbp_fd_v1.cp312-win_amd64.pyd  # Binario compilado (Windows)
│   │   │   └── cpp/                      # Funciones en C++
│   │   │       ├── psbp_fd_v1.cpp        # Implementación C++
│   │   │       ├── psbp_fd_v1.hpp        # Headers
│   │   │       ├── bindings.cpp          # Bindings pybind11
│   │   │       ├── CMakeLists.txt        # Configuración de build C++
│   │   │       └── .gitkeep
│   │   │
│   │   └── psbp_fd_vx/                   # Versión experimental/futura
│   │       ├── __init__.py
│   │       ├── m_vx/
│   │       │   └── psbp_fd_vx.py         # Clase del modelo completo: __init__, fit(), predict()
│   │       └── cpp/                      # Funciones en C++
│   │           └── .gitkeep
│   │
│   ├── pipelines/                        # Composición data → modelo → evaluación
│   │   ├── __init__.py
│   │   ├── simulation_pipeline.py        # Genera datos + ajusta + evalúa
│   │   └── real_data_pipeline.py         # Carga real + ajusta + evalúa
│   │
│   ├── fit/                              # Evaluación del ajuste posterior
│   │   ├── __init__.py
│   │   ├── diagnostics.py                # R-hat, ESS, trace plots
│   │   ├── posterior_checks.py           # PPC funcionales
│   │   └── metrics.py                    # RMSE, LPML, WAIC, distancias L2
│   │
│   ├── graphics/                         # Visualización
│   │   ├── __init__.py
│   │   ├── functional_plots.py           # Curvas funcionales, bandas credibles
│   │   └── mcmc_plots.py                 # Trace plots, densidades marginales
│   │
│   └── utils/                            # Utilidades transversales
│       ├── __init__.py
│       └── helpers.py                    # Guardar, lectura de cosas
│
├── artefact/                             # Objetos serializados (no versionados)
│   ├── reales/
│   │   └── models/                       # Modelos entrenados (objetos .pkl reutilizables)
│   └── simulaciones/
│       └── models/
│
├── configs/                              # Configuración centralizada
│   ├── default.yaml                      # Hiperparámetros base del modelo
│   └── experiments/                      # Configs por experimento
│       └── sim_var1_K4.yaml
│
├── data/                                 # Datos (reales y/o simulados)
│   ├── reales/
│   │   ├── raw/                          # Datos brutos
│   │   └── processed/                    # Datos procesados
│   │       ├── functional/               # Datos procesados (representación+estandarización si aplica)
│   │       └── predict/                  # Datos predichos
│   └── simulaciones/
│       ├── raw/                          # Datos brutos
│       └── processed/                    # Datos procesados
│           ├── functional/               # Datos procesados (representación+estandarización si aplica)
│           └── predict/                  # Datos predichos
│
├── notebooks/
│   ├── simulaciones/
│   └── reales/
│
├── reports/                              # Outputs finales (figuras, tablas)
│   ├── simulaciones/
│   └── reales/
│
├── references/                           # Documentos de referencia (PDFs, notas técnicas)
│
├── versioning/                           # Control experimental manual
│   ├── config.yaml
│   ├── experiment_registry.md
│   └── changelog.md
│
├── tests/                                # Unit tests
│   ├── test_samplers.py
│   └── test_preprocessors.py
│
├── pyproject.toml
├── environment.yml
├── ref.bib
├── id_model.md
├── README.md
├── .gitignore
└── __init__.py

### functions_models - Clases de funciones para apoyar el Modelo

El módulo `functions_models` proporciona clases auxiliares que permiten variar el comportamiento del modelo PSBP-FD en dos etapas clave: la representación funcional de las curvas discretas y la estandarización de los coeficientes resultantes (AQUI DESPUES SE PUEDE OPTAR POR ESTANDARIZAR LOS DATOS PRIMERO).

Estas clases están diseñadas siguiendo el patrón de composición, lo que permite al modelo principal (`PSBP_FD`) delegar responsabilidades específicas y mantener un código modular, testeable y fácil de extender.

**¿Por qué dos clases separadas?**

La separación entre representación funcional y estandarización responde a la necesidad de probar diferentes combinaciones de estrategias:

| Representación Funcional | Estandarización |
|--------------------------|-----------------|
| B-splines + Z-score | Datos suaves, coeficientes homogéneos |
| FPCA + Robust | Datos atípicos, reducción de dimensionalidad |
| Fourier + MinMax | Datos periódicos, coeficientes acotados |
| Precomputed + None | Datos ya procesados (passthrough) |

#### Funcionalidad 1: `FunctionalRepresentation`

Convierte curvas discretas `Y (T, G)` en coeficientes funcionales `THETA (T, K)`.

**Métodos soportados:**
- `"bspline"`: B-splines cúbicos (requiere `skfda`)
- `"fourier"`: base de Fourier (requiere `skfda`)
- `"fpca"`: Análisis de Componentes Principales Funcional (requiere `skfda`)
- `"precomputed"`: los datos ya son coeficientes (passthrough)

**Uso:**
```python
from model_psbp_fd.functions_models import FunctionalRepresentation

# Configuración
repre = FunctionalRepresentation(
    method="bspline",
    n_basis=10,
    order=4,                    # orden del spline (4 = cúbico)
    domain=(0, 1)               # opcional, se infiere si None
)

# Ajustar y transformar
THETA = repre.fit_transform(Y, grid)  # Y: (T, G) → THETA: (T, K)

# Reconstruir para diagnóstico
Y_hat = repre.reconstruct(THETA)

# Error de reconstrucción
error = repre.reconstruction_error(Y, grid)
print(f"RMSE medio: {error['rmse_mean']:.4f}")
```

#### Funcionalidad 2: `FunctionalStandardizer`

Estandariza los coeficientes funcionales `THETA (T, K)` después de la representación.

**Métodos soportados:**
| Método | Transformación | Cuándo usar |
|--------|---------------|-------------|
| `"zscore"` | `(x - mean) / std` | Por defecto, datos con escala homogénea |
| `"minmax"` | `(x - min) / range` | Coeficientes en un rango conocido |
| `"robust"` | `(x - median) / IQR` | Datos con outliers |
| `"none"` | identidad | Cuando los coeficientes ya están estandarizados |

**Uso:**
```python
from model_psbp_fd.functions_models import FunctionalStandardizer

# Configuración
scaler = FunctionalStandardizer(method="zscore")

# Ajustar y transformar
THETA_s = scaler.fit_transform(THETA)  # THETA: (T, K) → THETA_s: (T, K)

# Invertir (para predicciones)
pred_original = scaler.inverse_transform(pred_estandarizado)
```