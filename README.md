# Modelo PSBP extendido a prediccion de Series de tiempo Funcionales

> **PSBP extendido a series de tiempo funcionales**  
> Basado en Chung & Dunson (2009), *JASA* 104(488).

Este paquete implementa una extensión funcional del modelo **Product of Stick-Breaking Process (PSBP)** para series de tiempo, permitiendo trabajar con curvas funcionales discretas mediante representación en bases (B-splines, Fourier, FPCA) y estandarización de coeficientes.

## 🏗️ Estructura del proyecto

```
model_psbp_fd/
│
├── model_psbp_fd/                        # Paquete principal
│   ├── __init__.py
│   │
│   ├── functions_models/                 # Clases auxiliares de preprocesamiento
│   │   ├── __init__.py
│   │   ├── functions_standarize.py       # Estandarización de coeficientes
│   │   └── functions_repre_functional.py # Representación funcional (bases)
│   │
│   ├── models/                           # Versiones del modelo PSBP-FD
│   │   ├── __init__.py
│   │   ├── CHANGELOG.md
│   │   │
│   │   ├── psbp_fd_v1/                   # Primera versión funcional
│   │   │   ├── __init__.py
│   │   │   ├── psbp_fd_v1.py             # Clase principal: __init__, fit(), predict()
│   │   │   ├── functions/
│   │   │   │     ├── sampler.py          # Muestreador MCMC
│   │   │   │     └── predict.py          # Funciones de predicción
│   │   │   ├── psbp_fd_v1.cp312-win_amd64.pyd
│   │   │   └── cpp/                      # Extensiones C++ (pybind11)
│   │   │       ├── psbp_fd_v1.cpp
│   │   │       ├── psbp_fd_v1.hpp
│   │   │       ├── bindings.cpp
│   │   │       └── CMakeLists.txt
│   │   │
│   │   └── psbp_fd_vx/                   # Versión experimental/futura
│   │       └── ...
│   │
│   ├── pipelines/                        # Composición data → modelo → evaluación
│   │   ├── simulation_pipeline.py
│   │   └── real_data_pipeline.py
│   │
│   ├── fit/                              # Diagnósticos y métricas
│   │   ├── diagnostics.py                # R-hat, ESS, trace plots
│   │   ├── posterior_checks.py           # PPC funcionales
│   │   └── metrics.py                    # RMSE, LPML, WAIC, distancias L2
│   │
│   ├── graphics/                         # Visualización
│   │   ├── functional_plots.py
│   │   └── mcmc_plots.py
│   │
│   └── utils/                            # Utilidades transversales
│       └── helpers.py
│
├── artefact/                             # Objetos serializados (no versionados)
│   ├── reales/models/
│   └── simulaciones/models/
│
├── configs/                              # Configuración centralizada
│   ├── default.yaml
│   └── experiments/
│
├── data/                                 # Datos brutos y procesados
│   ├── reales/
│   └── simulaciones/
│
├── notebooks/                            # Análisis exploratorios
├── reports/                              # Outputs finales (figuras, tablas)
├── references/                           # Documentación de referencia
├── tests/                                # Tests unitarios
├── versioning/                           # Registro experimental manual
├── pyproject.toml
├── environment.yml
└── README.md
```

## 🔧 Módulo `functions_models`

El módulo `functions_models` proporciona dos clases auxiliares diseñadas con el **patrón de composición**, permitiendo al modelo principal (`PSBPFunctional`) delegar la representación funcional y la estandarización de manera modular, testeable y extensible.

### ¿Por qué separar representación y estandarización?

| Representación Funcional | Estandarización | Caso de uso recomendado |
|--------------------------|-----------------|------------------------|
| B-splines + Z-score | Datos suaves, coeficientes homogéneos | Series regulares, sin outliers |
| FPCA + Robust | Datos atípicos, reducción de dimensionalidad | Alta dimensionalidad, ruido estructurado |
| Fourier + MinMax | Datos periódicos, coeficientes acotados | Señales cíclicas, dominio temporal fijo |
| Precomputed + None | Datos ya procesados | Pipeline externo o reproducibilidad |

---

## 📐 1. `FunctionalRepresentation`

Convierte curvas discretas `Y (T, G)` en coeficientes funcionales `THETA (T, K)`, donde:
- `T` = número de observaciones (instantes de tiempo)
- `G` = puntos de grilla de evaluación
- `K` = número de coeficientes (funciones de base o FPCs retenidos)

### Métodos soportados

| Método | Descripción | Requiere `skfda` |
|--------|-------------|------------------|
| `"bspline"` | B-splines cúbicos (orden configurable) | ✅ |
| `"fourier"` | Base trigonométrica de Fourier | ✅ |
| `"fpca"` | Análisis de Componentes Principales Funcionales | ✅ |
| `"precomputed"` | Paso directo (datos ya son coeficientes) | ❌ |

### API

```python
from model_psbp_fd.functions_models import FunctionalRepresentation

# Inicializar
repre = FunctionalRepresentation(
    method="bspline",      # "bspline" | "fourier" | "fpca" | "precomputed"
    n_basis=10,            # número de funciones base (o FPCs)
    order=4,               # orden del spline (4 = cúbico). Solo bspline
    domain=(0, 1),         # (a, b) dominio. None → infiere de grid
    center_fpca=True,      # centrar curvas antes de FPCA
)

# Ajustar y transformar
THETA = repre.fit_transform(Y, grid)   # Y: (T, G) → THETA: (T, K)

# Reconstrucción (diagnóstico)
Y_hat = repre.reconstruct(THETA)       # THETA: (T, K) → Y_hat: (T, G)

# Error de reconstrucción
error = repre.reconstruction_error(Y, grid)
print(f"RMSE medio: {error['rmse_mean']:.4f}")
print(f"Error relativo medio: {error['rel_error_mean']:.4f}")

# Varianza explicada (solo FPCA)
if repre.method == "fpca":
    evr = repre.fpca_explained_variance()
    print(f"Varianza explicada acumulada: {np.cumsum(evr)[-1]*100:.1f}%")

# Serialización
params = repre.get_params()   # Nota: basis_ (skfda) requiere pickle por separado
```

### Flujo de trabajo

```
Y (T, G) ──► fit_transform() ──► THETA (T, K)
                  │
                  ▼
            [basis ajustada]
                  │
THETA (T, K) ──► reconstruct() ──► Y_hat (T, G)
```

### Notas importantes

- **Fourier**: `n_basis` debe ser impar (1 + 2·n_harmónicos). Si se pasa un valor par, se ajusta automáticamente al siguiente impar con un `UserWarning`.
- **FPCA**: `n_basis` controla `n_components` (FPCs retenidas). El objeto FPCA ajustado se guarda en `self.basis_`.
- **Precomputed**: `fit_transform()` actúa como passthrough. `reconstruct()` y `reconstruction_error()` lanzan `RuntimeError` porque no hay base funcional almacenada.
- **Validación de dimensiones**: `transform()` valida que `G` del nuevo array coincida con `G` del `fit()`, evitando errores crípticos de `matmul` de NumPy.

---

## ⚖️ 2. `FunctionalStandarizer`

Estandariza los coeficientes funcionales `THETA (T, K)` por columna, siguiendo el patrón `fit / transform / inverse_transform` de `sklearn`.

### Métodos soportados

| Método | Transformación | Cuándo usar |
|--------|---------------|-------------|
| `"zscore"` | `(x - μ) / σ` (ddof=0, poblacional) | Por defecto. Datos con escala homogénea |
| `"minmax"` | `(x - min) / (max - min)` | Coeficientes en rango conocido |
| `"robust"` | `(x - mediana) / IQR | Datos con outliers |
| `"none"` | Identidad | Coeficientes ya estandarizados |

### API

```python
from model_psbp_fd.functions_models import FunctionalStandarizer

# Inicializar
scaler = FunctionalStandarizer(method="zscore")   # "zscore" | "minmax" | "robust" | "none"

# Ajustar y transformar
THETA_s = scaler.fit_transform(THETA)   # THETA: (T, K) → THETA_s: (T, K)

# Invertir (predicciones en escala original)
pred = scaler.inverse_transform(pred_s)

# Serialización / deserialización
saved = {"scaler": scaler.get_params(), "chains": model.chains_}
# pickle.dump(saved, open("artefact/simulaciones/models/run01.pkl", "wb"))

# Recuperar sin datos originales
scaler = FunctionalStandarizer.from_params(saved["scaler"])
pred = scaler.inverse_transform(pred_s)
```

### Flujo de trabajo en el modelo

```
THETA (T, K) ──► fit_transform() ──► THETA_s (T, K)
                                          │
                                   [modelo PSBP]
                                          │
                                   pred_s (T, K)
                                          │
                                   inverse_transform()
                                          ▼
                                    pred (T, K)
```

### Notas importantes

- **Z-score**: usa `ddof=0` (estimador poblacional), consistente con `sklearn.StandardScaler`. Para muestras pequeñas (`T < 30`) la diferencia con `ddof=1` puede alcanzar un factor `√(T/(T-1))`.
- **Columnas constantes**: si `scale_ < 1e-12`, se reemplaza por `1.0` para evitar división por cero.
- **Validación de shape**: `inverse_transform()` valida que `K` de entrada coincida con `K` del `fit()`, evitando broadcasting silencioso de NumPy (especialmente crítico cuando `K=1` vs `K>1`).

---