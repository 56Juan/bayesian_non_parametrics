# Mezcla de Proceso de Dirichlet Completamente Aleatorizado

Este repositorio implementa la estimación de densidades mediante un **modelo de mezclas de procesos de Dirichlet completamente aleatorizado**, utilizando el **algoritmo de slice sampling** propuesto por (referenciar).

## Modelos
Se elaboraron dos versiones del modelo, que difieren en el kernel utilizado, pero siguen la **misma estructura de jerarquía aleatorizada**.

**Kernels utilizados:**
- **Normal**: `y_i ∣ θ_i ∼ N(μ_i, σ_i²)`  
- **Laplace**: `y_i ∣ θ_i ∼ Laplace(μ_i, b_i)`

**Nivel de cluster:**  
`θ_i ∣ G ∼ G`

**Proceso de Dirichlet:**  
`G ∣ M, G_0 ∼ DP(M, G_0)`

**Parámetro de concentración:**  
`M ∼ Gamma(α_M, β_M)`

**Medida base:**  
`G_0 = Normal-Inversa-Gamma(μ_0, κ_0, a_0, b_0)`

**Hiperparámetros de la medida base:**  

`μ_0 ∼ N(m_0, s_0²)`  
`κ_0 ∼ Gamma(α_κ, β_κ)`  
`a_0 ∼ Gamma(α_a, β_a)`  
`b_0 ∼ Gamma(α_b, β_b)`

## Estructura del Repositorio
```
model_dpm/
├── model_dpm/
│   ├── graphics/              # Clases y módulos de gráficas
│   ├── models/                # Clases de los modelos DPM
│   ├── simulations/           # Clases y lógica de simulación
│   ├── utils/                 # Módulos de utilidades
│   └── __init__.py           # Imports internos del paquete
│
├── data/                      # Datos (reales y/o simulados)
│   ├── reales/
│   └── simulaciones/
│
├── notebooks/
│   ├── simulaciones/
│   └── reales/
│
├── reports/
│   ├── simulaciones/
│   └── reales/
│
├── references/                # Documentos de referencia
│
├── versioning/                # Control experimental
│   ├── config.yaml
│   ├── experiment_registry.md
│   └── changelog.md
│
├── environment.yml            # Configuración de entorno
├── pyproject.toml             # Configuración del proyecto
├── ref.bib                    # Bibliografía formal
├── README.md                  # Documento principal
└── __init__.py                # Marca DPM como unidad lógica
```

## Extra

Este repositorio tiene la finalidad de documentar mi proyecto de tesis. No sigue un formato estándar de documentación, por lo que **no se incluyeron módulos de ETL ni pipelines de validación de datos**.  

Se crea un archivo `__init__.py` para permitir la interacción con otras branches en caso de futuras extensiones.