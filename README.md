#  Proceso de Dirichlet Dependientes

Este repositorio implementa la estimación de densidades mediante un **modelos de procesos de dirichlet dependientes**. 

Los modelos implemenentados se basan en los trabajos de:

- Modelo DDP con dependencia en Pesos con Kernel Probit/Logit (LSBP-PSBP), porpuesto en [Chung y Dunson, 2009](ref.bib)
- Modelo DDP con dependencia en Pesos con Kernel Gaussiano (KSBP), propuesto en  [Ren et al., 2011](ref.bib)
- Modelo DDP con dependencia Lineal en Atomos, propuesto en [Iorio et al., 2004 ](ref.bib)
- Modelo DDP con dependencia Espacial en Atomos, propuesto en [Gelfand, Kottas y MacEachern, 2005](ref.bib)  


## Modelos
Los modelos incluyen algunas variantes de los artículos originales. Además, para lograr una implementación computacionalmente eficiente, parte del código se desarrolló en C++, optimizando así el procesamiento.

### Modelo LSBP
En el modelo LSBP se utilizaron dos estructuras una basadas en mezclas de normales y otras en mezclas de Laplace, se busco la conjugacion para priorizar la optimizacion del metodo. 

**Modelo con kernel Normal y Logit Stick-Breaking: MODELO LISTO - EVALUAR Y COMPARAR CON OTROS** 
```
# Likelihood
y_i | z_i = h, μ_h, σ²_h ~ N(μ_h, σ²_h)

# Asignaciones dependientes
z_i | x_i, {w_h(x_i)} ~ Categorical(w_1(x_i), ..., w_T(x_i))

# Pesos dependientes (Logit stick-breaking)
w_h(x) = v_h(x) ∏_{ℓ<h} (1 - v_ℓ(x))
v_h(x) = logit⁻¹(η_h(x)) = exp(η_h(x)) / [1 + exp(η_h(x))]

# Predictor lineal con kernel de dependencia
η_h(x) = α_h - Σ_j ψ_{hj} |x_j - ℓ_{hj}|

# Priors para dependencia
α_h ~ N(μ, 1)
ψ_{hj} ~ N⁺(μ_ψ_j, τ⁻¹_ψ_j)
ℓ_{hj} ~ Discrete-Uniform{ℓ*_{jm}}_{m=1}^{M_j}

# Átomos globales (Normal-Inverse-Gamma)
σ²_h ~ InvGamma(a₀, b₀)
μ_h | σ²_h ~ N(μ₀, σ²_h/κ₀)

# Hiperparámetros
μ ~ N(μ_μ, τ⁻¹_μ)
μ₀ ~ N(m₀, s₀²)
κ₀ ~ Gamma(α_κ, β_κ)
a₀ ~ Gamma(α_a, β_a)
b₀ ~ Gamma(α_b, β_b)
```

**Modelo con kernel Laplace y Logit Stick-Breaking:** 
```
# Likelihood
y_i | z_i = h, μ_h, b_h ~ Laplace(μ_h, b_h)

# Asignaciones dependientes
z_i | x_i, {w_h(x_i)} ~ Categorical(w_1(x_i), ..., w_T(x_i))

# Pesos dependientes (Logit stick-breaking)
w_h(x) = v_h(x) ∏_{ℓ<h} (1 - v_ℓ(x))
v_h(x) = logit⁻¹(η_h(x)) = exp(η_h(x)) / [1 + exp(η_h(x))]

# Predictor lineal con kernel de dependencia
η_h(x) = α_h - Σ_j ψ_{hj} |x_j - ℓ_{hj}|

# Priors para dependencia
α_h ~ N(μ, 1)
ψ_{hj} ~ N⁺(μ_ψ_j, τ⁻¹_ψ_j)
ℓ_{hj} ~ Discrete-Uniform{ℓ*_{jm}}_{m=1}^{M_j}

# Átomos globales (Normal-Gamma para Laplace)
b_h ~ Gamma(a₀, β₀)
μ_h ~ N(μ₀, τ₀⁻¹)

# Hiperparámetros
μ ~ N(μ_μ, τ⁻¹_μ)
μ₀ ~ N(m₀, s₀²)
τ₀ ~ Gamma(α_τ, β_τ)
a₀ ~ Gamma(α_a, β_a)
β₀ ~ Gamma(α_β, β_β)
```

### Modelo PSBP
En el modelo PSBP se utilizaron dos estructuras una basadas en mezclas de normales y otras en mezclas de Laplace, se busco la conjugacion para priorizar la optimizacion del metodo. 

**Modelo con kernel Normal y Probit Stick-Breaking:**
``` 
# Likelihood
y_i | z_i = h, μ_h, σ²_h ~ N(μ_h, σ²_h)

# Asignaciones dependientes
z_i | x_i, {w_h(x_i)} ~ Categorical(w_1(x_i), ..., w_T(x_i))

# Pesos dependientes (Probit stick-breaking)
w_h(x) = v_h(x) ∏_{ℓ<h} (1 - v_ℓ(x))
v_h(x) = Φ(η_h(x)) = ∫_{-∞}^{η_h(x)} N(t|0,1) dt

# Variables latentes para conjugación probit
u_{ih} ~ N(η_h(x_i), 1), para i: z_i ≥ h
v_h(x_i) = I(u_{ih} > 0)
u_{ih} | v_h(x_i) ~ { N⁺(η_h(x_i), 1) si z_i = h
                    { N⁻(η_h(x_i), 1) si z_i > h

# Predictor lineal con kernel de dependencia
η_h(x) = α_h - Σ_j ψ_{hj} |x_j - ℓ_{hj}|

# Priors para dependencia
α_h ~ N(μ, 1)
ψ_{hj} ~ N⁺(μ_ψ_j, τ⁻¹_ψ_j)
ℓ_{hj} ~ Discrete-Uniform{ℓ*_{jm}}_{m=1}^{M_j}

# Priors con estructura de selección de variables
γ_{hj} ~ Bernoulli(κ_j)
ψ_{hj} | γ_{hj} = 0 = 0
ψ_{hj} | γ_{hj} = 1 ~ N⁺(μ_ψ_j, τ⁻¹_ψ_j)
κ_j ~ Beta(a_κ_j, b_κ_j)

# Átomos globales (Normal-Inverse-Gamma)
σ²_h ~ InvGamma(a₀, b₀)
μ_h | σ²_h ~ N(μ₀, σ²_h/κ₀)

# Hiperparámetros
μ ~ N(μ_μ, τ⁻¹_μ)
μ₀ ~ N(m₀, s₀²)
κ₀ ~ Gamma(α_κ, β_κ)
a₀ ~ Gamma(α_a, β_a)
b₀ ~ Gamma(α_b, β_b)
μ_ψ_j ~ N(m_ψ_j, s²_ψ_j)
τ_ψ_j ~ Gamma(α_τ_j, β_τ_j)
```
**Modelo con kernel Laplace y Probit Stick-Breaking:**


### Modelo DDP Lineal
En el modelo DDP Lineal se considero el uso de funciones para poder explicar componentes no lineares. De estas se selecionaron transformaciones Spline 


**Modelo con Kernel Normal**

**Modelo con Kernel Laplace** 



## Estructura del repositorio 
```
model_ddp/
├── model_ddp/
│   ├── fit/                   # Clases y módulos para evaluar el desempeño
│   ├── pipelines/             # Clases y módulos para separación de datos, filtros, etc.
│   ├── graphics/              # Clases y módulos de visualización
│   ├── models/                # Clases de los modelos DDP
│   │   ├── LSBP_normal_v1/                 # Ejemplo de Clase Python + C++
│   │   │   ├── __init__.py                 # Imports de funciones en C++
│   │   │   ├── LSBP_normal_v1.py           # Wrapper Python (clase pública)
│   │   │   ├── lsbp_cpp.cp312-win_amd64.pyd  # Módulo C++ compilado (pybind11)
│   │   │   ├── cpp/                        # Código fuente C++
│   │   │   │   ├── lsbp_core.cpp           # Implementación C++
│   │   │   │   ├── lsbp_core.hpp           # Headers
│   │   │   │   ├── bindings.cpp            # Bindings pybind11
│   │   │   │   └── CMakeLists.txt          # Configuración de build C++
│   │   ├── .../
│   │   ├── __init__.py                     # Imports internos del paquete
│   ├── simulations/           # Clases y lógica de simulación
│   ├── utils/                 # Módulos de utilidades
│   └── __init__.py            # Imports internos del paquete
│
├── artefact/
│   ├── reales/
│   │   └── models/            # Modelos entrenados (objetos .pkl reutilizables)
│   └── simulaciones/
│       └── models/
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
├── pyproject.toml             # ÚNICO punto de build del proyecto
├── environment.yml            # Configuración del entorno conda
├── ref.bib                    # Bibliografía formal
├── id_model.md                # Documento de ideas
├── README.md                  # Documento principal del proyecto
└── __init__.py                # Marca el proyecto como unidad lógica
```

## Aplicacion real 

## Extra

Este repositorio tiene la finalidad de documentar mi proyecto de tesis. No sigue un formato estándar de documentación, por lo que **no se incluyeron módulos de ETL**.  

Se crea un archivo `__init__.py` para permitir la interacción con otras branches en caso de futuras extensiones.