#  Proceso de Dirichlet Dependientes

Este repositorio aborda problemas de regresion mediante **modelos de procesos de dirichlet dependientes**. 

Los modelos implementados se basan en los trabajos de:

- Modelo DDP con dependencia en Pesos con Kernel Probit/Logit (LSBP-PSBP), porpuesto en [Chung y Dunson, 2009](ref.bib)
- Modelo DDP con dependencia en Pesos con Kernel Gaussiano (KSBP), propuesto en  [Ren et al., 2011](ref.bib)
- Modelo DDP con dependencia Lineal en Atomos, propuesto en [Iorio et al., 2004 ](ref.bib)
- Modelo DDP con dependencia Espacial en Atomos, propuesto en [Gelfand, Kottas y MacEachern, 2005](ref.bib)  

## Modelos
Los modelos incluyen variantes de los artículos originales. Además, para lograr una implementación computacionalmente eficiente, parte del código se desarrolló en C++, optimizando así el procesamiento.

### Modelo LSBP
En el modelo LSBP se utilizaron dos estructuras: una basada en mezclas de normales y otra en mezclas de Laplace. Se buscó la conjugación para priorizar la optimización del método.

**Modelo con kernel Normal y Logit Stick-Breaking:** 
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
  donde: p(y_i | μ_h, b_h) = (1/2b_h) exp(-|y_i - μ_h|/b_h)

# Asignaciones dependientes
z_i | x_i, {w_h(x_i)} ~ Categorical(w_1(x_i), ..., w_T(x_i))

# Pesos dependientes (Logit stick-breaking)
w_h(x) = v_h(x) ∏_{ℓ<h} (1 - v_ℓ(x))
v_h(x) = logit⁻¹(η_h(x)) = exp(η_h(x)) / [1 + exp(η_h(x))]

# Predictor lineal con kernel de dependencia
η_h(x) = α_h - Σ_j ψ_{hj} |x_j - ℓ_{hj}|

# Priors para dependencia (idénticos a Normal)
α_h ~ N(μ_α, σ²_α)
ψ_{hj} ~ N⁺(μ_ψ, σ²_ψ)
ℓ_{hj} ~ Discrete-Uniform{ℓ*_{jm}}_{m=1}^{M_j}

# Átomos globales (parámetros Laplace)
b_h ~ Gamma(a₀, β₀)
μ_h ~ N(μ₀, τ₀⁻¹)

# Hiperparámetros Nivel 1
μ_α ~ N(m_α, s²_α)
μ₀ ~ N(m₀, s²₀)
τ₀ ~ Gamma(α_τ, β_τ)

# Hiperparámetros Nivel 2  
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
v_h(x) = Φ(η_h(x)) donde Φ(·) es la CDF Normal estándar

# Variables latentes de aumentación (Data Augmentation)
u_{ih} | η_h(x_i), z_i ~ TruncatedNormal(η_h(x_i), 1, truncation)
  donde truncation = [0, ∞) si z_i = h     (v_h = 1)
        truncation = (-∞, 0] si z_i > h     (v_h = 0)

# Predictor lineal con kernel de dependencia
η_h(x) = α_h - Σ_j γ_{hj} · ψ_{hj} · |x_j - ℓ_{hj}|

# Priors para parámetros de dependencia
α_h ~ N(μ_α, σ²_α)
ψ_{hj} | γ_{hj} = 1 ~ N⁺(μ_ψ_j, τ⁻¹_ψ_j)
ψ_{hj} | γ_{hj} = 0 = 0
ℓ_{hj} ~ Discrete-Uniform{ℓ*_{jm}}_{m=1}^{M_j}

# Spike-and-slab para selección de variables
γ_{hj} ~ Bernoulli(κ_j)
κ_j ~ Beta(a_κ_j, b_κ_j)

# Átomos globales (Normal-Inverse-Gamma)
σ²_h ~ InvGamma(a₀, b₀)
μ_h | σ²_h ~ N(μ₀, σ²_h/κ₀)

# Hiperparámetros Nivel 1
μ_α ~ N(m_α, s²_α)
μ₀ ~ N(m₀, s²₀)
κ₀ ~ Gamma(α_κ, β_κ)

# Hiperparámetros Nivel 2
a₀ ~ Gamma(α_a, β_a)
b₀ ~ Gamma(α_b, β_b)

# Hiperparámetros de kernel (por covariable)
μ_ψ_j ~ N(m_ψ_j, s²_ψ_j)
τ_ψ_j ~ Gamma(α_τ_j, β_τ_j)
```
### Modelo DDP Lineal
En el modelo DDP Lineal se considero el uso de funciones para poder explicar componentes no lineares. De estas se selecionaron transformaciones Spline 

**Modelo con Kernel Normal DP para Mu_h y Sigma_h**

```
# Likelihood
y_i | z_i = h, λ_h, ξ_h ~ N(μ_h(x_i), σ²_h(x_i))

# Funciones dependientes de x (lineales)
μ_h(x_i) = λ_h0 + λ_h1 x_i1 + ... + λ_hp x_ip = λ_h' x̃_i
log(σ²_h(x_i)) = ξ_h0 + ξ_h1 x_i1 + ... + ξ_hp x_ip = ξ_h' x̃_i

donde:
x̃_i = [1, x_i1, ..., x_ip]'  # Vector aumentado (p+1 dimensiones)
λ_h = [λ_h0, λ_h1, ..., λ_hp]' ∈ ℝ^(p+1)
ξ_h = [ξ_h0, ξ_h1, ..., ξ_hp]' ∈ ℝ^(p+1)

# Asignaciones a componentes
z_i | {w_h} ~ Categorical(w_1, w_2, ..., w_T)

# Pesos stick-breaking
w_h = v_h ∏_{ℓ<h} (1 - v_ℓ)
v_h ~ Beta(1, M)

# Coeficientes de media por componente
λ_h | μ_λ, Σ_λ ~ N_K(μ_λ, Σ_λ)

# Coeficientes de log-varianza por componente
ξ_h | μ_ξ, Σ_ξ ~ N_K(μ_ξ, Σ_ξ)

# Prior conjugado jerárquico para coeficientes de media
μ_λ | Σ_λ, m_λ, κ_λ ~ N_K(m_λ, Σ_λ/κ_λ)
Σ_λ ~ Inv-Wishart(ν_λ, Ψ_λ)

# Prior conjugado jerárquico para coeficientes de log-varianza
μ_ξ | Σ_ξ, m_ξ, κ_ξ ~ N_K(m_ξ, Σ_ξ/κ_ξ)
Σ_ξ ~ Inv-Wishart(ν_ξ, Ψ_ξ)

# Prior para concentración
M ~ Gamma(a_M, b_M)

# Hiperparámetros Nivel 1 (para media)
m_λ ~ N_K(μ_m, Σ_m)  # Media base de μ_λ
κ_λ ~ Gamma(α_κ, β_κ)  # Escala de μ_λ
ν_λ ~ Gamma(α_ν, β_ν)  # Grados de libertad de Σ_λ
Ψ_λ ~ Wishart(ν_Ψ, Ω_Ψ)  # Matriz de escala de Σ_λ

# Hiperparámetros Nivel 1 (para log-varianza)
m_ξ ~ N_K(μ_m, Σ_m)  # Media base de μ_ξ
κ_ξ ~ Gamma(α_κ, β_κ)  # Escala de μ_ξ
ν_ξ ~ Gamma(α_ν, β_ν)  # Grados de libertad de Σ_ξ
Ψ_ξ ~ Wishart(ν_Ψ, Ω_Ψ)  # Matriz de escala de Σ_ξ

# Hiperparámetros Nivel 1 (para concentración)
a_M ~ Gamma(α_aM, β_aM)  # Shape de concentración
b_M ~ Gamma(α_bM, β_bM)  # Rate de concentración
```

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

## Carpeta `artefact`

Los artefactos entrenados no se incluyen en el repositorio. En algunos casos, su tamaño supera los **2 GB**, lo que excede los límites operativos de GitHub para la subida de contenido.

Por este motivo, se optó por versionar **únicamente los reportes y resultados resumidos**, manteniendo fuera del repositorio los artefactos pesados.

El **código se mantiene completamente funcional** y preparado para generar y guardar dichos artefactos de forma local cuando sea necesario.

## Extra

Este repositorio tiene la finalidad de documentar mi proyecto de tesis. No sigue un formato estándar de documentación, por lo que **no se incluyeron módulos de ETL**.  

Se crea un archivo `__init__.py` para permitir la interacción con otras branches en caso de futuras extensiones.