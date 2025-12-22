#  Proceso de Dirichlet Dependientes

Este repositorio implementa la estimación de densidades mediante un **modelos de procesos de dirichlet dependientes**. 

Los modelos implemenentados se basan en los trabajos de:

- Modelo DDP con dependencia en Pesos con Kernel Probit/Logit (LSBP-PSBP), porpuesto en [Chung
y Dunson, 2009](ref.bib)
- Modelo DDP con dependencia en Pesos con Kernel Gaussiano (KSBP), propuesto en  [](ref.bib)
- Modelo DDP con dependencia en Atomos con  


## Modelos

### Modelo LSBP
En el modelo LSBP se utilizaron dos estructuras una basadas en mezclas de normales y otras en mezclas de Laplace, se busco la conjugacion para priorizar la optimizacion del metodo. 

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

**Modelo con kernel Laplace y Probit Stick-Breaking:**