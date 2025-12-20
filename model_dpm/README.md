# Mezcla de Proceso de Dirichlet Completamente Aleatorizado

Este repositorio implementa la estimación de densidades mediante un **modelo de mezclas de procesos de Dirichlet completamente aleatorizado**, utilizando el **algoritmo de slice sampling** propuesto por (referenciar).

## Modelos
Se elaboraron dos versiones del modelo, que difieren en el kernel utilizado, pero siguen la **misma estructura de jerarquía aleatorizada**.

**Kernels utilizados:**
- **Normal**: \(y_i \mid \theta_i \sim \mathcal{N}(\mu_i, \sigma_i^2)\)  
- **Laplace**: \(y_i \mid \theta_i \sim \text{Laplace}(\mu_i, b_i)\)

**Nivel de cluster:**  
\(\theta_i \mid G \sim G\)

**Proceso de Dirichlet:**  
\(G \mid M, G_0 \sim DP(M, G_0)\)

**Parámetro de concentración:**  
\(M \sim \text{Gamma}(\alpha_M, \beta_M)\)

**Medida base:**  
\(G_0 = \text{Normal-Inversa-Gamma}(\mu_0, \kappa_0, a_0, b_0)\)

**Hiperparámetros de la medida base:**  
\[
\begin{aligned}
\mu_0 &\sim \mathcal{N}(m_0, s_0^2) \\
\kappa_0 &\sim \text{Gamma}(\alpha_\kappa, \beta_\kappa) \\
a_0 &\sim \text{Gamma}(\alpha_a, \beta_a) \\
b_0 &\sim \text{Gamma}(\alpha_b, \beta_b)
\end{aligned}
\]

## Estructura del Repositorio
dirichlet_mixture_process/
├── graphics/       : Módulos para generación de gráficas
├── models/         : Clases de los modelos
└── simulations/    : Módulos de simulaciones

data/
├── reales/         : Datos reales
└── simulaciones/   : Datos simulados

notebooks/
├── reales/         : Implementaciones con datos reales
└── simulaciones/   : Implementaciones con datos simulados

reports/
├── reales/         : Reportes con datos reales
└── simulaciones/   : Reportes con datos simulados

references/         : Bibliografía y artículos
versioning/         : Control de versiones
environments/       : Archivos de configuración de entorno
ref.bib             : Archivo bibliográfico
README.md           : Descripción general del repositorio


## Extra

Este repositorio tiene la finalidad de documentar mi proyecto de tesis. No sigue un formato estándar de documentación, por lo que **no se incluyeron módulos de ETL ni pipelines de validación de datos**.  

Se crea un archivo `__init__.py` para permitir la interacción con otras branches en caso de futuras extensiones.


