# Ideas de modelos - Proceso de Dirichlet Dependientes
Este archivo presenta solo ideas de posibles modelos. Se basan en los trabajos de:

- Modelo DDP con dependencia en Pesos con Kernel Probit/Logit (LSBP-PSBP), porpuesto en [Chung y Dunson, 2009](ref.bib)
- Modelo DDP con dependencia en Pesos con Kernel Gaussiano (KSBP), propuesto en  [Ren et al., 2011](ref.bib)
- Modelo DDP con dependencia Lineal en Atomos, propuesto en [Iorio et al., 2004 ](ref.bib)
- Modelo DDP con dependencia Espacial en Atomos, propuesto en [Gelfand, Kottas y MacEachern, 2005](ref.bib)  

## Modelos

---

### Modelo LSBP (Logit Stick-Breaking Process)

En el modelo LSBP, inducimos una representación funcional donde los **coeficientes de proyección** en bases ortonormales pasan a ser nuestras covariables. Para covariables funcionales $X_i(t)$, proyectamos en bases ortonormales $\{\psi_k(t)\}_{k=1}^p$:

$$
\xi_{ik} = \langle X_i, \psi_k \rangle, \quad \mathbf{\xi}_i = (\xi_{i1}, \dots, \xi_{ip})^\top
$$

Los pesos se modelan con un **kernel logístico** que depende de estos coeficientes:

$$
v_h(\mathbf{\xi}_i) = \text{logit}^{-1}\left(\alpha_h + \boldsymbol{\beta}_h^\top \mathbf{\xi}_i\right) = \frac{\exp(\alpha_h + \boldsymbol{\beta}_h^\top \mathbf{\xi}_i)}{1 + \exp(\alpha_h + \boldsymbol{\beta}_h^\top \mathbf{\xi}_i)}
$$

La medida aleatoria funcional es:

$$
G_{\mathbf{\xi}} = \sum_{h=1}^{\infty} w_h(\mathbf{\xi}) \delta_{\theta_h}, \quad w_h(\mathbf{\xi}) = v_h(\mathbf{\xi}) \prod_{\ell=1}^{h-1} (1 - v_\ell(\mathbf{\xi}))
$$

#### Ventaja con Bases Ortonormales
Cuando las bases son ortonormales, la representación es **eficiente e identificable**, ya que los coeficientes $\xi_{ik}$ no están correlacionados y capturan direcciones ortogonales de variación.

---

### Modelo PSBP (Probit Stick-Breaking Process)

Similar al LSBP, pero usando la **función probit** para modelar los pesos:

$$
v_h(\mathbf{\xi}_i) = \Phi\left(\alpha_h + \boldsymbol{\beta}_h^\top \mathbf{\xi}_i\right)
$$

donde $\Phi(\cdot)$ es la CDF normal estándar. La representación con variables latentes es:

$$
z_{ih} \sim \mathcal{N}(\alpha_h + \boldsymbol{\beta}_h^\top \mathbf{\xi}_i, 1), \quad v_h(\mathbf{\xi}_i) = \mathbb{P}(z_{ih} > 0)
$$

Esta formulación permite **Gibbs sampling eficiente** debido a la conjugación normal.

---

### Modelo KSBP (Kernel Stick-Breaking Process)

El KSBP extiende el enfoque utilizando **kernels gaussianos (RBF)** para modelar dependencias no lineales en el espacio de coeficientes:

$$
v_h(\mathbf{\xi}_i) = \Phi\left(\alpha_h + f_h(\mathbf{\xi}_i)\right)
$$

donde $f_h(\cdot) \sim \text{GP}(0, K_h(\cdot, \cdot; \boldsymbol{\eta}_h))$ es un Proceso Gaussiano con kernel RBF:

$$
K_h(\mathbf{\xi}, \mathbf{\xi}') = \sigma_h^2 \exp\left(-\frac{\|\mathbf{\xi} - \mathbf{\xi}'\|^2}{2\ell_h^2}\right)
$$

#### Caso Espacial-Funcional
Para datos con ubicaciones $\mathbf{s}_i$ y covariables funcionales, se puede definir:

$$
v_h(\mathbf{\xi}_i, \mathbf{s}_i) = \Phi\left(\alpha_h + f_h(\mathbf{\xi}_i) + g_h(\mathbf{s}_i)\right)
$$

con $f_h \sim \text{GP}$ sobre coeficientes y $g_h \sim \text{GP}$ sobre espacio.

---

### Modelo DDP Lineal Enfoque A

El modelo **DDP Lineal** se extiende para manejar covariables funcionales $X_i(t) = (x_{i1}(t), \dots, x_{im}(t))$, donde cada $x_{ij}(t)$ representa la evolución temporal de la covariable $j$, con $j=1,\dots,m$.

#### Representación mediante bases

Cada covariable funcional se proyecta en un conjunto de bases $\{\psi_k(t)\}_{k=1}^p$:

$$
X_i(t) \approx \sum_{k=1}^p \xi_{ik} \psi_k(t)
$$

donde $\xi_{ik} = \langle X_i, \psi_k \rangle$ son los coeficientes de proyección. Similarmente, los coeficientes funcionales se expanden como:

$$
\beta_i(t) = \sum_{k=1}^p \lambda_{ik} \psi_k(t)
$$

#### Modelo Jerárquico

La respuesta $Y_i$ se modela mediante la representación discreta del producto interno funcional:

$$
Y_i \stackrel{\text{ind}}{\sim} F\left(\sum_{k=1}^p \lambda_{ik} \xi_{ik}, \epsilon_i\right)
$$

que equivale a:

$$
Y_i = \lambda_i^\top \xi_i + \epsilon_i
$$

donde $\lambda_i = (\lambda_{i1}, \dots, \lambda_{ip})^\top$ y $\xi_i = (\xi_{i1}, \dots, \xi_{ip})^\top$.

Los coeficientes $\lambda_i$ siguen un **Dirichlet Process**:

$$
\lambda_i \sim G, \quad G \sim \text{DP}(M, G_0)
$$

con medida base $G_0 = \mathcal{N}_p(\mu_0, \Sigma_0)$.


Ademas:
- $M \sim \text{Gamma}(a_M, b_M)$ (parámetro de concentración del DP)
- $\mu_0, \Sigma_0$ ~ $\pi$

Esta formulación mantiene la estructura **lineal en parámetros** del DDP tradicional, pero permite modelar relaciones funcionales mediante la representación de bases.

---

### Modelo DDP Lineal Enfoque B 
Un enfoque alternativo en ves proyectar en bases se modelan los coeficientes funcionales $\beta_i(t)$ directamente como **procesos estocásticos**, se obtiene un modelo **DDP funcional** (no lineal en el sentido tradicional).

#### Modelo Jerárquico

$$
Y_i \stackrel{\text{ind}}{\sim} F\left(\int \beta_i(t) X_i(t) dt, \epsilon_i\right)
$$

donde los coeficientes funcionales siguen:

$$
\beta_i(\cdot) \sim G, \quad G \sim \text{DP}(M, G_0)
$$

con medida base $G_0 = \text{GP}\big(0, K(t, t';\boldsymbol{\eta})\big)$, siendo $K$ un kernel de covarianza (e.g., exponencial, Matérn, RBF).

Ademas:
- $M \sim \text{Gamma}(a_M, b_M)$ (concentración del DP)
- $\boldsymbol{\eta}$: parámetros del kernel (rango $\rho$, varianza $\sigma^2$, suavidad $\nu$)


#### Representación discreta para implementación

En la práctica, las funciones se observan en puntos discretos $\{t_1, \dots, t_T\}$, obteniendo vectores $\mathbf{X}_i = (X_i(t_1), \dots, X_i(t_T))^\top$ y $\boldsymbol{\beta}_i = (\beta_i(t_1), \dots, \beta_i(t_T))^\top$. La integral se aproxima mediante cuadratura:

$$
\int \beta_i(t) X_i(t) dt \approx \boldsymbol{\beta}_i^\top \mathbf{W} \mathbf{X}_i
$$

donde $\mathbf{W}$ es una matriz diagonal de pesos de cuadratura.

---

### Modelo DDP Espacial 

El modelo espacial DDP se formula mediante una **descomposición en bases**, donde la respuesta $Y_i$ en la ubicación $\mathbf{s}_i$ se modela como:

$$
Y_i \stackrel{\text{ind}}{\sim} F\left(\sum_{k=1}^p \lambda_{ik} \xi_{ik} + \theta(\mathbf{s}_i), \epsilon_i\right)
$$

donde:

- $\xi_{ik} = \langle X_i, \psi_k \rangle$ son las proyecciones de las covariables funcionales $X_i(t)$ en las bases $\{\psi_k(t)\}_{k=1}^p$
- $\lambda_{ik}$ son coeficientes específicos por sujeto y base
- $\theta(\mathbf{s}_i)$ es un **efecto espacial** en la ubicación $\mathbf{s}_i$

Con:
- $M_1, M_2 \sim \text{Gamma}$: parámetros de concentración de los DPs
- $\boldsymbol{\eta}_\theta$: parámetros del kernel espacial (rango $\rho_\theta$, varianza $\sigma_\theta^2$)
- Parámetros de las medidas base $G_{0\lambda}$ y $G_{0\theta}$
#### Estructura Jerárquica del Modelo

1. **Componente Funcional:**
   $$
   \lambda_i = (\lambda_{i1}, \dots, \lambda_{ip})^\top \sim G, \quad G \sim \text{DP}(M_1, G_{0\lambda})
   $$

2. **Componente Espacial:**
   $$
   \theta(\mathbf{s}) \sim H, \quad H \sim \text{DP}(M_2, G_{0\theta})
   $$
   con medida base $G_{0\theta} = \text{GP}(0, K_{\theta}(\mathbf{s}, \mathbf{s}'; \boldsymbol{\eta}_\theta))$

3. **Distribución Conjunta:** Ambas componentes pueden modelarse como **DDP independientes** o con **dependencia cruzada**.

#### Formulación Computacionalmente Eficiente

Para la implementación, el efecto espacial se discretiza en las $n$ ubicaciones observadas:

$$
\boldsymbol{\theta} = (\theta(\mathbf{s}_1), \dots, \theta(\mathbf{s}_n))^\top
$$

con distribución previa:

$$
\boldsymbol{\theta} \sim \mathcal{N}_n(\mathbf{0}, \mathbf{K}_{\theta})
$$

donde $\mathbf{K}_{\theta}[i,j] = K_{\theta}(\mathbf{s}_i, \mathbf{s}_j; \boldsymbol{\eta}_\theta)$.


