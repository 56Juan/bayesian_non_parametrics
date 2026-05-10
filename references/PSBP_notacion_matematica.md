# Notación Matemática del Modelo PSBP
## (Probit Stick-Breaking Process Regression con Selección de Variables)
### Chung & Dunson (2009)

---

## 1. Modelo de Mezcla

La distribución condicional de la respuesta dado los covariables es:

$$f(y_i \mid \mathbf{x}_i) = \sum_{h=1}^{N} \pi_h(\mathbf{x}_i) \, \mathcal{N}\!\left(y_i \;\middle|\; \mathbf{x}_i^\top \boldsymbol{\beta}_h,\; \tau_h^{-1}\right)$$

donde $N$ es el número máximo de componentes (truncamiento).

---

## 2. Pesos Dependientes de Covariables (Probit Stick-Breaking)

Los pesos $\pi_h(\mathbf{x})$ se construyen mediante un proceso stick-breaking probit:

$$v_h(\mathbf{x}) = \Phi\!\left(\alpha_h - \sum_{j=1}^{p} \psi_{jh} \,|\, x_j - \Gamma_{jh} \,|\right), \quad h = 1, \ldots, N-1$$

$$\pi_1(\mathbf{x}) = v_1(\mathbf{x})$$

$$\pi_h(\mathbf{x}) = v_h(\mathbf{x}) \prod_{\ell=1}^{h-1}\left(1 - v_\ell(\mathbf{x})\right), \quad h = 2, \ldots, N-1$$

$$\pi_N(\mathbf{x}) = \prod_{\ell=1}^{N-1}\left(1 - v_\ell(\mathbf{x})\right)$$

**Interpretación:** $\Gamma_{jh}$ es la localización de referencia del componente $h$ en la dimensión $j$, y $\psi_{jh} \geq 0$ es el ancho de banda. Mayor $\psi_{jh}$ implica que el componente $h$ tiene influencia más localizada en el eje $j$.

---

## 3. Selección de Variables

Para cada componente $h$ y variable $j$, se introduce un indicador binario:

$$\gamma_{jh} \in \{0, 1\}$$

La función de regresión del componente $h$ solo incluye las variables activas:

$$\mu_h(\mathbf{x}_i) = \beta_{0h} + \sum_{j=1}^{p} \gamma_{jh} \, \beta_{jh} \, x_{ij}$$

Si $\gamma_{jh} = 0$: $\psi_{jh} = 0$ y $\beta_{jh} = 0$ (variable excluida del componente $h$).

---

## 4. Variables Latentes del Proceso Probit (Augmentación de datos)

Se introduce $Z_{ih}$ latente para facilitar el muestreo de $\alpha_h$, $\psi_{jh}$ y $\Gamma_{jh}$.

Para la observación $i$ asignada al componente $S_i$:

$$Z_{i\ell} \mid S_i \sim \begin{cases}
\mathcal{N}\!\left(m_\ell(\mathbf{x}_i),\, 1\right)\, \mathbf{1}[Z_{i\ell} < 0] & \text{si } \ell < S_i \\
\mathcal{N}\!\left(m_\ell(\mathbf{x}_i),\, 1\right)\, \mathbf{1}[Z_{i\ell} \geq 0] & \text{si } \ell = S_i
\end{cases}$$

donde:

$$m_\ell(\mathbf{x}_i) = \alpha_\ell - \sum_{j=1}^{p} \psi_{j\ell} \,|\, x_{ij} - \Gamma_{j\ell} \,|$$

La variable aumentada $W_{i\ell} = Z_{i\ell} + \sum_j \psi_{j\ell} |x_{ij} - \Gamma_{j\ell}|$ concentra la información sobre $\alpha_\ell$.

---

## 5. Asignaciones de Componente

$$P(S_i = h \mid \mathbf{x}_i, \boldsymbol{\beta}, \boldsymbol{\tau}) \propto \pi_h(\mathbf{x}_i) \cdot \mathcal{N}\!\left(y_i \mid \mathbf{x}_i^\top \boldsymbol{\beta}_h,\; \tau_h^{-1}\right)$$

---

## 6. Prior de Zellner-g sobre los Coeficientes

$$\boldsymbol{\beta}_h^{(\gamma)} \mid \tau_h, g \sim \mathcal{N}\!\left(\mathbf{0},\; \frac{n}{g} \tau_h^{-1} \left(\mathbf{X}_h^\top \mathbf{X}_h\right)^{-1}\right)$$

donde $\mathbf{X}_h$ es la submatriz de diseño con las variables activas en $h$ (más el intercepto), y $g \sim \text{Gamma}(a_g, b_g)$.

---

## 7. Prior sobre la Precisión por Componente

$$\tau_h \sim \text{Gamma}(a_\tau, b_\tau)$$

**Posterior conjugada:**

$$\tau_h \mid \cdot \sim \text{Gamma}\!\left(a_\tau + \frac{n_h}{2} + \frac{p_h + 1}{2},\;\; b_\tau + \frac{1}{2}\text{RSS}_h + \frac{1}{2n} g \,\boldsymbol{\beta}_h^\top \mathbf{X}_h^\top \mathbf{X}_h \boldsymbol{\beta}_h \right)$$

donde $n_h = |\{i : S_i = h\}|$, $p_h = \sum_j \gamma_{jh}$, y $\text{RSS}_h = \sum_{i: S_i = h}(y_i - \mathbf{x}_i^\top \boldsymbol{\beta}_h)^2$.

---

## 8. Prior sobre los Umbrales del Proceso Probit

$$\alpha_h \mid \mu \sim \mathcal{N}(\mu, 1), \quad h = 1, \ldots, N-1$$

$$\mu \sim \mathcal{N}(\mu_\mu, \tau_\mu^{-1})$$

**Posterior conjugada de** $\alpha_h$:

Sea $\mathcal{A}_h = \{i : S_i \geq h\}$ (observaciones que "sobrevivieron" hasta el componente $h$):

$$\alpha_h \mid \cdot \sim \mathcal{N}\!\left(\hat{m}_h,\; \hat{v}_h\right)$$

$$\hat{v}_h = \frac{1}{1 + |\mathcal{A}_h|}, \qquad \hat{m}_h = \hat{v}_h \left(\mu + \sum_{i \in \mathcal{A}_h} W_{ih}\right)$$

**Posterior conjugada de** $\mu$:

$$\mu \mid \cdot \sim \mathcal{N}\!\left(\frac{\tau_\mu \mu_\mu + \sum_h \alpha_h}{N + \tau_\mu},\;\; \frac{1}{N + \tau_\mu}\right)$$

---

## 9. Prior sobre los Anchos de Banda

$$\psi_{jh} \mid \gamma_{jh} = 1 \sim \mathcal{N}_+\!\left(\mu_{\psi_j},\; \tau_{\psi_j}^{-1}\right)$$

(Normal truncada a $[0, +\infty)$)

**Posterior conjugada (Normal truncada):**

Sea $T_{ijh} = \alpha_h - Z_{ih} - \sum_{j' \neq j} \psi_{j'h} |x_{ij'} - \Gamma_{j'h}|$:

$$\hat{v}_{jh} = \left(\tau_{\psi_j} + \sum_{i \in \mathcal{A}_h} (x_{ij} - \Gamma_{jh})^2\right)^{-1}$$

$$\hat{m}_{jh} = \hat{v}_{jh} \left(\tau_{\psi_j} \mu_{\psi_j} + \sum_{i \in \mathcal{A}_h} T_{ijh} \,|\, x_{ij} - \Gamma_{jh} \,|\right)$$

$$\psi_{jh} \mid \cdot \sim \mathcal{N}_+\!\left(\hat{m}_{jh},\; \hat{v}_{jh}\right)$$

---

## 10. Prior sobre los Indicadores de Selección

$$\gamma_{jh} \mid \pi_j \sim \text{Bernoulli}(\pi_j)$$

$$\pi_j \mid w_j = 1 \sim \text{Beta}(a_{\pi_j}, b_{\pi_j})$$

**Posterior conjugada de** $\pi_j$ (si $w_j = 1$):

$$\pi_j \mid \cdot \sim \text{Beta}\!\left(a_{\pi_j} + \sum_h \gamma_{jh},\;\; b_{\pi_j} + N - \sum_h \gamma_{jh}\right)$$

---

## 11. Indicador Global de Relevancia

$$w_j \sim \text{Bernoulli}(p_w)$$

Si $w_j = 0$ entonces $\pi_j = 0$ (variable $j$ excluida globalmente).

---

## 12. Posterior de gamma_jh (Factor de Bayes analítico)

Para $h < N$, la probabilidad posterior de inclusión se obtiene comparando:

$$\log P(\gamma_{jh} = 1 \mid \cdot) \propto \log \pi_j + \log \mathcal{L}_{\text{marg}}(y_h, \beta_{jh}) + \log \mathcal{L}(Z_{ih} \mid \psi_{jh})$$

$$\log P(\gamma_{jh} = 0 \mid \cdot) \propto \log(1 - \pi_j) + \log \mathcal{L}(y_h \mid \beta_{-j}) + \log \mathcal{L}(Z_{ih} \mid \psi_{jh} = 0)$$

donde $\mathcal{L}_{\text{marg}}$ denota la verosimilitud marginalizada analíticamente sobre $\beta_{jh}$ (resultado de una integral Gaussiana-Gaussiana).

$$\gamma_{jh} \mid \cdot \sim \text{Bernoulli}\!\left(\frac{1}{1 + \exp(\log B_0 - \log B_1)}\right)$$

---

## 13. Predicción

La predicción para una nueva observación $\mathbf{x}^*$ es:

$$\hat{y}(\mathbf{x}^*) = \mathbb{E}[y \mid \mathbf{x}^*, \text{datos}] \approx \frac{1}{T - T_{\text{burn}}} \sum_{t=T_{\text{burn}}+1}^{T} \sum_{h=1}^{N} \pi_h^{(t)}(\mathbf{x}^*) \cdot \mathbf{x}^{*\top} \boldsymbol{\beta}_h^{(t)}$$

En el espacio original (desnormalizado):

$$\hat{y}_{\text{original}} = \bar{y}_{\text{train}} + s_{\text{train}} \cdot \hat{y}_{\text{std}}$$

---

## 14. Resumen de Hiperparámetros Utilizados en el Código

| Símbolo | Valor en código | Descripción |
|---------|----------------|-------------|
| $N$ | 20 | Truncamiento del proceso |
| $M$ | 50 | Puntos de grilla para $\Gamma_{jh}$ |
| $a_\tau = b_\tau$ | 0.5 | Prior Gamma para $\tau_h$ |
| $a_g = b_g$ | 0.5 | Prior Gamma para $g$ |
| $a_{\pi_j}$ | 1 | Prior Beta para $\pi_j$ |
| $b_{\pi_j}$ | 5 | Prior Beta para $\pi_j$ (penaliza inclusión) |
| $\mu_\mu$ | 0 | Media de la prior sobre $\mu$ |
| $\tau_\mu$ | 1 | Precisión de la prior sobre $\mu$ |
| $\tau_{\psi_j}$ | 1 | Precisión de la prior sobre $\psi_{jh}$ |
| $\mu_{\psi_j}$ | 0 | Media de la prior sobre $\psi_{jh}$ |
| $p_w$ | 0.5 | Prior Bernoulli para $w_j$ |
