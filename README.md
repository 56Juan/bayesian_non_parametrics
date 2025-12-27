# Bayesian Nonparametrics

Este repositorio reúne el desarrollo de **modelos bayesianos no paramétricos aplicados a problemas reales de predicción, segmentación y modelado de heterogeneidad**, con foco en **Procesos de Dirichlet y sus extensiones dependientes**.

El proyecto está orientado a **escenarios donde los modelos clásicos resultan insuficientes**, especialmente cuando es necesario **capturar heterogeneidad latente y cuantificar explícitamente la incertidumbre de las predicciones**.

---

## 💼 ¿Qué tipo de problemas aborda este repositorio?

Los modelos implementados están pensados para casos como:

- Estimación de densidades con **intervalos de credibilidad**
- Modelos predictivos con **heterogeneidad latente**
- Datos con **colas pesadas, asimetrías o multimodalidad**
- Efectos **no lineales dependientes de covariables**
- Evolución temporal de patrones (series de tiempo funcionales)
- Requerimientos de **interpretabilidad** y medición de incertidumbre

---

## 🧩 Estructura del proyecto

La branch `main` funciona como **punto de entrada** al repositorio.  
Cada branch representa una **línea de modelado independiente**, diseñada para adaptarse a distintos **casos de uso analíticos y de negocio**.

### 🔹 Branches principales

- **`model_dpm`** *(LISTA)*  
  Modelos de mezcla basados en **Procesos de Dirichlet**, orientados a:
  - Estimación de densidades
  - Construcción de **intervalos de credibilidad**
  
  Implementados mediante **MCMC** y kernels **Normal** y **Laplace**.

- **`model_ddp`** *(EN DESARROLLO)*  
  Modelos avanzados de **Procesos de Dirichlet Dependientes**, enfocados en:
  - Problemas de **regresión**
  - Obtención de **incertidumbre predictiva**
  - Personalización por covariables
  - Modelos predictivos adaptativos
  - Captura de **heterogeneidad estructural**
  
  Incluye optimización computacional mediante **Python + C++**.

- **`model_time_series_fd`** *(INACTIVA)*  
  Extensión de los modelos anteriores a **series de tiempo funcionales y multivariadas**, útiles para:
  - Análisis de comportamiento en el tiempo
  - Evolución de riesgo
  - Patrones dinámicos complejos

- **`cookiecutter-setup`** *(SANDBOX)*  
  Estructura reutilizable para levantar rápidamente nuevos proyectos analíticos siguiendo buenas prácticas.

---

## ⚙️ Enfoque técnico (resumido)

- Inferencia completamente bayesiana
- Enfoque **no paramétrico**
- Cuantificación explícita de incertidumbre
- Modelos modulares y extensibles
- Código orientado a reutilización y escalabilidad
- Separación clara entre modelado, simulación y reporting

---

## 🚀 Cómo usar este repositorio

Este repositorio está pensado como:

- Base para **prototipos analíticos avanzados**
- Soporte para **modelos productivos complejos**
- Evidencia de **capacidad técnica en modelado estadístico aplicado**

Cada branch incluye ejemplos, simulaciones y documentación específica para su uso y adaptación.

---

## 📌 Estado del proyecto

Repositorio en desarrollo activo.  
Algunas branches contienen avances incrementales y validaciones en curso.

