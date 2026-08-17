# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Qué es

Tesis: extensión del **PSBP** (Probit Stick-Breaking Process, Chung & Dunson 2009) a **series de tiempo funcionales**, bajo el nombre **PSBPM-FD**. El paquete `model_psbp_fd` genera datos, preprocesa a base + FPCA, consume trazas MCMC y evalúa predicciones. El **muestreo MCMC ocurre en MATLAB**, no en Python.

Código, docstrings y nombres están en español; mantener ese idioma. Los docstrings de módulo son extensos a propósito: documentan *por qué* se tomó cada decisión numérica y suelen contener la respuesta antes que el código.

**El contexto de qué se va a simular está en `docs/`, y esa carpeta manda sobre el código.** Ver la sección [El estudio de simulación](#el-estudio-de-simulación-qué-pide-docs-y-qué-existe-hoy).

## Instalación

```bash
pip install -e .
```

Alternativa conda en `environment.yml` (entorno `psbp_fd`). **No hay suite de tests** en el repo pese a que `pytest` está declarado en `[dev]`; no inventar comandos de test. Tampoco hay linter ni CI configurados.

## Arquitectura: el ciclo Python → MATLAB → Python

Esto es lo que no se deduce leyendo un solo archivo. Cada experimento recorre tres etapas acopladas **por archivos en disco**, no por llamadas:

1. **`<NN>_01_simulaciones.ipynb`** (Python) — genera datos con `pipelines/sim_escenario_k`, ajusta representación en base + FPCA + estandarizador, arma los datasets AR y escribe todos los artefactos vía `pipelines/artifacts.py`.
2. **`psbp_fd_iteracion.m`** (MATLAB) — lee `hyperparameters.json` y `datasets_manifest.json`, arma una lista plana de jobs *(cadena × componente FPCA)* y los reparte con `parfor`; cada job llama a `psbp_train.m`. Entrena **solo con el bloque train**.
3. **`<NN>_03_resultados.ipynb`** (Python) — carga las trazas `.mat`, construye la predictiva funcional con `psbp_fd_v3` y evalúa con `fit/`.

`hyperparameters.json` es la **única fuente de verdad del contrato Python ↔ MATLAB** (`n_iter`, `mcmc_config`, `hyperparams_list`, `partition`). No hardcodear esos valores en el `.m`.

### Dónde vive cada artefacto

Los dos directorios no son intercambiables y el `.m` depende de la distinción:

- `data/.../processed/functional/<EXPERIMENT_ID>/` → `datasets_manifest.json`, `dataset_fpc_<idx>_{train,test}.csv`, CSV de FPCA, estandarizador, curvas `.npy`.
- `artefact/simulaciones/<EXPERIMENT_ID>/` → `hyperparameters.json`, `eval_config.json` y las trazas `.mat`.

`pipelines/artifacts.py` centraliza escritura y lectura emparejadas, con el dict `ARCHIVOS` como definición única de nombres. **No leer ni escribir estos artefactos a mano desde un notebook**: el módulo existe porque duplicar la convención en cada notebook fue la fuente de error más persistente del proyecto. `verificar_contrato()` cruza manifest, hiperparámetros y artefactos FPCA antes de analizar.

**No existe un `config_paths` en Python.** Cada notebook construye a mano el dict `PATHS` con las claves `raw`, `functional`, `predict`, `out_report`, `out_artefact`; `config_paths.m` replica esas mismas rutas del lado MATLAB. Hay **una copia del `.m` por carpeta de experimento**, así que cualquier cambio de convención se propaga a mano a todas.

### Versiones del modelo

- **`psbp_fd_v3` es la que se usa** y no tiene muestreador: trazas `.mat` → `PSBPPredictor` (predictiva por score) → `muestrear_scores` (agrupa cadenas) → `PropagadorFuncional` (des-estandariza + reconstruye FPCA) → muestras de curvas `(S, n, G)`, que son el insumo de `fit/`.
- `v1`/`v2` son heredadas. `models/__init__.py` las importa de forma tolerante pero trazable (`v1` depende de un `.pyd` compilado por plataforma); usar `estado_versiones()` para diagnosticar. `v3` se importa estricto a propósito.
- Ojo con el cambio de contrato v2→v3: `predict(return_std=True)` devolvía la desviación posterior **de la media condicional**; en v3 devuelve la **predictiva** (ley de varianza total). La cantidad vieja sigue disponible como `sd_centro`. Usar la de v2 como banda produce subcobertura que se confunde con fracaso del modelo.

## Gotchas

- **La semilla registrada no reproduce la que MATLAB usó.** Todos los `psbp_fd_iteracion.m` tienen `SEED_BASE = 4123` hardcodeado y **no** leen `seed_base` del JSON, mientras los `hyperparameters.json` de los experimentos recientes registran `41232`. El notebook de resultados lee el valor del JSON. Los seeds reales por job son `4123 + chain*9973 + k*31`.
- **`M` significa dos cosas.** En FPCA es el número de componentes retenidas; en `mcmc_config` es el tamaño de la grilla de localización G\* del stick-breaking (`N` ahí es el truncamiento del número de átomos).
- **Índices base-0 vs base-1.** `component_idx` del manifest es base-0; los nombres de archivo usan `fpc_idx = component_idx[k] + 1`. Las trazas se llaman `chain_fpc_<fpc_idx>_iter<chain a 2 dígitos>.mat` (p. ej. `chain_fpc_2_iter03.mat`) — no cambiar, el flujo de resultados los busca por nombre.
- **Cuadratura y Cholesky tienen una sola definición.** `utils.quadrature.pesos_trapezoidales` y `utils.linalg.safe_chol`; `pipelines.sim_comun` las re-exporta (`factor_cholesky`) sin redefinirlas. Duplicarlas degrada en silencio la ortonormalidad FPCA o cambia las trayectorias ante la misma semilla.
- **`np.loadtxt` colapsa a 1D** con una sola columna; por eso `artifacts.py` fuerza 2D en `_MATRICES_2D`. Deja de ser hipotético con una sola componente FPCA retenida.
- **No usar `crps_gaussiano` / `lps_gaussiano` / `pit_gaussiano` con la predictiva del modelo propuesto**: son la aproximación de dos momentos y descartan la multimodalidad/asimetría que el estudio existe para medir. Usar las versiones muestrales.
- **El error de representación no es opcional** en las bandas funcionales: `PropagadorFuncional(modo_residuo=...)` con `"empirico"` es lo recomendado; `"ninguno"` solo si se evalúa la curva *proyectada*, y debe declararse al reportar.
- **`base_en_grilla` obtiene cada base por diferencia** (`reconstruct(e_k) - reconstruct(0)`) porque con `center=True` la reconstrucción es afín y contaminaría la base con la media funcional.
- **`pipelines/real_pipeline.py` es un stub `TODO`.** El flujo de datos reales vive solo en `notebooks/reales/`.
- **`versioning/changelog.md` está obsoleto**: describe un Gibbs de 8 pasos con prior MNIW y un directorio `configs/` que no existen en el diseño actual (stick-breaking probit, muestreador en MATLAB). `versioning/experiment_registry.md` está vacío.
- Los comentarios `% [FIX]` / `% [FIX N]` en los `.m` documentan correcciones deliberadas contra la implementación original (grid G\* sobre `Xnoint` sin el intercepto, `randi` que ignoraba parte del rango, factores `exp(1.2*·)` eliminados). No revertirlos sin entender el motivo.

## Convenciones

- **Ejes del estudio**, nombrados sin abreviar tras el `[FIX 12]`: `ESCENARIO_ID` (Algoritmo *k* del anexo), `REPLICA_ID` (réplica Monte Carlo), `chain` (cadena MCMC), `k` (componente FPCA). Los notebooks/`.m` de `05_sim_E1`, `06_sim_E2` y `07_sim_E3` ya usan esta convención; `03_Modelo`, `04_sim_E1` y `notebooks/reales/*` siguen con el `tt` antiguo.
- `EXPERIMENT_ID = f"{BASENAME}_{ESCENARIO_ID}"` (p. ej. `modelo_experimento_1_3`) nombra por igual `data/`, `artefact/` y `reports/`. Debe coincidir **exactamente** entre el notebook `_01`, el `.m` y el `_03`.
- Numeración de notebooks: `NN_01_simulaciones` (preprocesamiento) y `NN_03_resultados` (evaluación), con `NN` compartido con la carpeta.
- Retención temporal: `T0` marca el corte train/test y se registra en el manifest. El estandarizador se ajusta **solo** con el bloque de entrenamiento y guarda `n_ajuste` / `etiqueta_ajuste` para que eso sea auditable. `scores_scale` debe ser `"standardized_zscore_ddof0"`.
- FPCA con patrón `fit`/`transform`: `fit` recibe solo el bloque de entrenamiento, de modo que la ausencia de fuga es una propiedad de la clase y no una disciplina del notebook.
- `.gitignore` excluye `*.npy` y los compilados; los `.mat` y las figuras `.png` **sí** se versionan.

## Terminología del dominio

**FAR(1)** autorregresivo funcional · **FGARCH** GARCH funcional · **HS** norma de Hilbert-Schmidt (`< 1` ⇒ operador contractivo) · **FPC/FPCA** componente/análisis principal funcional en métrica L² (problema generalizado `C u = λ W u`, con `W` la Gram de la base) · **MISE** error cuadrático integrado · **CRPS**, **energy score**, **LPS**, **PIT**, cobertura: reglas de puntuación de `fit/metrics_distribucional` · **T0** corte de la partición temporal · **G\*** grilla de localización del stick-breaking probit.

Los escenarios de simulación 1–4 son los Algoritmos 1–4 del anexo: FAR(1) lineal gaussiano homogéneo; FGARCH(1,1); FAR con cambio de régimen (`mecanismo` = `probit` / `markov` / `quiebre`, siendo `probit` el escenario informativo y `quiebre` no estacionario por construcción); y FAR lineal homogéneo con innovación **SMSN** (mezcla de escala skew-normal: `sn`, `st`, `sl`, `scn`), que aísla el efecto de la asimetría dejando la dinámica idéntica al Escenario 1.

---

# El estudio de simulación: qué pide `docs/` y qué existe hoy

## La fuente de verdad del diseño

`docs/` contiene el avance de la tesis y **es la especificación de lo que hay que simular**. Cuando el código y `docs/` discrepan, gana `docs/`, salvo que la discrepancia sea deliberada y quede anotada aquí.

- **`docs/01 Anexo.tex`** — los **seis** algoritmos generadores, con proceso, rol de cada parámetro y pasos de generación. Define el esquema de observación común (`§ane_01_00_00`): grilla regular de `L` puntos con `tau_1=0`, `tau_L=1`, ruido `x_tl = X_t(tau_l) + eps_tl`, `eps ~ N(0, sigma_eps^2)`, burn-in descartado, `R` réplicas con semilla registrada.
- **`docs/03 Modelo.tex`** — motivación, estado del arte, PSBP/PSBPM (Chung & Dunson), el modelo **PSBPM-FD**, la reconstrucción funcional (esperanza, varianza por ley de varianza total, intervalos por cuantiles empíricos) y el **§03_06 Diseño del Estudio de Simulación**.

### Lo que fija `docs/03 Modelo.tex §03_06`

Tres ejes de evaluación, y el estudio existe para responderlos:

1. **Desempeño puntual y sensibilidad a `M`** — métricas sobre el bloque de prueba **para distintos valores de `M`**, contra los modelos de referencia sobre la misma representación. `M` interviene dos veces: como número de mezclas y como dimensión `p = qM` del predictor. Se acompaña de la **evolución del error sobre una ventana móvil** que recorre entrenamiento y prueba.
2. **Calibración condicional** — cobertura de los intervalos **estratificando los orígenes según el estado verdadero del generador** (régimen, nivel de volatilidad). La cobertura marginal no distingue un modelo calibrado de uno ancho en calma y angosto en estrés.
3. **Interpretabilidad** — **probabilidades posteriores de inclusión** (`gamma_hj`) contra la estructura de dependencia que usó el generador, conocida en cada escenario.

Configuración común a todos los escenarios (Cuadro `tab:escenarios`): `L = 48`, `sigma_eps = 0.1` (ruido de medición), `T = 300` con `200` de calentamiento, `R = 50` réplicas, `T0 = 240` (80 %), `mu(tau) = sin(2*pi*tau)`, innovación gaussiana con `sigma = 1` y `ell = 0.2`. En los Algoritmos 5 y 6, base de Fourier con `J = 10`.

### Los dos bloques de escenarios

El anexo los ordena por **qué someten a prueba**, y la lectura de los resultados difiere:

- **Bloque 1 (Algoritmos 1–4)** — intervienen la **ley condicional** dejando intactos los supuestos de la representación. Operan **sobre la grilla**, porque sus operadores actúan sobre la curva completa. Aquí la comparación relevante es PSBPM-FD *vs.* referencias.
- **Bloque 2 (Algoritmos 5–6)** — dejan la ley condicional dentro de lo representable y comprometen la **reducción de dimensión**. Se especifican **sobre los coeficientes de un sistema ortonormal fijo** y la curva sale de un único producto matricial `X = 1 mu^T + A Phi^T`. Aquí la degradación alcanza por igual a todo método sobre la misma representación: el resultado **no discrimina entre especificaciones dinámicas**, acota el alcance de la reducción.

## Inventario de generadores: implementado vs. especificado

| Alg. | Rasgo | Módulo | Estado |
|---|---|---|---|
| 1 | FAR(1) lineal gaussiano (control) | `pipelines/sim_escenario_1.py` | Implementado; revisar parámetros |
| 2 | FGARCH(1,1) | `pipelines/sim_escenario_2.py` | Implementado; revisar parámetros |
| 3 | FAR con cambio de régimen | `pipelines/sim_escenario_3.py` | Implementado (más general que el anexo); revisar parámetros |
| 4 | Innovación SMSN (asimetría) | `pipelines/sim_escenario_4.py` | Implementado; revisar parámetros |
| 5 | Predictibilidad en componente subordinada | `pipelines/sim_escenario_5.py` | Implementado con los parámetros del anexo como default |
| 6 | Covarianza no estacionaria | `pipelines/sim_escenario_6.py` | Implementado con los parámetros del anexo como default |

**Los Algoritmos 5 y 6 cambiaron de definición.** La nota antigua de este archivo ("Algoritmo 5 no implementado por su estacionariedad bajo skew-normal") describía un diseño anterior y ya no aplica: en el anexo vigente el 5 es *predictibilidad en componente subordinada* y el 6 es *covarianza no estacionaria*. Ambos son del Bloque 2 y **no reutilizan la maquinaria de `sim_comun`** para la dinámica — no hay operador integral, ni Cholesky de innovación funcional, ni norma HS. Comparten en cambio el esquema de observación, las réplicas y las semillas, de modo que lo que sí se reutiliza de `sim_comun` es `grilla_regular`, `evaluar_media`, `semillas_replicas`, `aplicar_ruido_observacion`, `diagnostico_comun`, `ConfigObservacion` y `SalidaSimulacion`.

### Divergencias de parámetros entre código y anexo

Los defaults actuales de las dataclasses **no** son los del Cuadro `tab:escenarios`, y las últimas corridas tampoco. Revisar antes de generar nada definitivo:

| Cantidad | `docs/` | Código / última corrida |
|---|---|---|
| `L` (grilla) | 48 | `ConfigObservacion.L = 48`; la corrida de E3 usó `G = 75` |
| `sigma_obs` | 0.1 | `0.5` en la dataclass y en los notebooks `_01` |
| `T` / `T0` | 300 / 240 | `T = 200` default; E3 corrió `T = 250`, `T0 = 200` |
| `R` | 50 | `R = 50` default, pero los notebooks fijan `R = 1` |
| `mu(tau)` | `sin(2*pi*tau)` | `media_nula` default; los notebooks pasan `media_senoidal` |
| Alg. 1 `||Psi||_HS` | 0.6 | `hs_norm = 0.7` |
| Alg. 2 alcance de `beta` | 0.3 | `alcance_beta = 0.15` (`alcance_gamma = 0.30` sí coincide) |
| Alg. 3 `||Psi^(2)||_HS`, dirección `e` | 0.8, `e = psi_1` | `hs_norms = (0.30, 0.85)`, `direccion_fn` constante |
| Alg. 4 familia, `delta`, `||Psi||_HS` | `U = 1` (⇒ `sn` puro), `delta = 0.8`, dinámica igual al Alg. 1 | `familia = "st"`, `nu = 5`, `delta_skew = 0.85`, `hs_norm = 0.5` |

`ConfigEscenario3` además agrega `desplazamientos` (corrimiento de nivel por régimen) que **el anexo no contempla**: allí los regímenes difieren solo por el operador. Con `desplazamientos = (0, 0)`, `nitidez = 1`, `umbrales = (0,)` y `direccion_fn = psi_1` el código reproduce el Algoritmo 3 tal como está escrito; conviene decidir explícitamente si el estudio usa la versión del anexo o la extendida, y declararlo.

## Inventario de infraestructura de evaluación

Existe y sirve tal cual:

- **Ciclo Python → MATLAB → Python** completo y probado (ver arriba), con `artifacts.py` como capa de E/S y `verificar_contrato()`.
- **Predictiva funcional** (`models/pspb_fd_v3`): `PSBPPredictor` → `muestrear_scores` → `PropagadorFuncional` → muestras `(S, n, G)`.
- **Métricas**: `fit/metrics_puntual` (RMSE, MSE/R² por coeficiente, MISE, razón de dispersión) y `fit/metrics_distribucional` (CRPS muestral, energy score, cobertura, PIT muestral, LPS). Cubren el eje 1 y la parte marginal del eje 2.
- **Diagnóstico MCMC**: ESS/Geweke/R-hat, hoy **dentro** de `graphics/viz_traces.py` (`plot_convergence_*`), no como cálculo separable.
- **Trazas de selección de variables**: `psbp_train.m` **sí guarda** `gammajhout (nsim, N, p)`, `pijout`, `wjout`, `osumout`, y `cargar_trazas_mat` los lee. La materia prima del eje 3 ya está en disco.

Falta para poder cerrar el capítulo:

1. **Orquestación Monte Carlo (`R = 50`)**. Hoy todo el flujo — notebooks, `EXPERIMENT_ID`, `hyperparameters.json`, nombres `.mat` — está construido para **una** réplica; `REPLICA_ID` existe como campo pero siempre vale 1. Es el cambio estructural más grande: 6 escenarios × 50 réplicas × cadenas × componentes no cabe en el patrón "un notebook por experimento".
2. **Barrido en `M`**. No hay ningún mecanismo para ajustar y evaluar el mismo escenario con varios `M`; `M` se fija una vez en el notebook `_01` y se propaga al manifest.
3. **Modelos de referencia**. `fit/baselines.py` tiene solo `prediccion_media_incondicional` y `prediccion_persistencia`. No hay FAR(1), ni VAR sobre scores (Aue et al.), ni ARIMA por score (Hyndman) — que es lo que el capítulo entiende por "los de referencia sobre la misma representación".
4. **Calibración condicional estratificada**. `cobertura()` es marginal. Falta cruzar la cobertura con el estado verdadero del generador, que **sí** está disponible: los generadores lo dejan en `SalidaSimulacion.internos` (`regimenes` en el Alg. 3, varianzas condicionales en el Alg. 2).
5. **Error sobre ventana móvil** a lo largo de entrenamiento y prueba.
6. **Probabilidades posteriores de inclusión como cantidad reportable**. Existe la traza y existen gráficos en `graphics/`, pero no una función que devuelva la matriz PIP por componente lista para contrastarla con la estructura del generador.
7. **Agregación entre réplicas** (media, error estándar Monte Carlo por escenario/métrica/`M`) y las tablas del capítulo.
8. ~~**Escenarios 5 y 6** completos, incluidos sus diagnósticos~~ — hecho (ver Etapa B). Queda pendiente el notebook `_01` que los ejecute y escriba artefactos.

## Zonas grises que `docs/` todavía no resuelve

No inventarlas: preguntar antes de codificar contra ellas.

- **`03 Modelo.tex` referencia `\ref{03_06_04_objetivos_evaluacion}` (línea 374) y ese label no existe.** Las subsecciones de §03_06 sobre modelos de referencia, métricas y objetivo de evaluación (curva proyectada vs. curva suavizada) están pendientes de escribir. De ahí sale el punto 3 de la lista anterior.
- **`§03_07 Resultados` es un bloque `% [PENDIENTE]`** con la organización propuesta en seis puntos. Esa lista es la mejor guía disponible de qué debe producir el código.
- Los criterios de evaluación se citan como `§02_02_03` del Capítulo 2, que **no está en `docs/`**.
- `q` (orden markoviano) no aparece en el Cuadro de escenarios; las corridas actuales usan `n_lags = 2`.

## Plan de trabajo

Orden pensado para que cada etapa sea utilizable antes de empezar la siguiente.

**Etapa A — alinear lo que ya existe con `docs/`.** Llevar los defaults de los cuatro escenarios y de `ConfigObservacion` al Cuadro `tab:escenarios`, decidir el caso del `desplazamientos` del Alg. 3 y del `familia`/`delta` del Alg. 4, y dejar los valores del anexo como default de la dataclass en vez de como argumento del notebook — el notebook debería poder no pasar nada y obtener el escenario del anexo.

**Etapa B — Escenarios 5 y 6. HECHA.** `sim_escenario_5.py` y `sim_escenario_6.py` implementan la generación sobre coeficientes + base de Fourier `J = 10` (`base_fourier`, ortonormal en L² continuo; la cuadratura trapezoidal sobre grilla regular la reproduce con error de máquina si `2*k_max < L-1`, condición que `validar()` verifica). Los defaults de ambas dataclasses son los del Cuadro `tab:escenarios` — `ConfigEscenario5()` / `ConfigEscenario6()` sin argumentos son el escenario del anexo, incluida `media_fn=media_senoidal` y `sigma_obs=0.1`. Ambos exportan por `pipelines/__init__.py`. Falta el notebook `_01` que los consuma. Nota de diagnóstico: `resumen_escenario_6` compara varianzas en la ventana posterior a `t*=270`, que tiene 30 períodos; con `R=1` el reordenamiento no se separa del ruido Monte Carlo y `reordenamiento_detectado` puede dar False sin que el generador falle (con `R≥20` se detecta).

**Etapa C — evaluación que hoy falta**, en `fit/`: baselines de referencia, cobertura condicional estratificada, ventana móvil, extracción de PIP, y separar los diagnósticos MCMC de sus gráficos (el `diagnostics_mcmc.py` que el propio `fit/__init__.py` anota como pendiente).

**Etapa D — barrido en `M` y réplicas.** Es donde el diseño actual de un-notebook-por-experimento deja de escalar. Requiere decidir la convención de nombres e `EXPERIMENT_ID` para `(escenario, réplica, M)` antes de escribir código, porque esa convención está replicada a mano en cada `config_paths.m`.

**Etapa E — agregación y tablas** siguiendo los seis puntos del `% [PENDIENTE]` de `§03_07`.
