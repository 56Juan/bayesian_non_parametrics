# CLAUDE.md — Guía de trabajo para simulaciones de tesis

Este archivo define cómo Claude Code debe asistir en este repositorio. Es la
única fuente de verdad sobre el *modo de trabajo*; el código y `docs/` son la
fuente de verdad sobre el *contenido*.

---

## 1. Propósito

Claude **no** ayuda a *completar* la tesis ni a redactar capítulos. Su rol es:

- Diseñar, ejecutar y depurar **simulaciones**.
- Implementar y ajustar **modelos** y **métricas**.
- Interpretar resultados **cuando yo lo indique**.
- Cambiar y mejorar redacción **cuando se le indique**.

Lo que queda fuera salvo pedido explícito: proponer secciones de la tesis,
escribir prosa para `docs/`, decidir qué resultados "cuentan", ampliar el
alcance del estudio.

---

## 2. Contexto del proyecto

Tesis: extensión del **PSBP** (Probit Stick-Breaking Process, Chung & Dunson
2009) a **series de tiempo funcionales**, bajo el nombre **PSBPM-FD**. El
paquete `model_psbp_fd` genera datos, consume trazas MCMC y evalúa
predicciones. **El muestreo MCMC ocurre en MATLAB**, no en Python.

Código, docstrings y nombres están en español; mantener ese idioma. Los
docstrings de módulo son extensos a propósito: documentan *por qué* se tomó cada
decisión numérica y suelen traer la respuesta antes que el código.

- En `docs/` está el material de la tesis (`01 Anexo.tex`, `01 Introduccion.tex`,
  `02 Marco Teorico.tex`, `03 Modelo.tex`). Es **referencia, de lo que se va a
  hacer**: se puede modificar, y de ahí salen indicaciones, no obligaciones.
- **Todo el estudio de simulación se especifica sobre los scores** de la
  representación funcional. El anexo vigente tiene tres escenarios, cada uno
  asociado a un rasgo de la ley condicional del score que un modelo de media
  lineal no representa: **TAR** (cambio de régimen por umbral), **GARCH**
  (heterocedasticidad condicional) y **multimodalidad**. **El conjunto no está
  cerrado**: pueden agregarse escenarios (cada uno = un algoritmo nuevo en el
  anexo + una copia de la 200 con otro generador). Cuáles se muestran en el
  documento es una decisión pendiente del autor (§8).
- **La predictiva del PSBPM-FD es una mezcla** y se resume de más de una forma.
  `docs/03 Modelo.tex §03_05` define cuatro predictores puntuales (esperanza,
  mediana puntual, mediana de Fréchet/medoide L¹, átomo más probable; la mediana
  por profundidad MBD se eliminó del código el 2026-10-11), tres intervalos
  (equi-colas, banda simultánea, región de máxima densidad puntual) y `§03_06` los
  **comportamientos predictivos** (k-medias sobre los draws de scores con reglas
  valle/crecimiento/masa). Regla de consistencia (Gneiting 2011): MAE y E_max se
  reportan con una mediana, RMSE con la esperanza; las seis métricas del Bloque B
  con cada construcción de intervalo. En 200–202 y las reales el pipeline calcula
  la esperanza y la equi-colas; en las 30x (`predictores=`, `intervalos=`) los
  cuatro predictores y los tres intervalos (§4, §7).
- El foco experimental son **escenarios donde el modelo puede ganar**: se trata de
  caracterizar *cuándo y por qué* el PSBPM-FD supera a las alternativas, no de
  probar que gana siempre. Un resultado negativo bien acotado es un resultado,
  no un fracaso (p. ej. GARCH deja la media lineal y se espera ventaja sólo en el
  Bloque B).

---

## 3. Invariantes

Estos tres elementos **no cambian entre escenarios**. Si una instrucción mía los
contradice, avisar **antes** de ejecutar.

### 3.1 Métricas — el conjunto es fijo y son diez

**La fuente de verdad es `docs/02 Marco Teorico.tex §02_02_03`**, y el conjunto
es exactamente el que ahí se define: ni una más.

Se calculan **una sola vez**, sobre la ventana móvil, con
`ventana_movil_funcional(..., bloque_A=True)`. No hay tabla agregada aparte:
todo lo demás (resumen mínimo/máximo/promedio, comparación entre `M`, quién gana
cada ventana) se **deriva** de esa misma tabla.

| # | Bloque | Métrica (`.tex`) | Columna |
|---|---|---|---|
| 1 | **A** — error, `§02_02_03_02` | MAE funcional | `mae_f` |
| 2 | | RMSE funcional | `rmse_f` |
| 3 | | Ē_max — promedio de los errores máximos por curva | `linf_medio` |
| 4 | | E_max^glob — máximo de los errores máximos en la ventana | `linf_max` |
| 5 | **B** — calibración, `§02_02_03_04` | MPIW | `mpiw` |
| 6 | | PICP — fracción del dominio cubierta | `picp` |
| 7 | | PICPB — fracción de curvas cubiertas por completo | `picp_simultaneo` |
| 8 | | Winkler integrado sobre el dominio | `winkler` |
| 9 | | W̄_max — promedio del peor Winkler por curva | `winkler_max_medio` |
| 10 | | W_max^glob — peor Winkler de la ventana | `winkler_max_glob` |

Se declaran en la celda `[CONFIG]` de `_04` y `_05` como `METRICAS_A` /
`METRICAS_B`, con el mismo contenido que `fit/evaluacion_barrido.py` (un `assert`
lo verifica), y `_01` las persiste en `eval_config.json`.

Relaciones verificadas con `assert`, que **no** deben romperse:
`mae_f <= l2_medio <= linf_medio <= linf_max`, `picp_simultaneo <= picp`, y
`winkler <= winkler_max_medio <= winkler_max_glob`.

**`rmse_f` NO pertenece a la cadena L¹ ≤ L² ≤ L^∞.** Es la raíz del MSE
*agregado*, una media **cuadrática** sobre los orígenes: por Jensen
`l2_medio <= rmse_f`, pero `rmse_f <= linf_medio` **es falso** cuando el error es
desigual entre los orígenes de la ventana. La cadena se verifica sobre `l2_medio`.

`linf_max` y `winkler_max_glob` dependen de una sola evaluación y son las cifras
frágiles del conjunto: se reportan porque el marco teórico las define. **`mise` y
`mise_rel` siguen en la tabla** pero son diagnóstico interno (salto en `T0`,
piso de truncamiento), no métricas reportables.

### 3.2 Ventanas temporales — la segmentación es fija

- Anchos `w ∈ {20, 30, 40}`, `paso = 1`, solapadas. `W_REF = VENTANAS_W[len // 2]`
  = **30**: las figuras superponen los tres anchos y las tablas usan `W_REF`.
- **Las ventanas que cruzan `T0` se excluyen** de todo agregado (`cruza_T0`).
- Partición: `PROP_TRAIN = 0.70`, `T0` marca el corte y va en el manifest.
- Se **leen** de `eval_config.json["ventana_movil"]`; **no se redeclaran** en
  `_04` / `_05`.

### 3.3 Evaluación sobre todos los `M`

`M_FPCA_LIST` se declara en **los cuatro notebooks y en el `.m`**, y los cuatro
recorren el barrido completo en una sola pasada.

- `_01` produce un juego de artefactos por cada `M` (bucle `procesar_punto(M)`);
  los objetos comunes —`fr`, `fpca`, `THETA`, la partición— se pasan por
  clausura y **no se recalculan**: ésa es la garantía de que los puntos
  comparten realización y base.
- `_03`, `_04` y `_05` cargan cada punto en `EST[M]` e iteran sobre él.
- `psbp_fd_iteracion.m` arma **una sola lista plana** de jobs
  `(M × cadena × componente)` con un único `parfor`, no uno por `M`.
- `SALTAR_M_SIN_TRAZAS` / `SALTAR_M_SIN_ARTEFACTOS` permiten mirar el barrido
  mientras MATLAB va terminando.
- Los puntos **deben compartir diseño**: `T`, `T0`, `n_lags`, `nivel`,
  `modo_residuo`, `objetivo_evaluacion`, ventanas y `mcmc_config` idénticos, y
  las mismas curvas. Si no, la comparación entre `M` mezclaría el efecto de `M`
  con el de otra cosa, y el notebook **falla en vez de callar**.

`M` significa dos cosas: en FPCA es el número de componentes retenidas; en
`mcmc_config` es el tamaño de la grilla de localización G\* del stick-breaking
(ahí `N` es el truncamiento de átomos).

---

## 4. Estructura canónica: la corrida 200

**`notebooks/simulaciones/200_sim_TAR/` es la plantilla.** Toda simulación nueva
sigue **esa misma estructura, sin agregados**: se cambian parámetros o
escenario, no la arquitectura del pipeline. Las corridas 201 y 202 son copias
suyas con otro generador (§7).

**Cuatro notebooks y un paso MATLAB**, acoplados por archivos en disco:

| Paso | Archivo | Qué hace |
|---|---|---|
| 1 | `200_01_simulaciones.ipynb` | **Genera los datos** y los prepara: simulación, partición temporal, representación en la base del generador, FPCA de oráculo, datasets AR, contrato. Un juego de artefactos por cada `M`. |
| 2 | `psbp_fd_iteracion.m` | **Entrena.** Lee el contrato, arma los jobs y llama a `psbp_train.m`. Sólo con el bloque train. No tiene notebook. |
| 3 | `200_03_convergencia.ipynb` | **Convergencia de las cadenas.** No toca el bloque de prueba. |
| 4 | `200_04_evaluacion.ipynb` | **Evaluación propia del modelo**: las diez métricas sobre la ventana móvil, PIP, monitoreo por componente. |
| 5 | `200_05_comparacion.ipynb` | **Comparación** del PSBPM-FD contra FAR y RF. |

> `_01` **no** es "el entrenamiento": genera los datos. El entrenamiento es el
> paso MATLAB. Si las cadenas no convergen, los números de `_04` no significan
> nada: por eso `_03` es un notebook aparte.

### La representación (scores como insumo)

El generador entrega los **scores**; los modelos de scores los reciben **crudos**
(`scores_scale = "raw_fpca_scores"`, estandarizador en identidad, sin ruido de
medición: `sigma_obs = 0`). La base de Fourier ortonormal del generador **es** la
representación: `theta_t = mu_theta + xi_t`, exacta (porque `mu = sin(2πτ)` cae en
la base), con `K = J = 10`. **No hay GCV, ni B-spline, ni FPCA estimada**:
`FunctionalRepresentation.desde_base` y `FPCA_L2.desde_scores_conocidos` arman los
objetos del contrato con la base y los scores conocidos, en el orden de la base.
El prior se mantiene equivariante a la escala con hiperparámetros relativos a la
dispersión de cada serie (§7), no reescalando los datos.

### Secciones de cada notebook

`_01`: §1 imports y `[CONFIG]` (§1.1 experimento y barrido en `M`) · §2
simulación (§2.1 `[CONFIG]` generador, §2.3 figuras, §2.5 partición) · §3
representación (§3.1 base conocida y coeficientes, §3.2 FPCA de oráculo) · §4
bucle sobre `M_FPCA_LIST` (§4.1 componentes, §4.2 estandarizador identidad, §4.3
diagnóstico de rezagos, §4.4 datasets AR, §4.5 hiperparámetros y MCMC, §4.6
evaluación y contrato) · §5 resumen del barrido.

`_03`: §1 artefactos (`[CONFIG]`, `cargar_artefactos`) · §2 trazas · §3 tabla de
diagnósticos · §4 ocupación de la mezcla (§4.1 gating) · §5 figuras por parámetro ·
§6 PIP (§6.1 contraste con el generador) · §7 verificación muestreador ↔
predictor (§7.1 paso de Γ) · §8 comparación entre puntos del barrido · §9
comparación entre variantes.

`_04`: §1 artefactos · §2 trazas · §3 predicción a `h=1` (§3.1 persistencia de la
banda) · §4 Bloque A sobre la ventana móvil (§4.1 qué `M` gana cada ventana) · §5
Bloque B · §6 muestra de predicciones · §7 PIP · §8 resumen mín/máx/promedio · §9
monitoreo univariado por componente · §10 banda de credibilidad de cada `xi_k`,
métricas de intervalo por componente y dispersión `xi_hat` contra `xi`.

`_05`: §1 artefactos · §2 validaciones · §3 competidores (§3.1 FAR, §3.2 RF, §3.3
PSBPM-FD) · §4 Bloque A (los tres modelos) · §5 Bloque B (sólo FAR y PSBPM-FD) ·
§6–§7 quién gana cada ventana · §8 resumen y métricas contra `M` · §9 intervalos
curva a curva.

En las 30x (`300_sim_TAR_estimadores/` es la de referencia) el `_04` y el `_05`
agregan, sin quitar nada: `_04` §3 los cuatro predictores, las tres construcciones de
intervalo (equi-colas, simultánea y HDR puntual, sobre las mismas extracciones) y
`draws_scores_psbp.npy`, §4 y §8 el Bloque A por predictor (§4 también en la ventana
móvil, §4.2 qué predictor gana cada ventana, §8.1 la tabla por predictor y `M`), §5.1
el Bloque B por construcción de intervalo (con la esperanza; qué construcción gana
cada ventana) y §8.2 su tabla por intervalo y `M`, **§11 diagnóstico de multimodalidad** (|esperanza −
mediana| por score; distancia de la mediana puntual a la extracción más cercana) y
**§12 comportamientos predictivos** (§12.1 `K` por origen, §12.2 curvas típicas y
bandas condicionales, §12.3 catálogo global, §12.4 validación contra el régimen
verdadero con la predictiva oráculo del generador; en la 301, sin régimen, distribución
de `K` y ancho de la banda contra `h_t` verdadera, para las tres construcciones; en la
302, contra el mecanismo `Z_t` y, además, la HDR y la simultánea contra las mismas
construcciones sobre la ley verdadera); `_05`
§3.3 una fila por predictor, §5 una banda por construcción del PSBPM-FD junto al FAR
(con `IC_PSBP_TAMBIEN_GAUSSIANA = False`, cuatro filas: FAR, equi-colas, simultánea,
HDR) y el contraste "misma predictiva, distinta construcción", §6 cada predictor contra
FAR y RF, §7 cada construcción contra el FAR cara a cara (`111_*`), §9 el FAR contra
cada construcción curva a curva, y **§10 comportamientos contra lo observado** (desde el
2026-10-11 `validar_observado(..., curvas_ref=)` agrega el MAE y RMSE en el origen de la curva
típica del comportamiento realizado y del más probable, y de la esperanza, el FAR y el RF en los
mismos orígenes: columnas `mae_*`/`rmse_*` en `82`, `83` y `105`; los `_05` de las 30x y la 32
tienen la celda, falta reejecutarla). Los artefactos de una 30x se **copian** de su 20x con
`00_copiar_200.py` (`--origen/--destino/--M`): no se corre `_01` ni MATLAB.

**Ventana de detalle** (`notebooks/simulaciones/PROMPT_30x_ventana_detalle.md`, 2026-10-11).
Las diez métricas de la ventana móvil, las tres bandas y los cuatro predictores en todos
los orígenes, y §12.1 (`K` por origen) siguen sobre toda la serie. El **análisis
predictivo por origen** va sobre `N_DETALLE = 10` orígenes consecutivos de prueba desde
`T0_DETALLE` (`[CONFIG]` de `_04` y `_05`; `None` = criterio automático del escenario,
calculado sobre los `internos`, nunca sobre lo predicho): `_04` §3.2 la elige y la escribe
en `112_ventana_detalle.json`; §6 (extractos `54`), §11, §12.2 (grilla `63` y una figura
por origen `63_detalle_t<t>.png`), §12.3, §12.4 y §12.5 (`112_ventana_detalle_por_origen.csv`)
van sobre ella; el `_05` la **lee** del JSON (§9, `79_ic_contraste_detalle{,_hdr,_simultanea}.png`;
§10). Con 10 orígenes aciertos y coberturas se reportan como cuentas (k de 10). Las
funciones del paquete que separan train/test reciben `ORIG_DET = {**ORIG, "es_train":
~MASK_DET}`: su bloque "test" es la ventana. No hubo cambios en el paquete. Criterios:
300, más cambios de régimen verdadero; 301, contiene el máximo de `h_t` agregada
(`sum_j lambda_j h_tj`, el máximo en la tercera posición); 302, más orígenes con `K ≥ 2` en
la ley verdadera.

### Dónde vive el código de cada notebook

Los notebooks **sólo orquestan**; el cálculo y el dibujo están en el paquete:

| Notebook | Cálculo | Figuras |
|---|---|---|
| `_01` | `fit/preparacion_barrido.py` (`preparar_punto(M, CTX)`: datasets AR, hiperparámetros, contrato) | `graphics/viz_fpca.py` (`plot_diagnostico_rezagos`, `plot_series_componentes`) |
| `_03` | `fit/convergencia_barrido.py` | `graphics/viz_convergencia.py` (+ `viz_traces`, `viz_global_components`) |
| `_04` | `fit/evaluacion_barrido.py` | `graphics/viz_evaluacion_barrido.py` (+ `viz_evaluacion`) |
| `_05` | `fit/comparacion_barrido.py` | `graphics/viz_comparacion.py` |
| 30x (`_04` §11–§12, `_05` §10) | `fit/resumenes_predictiva.py`, `fit/comportamientos.py` | `graphics/viz_comportamientos.py` |
| 30x (`_04` §3, §5.1, §8.2; `_05` §5, §7, §9) | `fit/intervalos_predictiva.py` (las tres construcciones y las métricas sobre una unión de tramos; `rolling.ventana_movil_funcional(..., region=)`) | `graphics/viz_evaluacion_barrido.py` (`plot_ventana_grupos_por_M`, `plot_ganador_grupo_vs_M`), `viz_comparacion.ESTILOS_BANDAS` |
| 301 (`_04` §12.4) | `fit/heterocedasticidad.py` (banda contra la varianza condicional del GARCH) | `graphics/viz_heterocedasticidad.py` |
| 302 (`_04` §12.4) | `pipelines/sim_escenario_C1.predictiva_oraculo_C1`; `fit/comportamientos.intervalos_contra_oraculo` (vía `regimen_barrido(..., intervalos=True)`: columnas `frac_multitramo_*`, `masa_*`, `ancho_*` en `65_*` y `104_*`; con la ventana de detalle se muestran como cuentas) | `viz_comportamientos.plot_validacion_regimen` (cuarto panel; sin uso desde la ventana de detalle) |

`cargar_artefactos(..., con_test=True)` carga cada punto y cruza el contrato
(`verificar_contrato`); `cargar_representaciones` fija el diseño común y exige que
los puntos coincidan. Los tres comparten el estado `EST[M]`.

### Los competidores del `_05`

`FAR(p)` · `RF` · y el PSBPM-FD. **No hay líneas base (media, persistencia) ni
GBT**. Hay una sola referencia lineal.

- **FAR(p)** sobre los coeficientes de la base, en la métrica L²: se estima
  blanqueando `THETA` con la Cholesky de la Gram (`W = LL^T`), verificado con
  `assert` (con base ortonormal `W = I`). Se ajusta con `pesos="conteo"` (pesar de
  nuevo con la trapezoidal aplicaría L² dos veces). **`p = N_LAGS` fijo, no se
  selecciona**: seleccionarlo daría una ventaja de especificación que el PSBPM-FD
  no tiene. Tope de `kn` = `K`.
- **RF**: un bosque **por componente**, con las **mismas covariables** que el
  PSBPM-FD (rezago propio en 200/201; todos los rezagos en 202). Barre el orden
  `1..N_LAGS` y se queda con el que gana más **ventanas de TRAIN** en `mae_f`;
  nunca test.
- **PSBPM-FD**: su predicción puntual y su banda se **leen** de
  `banda_funcional_psbp.npz`, que persiste `_04`; no se recalcula desde las trazas.
  En 200–202 y las reales hasta la 31 el punto es la **esperanza analítica**
  (`momentos()`). En las 30x `predecir_barrido(..., predictores=PREDICTORES)`
  calcula además mediana puntual, medoide L¹ y átomo modal
  (`fit/resumenes_predictiva.py`), el `.npz` los lleva como `X_pred_<predictor>`
  y `cargar_psbp(..., predictores=...)` registra una fila `PSBPM-FD (<predictor>)`
  por cada uno; `e["X_PSBP"]` (la esperanza) es la que acompaña a las bandas en el
  Bloque B. Con `intervalos=INTERVALOS` se agregan la banda simultánea y la HDR
  (`cargar_psbp(..., intervalos=...)`, `construir_bandas(..., intervalos_psbp=...)`,
  filas `G_PSS` y `G_PSH`). Sin `predictores` ni `intervalos` todo es como en la 200
  (sin columnas `predictor` ni `intervalo`).

El Bloque B se restringe a FAR y PSBPM-FD —los únicos con mecanismo de intervalo propio— con dos filas por `M`: la banda gaussiana del FAR y la banda nativa del PSBPM-FD (cuantiles
de su predictiva). `IC_PSBP_TAMBIEN_GAUSSIANA = False`: no se corre el control con banda gaussiana
sobre la media del PSBPM-FD.

---

## 5. Cómo trabajar conmigo

- Si una instrucción mía rompe los invariantes (§3) o la estructura de la corrida
  200 (§4), **avisar antes de ejecutar**.
- **No agregar pasos, métricas ni etapas** al pipeline sin que yo lo pida.
- **No ejecutar notebooks ni MATLAB** sin que lo pida (las corridas son largas y
  los artefactos se pisan con `limpiar=True`). Sí: chequeo estático de las celdas
  y scripts sueltos de verificación sobre datos ya generados.
- Ser conciso: priorizar código, comandos y diagnósticos sobre explicaciones largas.
- No leer ni escribir artefactos a mano desde un notebook: usar
  `pipelines/artifacts.py` y `utils/rutas.py` (§6.3).
- `200_sim_TAR/` es la plantilla y se edita con cuidado. Las copias 201/202 **no
  se actualizan solas**: un cambio en la 200 se propaga a mano. Lo histórico (§7)
  no se toca.
- **Las corridas 300–302 no regeneran ni reentrenan**: toman los artefactos de
  200–202 por copia en disco bajo IDs nuevos (`escenario_30x`) y sólo ejecutan
  `_04`/`_05`. Nunca escribir en los directorios de 200–202 desde una 30x.
- Los cambios al paquete que piden las 30x son **aditivos**: valores por defecto
  que dejan 200–202 y las reales exactamente como están.
- Tras cambiar el paquete, el kernel de los notebooks debe **reiniciarse** (los
  módulos ya importados no se recargan).

---

## 6. Referencia técnica

### 6.1 Instalación y pruebas

```bash
pip install -e .
```

Alternativa conda en `environment.yml` (entorno `psbp_fd`).

```bash
python -m pytest tests/test_metricas_bloques_AB.py tests/test_far_operador.py tests/test_sim_escenario_B_mezcla.py tests/test_resumenes_predictiva.py tests/test_comportamientos.py tests/test_intervalos_predictiva.py -v
```

Cubren los Bloques A y B y su enganche con la ventana móvil, el FAR contra salidas
de R, el generador de mezcla B, los cuatro predictores puntuales (con el
`PSBPPredictor` real sobre trazas sintéticas, el medoide contra su definición
directa, el `.npz` nuevo y uno antiguo, y que las llamadas de `200_04`/`200_05`
sigan ligando con las firmas), las reglas de comportamientos sobre muestras con
`K` conocido y las tres construcciones de intervalo (las métricas de tramos con
K = 1 reproducen las de siempre al bit; HDR de dos tramos en una bimodal, igual a
la equi-colas en una gaussiana; simultánea robusta a las colas de atau ≤ 1; el
`.npz` con intervalos y el Bloque B del `_05` por tramos). No hay configuración de pytest, CI ni linter.

### 6.2 El ciclo Python → MATLAB → Python

Las tres etapas se acoplan **por archivos en disco**, no por llamadas.
`hyperparameters.json` es la **única fuente de verdad del contrato Python ↔
MATLAB** (`n_iter`, `mcmc_config`, `hyperparams_list`, `partition`, `seed_base`,
`trazas_en`). **No hardcodear esos valores en el `.m`.**

`psbp_fd_iteracion.m` **lee `seed_base` del JSON** y falla si falta. La semilla
por job es `seed_base + chain*9973 + k*31`. Hay **una copia de `config_paths.m` y
`psbp_train.m` por carpeta de experimento**: un cambio de convención se propaga a
mano a todas.

Con **rezago propio** el score `xi_k` no depende de `M`, así que MATLAB entrena
**sólo** `M_ENTRENO = max(M_FPCA_LIST)` y los `M` menores reutilizan las trazas vía
`hyperparameters.json["trazas_en"]`. Con **covariables cruzadas** el diseño cambia
con `M` y **se entrena cada punto**.

### 6.3 Artefactos: dónde vive cada cosa

| Clave | Ruta | Contenido |
|---|---|---|
| `raw` | `data/<dominio>/raw/<EID>/` | `X_curves.npy`, `X_curves_true.npy`, `simulation_config.json`, `escenario_<id>.npz` |
| `functional` | `data/<dominio>/processed/functional/<EID>/` | `datasets_manifest.json`, `dataset_fpc_<idx>_{train,test}.csv`, CSV de FPCA, estandarizador, `theta.csv` |
| `predict` | `data/<dominio>/processed/predict/<EID>/` | `banda_funcional_psbp.npz` (`li`, `ls` = equi-colas, `X_pred` = esperanza y, en las 30x, `X_pred_<predictor>`, `li_simultanea`/`ls_simultanea` `(n, G)` y `hdr_L`/`hdr_U` `(n, G, hdr_kmax)` float32 con NaN donde no hay tramo); en las 30x además `draws_scores_psbp.npy` (extracciones de scores `(S, n, M)` float32, ~240 MB con `M = 10`, ignorado por git) y `comportamientos_psbp.npz` (`K`, masas, centroides, curvas típicas y bandas condicionales por origen) |
| `out_report` | `reports/<dominio>/<EID>/` | figuras y CSV del punto |
| `out_artefact` | `artefact/<dominio>/<EID>/` | `hyperparameters.json`, `eval_config.json`, trazas `.mat` |

- **`utils/rutas.py` construye los directorios** (`experiment_id`,
  `construir_paths`, `rutas_por_M`, `ruta_barrido_M`, `guardar_en_todos`,
  `replicar_figura`). Es el gemelo Python de `config_paths.m`; si una cambia, la
  otra también.
- **`pipelines/artifacts.py` decide los nombres de archivo** (dict `ARCHIVOS`) y
  centraliza escritura y lectura emparejadas. `verificar_contrato()` cruza
  manifest, hiperparámetros y artefactos FPCA.
- `construir_paths(limpiar=True)` vacía `raw`, `functional`, `predict` y
  `out_artefact`, pero **no** `out_report`: un dataset o una traza rancios se
  leerían como si fueran de la corrida actual.

### 6.4 El modelo

**`psbp_fd_v3` es la versión que se usa** y no tiene muestreador: trazas `.mat` →
predictiva por score → `PropagadorFuncional` (reconstruye la curva) → muestras de
curvas `(S, n, G)`, que son el insumo de `fit/`.

La puerta de entrada de los notebooks vivos es `ModeloTraza`: `leer_traza` da
`(traces, burn, feature_names)` y `ModeloTraza.momentos(df)` / `.muestrear(df, S)`
producen media, sd predictiva y extracciones por score. **No redefinir la
convención de nombres de traza en un notebook**: usar `ruta_traza`. (`N1out`,
`Nout` y `muout` no vienen en `leer_traza`: los agrega
`convergencia_barrido._leer_traza_completa`.)

`curva_media_desde_scores` es el mapa determinista (`modo_residuo="ninguno"`, sin
muestreo) con el que los competidores entran en la misma escala de curva que el
PSBPM-FD. `v1` / `v2` son heredadas: siguen en disco pero `models/__init__.py` ya no las importa (sólo `v3`).

**Contrato v2 → v3:** `predict(return_std=True)` devuelve la **predictiva** (ley de
varianza total), no la sd de la media condicional (`sd_centro`). Usar la de v2
como banda produce subcobertura que se confunde con fracaso del modelo. Con
`atau <= 1` esa sd **no existe** (diverge); las bandas por cuantiles sí.

### 6.5 El error se mide contra la curva, no contra la grilla cruda

**La fuente de verdad es `docs/03 Modelo.tex §03_05_00`**, con dos objetivos:

| objetivo | qué es | quién se mide contra él |
|---|---|---|
| **curva suavizada** | `fr.reconstruct(fr.transform(X_obs))` | **todos**: FAR, PSBPM-FD y RF |
| **representación FPCA** `X_t^(M)` | `mu + sum_{m<=M} xi_tm psi_m`, la expansión truncada | sólo los métodos sobre scores |

En 200–202 la representación es exacta, así que la curva suavizada **coincide con
la curva verdadera**. El segundo objetivo cambia con `M` y aísla la dinámica sobre
los scores. El **Bloque A** se reporta contra la curva suavizada; el **Bloque B**,
contra **los dos**, en una tabla con una columna que declara el objetivo (la banda
del FAR vive en `K` y la del PSBPM-FD en `M`).

`modo_residuo` **no es una perilla**: es `"ninguno"` por construcción del modelo
(la variabilidad vive dentro de la mezcla, no en un error aditivo). Sigue en
`eval_config.json` porque el `PropagadorFuncional` lo consume. `objetivo_evaluacion`
se registra desde `_01`; **no se decide en el notebook de evaluación**.

### 6.6 Convenciones

- **Ejes del estudio**, sin abreviar: `ESCENARIO_ID`, `REPLICA_ID`, `chain`,
  `k` (componente FPCA), `M` (componentes retenidas).
- **`EXPERIMENT_ID`** nombra por igual `data/`, `artefact/` y `reports/`, y debe
  coincidir **exactamente** entre los cuatro notebooks y el `.m`:
  `f"{BASENAME}_{ESCENARIO_ID}_r{REPLICA_ID:02d}_m{M_FPCA:02d}"` (p. ej.
  `escenario_200_chungEscG_1_r01_m03`). Datos reales: `real_<serie>_v<NN>_m<NN>`.
  `M` viaja en el id para que cada punto escriba sus propios datos sin pisar los
  demás; por eso `M_FPCA_LIST` se declara en `[CONFIG]` antes de construir rutas.
- **Constantes al inicio, igual en los cuatro notebooks**: `[CONFIG]` con
  `PROJECT_ROOT`, `VARIANTE`, `BASENAME`, `ESCENARIO_ID`, `REPLICA_ID` y
  `M_FPCA_LIST`, más las constantes propias de cada uno.
- **Índices base-0 vs base-1.** `component_idx` del manifest es base-0; los
  nombres de archivo usan `fpc_idx = component_idx[k] + 1`. Las trazas se llaman
  `chain_fpc_<fpc_idx>_iter<chain a 2 dígitos>.mat`; **no cambiar**.
- **Salidas del barrido**: lo que cruza los puntos va a
  `reports/<dominio>/<BASENAME>_<ESCENARIO>_r<NN>_barrido_M/`, hermano de los de
  cada `M`.

  | Notebook | Por `M` (`out_report`) | Barrido (`_barrido_M`) |
  |---|---|---|
  | `_01` | `01`–`09`, `30_baselines_test.csv` | — |
  | `_03` | `40`–`49b` | `80`–`83` |
  | `_04` | `54`–`57`, `66`–`67`, `90`–`94` | `84`–`89` |
  | `_05` | `72`–`81b` | `76`, `77`, `96`–`99` |
  | `_04` de las 30x, además | `60`–`65` (§11–§12) | `100`–`104` (`101b`, `102b`) |
  | `_05` de las 30x, además | `82`–`83` (§10) | `105`–`107` |
  | `_04` de las 30x, intervalos | `58`–`59` (§4 por predictor, §5.1 por intervalo) | `108`–`110` |
  | `_05` de las 30x, intervalos | `81b_*_PSBPM-FD-simultanea/-hdr` | `111` |
  | 30x, ventana de detalle | `_04`: `63_detalle_t<t>.png`; `_05`: `79_ic_contraste_detalle{,_hdr,_simultanea}.png` | `112_ventana_detalle.json`, `112_ventana_detalle_por_origen.csv` |

  En la 301 los prefijos de la validación contra régimen (`65` por `M`, `104` en el
  barrido) llevan el diagnóstico de banda contra `h_t`.

  Los prefijos `80`/`81` aparecen en dos notebooks pero en **directorios
  distintos**; no "arreglar" la numeración sin pedirlo.
- Retención temporal: todo lo **estimado** con datos (autovalores de la FPCA,
  `kn` del FAR, orden del RF, `sd` de los hiperparámetros) usa **sólo** el bloque
  de entrenamiento; el estandarizador (identidad) guarda `n_ajuste`.
- `.gitignore` excluye `*.npy` y los compilados; los `.mat` y las figuras `.png`
  **sí** se versionan.

### 6.7 Gotchas

- **MATLAB colapsa la dimensión singleton final al guardar.** Con `p = 1`
  (`M = 1` con rezago propio y `N_LAGS = 1`) `betajhout`, `psijhout`, `Gammajhout`
  y `gammajhout` llegan con dos ejes en vez de tres. `utils/trazas.py`
  (`normalizar_trazas_mat` / `asegurar_3d`) restaura el eje. **No leer
  `traces["betajhout"].shape[2]` a mano**: usar `predictor.n_features_`.
- **`np.loadtxt` colapsa a 1D** con una sola columna; `artifacts.py` fuerza 2D en
  `_MATRICES_2D`.
- **Cuadratura y Cholesky tienen una sola definición**:
  `utils.quadrature.pesos_trapezoidales` y `utils.linalg.safe_chol`. Duplicarlas
  degrada en silencio la ortonormalidad o cambia trayectorias ante la misma semilla.
- **`base_en_grilla` obtiene cada base por diferencia**
  (`reconstruct(e_k) - reconstruct(0)`) porque con `center=True` la reconstrucción
  es afín.
- **No usar `crps_gaussiano` / `lps_gaussiano` / `pit_gaussiano`** con la
  predictiva del modelo propuesto: descartan la multimodalidad/asimetría que el
  estudio existe para medir. Usar las versiones muestrales.
- **Los diagnósticos MCMC se calculan sobre cantidades invariantes a la
  permutación de etiquetas** (`extraer_traza_variable` promedia sobre átomos).
  `fit/diagnostics_mcmc.py` tiene la única definición.
- **`fit/rolling.py` no reentrena**: desliza la ventana de *evaluación* sobre una
  serie de predicciones a `h=1` con rezagos reales.
- Los comentarios `% [FIX]` / `% [FIX N]` en los `.m` documentan correcciones
  deliberadas (grid G\* por predictor, log-verosimilitud en el paso de Γ, etc.).
  **No revertirlos sin entender el motivo.**

### 6.8 Terminología

**FAR(p)** autorregresivo funcional · **TAR / SETAR** autorregresivo por umbral
(el umbral aquí es el rezago propio: autoexcitado) · **GARCH** varianza
condicional autorregresiva · **HS** norma de Hilbert-Schmidt · **FPC/FPCA** análisis
principal funcional en métrica L² · **MISE** error cuadrático integrado ·
**Winkler**, **PICP**, **MPIW**, cobertura simultánea: Bloque B · **T0** corte de la
partición temporal · **G\*** grilla de localización del stick-breaking probit ·
**oráculo** la media condicional verdadera, cota superior de lo alcanzable ·
**scores activos / pasivos** los `J_a = 4` primeros llevan la dinámica del escenario
y los otros seis son AR(1) de coeficiente 0.3.

---

## 7. Estado del repositorio

**Se trabaja sobre las corridas 200–202 (hechas) y sus reevaluaciones 300–302
(las tres hechas).** Todo lo anterior existe en el repo y no se borra, pero es
**historia**: no es plantilla, no se edita y no se cita como estado actual.

### Corridas vivas — `notebooks/simulaciones/`

| Carpeta | Escenario (anexo) | Generador | `BASENAME` | `M_FPCA_LIST` | covariables | estado |
|---|---|---|---|---|---|---|
| `200_sim_TAR` | **TAR de dos regímenes** (C-2) | `sim_escenario_TAR.py` | `escenario_200` | `(1, …, 10)` | rezago propio, `N_LAGS = 4` | entrenada y evaluada, 3 cadenas |
| `201_sim_GARCH` | **GARCH en los scores** (C-1) | `sim_escenario_GARCH.py` | `escenario_201` | `(1, …, 10)` | rezago propio, `N_LAGS = 4` | entrenada y evaluada, 3 cadenas |
| `202_sim_MULT` | **Multimodalidad** (C-3) | `sim_escenario_C1.py` | `escenario_202` | `(3, 4, 5, 6)` | **cruzadas**, `N_LAGS = L = 2` | entrenada y evaluada, 3 cadenas |
| `300_sim_TAR_estimadores` | el de la 200 | ninguno (copia de la 200 con `00_copiar_200.py`) | `escenario_300` | `(1, …, 10)` | las de la 200 | **evaluada** (`_04`, `_05`); `_01`/`.m`/`_03` copiados, no se ejecutan |
| `301_sim_GARCH_estimadores` | el de la 201 | ninguno (copia de la 201 con `00_copiar_200.py --origen escenario_201_chungEscG_1_r01 --destino escenario_301_chungEscG_1_r01`) | `escenario_301` | `(1, …, 10)` | las de la 201 | **evaluada** (`_04`, `_05`); `_01`/`.m`/`_03` copiados, no se ejecutan |
| `302_sim_MULT_estimadores` | el de la 202 | ninguno (copia de la 202 con `00_copiar_200.py --origen escenario_202_chungEscX_1_r01 --destino escenario_302_chungEscX_1_r01 --M 3 4 5 6`; cada `M` con sus trazas) | `escenario_302` | `(3, 4, 5, 6)` | las de la 202 | **evaluada** (`_04`, `_05`); `_01`/`.m`/`_03` copiados, no se ejecutan |

Las 30x reevalúan 200–202 con los predictores, intervalos y comportamientos de
`docs §03_05–§03_06`. El plan completo, con el contexto que lo justifica, está en
`notebooks/simulaciones/PROMPT_300_estimadores.md` (300, y al final 301 y 302).
Hallazgos que motivan las 30x: con predictiva multimodal la esperanza cae entre
los modos y pierde bajo MAE (tráfico −9 % con la mediana; Mapocho −13 % por
asimetría; 200 −1 %: sus regímenes difieren en dinámica, no en nivel; 201, 202 y
las demás reales sin diferencia); las reglas de comportamientos dan K = 2 en el
tráfico (69 % de los días) y K = 1 en 201, 202 y las reales restantes.

**Resultados de la 300** (test, w = 30; reproduce la 200 al bit en esperanza,
banda, FAR y RF). Bloque A contra la curva suavizada: la **mediana puntual** baja
el MAE 1.3 % y el E_max medio 2 % frente a la esperanza con M ≥ 5 (0.6087 → 0.6006
en M = 10); contra X^(M) hasta −5 % en M = 2. La **esperanza** gana el RMSE y el
E_max^glob (las medianas lo suben ~5 %). El medoide gana con M ≤ 5 y empeora con M
alto (es UNA extracción: arrastra el ruido de los scores pasivos); el **modal** es
el peor (+3.6 % MAE, +7 % RMSE). El RF sigue ganándole a todos en el Bloque A (MAE
0.591 en M = 10). |esperanza − mediana| > 0.25 sd robusta en 52–67 % de los
orígenes de test de los scores activos y 0 % de los pasivos. Comportamientos (todo
el test): K = 1 / 2 / 3 en 70 / 17 / 13 % con M ≥ 2 (lo fijan ξ_1, ξ_2); el modelo
declara K ≥ 2 en 30 % y la ley verdadera (`predictiva_oraculo_TAR`) en 9 %, porque
el gating deja ~11 % de masa en un modo tipo B donde P(B) ≈ 0; donde la ley sí es
bimodal, el modelo lo detecta en 22 de 25 (M = 1).
Intervalos (docs §03_05_05, con la esperanza; equi-colas, FAR y predictores
reproducidos al bit, RF a 1e-15): la **HDR** es la única construcción que se parte
(un solo tramo en 63 % de los (t, τ) con M = 1 y 88–90 % con M ≥ 5; fusiones ≤ 0.35 %) y
gana el Winkler a la equi-colas en 94 % de las ventanas (3.75 contra 3.91 en M = 5,
MPIW 3.35 contra 3.45, PICP 0.965 contra 0.957); las dos le ganan al FAR (4.86) en el
100 % de las ventanas desde M = 4. La **simultánea** lleva el PICPB a 0.98 (equi-colas
0.57, HDR 0.67, FAR 0.70) con MPIW 6.55 y pierde el Winkler (6.57). Con M ≤ 2 el
Winkler medio de las tres es peor que el del FAR (truncamiento).
Ventana de detalle (`PROMPT_30x_ventana_detalle.md`): t = 807..816, la de más cambios
de régimen verdadero en los scores activos (14). Cuentas sobre esos 10 orígenes, M = 5,
descriptivas: K ≥ 2 en 6 (oráculo 2), K igual al del oráculo en 6, el comportamiento
del régimen realizado es el más probable en 7 (oráculo 9); lo observado cae en el
comportamiento más probable en 7; curvas enteras dentro de la banda: equi-colas 8,
HDR 7, simultánea 10, FAR 4; la banda condicional del comportamiento realizado tiene
MPIW 3.03 contra 3.87 de la marginal.

**Resultados de la 301** (test, w = 30; reproduce la 201 a precisión de máquina en
esperanza, banda, FAR, RF y Bloque B). Predictiva unimodal: |esperanza − mediana| /
sd robusta tiene mediana 0.01–0.03 y p90 0.03–0.08 por score, > 0.25 en 0–0.3 % de
los orígenes; la **mediana puntual** queda a ≤ 0.3 % de la esperanza en MAE, RMSE y
Ē_max y a ≤ 0.8 % en E_max^glob (MAE 0.7113 / 0.7116 en M = 10), y la esperanza gana
por poco. El medoide empeora con M alto (+2.5 % MAE en M = 10) y el modal +0.7 % (la MBD
se eliminó; con ella la 301 reproduce al bit las filas de los otros cuatro).
Desde M = 4 el PSBPM-FD (esperanza) le gana en MAE y RMSE al FAR (0.7113 contra
0.7232; 0.8899 contra 0.9056 en M = 10) y al RF; con M ≤ 2 gana el FAR. Bloque B: el
Winkler le gana al FAR desde M = 4 (4.23 contra 4.35 en M = 10, 81 % de las
ventanas), W_max^glob desde M = 5, W̄_max desde M = 6 (9.85 contra 10.40) y PICPB
desde M = 7 (0.65 contra 0.62), con MPIW parecido (3.49 contra 3.42) y PICP 0.947
contra 0.940. Comportamientos: K = 1 en 99.7 % de test (≥ 98.7 % de train) y
catálogo global L = 1 en todo M; los K = 2 (t = 933 en test; 460–464 y 547–551 en
train) son orígenes con el score rezagado a 5–6 sd y h_t en su percentil 100, donde
la mezcla separa una curva de reversión (p ≈ 0.6) de una persistente y más ancha
(p ≈ 0.4). Banda contra h_t (fuera de las diez métricas; sobre todo el test, evaluación
previa a la ventana de detalle, mismo código): Spearman 0.61 en la curva
(0.58–0.70 por score activo), ancho medio 0.99 del oráculo, pero CV 0.07 contra 0.13:
la banda respira con h_t y sigue sus picos, con un piso más alto y menos amplitud
que la verdad.
Intervalos (docs §03_05_05, con la esperanza; equi-colas, FAR y predictores
reproducidos al bit): la **HDR** es un solo tramo en ≥ 99.85 % de los (t, g), sin
fusiones (en t = 933, el origen con K = 2, tampoco se parte: los dos comportamientos
se solapan en cada τ), y coincide con la equi-colas (Winkler 4.231 / 4.233 en M = 10,
MPIW 3.49 / 3.49; cara a cara le gana al FAR en 88 % de las ventanas contra 81 % de la
equi-colas, en M = 10). La
**simultánea** alcanza el nominal de curva completa (PICPB 0.94 / 0.97 en M = 5 / 10,
contra 0.60–0.66 de equi-colas y HDR y 0.62 del FAR) y baja W̄_max y W_max^glob (6.5 y
17–22 contra 9.6–10.7 y 44–49), pero es 1.5 veces más ancha (MPIW 5.0–5.3) y pierde en
Winkler (5.1–5.3 contra 4.35 del FAR; sólo es la mejor de las tres en M = 2, donde
equi-colas y HDR quedan cortas): el Winkler puntual elicita
la equi-colas. Contra h_t la HDR da lo mismo que la equi-colas y la simultánea algo
menos (Spearman 0.57): cambian el nivel del ancho, no la lectura.
**Ventana de detalle** t = 931..940 (contiene el máximo de h_t agregada en test, t = 933,
en la tercera posición; h_t agregada 1.28 → 2.34 → 0.55, mediana del test 0.70; métricas de
la ventana móvil reproducidas al bit). K = 2 sólo en t = 933 (1 de 10) y la HDR es un tramo
en los 10 orígenes. Banda contra h_t en la ventana (10 puntos, descriptivo): Spearman
0.79–0.88 para las tres construcciones, CV del ancho 0.19–0.24 contra 0.21–0.22 del
oráculo: en el episodio de volatilidad la banda sí acompaña a h_t. Contra lo observado: lo
observado cae en el comportamiento más probable en 10 de 10; equi-colas, HDR y simultánea
cubren la curva entera en 8 de 10 (fallan t = 932 y 940), el FAR en 7 de 10 (además t = 931);
t = 932, el origen anterior al pico, es el peor para todas (cobertura puntual 0.40 del FAR,
0.44–0.49 de equi-colas y HDR, 0.75 de la simultánea).

**Resultados de la 302** (test, w = 30, M = 4 / 6; las filas esperanza/equi-colas, la
banda del FAR y la validación previa se reproducen al bit). Predictiva unimodal:
|esperanza − mediana| > 0.25 sd robusta en 0 % de los orígenes (3 % sólo en ξ_5 con
M = 6). Bloque A: la
**esperanza** es el mejor predictor en MAE y RMSE en todo M (0.6895 / 0.6890 y
0.8658 / 0.8655); mediana +0.0–0.3 %, medoide +0.9–1.6 %, modal +1.4–1.6 %; FAR 0.6997 /
0.8804 y RF 0.699 / 0.877. Bloque B contra la curva suavizada: equi-colas Winkler 4.15 /
4.11 contra 4.23 del FAR (gana 76 / 81 % de las ventanas), PICP 0.931 / 0.944; la **HDR**
coincide con la equi-colas (4.16 / 4.11, un tramo en el 99.99 % de los (t, τ), sin
fusiones); la **simultánea** lleva el PICPB a 0.90 / 0.93 (FAR 0.56) con MPIW 4.6 / 4.9 y
pierde el Winkler (4.71 / 4.99, gana 3 / 1 % de las ventanas al FAR): el Winkler elicita
el intervalo central. Comportamientos (§12.1, todo el test): K = 1 en el 100 % de los
orígenes. Sobre todo el test (evaluación previa a la ventana de detalle, mismo código de
§12.4): la ley verdadera (`predictiva_oraculo_C1`) tiene K ≥ 2 en 3.0–3.3 % de test y
sólo en la **conjunta** (la marginal de ξ_1 nunca es bimodal); argmax π_t acierta Z_t en
74 %; la HDR del modelo tiene más de un tramo en 0.00–0.01 % de los (t, τ) contra
0.95–2.0 % del oráculo, y la masa verdadera que cubre es 0.956–0.959 (simultánea
0.959–0.962) contra 0.951 / 0.950 del oráculo, con la HDR ~8 % más ancha. **Ventana de
detalle** t = 776..785 (criterio: la de más orígenes con K ≥ 2 en la ley verdadera; 3 de
10, t = 782, 784, 785), cuentas descriptivas: el modelo da K = 1 en los 10 y en todo M
(acuerdo en K 7 de 10; catálogo L = 1); la HDR del modelo no se parte en ningún (t, τ)
y la del oráculo en el 4–8 % de ellos; la observada cae entera en la banda equi-colas
en 4–8 de 10 orígenes contra X^(M), según M. Lectura: la multimodalidad de C-3 casi no llega a
la ley marginal en τ; la banda unimodal del modelo está bien calibrada contra la verdad,
y su ventaja en el Bloque B no viene de la multimodalidad.

El generador común (`pipelines/sim_scores_comun.py`) arma la salida a partir de un
simulador de `z`, calcula `mu_theta`, `theta` y el control de calidad
`resumen_scores`. En `internos` van los scores, el oráculo, `Phi`, `soporte` y lo
propio de cada escenario (regímenes, varianza condicional, mecanismos); **nada de
eso se entrega a la estimación**.

- **200 — TAR.** Cada score activo `ξ_1..ξ_4` cambia de régimen (A tranquilo y
  persistente, B volátil y reversivo) según **su propio rezago** con
  `P(B|u) = Φ((u − c)/h)`; `ξ_5..ξ_10` son AR(1) pasivos. Calibración
  **deliberadamente favorable al gating** (hay que decirlo al presentarla): brecha
  oráculo−AR(2) 0.41/0.51/0.42/0.44 (0.386 agregada en L²). Parámetros en
  `PARAMETROS_TAR`, momentos poblacionales en `MOMENTOS_TAR`.
- **201 — GARCH.** Cada score activo es AR(1) con innovaciones GARCH(1,1)
  (`φ=0.5`, `α₁=0.25`, `β₁=0.60`, `σ_a²=1`): media **lineal**, varianza persistente
  (memoria de ~7 curvas, acf1 de los cuadrados ≈ 0.36). La innovación reciente `a_{t-1} = u_{t-1} − φ u_{t-2}`
  es función de los dos rezagos propios; con `N_LAGS ≥ 2` el término `α₁a²` es
  aprendible, `β₁` no. Se espera ventaja en el Bloque B.
- **202 — Multimodalidad.** Mezcla de 3 mecanismos lineales con asignación softmax
  sobre `ξ_{t-1}, ξ_{t-2}`; los mecanismos comparten el **patrón** de `A_{k,l}` y
  difieren en valores. La **media condicional es casi lineal** (R² oráculo 0.45
  contra 0.42 lineal): la multimodalidad está en la ley. El régimen depende de
  `ξ_1, ξ_2`, así que se usan **covariables cruzadas** (`p = M·L`) y se entrena
  **cada `M`**.

**Configuración común** (leída de los `_01`):

| Cantidad | Valor |
|---|---|
| `L_GRILLA` / `T_CURVAS` | **75** / **1000** (⇒ `T0 = 700`, test = 300) |
| `PROP_TRAIN` | 0.70 |
| `SIGMA_OBS` | **0** (los scores son el insumo) |
| `SEED` | 41232 |
| `REPLICA_ID` / `R` | 1 |
| `J` / `K` | 10 / 10 (base de Fourier del generador; acota el barrido en `M`) |
| `MCMC_CONFIG` | `{"nsim": 2500, "burn": 500, "N": 30, "M": 50}`, `N_CHAINS = 3` |
| ventanas | `w ∈ {20, 30, 40}`, `W_REF = 30` |
| scores | **sin estandarizar**: `scores_scale = "raw_fpca_scores"` |
| `M` (regla) | primera `M` con varianza acumulada `>= 95 %` (5 en estas corridas) |

**Hiperparámetros: Chung y Dunson §4.2 en escala cruda, sin escalera.** Valen igual
para todas las componentes: `atau = ag = bg = 0.5`, `btau = 0.5·sd_y²`,
`taupsij = taupsij_z·sd_x²` (las `sd` son de **train**; así el prior es
proporcional a la varianza de cada componente), `mupsij = 0`, `mumu = 0`,
`taumu = 1`, `pwj = 0.5`. Inclusión: `apij/bpij = 0.5/0.5` en el rezago 1 y
`0.5/5.0` del rezago 2 en adelante (y en los cruzados). La variante **`chungEscG`**
(200, 201) usa `taupsij_z = 10` (gating libre; sd a priori de ψ 0.32 en z) y
**`chungEscX`** (202) lo mismo con covariables cruzadas; `chungEsc` usa 100 (el
prior del paper). `G*` por predictor, sobre el rango de train con extremos.

Código del modelo que se conserva y se usa (§4): `fit/` (`convergencia_barrido`,
`evaluacion_barrido`, `comparacion_barrido`, `rolling`, `metrics_*`, `intervalos`,
`far_operador`, `inclusion`, `diagnostics_mcmc`), `functions_models/`
(`FunctionalRepresentation.desde_base`, `FPCA_L2.desde_scores_conocidos`),
`graphics/viz_*` y `models/pspb_fd_v3`. `functions_odpc.py` (ODPC) se conserva de
corridas anteriores y no lo usa ninguna corrida viva.

### Datos reales — `notebooks/reales/`

Dos generaciones. `21_real_nivel/` a `25_real_cripto/` siguen **la arquitectura
anterior** (base B-spline por GCV, FPCA estimada, escalera de hiperparámetros):
nivel diario del RÍO MAPOCHO EN LOS ALMENDROS (21–24, 48 mediciones por día) y
BTCUSDT de Binance en velas de 5 min (25, 288 barras por día UTC). `24` y `25`
son las vivas de esa generación (rezago propio `N_LAGS = 7`, `MCMC_CONFIG =
{5000, 2000, 40, 60}`, `N_CHAINS = 2`); 21–23 **no son comparables** con la 24.

`26`–`31` llevan el **pipeline de la 200 portado a reales** (B-spline 10/3 elegida
por el analista, FPCA en train, scores crudos, `chungEscG`, `atau = 2`,
covariables **cruzadas con un rezago** salvo que se indique): `26` BTC cierre,
`27` precio SA1 (AEMO), `28` PM2.5 Parque O'Higgins (SINCA), `29` BTC volatilidad
realizada (+ exógenas), `30` tráfico I-94 barrido `M = 1..6` y `31` tráfico
`M = 10` (1 cadena). Los datasets de la tesis de Katerin (`references/`) están en
`data/reales/raw/katerin_*`; el tráfico (régimen semanal) y el PM2.5 crudo
(volatilidad) son los candidatos; household no. **`32` = la 31 con rezagos `{1, 7}`**
(`notebooks/reales/32_real_trafico_l17/`, `SERIE_ID = trafico_l17`, M = 10, 1 cadena,
inclusión (1, 5) también para el rezago 7): los cuatro notebooks y los `.m` están creados
(2026-10-11) y **sin ejecutar**. `32_01`/`32_03`/`.m` son la 31 con `LAGS = (1, 7)`;
`32_04`/`32_05` son los `300_04`/`300_05` (plantilla 30x: cuatro predictores, tres
intervalos, comportamientos, ventana de detalle) adaptados a reales: identidad
`real_<serie>_v<NN>`, `BURN_EVAL = 1000`, sin generador ni oráculo; la ventana de detalle
se elige por **calendario** (la de 10 días de prueba con más días no laborables, con la
columna `holiday` del CSV crudo) y el §12.4 cruza `K` con el día de la semana y los feriados
(`65_comportamientos_calendario.csv`, `104_comportamientos_calendario_por_M.csv`). Orden de
ejecución: `32_01`, MATLAB (10 jobs), `32_03`, `32_04`, `32_05`. Rezagos no contiguos: el
manifest lleva `lags = [1, 7]` y `n_lags = 7` (= `max(lags)`, el desplazamiento y
`t_offset`); `verificar_contrato` acepta `lags` (cambio aditivo en `artifacts.py`);
MATLAB lee `feature_names` y no cambia. Decisiones para el `_05`: el FAR de referencia es
el **FAR(7) completo** (`p = N_LAGS = 7`, sale solo), y el RF debe barrer los
subconjuntos `{1}` y `{1, 7}` (`ajustar_rf(..., ordenes=LAGS)`, argumento aditivo: `r = 1`
usa `{1}` y `r = 7` usa `{1, 7}` porque `_cols_propias` filtra `lag <= r`; sin `ordenes`,
`1..N_LAGS` como siempre). Los datos de cada serie, su calidad y su calibración están en los
`_01`, no se repiten aquí.

### Historia — no se edita ni se cita como estado actual

- `notebooks/simulaciones/06_07_Testo_cosas/` (corridas 61–66, 71–73),
  `07_01 Antiguos casi listos/` (101–114: variantes de rezago propio, mezclas de
  mecanismos y de scores, ODPC, variantes MCMC y el TAR de la 114, antecedente
  directo de la 200) y `00_Estudio_Modelo/`, `01_Inicio_Formato/`,
  `02_Antiguas_pruebas/`, `03_04_Antiguas_pruebas/`, `05_Testeo_hiperparametros/`.
  Usaban base B-spline por GCV, ruido de medición `sigma_obs = 0.25`, scores
  estandarizados o escalera de hiperparámetros, y `N_LAGS`/`MCMC` distintos: **sus
  cifras no son comparables una a una** con las de 200–202.
- Las corridas `30`–`34` y `50`–`53`: datos, reportes y artefactos siguen en disco.
- `model_psbp_fd/pipelines/deprecated/` — generadores de esas corridas; siguen
  importables desde ahí, **no** desde `pipelines` directamente. Incluye ahora
  `sim_escenario_A1/A2/A3` y `C2/C3`. Siguen en `pipelines` `B1/B2/B3` y `C`, sin
  corrida viva, por dependencias; `C1` es el generador de la 202.
- `versioning/`, `escenarios_simulados.txt` y `modelo_e_hiperparametros.txt`
  describen corridas y un Gibbs anteriores y **están desactualizados**.

---

## 8. Contradicciones abiertas

Señaladas, no resueltas. **Preguntar antes de codificar contra ellas.**

1. **`docs/03 Modelo.tex`: "Diseño del Estudio de Simulación" tiene dos párrafos
   y una subsección "Escenarios" vacía; "Resultados" es un esqueleto con
   `% [PENDIENTE]`.** Falta escribir objetivos de evaluación, modelos de
   referencia, especificación del PSBPM-FD, qué resumen va con qué métrica y qué
   se reporta. Esquema propuesto en la sesión del 2026-10-10 (seis subsecciones);
   no redactar sin pedido.
2. **Qué escenarios se muestran en el documento.** Hoy son tres (C-1 GARCH, C-2
   TAR, C-3 multimodal) pero el conjunto puede crecer. Con las reglas de
   comportamientos, sólo el tráfico produce predictivas multimodales; 200–202 tal
   como están calibradas no (el TAR separa dinámica, no nivel). Si se quiere un
   escenario simulado donde los comportamientos aparezcan, hace falta un
   generador con regímenes que difieran en **nivel** (tipo calendario).
3. ~~Prefijo `03_06_` duplicado~~ Resuelto el 2026-10-11: Diseño del estudio es
   `03_07_*`, Resultados `03_08_00`, Intervalos `03_05_05_*`. El prompt de la 300
   puede citar las etiquetas viejas; las `
ef` de `docs/` ya están actualizadas.
4. **El anexo (`docs/01 Anexo.tex`) y el capítulo 03 no coinciden en el predictor.**
   El capítulo describe un predictor común con los rezagos de **todas** las
   componentes; 200 y 201 usan **rezago propio** (sólo 202 es cruzado). Se
   resuelve escribiéndolo como decisión por escenario en el diseño del estudio.
5. **Una réplica por escenario, por decisión (2026-10-10).** No habrá réplicas;
   `REPLICA_ID` se conserva en el `EXPERIMENT_ID` y en las rutas para que el
   código lo permita, pero siempre vale 1. Consecuencia que el documento declara:
   las diferencias entre modelos son evidencia descriptiva, y una diferencia de
   pocos puntos porcentuales no se lee como superioridad. No proponer réplicas.
6. **`200_sim_TAR` está calibrada para favorecer al gating** (calibración de la
   114 "exagerada"). Las conclusiones sobre el TAR valen para esa calibración, no
   para un TAR genérico.
7. **`atau = 0.5` en 200–202 deja átomos con precisión ≈ 0**: los draws tienen
   colas no acotadas (curtosis 10⁵–10⁶). La esperanza analítica y los cuantiles no
   lo sufren; **la media de los draws sí** y nunca se usa como estimador; antes de
   clusterizar se recortan los draws a los cuantiles 0.5–99.5 %. Las reales usan
   `atau = 2`. Si se reentrena una simulación, decidir `atau` antes.
8. **La banda equi-colas es puntual y PICPB mide cobertura simultánea**: queda bajo
   el nominal para todos los modelos por construcción (0.72 FAR / 0.75 PSBPM-FD en
   tráfico). Compara modelos, no mide calibración. Desde el 2026-10-11 la banda
   simultánea y la HDR de `§03_05_05` están implementadas
   (`fit/intervalos_predictiva.py`) y se reportan en las 30x; PICPB sigue siendo
   cobertura de curva completa y sólo la simultánea está calibrada para ella.
9. **Las etiquetas de los átomos no convergen** (ψ, Γ, α; los β sí) y la mezcla
   toca `N = 30` en varias componentes de las reales. No afecta a la predictiva
   (invariante a etiquetas), pero R̂ no se calcula con 1 cadena (30/31).
10. **`pipelines/real_pipeline.py` es un stub `TODO`** de seis líneas. El flujo de
    datos reales vive en `notebooks/reales/`.
11. **La regla de valle (`docs` eq. `valle`) evalúa la densidad en los
    centroides**, pero el texto dice "el menor de los dos picos". Con tres modos
    colineales, k-medias con K = 2 junta dos y su centroide cae en el valle entre
    ellos: la razón no baja de 0.5 y la regla da K = 1 (a veces 2), nunca 3
    (`tests/test_comportamientos.py`). En R² no colineales funciona. Implementado
    como dice la ecuación; evaluar en los picos lo arreglaría, pero cambia la regla.
12. ~~"Cinco predictores" en el docs~~ Resuelto el 2026-10-11 por decisión: el
    documento define y reporta **cuatro** (sin la mediana MBD), y la MBD se eliminó
    del código (`PREDICTORES` tiene cuatro; sin `profundidad_mbd` ni `mediana_mbd`,
    sin filas `PSBPM-FD (mbd)`).
13. **Las 30x duplican las trazas de su 20x** (~220 MB de `.mat` en la 300), y los
    `.mat` se versionan. `trazas_en` podría apuntar a la 20x sin copiarlas, pero el
    plan pidió una 30x autocontenida.

Resueltas en la sesión del 2026-10-10 (no reabrir): la referencia a
`03_06_04_objetivos_evaluacion` ya no existe; `§03_05` quedó sin estandarización
(`ξ` crudo, sin `(c_m, d_m)`); 201 y 202 están entrenadas y evaluadas.
