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
paquete `model_psbp_fd` genera datos, preprocesa a base + FPCA, consume trazas
MCMC y evalúa predicciones. **El muestreo MCMC ocurre en MATLAB**, no en Python.

Código, docstrings y nombres están en español; mantener ese idioma. Los
docstrings de módulo son extensos a propósito: documentan *por qué* se tomó cada
decisión numérica y suelen traer la respuesta antes que el código.

- En `docs/` está el material actual de la tesis (`01 Anexo.tex`,
  `02 Marco Teorico.tex`, `03 Modelo.tex`).
- Ese material es **referencia, no contrato**: se puede modificar. De ahí salen
  indicaciones, no obligaciones.
- El foco experimental son **escenarios donde el modelo puede ganar**. El diseño
  se orienta a caracterizar *cuándo y por qué* el PSBPM-FD supera a las
  alternativas, no a probar que gana siempre. Un resultado negativo bien acotado
  (p. ej. el Escenario 1, donde la media condicional es lineal y el FAR está
  correctamente especificado) es un resultado, no un fracaso.

---

## 3. Invariantes

Estos tres elementos **no cambian entre escenarios**. Si una instrucción mía los
contradice, avisar **antes** de ejecutar.

### 3.1 Métricas — el conjunto es fijo y son diez

**La fuente de verdad es `docs/02 Marco Teorico.tex §02_02_03`**, y el conjunto
es exactamente el que ahí se define: ni una más. Si una métrica no está en esa
sección, no se reporta; si se define ahí, se reporta.

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
`METRICAS_B`, con el mismo contenido en ambos notebooks, y `_01` las persiste en
`eval_config.json` bajo `metricas_bloque_A` / `metricas_bloque_B`.

Relaciones verificadas con `assert`, que **no** deben romperse:
`mae_f <= l2_medio <= linf_medio <= linf_max`, `picp_simultaneo <= picp`, y
`winkler <= winkler_max_medio <= winkler_max_glob`.

**`rmse_f` NO pertenece a la cadena L¹ ≤ L² ≤ L^∞.** Es la raíz del MSE
*agregado*, es decir una media **cuadrática** sobre los orígenes, no aritmética;
por Jensen `l2_medio <= rmse_f`, pero `rmse_f <= linf_medio` **es falso** cuando
el error es desigual entre los orígenes de la ventana. La cadena se verifica
sobre `l2_medio`.

**`linf_medio` es el máximo real por curva promediado sobre la ventana**, no un
cuantil que lo aproxima. `Q_EXTREMO` / `q95_abs` fueron eliminados. `linf_max` y
`winkler_max_glob` dependen de una sola evaluación y son las cifras frágiles del
conjunto: se reportan porque el marco teórico las define, no porque sean
robustas.

**`mise` y `mise_rel` siguen en la tabla** pero son diagnóstico interno (el salto
en `T0`, el piso de truncamiento), no métricas reportables: no están en
`§02_02_03`.

### 3.2 Ventanas temporales — la segmentación es fija

- Anchos `w ∈ {10, 20, 40}`, `paso = 1`, solapadas.
- Ancho de referencia: `W_REF = VENTANAS_W[len // 2]` = **20**. Las figuras
  superponen los tres anchos; las tablas usan `W_REF`.
- **Las ventanas que cruzan `T0` se excluyen** de todo agregado (`cruza_T0`).
- Partición: `PROP_TRAIN = 0.70`, `T0` marca el corte y va en el manifest.
- Se **leen** de `eval_config.json["ventana_movil"]`; **no se redeclaran** en
  `_04` / `_05`.

### 3.3 Evaluación sobre todos los `M`

`M_FPCA_LIST` se declara en **los cuatro notebooks y en el `.m`**, y los cuatro
recorren el barrido completo en una sola pasada.

- `_01` produce un juego de artefactos por cada `M` (bucle `procesar_punto(M)`
  en §4); los objetos comunes —`fpca`, `THETA`, la partición, la base— se pasan
  por clausura y **no se recalculan**: ésa es la garantía de que los puntos
  comparten realización y base.
- `_03`, `_04` y `_05` cargan cada punto en `EST[M]` e iteran sobre él.
- `psbp_fd_iteracion.m` arma **una sola lista plana** de jobs
  `(M × cadena × componente)` y la reparte con un único `parfor` — no un
  `parfor` por `M`, porque el último punto tiene más jobs que el primero y con
  tandas separadas los workers quedan ociosos al final de cada una.
- `SALTAR_M_SIN_TRAZAS` / `SALTAR_M_SIN_ARTEFACTOS` permiten mirar el barrido
  mientras MATLAB va terminando.
- Los puntos **deben compartir diseño**: `T`, `T0`, `n_lags`, `nivel`,
  `modo_residuo`, `objetivo_evaluacion`, ventanas y `mcmc_config` idénticos, y
  las mismas curvas verdaderas. Si no, la comparación entre `M` mezclaría el
  efecto de `M` con el de otra cosa, y el notebook **falla en vez de callar**.

`M` significa dos cosas y hay que tenerlo presente: en FPCA es el número de
componentes retenidas; en `mcmc_config` es el tamaño de la grilla de
localización G\* del stick-breaking (ahí `N` es el truncamiento de átomos).

---

## 4. Estructura canónica: la corrida 30

**`notebooks/simulaciones/30_sim_E1/` es la plantilla.** Toda simulación nueva
sigue **esa misma estructura, sin agregados**: se cambian parámetros o
escenario, no la arquitectura del pipeline.

La secuencia canónica son **cuatro notebooks y un paso MATLAB**, acoplados por
archivos en disco:

| Paso | Archivo | Qué hace |
|---|---|---|
| 1 | `30_01_simulaciones.ipynb` | **Genera los datos** y los prepara: simulación, partición temporal, base B-spline por GCV, FPCA, estandarizador, datasets AR, contrato. Un juego de artefactos por cada `M`. |
| 2 | `psbp_fd_iteracion.m` | **Entrena.** Lee el contrato, arma los jobs y llama a `psbp_train.m`. Sólo con el bloque train. No tiene notebook. |
| 3 | `30_03_convergencia.ipynb` | **Evaluación de convergencia.** No toca el bloque de prueba. |
| 4 | `30_04_evaluacion.ipynb` | **Evaluación propia del modelo**: las 7 métricas sobre la ventana móvil, PIP, monitoreo por componente. |
| 5 | `30_05_comparacion.ipynb` | **Comparación** contra los modelos de referencia. |

> **Corrección respecto de la formulación informal de esta regla:** `30_01` no
> es "el entrenamiento" — genera los datos. El entrenamiento es el paso MATLAB.
> El "mapa base" es la carpeta `30_sim_E1/` completa, no el notebook `30_01`.

Si las cadenas no convergen, los números de `_04` no significan nada: ésa es la
razón de que `_03` sea un notebook aparte y con su propia condición de parada.

### Secciones de cada notebook

`_01`: §1 imports y rutas · §1.1 `[CONFIG]` experimento y barrido en `M` · §2
simulación (§2.1 `[CONFIG]` generador) · §3 representación B-spline (§3.1 GCV,
§3.2 `[CONFIG]` base elegida, §3.3 FPCA) · §4 bucle sobre `M_FPCA_LIST` · §5
resumen del barrido.

`_03`: §1 artefactos · §2 trazas · §3 tabla de diagnósticos · §4 ocupación de la
mezcla · §5 figuras por parámetro · §6 PIP (§6.1 contraste con el generador) ·
§7 verificación muestreador ↔ predictor · §8 comparación entre puntos del
barrido.

`_04`: §1 artefactos · §2 trazas · §3 predicción a `h=1` (§3.1 persistencia) ·
§4 Bloque A sobre la ventana móvil (§4.1 qué `M` gana cada ventana) · §5 Bloque
B · §6 muestra de predicciones · §7 PIP · §8 resumen mín/máx/promedio · §9
monitoreo univariado por componente FPCA.

`_05`: §1 artefactos · §2 validaciones · §3 competidores (§3.1 FAR sobre
coeficientes B-spline, §3.1.1 contraste con el estimador retirado, §3.2 RF/GBT,
§3.3 líneas base, §3.4 predicción puntual del PSBPM-FD) · §4 Bloque A todos los
modelos · §5 Bloque B sólo FAR y PSBPM-FD · §6–§7 quién gana cada ventana · §8
resumen · §9 intervalos curva a curva.

### Los competidores del `_05`

`FAR` (sobre los coeficientes B-spline, en la métrica L²) · `RF` · `GBT` ·
`media` (incondicional) · `persistencia`.

**Hay una sola referencia lineal, no dos.** El `VAR` sobre scores y el `FAR1`
por momentos sin centrar eran el mismo estimador salvo el intercepto; se
retiraron. `_far_scores_viejo` se conserva **sólo** como término de comparación
en §3.1.1 y no predice nada. Al citar resultados: *una* referencia lineal.

El FAR se estima blanqueando `THETA` con la Cholesky de la Gram de la base
(`W = LL^T`, `theta_w = THETA @ L`), de modo que el producto escalar euclídeo de
`theta_w` **es** el producto L² exacto — verificado con `assert`. Por eso se
ajusta con `pesos="conteo"`: pesar de nuevo con la trapezoidal aplicaría la
métrica L² dos veces. El tope de `kn` es `K` (funciones base del GCV), no `L`.

El Bloque B se restringe a FAR y PSBPM-FD —los únicos con mecanismo de intervalo
propio— con tres filas por `M`: la banda nativa de cada uno y una banda
gaussiana del PSBPM-FD con el mismo mecanismo del FAR, para separar "gana por el
modelo" de "gana por la forma de la predictiva".

---

## 5. Cómo trabajar conmigo

- Si una instrucción mía rompe los invariantes (§3) o la estructura de la
  corrida 30 (§4), **avisar antes de ejecutar**.
- **No agregar pasos, métricas ni etapas** al pipeline sin que yo lo pida
  explícitamente.
- Ser conciso: priorizar código, comandos y diagnósticos sobre explicaciones
  largas.
- No leer ni escribir artefactos a mano desde un notebook: usar
  `pipelines/artifacts.py` y `utils/rutas.py` (§6.3).
- `notebooks/simulaciones/30_sim_E1/` es la plantilla y se edita con cuidado.
  Las corridas archivadas (§7) no se tocan.

---

## 6. Referencia técnica

### 6.1 Instalación y pruebas

```bash
pip install -e .
```

Alternativa conda en `environment.yml` (entorno `psbp_fd`).

**Sí hay tests**, aunque no hay configuración de pytest ni CI:

```bash
python -m pytest tests/test_metricas_bloques_AB.py tests/test_far_operador.py -v
```

`tests/test_metricas_bloques_AB.py` cubre los Bloques A y B y su enganche con la
ventana móvil; `tests/test_far_operador.py` contrasta el FAR contra salidas de
R. Ambos corren también como script. No hay linter configurado.

### 6.2 El ciclo Python → MATLAB → Python

Las tres etapas se acoplan **por archivos en disco**, no por llamadas.
`hyperparameters.json` es la **única fuente de verdad del contrato Python ↔
MATLAB** (`n_iter`, `mcmc_config`, `hyperparams_list`, `partition`,
`seed_base`). **No hardcodear esos valores en el `.m`.**

`psbp_fd_iteracion.m` **lee `seed_base` del JSON** y falla si falta. La semilla
por job es `seed_base + chain*9973 + k*31`.

Hay **una copia de `config_paths.m` y `psbp_train.m` por carpeta de
experimento**: un cambio de convención se propaga a mano a todas.

### 6.3 Artefactos: dónde vive cada cosa

Los directorios no son intercambiables y el `.m` depende de la distinción:

| Clave | Ruta | Contenido |
|---|---|---|
| `raw` | `data/<dominio>/raw/<EID>/` | `X_curves.npy`, `X_curves_true.npy`, `simulation_config.json` |
| `functional` | `data/<dominio>/processed/functional/<EID>/` | `datasets_manifest.json`, `dataset_fpc_<idx>_{train,test}.csv`, CSV de FPCA, estandarizador, `theta.csv` |
| `predict` | `data/<dominio>/processed/predict/<EID>/` | `banda_funcional_psbp.npz` |
| `out_report` | `reports/<dominio>/<EID>/` | figuras y CSV del punto |
| `out_artefact` | `artefact/<dominio>/<EID>/` | `hyperparameters.json`, `eval_config.json`, trazas `.mat` |

- **`utils/rutas.py` construye los directorios** (`experiment_id`,
  `construir_paths`, `rutas_por_M`, `ruta_barrido_M`, `guardar_en_todos`,
  `replicar_figura`). Es el gemelo Python de `config_paths.m`; si una cambia, la
  otra también. `CLAVES_PATHS` es el juego exacto de cinco claves que esperan
  todos los consumidores.
- **`pipelines/artifacts.py` decide los nombres de archivo** (dict `ARCHIVOS`) y
  centraliza escritura y lectura emparejadas. `verificar_contrato()` cruza
  manifest, hiperparámetros y artefactos FPCA antes de analizar.
- `construir_paths(limpiar=True)` vacía `raw`, `functional`, `predict` y
  `out_artefact`, pero **no** `out_report`: sus figuras se sobrescriben por
  nombre y sirven de registro visual de la corrida anterior, mientras que un
  dataset o una traza rancios se leerían como si fueran de ésta.

### 6.4 El modelo

**`psbp_fd_v3` es la versión que se usa** y no tiene muestreador: trazas `.mat` →
predictiva por score → `PropagadorFuncional` (des-estandariza + reconstruye
FPCA) → muestras de curvas `(S, n, G)`, que son el insumo de `fit/`.

La puerta de entrada de los notebooks vivos es `ModeloTraza`: `leer_traza` da
`(traces, burn, feature_names)` y `ModeloTraza.momentos(df)` / `.muestrear(df, S)`
producen media, sd predictiva y extracciones por score. `PSBPPredictor`,
`muestrear_scores` y `bandas_puntuales` siguen exportados y son la capa de
abajo. **No redefinir la convención de nombres de traza en un notebook**: usar
`ruta_traza`.

`curva_media_desde_scores` es el mapa determinista (`modo_residuo="ninguno"`, sin
muestreo) que necesitan los competidores para entrar en la misma escala de curva
que el PSBPM-FD. `residuos_representacion` estima el residuo de representación
para el modo `"empirico"` — sólo sobre el bloque de entrenamiento.

`v1` / `v2` son heredadas. `models/__init__.py` las importa de forma tolerante
pero trazable (`v1` depende de un `.pyd` compilado por plataforma); usar
`estado_versiones()` para diagnosticar. `v3` se importa estricto a propósito.

**Cambio de contrato v2 → v3:** `predict(return_std=True)` devolvía la desviación
posterior *de la media condicional*; en v3 devuelve la **predictiva** (ley de
varianza total). La cantidad vieja sigue disponible como `sd_centro`. Usar la de
v2 como banda produce subcobertura que se confunde con fracaso del modelo.

### 6.5 El error se mide contra la curva verdadera

`SalidaSimulacion` distingue `observaciones` (con ruido `sigma_obs`) de `curvas`
(la curva verdadera `X_t(tau)`).

- `guardar_curvas(paths, X, grilla, X_true=...)` persiste las dos por separado;
  `cargar_curvas_true()` la lee y **falla si no existe**.
- La **observada** es el único objeto que alimenta la estimación: base, FPCA,
  estandarizador, scores. La **verdadera** entra sólo como objetivo de
  evaluación.
- Se combina con `modo_residuo="ninguno"`: la banda cubre la curva **proyectada**
  sobre las `M` autofunciones y se contrasta con `X_t(tau)`, de modo que lo que
  queda fuera es truncamiento FPCA puro — cantidad interpretable y acotable
  subiendo `M`. **Declararlo al reportar.**
- `objetivo_evaluacion` y `modo_residuo` se registran en `eval_config.json`
  desde `_01`; **no se deciden en el notebook de evaluación**.

En datos reales no hay curva verdadera: el objetivo es la observada y
`modo_residuo="empirico"`, porque el error de representación forma parte de lo
que la banda tiene que cubrir. **Las cifras de simulación y de datos reales no
son comparables**: allí el denominador excluye el ruido de medición y aquí lo
incluye.

### 6.6 Convenciones

- **Ejes del estudio**, sin abreviar: `ESCENARIO_ID`, `REPLICA_ID`, `chain`
  (cadena MCMC), `k` (componente FPCA), `M` (componentes retenidas).
- **`EXPERIMENT_ID`** nombra por igual `data/`, `artefact/` y `reports/`, y debe
  coincidir **exactamente** entre los cuatro notebooks y el `.m`. Convención
  vigente: `f"{BASENAME}_{ESCENARIO_ID}_r{REPLICA_ID:02d}_m{M_FPCA:02d}"` (p. ej.
  `escenario_1_r01_m02`, `escenario_J_1_r01_m03`). Datos reales:
  `real_<serie>_v<NN>_m<NN>`.
  `M` viaja en el id para que cada punto del barrido escriba sus propios datos,
  trazas y reportes sin pisar los demás — por eso `M_FPCA_LIST` se declara en la
  celda `[CONFIG]` de §1.1, **antes** de construir las rutas.
- **Índices base-0 vs base-1.** `component_idx` del manifest es base-0; los
  nombres de archivo usan `fpc_idx = component_idx[k] + 1`. Las trazas se llaman
  `chain_fpc_<fpc_idx>_iter<chain a 2 dígitos>.mat` (p. ej.
  `chain_fpc_2_iter03.mat`) — **no cambiar**, el flujo de resultados los busca
  por nombre.
- **Salidas del barrido**: lo que cruza los puntos va a
  `reports/<dominio>/<BASENAME>_<ESCENARIO>_r<NN>_barrido_M/`, hermano de los de
  cada `M`.

  | Notebook | Por `M` (`out_report`) | Barrido (`_barrido_M`) |
  |---|---|---|
  | `_01` | `01`–`08`, `30_baselines_test.csv` | — |
  | `_03` | `40`–`47` | `80`–`83` |
  | `_04` | `54`–`57`, `60`–`61`, `66`–`67`, `90`–`91` | `84`, `89` |
  | `_05` | `72`, `78`–`81b`, `94`–`95` | `96`–`99` |

  Los prefijos `80`/`81` aparecen en dos notebooks pero en **directorios
  distintos** (`_03` escribe al barrido, `_05` al de cada `M`): no hay colisión,
  pero la numeración ya no es una partición limpia. No "arreglarla" sin pedirlo.
- Retención temporal: el estandarizador se ajusta **sólo** con el bloque de
  entrenamiento y guarda `n_ajuste` / `etiqueta_ajuste` para que eso sea
  auditable. `scores_scale` debe ser `"standardized_zscore_ddof0"`.
- FPCA con patrón `fit`/`transform`: `fit` recibe sólo el bloque de
  entrenamiento, de modo que la ausencia de fuga es una propiedad de la clase y
  no una disciplina del notebook.
- `.gitignore` excluye `*.npy` y los compilados; los `.mat` y las figuras `.png`
  **sí** se versionan.

### 6.7 Gotchas

- **MATLAB colapsa la dimensión singleton final al guardar.** `zeros(nsim, N, p)`
  con `p = 1` se escribe como `nsim x N`, de modo que `betajhout`, `psijhout`,
  `Gammajhout` y `gammajhout` llegan a Python con dos ejes en vez de tres. Ocurre
  exactamente en el punto **`M = 1`** del barrido; hacía fallar a
  `PSBPPredictor` con `IndexError` y a `pip_por_componente` **en silencio**.
  `utils/trazas.py` (`normalizar_trazas_mat` / `asegurar_3d`) restaura el eje. La
  reconstrucción es inequívoca porque `beta0hout` es `(nsim, N)` por definición.
  **No leer `traces["betajhout"].shape[2]` a mano**: usar `predictor.n_features_`.
- **Con `M = 1` hay dos trampas más de forma**, ambas en el `_05`:
  `np.cov(SCORES.T)` devuelve un escalar 0-d y rompe `np.linalg.cond` (se envuelve
  en `np.atleast_2d`), y `RandomForestRegressor` ajustado con un objetivo `(n, 1)`
  lo aplana y `predict` devuelve `(n,)` (se hace `reshape` a la forma del
  contrato). Ninguna aparece con `M >= 2`.
- **`np.loadtxt` colapsa a 1D** con una sola columna; por eso `artifacts.py`
  fuerza 2D en `_MATRICES_2D`. Deja de ser hipotético con una sola componente.
- **Cuadratura y Cholesky tienen una sola definición.**
  `utils.quadrature.pesos_trapezoidales` y `utils.linalg.safe_chol`;
  `pipelines.sim_comun` las re-exporta (`factor_cholesky`) sin redefinirlas.
  Duplicarlas degrada en silencio la ortonormalidad FPCA o cambia las
  trayectorias ante la misma semilla.
- **`base_en_grilla` obtiene cada base por diferencia**
  (`reconstruct(e_k) - reconstruct(0)`) porque con `center=True` la
  reconstrucción es afín y contaminaría la base con la media funcional.
- **No usar `crps_gaussiano` / `lps_gaussiano` / `pit_gaussiano`** con la
  predictiva del modelo propuesto: son la aproximación de dos momentos y
  descartan la multimodalidad/asimetría que el estudio existe para medir. Usar
  las versiones muestrales.
- **Los diagnósticos MCMC se calculan sobre cantidades invariantes a la
  permutación de etiquetas** de la mezcla (`extraer_traza_variable` promedia
  sobre átomos): con un `h` fijo, R-hat mide el etiquetado y no la convergencia.
  `fit/diagnostics_mcmc.py` tiene la única definición; `graphics/viz_traces.py`
  la **importa** en vez de redefinirla.
- **`fit/rolling.py` no reentrena**: desliza la ventana de *evaluación* sobre una
  serie de predicciones a `h=1` con rezagos reales, etiqueta cada ventana por
  bloque y marca las que cruzan `T0`.
- Los comentarios `% [FIX]` / `% [FIX N]` en los `.m` documentan correcciones
  deliberadas contra la implementación original (grid G\* sobre `Xnoint` sin el
  intercepto, `randi` que ignoraba parte del rango, factores `exp(1.2*·)`
  eliminados). **No revertirlos sin entender el motivo.**
- **`pipelines/real_pipeline.py` es un stub `TODO`** de seis líneas. El flujo de
  datos reales vive en `notebooks/reales/`.
- **`versioning/` está obsoleto**: `changelog.md` describe un Gibbs de 8 pasos
  con prior MNIW y un directorio `configs/` que no existen en el diseño actual;
  `experiment_registry.md` está vacío.

### 6.8 Terminología

**FAR(1)** autorregresivo funcional · **FGARCH** GARCH funcional · **HS** norma
de Hilbert-Schmidt (`< 1` ⇒ operador contractivo) · **FPC/FPCA**
componente/análisis principal funcional en métrica L² (problema generalizado
`C u = λ W u`, con `W` la Gram de la base) · **MISE** error cuadrático integrado
· **Winkler**, **PICP**, **MPIW**, cobertura simultánea: Bloque B de
`fit/metrics_distribucional` · **T0** corte de la partición temporal · **G\***
grilla de localización del stick-breaking probit · **oráculo** la media
condicional verdadera proyectada sobre las mismas `M` autofunciones, cota
superior de lo alcanzable.

---

## 7. Estado del repositorio

### Corridas vivas — `notebooks/simulaciones/`

| Carpeta | Escenario | `BASENAME` | `M_FPCA_LIST` |
|---|---|---|---|
| `30_sim_E1` | Algoritmo 1 del anexo (FAR(1) lineal gaussiano) | `escenario` | `(1, 2, 3)` |
| `31_sim_T1` | núcleo cuadrático + tendencia cuadrática | `escenario_T` | `(1, 2, 3)` |
| `32_sim_R1` | recursión **par** + tendencia cúbica | `escenario_R` | `(1, 2, 3)` |
| `33_sim_EJ` | Escenario J: cuatro rezagos, grado polinomial decreciente | `escenario_J` | `(2, 3, 4)` |
| `34_sim_EK` | Escenario K: heterocedasticidad condicional en tau | `escenario_K` | `(1, 2, 3)` |

`30_sim_E1` es la plantilla. Los cinco comparten esquema de observación,
`mcmc_config`, priors y `N_CHAINS`: cambiarlos convertiría la comparación entre
escenarios en una comparación entre ajustes.

**Configuración común de las corridas vivas** (leída de los `_01`):

| Cantidad | Valor |
|---|---|
| `L_GRILLA` | **100** |
| `T_CURVAS` | **800** (⇒ `T0 = 560`, test = 240) |
| `PROP_TRAIN` | 0.70 |
| `SIGMA_OBS` | 0.25 |
| `mu(tau)` | `sin(2 pi tau)` (salvo `31`, que usa una media cuadrática) |
| `SEED` | 41232 |
| `N_LAGS` | 1 |
| `N_CHAINS` | 2 |
| `MCMC_CONFIG` | `{"nsim": 2500, "burn": 500, "N": 35, "M": 35}` ⇒ 4000 draws posteriores por score |
| priors | `E[pi] = 0.90` sobre el propio rezago, `0.50` sobre los cruzados |
| `M` (regla) | primera `M` con varianza acumulada `>= 95 %`; el barrido la rodea |
| base B-spline | elegida por **GCV en cada escenario** ⇒ `K` no es invariante y **acota el barrido en `M`** |

**La base no es invariante entre escenarios** y eso hay que declararlo: un `M`
alto en un escenario con `K` chico cae sobre componentes que son en buena medida
artefacto de una base corta, y no mide lo mismo que en uno con `K` grande.

### Datos reales — `notebooks/reales/`

`21_real_nivel/` sigue la misma arquitectura (`21_01_datos`, `21_03`, `21_04`,
`21_05`). Serie: nivel diario del RÍO MAPOCHO EN LOS ALMENDROS, 48 mediciones
por día, cada día una curva. Reportes existentes: `real_nivel_v01_m03`,
`real_nivel_v02_m03`, más varias corridas `datos_{nivel,caudal}_*`.
`Ejemplo_{1,2,3}/` siguen la arquitectura antigua y no se usan.

### Archivadas — no se editan

- `notebooks/simulaciones/00_Estudio_Modelo/` — validaciones del modelo.
- `notebooks/simulaciones/01_Inicio_Formato/` — corridas `03`–`10`, formato
  antiguo (evaluación completa en un solo `_03`, eje `tt`).
- `notebooks/simulaciones/02_Antiguas_pruebas/` — corridas `11`–`27`: los seis
  Algoritmos del anexo, el Escenario B y la familia con tendencia (C–I), más la
  `20`. Sus generadores siguen en `pipelines/` (`sim_escenario_1`…`_6`, `_B`,
  `_T`) y son reutilizables; los notebooks no son plantilla.

---

## 8. Contradicciones abiertas

Señaladas, no resueltas. **Preguntar antes de codificar contra ellas.**

1. **Las corridas `31`–`34` y la de datos reales siguen con el conjunto viejo de
   métricas.** La alineación con `§02_02_03` se hizo sólo en la corrida `30`:
   falta propagar a los otros cuatro `_01` el bloque `metricas_bloque_A/B` del
   `eval_config`, y a sus `_04`/`_05` las listas `METRICAS_A`/`METRICAS_B`. Sus
   `eval_config.json` ya escritos todavía declaran `metrics_scores`,
   `metrics_curvas` y `metrics_dist` (`CRPS`, `energy_score`, `PIT`), que ningún
   notebook consume; se regeneran al re-correr el `_01`.
2. **Escenario J: el módulo pide `N_LAGS = 4` y el notebook usa `1`.** El
   docstring de `sim_escenario_J.py` advierte que con `N_LAGS = 1` el diseño AR
   no contiene las columnas de los rezagos 2–4 y "el escenario queda roto en
   silencio". `33_01` usa `N_LAGS = 1` **deliberadamente** —para que todos los
   escenarios entrenen con la misma máquina— y lo registra como
   `n_lags_estudio: 1`. La decisión está tomada; lo que falta es que el
   docstring deje de contradecirla.
3. **Las corridas `31` y `32` definen su generador dentro del notebook**
   (`nucleo_cuadratico`, `mapa_par`, `trayectoria`, `calibrar`) en vez de en
   `pipelines/sim_escenario_*.py`, partiendo de `ConfigEscenario1`. Es una
   desviación del patrón: el generador no es reutilizable ni testeable fuera del
   notebook.
4. **Las corridas `33` y `34` agregan secciones** que la `30` no tiene: §9.1 en
   el `_04` (error y cobertura por cuartil del rezago 1 no lineal) y §10 en el
   `_05` (techo del oráculo, `fraccion_brecha_recuperada`). Son específicas del
   escenario y están justificadas, pero **son agregados** respecto de la
   plantilla de §4. Si la regla "sin agregados" es estricta, hay que decidir si
   suben a la plantilla o bajan a un apéndice.
5. **`docs/03 Modelo.tex` referencia `\ref{03_06_04_objetivos_evaluacion}`
   (línea 371) y ese label no existe.** Las subsecciones de §03_06 sobre modelos
   de referencia, métricas y objetivo de evaluación están sin escribir.
6. **`§03_07 Resultados` es un bloque `% [PENDIENTE]`** con la organización
   propuesta en seis puntos.
7. Los criterios de evaluación se citan como `§02_02_03` del Capítulo 2.
8. **El Cuadro `tab:escenarios` de `docs/` no describe las corridas vivas**
   (`L = 48`, `sigma_eps = 0.1`, `T = 300`, `T0 = 240`, `R = 50` contra
   `L = 100`, `0.25`, `800`, `560`, `R = 1`). La divergencia es deliberada
   —`docs/` es referencia, no contrato— pero el cuadro sigue sin actualizarse.
9. **`R = 50` réplicas no está implementado.** `REPLICA_ID` existe y viaja en el
   `EXPERIMENT_ID`, pero siempre vale 1. Con `R = 1` una diferencia de pocos
   puntos porcentuales entre modelos es indistinguible del ruido Monte Carlo:
   **ninguna afirmación comparativa es concluyente hoy**, y así hay que
   presentarla.
