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

- **La semilla registrada no reproduce la que MATLAB usó, salvo en las corridas 11, 12 y 13.** Los `psbp_fd_iteracion.m` de `03`–`10` tienen `SEED_BASE = 4123` hardcodeado y **no** leen `seed_base` del JSON, mientras sus `hyperparameters.json` registran `41232`; los seeds reales por job son `4123 + chain*9973 + k*31` y ningún resultado de esas corridas es reproducible desde sus artefactos. **los `psbp_fd_iteracion.m` de `11_sim_E1/`, `12_sim_E2/` y `13_sim_E3/` lo corrigen**: lee `hp_json.seed_base` y falla si falta. Propagar esa corrección al resto al regenerarlos.
- **`M` significa dos cosas.** En FPCA es el número de componentes retenidas; en `mcmc_config` es el tamaño de la grilla de localización G\* del stick-breaking (`N` ahí es el truncamiento del número de átomos).
- **Índices base-0 vs base-1.** `component_idx` del manifest es base-0; los nombres de archivo usan `fpc_idx = component_idx[k] + 1`. Las trazas se llaman `chain_fpc_<fpc_idx>_iter<chain a 2 dígitos>.mat` (p. ej. `chain_fpc_2_iter03.mat`) — no cambiar, el flujo de resultados los busca por nombre.
- **Cuadratura y Cholesky tienen una sola definición.** `utils.quadrature.pesos_trapezoidales` y `utils.linalg.safe_chol`; `pipelines.sim_comun` las re-exporta (`factor_cholesky`) sin redefinirlas. Duplicarlas degrada en silencio la ortonormalidad FPCA o cambia las trayectorias ante la misma semilla.
- **`VAR` y `FAR1` del `_05` son el mismo estimador salvo el intercepto**, aunque conceptualmente no lo sean. `ajustar_var` es MCO **con** intercepto; `ajustar_far1` calcula `Psi = C1 C0^-1` con `C0 = A'A/n` y `C1 = B'A/n`, que son momentos **sin centrar** —no covarianzas—, y eso es MCO **sin** intercepto. Como los scores están centrados sobre el bloque de entrenamiento, ese intercepto vale ~0 y MCO es equivariante bajo cambios afines de escala, de modo que la estandarización tampoco los separa. Medido en la corrida 20: `corr(ajuste) = 1.00000000` con `M=1` y `0.99999486` con `M=2`; el MISE de prueba difiere en ~1e-5 relativo, cuatro órdenes por debajo del ruido Monte Carlo con `R=1`. Por eso **el `FAR1` se excluye de las figuras del `_05`** —dibujaba una línea encima del `VAR`— pero se conserva en tablas y CSV, y su `||Psi_hat||_HS` se sigue reportando, que es la cifra genuinamente propia suya (la que se contrasta con el 0.70 del generador). La exclusión **no está hardcodeada**: el notebook mide la separación en cada corrida y la revierte sola si supera `TOL_VAR_FAR = 2 %` de la sd del ajuste, que es lo que pasaría con `RIDGE_FAR > 0`, con otro centrado del FPCA o en un escenario con salto de nivel. Consecuencia para el capítulo: hoy **hay un solo competidor lineal**, no dos, y el promedio "sobre los competidores" no debe contarlo dos veces.
- **MATLAB colapsa la dimensión singleton final al guardar.** `zeros(nsim, N, p)` con `p = 1` se escribe en el `.mat` como `nsim x N`, de modo que `betajhout`, `psijhout`, `Gammajhout` y `gammajhout` llegan a Python con dos ejes en vez de tres. Ocurre exactamente cuando el diseño tiene una sola covariable, es decir en el punto **`M = 1`** del barrido, y hacía fallar a `PSBPPredictor` con `IndexError: tuple index out of range` y a `pip_por_componente` **en silencio**, devolviendo `(N,)` en vez de `(N, 1)`. `utils/trazas.py` (`normalizar_trazas_mat` / `asegurar_3d`) restaura el eje; `PSBPPredictor.__init__` y `fit/inclusion.py` lo aplican al leer. La reconstrucción es inequívoca porque `beta0hout` es `(nsim, N)` por definición y nunca pierde ejes. **No leer `traces["betajhout"].shape[2]` a mano** en un notebook: usar `predictor.n_features_`.
- **Con `M = 1` hay dos trampas más de forma**, ambas en el `_05` y ambas anteriores al barrido: `np.cov(SCORES.T)` devuelve un escalar 0-d y rompe `np.linalg.cond` (se envuelve en `np.atleast_2d`), y `RandomForestRegressor` ajustado con un objetivo `(n, 1)` lo aplana y `predict` devuelve `(n,)` en vez de `(n, 1)` (se hace `reshape` a la forma del contrato). Ninguna de las dos aparece con `M >= 2`.
- **`np.loadtxt` colapsa a 1D** con una sola columna; por eso `artifacts.py` fuerza 2D en `_MATRICES_2D`. Deja de ser hipotético con una sola componente FPCA retenida.
- **No usar `crps_gaussiano` / `lps_gaussiano` / `pit_gaussiano` con la predictiva del modelo propuesto**: son la aproximación de dos momentos y descartan la multimodalidad/asimetría que el estudio existe para medir. Usar las versiones muestrales.
- **El error de representación no es opcional** en las bandas funcionales: `PropagadorFuncional(modo_residuo=...)` con `"empirico"` es lo recomendado; `"ninguno"` solo si se evalúa la curva *proyectada*, y debe declararse al reportar.
- **`base_en_grilla` obtiene cada base por diferencia** (`reconstruct(e_k) - reconstruct(0)`) porque con `center=True` la reconstrucción es afín y contaminaría la base con la media funcional.
- **`pipelines/real_pipeline.py` es un stub `TODO`.** El flujo de datos reales vive solo en `notebooks/reales/`; desde la corrida 21 usa la misma arquitectura y los mismos artefactos que las simulaciones.
- **`versioning/changelog.md` está obsoleto**: describe un Gibbs de 8 pasos con prior MNIW y un directorio `configs/` que no existen en el diseño actual (stick-breaking probit, muestreador en MATLAB). `versioning/experiment_registry.md` está vacío.
- Los comentarios `% [FIX]` / `% [FIX N]` en los `.m` documentan correcciones deliberadas contra la implementación original (grid G\* sobre `Xnoint` sin el intercepto, `randi` que ignoraba parte del rango, factores `exp(1.2*·)` eliminados). No revertirlos sin entender el motivo.

## Parámetros fijos del estudio

**Decididos y cerrados a partir de la corrida 11.** No se cambian entre escenarios: solo así las comparaciones entre los Algoritmos 1–6 son entre generadores y no entre diseños de observación. Divergen del Cuadro `tab:escenarios` y esa divergencia es deliberada (ver la tabla de §Divergencias).

| Cantidad | Valor | Dónde |
|---|---|---|
| `L` (grilla) | **75** | `ConfigEscenario*.L`, constante `L_GRILLA` del `_01` |
| `T` (curvas) | **400** | constante `T_CURVAS` |
| `PROP_TRAIN` | **0.70** ⇒ `T0 = 280`, test = 120 | constante `PROP_TRAIN` |
| `sigma_obs` | **0.25** | `SIGMA_OBS` |
| `mu(tau)` | **`sin(2 pi tau)`** | `media_senoidal` |
| `burn_in` | 200 | `ConfigEscenario*.burn_in` |
| **`M` (componentes FPCA)** | **regla: primera `M` con varianza acumulada >= 95 %** | `fpca.seleccionar_M(0.95)`; `M_FPCA` es eje de barrido |
| base B-spline | **elegida por GCV en cada escenario** | `NB_ELEGIDO` / `ORD_ELEGIDO` del `_01` |

**`M` dejó de ser una constante y pasó a ser un eje.** La decisión vigente (2026-08-30) es que `M` sale de la **regla del 95 %** de varianza acumulada, y que alrededor de ese valor se corren puntos por encima y por debajo para medir la sensibilidad que pide `§03_06`. Consecuencias: `M` **no** es invariante entre escenarios —depende del espectro de cada uno— y por tanto las diferencias entre escenarios incluyen diferencias de `M`; hay que declararlo al reportar.

**La base B-spline tampoco es invariante**, y esto no estaba documentado. La elige el GCV en cada `_01` y difiere mucho entre corridas:

| | `(n_basis, order)` | `K` disponible | `M` corrido | `M(95 %)` |
|---|---|---|---|---|
| E1 (11) | (6, 4) | 6 | 2 | 2 |
| E2 (12) | (4, 3) | **4** | 2 | 2 |
| E3 (13) | (4, 3) | **4** | 3 | 2 |
| E4 (14) | (8, 4) | 8 | 3 | 3 |
| E5 (15) | (12, 3) | 12 | 5 | 5 |
| E6 (16) | (12, 3) | 12 | 5 | 5 |

`K` **acota el barrido en `M`**: en E2 y E3 hay sólo 4 componentes en total, y la 3ª y 4ª son en buena medida artefacto de una base de 4 funciones. Un barrido en esos escenarios no mide lo mismo que en E5/E6. Se evaluó fijar la base en `(12, 3)` para los seis y **se decidió mantener el GCV por escenario**; la incomparabilidad queda declarada, no resuelta. La única corrida que diverge de la regla del 95 % es la 13 (M=3 contra 2).

**Ojo con la corrida 11 ya ejecutada.** Sus artefactos (`data/simulaciones/raw/escenario_1_r01/simulation_config.json`) registran `sigma_obs = 0.25` pero `mu(tau) = 5 + 2 sin(2 pi tau)`. La decisión vigente, tomada al construir la corrida 12, es `mu(tau) = sin(2 pi tau)` —el Cuadro `tab:ane_esquema` del anexo— con `sigma_obs = 0.25`, que **no** es el 0.5 que ese mismo cuadro declara. Consecuencias: (a) hay que corregir `sigma_epsilon` a 0.25 en `docs/01 Anexo.tex`; (b) para que la comparación 1 vs 2 sea limpia conviene regenerar la corrida 11 con la media del anexo. El efecto práctico de la media es menor —el FPCA centra con la media empírica— pero la divergencia no debe quedar sin declarar.

Se evaluó subir `T` a 500 y **se descartó**: la resolución de la ventana móvil la da el deslizamiento (con test = 120 y `w = 20` hay 101 posiciones), no el tamaño del test, mientras que `T = 500` sube `n_train` de 280 a 350 y con ello ~25 % el tiempo de MCMC de cada job. Si en algún momento hace falta más test, sale más barato bajar `PROP_TRAIN` que crecer `T`.

## El error se mide contra la curva verdadera

**Decisión de la corrida 11, y cambia el significado de toda la evaluación.** `SalidaSimulacion` distingue `observaciones` (con ruido `sigma_obs`) de `curvas` (la curva verdadera `X_t(tau)`). Hasta la corrida 10 solo se persistía la primera y todas las métricas funcionales — MISE, RMSE funcional, cobertura, energy score — se calculaban **contra los datos ruidosos**, atribuyéndole `sigma_eps` al modelo.

- `guardar_curvas(paths, X, grilla, X_true=...)` persiste las dos por separado: `X_curves.npy` (observada) y `X_curves_true.npy` (verdadera). `cargar_curvas_true()` la lee y **falla si no existe**.
- La observada es el **único** objeto que alimenta la estimación: base, FPCA, estandarizador, scores. La verdadera entra **solo** como objetivo de evaluación.
- Se combina con `modo_residuo="ninguno"`: la banda cubre la curva **proyectada** sobre las `M` autofunciones y se contrasta con `X_t(tau)`, de modo que lo que queda fuera es truncamiento FPCA puro, cantidad interpretable y acotable subiendo `M`. Declararlo al reportar.
- **Las cifras de la corrida 11 no son comparables con las de 05–10**, que incorporaban el residuo empírico y evaluaban contra los datos.

`objetivo_evaluacion` y `modo_residuo` se registran en `eval_config.json` desde el notebook `_01`, no se deciden en el de evaluación.

## Convenciones

- **Ejes del estudio**, nombrados sin abreviar tras el `[FIX 12]`: `ESCENARIO_ID` (Algoritmo *k* del anexo), `REPLICA_ID` (réplica Monte Carlo), `chain` (cadena MCMC), `k` (componente FPCA). Los notebooks/`.m` de `05_sim_E1` a `08_sim_E4` y las corridas `11_sim_E1`, `12_sim_E2` y `13_sim_E3` usan esta convención; los archivados en `notebooks/simulaciones/01_Inicio_Formato/` (`03_Modelo`, `04_sim_E1_version_preliminar`, `09_sim_E5`, `10_sim_E6`) y `notebooks/reales/*` siguen con el `tt` antiguo.
- `EXPERIMENT_ID` nombra por igual `data/`, `artefact/` y `reports/`, y debe coincidir **exactamente** entre el notebook `_01`, el `.m` y los de evaluación. Hay dos convenciones vivas:
  - `03`–`10`: `f"{BASENAME}_{ESCENARIO_ID}"` (p. ej. `escenario_3`).
  - `11`–`17`: `f"{BASENAME}_{ESCENARIO_ID}_r{REPLICA_ID:02d}"` (p. ej. `escenario_1_r01`). Incluye la réplica para que el barrido Monte Carlo de la Etapa D no obligue a cambiar la convención. `psbp_fd_iteracion.m` llama `config_paths(EXPERIMENT_ID)` con el id ya construido en vez de `config_paths(basename, tt, seed)`.
  - **`20` en adelante: `f"{BASENAME}_{ESCENARIO_ID}_r{REPLICA_ID:02d}_m{M_FPCA:02d}"`** (p. ej. `escenario_1_r01_m02`). `M` viaja en el id para que cada punto del barrido escriba sus propios datos, trazas y reportes sin pisar los demás. Esto obliga a declarar `M_FPCA` **antes** de construir `PATHS`: en el `_01` subió de §3.4 a la celda `[CONFIG]` de §1.1, y §3.4 pasó a consumirlo y contrastarlo contra `M_SUGERIDO`. `config_paths.m` no cambia (ya recibía el id armado); `psbp_fd_iteracion.m` sí.
  - **`M` es un conjunto, no un escalar, en todo salvo el `_01`.** Los notebooks `_03`, `_04`, `_05` y `psbp_fd_iteracion.m` declaran **`M_FPCA_LIST`** (p. ej. `(1, 2, 3)`) y recorren el barrido completo en una sola pasada: cada punto se carga en un dict `EST[M]` y las celdas iteran sobre él, de modo que las figuras y tablas por `M` siguen escribiéndose en el directorio de ese `M` con los nombres de siempre. El `.m` arma **una sola lista plana de jobs** `(M × cadena × componente)` y la reparte con un único `parfor` —no un `parfor` por `M`— porque el último punto tiene más jobs que el primero y con tandas separadas los workers quedan ociosos al final de cada una. `_01` sigue siendo escalar (`M_FPCA`) y se corre una vez por punto: es el único que produce datos, y hacerlo iterar destruiría su narrativa exploratoria.
  - Los tres notebooks **saltan con aviso** los puntos sin artefactos o sin trazas (`SALTAR_M_SIN_TRAZAS`), para poder mirar el barrido mientras MATLAB va terminando. Y **exigen que los puntos compartan diseño**: `T`, `T0`, `n_lags`, `nivel`, `modo_residuo`, `objetivo_evaluacion`, ventanas y `mcmc_config` idénticos, y las mismas curvas verdaderas. Si no, la comparación entre `M` mezclaría el efecto de `M` con el de otra cosa, y el notebook falla en vez de callar.
- Numeración de notebooks. El paso `_02` es siempre MATLAB y no tiene notebook:
  - `03`–`10`: `NN_01_simulaciones` y `NN_03_resultados` (evaluación completa en un solo notebook).
  - **`11` en adelante: `NN_01_simulaciones`, `NN_03_convergencia`, `NN_04_evaluacion`, `NN_05_comparacion`.** La evaluación se partió porque son preguntas distintas y con distinta condición de parada: si las cadenas no convergen, los números de `_04` no significan nada. `_03` no toca el bloque de prueba. **`_05` es donde viven los modelos de referencia** (VAR sobre scores, FAR(1), Random Forest, Gradient Boosting) y escribe los archivos `70`–`75`; los `_04` ocupan `50`–`65`. La corrida 17 no tiene `_05`.
- **Salidas del barrido en `M`.** Lo que cruza los puntos no cabe en el directorio de ninguno de ellos y va a `reports/simulaciones/<BASENAME>_<ESCENARIO>_r<NN>_barrido_M/`, hermano de los de cada `M`. Prefijos: `80`–`83` del `_03` (convergencia, ocupación y PIP contra `M`), `85`–`89` del `_04` (métricas, error, calibración y ventana móvil contra `M`), `90`–`93` del `_05` (todos los modelos contra `M`). Los `50`–`75` de cada punto no cambian de nombre ni de sitio.
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
| 5 | Predictibilidad en componente subordinada | `pipelines/sim_escenario_5.py` | Implementado, **descartado del estudio**: pasa a cota analítica en limitaciones (ver Etapa B) |
| 6 | Covarianza no estacionaria | `pipelines/sim_escenario_6.py` | Implementado con los parámetros del anexo como default |
| **B** | **FAR con signo conmutado por umbral** | `pipelines/sim_escenario_B.py` | Implementado; **no es del anexo**, es de diagnóstico (ver §El Escenario B) |
| **C–I** | **Tendencia (lineal, cuadrática, logarítmica, sinusoidal, por tramos, volátil) × no linealidad (interacción intra-curva / mezcla de J regímenes)** | `pipelines/sim_escenario_T.py` | Implementado; **no son del anexo**, son de diagnóstico. Un módulo, siete corridas (ver §La familia con tendencia) |

**Los Algoritmos 5 y 6 cambiaron de definición.** La nota antigua de este archivo ("Algoritmo 5 no implementado por su estacionariedad bajo skew-normal") describía un diseño anterior y ya no aplica: en el anexo vigente el 5 es *predictibilidad en componente subordinada* y el 6 es *covarianza no estacionaria*. Ambos son del Bloque 2 y **no reutilizan la maquinaria de `sim_comun`** para la dinámica — no hay operador integral, ni Cholesky de innovación funcional, ni norma HS. Comparten en cambio el esquema de observación, las réplicas y las semillas, de modo que lo que sí se reutiliza de `sim_comun` es `grilla_regular`, `evaluar_media`, `semillas_replicas`, `aplicar_ruido_observacion`, `diagnostico_comun`, `ConfigObservacion` y `SalidaSimulacion`.

### Divergencias de parámetros entre código y anexo

**`docs/01 Anexo.tex` fue reescrito** (cambios sin commitear al 2026-08-25): ya no remite al Capítulo 3 para los valores numéricos, sino que trae un cuadro por algoritmo —`tab:ane_esquema` para el esquema de observación y `tab:ane_alg1`…`tab:ane_alg6`— y esos cuadros **ya recogen los defaults del código**. La tabla de divergencias quedó por tanto casi vacía; lo que sigue es lo que aún no coincide.

| Cantidad | `docs/01 Anexo.tex` | Estudio | Estado |
|---|---|---|---|
| `sigma_epsilon` | 0.5 (`tab:ane_esquema`) | **0.25** | decidido; **corregir el cuadro del anexo** |
| `mu(tau)` | `sin(2*pi*tau)` | `sin(2*pi*tau)` desde la corrida 12; la 11 corrió con `5 + 2 sin(2*pi*tau)` | regenerar la 11 |
| `R` | 50 | 1 por corrida, `REPLICA_ID` en el `EXPERIMENT_ID` | pendiente (Etapa D) |
| `q` (orden markoviano) | no aparece | `n_lags = 1` en las corridas 11–12 | declarar en la tesis |
| Alg. 3 `||Psi^(2)||_HS` | 0.80 (`tab:ane_alg3`) | **0.80**, pasado explícito en el `_01` de la corrida 13 | decidido; el default del código sigue en 0.85 |
| Alg. 4 `||Psi||_HS` | 0.7 (`tab:ane_alg4`) | **0.70**, pasado explícito en el `_01` de la corrida 14 | decidido; el default del código sigue en 0.50. Es el mismo valor del Algoritmo 1, condición para que la comparación 1 vs 4 aísle la asimetría |
| Alg. 6 `t*` | 270 (`tab:ane_alg6`) | **280 = T0**, pasado explícito en el `_01` de la corrida 16 | decidido; **corregir el cuadro del anexo**. El 270 está dimensionado para `T=300`/`T0=240`; con `T=400`/`T0=280` hay que reubicarlo o el quiebre caería dentro del entrenamiento |
| `M` (componentes FPCA) | no aparece | **2** (11, 12), **3** (13, 14), **5** (15, 16) | ya no es invariante: sale de la **regla del 95 %** y es eje de barrido. La 13 es la única que diverge de su propia regla (M=3 contra 2) |

`L = 75`, `T = 400`, `T0 = 280`, `burn_in = 200`, y los parámetros de los Algoritmos 1 y 2 (`gamma = 0.3`, `||Psi||_HS = 0.7`; `persistencia = 0.85`, `prop_arch = 0.25`, `var_objetivo = 1.0`, `alcance_beta = 0.15`, `alcance_gamma = 0.30`) **ya coinciden** entre código y anexo.

**`ell` es una trampa activa.** Los cuadros del anexo piden `ell = 0.5` en los Algoritmos 1, 2 y 3 y `ell = 0.2` en el 4; los **defaults de las dataclasses siguen en `0.2` para los cuatro**. Las corridas 11–13 pasan `0.5` explícito y la 14 pasa `0.2`, de modo que hoy código y anexo coinciden *sólo porque el notebook lo pasa a mano*: un `_01` que omita `ell` obtiene un escenario distinto del anexo sin que nada falle. Y la divergencia 0.5 vs 0.2 entre los Algoritmos 1 y 4 **rompe la comparación que el Algoritmo 4 existe para hacer**: `ell` fija la suavidad, luego el decaimiento del espectro FPCA, luego cuánto retiene la regla del 95 % (M=2 en E1 contra M=3 en E4). Está decidido corregir el Algoritmo 4 a `ell = 0.5`.

**El anexo reescrito ya incorpora `desplazamientos`** (`d_1 = -1.5`, `d_2 = +1.5`, forma `f` constante) y argumenta por qué: sin ellos los dos regímenes tendrían igual media condicional, la mezcla sería unimodal y el escenario no distinguiría un modelo capaz de representar mezclas de uno que no lo es. La versión "extendida" del código **es** ahora la del anexo, y la duda que figuraba aquí queda cerrada. Los defaults de `ConfigEscenario3` coinciden con `tab:ane_alg3` en todo salvo `hs_norms[1]` (0.85 en el código, 0.80 en el cuadro).

## Inventario de infraestructura de evaluación

Existe y sirve tal cual:

- **Ciclo Python → MATLAB → Python** completo y probado (ver arriba), con `artifacts.py` como capa de E/S y `verificar_contrato()`.
- **Predictiva funcional** (`models/pspb_fd_v3`): `PSBPPredictor` → `muestrear_scores` → `PropagadorFuncional` → muestras `(S, n, G)`.
- **Métricas**: `fit/metrics_puntual` (RMSE, MSE/R² por coeficiente, MISE, razón de dispersión) y `fit/metrics_distribucional` (CRPS muestral, energy score, cobertura, PIT muestral, LPS). Cubren el eje 1 y la parte marginal del eje 2.
- **Diagnóstico MCMC**: `fit/diagnostics_mcmc.py` — `ess_geyer`, `geweke_z`, `gelman_rubin`, `tabla_diagnosticos`, `resumen_convergencia`. `graphics/viz_traces.py` los **importa** en vez de redefinirlos, de modo que hay una sola definición. Los diagnósticos se calculan sobre cantidades invariantes a la permutación de etiquetas de la mezcla (`extraer_traza_variable` promedia sobre átomos): con un `h` fijo, R-hat mide el etiquetado y no la convergencia.
- **Ventana móvil**: `fit/rolling.py` — `ventana_movil_scores` y `ventana_movil_funcional`. No reentrena: desliza la ventana de *evaluación* sobre una serie de predicciones a `h=1` con rezagos reales, etiqueta cada ventana por bloque y marca las que cruzan `T0`. `w` es configurable y el notebook superpone varios anchos.
- **Probabilidades de inclusión**: `fit/inclusion.py` — `matriz_pip` (PIP global desde `osumout`, media y sd entre cadenas) y `contraste_con_verdad` (VP/FP/FN/VN, sensibilidad, AUC contra la estructura declarada del generador). Distingue la PIP **global** de la **promedio por átomo**, que no son la misma cantidad.
- **Figuras de evaluación**: `graphics/viz_evaluacion.py` — `plot_ventana_movil` (train y test en el mismo eje, corte marcado, varios `w`), `plot_bandas_serie` (bandas por score en ambos bloques), `plot_extractos_curvas` (intervalos sobre la curva, un extracto cada `cada` períodos) y `plot_calibracion_pit` (PIT train vs test).
- **Trazas de selección de variables**: `psbp_train.m` **sí guarda** `gammajhout (nsim, N, p)`, `pijout`, `wjout`, `osumout`, y `cargar_trazas_mat` los lee. La materia prima del eje 3 ya está en disco.

Falta para poder cerrar el capítulo:

1. **Orquestación Monte Carlo (`R = 50`)**. Hoy todo el flujo — notebooks, `EXPERIMENT_ID`, `hyperparameters.json`, nombres `.mat` — está construido para **una** réplica; `REPLICA_ID` existe como campo pero siempre vale 1. Es el cambio estructural más grande: 6 escenarios × 50 réplicas × cadenas × componentes no cabe en el patrón "un notebook por experimento".
2. **Barrido en `M`**. No hay ningún mecanismo para ajustar y evaluar el mismo escenario con varios `M`; `M` se fija una vez en el notebook `_01` y se propaga al manifest.
3. **Predictiva para los modelos de referencia.** Los modelos ya existen —`NN_05_comparacion.ipynb` ajusta VAR sobre scores, FAR(1), Random Forest y Gradient Boosting, y escribe `70_comparacion_modelos.csv`, `74_tabla_final.csv` y `75_resumen_comparacion.csv`— pero **reportan solo RMSE, R² y MISE**. `fit/baselines.py` sigue teniendo únicamente `prediccion_media_incondicional` y `prediccion_persistencia`; el resto vive en el notebook. Sin una predictiva para las referencias no hay CRPS, energy, PIT ni cobertura comparables, y **ese es el cuello de botella del capítulo**: en el único eje comparable el PSBPM-FD queda igual o levemente peor que un FAR(1) lineal (E1: MISE test 1.0908 contra 1.0549; E2: 1.0143 contra 1.0347, con la media incondicional en 1.0093). Es una métrica de media condicional aplicada a un método diseñado para ganar en forma de la predictiva. Ningún ajuste de generador lo arregla.
4. ~~**Calibración condicional estratificada por el estado del generador**~~ — hecho en la corrida 12: `fit/metrics_distribucional.py` aporta `estratos_por_cuantil` y `cobertura_condicional` (cobertura, ancho medio, `desvio` respecto de la marginal y el puntaje propio del estrato —CRPS para un score, energía para la curva—), y `12_04 §9` los consume con `sigma2` como estado verdadero. Para el Alg. 3 el estrato es el régimen y se pasa directo, sin cuantilizar. En el Escenario 1 no rinde: no hay estado que estratificar.
5. ~~**Error sobre ventana móvil**~~ — hecho: `fit/rolling.py`.
6. ~~**Probabilidades posteriores de inclusión como cantidad reportable**~~ — hecho: `fit/inclusion.py`.
7. **Agregación entre réplicas** (media, error estándar Monte Carlo por escenario/métrica/`M`) y las tablas del capítulo.
8. ~~**Escenarios 5 y 6** completos, incluidos sus diagnósticos y sus notebooks `_01`~~ — hecho (corridas 15 y 16). El Escenario 5 quedó después descartado del estudio.

## Zonas grises que `docs/` todavía no resuelve

No inventarlas: preguntar antes de codificar contra ellas.

- **`03 Modelo.tex` referencia `\ref{03_06_04_objetivos_evaluacion}` (línea 374) y ese label no existe.** Las subsecciones de §03_06 sobre modelos de referencia, métricas y objetivo de evaluación (curva proyectada vs. curva suavizada) están pendientes de escribir. De ahí sale el punto 3 de la lista anterior.
- **`§03_07 Resultados` es un bloque `% [PENDIENTE]`** con la organización propuesta en seis puntos. Esa lista es la mejor guía disponible de qué debe producir el código.
- Los criterios de evaluación se citan como `§02_02_03` del Capítulo 2, que **no está en `docs/`**.
- `q` (orden markoviano) no aparece en los cuadros del anexo; las corridas 03–10 usaban `n_lags = 2` y las 11–13 usan `n_lags = 1`. Hay que declararlo en la tesis.

## Plan de trabajo

Orden pensado para que cada etapa sea utilizable antes de empezar la siguiente.

**Etapa A — alinear lo que ya existe con `docs/`.** El esquema de observación ya está decidido (§Parámetros fijos del estudio) y difiere del Cuadro `tab:escenarios` a propósito. Queda decidir el caso del `desplazamientos` del Alg. 3 y del `familia`/`delta` del Alg. 4, y dejar los valores del anexo como default de la dataclass en vez de como argumento del notebook — el notebook debería poder no pasar nada y obtener el escenario del anexo.

**Etapa B — Escenarios 5 y 6. HECHA.** `sim_escenario_5.py` y `sim_escenario_6.py` implementan la generación sobre coeficientes + base de Fourier `J = 10` (`base_fourier`, ortonormal en L² continuo; la cuadratura trapezoidal sobre grilla regular la reproduce con error de máquina si `2*k_max < L-1`, condición que `validar()` verifica). Los defaults de ambas dataclasses son los del Cuadro `tab:escenarios` — `ConfigEscenario5()` / `ConfigEscenario6()` sin argumentos son el escenario del anexo, incluida `media_fn=media_senoidal` y `sigma_obs=0.1`. Ambos exportan por `pipelines/__init__.py`. **Cuidado con esos defaults**: son los del Cuadro del Capítulo 3 (`L=48`, `T=300`, `sigma_obs=0.1`, `R=50`, `t_quiebre=270`) y **no** los parámetros fijos del estudio, así que los `_01` de las corridas 15 y 16 pasan los seis campos del esquema de observación explícitos y lo verifican con un assert; omitir uno rompería la comparabilidad sin dar error. Nota de diagnóstico: `resumen_escenario_6` compara varianzas en la ventana posterior a `t*`, y con el default de 270 esa ventana tiene 30 períodos, con lo que `reordenamiento_detectado` puede dar False con `R=1` sin que el generador falle. Con el `t*=280` de la corrida 16 (y con el 340 decidido) el problema desaparece.

**Los notebooks `_01` de ambos ya existen** (corridas 15 y 16); el punto 8 de la lista de pendientes está cerrado. El Escenario 5 quedó luego **descartado del estudio** por la cota `lambda_j * phi_j^2` (ver la corrida 15).

**Etapa C — evaluación. PARCIALMENTE HECHA.** Listos y verificados sobre trazas reales: `fit/rolling.py` (ventana móvil), `fit/inclusion.py` (PIP y contraste con la verdad), `fit/diagnostics_mcmc.py` (ESS/Geweke/R-hat separados del dibujo) y `graphics/viz_evaluacion.py`. También hechos: los modelos de referencia (en `NN_05_comparacion.ipynb`, no en `fit/baselines.py`) y la cobertura condicional estratificada. **Queda pendiente lo único que bloquea el capítulo**: dotar a esas referencias de una predictiva para poder darles CRPS, energy, PIT y cobertura.

### Qué corridas hay y cuáles son informativas

| Carpeta | Escenario | `EXPERIMENT_ID` | Estado |
|---|---|---|---|
| `11_sim_E1` | Alg. 1 | `escenario_1_r01` | corrida; usó `mu = 5 + 2 sin`, **superada por la 20** |
| `12_sim_E2` | Alg. 2 | `escenario_2_r01` | corrida |
| `13_sim_E3` | Alg. 3 | `escenario_3_r01` | corrida |
| `14_sim_E4` | Alg. 4 | `escenario_4_r01` | corrida |
| `15_sim_E5` | Alg. 5 | `escenario_5_r01` | corrida; escenario **descartado** |
| `16_sim_E6` | Alg. 6 | `escenario_6_r01` | corrida |
| `17_sim_EA` | **Escenario A: FAR(1) con tendencia** | `escenario_A_r01` | escenario de **diagnóstico**, no es un Algoritmo del anexo; sin `_05` |
| `18_sim_EB` | **Escenario B: FAR con signo conmutado** | `escenario_B_r01_m01`…`_m03` | escenario de **diagnóstico**, no es un Algoritmo del anexo; **construido para que el FAR muera**. Arquitectura de la 20 (barrido en `M`). **Sin correr** |
| `19_sim_EC` | **Escenario C: interacciones intra-curva + tendencia lineal** | `escenario_C_r01_m02`…`_m04` | familia con tendencia (`sim_escenario_T.py`). **Sin correr** |
| `22_sim_ED` | **Escenario D: interacciones + tendencia cuadrática** | `escenario_D_r01_m02`…`_m04` | ídem, eje tendencia. **Sin correr** |
| `23_sim_EE` | **Escenario E: multimodalidad + tendencia por régimen (lineal)** | `escenario_E_r01_m01`…`_m03` | ídem, eje no linealidad. **Sin correr** |
| `24_sim_EF` | **Escenario F: multimodalidad + tendencia por régimen (cuadrática)** | `escenario_F_r01_m01`…`_m03` | ídem, las dos. **Sin correr** |
| `25_sim_EG` | **Escenario G: 3 regímenes, una forma de tendencia por régimen** | `escenario_G_r01_m01`…`_m03` | familia con tendencia, segunda tanda. **Sin correr** |
| `26_sim_EH` | **Escenario H: tendencia sinusoidal volátil** | `escenario_H_r01_m01`…`_m03` | la pendiente misma es estocástica. **Sin correr** |
| `27_sim_EI` | **Escenario I: tendencia por tramos (cuad → log → sen)** | `escenario_I_r01_m01`…`_m03` | cambios de régimen de tendencia. **Sin correr** |
| `20_sim_E1` | Alg. 1 | `escenario_1_r01_m01`, `..._m02` | **plantilla vigente**: `mu = sin(2 pi tau)`, `M` en el id. Barrido en curso |
| `21_real_nivel` | **datos reales** | `real_nivel_v01_m03` | la arquitectura de la 20 sobre la serie observada; ver §Datos reales |

**Veredicto de informatividad (2026-08-30), y difiere del orden intuitivo.** El criterio no es "¿el generador rompe un supuesto?" sino "¿rompe uno que el modelo propuesto puede aprovechar y las referencias no?". Bajo ese criterio:

- **E2 es el mejor de los seis hoy** — el único con evidencia positiva y limpia (el modelo ensancha con la volatilidad). Amplitud corta, arreglable con `prop_arch`.
- **E3 debería ser el mejor** (multimodalidad es literalmente para lo que existe una mezcla) y hoy es el que falló más claramente. Pendiente el diagnóstico oracle que decide si es hallazgo o defecto.
- **E1 es informativo por su valor negativo**: cuantifica el precio de la flexibilidad bajo especificación correcta. Imprescindible como control.
- **E4** tiene contraste en el generador que el modelo no recupera; mismo patrón que E3.
- **E5 y E6 son Bloque 2 y no discriminan métodos**: su degradación alcanza por igual a PSBPM-FD, FAR(1), VAR y RF, porque la base congelada les falla a todos y no hay covariable rezagada que anticipe el cambio. Son **limitación, no resultado**. E6 conserva un aporte propio que ningún otro da: `e_t = ||X_t - Pi_M X_t||^2`, métrica independiente del modelo predictivo que separa "falla la representación" de "falla la predicción".

El peso del capítulo tiene que caer en **2, 3 y 4**, y —desde que existe— en el
**Escenario B de la corrida 18**, que es el único con brecha medible entre la
clase lineal y el óptimo (ver §El Escenario B).

**La corrida 20 (`notebooks/simulaciones/20_sim_E1/`) es la plantilla vigente**, copia de la 11 con dos cambios: `mu(tau) = sin(2 pi tau)` (la 11 corrió con `5 + 2 sin`) y `M_FPCA` como eje de barrido en el `EXPERIMENT_ID`. El resto de `SIM_CFG` es el del anexo y no se tocó. Para las corridas de E2, E3 y E4 con las constantes nuevas, partir de la 20 y no de la 11.

**Dato del barrido en curso, y no es menor:** con `mu = sin(2 pi tau)` el GCV eligió una base distinta de la de la 11 — `K = 4` contra los 6 de aquella— y la varianza acumulada cambió (`M=1` → 83.24 %, `M=2` → 97.69 %, contra 97.54 % de la 11). O sea que **la regla del 95 % sigue dando `M = 2`, pero sobre un espectro distinto**: cambiar la media cambió la base, y con ella el techo `K` del barrido. Con `K = 4` el punto `M = 3` de este escenario cae en el mismo terreno que E2/E3 —componentes que son en buena medida artefacto de una base corta— y hay que leerlo con esa advertencia, no como "una FPC más del proceso".

**La corrida 11 (`notebooks/simulaciones/11_sim_E1/`) es la plantilla de referencia** para regenerar los escenarios 2–6: parámetros fijos del estudio, evaluación contra la curva verdadera, `seed_base` leída del JSON, `EXPERIMENT_ID` con réplica y la evaluación partida en convergencia + predicción. Lo que cambia por escenario es el generador del `_01`, el `ESCENARIO_ID` y el diccionario `VERDAD` de `_03` §6.1, que declara qué covariables usó realmente el generador.

**La corrida 12 (`notebooks/simulaciones/12_sim_E2/`) es la instancia de esa plantilla para el Algoritmo 2**, y fija lo que hay que replicar en las siguientes:

- Sólo cambia el generador. `N_LAGS = 1`, priors, `mcmc_config` y `N_CHAINS` son idénticos a los de la 11 **a propósito**: cambiarlos convertiría la comparación entre escenarios en una comparación entre ajustes. La base y `M` **no** son idénticos —los elige el GCV y la regla del 95 % sobre cada espectro; la 11 corrió con (6,4)/M=2 y la 12 con (4,3)/M=2— y eso es una diferencia de ajuste que sí queda entre escenarios. El prior `E[pi] = 0.90` sobre el propio rezago se mantiene aunque en este escenario esté mal orientado —no hay dependencia en la media—, y que el modelo tenga que desmentirlo es parte de la prueba del eje 3.
- **El estado verdadero se persiste.** `_01` guarda con `incluir_internos=True` (deja `interno_sigma2` en el `.npz`) y además escribe `reports/.../10_estado_volatilidad.csv`, que es lo que `_04 §9` lee; `eval_config.json` gana un bloque `estratificacion` que declara variable, fuente, número de estratos y etiquetas. Replicar el patrón en el Alg. 3 con `regimenes`.
- **La lectura cambia con el escenario.** En el Algoritmo 2 la media condicional es constante, de modo que RMSE y R² **no discriminan** (R² ≈ 0 es el resultado correcto y la media incondicional es la predicción puntual óptima); lo que separa modelos es CRPS, energía y la cobertura estratificada. `VERDAD` de `_03 §6.1` es `{FPC k: []}` —ninguna covariable activa en la media— y por eso la cifra reportable es `pip_media_inactivas`, leída como cuánta señal recogen los rezagos por el canal de varianza, no como falsos positivos.
- Diagnóstico de la corrida ejecutada (`simulation_config.json`): `cv_temporal_sigma2 = 0.381`, `acf1_residuos_cuadrado = 0.184`, `radio_espectral_BG = 0.748`, `exceso_curtosis = +0.338`, `sigma2 in [0.58, 4.13]`. El generador tiene contraste real.
- **Resultado: es el escenario que mejor funciona hoy.** El modelo ensancha la banda con la volatilidad —`corr_ancho_sigma = 0.679`, `razon_ancho_alto_bajo = 1.13`, cobertura 0.935 (estrato bajo) contra 0.918 (alto)— que es exactamente lo que un FAR(1) homocedástico no puede hacer. La amplitud es corta y el techo lo pone `prop_arch`: con `n_lags = 1` el modelo sólo ve la volatilidad por el canal **ARCH** (β, que responde a `X_{t-1}` observable), no por el GARCH (γ, latente). Está decidido subir `prop_arch` de 0.25 a **0.35** manteniendo `persistencia = 0.85`; la condición escalar de cuarto momento `3a^2+2ab+b^2 < 1` da 0.90 con 0.35, 0.95 con 0.40 y **falla (1.02) con 0.45**.

**La corrida 13 (`notebooks/simulaciones/13_sim_E3/`) instancia el Algoritmo 3** y es el escenario informativo del estudio. Además de lo que ya fijaban la 11 y la 12:

- Parámetros del Cuadro `tab:ane_alg3`, con `hs_norms = (0.30, 0.80)` pasado explícito (el default del código es 0.85). `mecanismo = "probit"`: los pesos de la mezcla dependen del estado rezagado. Los otros dos mecanismos —`"markov"`, pesos independientes del estado, y `"quiebre"`, no estacionario— quedan fuera.
- **El estado verdadero es discreto.** `internos["regimenes"]` es `(R, T)` base-0 y se pasa **directo** a `cobertura_condicional`, sin `estratos_por_cuantil`. `eval_config.json` lo declara con `metodo = "estado discreto del generador"`.
- **`_01` reconstruye el mecanismo latente y lo persiste** en `reports/.../10_estado_regimen.csv`: régimen, `p_regimen1 = Phi(kappa*(c - z_{t-1}))`, `ambiguedad = 0.5 - |p - 0.5|`, `z_lag` y nivel de la curva. La proyección se calcula sobre `Y_t = X_t - mu - d_{R_t}` —restando el desplazamiento **del régimen vigente**, no sólo la media—; restar sólo `mu` da una `p_t` que no es la que se usó. Con los parámetros del anexo eso da `E[p] ≈ 0.378` contra una frecuencia empírica de `0.353` y ~56 orígenes ambiguos de 400, y `_01` verifica esa coincidencia con un assert.
- **La lectura vuelve a invertirse.** Aquí RMSE y R² **sí** discriminan (hay dependencia en la media por dos vías: el operador vigente y el desplazamiento de régimen), y `VERDAD` de `_03 §6.1` vuelve a ser **todas las covariables activas**, como en el Escenario 1: ambos operadores actúan sobre la curva completa y el régimen depende del nivel rezagado. Sensibilidad y AUC sí están definidas, al revés que en la 12.
- **`_04 §9.1` es la sección nueva**: en los orígenes con `p_t ≈ 1/2` la condicional es genuinamente bimodal y un modelo gaussiano sólo puede responder con una moda situada entre las dos. Se mide con el coeficiente de bimodalidad de Sarle `b = (g1^2 + 1)/g2` (referencia: uniforme 5/9, gaussiana 1/3), comparando orígenes ambiguos contra deterministas **del mismo modelo** — es una heurística descriptiva, no un test, y por eso la comparación interna importa más que el valor absoluto.
- **Resultado: el escenario que debía ser el más informativo no discriminó.** `sarle_ambiguos = 0.4534` contra `sarle_deterministas = 0.4372` — una diferencia de 0.016, ruido. `r2_scores_test = 0.160`, `amplitud_cobertura_regimenes = 0.033`. **No está establecido si falló el generador o el modelo**: con brecha `2d = 3.0` y sd condicional ≈ 1.1 la condicional en `p ≈ 0.5` es bimodal en la población, así que la predictiva del modelo debería serlo. El diagnóstico que separa las dos hipótesis —y que nadie corrió— es el **coeficiente de Sarle de la predictiva *oracle*** (pesos `p_t` y operadores verdaderos, proyectada sobre las mismas autofunciones). Es barato y decide si esto es un hallazgo negativo sobre PSBPM-FD o un defecto de diseño.
- Cambios decididos si el diagnóstico dice que es de diseño: `desplazamientos` de ±1.5 a **±2.0** y `umbrales` de 0 a un `c` calibrado numéricamente para proporciones ≈ 0.5/0.5 (hoy son `[0.355, 0.645]` porque el desplazamiento se retroalimenta: el régimen 2 sube el nivel y el nivel alto lo sostiene). Equilibrar maximiza los orígenes ambiguos, que son los únicos con contenido: hoy son ~56 de 400, ~17 en test. **No subir `nitidez`**: κ alto vuelve el régimen determinista y elimina justamente esos orígenes.
- **Cuidado: ese cambio choca con la regla del 95 %.** `λ1` ya se lleva el **94.2 %** de la varianza porque el desplazamiento es de rango uno sobre la dirección constante. Subir a ±2.0 lleva su contribución de `p(1-p)(2d)^2 = 2.06` a ≈ 3.66 y empuja `λ1` sobre el 95 %, con lo que **la regla devuelve M = 1** y el escenario de multimodalidad se queda sin eje 3. Equilibrar los regímenes lo empeora (`p(1-p)` sube de 0.229 a 0.25). La salida candidata —no verificada— es `forma_desplazamiento_fn` no constante, p. ej. `f(tau) = cos(2 pi tau)`: el desplazamiento sigue siendo de rango uno pero carga sobre una dirección casi ortogonal a la constante, infla `λ2` en vez de `λ1` y deja `M = 2`.
- Diagnóstico con los parámetros elegidos: `radio_espectral` por régimen `[0.27, 0.71]`, proporciones `[0.35, 0.65]`, `separacion_en_sd_puntual ≈ 1.47`, `duracion_media_racha ≈ 2.86`. Las cifras que deciden si el escenario tiene contenido son `separacion_en_sd_puntual` (si es baja, la mezcla es unimodal y el escenario no prueba nada) y `proporcion_regimen_minima` (si un régimen es raro, el test no tiene orígenes suficientes para estratificar).

**La corrida 14 (`notebooks/simulaciones/14_sim_E4/`) instancia el Algoritmo 4**, y su razón de ser es el **contraste con la corrida 11**: la dinámica es idéntica y sólo cambia la ley de la innovación.

- Parámetros del Cuadro `tab:ane_alg4` con **`hs_norm = 0.70` pasado explícito** (el default del código es 0.50). Es la decisión que hace limpio el contraste: con 0.50 la comparación 1 vs 4 mezclaría asimetría con persistencia. `gamma=0.30`, `sigma_eps=1.0`, `ell=0.2`, `familia="st"`, `delta_skew=0.85`, `nu=5`.
- **No hay estado latente que estratificar**, y por eso `eval_config.json` declara `estratificacion: null` **con su motivo**, en vez de omitir la clave: la asimetría es una propiedad constante de la ley de la innovación, no un régimen. `_04` omite la sección de calibración condicional (como la corrida 11) y `assert`ea que el artefacto declare `None`, para que el notebook falle en vez de callar si se le pasan datos de otro escenario. Lo que sí se persiste es `reports/.../10_momentos_curva.csv` (asimetría y curtosis puntuales de curva y residuo AR) y `10_serie_residuo.csv`; el bloque `forma_predictiva` de `eval_config` lleva las cifras del generador contra las que `_04 §9` contrasta.
- **`_01 §2.5` reconstruye el residuo AR** `X_t - mu - Psi(X_{t-1} - mu)` con el operador guardado en `internos` y verifica con un assert que su asimetría reproduzca `asimetria_innovacion_empirica`. Es el análogo de la reconstrucción de `p_t` en la corrida 13.
- **La lectura se invierte respecto de la 13.** RMSE y R² **no** discriminan —la media condicional es la del Algoritmo 1, correctamente especificada por cualquier método lineal— y que coincidan con los de la corrida 11 es la señal de que el diseño funciona. Lo que separa es CRPS, energía y la **forma del PIT**: una predictiva simétrica sobre datos asimétricos produce un PIT **inclinado**, patrón distinto de la U y de la campana. `VERDAD` de `_03 §6.1` es **todas las covariables activas**, como en el Escenario 1.
- **`_04 §9` es la sección propia**: asimetría y curtosis de los draws predictivos *por origen* contra las de los residuos realizados y contra `asimetria_scores_train`. Tres cifras comparadas entre sí, no contra un umbral. Se acompaña de la razón de semianchos de la banda y del desequilibrio de excesos por lado.
- **Resultado: el escenario no logró que se notara la asimetría.** `fraccion_asimetria_recuperada = 0.349` (predictiva −0.44 contra residuo realizado −1.26) y `pit_ks_test_max = 0.104` contra un crítico ≈ 0.124 con n=120: **el PIT no rechaza uniformidad**. Como en la corrida 13, el generador sí tiene contraste —el residuo a nivel de score tiene asimetría −1.26— y el nulo es del lado del modelo. Por componente: FPC1 −0.74, FPC2 −0.61, FPC3 +0.10, o sea que el sesgo llega a dos componentes y muere en la tercera.
- Cambios decididos: **`ell` de 0.2 a 0.5** (es el único parámetro dinámico que difiere de E1 y confunde asimetría con representabilidad; ver §Divergencias) y `delta_skew` de 0.85 a **0.95** (el PIT está al borde de rechazar). `nu` de 5 a 4 queda como opcional: aleja aún más E4 de E1 en momentos de cuarto orden, y `curt_residuo` ya es 11.3. **No bajar `hs_norm`**: recupera asimetría marginal pero rompe la comparación 1↔4, que es la razón de ser del escenario. `forma_skew_fn` no constante (el código ya lo soporta vía `delta_l = delta_skew * d(tau_l)`) es un arreglo de segundo orden y queda fuera de esta vuelta.
- Diagnóstico con los parámetros elegidos (R=1): `hs_norm_efectiva = 0.700000` exacta, `radio_espectral = 0.622`, `var_innovacion_teorica_error_max = 0.0` (calibración exacta), `asimetria_innovacion_empirica ≈ 1.726`, `curtosis_innovacion ≈ 15.88`, `asimetria_curvas_media ≈ 0.572`, `razon_asimetria_curva_innovacion ≈ 0.332`, `razon_senal_ruido ≈ 24.2`. **`correlacion_larga_distancia_extra ≈ 0.545`** es el precio de introducir asimetría con un choque común de rango uno: la varianza puntual queda igualada a la del Escenario 1 pero la correlación a larga distancia no. No es un defecto —es consecuencia necesaria de que `U_0` sea escalar— pero hay que declararlo al comparar el espectro FPCA con el de la corrida 11.

**La corrida 15 (`notebooks/simulaciones/15_sim_E5/`) instancia el Algoritmo 5** y es la primera del **Bloque 2**: no interviene la ley condicional sino la reducción de dimensión.

- Parámetros de `tab:ane_alg5`: `J=10`, `razon_espectro=0.5`, `indice_predecible=3` (base-1), `phi_predecible=0.9`. **Los seis campos del esquema de observación se pasan explícitos y `_01` lo verifica con un assert**, porque `ConfigEscenario5` redefine sus defaults a los del Cuadro del Capítulo 3 (`L=48`, `T=300`, `sigma_obs=0.1`, `R=50`) y omitir uno rompería la comparabilidad sin dar error.
- **DESCARTADO como escenario del estudio (decisión del 2026-08-30).** Pasa a ser una cota analítica en la sección de limitaciones, y no se le corre MCMC.

  La corrida 15 se ejecutó con `M = 5` (no 4: la regla del 95 % sobre el espectro empírico da 5) y quedó en **caso nulo**, porque el truncamiento retiene `j* = 3`. Pero el problema es más profundo que elegir mal `j*`. Adoptar la regla del 95 % **no inmuniza** contra el Algoritmo 5: sólo mueve dónde tiene que estar `j*` para romperla —con `rho = 0.5` haría falta `j* = 6`— y ahí el efecto es despreciable por construcción.

  La cota: la ganancia máxima de modelar la componente `j` en vez de tratarla como ruido blanco es **`lambda_j * phi_j^2`**. Con `lambda_6 = 0.031` y `phi = 0.9` eso da `0.025` contra una varianza total de `1.97`, o sea **1.3 % del MISE**. Generalizando: bajo una regla de 95 %, toda componente excluida vale menos del 5 % de la varianza, luego el efecto del Algoritmo 5 está acotado por debajo del ~3 % del MISE — dentro del ruido Monte Carlo con `R = 1` y marginal con `R = 50`.

  **Eso es un resultado, no un vacío**, y así hay que escribirlo: la regla de varianza acumulada acota el daño que el desalineamiento varianza/predictibilidad puede causar, porque toda dirección que descarta contribuye poco a la curva aunque sea perfectamente predecible. Se demuestra con la cota, sin correr nada. Los artefactos de la 15 quedan en disco como evidencia del caso nulo.
- **`VERDAD` no se escribe a mano**: `_01 §4` persiste `reports/.../10_alineacion_fpca_generador.csv` (fpc, fourier alineada por correlación, `var_ratio`, `ar1_score`, `phi_generador`, `retenida`) y `_03 §6.1` la deriva de ahí, contrastándola con `internos["coeficientes_ar"]`. Las recursiones son escalares e independientes ⇒ ninguna covariable cruzada es activa y sólo el propio rezago lo es, y sólo donde `phi != 0`.
- **Es el único escenario en que sensibilidad, especificidad y AUC están las tres bien definidas a la vez**: el conjunto activo verdadero es pequeño y no vacío, al revés que en la 12 (vacío) y en la 11/13/14 (total). Y el prior `E[pi]=0.90` está mal orientado en casi todas las componentes, de modo que desmentirlo es una prueba genuina.
- Ocupación de la mezcla (`_03 §4`): el generador es gaussiano y homogéneo ⇒ una ocupación **cercana a uno es el resultado correcto** y debería ser la más baja de las seis corridas. Es la lectura opuesta a la de la 13.
- Diagnóstico (R=1): `base_ortonormal` con error `6.7e-16`, `reproyeccion_error_max = 1.6e-15`, `desalineacion_confirmada = True` (varianza máx en j=1, predictibilidad máx en j=3), `acf1[j*] ≈ 0.864`, `varianza_acumulada_dos_primeras = 75.1 %`, `razon_senal_ruido ≈ 30.3`. **`espectro_error_relativo_max ≈ 18.8 %` es informativo, no criterio**: con R=1 y T=400 el error de muestreo de una varianza es ~7 % y en la componente con `phi=0.9` el tamaño muestral efectivo es mucho menor. Lo que sí sería un fallo es un error creciente **sistemáticamente** en j, y `_01` reporta esa correlación.

**La corrida 16 (`notebooks/simulaciones/16_sim_E6/`) instancia el Algoritmo 6**, Bloque 2 y el único escenario no estacionario del estudio.

- Parámetros de `tab:ane_alg6`: `J=10`, `razon_espectro=0.5`, `phi_comun=0.7`, `indices_intercambio=(1,2)`, `modo="quiebre"`. Mismos asserts sobre los defaults de la dataclass que en la 15.
- La corrida 16 usó **`t_quiebre = T0 = 280`** y no el 270 del default, que está dimensionado para `T=300`/`T0=240`. Con `t* = T0` el entrenamiento queda homogéneo, todo el test queda post-quiebre y la ventana posterior pasa de 30 a 120 períodos. El motivo queda escrito en `simulation_config.json`.
- **Cambio decidido: `t_quiebre` a 340.** Con `t* = T0` el estrato pre/post **coincide** con la partición train/test y la cobertura condicional de `_04 §9` mide lo mismo que la comparación train/test de `§4` — la advertencia que hoy viaja en `eval_config.json`. Con `t* = 340` el entrenamiento sigue homogéneo y el test queda partido en 60 pre y 60 post, de modo que la estratificación pasa a ser **interna al test** y separa "pérdida de vigencia de la base" de "pérdida de generalización", que es toda la tesis del escenario. Se pierde ventana posterior (120 → 60), pero 60 sigue siendo el doble de la que hacía poco fiable la detección con el default de 270. La corrida 16 usó `M = 5`, no 4. **No hace falta tocar la amplitud**: `indices_intercambio = (1,2)` ya detecta el reordenamiento (quiebre estimado en 285 contra 280).
- **La advertencia que hay que arrastrar al reporte, y que los notebooks repiten en tres lugares**: con `t* = T0` el estrato pre/post **coincide con la partición train/test**, de modo que la cobertura condicional de `_04 §9` y la comparación train/test de `§4` miden **lo mismo**. No es un defecto —es la tesis del escenario— pero sin declararlo se leería el desplome de cobertura como pérdida de **generalización** cuando es pérdida de **vigencia de la base**. Está en el propio `eval_config.json`, bajo `estratificacion.advertencia`, para que viaje con los datos. Para separar ambos efectos habría que mover `t*` dentro del test (p. ej. 340) a costa de acortar la ventana posterior.
- **Estado verdadero discreto y persistido**: `reports/.../10_estado_quiebre.csv` con `quiebre_idx` (base-0, el que consume `cobertura_condicional` sin cuantilizar), `en_adaptacion` (los `periodos_adaptacion = 5` en que la varianza aún converge como `phi^(2n)`), nivel y energía de la curva y los dos coeficientes intercambiados. `_01` verifica fila a fila que el estrato coincida con `internos["trayectoria_espectro"]`, y `_04` que el primer período post sea exactamente `t*`.
- **`_04 §9.1` es la sección propia y el resultado central**: el error de representación `e_t = ||X_t - Pi_M X_t||^2` con la base ajustada en entrenamiento **no depende del modelo predictivo**, de modo que si crece tras `t*` la degradación es de la base. Se contrasta contra el nivel de entrenamiento y contra una base **reajustada sobre el test**, que es una **cota optimista** —se ajusta a los mismos datos sobre los que se evalúa— y por eso no debe reportarse como el desempeño de un método alternativo. `_04 §7` superpone además el MISE del modelo y el del truncamiento en la misma ventana móvil.
- `VERDAD` se deriva del mismo CSV de alineación que en la 15; aquí `phi = 0.7` en **todas** las componentes ⇒ el propio rezago es activo en todas y ninguna cruzada. El eje 3 es por eso **el menos informativo de los seis**: el prior 0.90 y la verdad apuntan en la misma dirección en la diagonal. La cifra reportable son las **cruzadas** (prior 0.5, verdad cero).
- Diagnóstico (R=1, `t*=280`): `base_ortonormal` con error `6.7e-16`, `reproyeccion_error_max = 1.3e-15`, `orden_pre_correcto = True`, `reordenamiento_detectado = True`, intercambio efectivo de las componentes 1 y 2 (`var_pre` 0.878 vs 0.628 → `var_post` 0.302 vs 0.764), `instante_quiebre_estimado = 285` contra 280 (error 5 = `periodos_adaptacion`), `acf1` por componente ≈ 0.65–0.76 contra `phi=0.7`, `razon_senal_ruido ≈ 31.8`. **`orden_post_correcto = False` es informativo y no criterio**: discrepa sólo en el par (8, 9), componentes de varianza ~0.006 indistinguibles entre sí con una sola trayectoria. El criterio estructural que `_01` sí exige con assert es el intercambio de las **dos componentes intercambiadas**, no el orden de las diez.

**Etapa D — barrido en `M` y réplicas. LA CONVENCIÓN YA ESTÁ DECIDIDA E IMPLEMENTADA** en la corrida 20 (ver §Convenciones): `M` viaja en el `EXPERIMENT_ID` como sufijo `_m<NN>` y se declara en la celda `[CONFIG]` de §1.1. Lo que queda:

- **El barrido en `M` propiamente dicho**, que es la corrida 20 con `M = 1, 2, 3`. **La mecánica ya está**: `_03`, `_04`, `_05` y el `.m` toman `M_FPCA_LIST` y recorren el barrido completo en una pasada, con su propia sección de comparación entre `M` (ver §Convenciones). Sólo el `_01` se corre una vez por punto. Estado al 2026-08-30: `M = 1` y `M = 2` corridos; falta `M = 3`.
- **Lo que el barrido ya dice con dos puntos, y hay que tomar con pinzas.** De `M=1` a `M=2` el MISE de prueba baja poco (1.0829 → 1.0662, −1.5 %) mientras el piso de truncamiento se desploma (0.2595 → 0.0339), de modo que la razón `MISE/piso` pasa de 4.2 a 31.5: **la ganancia es casi toda de la representación y el modelo no la aprovecha**. Pero el `_05` muestra que la degradación relativa al VAR es *menor* en el PSBPM-FD (+0.005) que en el promedio de los competidores (+0.080), o sea que el efecto **no** es específico del modelo propuesto. Con `R = 1` y dos puntos nada de esto es concluyente; es la dirección, no la magnitud.
- **Aviso de convergencia sobre lo ya corrido:** en `M = 1` y `M = 2` el `_03` da veredicto ✗ —`ESS` mínimo de 6.5 y 15.3 contra un umbral de 100, `|Geweke|` hasta 11.5— con `nsim = 2000`, `burn = 500` y 3 cadenas. Mientras eso siga así, las diferencias entre puntos del barrido mezclan efecto de `M` con muestreo insuficiente, que es justamente lo que la §8 del `_03` existe para detectar.
- **La hipótesis que el barrido va a probar**, formulada al montar la 20: *agregar componentes de varianza baja deteriora el modelo, no sólo deja de aportar*. El mecanismo propuesto tiene dos partes. (a) El estandarizador lleva todo score a varianza unitaria, así que una componente con `lambda ≈ 1e-3` se amplifica ~30x y llega al muestreador indistinguible de una covariable informativa, con su ruido de medición y su error de estimación de la autofunción dentro. (b) `M` entra **dos veces** —como dimensión de respuesta y como `p = q*M` covariables en la regresión probit de los pesos—, y esos pesos son **compartidos por todos los átomos**: una componente basura no es sólo una salida mal predicha, degrada la asignación de la mezcla también para las componentes buenas. Por eso el daño puede ser más que compensatorio. Si se confirma, responde `§03_06` con la lectura invertida de la esperada: no "M chico pierde información" sino "M grande importa ruido, y el prior de selección no alcanza a filtrarlo".
- **`K` acota el barrido y `K` lo fija el GCV**, que difiere por escenario (ver §Parámetros fijos). En E2 y E3 hay sólo 4 componentes en total.
- **Réplicas (`R = 50`)**, que sigue siendo el cambio estructural grande y no está empezado. Con `R = 1` una diferencia del 2–3 % en MISE entre PSBPM-FD y FAR(1) es indistinguible del ruido Monte Carlo, de modo que **ninguna afirmación comparativa del capítulo es concluyente hoy**.

**Etapa E — agregación y tablas** siguiendo los seis puntos del `% [PENDIENTE]` de `§03_07`.

---

---

# El Escenario B: la corrida 18

`notebooks/simulaciones/18_sim_EB/` es un escenario de **diagnóstico** —como el A
de la 17, se nombra con letra y no es un Algoritmo del anexo— construido para
responder al único hueco que ninguno de los seis cubre: **un generador donde la
clase lineal homogénea no pueda competir en media condicional.**

## Por qué existe

El cuello de botella del capítulo no es que falten escenarios que rompan un
supuesto, sino que **los que existen rompen supuestos que la clase lineal
tampoco necesita**: en el E1 la media condicional es lineal (el FAR(1) está
correctamente especificado), en el E2 es constante (la media incondicional es
óptima), en el E4 es la del E1, y en la corrida 21 el RESET no rechaza
linealidad. En los cuatro casos el MISE no discrimina, y **no por defecto del
modelo**. El E3 debía ser la excepción y no lo fue.

## El generador

    X_t = mu + s_t · Psi(X_{t-1} - mu) + eps_t,   P(s_t = +1 | X_{t-1}) = Phi(kappa (z_{t-1} - c)),
    z_{t-1} = <X_{t-1} - mu, e>,  e(tau) ∝ sin(2 pi tau)

es decir el **FAR(1) del Algoritmo 1 con el signo del operador conmutado** por
un umbral sobre el estado rezagado. Los dos regímenes son `Psi` y `-Psi`:
**antisimétricos**.

**El mecanismo es de simetría, y es exacto.** El mejor predictor lineal depende
del proceso sólo a través de `C1 = E[Y_t ⊗ Y_{t-1}] = Psi E[s(Y_{t-1}) Y_{t-1} ⊗ Y_{t-1}]`,
cuyo integrando es par en el signo e impar en `Y`: si la ley de `Y` fuera
exactamente simétrica, `C1` sería el operador **nulo**. No lo es —la deriva
conmutada es par y eso desplaza la media— pero la cancelación es casi completa
cuando `e` es casi ortogonal a la dirección dominante del proceso, que es lo que
fija la elección de `e ∝ sin(2 pi tau)` y de `gamma = 0.60`.

**Diagnóstico del generador con los parámetros elegidos** (`seed=41232`, `T=400`,
`R=1`), todo calculado por `resumen_escenario_B`:

| Cantidad | Valor | Lectura |
|---|---|---|
| `r2_lineal_fuera_de_muestra` | **−0.038** | el mejor lineal **no supera a la media incondicional** |
| `r2_oraculo_fuera_de_muestra` | **0.376** | y sin embargo un tercio largo de la varianza es predecible |
| `acf1_media` | 0.049 | la serie **parece ruido blanco** a cualquier diagnóstico lineal |
| `fraccion_origenes_ambiguos` | 0.353 (141 de 400) | contra el 14 % de la corrida 13 |
| `fraccion_origenes_bimodales` | 0.188 (75) | ambiguos **y** con estado grande |
| `separacion_modas_en_sd_innovacion` | 2.29 | contra 1.47 de la corrida 13 |
| `sarle_oraculo_bimodales` | **0.581** | por encima del 5/9 uniforme: bimodalidad genuina |
| `sarle_oraculo_ambiguos` / `_deterministas` | 0.473 / 0.386 | el contraste que el modelo debería reproducir |
| espectro FPCA | 89.8 % / 96.9 % | la regla del 95 % da **M = 2** |
| `\|<e, phi_2>\|` | 0.85 | **la dirección de conmutación es la 2ª componente** |

`r2_lineal_dentro_de_muestra` (0.007) se reporta aparte a propósito: con L=75
regresores y T=400 el `R²` dentro de muestra está inflado por sobreajuste, y es
el sesgo que haría parecer favorable al escenario sin serlo. La cifra citable es
la de fuera de muestra.

## Lo que este escenario permite y ningún otro

1. **El MISE discrimina.** `18_05 §7.1` es la sección nueva: pone en la misma
   escala la media incondicional, el mejor lineal, el PSBPM-FD y el **oráculo**
   —la media condicional verdadera, leída de `interno_media_condicional` del
   `.npz` y proyectada sobre las mismas `M` autofunciones—. La cifra reportable
   es `fraccion_brecha_recuperada`: 0 significa que el modelo se comporta como
   un lineal, 1 que alcanza el óptimo teórico. **Ningún otro escenario del
   estudio puede producir esa cifra**, porque en los demás la brecha es nula.
2. **Referencia *oracle* para la bimodalidad.** `coeficiente_sarle_mezcla_simetrica`
   calcula de forma **exacta** el Sarle de la ley condicional verdadera
   proyectada, y viaja en `eval_config["forma_predictiva"]`. `18_04 §9.1` lo
   contrasta con el mismo coeficiente sobre la predictiva del modelo, separando
   deterministas / ambiguos / **bimodales** (ambiguos y con estado grande). Es
   exactamente el diagnóstico que a la corrida 13 le faltó y sin el cual un
   Sarle bajo no distingue un fallo del modelo de un generador sin contenido.
3. **El barrido en `M` tiene un punto de corte predicho de antemano.** Con
   `M = 1` el modelo **no observa** la dirección que gobierna la conmutación
   —carga sobre la 2ª componente— y no puede sino fallar como el FAR; con
   `M >= 2` sí la observa. La predicción es un **salto** entre 1 y 2, no la
   degradación suave de la corrida 20 (después de `M=2` vuelve a regir aquélla).
   `18_01 §3.5` persiste `10_alineacion_conmutacion.csv` con
   `fraccion_e_explicada = sum_{k<=M} <e, phi_k>^2`, que es la cifra que decide
   de qué lado cae cada punto del barrido.

## Divergencias deliberadas respecto del Algoritmo 1

Sólo dos, y ambas están argumentadas en `18_01 §2.1`. **Hay que declararlas al
comparar la 18 con la 20**, porque la comparación no es limpia:

| | Alg. 1 | Escenario B | Por qué |
|---|---|---|---|
| `gamma` | 0.30 | **0.60** | núcleo más ancho ⇒ el rango de `Psi` se concentra en la dirección casi constante y queda más ortogonal a `e` ⇒ más cancelación. `r2_lineal` pasa de 0.031 a 0.002 en corrida larga |
| `hs_norm` | 0.70 | **0.90** | la media condicional es `s_t Psi Y_{t-1}`: su varianza —y con ella el `R²` del oráculo— crece con `hs_norm` |

`nitidez = 4.0` (≈ 1.5 en unidades de `sd(z)`) es el punto elegido del
intercambio central del escenario: **subirla agranda la brecha oráculo/lineal y
elimina los orígenes bimodales**, que son los únicos con contenido para el eje
distribucional. `ell = 0.5`, `sigma_eps = 1.0`, `umbral = 0` y los parámetros
fijos del estudio (`L=75`, `T=400`, `PROP_TRAIN=0.70`, `sigma_obs=0.25`,
`mu = sin(2 pi tau)`) son los de siempre.

## Archivos propios

- `pipelines/sim_escenario_B.py` — `ConfigEscenarioB`, `generar_escenario_B`,
  `resumen_escenario_B`, `simular_trayectoria_far_signo`,
  `direccion_oscilatoria`, `coeficiente_sarle_mezcla_simetrica`.
  `internos` lleva `signos`, `proyeccion_estado`, `prob_signo_positivo`,
  **`media_condicional`** (el oráculo, `(R, T, L)`) y el operador.
- `18_01 §2.3` — la celda que decide si el escenario sirve: tres `assert`
  (`r2_lineal < 0.05`, `r2_oraculo > 0.15`, `fraccion_ambiguos > 0.15`). Si
  alguno falla no tiene sentido gastar MCMC.
- `reports/.../10_estado_signo.csv` — estado verdadero por período (`signo`,
  `p_signo_pos`, `z_lag`, `ambiguo`, `norma_arrastre`, `bimodal`). A diferencia
  de la corrida 16, **el estrato no coincide con la partición train/test**, de
  modo que la cobertura condicional mide algo distinto de la comparación
  train/test.
- `reports/.../10_alineacion_conmutacion.csv` — carga de `e` sobre cada FPC.
- `reports/.../11_tendencia_nivel.png` — el nivel con la tendencia determinista,
  la de cada rama y la recta de MCO superpuestas. Es la figura que hace
  verificable que la tendencia existe y que su forma no es lineal.
- Salidas nuevas: `61_bimodalidad_predictiva.csv` y `62_cobertura_condicional.csv`
  del `_04`; `76_techo_oraculo.csv` del `_05`.

**Estado: los cuatro notebooks y los tres `.m` están escritos, pero la corrida
no se ha ejecutado.** El generador sí está verificado numéricamente (las cifras
de la tabla de arriba salen de ejecutar `18_01` hasta §2.7).


---

# La familia con tendencia: corridas 19, 22, 23, 24, 25, 26 y 27

`pipelines/sim_escenario_T.py` genera **siete** escenarios de diagnóstico —C a
I— en dos tandas. Un solo módulo, porque comparten la tendencia, el esquema de
observación, el oráculo y el control de calidad.

**Primera tanda: un factorial 2×2 limpio.**

| | tendencia **lineal** | tendencia **acelerada** (`potencia p=3`) |
|---|---|---|
| **interacciones** intra-curva (unimodal) | **C · corrida 19** | **D · corrida 22** |
| **multimodalidad**, 2 regímenes | **E · corrida 23** | **F · corrida 24** |

Cualquier diferencia entre 19 y 22 (o 23 y 24) es atribuible a la **forma de la
tendencia** —lineal contra `potencia p=3`, que es la que sí se ve—; entre las dos filas, al **tipo de no linealidad**.

**Segunda tanda: tendencias con estructura propia.** Ya no es un factorial —cada
una interviene una cosa distinta— y todas son variantes del mecanismo `mezcla`:

| corrida | regímenes | tendencia | qué rompe |
|---|---|---|---|
| **G · 25** | **3** | una **forma distinta por régimen** (cuadrática / sinusoidal / logarítmica) | las modas se separan **a ritmos distintos y con formas distintas** |
| **H · 26** | 2 | **sinusoidal creciente y volátil**: sube ondulando y el factor de escala es un AR(1) | la tendencia deja de ser determinista y de ser monótona |
| **I · 27** | **3** | **por tramos**: cuadrática → logarítmica → sinusoidal creciente, con quiebres en `0.45T` y `0.75T` | la **derivada** de la tendencia es discontinua: cambia el régimen de tendencia |

Todo lo demás —esquema de observación, `gamma=0.30`, `hs_norm=0.70`, `ell=0.5`,
priors, `mcmc_config`— es idéntico a las corridas 18 y 20 en las siete.

    Y_t = m(Y_{t-1}, S_t) + eps_t,   X_t = mu + b_{S_t}(t) g(tau) + Y_t

`deriva` es el desplazamiento total entre t=1 y t=T: **3.0** en las corridas de
interacción (19, 22) y **6.0** en las de mezcla (23–27), donde hace falta más
porque las ramas reparten la deriva. `inclinacion = 0` (nivel puro). `perfil_tendencia` está normalizado como en el Escenario A: `b(1)=0`,
`b(T)=deriva`, de modo que las corridas con tendencia son comparables. Con
`T0 = 0.7T`: la lineal deja `b(T0)=0.70·deriva`, la cuadrática `0.49·deriva`, y
el desfase entre bloques pasa de 0.35 a 0.44 de la deriva — **la cuadrática es
el caso adverso, no una variante cosmética**.

## La tendencia tiene que VERSE, y eso costó dos correcciones

Las dos primeras versiones de esta familia producían series en las que **la
tendencia no era visible en el histórico**, por motivos distintos y ambos
instructivos:

1. **Derivas antisimétricas por régimen.** Con `derivas_regimen = (+1, -1)` y
   ocupaciones parecidas, la tendencia se **cancela en el nivel medio**: medido,
   la deriva neta de la serie era `-0.06` desviaciones. Cada rama tenía su
   tendencia y la separación entre modas crecía, pero un gráfico de la serie
   parecía estacionario. La corrección es que **todas las derivas sean del mismo
   signo y difieran en magnitud** —`(1.0, 0.25)` con dos ramas, `(1.0, 0.6, 0.2)`
   con tres—: la separación entre modas depende de la *diferencia* de derivas y
   no de sus signos, de modo que no se pierde nada y se gana una tendencia neta
   de ~2.7–3.4 sd.
2. **Formas por régimen que se promedian a una recta.** En el Escenario G cada
   rama tiene su propia forma (cuadrática, sinusoidal creciente, logarítmica) y
   el promedio ponderado de las tres es **casi lineal**: la curvatura de la
   tendencia media es sólo 0.30 sd. No es un defecto —lo que distingue al
   escenario vive en la *dispersión* entre ramas y no en su promedio— pero hay
   que saberlo antes de mirar el gráfico y concluir que la tendencia es lineal.
   Por eso la figura `11_tendencia_nivel.png` dibuja **las J tendencias por
   rama** además de la media.

`_01 §2.5` produce esa figura —nivel realizado, tendencia determinista,
tendencia de cada rama y **la recta ajustada por MCO**— y falla con `assert` si
la deriva neta no supera una desviación estándar. La distancia entre la
tendencia y su propia recta es la cifra que dice si la forma es genuinamente no
lineal o si una recta la resume.

## El catálogo de formas, y cuánto se ve cada una

La columna que decide si una forma se distingue de una recta **a ojo** es la
curvatura: la máxima distancia entre `b(t)` y su propia recta de mínimos
cuadrados, en unidades de `deriva`. Multiplicada por `deriva` y dividida por la
desviación de la componente estable (~1.34) da `curvatura_en_sd`, que el
diagnóstico reporta. **Por debajo de ~0.5 sd la forma se lee como lineal.**

| forma | expresión | curvatura | qué hace |
|---|---|---|---|
| `lineal` | `u` | 0 | reparte la deriva de forma uniforme |
| `cuadratica` | `u²` | **0.166** | la concentra al final; es de las formas **menos curvas** que hay |
| `potencia` | `u^p` | 0.30 con `p=3` | generaliza a la cuadrática; `p<1` da formas cóncavas |
| `exponencial` | `(e^{cu}−1)/(e^c−1)` | 0.28 con `c=3` | acelera al final, como la potencia |
| `logistica` | sigmoide normalizada | 0.16 | **dos cambios de concavidad**: plana, sube, se aplana |
| `logaritmica` | `log(1+cu)/log(1+c)` | 0.25 | **espejo de la cuadrática**: casi toda la deriva al principio |
| `sinusoidal` | `sin(2π·κ·u)` | 0.30 | **vuelve a cero**: `b(1)=b(T)=0`, la serie no tiene tendencia neta |
| `sinusoidal_creciente` | `[u + a·sin(2π·κ·u)]/(1 + a·sin(2π·κ))` | 0.20 | **sube ondulando**, con retrocesos locales |
| `escalon` | `1{u ≥ f}` | 0.54 | quiebre de nivel; no es tendencia, es el caso extremo |

En todas salvo `sinusoidal` —donde es la amplitud— `deriva` es el desplazamiento
total entre `t=1` y `t=T`.

**La lección de la cuadrática.** Su máxima separación respecto de su propia recta
es `deriva/4`, la segunda más baja del catálogo: con `deriva = 6` son 0.51 sd y
sobre 400 puntos con ruido **se lee como una recta**. Por eso las corridas 22 y
24 usan `potencia` con `p = 3` en vez de `u²`: con `deriva = 8` la curvatura
llega a **1.22 sd**, la forma es inequívoca, y —esto es lo que importa— el `R²`
del mejor predictor lineal **cae a 0.018**, porque una recta ya no la resume. Lo
que distingue a una cuadrática de una lineal en el pipeline nunca fue su aspecto
sino el desfase entre bloques; con `p = 3` además se ve.

`sinusoidal` pura es la que hay que evitar cuando lo que se quiere es una
tendencia visible: al volver a su punto de partida, el histórico parece no
tener tendencia. `sinusoidal_creciente` es la que usan las corridas 26 y 27:
sube como la lineal pero con `κ` ondas encima, y con `amplitud_sinusoidal`
grande la **pendiente instantánea llega a ser negativa** por tramos sin que la
tendencia deje de ser creciente en el largo plazo.

La logarítmica produce el modo de fallo **opuesto** al de la cuadrática: el
entrenamiento ve una pendiente grande y el test una casi nula, de modo que un
modelo que extrapole sobrecorrige. La sinusoidal rompe la **monotonía**, y con
ella la coincidencia entre un estrato por nivel de tendencia y la partición
train/test —el defecto de la corrida 16—: el mismo nivel de `b(t)` ocurre en los
dos bloques. Con `ciclos_sinusoidal` **no entero** (1.25) los dos bloques caen
además en fases distintas, que es lo que impide que el modelo aprenda el ciclo
del entrenamiento y lo continúe.

`perfil_tramos` encadena varias formas **con continuidad**: cada tramo arranca
donde terminó el anterior, de modo que el perfil es continuo pero **su derivada
no lo es** — y esa discontinuidad *es* el cambio de régimen de tendencia. El
perfil global se normaliza por su máxima excursión para que `deriva` conserve su
lectura y encadenar tres tramos no la multiplique por tres.

## Tendencia volátil

`volatilidad_tendencia > 0` multiplica la tendencia por `1 + v·eta_t`, con
`eta_t` un AR(1) estandarizado de persistencia `persistencia_volatilidad`. Tres
decisiones que hay que conocer:

- **`eta_t` se sortea de un generador propio**, derivado de la semilla pero
  independiente del de la dinámica. Así activar la volatilidad **no desplaza el
  consumo del generador de `Y`**, y una corrida volátil y su gemela determinista
  comparten exactamente la misma trayectoria de la componente estable: la
  diferencia entre ambas es atribuible sólo a la tendencia.
- **Se permite que el factor sea negativo** (`v·eta_t < -1`): habría tramos con
  la tendencia invertida. `fraccion_tendencia_invertida` lo reporta; con la
  `volatilidad_tendencia = 0.5` de la corrida 26 vale 0, es decir el factor se
  mantiene positivo y lo que cambia es **cuán rápido sube**, entre 0.2x y 2.4x.
  Los retrocesos que sí se ven vienen de la ondulación de
  `sinusoidal_creciente`, no del factor.
- **El oráculo condiciona sobre `eta_{t-1}`**, que es un estado **latente que
  ningún método observa**. En la corrida 26 el oráculo es por tanto una cota más
  generosa que en el resto y parte de la brecha es **irreducible**. Hay que
  declararlo al reportar la fracción recuperada.

## Los dos mecanismos

**`"interaccion"` (C y D), unimodal.** La media condicional suma al operador
lineal un término cuadrático que hace interactuar puntos del dominio **dentro de
la misma curva rezagada**:

    m(Y) = Psi Y + lambda sum_j peso_j s_j tanh((Y(a_j) Y(b_j) - c_j)/s_j) h_j

con pares por defecto `(0.25, 0.75)` —un extremo interactúa con el otro— y
`(0.50, 0.50)` —el cuadrado del nivel local—. `Y(a)` es un **promedio local**
(`nucleo_local`, integral unitaria) y no el valor de una celda: un valor puntual
no sobrevive a la base ni al truncamiento FPCA.

- **La saturación `tanh` no es cosmética y hay que declararla.** Una recursión
  bilineal sin acotar es **explosiva con probabilidad positiva**; medido, con
  `razon_interaccion = 0.6` la serie desborda antes de los 600 períodos (fue el
  primer intento y falló con `overflow`). La zona lineal se fija en `saturacion`
  desviaciones del producto: en el rango central el término **es** el producto.
  `fraccion_saturada` = **9.5 %** de pares-instante con los valores por defecto.
- **`razon_interaccion` se CALIBRA**, no se fija a ciegas: un piloto sin el
  término mide la sd L² de la parte lineal y de la cuadrática y ajusta `lambda`
  para que su cociente sea el pedido. Es de primer orden —al activar el término
  la varianza cambia— y por eso el diagnóstico reporta
  `razon_interaccion_efectiva` (0.644 contra 0.60 nominal) en vez de dar por
  buena la nominal.
- **Aquí la no linealidad NO cancela la parte lineal**, al revés que en el
  Escenario B: `Psi` sigue estando y sigue siendo estimable, de modo que **la
  referencia lineal no muere, se queda corta**. Es el caso realista.

**`"mezcla"` (E, F, G, H, I), multimodal.** `J` ramas ---la longitud de
`factores_operador`--- sorteadas por un **probit ordenado** sobre
`z = <Y_{t-1}, e>`, con `J-1` cortes: es el mismo mecanismo del Algoritmo 3 y por
eso un escenario de esta familia con tres ramas es comparable con aquél. Cada
rama tiene su factor sobre el operador, su deriva **y opcionalmente su propia
forma de tendencia** (`formas_regimen`), que es lo que hace multimodal a la
tendencia misma en la corrida 25. El rasgo propio:

    separacion(t) = |f_0-f_1| ||Psi Y_{t-1}||  +  |d_0-d_1| b(t) ||g||
                     (dinámica, constante)         (tendencia, creciente)

**la distancia entre las modas extremas crece con t**: 5.32 sd de la innovación
en `T0` contra 7.12 en `T` (corrida 23). Un modelo unimodal responde con una
moda en el medio y su error **crece con el tiempo** — modo de fallo que ningún
escenario anterior produce, y que la ventana móvil de `_04 §7` debería mostrar.

**Con `J = 3` el coeficiente de Sarle deja de ser la cifra principal, y no es un
fracaso.** Con tres modas aproximadamente equiespaciadas la central **rellena el
hueco** entre las extremas y la densidad se aplana: Sarle detecta *bi*modalidad
y baja aunque la multimodalidad sea mayor (0.44 en la corrida 25 contra 0.66 en
la 23, con más modas). Por eso el diagnóstico agrega **`modas_efectivas`**
`= 1/sum_j p_j^2`, el número efectivo de componentes activas por origen, y con
`J = 3` la lectura correcta es el par (modas efectivas, separación máxima). El
valor absoluto del Sarle **no es comparable entre escenarios con distinto número
de ramas**; la comparación interna —ambiguos contra deterministas, train contra
test— sí lo sigue siendo.

## Diagnóstico medido de las siete corridas (seed 41232, T=400, R=1)

| | C (19) | D (22) | E (23) | F (24) | G (25) | H (26) | I (27) |
|---|---|---|---|---|---|---|---|
| mecanismo | interacción | interacción | 2 reg. | 2 reg. | **3 reg.** | 2 reg. | **3 reg.** |
| forma de la tendencia | lineal | **potencia p=3** | lineal | **potencia p=3** | una por rama | sinus. creciente + volátil | tramos |
| `deriva` | 3.0 | 4.5 | 6.0 | 8.0 | 6.0 | 6.0 | 6.0 |
| **deriva neta del nivel** | +2.16 sd | +2.79 sd | +2.73 sd | **+3.34 sd** | +2.86 sd | **+3.41 sd** | +3.23 sd |
| **`curvatura_en_sd`** | 0.00 | 0.80 | 0.00 | **1.22** | 0.30 | 0.62 | 0.30 |
| `r2_lineal_fuera_de_muestra` | 0.729 | 0.749 | 0.294 | **0.018** | 0.213 | 0.212 | 0.427 |
| `r2_oraculo_fuera_de_muestra` | 0.847 | 0.901 | 0.690 | 0.772 | 0.767 | 0.775 | 0.778 |
| **brecha destendenciada** | 0.067 | 0.067 | 0.207 | 0.337 | 0.350 | **0.435** | 0.262 |
| `acf1_media` | 0.760 | 0.827 | 0.279 | 0.442 | 0.420 | 0.381 | 0.407 |
| `desfase_en_sd` (train↔test) | 0.89 | 1.47 | 2.25 | **3.29** | 2.71 | 1.96 | 2.52 |
| `monotona` | sí | sí | sí | sí | sí | **no** | **no** |
| ambiguos | — | — | 0.33 | 0.33 | **0.60** | 0.33 | **0.60** |
| `modas_efectivas_media` | — | — | 1.41 | 1.41 | **1.88** | 1.41 | **1.88** |
| separación T0 → T (sd innov.) | — | — | 4.3→5.6 | 3.2→7.1 | 2.8→5.7 | **8.0→9.7** | 4.5→5.7 |
| Sarle oráculo ambiguos train→test | — | — | 0.43→0.59 | 0.38→0.58 | 0.34→0.41 | 0.47→**0.64** | 0.35→0.41 |
| espectro acumulado (M=1,2,3) | .873/.937/.981 | .907/.954/.986 | .926/.975/.985 | .931/.977/.986 | .901/.967/.980 | **.959**/.986/.992 | .924/.975/.985 |
| **M por la regla del 95 %** | **3** | **2** | **2** | **2** | **2** | **1** | **2** |

Las cifras son las de los `_01` **ya ejecutados** (los 21 puntos del barrido, seed
41232), no estimaciones.

Cuatro lecturas de esta tabla:

- **Las brechas destendenciadas de C y D son idénticas por construcción** (0.067):
  la tendencia no realimenta la dinámica, de modo que la componente `Y` es la
  misma serie en las dos corridas y sólo cambia la parte determinista. Es la
  prueba de que el eje "forma de la tendencia" está aislado.
- **F es el caso que muestra para qué sirve el eje de la tendencia.** Con la
  forma `potencia p=3` el `R²` del mejor predictor lineal cae a **0.018** —una
  recta no resume una tendencia con esa curvatura— mientras el oráculo se
  mantiene en 0.77. La forma de la tendencia, y no sólo la no linealidad de la
  dinámica, puede por sí sola matar a la referencia lineal.
- **Los `R²` en bruto ya NO son la cifra a citar.** Con una tendencia neta de
  ~3 sd, el predictor lineal recoge la persistencia que la propia tendencia
  induce (`acf1` sube de ~0.02 a ~0.4) y su `R²` pasa de ~0 a 0.14–0.43 sin
  haber aprendido nada de la dinámica. **La columna que discrimina es la brecha
  destendenciada**, y ahí H y G son los que más margen dejan (0.44 y 0.35).
  Es el mismo fenómeno que la corrida 21 encontró en el Mapocho.
- **H mantiene el mayor margen** (0.435 destendenciada) y además su oráculo
  condiciona sobre un estado latente (`eta_{t-1}`), de modo que es una cota más
  generosa que la de los demás: parte de esa brecha es irreducible y hay que
  declararlo.
- **G e I tienen el Sarle más bajo y son los MÁS multimodales**: con tres modas
  la central rellena el hueco y el coeficiente baja. `modas_efectivas` (1.88 de
  3 posibles, contra 1.41 de 2 en E/F) es la cifra que hay que citar ahí.
- **En H el espectro cambia de forma**: la primera componente se lleva el 96 %
  porque la tendencia de nivel puro domina, y la regla del 95 % da `M = 1`. El
  barrido sigue siendo `(1, 2, 3)` y el punto informativo sigue siendo `M = 2`
  —donde entra la dirección de conmutación—, pero el punto `M = 1` deja de ser
  el punto "pobre" que es en las demás corridas de mezcla.

## Lo que hay que arrastrar al reporte

- **Ninguno de los cuatro es estacionario**, y eso trae de vuelta la lección de
  la corrida 17: el centrado del FPCA y del estandarizador se ajustan con el
  bloque de entrenamiento y **dejan de ser válidos en el de prueba**. Parte del
  error de test es de *vigencia del centrado*, no de predicción.
- **Por eso todo se reporta por duplicado**, en bruto y destendenciado. Con
  `acf1 = 0.76` un VAR sobre scores parecerá excelente sin haber aprendido nada
  de la dinámica — el mismo fenómeno que la corrida 21 encontró en el Mapocho
  (`acf1 = 0.989`, la persistencia como piso). El destendenciado usa la
  tendencia **determinista** (la esperada bajo la ocupación de las ramas), no la
  realizada: restar la realizada rompería la relación entre la media condicional
  y el pasado, y el diagnóstico dejaría de medir lo que dice.
- **El error estándar de la pendiente va corregido por autocorrelación.** El
  residuo de la regresión del nivel sobre `b(t)` es `Y` promediada sobre `tau`,
  fuertemente autocorrelacionada (`acf1 = 0.71` en C), y el EE de MCO la
  subestima por un factor 2.4: con el de MCO la pendiente parecería
  significativamente distinta de la esperada sin serlo. El `assert` usa
  `desvio_pendiente_en_ee` sobre el EE corregido.
- **El estrato de `_04 §9` nunca es `b(t)`.** `b(t)` es monótona en `t`, de modo
  que su estrato sería un intervalo contiguo y coincidiría con la partición
  train/test — el defecto declarado de la corrida 16. En C y D el estrato es
  **el cuartil de la magnitud de la interacción** (`q_t`), y en E y F el
  **estado discreto del régimen**. Las dos celdas verifican con `assert` que
  ningún estrato caiga casi entero en un bloque.

## Barrido en `M` y archivos propios

- **C y D corren `M = 2, 3, 4`** y no `1, 2, 3`: la regla del 95 % da `M = 3` y,
  sobre todo, los núcleos de lectura son funciones **localizadas** y necesitan
  más componentes que una dirección suave para sobrevivir al truncamiento.
  `19_01 §3.5` persiste `10_alineacion_interaccion.csv` con
  `fraccion_explicada` de cada lectura.
- **E, F, G, H e I corren `M = 1, 2, 3`**, como la 18: la dirección de
  conmutación carga sobre la **segunda** componente (`|<e, phi_2>| = 0.85` en las
  cinco), de modo que el punto de corte del barrido está predicho de antemano.
  `10_alineacion_conmutacion.csv`. La excepción a vigilar es la 26, donde la
  regla del 95 % ya da `M = 1` y el punto bajo del barrido no es tan pobre como
  en las demás.
- Estado verdadero: `10_estado_tendencia.csv` (C, D: `b_t`, `q_interaccion`,
  `estrato_q`) y `10_estado_signo_tendencia.csv` (E–I: régimen, una columna
  `p_regimen_j` por rama, `p_max`, `modas_efectivas`, `sep_dinamica`,
  `sep_tendencia`, `factor_volatil`, `bimodal`). En las corridas 25 a 27 el
  archivo tiene **J columnas de probabilidad** y `ambiguo` pasa a definirse como
  `p_max < 0.75` —ninguna rama domina—, que con `J = 2` coincide exactamente con
  la definición de las corridas 18, 23 y 24.
- `_04 §9.1` cambia de contenido según el mecanismo: **error y cobertura por
  cuartil de interacción** en C y D (`61_error_por_interaccion.csv`), y
  **bimodalidad contra el oráculo, comparando train con test**, en E y F
  (`61_bimodalidad_predictiva.csv`). `62_cobertura_condicional.csv` en los
  cuatro. `_05 §7.1` (techo del oráculo, `76_techo_oraculo.csv`) es común.

**Estado: los veintiocho notebooks y los siete `.m` están escritos; ninguna de
las siete corridas se ha ejecutado.** Los generadores sí están verificados: las
cifras de la tabla salen de ejecutar los `_01` hasta §2.7.

**Nota de reproducibilidad:** al generalizar el mecanismo `mezcla` a `J` ramas,
el sorteo del régimen pasó de `rng.random() < p` a `rng.choice(J, p=...)`, que
consume el generador de otra forma. Las corridas **23 y 24 cambiaron de
realización** —no de distribución— y las cifras de la tabla son las nuevas; las
19 y 22 son bit-idénticas a las de antes del cambio, y eso se verificó
explícitamente.


# Datos reales: la corrida 21

`notebooks/reales/21_real_nivel/` es la **corrida 20 aplicada a una serie observada**, y sustituye a los
`notebooks/reales/Ejemplo_{1,2,3}/`, que siguen la arquitectura antigua (sin partición train/test, sin
`artifacts.py`, evaluación en un solo `03_03`). Los notebooks son `21_01_datos`, `21_03_convergencia`,
`21_04_evaluacion` y `21_05_comparacion`; `config_paths.m` apunta a `data/reales/` y `artefact/reales/`.

**Datos.** `data/reales/raw/Tabla_de_Mediciones_Wed_Jun_17_2026.xlsx` — estación RIO MAPOCHO EN LOS ALMENDROS,
48 mediciones diarias (cada media hora), 2020-01-01 a 2026-05-31. Cada **día** es una curva. La variable
modelada es `Nivel de Agua (m)`; el archivo también trae turbiedad, precipitación y caudal.

**`EXPERIMENT_ID`**: `real_<serie>_v<ventana a 2 dígitos>_m<M a 2 dígitos>` (p. ej. `real_nivel_v01_m03`).
Sustituye a `ESCENARIO_ID`/`REPLICA_ID`, que no aplican: no hay generador ni réplicas Monte Carlo. Como en la
corrida 20, `M` viaja en el id y está escrito a mano en los cuatro notebooks y en el `.m`.

**Lo que cambia respecto de una simulación, y por qué:**

- **No hay curva verdadera.** No se escribe `X_curves_true.npy`; `cargar_curvas_true(PATHS, estricto=False)`
  devuelve `None` y `_04`/`_05` toman la curva **observada** como objetivo. `eval_config.json` lo declara con
  `objetivo_evaluacion = "curva_observada"`.
- **`modo_residuo = "empirico"`**, no `"ninguno"`: la banda tiene que cubrir la curva observada, de modo que el
  error de representación forma parte de lo que hay que cubrir. `_04` estima los residuos con
  `residuos_representacion` **sólo sobre el bloque de entrenamiento**. Con `"ninguno"` aparecería subcobertura
  atribuible a la representación y no al modelo.
- **Las cifras NO son comparables con las de los escenarios simulados**: allí el denominador es la curva sin
  ruido y aquí incluye el ruido de medición, que ningún modelo puede predecir.
- **`_03 §6.1` deja de ser validación y pasa a ser hallazgo**: no hay estructura verdadera contra la cual
  contrastar las PIP, de modo que `contraste_con_verdad` no se usa y se reportan las PIP contra el **prior**
  (0.90 sobre el propio rezago, 0.50 sobre los cruzados).
- **`_05 §3.2` cambia de sentido**: sin operador verdadero, `||Psi_hat||_HS` pasa de verificación a estimación,
  y depende del `M` con que se calculó.

**Higiene de datos que la simulación da gratis** (todo en `21_01 §2.2`, con asserts): grilla intra-diaria
canónica (horas presentes en ≥90 % de los días), interpolación de huecos **a lo largo de tau** dentro de cada
día, descarte de días con >20 % ausente, detección de marcas `(día, hora)` duplicadas, y reporte de **saltos en
la secuencia de días** — un día ausente rompe el supuesto AR porque el rezago de `t` deja de ser `t-1` en tiempo
real. `MARGEN_DIAS` hace que la ventana lea de más para llegar a `T_CURVAS` tras los descartes; la fecha final
efectiva queda en `dataset_config.json`, el análogo real de `simulation_config.json`.

**Ancla de continuidad**: la curva del día `d` gana un punto en `tau = 0` igual a la última medición del día
`d-1`, de modo que `G = 49` y no 48. Viene de `Ejemplo_3` y se conserva.

## Dos secciones nuevas en `21_05`, y responden a una pregunta abierta del capítulo

**§3.2.1 — `VAR` y `FAR1` son el mismo modelo con `p = 1`.** En las figuras de ventana móvil la curva del VAR
queda oculta bajo la del FAR1, y no es un problema de dibujo: la estandarización es afín, luego el FAR1 en
escala estandarizada es `D^-1 Psi D` más un intercepto, exactamente la familia sobre la que el VAR minimiza por
MCO. Sólo difieren en el estimador (MCO contra Yule-Walker) y en el intercepto. La sección lo **cuantifica**;
en `real_nivel_v01_m03` da `corr = 0.999986` y `max|VAR - FAR1| = 0.85 %` de la sd del objetivo.
**Consecuencia para el capítulo: son UNA referencia lineal, no dos, y así hay que citarlas.** Para separarlas
haría falta `p > 1` en el VAR o `RIDGE_FAR > 0`.

**§3.2.2 — RESET de Ramsey sobre el bloque de entrenamiento.** Contrasta si la media condicional de `xi_t` dado
`xi_{t-1}` es lineal. Es la condición **necesaria** para que el PSBPM-FD pueda ganar en MISE: si no se rechaza,
la referencia lineal está correctamente especificada y el MISE no discrimina, exactamente como en el Escenario 1
simulado. En `real_nivel_v01_m03` **no rechaza** en ninguna de las tres componentes (`p` = 0.94, 0.58, 0.80).

## Diagnóstico de la primera corrida ejecutada (`real_nivel_v01_m03`)

Ventana 2024-01-01 … 2025-02-06, `T = 400`, `T0 = 280`, base B-spline `(10, 2)` por GCV, `K = 10`, `M = 3`
(99.51 % de varianza), `N_LAGS = 1`, `cond(W) = 3.86`, contrato verificado.

- **`acf1_nivel_diario = 0.989`.** El nivel medio diario está cerca de una raíz unitaria. La línea base de
  **persistencia** da `MISE = 0.00051` contra `0.0254` de la media incondicional: **el piso a batir es
  `y_hat(t) = y(t-1)`**, no la media. Cualquier modelo lineal razonable queda muy cerca de ese piso, y la
  brecha disponible para un modelo flexible es diminuta.
- **La regla del 95 % da `M = 1`**: `lambda_1` se lleva casi toda la varianza porque el nivel diario domina el
  espectro. `M = 3` es una elección por encima de la regla, y hay que declararla.
- Combinado con el RESET que no rechaza, el veredicto es el mismo que en el Escenario 1: **en la media
  condicional no hay nada que ganar en esta serie**, y el argumento del capítulo tiene que salir de `21_04`
  —CRPS, energía, cobertura, PIT—, no de la tabla de MISE.
- **Camino natural si se busca contraste en la media**: modelar el **incremento** `X_t - X_{t-1}` en vez del
  nivel (quita la raíz unitaria y deja al descubierto la dinámica), o pasar a `Caudal (m3/seg)`, que es mucho
  menos persistente que el nivel. Ninguna de las dos está implementada.
