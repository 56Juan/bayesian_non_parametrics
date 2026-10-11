# Corrida 300: la 200 reevaluada con predictores, intervalos y comportamientos

Prompt autocontenido para una sesión nueva. Leer primero `CLAUDE.md` (modo de trabajo,
invariantes, estructura de la corrida 200) y `docs/03 Modelo.tex` §03_05 (Reconstrucción
funcional: esperanza, extracciones, predictores puntuales, intervalos) y §03_06
(Comportamientos predictivos). Esas dos secciones definen *qué* se calcula; este prompt
dice *cómo* se incorpora al pipeline sin romper lo que ya corre.

## Contexto que no está en el repo (hallazgos de la sesión del 2026-10-10)

- El pipeline persiste como predicción puntual del PSBPM-FD la **esperanza analítica**
  (`momentos()`), y el Bloque A la evalúa con MAE. Cuando la predictiva es multimodal la
  esperanza cae entre los modos y pierde por construcción bajo MAE (Gneiting 2011: el
  funcional debe ser consistente con la métrica; MAE ↔ mediana, RMSE ↔ esperanza).
- Verificado sobre todas las corridas con trazas (MAE funcional en test, objetivo X^(M)):
  tráfico (corrida 31, real) esperanza 0.409 → mediana puntual 0.373 (bimodal en ξ1,
  régimen laborable/fin de semana); Mapocho (24) 0.0091 → 0.0079 (asimetría, no
  bimodal); 200 TAR 0.602 → 0.595 (bimodal débil: los regímenes del TAR difieren en
  dinámica, no en nivel, y sus medias condicionales cerca del umbral están juntas);
  201, 202, SA1, PM2.5, BTC: sin diferencia (unimodales).
- En 200–202 `atau = 0.5` deja átomos con precisión ≈ 0: los draws tienen colas no
  acotadas (curtosis 1e5–1e6). La esperanza analítica y los cuantiles no lo sufren; la
  **media de los draws sí**. Nunca usar la media de los draws como estimador; recortar los
  draws a los cuantiles 0.5 %–99.5 % por score antes de cualquier clustering.
- Las etiquetas de los átomos saltan entre cadenas e iteraciones (ψ, Γ, α no convergen
  aunque los β sí). Todo lo que use el índice h se decide dentro de cada iteración; nunca
  se promedia por etiqueta. La predictiva no cambia con burn 500 → 1000.
- Reglas para contar comportamientos, probadas en sintéticos y en las 8 corridas:
  k-medias sobre los draws de scores (distancia euclídea = L² entre curvas; no
  estandarizar); (0) compuerta de valle: con K = 2, proyectar sobre el eje entre
  centroides, KDE gaussiana (Scott), min f en [a,b] / min(f(a), f(b)) < 0.5, si no K = 1;
  (1) crecimiento: aceptar K+1 si R²_{K+1} ≥ R²_K (K+1)/K; (2) masa ≥ α = 0.05 por grupo.
  Resultados: gaussiana, asimétrica y lognormal → K = 1 (0 % falsos positivos); bimodal
  60/40 y 90/10 → K = 2; cola rala de 0.6 % a 8 sd (Mapocho) → K = 1 por masa; tráfico
  K = 2 en 69 % de los días; 200 K = 1 en 77 %, K = 2 11 %, K = 3 12 %; 201 y 202 K = 1
  en 100 %. Alternativas descartadas: silueta (fallaba con asimetría), umbral absoluto de
  R² (depende de M), Ashman D sobre clusters de k-medias (una gaussiana partida da D ≈ 2.7),
  GMM por BIC (válido pero más parámetros), mean-shift (O(S²), minutos por corrida).
- Scripts de diagnóstico de esa sesión (no forman parte del pipeline) en
  `reports/reales/candidatos_katerin/candidatos_katerin.py` y figuras/CSV `diag_*` en
  `reports/reales/real_trafico_m10_v01_m10/`.

## Qué es y qué no es la 300
La 300 es la corrida 200 (TAR, `escenario_200_chungEscG_1_r01`, M = 1..10, rezago
propio, N_LAGS = 4, 3 cadenas, atau = 0.5) evaluada con los resúmenes de §03_05 y §03_06.
NO cambia el generador, el contrato, el entrenamiento ni la convergencia. NO se ejecutan
ni _01 ni MATLAB ni _03 de la 300: sus artefactos son los de la 200, copiados. La 200 NO
se toca: ni se reejecuta, ni se limpia, ni se escribe en sus directorios.

## Paso 0 — Copiar artefactos sin ejecutar nada
Script suelto `notebooks/simulaciones/300_sim_TAR_estimadores/00_copiar_200.py`,
parametrizado por (eid_origen, eid_destino, M_FPCA_LIST) para reutilizarlo en 301/302,
que para cada M en 1..10:
  - copia `data/simulaciones/raw/<eid200>/`, `processed/functional/<eid200>/` y
    `artefact/simulaciones/<eid200>/` (incluidas las trazas .mat) a los mismos
    directorios con `eid300 = escenario_300_chungEscG_1_r01_m<MM>`, usando
    `construir_paths(limpiar=False)` para los destinos;
  - reescribe en los JSON copiados toda aparición del eid200 por el eid300
    (`hyperparameters.json`: experiment_id, trazas_en; `eval_config.json`;
    `datasets_manifest.json`; `simulation_config.json`; `fpca_meta.json` si lo lleva).
    Con rezago propio `trazas_en` apunta al M entrenado (m10): mantener esa relación
    con el eid300 correspondiente;
  - NO copia `reports/` ni `predict/` de la 200 (la 300 los genera);
  - aborta si algún destino ya existe con contenido, salvo flag `--sobrescribir`;
  - termina llamando a `verificar_contrato` sobre cada eid300.
Dejar en la carpeta 300 copias de `300_01_simulaciones.ipynb`, `psbp_fd_iteracion.m`,
`config_paths.m`, `psbp_train.m` y `300_03_convergencia.ipynb` con
`BASENAME = "escenario_300"`, `LIMPIAR_DIRECTORIOS = False` y una celda de aviso al inicio
de _01: la 300 se construye por copia (paso 0); ejecutar _01 regenera datos que coinciden
por semilla pero obliga a reentrenar. _03 puede ejecutarse (sólo lee).

## Paso 1 — Paquete: cambios aditivos, compatibles con 200/201/202 y reales
Todo lo nuevo con valores por defecto que reproducen el comportamiento actual, de modo
que los notebooks existentes sigan corriendo sin cambios. Tras cambiar el paquete,
reiniciar kernels.

1a. `fit/evaluacion_barrido._predecir_punto` persiste los draws de scores `SC_draws`
    (S, n, M) en float32, en `banda_funcional_psbp.npz` o en un npz hermano con nombre
    en `pipelines/artifacts.ARCHIVOS`. La lectura devuelve None si no existe.

1b. Módulo nuevo `fit/resumenes_predictiva.py`, una sola definición de cada funcional
    sobre muestras de curvas (S, n, G) y de scores (S, n, M):
      - `mediana_puntual(X)` → (n, G)
      - `medoide_l1(X, pesos_tau)` → (n, G), distancia integrada con
        `utils.quadrature.pesos_trapezoidales`
      - `mediana_mbd(X)` → (n, G), profundidad de banda modificada J = 2 por rangos
        (Sun y Genton 2011): MBD(s) = (1/G) Σ_g [(S − r)(r − 1) + (S − 1)] / C(S,2)
      - `atomo_modal(models_chains, dfs)` → (n, M): por iteración y componente
        h* = argmax `pesos_probit`, media por `medias_componente`
        (`models/pspb_fd_v3/functions/predict.py`); mediana sobre iteraciones y cadenas;
        reconstruir con `curva_media_desde_scores`
      - `distancia_a_muestra_mas_cercana(X, curva)` como control del collage de la
        mediana puntual
    Ninguno usa la media de los draws; la esperanza sigue siendo `momentos()`.

1c. `_predecir_punto` persiste `X_pred` (esperanza, sin cambios), `X_pred_mediana`,
    `X_pred_medoide`, `X_pred_mbd`, `X_pred_modal`.

1d. `fit/evaluacion_barrido.tablas_ventana` / `apilar_tablas` / `resumen_metricas`:
    la ventana móvil se recorre por predictor y la tabla lleva la columna `predictor`
    (esperanza | mediana | medoide | mbd | modal). `verificar_relaciones_metricas` se
    aplica por predictor. Con `predictores=("esperanza",)` la salida es idéntica a la
    actual.

1e. `fit/comparacion_barrido.cargar_psbp` registra una fila por predictor:
    "PSBPM-FD (esperanza)", "(mediana)", "(medoide)", "(mbd)", "(modal)". FAR y RF no
    cambian. Bloque B no cambia. Corregir el mensaje de `ajustar_rf` que imprime
    "rezagos propios" también con diseño cruzado.

1f. Módulo nuevo `fit/comportamientos.py`, por origen sobre Z_t (S, M):
      - recorte a los cuantiles 0.5 %–99.5 % por score si atau ≤ 1 (leer de
        `hyperparameters.json`);
      - k-medias K = 2 (n_init ≥ 10, semilla fija); compuerta de valle como arriba;
      - regla de crecimiento; regla de masa con alpha = 1 − nivel;
      - salida: K, etiquetas, masas p_c, centroides, curvas típicas (centroide
        reconstruido), bandas condicionales (cuantiles equi-colas dentro del grupo),
        Δ entre curvas típicas, `X_pred_cluster` (curva típica del más probable).
    `catalogo_global(centroides, masas)`: el mismo procedimiento sobre los centroides de
    todos los orígenes ponderados por masa; vector de probabilidades por origen.
    `validar(X_obs, ...)`: probabilidad del comportamiento en que cae la observada (grupo
    más cercano en L²), acierto del de mayor masa, cobertura y amplitud de la banda
    condicional del realizado frente a la marginal. En simulación: cruce con el régimen
    verdadero de `internos` en `escenario_200.npz` (régimen por score activo y período).
    Las constantes (0.5 del valle, alpha) en un solo lugar.

1g. `graphics/viz_evaluacion_barrido.py` y `graphics/viz_comportamientos.py` (nuevo):
    predictores contra M; histogramas de draws por score con esperanza, mediana y
    observado; curvas simuladas con los cinco predictores; comportamientos por origen
    (curvas típicas y bandas condicionales); K por origen; catálogo.

## Paso 2 — Notebooks de la 300
`300_04_evaluacion.ipynb` = copia del `200_04` con `BASENAME = "escenario_300"` y:
  - §3 persiste draws y los cinco predictores;
  - §4 y §8 reportan el Bloque A por predictor (MAE y E_max con cada mediana, RMSE con
    la esperanza; columna `predictor`);
  - §11 (nueva) diagnóstico de multimodalidad: fracción de orígenes con
    |esperanza − mediana| > 0.25 sd por score, distancia de la mediana puntual a la
    muestra más cercana;
  - §12 (nueva) comportamientos: K por origen y por M, masas, catálogo global, curvas
    típicas, validación contra el régimen verdadero; salidas por M en prefijos 60–65 y
    en el barrido 100–104 (verificar que no colisionen con los existentes).
`300_05_comparacion.ipynb` = copia del `200_05` con `BASENAME = "escenario_300"` y:
  - §3.3 lee los cinco predictores;
  - §4, §6, §8 Bloque A con las filas por predictor junto a FAR y RF;
  - §5 y §7 sin cambios;
  - §10 (nueva) validación de comportamientos (las tres cifras) por M; salidas 82–83
    por M y 105–107 en el barrido.
Ambos declaran los mismos [CONFIG] que la 200 salvo BASENAME; las diez métricas, las
ventanas y los objetivos se leen de `eval_config.json`.

## Paso 3 — Pruebas antes de ejecutar nada
`tests/test_resumenes_predictiva.py`: mezcla bimodal sintética (60/40, separación 1.8,
sd 0.17) donde mediana, medoide, MBD y modal caen en el modo mayoritario y la esperanza
en el valle; unimodal donde los cinco coinciden salvo error Monte Carlo; MBD por rangos
contra la definición directa con S pequeño.
`tests/test_comportamientos.py`: gaussiana, asimétrica (skewnorm a = 10) y lognormal →
K = 1; bimodal 60/40 y 90/10 → K = 2; dos modos pegados (sep 1.0, sd 0.3) → K = 1; cola
rala de masa 0.6 % a 8 sd → K = 1 por masa; recorte con atau ≤ 1.
Guardado/lectura del npz con los nuevos arreglos y compatibilidad con un npz antiguo.
Chequeo estático de que `200_04`/`200_05` (sin tocar) siguen importando.

## Paso 4 — Ejecución (sólo esto se ejecuta)
`PYTHONUTF8=1 PYTHONIOENCODING=utf-8 python -m jupyter execute --inplace` sobre
`300_04_evaluacion.ipynb` y luego `300_05_comparacion.ipynb`. Nada más. Reportar al
terminar: Bloque A por predictor y M (test), Bloque B, K por M y acierto de
comportamientos contra el régimen verdadero.

## Cuidados
- No cambiar la convención de nombres de trazas ni `ruta_traza`.
- La mediana puntual puede ser un collage si varias componentes son bimodales a la vez;
  por eso se reporta la distancia a la muestra más cercana y se prefieren medoide/MBD
  cuando difieren.
- Actualizar `CLAUDE.md`: §4 (predicción puntual → cinco predictores; _04 §11–§12;
  _05 §10), §6.3 (nuevos arreglos del npz), tabla de prefijos de salida, §8.

## 301 y 302
Mismo procedimiento con el script de copia parametrizado. 301 = 201 (GARCH, rezago
propio, N_LAGS = 4, M = 1..10): el generador no tiene regímenes (`internos` trae la
varianza condicional h_t), así que la validación contra régimen no aplica; en su lugar
reportar la distribución de K (se espera 1) y la correlación entre la amplitud de la
banda y h_t. 302 = 202 (multimodal, cruzado, N_LAGS = 2, M = 3..6): `internos` trae el
mecanismo Z_t y la validación sí aplica; con diseño cruzado cada M tiene sus propias
trazas (`trazas_en` apunta a sí mismo). El diagnóstico previo dio K = 1 en todos los
orígenes de la 202: el resultado interesante es si las reglas encuentran los tres
mecanismos y, si no, qué dice eso del escenario.
