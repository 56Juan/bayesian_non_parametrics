# CLAUDE.md — Guía de trabajo

Define cómo Claude Code asiste en este repositorio. Es la fuente de verdad sobre el
*modo de trabajo*; el código, `docs/` y la presentación lo son sobre el *contenido*.

---

## 1. Estado y propósito

**El estudio está cerrado** (noviembre de 2026). Las corridas están ejecutadas, la tesis
está redactada y el paquete no va a recibir integraciones ni pruebas nuevas. **El trabajo
actual es pulir la presentación de la defensa** (`ppts/FINAL/`, §2).

Qué hace Claude:

- Editar la presentación y el guion **cuando se le indique y sólo lo indicado**.
- Regenerar o ajustar figuras de la presentación.
- Responder preguntas sobre el modelo, los resultados y el código.
- Cambios puntuales al paquete o a `docs/` cuando se pidan.

Fuera salvo pedido explícito: redactar capítulos, proponer secciones, ejecutar corridas,
agregar métricas o etapas, decidir qué resultados "cuentan".

Código, docstrings y nombres están en español; mantener ese idioma.

---

## 2. La presentación — `ppts/FINAL/`

`ppts/` está en `.gitignore`: **no hay historial que la recupere**. Respaldar el `.pptx` en
el scratchpad antes de cada edición.

| Archivo | Qué es |
|---|---|
| `PRESENTACION_TESIS.pptx` | **La fuente de verdad.** 38 láminas, 16:9. El autor la edita a mano. |
| `PRESENTACION_TESIS.pdf` | Exportación de revisión. |
| `GUION_PRESENTACION.docx` | Guion hablado. El autor lo edita en Word. |
| `build_presentacion_tesis.py` | Script que armó la primera versión. **Ya no reproduce el deck: no reconstruir láminas con él.** |
| `tesis/estilo.py` | Colores, tarjetas, `add_text`, `fit_image`, fórmulas (`eq_png`, `eq_nat`). Usar estos helpers al agregar formas. |
| `figuras/figuras.py` | Todas las figuras, a `IMG/`. `python ppts/FINAL/figuras/figuras.py t05 m01` regenera sólo las que empiezan así. |
| `animar_tesis.py` | Agrega animaciones con PowerPoint (COM) desde `tesis/anim.json` y exporta el PDF. |
| `medir_desbordes.py` | Textos que no caben en su caja y formas fuera del área útil (COM). |
| `revisar.py <carpeta>` | Exporta a PDF y arma hojas de contacto de 6 láminas. |

Orden del deck: portada e índice (1–3) · datos reales, los dos problemas (4–6) · marco
teórico: datos funcionales, FPCA, series, DP, DPM, DDP (7–16) · modelo: predictiva,
predictores, intervalos, comportamientos (17–22) · resultados: evaluación, tráfico,
comportamientos (23–31) · síntesis (32–33) · anexos (34–38). Los números cambian cuando el
autor quita o agrega láminas: **ubicar una lámina por su kicker o su título, no por número**.

### Cómo editar

- **Parchar el `.pptx` en sitio** con python-pptx, dando nombre explícito a las formas
  nuevas, y guardar en un temporal. Después un paso COM hace el `SaveAs` al definitivo.
  Si `POWERPNT` está abierto, abortar sin tocar nada.
- Verificar exportando la lámina a PNG y mirándola. `medir_desbordes.py` para los textos.
- **Tocar sólo lo que el autor nombró.** No reescribir ni agregar frases en otros cuadros;
  si el cambio pedido obliga a tocar otro cuadro, preguntar antes. No agregar acotaciones.
- En las definiciones se redacta en general, sin el ejemplo del tráfico, salvo que se pida.
- **Guion**: leer el `.docx` actual y usarlo como base; aplicar sólo lo pedido y comprobar
  que los cortes del autor siguen. **Nunca restablecer algo que él quitó.** El guion no
  remite al anexo: explica lo que la lámina muestra. Las indicaciones escénicas van entre
  corchetes.

### Estilo (el de `ppts/AVANCE_2/AVANCE_2.pptx`)

- Fondo blanco, kicker celeste `1C7293` con letras espaciadas ("2 · MARCO TEÓRICO · FPCA"),
  título serif Cambria en navy `21295C`, logo USACH arriba a la derecha.
- Contenido en tarjetas redondeadas gris `EEF3F6`; definiciones y teoremas en tarjetas navy
  con título menta `02C39A`; entradas en negrita celeste. Pie "PSBPM-FD · Series de Tiempo
  Funcionales".
- **Letra: cuerpo 16 pt, encabezados de tarjeta 18 pt**, kicker y pie 14 pt. Se probaron
  17 y 18 pt y se descartaron: no volver a proponerlo.
- Figuras con fondo `EEF3F6` y fuente Calibri, dibujadas al tamaño exacto de su caja
  (`CAJAS` en `figuras.py`) para insertarlas a escala 1; ningún texto bajo 14 pt.
- Las fórmulas son **imágenes** (mathtext, caché en `tesis/eq/`). Cambiar una letra obliga
  a re-renderizarla y escalarla a la altura de la original.

### Notación del deck — difiere de la tesis a propósito

| Cantidad | Deck | Tesis (`CAP/00_Simbologia.tex`) |
|---|---|---|
| concentración del DP | **M** | c |
| componentes FPCA retenidas | **Q** | M |

---

## 3. El proyecto

Tesis: extensión del **PSBP** (Probit Stick-Breaking Process, Chung & Dunson 2009) a
**series de tiempo funcionales**, con el nombre **PSBPM-FD**. El paquete `model_psbp_fd`
genera datos, consume trazas MCMC y evalúa predicciones. **El muestreo MCMC ocurre en
MATLAB.**

- El estudio de simulación se especifica **sobre los scores** de una base de Fourier
  conocida, con tres escenarios: **TAR** (C-2, cambio de régimen por umbral), **GARCH**
  (C-1, heterocedasticidad condicional) y **multimodalidad** (C-3).
- La predictiva es una **mezcla**. `docs §03_05` define tres predictores puntuales
  (esperanza, mediana puntual, átomo modal; el código calcula además el medoide L¹, que no
  se reporta) y tres intervalos (equi-colas, banda simultánea, HDR); `§03_06`, los
  **comportamientos predictivos** (k-medias sobre los draws de scores).
- Regla de consistencia (Gneiting 2011): MAE y E_max con una mediana, RMSE con la esperanza.
- El foco es caracterizar **cuándo y por qué** el PSBPM-FD supera a las alternativas. Un
  resultado negativo bien acotado es un resultado.

### `docs/` — la tesis

`main.tex`, `CAP/` (`00_Antes`, `00_Resumen`, `00_Simbologia`, `01 Introduccion`,
`02 Marco Teorico`, `03 Modelo`, `04 Aplicacion`, `05 Conclusiones`) y `ANE/` (`01 Anexo`,
`02 Anexo Resultados`, `03 Anexo trafico`, `04 Anexo otros`). Compila en Overleaf; esta
máquina no tiene LaTeX. `docs/` figura en `.gitignore` pero sus archivos están versionados:
un archivo nuevo necesita `git add -f`.

Convenciones del texto: prosa sin dos puntos ni listas, un párrafo por idea cerrado con
`\\`, `\citep` entre paréntesis y `\citet` cuando el autor es sujeto.

**El documento no tiene que calzar con el código**: el código genera insumos de más. No
cambiar código para seguir un valor del documento.

---

## 4. Invariantes del estudio

Referencia de cómo se evaluó todo. Si una instrucción los contradice, avisar antes.

### 4.1 Diez métricas, fijadas por `docs §02_02_03`

Se calculan una sola vez sobre la ventana móvil (`ventana_movil_funcional(..., bloque_A=True)`);
todo lo demás se deriva de esa tabla.

| Bloque | Métrica | Columna |
|---|---|---|
| **A** error | MAE funcional | `mae_f` |
| | RMSE funcional | `rmse_f` |
| | Ē_max, promedio de los máximos por curva | `linf_medio` |
| | E_max^glob, máximo de la ventana | `linf_max` |
| **B** calibración | MPIW | `mpiw` |
| | PICP, fracción del dominio cubierta | `picp` |
| | PICPB, curvas cubiertas por completo | `picp_simultaneo` |
| | Winkler integrado | `winkler` |
| | W̄_max, promedio del peor Winkler por curva | `winkler_max_medio` |
| | W_max^glob, peor Winkler de la ventana | `winkler_max_glob` |

Relaciones verificadas con `assert`: `mae_f <= l2_medio <= linf_medio <= linf_max`,
`picp_simultaneo <= picp`, `winkler <= winkler_max_medio <= winkler_max_glob`. **`rmse_f` no
pertenece a esa cadena** (es una media cuadrática sobre orígenes). `mise` y `mise_rel` son
diagnóstico interno, no métricas reportables.

### 4.2 Ventanas

`w ∈ {20, 30, 40}`, `paso = 1`, solapadas; se reporta **`w = 30`**. Las ventanas que cruzan
`T0` se excluyen. Partición 70/30. Se leen de `eval_config.json["ventana_movil"]`.

### 4.3 Barrido en `M`

`M_FPCA_LIST` se declara en todos los notebooks y en el `.m`; los puntos comparten
realización, base, partición y `mcmc_config`. `M` significa dos cosas: en FPCA, componentes
retenidas; en `mcmc_config`, tamaño de la grilla G\* del stick-breaking (`N` es el
truncamiento de átomos).

### 4.4 El error se mide contra la curva

Nunca contra la grilla cruda (`docs §03_05_00`). El Bloque A, contra la curva suavizada; el
Bloque B, contra la curva suavizada y contra la representación truncada `X_t^(M)`.
`modo_residuo = "ninguno"` por construcción del modelo.

---

## 5. El pipeline

Cada corrida es una carpeta con notebooks y un paso MATLAB, acoplados **por archivos en disco**.

| Paso | Archivo | Qué hace |
|---|---|---|
| 1 | `_01` | Genera o carga los datos, representación, FPCA, datasets AR y contrato. Un juego de artefactos por `M`. |
| 2 | `psbp_fd_iteracion.m` | Lee el contrato y entrena con `psbp_train.m`, sólo con el bloque train. |
| 3 | `_03_convergencia` | Diagnósticos de las cadenas. No toca el bloque de prueba. |
| 4 | `_04_evaluacion` | Predictiva, métricas sobre la ventana móvil, PIP, comportamientos. |
| 5 | `_05_comparacion` | PSBPM-FD contra FAR(p) y RF. Bloque B sólo FAR y PSBPM-FD. |
| 6 | `_06_opciones` | Comportamientos como opciones de pronóstico: **más probable** y **mejor comportamiento** (cota). |

`hyperparameters.json` es la **única fuente de verdad del contrato Python ↔ MATLAB**
(`n_iter`, `mcmc_config`, `hyperparams_list`, `partition`, `seed_base`, `trazas_en`). Hay una
copia de `config_paths.m` y `psbp_train.m` por carpeta.

Los notebooks sólo orquestan:

| Notebook | Cálculo (`fit/`) | Figuras (`graphics/`) |
|---|---|---|
| `_01` | `preparacion_barrido` | `viz_fpca` |
| `_03` | `convergencia_barrido`, `diagnostics_mcmc` | `viz_convergencia`, `viz_traces`, `viz_global_components` |
| `_04` | `evaluacion_barrido`, `rolling`, `resumenes_predictiva`, `intervalos_predictiva`, `comportamientos` | `viz_evaluacion_barrido`, `viz_evaluacion`, `viz_comportamientos` |
| `_05`, `_06` | `comparacion_barrido`, `far_operador` | `viz_comparacion` |

**Competidores.** FAR(p) sobre los coeficientes en métrica L², con `p = N_LAGS` fijo. RF:
un bosque por componente con las mismas covariables del PSBPM-FD; elige el orden por
ventanas de *train*. El PSBPM-FD se **lee** de `banda_funcional_psbp.npz`, no se recalcula.

### Artefactos

| Clave | Ruta | Contenido |
|---|---|---|
| `raw` | `data/<dominio>/raw/<EID>/` | curvas, configuración, `escenario_<id>.npz` |
| `functional` | `data/<dominio>/processed/functional/<EID>/` | manifest, datasets por componente, FPCA, `theta.csv` |
| `predict` | `data/<dominio>/processed/predict/<EID>/` | `banda_funcional_psbp.npz`, `draws_scores_psbp.npy` (ignorado por git), `comportamientos_psbp.npz` |
| `out_report` | `reports/<dominio>/<EID>/` | figuras y CSV del punto |
| `out_artefact` | `artefact/<dominio>/<EID>/` | `hyperparameters.json`, `eval_config.json`, trazas `.mat` |

`<EID>` es `escenario_<NNN>_<variante>_<id>_r<NN>_m<NN>` en simulaciones y
`real_<serie>_v<NN>_m<NN>` en reales. Lo que cruza los puntos del barrido va a
`reports/<dominio>/..._barrido_M/`. `utils/rutas.py` construye los directorios (gemelo de
`config_paths.m`) y `pipelines/artifacts.py` decide los nombres de archivo.

### Modelo

`models/pspb_fd_v3` es la versión en uso y no tiene muestreador: trazas `.mat` → predictiva
por score → `PropagadorFuncional` → muestras de curvas `(S, n, G)`. La entrada de los
notebooks es `ModeloTraza` (`leer_traza`, `momentos`, `muestrear`). `psbp_fd_v1` y `v2` son
heredadas y no se importan.

Hiperparámetros: Chung y Dunson §4.2 en escala cruda, relativos a la dispersión de cada
serie en train. Variante `chungEscG` (rezago propio, `taupsij_z = 10`) y `chungEscX`
(covariables cruzadas). `atau = 0.5` en simulaciones, `atau = 2` en reales.

---

## 6. Corridas y dónde está cada cosa

### Simulaciones — `notebooks/simulaciones/`

| Carpeta | Escenario | Estado |
|---|---|---|
| `08_02 3 escenarios/200_sim_TAR`, `201_sim_GARCH`, `202_sim_MULT` | generación y entrenamiento originales | notebooks con salidas embebidas; **sus artefactos ya no están en disco** |
| `300_sim_TAR_estimadores` | TAR (C-2), `M = 1..10`, rezago propio | evaluada; tiene sus propios datos y trazas |
| `301_sim_GARCH_estimadores` | GARCH (C-1), `M = 1..10`, rezago propio | evaluada |
| `302_sim_MULT_estimadores` | multimodalidad (C-3), `M = 3..6`, cruzadas | evaluada |

**Las cifras del documento salen de las 30x**, que son las 20x reevaluadas con todos los
predictores e intervalos. En 300 y 301 los `.mat` están sólo en `m10` y los demás puntos
apuntan ahí con `trazas_en`.

Configuración común: `T = 1000` (`T0 = 700`), grilla de 75 puntos, `J = K = 10`,
`MCMC = {nsim 2500, burn 500, N 30, M 50}`, 3 cadenas, semilla 41232, una réplica.

Lectura: la esperanza gana el RMSE en los tres; la mediana puntual baja el MAE ~1.3 % en la
300 y coincide con la esperanza en 301 y 302; el RF gana el Bloque A en la 300. La HDR sólo
se parte en la 300 y ahí le gana el Winkler a la equi-colas; la simultánea es la única
calibrada en PICPB; las tres le ganan al FAR desde `M = 4`. `K ≥ 2` en 30 % del test de la
300 (ley verdadera 9 %), 0.3 % en la 301 y 0 % en la 302.

### Datos reales — `notebooks/reales/`

Todas con `M = 10`, una cadena, `chungEscG`, `atau = 2`.

| Carpeta | `SERIE_ID` | Serie |
|---|---|---|
| `101_real_trafico` | `trafico_101` | tráfico I-94, rezago 1 |
| `102_real_trafico_l17` | `trafico_102` | tráfico I-94, rezagos `{1, 7}` |
| `103_real_sa_precio` | `sa1_103` | precio SA1 (AEMO) |
| `104_real_pm25` | `pm25_104` | PM2.5 Parque O'Higgins (SINCA) |
| `105_real_beijing_pm25` | `beijing_105` | PM2.5 de Beijing |
| `106_real_eolica_smard` | `eolica_106` | factor de planta eólico DE (SMARD) |
| `107_real_btc_vol` | `btc_107` | volatilidad realizada de BTCUSDT |

**El capítulo 4 y la presentación usan sólo el tráfico (101 y 102)**; 103–107 van en el
anexo D. Fuera del tráfico la predictiva es unimodal (`K = 1` en más del 99 % del test) y el
Bloque A queda empatado o del FAR, salvo la 104; el Bloque B es del PSBPM-FD en Beijing y la
104. `102_04` no se ejecuta: sus salidas son copia de la 32.

### Historia — no se edita ni se cita como estado actual

- `notebooks/simulaciones/00_*` a `07_*`, y `notebooks/reales/01 Antiguos primera tanda` y
  `02 Antiguos segunda tanda` (21–32). Otra arquitectura (B-spline por GCV, ruido de
  medición, escalera de hiperparámetros): sus cifras no son comparables con las vigentes.
  Sus artefactos ya no están en disco; valen por sus salidas embebidas.
- `model_psbp_fd/pipelines/deprecated/`: generadores de esas corridas.
- En disco quedan artefactos sin notebook vivo: `real_sa1_precio_v02`, `real_pm25_poh_v02`,
  `real_trafico_{mi,m10,l17}`, `real_btcusdt_vr`, `real_pm25_104_v02_m01..m06`,
  `avance_tesis_1`. **No se borran sin que se pida.**

---

## 7. Cómo trabajar conmigo

- **No ejecutar notebooks ni MATLAB** sin que lo pida. Sí: chequeo estático y scripts
  sueltos de verificación sobre datos ya generados.
- **Los notebooks no se tocan** salvo pedido explícito.
- Ser conciso: código, comandos y diagnósticos antes que explicaciones largas. Comentarios
  escuetos.
- No leer ni escribir artefactos a mano: usar `pipelines/artifacts.py` y `utils/rutas.py`.
- Tras cambiar el paquete hay que reiniciar el kernel de los notebooks.
- Si un notebook importa código viejo, el `pip install -e .` apunta a otra copia: reinstalar
  desde la raíz.

```bash
pip install -e .
```

```bash
python -m pytest tests -q
```

En Windows, `jupyter execute` necesita `PYTHONUTF8=1` o corrompe las tildes del `.ipynb`.

### Gotchas

- **MATLAB colapsa la dimensión singleton final** al guardar (`p = 1`).
  `utils/trazas.normalizar_trazas_mat` restaura el eje; no leer `shape[2]` a mano, usar
  `predictor.n_features_`.
- **Cuadratura y Cholesky tienen una sola definición**: `utils.quadrature.pesos_trapezoidales`
  y `utils.linalg.safe_chol`.
- **Con `atau <= 1` la sd predictiva no existe** (diverge); las bandas por cuantiles sí.
  Nunca la media de los draws como estimador; recorte a 0.5–99.5 % antes de agrupar.
- `predict(return_std=True)` de v3 devuelve la sd **predictiva**, no la de la media
  condicional.
- Los diagnósticos MCMC se calculan sobre cantidades invariantes a la permutación de
  etiquetas (`fit/diagnostics_mcmc.py`).
- `fit/rolling.py` no reentrena: desliza la ventana de *evaluación*.
- Índices: `component_idx` del manifest es base-0; los nombres de archivo usan
  `fpc_idx = component_idx + 1`. Trazas: `chain_fpc_<fpc_idx>_iter<chain>.mat`.
- Los comentarios `% [FIX]` de los `.m` documentan correcciones deliberadas.
- `.git` pesa ~23 GB: borrar archivos no lo reduce.

---

## 8. Decisiones cerradas — no reabrir

- Escenarios del documento: C-1, C-2 y C-3. Un escenario simulado con regímenes que
  difieran en nivel es trabajo futuro.
- Una réplica por escenario; las cifras son descriptivas y el documento lo declara.
- La 200/300 está calibrada a favor del gating; las conclusiones valen para esa calibración.
- Equi-colas y HDR son puntuales; sólo la banda simultánea está calibrada para el PICPB.
- Las etiquetas de los átomos no convergen (ψ, Γ, α; los β sí); la predictiva es
  invariante. R̂ no se calcula con una cadena.
- La regla de valle evalúa la densidad en los centroides; con tres modos colineales nunca
  da `K = 3`. No se cambia.
- Se reportan tres predictores; el medoide se calcula y no se reporta.
- La **cota** (mejor comportamiento) se lee primero como diagnóstico de la predictiva y
  después como costo de elegir. El exceso de comportamientos en C-2 (30 % contra 9 %) **no**
  se llama sobredetección: es masa repartida cerca del umbral por la transición probit suave.
- C-3 se describe como "multimodalidad en la ley conjunta de los scores" que no llega a la
  marginal de la curva.
- Capítulo 4: sólo tráfico I-94 (101 y 102). El cuadro de comportamientos contra tipo de día
  se quitó; el cálculo sigue reproducible desde `comportamientos_psbp.npz`.
- Sin control de extracciones al azar en el `_06`.

### Terminología

**FAR(p)** autorregresivo funcional · **TAR** autorregresivo por umbral (aquí el umbral es el
rezago propio) · **GARCH** varianza condicional autorregresiva · **FPCA** análisis principal
funcional en métrica L² · **Winkler**, **PICP**, **PICPB**, **MPIW**: Bloque B · **HDR**
región de máxima densidad · **T0** corte de la partición · **G\*** grilla de localización
del stick-breaking probit · **oráculo** la media condicional verdadera · **scores activos /
pasivos**: los 4 primeros llevan la dinámica del escenario, los otros 6 son AR(1) de
coeficiente 0.3.
