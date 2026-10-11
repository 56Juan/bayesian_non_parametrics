# experimento — PSBPM-FD conjunto (multivariado)

Variante del PSBPM-FD en la que **las K coordenadas de la representación se
modelan juntas**: un solo stick-breaking probit (gating compartido), átomos
con regresión vectorial `B_h` y covarianza `Σ_h`, respuesta = los coeficientes
blanqueados de la base y predictores = **dos rezagos funcionales**. Con un
átomo es el FAR(2) con `kn = K`; todo lo que gane sobre el FAR es la mezcla.

Todo vive en esta carpeta. Del paquete `model_psbp_fd` se usan, sin
modificarlos, `FunctionalRepresentation`, `FARp`/`seleccionar_kn`,
`ventana_movil_funcional`, las métricas, las bandas gaussianas y los
diagnósticos MCMC. De las corridas vivas 200–202 se **leen** las curvas y la
base del generador (`data/simulaciones/raw/escenario_20X_*/escenario_1.npz`) y
la banda del PSBPM-FD univariado (`banda_funcional_psbp.npz`); no se escribe
nada fuera de `notebooks/experimento/`.

## Estructura

```
mv_psbp/                 código del experimento
  rutas.py               data/ artefact/ reports/ bajo esta carpeta, misma convención de nombres
  datos.py               insumos: corridas vivas (lectura) y npz real
  preparacion.py         representación, blanqueo theta_w = theta L (W = L L^T), datasets, contrato
  trazas.py              lectura de chain_mv_iter<NN>.mat y predictor (pesos, media, muestras)
  convergencia.py        paso 03   evaluacion.py  paso 04   comparacion.py  paso 05
  pipeline.py            paso01_simulacion / paso01_real / registrar_jobs
  notebooks_gen.py       genera y ejecuta los notebooks E_01..E_05
matlab/
  psbp_train_mv.m        muestreador Gibbs conjunto (una cadena)
  psbp_fd_iteracion_mv.m driver: lee jobs_*.json, parfor sobre (caso × cadena)
200_TAR/ 201_GARCH/ 202_MULT/ real_eolica_DE/
  E_01_datos  E_02_entrenamiento  E_03_convergencia  E_04_evaluacion  E_05_comparacion
200_TAR_bloque/ 201_GARCH_bloque/ 202_MULT_bloque/   variante "bloque" (abajo), mismos cinco pasos
data/<dominio>/{raw,processed/functional,processed/predict}/<EID>/
artefact/<dominio>/<EID>/   hyperparameters.json, eval_config.json, chain_mv_iter<NN>.mat
reports/<dominio>/<EID>/    tablas y figuras
```

`EID = mv_<basename>_<esc>_r01_K<KK>`: `K` (coeficientes) ocupa el lugar de `M`.
La variante bloque usa el prefijo `mvb_`.

### Variante "bloque conjunto parcial"

Conjunto sólo sobre los scores activos del generador (ξ₁..ξ₄: `bloque = (0,1,2,3)`),
con los dos rezagos funcionales de **todas** las coordenadas como predictores
(`p = 20`), y las seis coordenadas pasivas con un PSBP univariado cada una
(el mismo muestreador con `q = 1`) sobre sus dos rezagos propios. Un dataset y
un `hyperparameters_<sub>.json` por submodelo; trazas `chain_bloque_iter<NN>.mat`
y `chain_uni_<k>_iter<NN>.mat`; `ModeloBloque` arma la predictiva de las K
coordenadas por concatenación (independencia condicional entre submodelos dado
el pasado, como en el univariado). El `E_05` compara contra el FAR, el
univariado de la corrida viva y el conjunto completo de este experimento.

## Modelo y muestreador (`psbp_train_mv.m`)

```
y_t | S_t = h ~ N_q( B_h [1; x_t], Σ_h )
P(S_t = h | x_t) = v_h(x_t) Π_{l<h} (1 − v_l(x_t)),  v_h(x) = Φ(α_h − Σ_j ψ_hj |x_j − Γ_hj|)
B_h' | Σ_h ~ MN(0, (n/g)(X_h'X_h)^{-1}, Σ_h)      g-prior de Zellner, como el univariado
Σ_h ~ IW(ν0 = q+2, S0 = 0.5·diag(var_train))      análogo de btau = 0.5·sd²
Γ_hj en la grilla G* por predictor; inclusión γ_hj con apij/bpij (0.5/0.5 en el rezago 1,
0.5/5 en el 2); ψ_hj ~ N+(0, 1/taupsij_j), taupsij_j = 10·var_train(x_j)
```

Pasos del Gibbs: S (vectorizado), Z latentes del probit, `B_h` (matriz-normal),
`Σ_h` (Wishart inverso por Bartlett), `g`, `w_j`, `α_h`, `μ`, `π_j`, `Γ_hj`
(grilla; el resto de la distancia se precalcula una vez por `(h, j)`),
`ψ_hj` (normal truncada exacta), `γ_hj` (razón de marginales con la columna
integrada bajo el g-prior condicional + término del gating). Trazas: `Bout
(nsim,N,q,p+1)`, `Sigout (nsim,N,q,q)`, gating, `Sout`, `loglikout`,
`mse_inout`, `entropout`, `gamdiagout`, `inE_media`. Costo: ~0.1 s/iter con
n=698, q=10, p=20; las 6 000 iteraciones (burn 1 500) del contrato ≈ 10 min por
cadena. Con 2 500 las cadenas de la 200 no convergian (R-hat hasta 2).

## Ciclo

1. `E_01` escribe curvas, representación, `theta_w`, `dataset_mv_{train,test}.csv`,
   `hyperparameters.json`, `eval_config.json` y `artefact/jobs_<EID>.json`.
2. `E_02` (o a mano): `matlab -batch "psbp_fd_iteracion_mv('<jobs.json>', n_workers)" -wait -logfile <log>`
   desde `matlab/` (en Windows `matlab` devuelve el control al instante si no se pasa `-wait`). Entrena sólo con train. `nsim_override` y `max_chains`
   opcionales para pruebas de humo.
3. `E_03` convergencia (R-hat, ESS, Geweke sobre log-verosimilitud, error
   in-sample, átomos activos, entropía, μ, g, π_j; PIP por predictor).
4. `E_04` predicción a h=1, banda por cuantiles, las diez métricas sobre la
   ventana móvil `w ∈ {20,30,40}` contra la curva suavizada; persiste
   `banda_funcional_psbp_mv.npz`.
5. `E_05` contra el FAR(2) (`kn` por hold-out, `pesos="conteo"` sobre `theta_w`,
   exactamente como el `_05`) y, en 200–202, contra el PSBPM-FD univariado de
   la corrida viva (`M = 10` en 200/201, `M = 6` en 202, alineado por el índice
   del objetivo). Tablas 72/73 (ventanas), 96–99 (razones, ganadores, resumen),
   figuras 75–77.

Regenerar y ejecutar todo: `python -m mv_psbp.notebooks_gen generar` y
`python -m mv_psbp.notebooks_gen ejecutar <200|201|202|real>` desde esta carpeta.

## Casos

| caso | insumo | K | p | T0 |
|---|---|---|---|---|
| 200 / 201 / 202 | curvas verdaderas y base de Fourier del generador (W = I) | 10 | 20 | 700 |
| real_eolica_DE | factor de planta eólica onshore DE (SMARD, CC BY 4.0), 2015–2026, 96 puntos/día; B-spline por GCV con tope 12 | 12 | 24 | 3009 |

Las trazas `.mat` de esta carpeta no se versionan (ver `.gitignore` local): son
~100 MB por cadena.

## Resultados (test, w = 30; `00_RESUMEN.ipynb`)

Bloque A, razón de MAE contra el FAR(2) y % de ventanas ganadas en MAE:

| caso | PSBPM-FD conjunto | PSBPM-FD univariado (corrida viva) | FAR |
|---|---|---|---|
| 200 TAR | **0.93** (0 %) | 0.79 (100 %) | 1 |
| 201 GARCH | 1.00 (20 %) | 0.99 (76 %) | 1 |
| 202 MULT | 1.00 (14 %) | 0.98 (78 %) | 1 |
| real eólica DE | 1.00 (50 %) | — | 1 (50 %) |

Bloque B, Winkler (PICP) y % de ventanas ganadas en Winkler:

| caso | FAR gaussiana | PSBPM-FD conjunto | PSBPM-FD univariado |
|---|---|---|---|
| 200 TAR | 4.98 (0.94) | **4.48** (0.96), 0 % | 3.91 (0.97), 100 % |
| 201 GARCH | 4.33 (0.94) | 4.32 (0.95), 24 % | 4.23 (0.95), 53 % |
| 202 MULT | 4.24 (0.94) | 4.19 (0.94), 23 % | 4.11 (0.94), 62 % |
| real eólica DE | 0.579 (0.94) | **0.521** (0.96), **89 %** | — |

Lectura:

- **El conjunto nunca supera al univariado** en 200–202. En la 200 recupera un
  tercio de la ventaja del univariado (7 % contra 21 % sobre el FAR): los
  regímenes del generador son independientes por score y un gating compartido
  no los puede representar. En 201 y 202 empata con el FAR en la media; el
  univariado con cruzadas (202) sigue 1,5 % por delante.
- **En el caso real gana el Bloque B**: Winkler 10 % mejor que la banda
  gaussiana del FAR en el 89 % de las ventanas, con la misma anchura y mejor
  cobertura (0.96 contra 0.94; PICPB 0.84 contra 0.76). El modelo usa
  exactamente **dos átomos** (`N_activos = 2` en toda la cadena): dos regímenes
  de varianza según el nivel de viento rezagado. En la media empata con el FAR.
- **Convergencia.** 201, 202 y el caso real convergen (R-hat ≤ 1.10 en todas
  las cantidades invariantes; `mse_in` con ESS > 300). **La 200 no**: con 6 000
  iteraciones una cadena queda en un modo peor (`mse_in` 0.95 contra 0.87 de las
  otras dos; R-hat de `mse_in` 9). El predictor mezcla las tres cadenas, como
  v3, así que la cifra de la 200 arrastra esa cadena. Las π_j no convergen en
  ningún caso: son la parte lenta del muestreador, igual que en el univariado.
- El caso real se entrenó con 2 500 iteraciones (contrato pinned en su
  `E_01`); sus cadenas convergen y cada una tarda ~25 min con n = 3 007, p = 24.

Lo que no está hecho: `R = 1`, así que diferencias de pocos puntos son ruido
Monte Carlo; no se probó el bloque conjunto parcial (3–4 scores activos juntos,
el resto univariado) ni separar las variables del gating de las de la
regresión, que son las dos variantes que discutimos como más prometedoras.

### Variante bloque conjunto parcial (ξ₁..ξ₄ juntos, ξ₅..ξ₁₀ univariados; 6 000 iteraciones)

Bloque A, razón de MAE contra el FAR(2) y % de ventanas ganadas en MAE (test, w = 30):

| caso | PSBPM-FD bloque | conjunto K | univariado (corrida viva) |
|---|---|---|---|
| 200 TAR | 0.87 (0 %) | 0.93 (0 %) | **0.79** (100 %) |
| 201 GARCH | 0.99 (39 %) | 1.00 (15 %) | 0.99 (45 %) |
| 202 MULT | **0.97** (65 %) | 1.00 (7 %) | 0.98 (21 %) |

Bloque B, Winkler y % de ventanas ganadas en Winkler:

| caso | FAR gaussiana | PSBPM-FD bloque | conjunto K | univariado |
|---|---|---|---|---|
| 200 TAR | 4.98 | 4.48 (0 %) | 4.48 (0 %) | **3.91** (100 %) |
| 201 GARCH | 4.33 | 4.25 (38 %) | 4.32 (24 %) | 4.23 (28 %) |
| 202 MULT | 4.24 | **4.02** (79 %) | 4.19 (1 %) | 4.11 (17 %) |

- **202**: el bloque es el mejor modelo en los dos bloques (MAE 2,7 % bajo el FAR
  y 1,2 % bajo el univariado; Winkler 2 % mejor que el univariado, 79 % de las
  ventanas). Es el único escenario con régimen compartido por construcción, y
  es donde compartir el gating entre los scores activos rinde.
- **200**: el bloque recupera la mitad de la ventaja del univariado (0.87 contra
  0.79 y 0.93 del conjunto completo). Los regímenes siguen siendo independientes
  por score y el gating compartido no los representa.
- **201**: empate a tres (0.99) en la media; en Winkler el bloque y el univariado
  empatan (4.25 contra 4.23) y los dos mejoran al conjunto completo.
- **Convergencia**: los seis univariados convergen en los tres casos; el bloque
  conjunto converge en 201 (R-hat ≤ 1.05) pero **no en 200 ni en 202** (R-hat
  2.4 y 2.3 en `mse_in`): mismo problema de modos que el conjunto completo en
  la 200. Las cifras de 200 y 202 bloque arrastran cadenas en modos distintos.
- Costo: 63 jobs (3 × (1 bloque + 6 univariados) × 3 cadenas); el bloque
  (q = 4, p = 20) tarda lo mismo que el conjunto completo, los univariados
  (q = 1, p = 2) ~3 min cada uno. Con 8 workers la máquina (14 GB) entró en
  intercambio a disco; se corrió con 4 (`saltar_existentes = true` reanuda).
