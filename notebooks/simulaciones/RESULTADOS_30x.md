# Resultados de las 30x (300, 301, 302)

Bitácora de cifras, movida desde `CLAUDE.md` el 2026-10-12. Todas en test, `w = 30`, sobre las
salidas de `_04`/`_05` previas a `W_REF_DOC = 40` y al §10 sobre todo el test; se actualizan al
reejecutar. `CLAUDE.md` §7 conserva sólo la lectura.

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

