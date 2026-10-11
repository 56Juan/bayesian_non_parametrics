# Insumos reales del experimento

`eolica_onshore_DE_smard.npz`: generacion eolica onshore de Alemania, SMARD
(Bundesnetzagentur, https://www.smard.de, licencia CC BY 4.0, filtro 4067,
resolucion cuarto de hora), 2015-01-01 a 2026-10-08, dias UTC de 96 puntos.
`X` (T, 96) es el factor de planta: MW sobre el maximo movil de 365 dias de la
serie diaria de maximos; `grilla` en [0, 1]; `fechas` ISO. Dias con mas de 5 %
de faltantes descartados y huecos menores interpolados (ninguno en la racha
usada). Descargado el 2026-10-08 con `preparar.py smard` (scratchpad de la
sesion); no se versiona el crudo semanal.
