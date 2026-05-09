# Ideas y decisiones del modelo

## Decisiones clave
- Función de pesos: norma L1 (preserva conjugacy del paper original)
- Átomos: prior MNIW con parámetros fijos (evita romper conjugacy)
- Jerarquía: solo prior sobre M_0 (única capa con posterior analítica)

## Pendientes
- [ ] Variable selection (SSVS) en v2
- [ ] Implementación C++ del inner loop del Gibbs
- [ ] Inferencia variacional (mean-field) en v3
