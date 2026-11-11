# PSBPM-FD: PSBP extendido a series de tiempo funcionales

Extensión del **Probit Stick-Breaking Process** (Chung & Dunson, 2009) a la predicción de series de tiempo funcionales. El paquete `model_psbp_fd` genera los datos de simulación, prepara la representación funcional, consume las trazas MCMC y evalúa las predicciones.

**El muestreo MCMC ocurre en MATLAB**, no en Python. Las etapas se acoplan por archivos en disco.

## Flujo de trabajo

Cada corrida es una carpeta de `notebooks/` con cuatro o cinco notebooks y un paso MATLAB.

| Paso | Archivo | Qué hace |
|---|---|---|
| 1 | `NNN_01_*.ipynb` | Genera o carga los datos, representación funcional, FPCA, datasets y contrato (`hyperparameters.json`). |
| 2 | `psbp_fd_iteracion.m` | Lee el contrato y entrena con `psbp_train.m`. Escribe las trazas `.mat`. |
| 3 | `NNN_03_convergencia.ipynb` | Diagnósticos de convergencia de las cadenas. |
| 4 | `NNN_04_evaluacion.ipynb` | Predictiva del modelo y las diez métricas sobre la ventana móvil. |
| 5 | `NNN_05_comparacion.ipynb` | Comparación contra FAR y Random Forest. |
| 6 | `NNN_06_opciones.ipynb` | Comportamientos predictivos como opciones de pronóstico (sólo en 300–302 y 101–107). |

## Instalación

Requiere **Python ≥ 3.11**. Los comandos se ejecutan desde la raíz del repositorio.

```bash
python -m venv .venv
```

Activar el entorno:

```bash
# Windows, PowerShell
.venv\Scripts\Activate.ps1

# Windows, Git Bash
source .venv/Scripts/activate

# Linux / macOS
source .venv/bin/activate
```

Instalar el paquete en modo editable, con sus dependencias (`numpy`, `scipy`, `pandas`, `scikit-fda`, `scikit-learn`, `matplotlib`, `seaborn`, `jinja2`, `openpyxl`):

```bash
pip install -e .
```

Para correr las pruebas hace falta además `pytest`:

```bash
pip install -e ".[dev]"
```

Verificar la instalación:

```bash
python -c "import model_psbp_fd; print(model_psbp_fd.__version__, model_psbp_fd.__file__)"
```

La ruta impresa debe ser la de este repositorio. Si apunta a otra copia, reinstalar con `pip install -e .` desde la raíz.

Hay una alternativa con conda en `environment.yml` (entorno `psbp_fd`). Para ejecutar los `.m` se necesita MATLAB por separado.

## Pruebas

```bash
python -m pytest tests -q
```

Cubren las métricas de los Bloques A y B, el FAR, los predictores puntuales, los intervalos y los comportamientos predictivos.

## Estructura

```
├── model_psbp_fd/              Paquete
│   ├── pipelines/              Generadores de simulación y contrato de artefactos
│   │   ├── sim_escenario_TAR.py, sim_escenario_GARCH.py, sim_escenario_C1.py
│   │   ├── sim_scores_comun.py, sim_comun.py
│   │   ├── artifacts.py        Escritura y lectura emparejadas de artefactos
│   │   └── deprecated/         Generadores de corridas retiradas
│   ├── functions_models/       Representación funcional, FPCA y estandarizador
│   ├── models/
│   │   ├── pspb_fd_v3/         Versión en uso: trazas → predictiva → curvas
│   │   └── psbp_fd_v1/, psbp_fd_v2/   Versiones anteriores, sin uso
│   ├── fit/                    Métricas, ventana móvil, convergencia, comparación
│   ├── graphics/               Figuras
│   └── utils/                  Rutas, cuadratura, álgebra lineal, trazas
│
├── notebooks/
│   ├── simulaciones/
│   │   ├── 08_02 3 escenarios/ 200 (TAR), 201 (GARCH), 202 (multimodalidad)
│   │   ├── 300–302_*           Reevaluación de 200–202 con todos los predictores e intervalos
│   │   └── 00_* … 07_*         Corridas históricas
│   ├── reales/
│   │   ├── 101–107_*           Tráfico I-94, precio SA1, PM2.5, Beijing, eólica, BTC
│   │   └── 01 …, 02 Antiguos … Corridas históricas
│   └── auxiliar/               Descarga y consolidado de datos de Binance
│
├── data/<dominio>/             raw/, processed/functional/, processed/predict/
├── artefact/<dominio>/         hyperparameters.json, eval_config.json, trazas .mat
├── reports/<dominio>/          Figuras y tablas por corrida
├── docs/                       Tesis en LaTeX (compila en Overleaf)
├── references/                 Código y notas de referencia de Chung & Dunson
├── tests/
├── pyproject.toml
└── environment.yml
```

`<dominio>` es `simulaciones` o `reales`. Cada corrida usa el mismo identificador en `data/`, `artefact/` y `reports/`; lo construye `model_psbp_fd/utils/rutas.py`, y `config_paths.m` es su gemelo en MATLAB.

## Referencia

Chung, Y. y Dunson, D. B. (2009). Nonparametric Bayes conditional distribution modeling with variable selection. *Journal of the American Statistical Association*, 104(488), 1646–1660.
