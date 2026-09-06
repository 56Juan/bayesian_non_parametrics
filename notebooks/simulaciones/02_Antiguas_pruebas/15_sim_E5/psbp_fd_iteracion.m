% psbp_fd_iteracion.m — Escenario 5 (Algoritmo 5 del anexo), corrida 15
%
% Paso 2 del ciclo Python -> MATLAB -> Python. Lee el contrato que escribio
% 15_01_simulaciones.ipynb, arma una lista plana de jobs (cadena x componente
% FPCA) y la reparte con parfor. Entrena SOLO con el bloque de entrenamiento.
%
% Ejes del estudio:
%   ESCENARIO_ID : Algoritmo k del anexo.
%   REPLICA_ID   : replica Monte Carlo de la trayectoria simulada.
%   chain        : cadena MCMC dentro de un mismo conjunto de datos.
%   k            : componente FPCA.
%
% Salida: chain_fpc_<idx>_iter<chain>.mat  (nombre fijo; los notebooks de
% convergencia y evaluacion los buscan por ese patron).

clear; clc; close all;

% ════════════════════════════════════════════════════════════════════════════
% 1. CONFIGURACIÓN  — lo único que se toca a mano en este archivo
% ════════════════════════════════════════════════════════════════════════════

N_WORKERS    = 8;            % workers del pool
ESCENARIO_ID = 5;            % Algoritmo del anexo
REPLICA_ID   = 1;            % replica Monte Carlo
BASENAME     = "escenario";

% EXPERIMENT_ID incluye la replica: <basename>_<escenario>_r<replica a 2 digitos>.
% Debe coincidir EXACTAMENTE con el de 15_01_simulaciones.ipynb.
EXPERIMENT_ID = sprintf("%s_%d_r%02d", BASENAME, ESCENARIO_ID, REPLICA_ID);
paths         = config_paths(EXPERIMENT_ID);

% ════════════════════════════════════════════════════════════════════════════
% 2. LEER HIPERPARÁMETROS Y MANIFEST
%    hyperparameters.json es la unica fuente de verdad del contrato con Python.
% ════════════════════════════════════════════════════════════════════════════

hp_path = fullfile(paths.out_artefact, "hyperparameters.json");
assert(isfile(hp_path), "No se encontró hyperparameters.json en: %s", hp_path);
hp_json = jsondecode(fileread(hp_path));

assert(isfield(hp_json, "n_iter"), "hyperparameters.json no contiene 'n_iter'.");
assert(isfield(hp_json, "mcmc_config"), "hyperparameters.json no contiene 'mcmc_config'.");

N_ITER    = hp_json.n_iter;
mcmc.nsim = hp_json.mcmc_config.nsim;
mcmc.burn = hp_json.mcmc_config.burn;
mcmc.N    = hp_json.mcmc_config.N;
mcmc.M    = hp_json.mcmc_config.M;

% [FIX] SEED_BASE se LEE del JSON. En las corridas 03-10 estaba escrita a mano
% en este archivo (4123) mientras el JSON registraba otra (41232), de modo que
% la semilla documentada no reproducia la corrida y ningun resultado era
% reproducible a partir de los artefactos. Ahora hay una sola fuente.
assert(isfield(hp_json, "seed_base"), ...
    "hyperparameters.json no contiene 'seed_base'; regenera con 11_01.");
SEED_BASE = hp_json.seed_base;

fprintf("════════════════════════════════════════════════════════\n");
fprintf("  EXPERIMENT_ID = %s\n", EXPERIMENT_ID);
fprintf("  N_ITER=%d   N_WORKERS=%d   SEED_BASE=%d  (desde JSON)\n", ...
        N_ITER, N_WORKERS, SEED_BASE);
fprintf("  MCMC: nsim=%d burn=%d N=%d M=%d\n\n", ...
        mcmc.nsim, mcmc.burn, mcmc.N, mcmc.M);

manifest_path = fullfile(paths.functional, "datasets_manifest.json");
assert(isfile(manifest_path), "No se encontró datasets_manifest.json en: %s", manifest_path);
manifest      = jsondecode(fileread(manifest_path));
component_idx = manifest.component_idx;
n_components  = numel(hp_json.hyperparams_list);

fprintf("✓ Configuración cargada  (n_components=%d, n_lags=%d)\n\n", ...
        n_components, manifest.n_lags);

% ════════════════════════════════════════════════════════════════════════════
% 3. PRE-CARGAR LOS DATASETS
%    Se leen una sola vez en el proceso principal y se serializan a los
%    workers, en lugar de que cada worker abra los mismos archivos.
% ════════════════════════════════════════════════════════════════════════════

fprintf("Cargando datasets en memoria...\n");
Y_cell      = cell(n_components, 1);
X_cell      = cell(n_components, 1);
fname_cell  = cell(n_components, 1);
fpc_idx_vec = zeros(n_components, 1);

for k = 1:n_components
    fpc_idx        = component_idx(k) + 1;   % component_idx es base-0
    fpc_idx_vec(k) = fpc_idx;

    % Solo el bloque de ENTRENAMIENTO. El de prueba queda reservado a Python.
    fpath = fullfile(paths.functional, sprintf("dataset_fpc_%d_train.csv", fpc_idx));
    assert(isfile(fpath), "No se encontró dataset: %s", fpath);

    Tbl           = readtable(fpath);
    dt            = table2array(Tbl);
    Y_cell{k}     = dt(:, 1);
    X_cell{k}     = dt(:, 2:end);
    fname_cell{k} = strjoin(Tbl.Properties.VariableNames(2:end), ",");

    fprintf("  fpc_%d  T_eff=%d  p=%d\n", fpc_idx, size(dt,1), size(dt,2)-1);
end
fprintf("\n");

% ════════════════════════════════════════════════════════════════════════════
% 4. LISTA PLANA DE JOBS
% ════════════════════════════════════════════════════════════════════════════

n_jobs = N_ITER * n_components;
jobs   = cell(n_jobs, 1);
idx    = 0;

for chain = 1:N_ITER
    for k = 1:n_components
        idx  = idx + 1;
        hp_k = hp_json.hyperparams_list(k).hyperparams;

        job.chain         = chain;
        job.escenario     = ESCENARIO_ID;
        job.replica       = REPLICA_ID;
        job.k             = k;
        job.fpc_idx       = fpc_idx_vec(k);
        job.seed          = SEED_BASE + chain * 9973 + k * 31;   % única por job
        job.y             = Y_cell{k};
        job.Xnoint        = X_cell{k};
        job.feature_names = fname_cell{k};

        job.hp.atau    = hp_json.global.atau;
        job.hp.btau    = hp_json.global.btau;
        job.hp.ag      = hp_json.global.ag;
        job.hp.bg      = hp_json.global.bg;
        job.hp.mumu    = hp_json.global.mumu;
        job.hp.taumu   = hp_json.global.taumu;
        job.hp.pwj     = hp_json.global.pwj;
        job.hp.apij    = hp_k.apij(:);
        job.hp.bpij    = hp_k.bpij(:);
        job.hp.mupsij  = hp_k.mupsij(:);
        job.hp.taupsij = hp_k.taupsij(:);

        job.out_path = fullfile(paths.out_artefact, ...
            sprintf("chain_fpc_%d_iter%02d.mat", fpc_idx_vec(k), chain));

        jobs{idx} = job;
    end
end

fprintf("✓ %d jobs construidos (%d cadenas × %d componentes)\n\n", ...
        n_jobs, N_ITER, n_components);

% ════════════════════════════════════════════════════════════════════════════
% 5. POOL Y PARFOR
% ════════════════════════════════════════════════════════════════════════════

pool = gcp('nocreate');
if isempty(pool)
    pool = parpool('local', N_WORKERS);
    fprintf("✓ Pool iniciado con %d workers\n\n", pool.NumWorkers);
else
    fprintf("✓ Pool ya activo con %d workers\n\n", pool.NumWorkers);
end

fprintf("Iniciando parfor (%d jobs, %d workers)...\n\n", n_jobs, N_WORKERS);
t_total  = tic;
mcmc_par = mcmc;   % copia explícita para la serialización de parfor

parfor i = 1:n_jobs
    job_i = jobs{i};
    fprintf("[job %2d/%d]  fpc_%d  iter%02d  seed=%d\n", ...
            i, n_jobs, job_i.fpc_idx, job_i.chain, job_i.seed);
    psbp_train(job_i.y, job_i.Xnoint, job_i.hp, mcmc_par, ...
               job_i.out_path, job_i.feature_names, job_i.seed);
end

elapsed = toc(t_total);
fprintf("\n════════════════════════════════════════════════════════\n");
fprintf("✅ Entrenamiento completo\n");
fprintf("   Jobs: %d   Tiempo total: %.1f min\n", n_jobs, elapsed/60);
fprintf("   Por job: %.2f min de pared   |   %.2f min de CPU\n", ...
        (elapsed/60)/n_jobs, (elapsed/60)*N_WORKERS/n_jobs);
fprintf("   Archivos en: %s\n", paths.out_artefact);
