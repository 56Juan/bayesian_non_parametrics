% psbp_fd_iteracion.m
% Script principal con parfor plano sobre todos los jobs (tt × componente).
% Estructura: pre-carga todos los datasets, construye lista de jobs,
% luego lanza parfor único sobre n_iter × n_components cadenas.

clear; clc; close all;

% ════════════════════════════════════════════════════════════════════════════
% 1. CONFIGURACIÓN
% ════════════════════════════════════════════════════════════════════════════

% [FIX] N_ITER y la configuración MCMC ya NO se definen aquí: se leen de
% hyperparameters.json (generado por 02_01_simulaciones), que es la única
% fuente de verdad del contrato Python ↔ MATLAB.
N_WORKERS = 8;     % workers del pool (ajusta según carga del sistema)
SEED_BASE = 4123;  % semilla base; cada job usa SEED_BASE + tt*9973 + k*31

TT       = 3;
BASENAME = "modelo_unificado";
paths    = config_paths(BASENAME, TT, SEED_BASE);

% ════════════════════════════════════════════════════════════════════════════
% 2. LEER HIPERPARÁMETROS Y MANIFEST
% ════════════════════════════════════════════════════════════════════════════

hp_path = fullfile(paths.out_artefact, "hyperparameters.json");
assert(isfile(hp_path), "No se encontró hyperparameters.json en: %s", hp_path);
hp_json = jsondecode(fileread(hp_path));

assert(isfield(hp_json, "n_iter"), ...
    "hyperparameters.json no contiene 'n_iter'; regenera con 02_01.");
assert(isfield(hp_json, "mcmc_config"), ...
    "hyperparameters.json no contiene 'mcmc_config'; regenera con 02_01.");
N_ITER    = hp_json.n_iter;
mcmc.nsim = hp_json.mcmc_config.nsim;
mcmc.burn = hp_json.mcmc_config.burn;
mcmc.N    = hp_json.mcmc_config.N;
mcmc.M    = hp_json.mcmc_config.M;

fprintf("════════════════════════════════════════════════════════\n");
fprintf("  N_ITER=%d   N_WORKERS=%d   SEED_BASE=%d\n", ...
        N_ITER, N_WORKERS, SEED_BASE);
fprintf("  MCMC: nsim=%d burn=%d N=%d M=%d  (desde hyperparameters.json)\n\n", ...
        mcmc.nsim, mcmc.burn, mcmc.N, mcmc.M);

manifest_path = fullfile(paths.functional, "datasets_manifest.json");
assert(isfile(manifest_path), "No se encontró datasets_manifest.json en: %s", manifest_path);
manifest      = jsondecode(fileread(manifest_path));
component_idx = manifest.component_idx;
n_components  = numel(hp_json.hyperparams_list);

fprintf("✓ Configuración cargada  (n_components=%d, n_lags=%d)\n\n", ...
        n_components, manifest.n_lags);

% ════════════════════════════════════════════════════════════════════════════
% 3. PRE-CARGAR TODOS LOS DATASETS EN MEMORIA
%    Mejor práctica: leer archivos una sola vez en el proceso principal,
%    luego serializar las matrices a los workers (evita I/O concurrente).
% ════════════════════════════════════════════════════════════════════════════

fprintf("Cargando datasets en memoria...\n");
Y_cell         = cell(n_components, 1);   % respuestas
X_cell         = cell(n_components, 1);   % predictores
fname_cell     = cell(n_components, 1);   % feature_names
fpc_idx_vec    = zeros(n_components, 1);  % índice fpc (base-1)

for k = 1:n_components
    fpc_idx          = component_idx(k) + 1;
    fpc_idx_vec(k)   = fpc_idx;
    fpath            = fullfile(paths.functional, ...
                           sprintf("dataset_fpc_%d.csv", fpc_idx));
    assert(isfile(fpath), "No se encontró dataset: %s", fpath);

    Tbl              = readtable(fpath);
    dt               = table2array(Tbl);
    Y_cell{k}        = dt(:, 1);
    X_cell{k}        = dt(:, 2:end);
    fname_cell{k}    = strjoin(Tbl.Properties.VariableNames(2:end), ",");

    fprintf("  fpc_%d  T_eff=%d  p=%d\n", fpc_idx, size(dt,1), size(dt,2)-1);
end
fprintf("\n");

% ════════════════════════════════════════════════════════════════════════════
% 4. CONSTRUIR LISTA PLANA DE JOBS
%    Cada job es un struct con todo lo necesario para una cadena MCMC.
%    Orden: [tt=1,k=1], [tt=1,k=2], ..., [tt=N_ITER,k=n_components]
% ════════════════════════════════════════════════════════════════════════════

n_jobs = N_ITER * n_components;
jobs   = cell(n_jobs, 1);
idx    = 0;

for tt = 1:N_ITER
    for k = 1:n_components
        idx      = idx + 1;
        hp_k     = hp_json.hyperparams_list(k).hyperparams;

        job.tt           = tt;
        job.k            = k;
        job.fpc_idx      = fpc_idx_vec(k);
        job.seed         = SEED_BASE + tt * 9973 + k * 31;  % semilla única por job
        job.y            = Y_cell{k};
        job.Xnoint       = X_cell{k};
        job.feature_names = fname_cell{k};

        % hiperparámetros globales
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

        % ruta de salida: chain_fpc_<k>_iter<tt>.mat
        job.out_path = fullfile(paths.out_artefact, ...
            sprintf("chain_fpc_%d_iter%02d.mat", fpc_idx_vec(k), tt));

        jobs{idx} = job;
    end
end

fprintf("✓ %d jobs construidos (%d iter × %d componentes)\n\n", ...
        n_jobs, N_ITER, n_components);

% ════════════════════════════════════════════════════════════════════════════
% 5. POOL DE WORKERS
% ════════════════════════════════════════════════════════════════════════════

pool = gcp('nocreate');
if isempty(pool)
    pool = parpool('local', N_WORKERS);
    fprintf("✓ Pool iniciado con %d workers\n\n", pool.NumWorkers);
else
    fprintf("✓ Pool ya activo con %d workers\n\n", pool.NumWorkers);
end

% ════════════════════════════════════════════════════════════════════════════
% 6. PARFOR PLANO — distribuye los  jobs entre  workers
% ════════════════════════════════════════════════════════════════════════════

fprintf("Iniciando parfor (%d jobs, %d workers)...\n\n", n_jobs, N_WORKERS);
t_total = tic;

mcmc_par = mcmc;   % copia explícita para serialización parfor

parfor i = 1:n_jobs
    job_i = jobs{i};
    fprintf("[job %2d/%d]  fpc_%d  iter%02d  seed=%d\n", ...
            i, n_jobs, job_i.fpc_idx, job_i.tt, job_i.seed);
    psbp_train(job_i.y, job_i.Xnoint, job_i.hp, mcmc_par, ...
               job_i.out_path, job_i.feature_names, job_i.seed);
end

elapsed = toc(t_total);
fprintf("\n════════════════════════════════════════════════════════\n");
fprintf("✅ Entrenamiento completo\n");
fprintf("   Jobs: %d   Tiempo total: %.1f min   (%.1f min/job promedio)\n", ...
        n_jobs, elapsed/60, elapsed/60/N_WORKERS);
fprintf("   Archivos en: %s\n", paths.out_artefact);
