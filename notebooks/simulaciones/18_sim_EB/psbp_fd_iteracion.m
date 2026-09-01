% psbp_fd_iteracion.m - Escenario B (FAR con signo conmutado), corrida 18
%
% Paso 2 del ciclo Python -> MATLAB -> Python. Lee el contrato que escribio
% 18_01_simulaciones.ipynb, arma una lista plana de jobs y la reparte con
% parfor. Entrena SOLO con el bloque de entrenamiento.
%
% BARRIDO EN M. A diferencia de las corridas 11-16, este archivo NO entrena un
% solo M: recorre M_FPCA_LIST y arma una unica lista de jobs
%
%     (M) x (cadena) x (componente FPCA)
%
% de modo que un solo parfor cubre todo el barrido. Cada M es un EXPERIMENT_ID
% distinto -escenario_B_r01_m01, _m02, _m03- con sus propios datos, contrato y
% trazas, asi que los jobs de distintos M no comparten nada mas que el pool: no
% hay riesgo de que uno pise los artefactos del otro.
%
% Por que en una sola lista y no un bucle de parfor por M: el ultimo M del
% barrido tiene mas jobs que el primero (uno por componente FPCA), y con un
% parfor por M los workers quedan ociosos al final de cada tanda. Con la lista
% plana el balanceo es global.
%
% Ejes del estudio:
%   ESCENARIO_ID : identificador del escenario. En esta corrida NO es un entero:
%                  vale "B" y por eso el sprintf del EXPERIMENT_ID usa %s y no %d,
%                  igual que en la corrida 17. Si se cambia a un numero hay que
%                  cambiar tambien ese formato.
%   REPLICA_ID   : replica Monte Carlo de la trayectoria simulada.
%   M            : numero de componentes FPCA retenidas (eje de barrido).
%   chain        : cadena MCMC dentro de un mismo conjunto de datos.
%   k            : componente FPCA.
%
% Salida: <artefact>/<EXPERIMENT_ID>/chain_fpc_<idx>_iter<chain>.mat
% (nombre fijo; los notebooks de convergencia y evaluacion los buscan asi).

clear; clc; close all;

% ════════════════════════════════════════════════════════════════════════════
% 1. CONFIGURACIÓN  — lo único que se toca a mano en este archivo
% ════════════════════════════════════════════════════════════════════════════

N_WORKERS    = 8;            % workers del pool
ESCENARIO_ID = "B";          % escenario de diagnostico (no es Algoritmo del anexo)
REPLICA_ID   = 1;            % replica Monte Carlo
BASENAME     = "escenario";

% [BARRIDO] componentes FPCA. DEBE coincidir con el M_FPCA_LIST de 18_01,
% 18_03, 18_04 y 18_05. Cada valor tiene que estar procesado por 18_01 antes
% de correr esto.
M_FPCA_LIST  = [1 2];

% Con true, un M sin artefactos de 18_01 se SALTA con aviso en vez de abortar
% todo el barrido. Con false, la falta de artefactos es un error.
SALTAR_M_SIN_ARTEFACTOS = true;

% ════════════════════════════════════════════════════════════════════════════
% 2. LEER EL CONTRATO DE CADA M Y PRE-CARGAR LOS DATASETS
%    hyperparameters.json es la unica fuente de verdad del contrato con Python:
%    de ahi salen n_iter, mcmc_config, seed_base y los hiperparametros por
%    componente. Nada de eso se escribe a mano en este archivo.
% ════════════════════════════════════════════════════════════════════════════

M_FPCA_LIST = unique(M_FPCA_LIST(:))';
assert(~isempty(M_FPCA_LIST), "M_FPCA_LIST esta vacio.");
assert(all(M_FPCA_LIST >= 1), "M debe ser >= 1.");

fprintf("════════════════════════════════════════════════════════\n");
fprintf("  BARRIDO EN M : [%s]\n", strjoin(string(M_FPCA_LIST), " "));
fprintf("  N_WORKERS    : %d\n", N_WORKERS);
fprintf("════════════════════════════════════════════════════════\n\n");

jobs      = {};
n_saltados = 0;

for M_FPCA = M_FPCA_LIST

    % EXPERIMENT_ID incluye la replica y M:
    %   <basename>_<escenario>_r<replica a 2 digitos>_m<M a 2 digitos>
    % Debe coincidir EXACTAMENTE con el de 18_01_simulaciones.ipynb. M viaja en
    % el ID porque cada valor del barrido escribe sus propios artefactos.
    EXPERIMENT_ID = sprintf("%s_%s_r%02d_m%02d", ...
                            BASENAME, ESCENARIO_ID, REPLICA_ID, M_FPCA);
    paths         = config_paths(EXPERIMENT_ID);

    hp_path       = fullfile(paths.out_artefact, "hyperparameters.json");
    manifest_path = fullfile(paths.functional, "datasets_manifest.json");

    if ~isfile(hp_path) || ~isfile(manifest_path)
        % OJO ["a" "b"] es un ARRAY de strings 1x2, no una concatenacion, y
        % sprintf/fprintf/assert exigen un formato UNICO. La concatenacion de
        % literales multilinea en MATLAB va con comillas simples: ['a' 'b'].
        msg = sprintf(['M=%d (%s): faltan artefactos de 18_01.\n' ...
                       '    hyperparameters.json: %d\n' ...
                       '    datasets_manifest.json: %d'], ...
                      M_FPCA, EXPERIMENT_ID, isfile(hp_path), isfile(manifest_path));
        if SALTAR_M_SIN_ARTEFACTOS
            fprintf("⚠ %s\n  -> saltado\n\n", msg);
            n_saltados = n_saltados + 1;
            continue
        else
            error("%s", msg);
        end
    end

    hp_json = jsondecode(fileread(hp_path));

    assert(isfield(hp_json, "n_iter"), ...
        "[M=%d] hyperparameters.json no contiene 'n_iter'.", M_FPCA);
    assert(isfield(hp_json, "mcmc_config"), ...
        "[M=%d] hyperparameters.json no contiene 'mcmc_config'.", M_FPCA);

    % [FIX] SEED_BASE se LEE del JSON. En las corridas 03-10 estaba escrita a
    % mano en este archivo (4123) mientras el JSON registraba otra (41232), de
    % modo que la semilla documentada no reproducia la corrida y ningun
    % resultado era reproducible a partir de sus artefactos. Una sola fuente.
    assert(isfield(hp_json, "seed_base"), ...
        "[M=%d] hyperparameters.json no contiene 'seed_base'; regenera con 18_01.", ...
        M_FPCA);
    SEED_BASE = hp_json.seed_base;

    N_ITER    = hp_json.n_iter;
    mcmc_M.nsim = hp_json.mcmc_config.nsim;
    mcmc_M.burn = hp_json.mcmc_config.burn;
    mcmc_M.N    = hp_json.mcmc_config.N;
    mcmc_M.M    = hp_json.mcmc_config.M;

    manifest      = jsondecode(fileread(manifest_path));
    component_idx = manifest.component_idx;
    n_components  = numel(hp_json.hyperparams_list);

    % El ID declara M; el manifest lo determina. Si discrepan, el directorio no
    % corresponde a este punto del barrido y todo lo que siga estaria mal.
    assert(n_components == M_FPCA, ...
        ['[M=%d] %s declara M=%d pero su hyperparameters.json tiene %d ' ...
         'componentes. Regenera ese punto con 18_01.'], ...
        M_FPCA, EXPERIMENT_ID, M_FPCA, n_components);
    assert(numel(component_idx) == n_components, ...
        "[M=%d] component_idx tiene %d entradas y hyperparams_list %d.", ...
        M_FPCA, numel(component_idx), n_components);

    fprintf("── M=%d  (%s) ─────────────────────────\n", M_FPCA, EXPERIMENT_ID);
    fprintf("   N_ITER=%d   SEED_BASE=%d  (desde JSON)\n", N_ITER, SEED_BASE);
    fprintf("   MCMC: nsim=%d burn=%d N=%d M=%d\n", ...
            mcmc_M.nsim, mcmc_M.burn, mcmc_M.N, mcmc_M.M);
    fprintf("   n_components=%d  n_lags=%d\n", n_components, manifest.n_lags);

    % ── datasets: se leen una sola vez en el proceso principal y se serializan
    %    a los workers, en lugar de que cada worker abra los mismos archivos.
    Y_cell      = cell(n_components, 1);
    X_cell      = cell(n_components, 1);
    fname_cell  = cell(n_components, 1);
    fpc_idx_vec = zeros(n_components, 1);

    for k = 1:n_components
        fpc_idx        = component_idx(k) + 1;   % component_idx es base-0
        fpc_idx_vec(k) = fpc_idx;

        % Solo el bloque de ENTRENAMIENTO. El de prueba queda reservado a Python.
        fpath = fullfile(paths.functional, sprintf("dataset_fpc_%d_train.csv", fpc_idx));
        assert(isfile(fpath), "[M=%d] no se encontró dataset: %s", M_FPCA, fpath);

        Tbl           = readtable(fpath);
        dt            = table2array(Tbl);
        Y_cell{k}     = dt(:, 1);
        X_cell{k}     = dt(:, 2:end);
        fname_cell{k} = strjoin(Tbl.Properties.VariableNames(2:end), ",");

        fprintf("     fpc_%d  T_eff=%d  p=%d\n", fpc_idx, size(dt,1), size(dt,2)-1);
    end

    % ── jobs de este M, agregados a la lista global
    for chain = 1:N_ITER
        for k = 1:n_components
            hp_k = hp_json.hyperparams_list(k).hyperparams;

            job = struct();
            job.experiment_id = EXPERIMENT_ID;
            job.M             = M_FPCA;
            job.chain         = chain;
            job.escenario     = ESCENARIO_ID;
            job.replica       = REPLICA_ID;
            job.k             = k;
            job.fpc_idx       = fpc_idx_vec(k);
            job.seed          = SEED_BASE + chain * 9973 + k * 31;   % única por job
            job.y             = Y_cell{k};
            job.Xnoint        = X_cell{k};
            job.feature_names = fname_cell{k};
            job.mcmc          = mcmc_M;   % viaja con el job: cada M lee el suyo

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

            jobs{end+1} = job;   %#ok<SAGROW>
        end
    end
    fprintf("   -> %d jobs (%d cadenas × %d componentes)\n\n", ...
            N_ITER * n_components, N_ITER, n_components);
end

n_jobs = numel(jobs);
assert(n_jobs > 0, ...
    ['Ningun punto del barrido tiene artefactos de 18_01. Corre ' ...
     '18_01_simulaciones.ipynb para cada M de M_FPCA_LIST.']);

% La semilla es SEED_BASE + chain*9973 + k*31, y SEED_BASE sale del JSON de
% cada M. Si dos M comparten seed_base, sus jobs comparten semilla: no es un
% error -son datos distintos- pero conviene saberlo al reportar.
seeds_por_job = cellfun(@(j) j.seed, jobs);
if numel(unique(seeds_por_job)) < n_jobs
    fprintf(['i Hay semillas repetidas ENTRE puntos del barrido (mismo ' ...
             'seed_base en\n  varios M). Los datos de cada M son distintos, ' ...
             'asi que no hay colision\n  real, pero declararlo al reportar.\n\n']);
end

fprintf("✓ %d jobs construidos en total", n_jobs);
if n_saltados > 0
    fprintf("   (%d punto(s) del barrido saltado(s))", n_saltados);
end
fprintf("\n\n");

% ════════════════════════════════════════════════════════════════════════════
% 3. POOL Y PARFOR
% ════════════════════════════════════════════════════════════════════════════

pool = gcp('nocreate');
if isempty(pool)
    pool = parpool('local', N_WORKERS);
    fprintf("✓ Pool iniciado con %d workers\n\n", pool.NumWorkers);
else
    fprintf("✓ Pool ya activo con %d workers\n\n", pool.NumWorkers);
end

fprintf("Iniciando parfor (%d jobs, %d workers)...\n\n", n_jobs, N_WORKERS);
t_total = tic;

parfor i = 1:n_jobs
    job_i = jobs{i};
    fprintf("[job %3d/%d]  M=%d  fpc_%d  iter%02d  seed=%d\n", ...
            i, n_jobs, job_i.M, job_i.fpc_idx, job_i.chain, job_i.seed);
    psbp_train(job_i.y, job_i.Xnoint, job_i.hp, job_i.mcmc, ...
               job_i.out_path, job_i.feature_names, job_i.seed);
end

elapsed = toc(t_total);
fprintf("\n════════════════════════════════════════════════════════\n");
fprintf("✅ Entrenamiento completo del barrido\n");
fprintf("   M evaluados: [%s]\n", strjoin(string(M_FPCA_LIST), " "));
fprintf("   Jobs: %d   Tiempo total: %.1f min\n", n_jobs, elapsed/60);
fprintf("   Por job: %.2f min de pared   |   %.2f min de CPU\n", ...
        (elapsed/60)/n_jobs, (elapsed/60)*N_WORKERS/n_jobs);
fprintf("\n   Archivos por punto del barrido:\n");
for M_FPCA = M_FPCA_LIST
    eid = sprintf("%s_%s_r%02d_m%02d", BASENAME, ESCENARIO_ID, REPLICA_ID, M_FPCA);
    p   = config_paths(eid);
    fprintf("     M=%d -> %s\n", M_FPCA, p.out_artefact);
end
fprintf("\n   Continuar con 18_03_convergencia, con el mismo M_FPCA_LIST.\n");
