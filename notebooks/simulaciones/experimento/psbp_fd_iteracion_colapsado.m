% psbp_fd_iteracion_colapsado.m — EXPERIMENTO [EXP-1] sobre la corrida 107
%
% Paso MATLAB del ciclo Python -> MATLAB -> Python. Reentrena el punto M de la
% 107 con psbp_train_colapsado.m (Gamma con psi integrado) sobre EXACTAMENTE
% los mismos datasets, hiperparametros, mcmc_config y semillas de la 107.
%
% NO toca la 107: su contrato y sus datasets se LEEN de escenario_107_1_r01_mNN;
% las trazas se escriben en experimento_107gc_1_r01_mNN/, con el nombre de
% traza del estudio (chain_fpc_<idx>_iter<chain>.mat), para que la evaluacion
% use el mismo lector que la 107.
%
% Despues: exp107gc_04_evaluacion.ipynb (contraste contra las trazas de la 107).

clear; clc; close all;

% ════════════════════════════════════════════════════════════════════════════
% 1. CONFIGURACIÓN — debe coincidir con exp107gc_04_evaluacion.ipynb
% ════════════════════════════════════════════════════════════════════════════
cfg.basename_fuente  = "escenario_107";
cfg.basename_destino = "experimento_107gc";
cfg.escenario        = 1;
cfg.replica          = 1;
cfg.M                = 4;
N_WORKERS            = 8;

% ════════════════════════════════════════════════════════════════════════════
% 2. JOBS (cadena x componente), leidos del contrato de la 107
% ════════════════════════════════════════════════════════════════════════════
[jobs, pf, pd] = jobs_psbp_107(cfg);
n_jobs = numel(jobs);
fprintf("fuente : %s\ndestino: %s\n%d jobs\n\n", pf.experiment_id, pd.experiment_id, n_jobs);

% ════════════════════════════════════════════════════════════════════════════
% 3. POOL Y PARFOR
% ════════════════════════════════════════════════════════════════════════════
pool = gcp('nocreate');
if isempty(pool)
    pool = parpool('local', min(N_WORKERS, n_jobs));
end

t_total = tic;
parfor i = 1:n_jobs
    job_i = jobs{i};
    fprintf("[job %d/%d] fpc_%d iter%02d seed=%d\n", i, n_jobs, job_i.fpc_idx, job_i.chain, job_i.seed);
    psbp_train_colapsado(job_i.y, job_i.Xnoint, job_i.hp, job_i.mcmc, ...
                         job_i.out_path, job_i.feature_names, job_i.seed);
end
fprintf("\n✅ %d jobs en %.1f min -> %s\n", n_jobs, toc(t_total) / 60, pd.out_artefact);
fprintf("Continuar con exp107gc_04_evaluacion.ipynb\n");
