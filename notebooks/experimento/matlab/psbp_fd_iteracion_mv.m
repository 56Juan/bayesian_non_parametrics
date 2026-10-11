function psbp_fd_iteracion_mv(jobs_json, n_workers, nsim_override, max_chains, saltar_existentes)
% PSBP_FD_ITERACION_MV  Paso 2 del ciclo Python -> MATLAB -> Python (conjunto).
%
%   psbp_fd_iteracion_mv(jobs_json, n_workers, nsim_override)
%
% Lee la lista de contratos que escribio Python (`jobs_mv.json`: una entrada
% por EXPERIMENT_ID con rutas absolutas a hyperparameters.json y al dataset de
% TRAIN), arma una lista plana de jobs (experimento x cadena) y la reparte con
% parfor. Entrena SOLO con el bloque de entrenamiento. hyperparameters.json es
% la unica fuente de verdad del contrato: nsim, burn, N, M, n_iter, seed_base
% e hiperparametros salen de ahi. `nsim_override` (opcional) acorta nsim para
% pruebas de humo y deja burn = floor(nsim/5); `max_chains` limita las cadenas;
% `saltar_existentes` no vuelve a entrenar los .mat que ya estan (reanudar).
%
% Salida: <out_artefact>/<out_prefix>_iter<NN>.mat  (out_prefix opcional, 'chain_mv')

    if nargin < 2 || isempty(n_workers), n_workers = 8; end
    if nargin < 3, nsim_override = []; end
    if nargin < 4 || isempty(max_chains), max_chains = Inf; end
    if nargin < 5 || isempty(saltar_existentes), saltar_existentes = false; end
    this_dir = fileparts(mfilename('fullpath'));
    addpath(this_dir);

    lista = jsondecode(fileread(jobs_json));
    if isstruct(lista), lista = num2cell(lista); end
    jobs = {};
    for i = 1:numel(lista)
        L = lista{i};  if iscell(L), L = L{1}; end
        hp_json = jsondecode(fileread(L.hyperparameters));
        prefijo = 'chain_mv';  if isfield(L, 'out_prefix'), prefijo = L.out_prefix; end
        Tbl = readtable(L.dataset_train);
        q = hp_json.q;  p = hp_json.p;
        Y = table2array(Tbl(:, 2:1+q));
        Xn = table2array(Tbl(:, 2+q:1+q+p));
        assert(size(Xn, 2) == p, 'el dataset no tiene p = %d predictores', p);
        fnames = strjoin(Tbl.Properties.VariableNames(2+q:1+q+p), ',');

        hp = struct();
        hp.ag = hp_json.global.ag;  hp.bg = hp_json.global.bg;
        hp.mumu = hp_json.global.mumu;  hp.taumu = hp_json.global.taumu;  hp.pwj = hp_json.global.pwj;
        hp.nu0 = hp_json.global.nu0;  hp.S0 = diag(hp_json.global.S0_diag(:));
        hp.apij = hp_json.por_predictor.apij(:);  hp.bpij = hp_json.por_predictor.bpij(:);
        hp.mupsij = hp_json.por_predictor.mupsij(:);  hp.taupsij = hp_json.por_predictor.taupsij(:);

        mc = struct();
        mc.nsim = hp_json.mcmc_config.nsim;  mc.burn = hp_json.mcmc_config.burn;
        mc.N = hp_json.mcmc_config.N;  mc.M = hp_json.mcmc_config.M;
        if ~isempty(nsim_override), mc.nsim = nsim_override; mc.burn = floor(nsim_override/5); end

        fprintf('-- %s : n=%d q=%d p=%d  nsim=%d burn=%d N=%d G*=%d  cadenas=%d\n', ...
            L.experiment_id, size(Y,1), q, p, mc.nsim, mc.burn, mc.N, mc.M, hp_json.n_iter);
        for chain = 1:min(hp_json.n_iter, max_chains)
            job = struct('eid', L.experiment_id, 'chain', chain, 'Y', Y, 'Xn', Xn, 'hp', hp, 'mc', mc, ...
                         'fnames', fnames, 'seed', hp_json.seed_base + chain*9973, ...
                         'out', fullfile(L.out_artefact, sprintf('%s_iter%02d.mat', prefijo, chain)));
            if saltar_existentes && isfile(job.out), fprintf('   ya existe, se salta: %s\n', job.out); continue; end
            jobs{end+1} = job; %#ok<AGROW>
        end
    end
    n_jobs = numel(jobs);
    fprintf('%d jobs en total\n', n_jobs);

    pool = gcp('nocreate');
    if isempty(pool) && n_workers > 1, parpool('local', min(n_workers, n_jobs)); end
    t0 = tic;
    if n_workers > 1
        parfor i = 1:n_jobs
            j = jobs{i};
            fprintf('[job %d/%d] %s cadena %d seed %d\n', i, n_jobs, j.eid, j.chain, j.seed);
            psbp_train_mv(j.Y, j.Xn, j.hp, j.mc, j.out, j.fnames, j.seed);
        end
    else
        for i = 1:n_jobs
            j = jobs{i};
            fprintf('[job %d/%d] %s cadena %d seed %d\n', i, n_jobs, j.eid, j.chain, j.seed);
            psbp_train_mv(j.Y, j.Xn, j.hp, j.mc, j.out, j.fnames, j.seed);
        end
    end
    fprintf('Entrenamiento completo: %d jobs en %.1f min\n', n_jobs, toc(t0)/60);
end
