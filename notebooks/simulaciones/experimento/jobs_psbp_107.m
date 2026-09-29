function [jobs, paths_fuente, paths_destino] = jobs_psbp_107(cfg)
% JOBS_PSBP_107  Arma la lista plana de jobs (cadena x componente) de UN punto
% del barrido de la corrida 107, leyendo su contrato SIN modificarlo, y dirige
% las trazas a un EXPERIMENT_ID propio.
%
%   cfg.basename_fuente  "escenario_107"      (contrato y datasets: solo lectura)
%   cfg.basename_destino "experimento_107gc"  (aqui se escriben las trazas)
%   cfg.escenario, cfg.replica, cfg.M
%
% Es la seccion 2 de psbp_fd_iteracion.m de la 107 para un solo M, con dos
% diferencias: (1) lee los CSV con dlmread en vez de readtable, para que corra
% tambien en Octave; (2) el out_path apunta al destino. La semilla es la misma
% convencion del estudio: seed_base + chain*9973 + k*31.

    eid_f = sprintf("%s_%d_r%02d_m%02d", cfg.basename_fuente,  cfg.escenario, cfg.replica, cfg.M);
    eid_d = sprintf("%s_%d_r%02d_m%02d", cfg.basename_destino, cfg.escenario, cfg.replica, cfg.M);
    paths_fuente  = config_paths(eid_f);
    paths_destino = config_paths(eid_d);

    hp_path       = fullfile(paths_fuente.out_artefact, "hyperparameters.json");
    manifest_path = fullfile(paths_fuente.functional, "datasets_manifest.json");
    assert(exist(hp_path, "file") == 2, "No se encontro %s.", hp_path);
    assert(exist(manifest_path, "file") == 2, "No se encontro %s.", manifest_path);

    hp_json  = jsondecode(fileread(hp_path));
    manifest = jsondecode(fileread(manifest_path));
    campos = {'n_iter', 'mcmc_config', 'seed_base', 'hyperparams_list'};
    for i = 1:numel(campos)
        assert(isfield(hp_json, campos{i}), "hyperparameters.json no contiene '%s'.", campos{i});
    end
    % "global" es palabra reservada: jsondecode (makeValidName) la entrega como
    % xGlobal. Se acepta cualquiera de las dos y se lee con getfield.
    if isfield(hp_json, 'global')
        gl = getfield(hp_json, 'global');
    elseif isfield(hp_json, 'xGlobal')
        gl = hp_json.xGlobal;
    else
        error("hyperparameters.json no contiene el bloque 'global'.");
    end

    mcmc.nsim = hp_json.mcmc_config.nsim;
    mcmc.burn = hp_json.mcmc_config.burn;
    mcmc.N    = hp_json.mcmc_config.N;
    mcmc.M    = hp_json.mcmc_config.M;

    component_idx = manifest.component_idx;
    n_comp = numel(hp_json.hyperparams_list);
    assert(n_comp == cfg.M, "%s declara M=%d y tiene %d componentes.", eid_f, cfg.M, n_comp);

    jobs = {};
    for chain = 1:hp_json.n_iter
        for k = 1:n_comp
            fpc_idx = component_idx(k) + 1;
            fpath = fullfile(paths_fuente.functional, sprintf("dataset_fpc_%d_train.csv", fpc_idx));
            fid = fopen(fpath, "r");
            cabecera = strtrim(fgetl(fid));
            fclose(fid);
            dt = dlmread(fpath, ",", 1, 0);
            nombres = strsplit(cabecera, ",");

            if iscell(hp_json.hyperparams_list)
                hp_k = hp_json.hyperparams_list{k}.hyperparams;
            else
                hp_k = hp_json.hyperparams_list(k).hyperparams;
            end

            job = struct();
            job.chain   = chain;
            job.k       = k;
            job.fpc_idx = fpc_idx;
            job.seed    = hp_json.seed_base + chain * 9973 + k * 31;
            job.y       = dt(:, 1);
            job.Xnoint  = dt(:, 2:end);
            job.feature_names = strjoin(nombres(2:end), ",");
            job.mcmc    = mcmc;
            if isfield(hp_k, "atau"), job.hp.atau = hp_k.atau; else, job.hp.atau = gl.atau; end
            if isfield(hp_k, "btau"), job.hp.btau = hp_k.btau; else, job.hp.btau = gl.btau; end
            job.hp.ag      = gl.ag;
            job.hp.bg      = gl.bg;
            job.hp.mumu    = gl.mumu;
            job.hp.taumu   = gl.taumu;
            job.hp.pwj     = gl.pwj;
            job.hp.apij    = hp_k.apij(:);
            job.hp.bpij    = hp_k.bpij(:);
            job.hp.mupsij  = hp_k.mupsij(:);
            job.hp.taupsij = hp_k.taupsij(:);
            % mismo nombre de traza que el estudio, en el directorio de DESTINO
            job.out_path = fullfile(paths_destino.out_artefact, ...
                                    sprintf("chain_fpc_%d_iter%02d.mat", fpc_idx, chain));
            jobs{end + 1} = job; %#ok<AGROW>
        end
    end
end
