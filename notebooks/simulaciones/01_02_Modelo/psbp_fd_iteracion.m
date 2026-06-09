% psbp_fd_iteracion.m
% Script principal: lee configuración, datasets e hiperparámetros, y entrena
% un modelo PSBP por cada componente FPCA llamando a psbp_train().

clear; clc; close all;

% ════════════════════════════════════════════════════════════════════════════
% 1. CONFIGURACIÓN DEL EXPERIMENTO
% ════════════════════════════════════════════════════════════════════════════

TT       = 1;
SEED     = 4123;
BASENAME = "modelo_unificado";
paths    = config_paths(BASENAME, TT, SEED);

rng(SEED);
fprintf("RNG seed establecida: %d\n\n", SEED);

% ════════════════════════════════════════════════════════════════════════════
% 2. LEER HIPERPARÁMETROS DESDE ARTEFACT
% ════════════════════════════════════════════════════════════════════════════

hp_path = fullfile(paths.out_artefact, "hyperparameters.json");
assert(isfile(hp_path), "No se encontró hyperparameters.json en: %s", hp_path);
hp_json = jsondecode(fileread(hp_path));
fprintf("✓ Hiperparámetros cargados\n\n");

n_components = numel(hp_json.hyperparams_list);

% ════════════════════════════════════════════════════════════════════════════
% 3. LEER MANIFEST Y DATASETS AR(p)
% ════════════════════════════════════════════════════════════════════════════

manifest_path = fullfile(paths.functional, "datasets_manifest.json");
assert(isfile(manifest_path), "No se encontró datasets_manifest.json en: %s", manifest_path);
manifest      = jsondecode(fileread(manifest_path));
cov_names     = manifest.cov_names;
component_idx = manifest.component_idx;

fprintf("✓ Manifest cargado  (n_components=%d, n_lags=%d)\n\n", n_components, manifest.n_lags);

% ════════════════════════════════════════════════════════════════════════════
% 4. CONFIGURACIÓN MCMC
% ════════════════════════════════════════════════════════════════════════════

mcmc.nsim = 2000;
mcmc.burn = 500;
mcmc.N    = 15;
mcmc.M    = 50;

% ════════════════════════════════════════════════════════════════════════════
% 5. BUCLE PRINCIPAL — entrenar PSBP por componente FPCA
% ════════════════════════════════════════════════════════════════════════════

for k = 1:n_components

    fpc_idx = component_idx(k) + 1;    % base-1
    fname   = sprintf("dataset_fpc_%d.csv", fpc_idx);
    fpath   = fullfile(paths.functional, fname);
    assert(isfile(fpath), "No se encontró dataset: %s", fpath);

    % ── Cargar dataset ───────────────────────────────────────────────────────
    Tbl    = readtable(fpath);
    names  = Tbl.Properties.VariableNames;
    dt     = table2array(Tbl);

    y      = dt(:, 1);           % (T_eff, 1) — respuesta: score fpc_k
    Xnoint = dt(:, 2:end);       % (T_eff, p) — predictores: fpc_j_lagL
    feature_names = strjoin(names(2:end), ",");

    fprintf("════════════════════════════════════════\n");
    fprintf("  k=%d  fpc_%d   T_eff=%d   p=%d\n", k, fpc_idx, size(y,1), size(Xnoint,2));
    fprintf("  Predictores: %s\n\n", feature_names);

    % ── Construir struct de hiperparámetros ──────────────────────────────────
    hp_k = hp_json.hyperparams_list(k).hyperparams;

    hp.atau    = hp_json.global.atau;
    hp.btau    = hp_json.global.btau;
    hp.ag      = hp_json.global.ag;
    hp.bg      = hp_json.global.bg;
    hp.mumu    = hp_json.global.mumu;
    hp.taumu   = hp_json.global.taumu;
    hp.pwj     = hp_json.global.pwj;
    hp.apij    = hp_k.apij(:);
    hp.bpij    = hp_k.bpij(:);
    hp.mupsij  = hp_k.mupsij(:);
    hp.taupsij = hp_k.taupsij(:);

    % ── Ruta de salida ───────────────────────────────────────────────────────
    out_path = fullfile(paths.out_artefact, sprintf("chain_fpc_%d.mat", fpc_idx));

    % ── Entrenar ─────────────────────────────────────────────────────────────
    psbp_train(y, Xnoint, hp, mcmc, out_path, feature_names);

end

fprintf("\n✅ Entrenamiento completo: %s\n", paths.experiment_id);
