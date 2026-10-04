% mezcla_gp_iteracion.m — Experimento: mezcla probit stick-breaking con atomos
% GP sobre la curva, datos de la corrida 107 (C-1)
%
% Paso 2 del ciclo Python -> MATLAB -> Python. Lee el contrato que escribio
% exp107_01_datos.ipynb (hyperparameters.json + CSV en `functional`), arma un
% job por cadena y los reparte con parfor. Entrena SOLO con el bloque train.
%
% A diferencia de psbp_fd_iteracion.m no hay un job por componente FPCA: la
% mezcla es UNA para la curva completa (una asignacion S_t por curva), asi que
% el unico eje paralelo son las cadenas.
%
% Salida: <artefact>/<EXPERIMENT_ID>/traza_mezcla_gp_cadena<NN>.mat
% (el nombre lo fija ARCHIVOS_MEZCLA["traza"] en mezcla_gp_funcional.py y
% viaja en el contrato).

clear; clc; close all;

% ════════════════════════════════════════════════════════════════════════════
% 1. CONFIGURACIÓN  — debe coincidir con exp107_01 / exp107_04
% ════════════════════════════════════════════════════════════════════════════

BASENAME     = "experimento_107";
ESCENARIO_ID = 1;            % C-1
REPLICA_ID   = 1;
M_COV        = 4;            % scores FPCA de la 107 usados como covariables
N_WORKERS    = 2;            % un job por cadena: mas workers no sirven

EXPERIMENT_ID = sprintf("%s_%d_r%02d_m%02d", BASENAME, ESCENARIO_ID, REPLICA_ID, M_COV);
paths = config_paths(EXPERIMENT_ID);

% ════════════════════════════════════════════════════════════════════════════
% 2. CONTRATO: hyperparameters.json es la unica fuente de verdad
% ════════════════════════════════════════════════════════════════════════════

hp_path = fullfile(paths.out_artefact, "hyperparameters.json");
assert(isfile(hp_path), "No se encontro %s. Corre exp107_01_datos.ipynb.", hp_path);
hp_json = jsondecode(fileread(hp_path));
campos = {'n_iter', 'mcmc_config', 'seed_base', 'modelo', 'dims', 'archivos'};
for i = 1:numel(campos)
    assert(isfield(hp_json, campos{i}), "hyperparameters.json no contiene '%s'.", campos{i});
end

arch = hp_json.archivos;
leer = @(clave) readmatrix(fullfile(paths.functional, arch.(char(clave))), "FileType", "text");

Y     = leer("Y_train");
X     = leer("X_train");
S_lam = leer("S_lam");
Ust   = leer("S_U");
M0    = leer("M0");
Lam0  = leer("Lam0");

dims = hp_json.dims;
[nE, K] = size(S_lam);
assert(isequal(size(Y), [dims.n_train, dims.K]), "Y_train no coincide con dims.");
assert(isequal(size(X), [dims.n_train, dims.q]), "X_train no coincide con dims.");
assert(nE == dims.nE && K == dims.K, "S_lam no coincide con dims.");
assert(isequal(size(Ust), [nE * K, K]), "S_U no tiene %d bloques de %dx%d.", nE, K, K);
assert(isequal(size(M0), [dims.q, K]), "M0 no es %dx%d.", dims.q, K);
assert(isequal(size(Lam0), [K, K]), "Lam0 no es %dx%d.", K, K);
S_U = zeros(K, K, nE);
for a = 1:nE
    S_U(:, :, a) = Ust((a - 1) * K + (1:K), :);
end

mod_ = hp_json.modelo;
hp.covarianza = char(mod_.covarianza);
hp.SIG     = mod_.SIG(:);
hp.NUG     = mod_.NUG(:);
hp.prior_A = char(mod_.prior_A);
hp.g       = mod_.g;
hp.V0_diag = mod_.V0_diag(:);      % vacio con prior_A = 'g'
hp.M0      = M0;
hp.s_omega = mod_.s_omega;
hp.mu_mu   = mod_.mu_mu;
hp.tau_mu  = mod_.tau_mu;
hp.prior_mu_alpha = char(mod_.prior_mu_alpha);
hp.conc    = [mod_.conc_a, mod_.conc_b];
hp.nu0     = mod_.nu0;
hp.Lam0    = (Lam0 + Lam0') / 2;

mc = hp_json.mcmc_config;
mcmc.nsim = mc.nsim;  mcmc.burn = mc.burn;  mcmc.thin = mc.thin;
mcmc.N    = mc.N;     mcmc.n_inicial = mc.n_inicial;
mcmc.n_mov = mc.n_mov;

N_CHAINS  = hp_json.n_iter;
SEED_BASE = hp_json.seed_base;

fprintf("════════════════════════════════════════════════════════\n");
fprintf("  %s\n", EXPERIMENT_ID);
fprintf("  n_train=%d  K=%d  q=%d  (p=%d)   covarianza=%s   prior_A=%s   lam0=%s\n", ...
        dims.n_train, K, dims.q, dims.p, hp.covarianza, hp.prior_A, char(mod_.lam0));
fprintf("  MCMC: nsim=%d burn=%d thin=%d N=%d   cadenas=%d   seed_base=%d   n_mov=%d   mu_alpha=%s\n", ...
        mcmc.nsim, mcmc.burn, mcmc.thin, mcmc.N, N_CHAINS, SEED_BASE, mcmc.n_mov, hp.prior_mu_alpha);
fprintf("  grilla GP: %d ell x %d sigma x %d pepita\n", nE, numel(hp.SIG), numel(hp.NUG));
fprintf("════════════════════════════════════════════════════════\n\n");

% ════════════════════════════════════════════════════════════════════════════
% 3. JOBS Y PARFOR
% ════════════════════════════════════════════════════════════════════════════

out_paths = strings(N_CHAINS, 1);
seeds     = zeros(N_CHAINS, 1);
for chain = 1:N_CHAINS
    seeds(chain)     = SEED_BASE + chain * 9973;
    out_paths(chain) = fullfile(paths.out_artefact, ...
                                strrep(arch.traza, "{cadena:02d}", sprintf("%02d", chain)));
end

pool = gcp('nocreate');
if isempty(pool)
    pool = parpool('local', min(N_WORKERS, N_CHAINS));
end
fprintf("✓ Pool con %d workers\n\n", pool.NumWorkers);

t_total = tic;
parfor chain = 1:N_CHAINS
    fprintf("[cadena %d] seed=%d\n", chain, seeds(chain));
    mezcla_gp_train(Y, X, hp, mcmc, S_lam, S_U, char(out_paths(chain)), seeds(chain));
end

fprintf("\n✅ %d cadenas en %.1f min\n", N_CHAINS, toc(t_total) / 60);
for chain = 1:N_CHAINS
    fprintf("   %s\n", out_paths(chain));
end
fprintf("\nContinuar con exp107_04_evaluacion.ipynb.\n");
