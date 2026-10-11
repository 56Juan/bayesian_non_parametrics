% config_paths.m
% Alineación de rutas MATLAB ↔ Python (proyecto model_psbp_fd)

function paths = config_paths(varargin)
    % INPUTS:
    %   paths = config_paths();                              % Defaults
    %   paths = config_paths('modelo_unificado_1');          % ID directo
    %   paths = config_paths(BASENAME, TT, SEED);            % Construye ID
    
    % ─── Raíz del proyecto ────────────────────────────────────────────
    project_root = find_project_root('README.md');
    fprintf("PROJECT_ROOT : %s\n", project_root);
    
    % ─── Identificación del experimento ───────────────────────────────
    % Defaults
    basename = "modelo_unificado";
    tt       = 1;
    seed     = 4123;
    
    if nargin == 0
        % Usar defaults
        experiment_id = sprintf("%s_%d", basename, tt);
    elseif nargin == 1
        % Primer argumento es el experiment_id directo
        experiment_id = varargin{1};
    elseif nargin >= 3
        % Tres argumentos: basename, tt, seed
        basename = varargin{1};
        tt       = varargin{2};
        seed     = varargin{3};
        experiment_id = sprintf("%s_%d", basename, tt);
    else
        error("Uso: config_paths() | config_paths(experiment_id) | config_paths(basename, tt, seed)");
    end
    
    fprintf("Experiment ID : %s\n", experiment_id);
    fprintf("Seed (base)   : %d\n", seed);
    
    % ─── Rutas canónicas (espejo de Python) ──────────────────────────
    raw_dir        = fullfile(project_root, "data", "reales", "raw", experiment_id);
    functional_dir = fullfile(project_root, "data", "reales", "processed", "functional", experiment_id);
    predict_dir    = fullfile(project_root, "data", "reales", "processed", "predict", experiment_id);
    report_dir     = fullfile(project_root, "reports", "reales", experiment_id);
    artefact_dir   = fullfile(project_root, "artefact", "reales", experiment_id);
    
    % Estructura de rutas (dict-like)
    paths = struct();
    paths.project_root = project_root;
    paths.experiment_id = experiment_id;
    paths.seed = seed;
    paths.raw = raw_dir;
    paths.functional = functional_dir;
    paths.predict = predict_dir;
    paths.out_report = report_dir;
    paths.out_artefact = artefact_dir;
    paths.out = report_dir;  % alias retrocompatible
    
    % ─── Crear directorios si no existen ──────────────────────────────
    dirs_to_create = {
        paths.raw,
        paths.functional,
        paths.predict,
        paths.out_report,
        paths.out_artefact
    };
    
    for i = 1:length(dirs_to_create)
        if ~isfolder(dirs_to_create{i})
            mkdir(dirs_to_create{i});
        end
    end
    
    % ─── Mostrar configuración ────────────────────────────────────────
    fprintf("\n");
    fprintf("  raw         → %s\n", paths.raw);
    fprintf("  functional  → %s\n", paths.functional);
    fprintf("  predict     → %s\n", paths.predict);
    fprintf("  out_report  → %s\n", paths.out_report);
    fprintf("  out_artefact → %s\n", paths.out_artefact);
    fprintf("  out         → %s\n", paths.out);
    fprintf("\n");
    
end

% ─────────────────────────────────────────────────────────────────────
% Función auxiliar: Encontrar raíz del proyecto
% ─────────────────────────────────────────────────────────────────────
function root = find_project_root(marker)
    if nargin < 1
        marker = "README.md";
    end
    
    current = pwd();
    
    while true
        if isfile(fullfile(current, marker))
            root = current;
            return;
        end
        
        parent = fileparts(current);
        if strcmp(parent, current)
            root = pwd();
            warning("No se encontró '%s'. Usando: %s", marker, root);
            return;
        end
        
        current = parent;
    end
end