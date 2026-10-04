function mezcla_gp_train(Y, X, hp, mcmc, S_lam, S_U, out_path, seed)
% MEZCLA_GP_TRAIN  Gibbs de la mezcla probit stick-breaking con atomos GP.
%                  Una cadena. Experimento sobre la corrida 107.
%
%   mezcla_gp_train(Y, X, hp, mcmc, S_lam, S_U, out_path, seed)
%
% MODELO (una asignacion S_t por curva completa)
%   c_t | S_t=h, x_t ~ N_K( A_h' x_t , C_h )
%   Pr(S_t=h | x_t)  = pi_h(x_t) = v_h prod_{l<h}(1-v_l),  v_h = Phi(eta_h),  v_N = 1
%   eta_h(x)         = alpha_h + omega_h' z,     x = [1, z]
%
%   hp.covarianza = 'gp':  C_h = SIG(b)^2 * S(ell_a) + NUG(c)^2 * I,
%                          S(ell_a) = U_a diag(lam_a) U_a'  (GP proyectado a la base)
%   hp.covarianza = 'iw':  C_h ~ Inv-Wishart(nu0, Lam0)
%   hp.covarianza = 'iw_gp_comun':  C_h = C para todo h (el error no depende del
%                          regimen), C ~ IW(nu0, kappa [SIG(b)^2 S(ell_a) + NUG(c)^2 I]),
%                          kappa = nu0 - K - 1, de modo que E[C | a,b,c] es el GP:
%                          el GP es el CENTRO del prior, no una restriccion
%                          (Yang, Zhu, Choi y Cox 2016, IWP centrado en Matern).
%
% PRIORS
%   A_h | C_h ~ MN(M0, V0, C_h)
%     prior_A = 'g'        : M0 = 0, V0 = (n/g) (X'X)^{-1}   (g-prior matricial)
%     prior_A = 'encogido' : M0 = MCO comun de train, V0 = diag(V0_diag)
%   (a, b, c) uniforme sobre la grilla  |  C_h ~ IW(nu0, Lam0)
%   alpha_h ~ N(mu_a, 1)
%     prior_mu_alpha = 'normal'   : mu_a ~ N(mu_mu, 1/tau_mu)
%     prior_mu_alpha = 'disperso' : mu_a inducido por una concentracion DP
%       a ~ G(conc(1), conc(2)) igualando E[v | mu_a] = Phi(mu_a / c) = 1/(1+a),
%       c = sqrt(2 + s_omega^2). Con G(1, 20) es el DPM "disperso" de
%       Fruhwirth-Schnatter y Malsiner-Walli (2019): vacia los atomos sobrantes.
%   omega_h ~ N(0, (s_omega^2/p) I)
%
% GIBBS (exacto; Metropolis solo en los movimientos de etiqueta)
%   1. S_t        : discreta, pi_h(x_t) N_K(c_t; A_h'x_t, C_h)
%   2. atomo h    : con A_h INTEGRADO, Pn = V0^{-1} + X_h'X_h,
%                   B = Pn^{-1} (X_h'Y_h + V0^{-1} M0),
%                   Q = Y_h'Y_h + M0'V0^{-1}M0 - B'Pn B:
%                     log p(C | Y_h) = -(n_h/2) log|C| - (1/2) tr(C^{-1} Q)
%                   'gp': con C = U_a diag(d) U_a', la traza y el determinante
%                         cuestan O(K) por punto de la grilla -> se muestrea
%                         (a, b, c) EXACTO sobre la grilla completa.
%                   'iw': C_h ~ IW(nu0 + n_h, Lam0 + Q).
%                   Luego A_h | C_h ~ MN(B, Pn^{-1}, C_h).
%                   'iw_gp_comun': C ~ IW(nu0 + n, Psi_abc + sum_h Q_h), luego
%                   A_h | C, y (a, b, c) | C EXACTO sobre la grilla con
%                     log p(a,b,c | C) = (nu0/2) sum_k log d_k - (kappa/2) sum_k d_k (U_a'C^{-1}U_a)_kk
%                   (d = SIG_b^2 lam_a + NUG_c^2), O(K) por punto de la grilla.
%   3. Z*         : normales truncadas (Albert-Chib), por inversa de erfc en
%                   la cola, con el algoritmo de Robert (1995) cuando |eta| > 25.
%   4. gating     : (alpha_h, omega_h) | Z* normal conjugada; mu_a | alpha normal
%                   ('normal') o exacto sobre una grilla fina ('disperso').
%   2b. etiquetas : mcmc.n_mov intentos por tipo (Papaspiliopoulos y Roberts
%                   2008; Hastie, Liverani y Richardson 2015), entre 2 y 3 para
%                   que Z* se regenere con (S, alpha, omega) ya movidos:
%                   (a) intercambia dos atomos OCUPADOS j, k (A, C, S); el
%                       gating queda fijo. log R = sum_{S=j}(lp_k - lp_j) + sum_{S=k}(lp_j - lp_k).
%                   (b) intercambia h y h+1 (h uniforme en 1..N-2) CON su
%                       gating (alpha, omega). log R = sum_t lp'_{S'_t} - lp_{S_t}.
%                   El prior de atomos y gating es iid: no entra en R.
%
% ENTRADAS
%   Y      (n,K)      respuesta: coeficientes blanqueados de la curva (SOLO train)
%   X      (n,q)      [1, covariables estandarizadas]
%   hp     struct     .covarianza .SIG .NUG .prior_A .g .V0_diag .M0 .s_omega
%                     .mu_mu .tau_mu .nu0 .Lam0
%   mcmc   struct     .nsim .burn .thin .N .n_inicial .n_mov
%   S_lam  (nE,K)     autovalores de S(ell) por ell
%   S_U    (K,K,nE)   autovectores de S(ell) por ell
%   out_path char     .mat de salida
%   seed   int        semilla (unica por cadena)
%
% SALIDA (.mat -v7). Los indices (idx_cov, S) son BASE 1.
%   A (nk,N,q,K) · idx_cov (nk,N,3) · C (nk,N,K,K, 'iw' | nk,K,K, 'iw_gp_comun')
%   idx_hyp (nk,3, solo 'iw_gp_comun') · alpha (nk,N-1)
%   omega (nk,N-1,p) · mu_a (nk,1) · S (nk,n) · loglik, n_ocupados, masa_max (nsim,1)
%   ms_por_iter · ms_por_paso (1,5) = [S, atomos, Z*, gating, etiquetas] · seed
%   acept_mov (1,3) = tasa de aceptacion de (a), (b) y (b) con algun atomo ocupado

    rng(seed, 'twister');

    [n, K] = size(Y);
    q = size(X, 2);
    p = q - 1;
    N = mcmc.N;
    modo  = char(hp.covarianza);
    gp    = strcmp(modo, 'gp');
    comun = strcmp(modo, 'iw_gp_comun');
    assert(gp || comun || strcmp(modo, 'iw'), 'covarianza %s desconocida.', modo);

    SIG = hp.SIG(:);  NUG = hp.NUG(:);
    nE = size(S_lam, 1);  nS = numel(SIG);  nN = numel(NUG);
    assert(size(S_lam, 2) == K && size(S_U, 1) == K && size(S_U, 3) == nE, ...
        'S_lam / S_U no coinciden con K=%d.', K);

    % ── prior de A_h ─────────────────────────────────────────────────────────
    M0 = hp.M0;
    if strcmp(char(hp.prior_A), 'encogido')
        V0i = diag(1 ./ hp.V0_diag(:));
    else
        V0  = (n / hp.g) * inv(X' * X);
        V0  = (V0 + V0') / 2;
        V0i = inv(V0);
    end
    assert(isequal(size(M0), [q K]) && isequal(size(V0i), [q q]), 'prior de A mal dimensionado.');
    Lam0  = hp.Lam0;
    nu0   = hp.nu0;
    kappa = nu0 - K - 1;
    assert(~comun || kappa > 0, 'iw_gp_comun requiere nu0 > K + 1.');

    % ── gating ───────────────────────────────────────────────────────────────
    s_om = hp.s_omega / sqrt(max(p, 1));
    P0   = diag([1; repmat(1 / s_om^2, p, 1)]);

    % ── grilla de covarianzas: d(b,c,k) = SIG_b^2 lam_{a,k} + NUG_c^2 ──────────
    dG    = cell(nE, 1);
    logdG = zeros(nE, nS, nN);
    for a = 1:nE
        lam      = reshape(S_lam(a, :), 1, 1, K);
        dG{a}    = reshape(SIG.^2, nS, 1, 1) .* lam + reshape(NUG.^2, 1, nN, 1);
        logdG(a, :, :) = reshape(sum(log(dG{a}), 3), 1, nS, nN);
    end

    % ── estado inicial ───────────────────────────────────────────────────────
    S     = randi(min(N, mcmc.n_inicial), n, 1);
    A     = repmat(M0, 1, 1, N);
    idx   = repmat([floor(nE/2) + 1, floor(nS/2) + 1, floor(nN/2) + 1], N, 1);
    Csig  = repmat(diag(var(Y, 1, 1)), 1, 1, N);
    Cc    = diag(var(Y, 1, 1));                               % 'iw_gp_comun'
    idh   = idx(1, :);
    alpha = zeros(N - 1, 1);
    om    = zeros(N - 1, p);
    mu_a  = 0;

    % ── trazas ───────────────────────────────────────────────────────────────
    guardar = (mcmc.burn + 1):mcmc.thin:mcmc.nsim;
    nk      = numel(guardar);
    trA     = zeros(nk, N, q, K, 'single');
    trIdx   = zeros(nk, N, 3, 'int16');
    if comun
        trC   = zeros(nk, K, K, 'single');
        trHyp = zeros(nk, 3, 'int16');
    elseif ~gp
        trC = zeros(nk, N, K, K, 'single');
    end
    trAlpha = zeros(nk, N - 1);
    trOm    = zeros(nk, N - 1, p);
    trMu    = zeros(nk, 1);
    trS     = zeros(nk, n, 'int16');
    loglik     = zeros(mcmc.nsim, 1);
    n_ocupados = zeros(mcmc.nsim, 1);
    masa_max   = zeros(mcmc.nsim, 1);
    t_pasos    = zeros(1, 5);
    n_prop     = zeros(1, 3);  n_acep = zeros(1, 3);            % [a, b, b con algun atomo ocupado]

    % ── prior de mu_a ────────────────────────────────────────────────────────
    disperso = strcmp(char(hp.prior_mu_alpha), 'disperso');
    if disperso
        [gmu, lpmu] = grilla_mu_disperso(hp.conc(1), hp.conc(2), hp.s_omega);
        mu_a = gmu(find(lpmu == max(lpmu), 1));
    end

    lidx   = 1:(N - 1);
    kg     = 1;
    t_ini  = tic;
    cada   = max(floor(mcmc.nsim / 10), 1);

    for it = 1:mcmc.nsim

        % ── 1. asignaciones S_t ──────────────────────────────────────────────
        t0  = tic;
        eta = alpha' + X(:, 2:end) * om';                     % (n, N-1)
        lp  = log_pesos(eta);                                 % (n, N)
        if comun, [Uc, dc] = factor_atomo(false, [], [], [], [], [], Cc); end
        for h = 1:N
            if comun
                U = Uc;  d = dc;
            else
                [U, d] = factor_atomo(gp, idx(h, :), S_U, S_lam, SIG, NUG, Csig(:, :, h));
            end
            R = (Y - X * A(:, :, h)) * U;
            lp(:, h) = lp(:, h) - 0.5 * sum((R.^2) ./ d', 2) - 0.5 * sum(log(d));
        end
        mx  = max(lp, [], 2);
        lse = mx + log(sum(exp(lp - mx), 2));
        loglik(it) = sum(lse) - 0.5 * n * K * log(2 * pi);
        Cp = cumsum(exp(lp - lse), 2);
        Cp(:, end) = 1;
        [~, S] = max(Cp > rand(n, 1), [], 2);
        cuenta = accumarray(S, 1, [N 1]);
        n_ocupados(it) = sum(cuenta > 0);
        masa_max(it)   = max(cuenta) / n;
        t_pasos(1) = t_pasos(1) + toc(t0);

        % ── 2. atomos: covarianza con A integrado, luego A | C ──────────────
        t0 = tic;
        if comun
            Qt = zeros(K);  Bs = zeros(q, K, N);  Ls = zeros(q, q, N);
            for h = 1:N
                m  = (S == h);
                Xh = X(m, :);  Yh = Y(m, :);
                Pn = V0i + Xh' * Xh;
                Vn = inv(Pn);  Vn = (Vn + Vn') / 2;
                B  = Vn * (Xh' * Yh + V0i * M0);
                Q  = Yh' * Yh + M0' * V0i * M0 - B' * Pn * B;
                Qt = Qt + (Q + Q') / 2;
                Bs(:, :, h) = B;  Ls(:, :, h) = chol_seguro(Vn);
            end
            Ua  = S_U(:, :, idh(1));
            dh  = SIG(idh(2))^2 * S_lam(idh(1), :)' + NUG(idh(3))^2;
            Psi = kappa * (Ua .* dh') * Ua';
            Cc  = iwishrnd((Psi + Qt + (Psi + Qt)') / 2, nu0 + n);
            Cc  = (Cc + Cc') / 2;
            [Uc, dc] = factor_atomo(false, [], [], [], [], [], Cc);
            Lc = (Uc .* sqrt(dc)')';
            for h = 1:N
                A(:, :, h) = Bs(:, :, h) + Ls(:, :, h) * randn(q, K) * Lc;
            end
            Ci  = inv(Cc);  Ci = (Ci + Ci') / 2;
            lpg = zeros(nE, nS, nN);
            for a = 1:nE
                Ua = S_U(:, :, a);
                wd = sum(Ua .* (Ci * Ua), 1);                     % diag(U'C^{-1}U)
                lpg(a, :, :) = 0.5 * nu0 * logdG(a, :, :) ...
                    - 0.5 * kappa * reshape(sum(reshape(wd, 1, 1, K) .* dG{a}, 3), 1, nS, nN);
            end
            idh = muestrear_grilla(lpg);
        else
            for h = 1:N
                m  = (S == h);
                nh = cuenta(h);
                Xh = X(m, :);  Yh = Y(m, :);
                Pn = V0i + Xh' * Xh;
                Vn = inv(Pn);  Vn = (Vn + Vn') / 2;
                B  = Vn * (Xh' * Yh + V0i * M0);
                Q  = Yh' * Yh + M0' * V0i * M0 - B' * Pn * B;  Q = (Q + Q') / 2;
                if gp
                    if nh == 0
                        idx(h, :) = [randi(nE), randi(nS), randi(nN)];
                    else
                        lpg = zeros(nE, nS, nN);
                        for a = 1:nE
                            Ua = S_U(:, :, a);
                            qd = sum(Ua .* (Q * Ua), 1);                  % diag(U'QU)
                            lpg(a, :, :) = -0.5 * nh * logdG(a, :, :) ...
                                - 0.5 * reshape(sum(reshape(qd, 1, 1, K) ./ dG{a}, 3), 1, nS, nN);
                        end
                        idx(h, :) = muestrear_grilla(lpg);
                    end
                else
                    Csig(:, :, h) = iwishrnd(Lam0 + Q, nu0 + nh);
                end
                [U, d] = factor_atomo(gp, idx(h, :), S_U, S_lam, SIG, NUG, Csig(:, :, h));
                A(:, :, h) = B + chol_seguro(Vn) * randn(q, K) * (U .* sqrt(d)')';
            end
        end
        t_pasos(2) = t_pasos(2) + toc(t0);

        % ── 2b. movimientos de etiqueta ──────────────────────────────────────
        t0 = tic;
        if mcmc.n_mov > 0
            lp = log_pesos(alpha' + X(:, 2:end) * om');
            fil = (1:n)';
            for r = 1:mcmc.n_mov
                % (a) dos atomos ocupados, gating fijo
                occ = find(accumarray(S, 1, [N 1]) > 0);
                if numel(occ) >= 2
                    jk = occ(randperm(numel(occ), 2));  j = jk(1);  k = jk(2);
                    mj = (S == j);  mk = (S == k);
                    lr = sum(lp(mj, k) - lp(mj, j)) + sum(lp(mk, j) - lp(mk, k));
                    n_prop(1) = n_prop(1) + 1;
                    if log(rand) < lr
                        S(mj) = k;  S(mk) = j;
                        [A, Csig, idx] = cambiar_atomos(A, Csig, idx, j, k);
                        n_acep(1) = n_acep(1) + 1;
                    end
                end
                % (b) h <-> h+1 con su gating; h uniforme en 1..N-2 (conjunto
                % fijo: si dependiera de max(S) la propuesta no seria simetrica)
                if N >= 3
                    h  = randi(N - 2);  hh = [h, h + 1];
                    a2 = alpha;  a2(hh) = alpha(fliplr(hh));
                    o2 = om;     o2(hh, :) = om(fliplr(hh), :);
                    lp2 = log_pesos(a2' + X(:, 2:end) * o2');
                    S2 = S;  S2(S == h) = h + 1;  S2(S == h + 1) = h;
                    lr = sum(lp2(sub2ind([n N], fil, S2))) - sum(lp(sub2ind([n N], fil, S)));
                    n_prop(2) = n_prop(2) + 1;
                    ocup = any(S == h | S == h + 1);
                    n_prop(3) = n_prop(3) + ocup;
                    if log(rand) < lr
                        S = S2;  alpha = a2;  om = o2;  lp = lp2;
                        [A, Csig, idx] = cambiar_atomos(A, Csig, idx, h, h + 1);
                        n_acep(2) = n_acep(2) + 1;
                        n_acep(3) = n_acep(3) + ocup;
                    end
                end
            end
        end
        t_pasos(5) = t_pasos(5) + toc(t0);

        % ── 3. latentes Z* ───────────────────────────────────────────────────
        t0 = tic;
        eta      = alpha' + X(:, 2:end) * om';                % los movimientos cambian alpha, omega
        activo   = lidx <= min(S, N - 1);                     % (n, N-1)
        positivo = lidx == S;
        Zs  = zeros(n, N - 1);
        Zs(positivo)  = eta(positivo)  + rtnorm_sup(-eta(positivo));    % Z > 0
        Zs(~positivo) = eta(~positivo) - rtnorm_sup( eta(~positivo));   % Z < 0
        t_pasos(3) = t_pasos(3) + toc(t0);

        % ── 4. gating (alpha_h, omega_h) y mu_a ──────────────────────────────
        t0 = tic;
        m0 = [mu_a; zeros(p, 1)];
        for h = 1:(N - 1)
            r  = activo(:, h);
            D  = X(r, :);
            Qg = P0 + D' * D;
            Lg = chol(Qg, 'lower');
            b  = Qg \ (D' * Zs(r, h) + P0 * m0) + Lg' \ randn(q, 1);
            alpha(h) = b(1);
            om(h, :) = b(2:end)';
        end
        if disperso
            lpm  = lpmu + gmu * sum(alpha) - 0.5 * (N - 1) * gmu.^2;
            pr   = exp(lpm - max(lpm));
            mu_a = gmu(find(cumsum(pr) >= rand * sum(pr), 1));
        else
            prec = hp.tau_mu + (N - 1);
            mu_a = (hp.tau_mu * hp.mu_mu + sum(alpha)) / prec + randn / sqrt(prec);
        end
        t_pasos(4) = t_pasos(4) + toc(t0);

        % ── guardar ──────────────────────────────────────────────────────────
        if kg <= nk && it == guardar(kg)
            trA(kg, :, :, :)  = reshape(single(permute(A, [3 1 2])), [1 N q K]);
            trIdx(kg, :, :)   = reshape(int16(idx), [1 N 3]);
            if comun
                trC(kg, :, :)  = reshape(single(Cc), [1 K K]);
                trHyp(kg, :)   = int16(idh);
            elseif ~gp
                trC(kg, :, :, :) = reshape(single(permute(Csig, [3 1 2])), [1 N K K]);
            end
            trAlpha(kg, :)    = alpha';
            trOm(kg, :, :)    = reshape(om, [1 N-1 p]);
            trMu(kg)          = mu_a;
            trS(kg, :)        = int16(S');
            kg = kg + 1;
        end

        if it == 1 || mod(it, cada) == 0
            fprintf('  [seed %d] it %5d/%d  loglik=%.1f  ocupados=%d  masa_max=%.2f  (%.0f ms/it)\n', ...
                seed, it, mcmc.nsim, loglik(it), n_ocupados(it), masa_max(it), ...
                toc(t_ini) / it * 1000);
        end
    end

    ms_por_iter = toc(t_ini) / mcmc.nsim * 1000;
    ms_por_paso = t_pasos / mcmc.nsim * 1000;
    acept_mov   = n_acep ./ max(n_prop, 1);

    A = trA;  idx_cov = trIdx;  omega = trOm;  %#ok<NASGU>
    S = trS;  mu_a = trMu;      alpha = trAlpha;  %#ok<NASGU>
    vars = {'A', 'idx_cov', 'alpha', 'omega', 'mu_a', 'S', 'loglik', ...
            'n_ocupados', 'masa_max', 'ms_por_iter', 'ms_por_paso', 'acept_mov', 'seed'};
    if ~gp
        C = trC; %#ok<NASGU>
        vars{end + 1} = 'C';
    end
    if comun
        idx_hyp = trHyp; %#ok<NASGU>
        vars{end + 1} = 'idx_hyp';
    end
    save(out_path, vars{:}, '-v7');
    fprintf('  -> %s  (%.1f ms/it, aceptacion etiquetas a=%.3f b=%.3f b_ocupados=%.3f)\n', ...
            out_path, ms_por_iter, acept_mov(1), acept_mov(2), acept_mov(3));
end


% ════════════════════════════════════════════════════════════════════════════
% FUNCIONES LOCALES
% ════════════════════════════════════════════════════════════════════════════

function [U, d] = factor_atomo(gp, idx_h, S_U, S_lam, SIG, NUG, C_h)
% C_h = U diag(d) U'. En 'gp' U es el de S(ell_a): no hay descomposicion aqui.
    if gp
        U = S_U(:, :, idx_h(1));
        d = SIG(idx_h(2))^2 * S_lam(idx_h(1), :)' + NUG(idx_h(3))^2;
    else
        [U, D] = eig((C_h + C_h') / 2);
        d = max(diag(D), 1e-12);
    end
end


function [A, Csig, idx] = cambiar_atomos(A, Csig, idx, j, k)
% Intercambia los parametros de los atomos j y k.
    A(:, :, [j k])    = A(:, :, [k j]);
    Csig(:, :, [j k]) = Csig(:, :, [k j]);
    idx([j k], :)     = idx([k j], :);
end


function [g, lp] = grilla_mu_disperso(a_s, b_r, s_omega)
% Log-prior de mu_a en una grilla, inducido por a ~ G(a_s, b_r) (tasa b_r) con
% E[v | mu] = Phi(mu / c) = 1/(1+a), c = sqrt(2 + s_omega^2):
%   a(mu) = 1/Phi(mu/c) - 1,  |da/dmu| = phi(mu/c) / (c Phi(mu/c)^2).
    g  = linspace(-3, 12, 3001)';
    c  = sqrt(2 + s_omega^2);
    lP = log_Phi(g / c);
    a  = expm1(-lP);
    lp = a_s * log(b_r) - gammaln(a_s) + (a_s - 1) * log(max(a, realmin)) - b_r * a ...
         - 0.5 * (g / c).^2 - 0.5 * log(2 * pi) - log(c) - 2 * lP;
end


function i3 = muestrear_grilla(lpg)
% Un punto (a, b, c) de la grilla con probabilidad proporcional a exp(lpg).
    pr = exp(lpg(:) - max(lpg(:)));
    k  = find(cumsum(pr) >= rand * sum(pr), 1);
    if isempty(k), k = numel(pr); end
    [ia, ib, ic] = ind2sub(size(lpg, 1:3), k);
    i3 = [ia, ib, ic];
end


function lp = log_pesos(eta)
% log pi_h para h = 1..N a partir de eta (n, N-1). Gemelo de `log_pesos` en
% mezcla_gp_funcional.py.
    lv  = log_Phi(eta);
    l1v = log_Phi(-eta);
    n   = size(eta, 1);
    acum = [zeros(n, 1), cumsum(l1v, 2)];
    lp = [lv, zeros(n, 1)] + acum;
end


function y = log_Phi(x)
% log Phi(x) estable en ambas colas. Phi(x) = 0.5 erfcx(-x/sqrt2) exp(-x^2/2).
    y  = zeros(size(x));
    ng = x < 0;
    y(ng)  = log(0.5 * erfcx(-x(ng) / sqrt(2))) - x(ng).^2 / 2;
    y(~ng) = log1p(-0.5 * erfc(x(~ng) / sqrt(2)));
end


function x = rtnorm_sup(a)
% x ~ N(0,1) truncada a (a, inf), vectorizado. Inversa por erfc mientras la
% cola es representable; Robert (1995) cuando a > 25.
    x  = zeros(size(a));
    u  = rand(size(a));
    lo = a < 25;
    x(lo) = sqrt(2) * erfcinv(u(lo) .* erfc(a(lo) / sqrt(2)));
    for i = find(~lo)'
        ai  = a(i);
        lam = (ai + sqrt(ai^2 + 4)) / 2;
        while true
            z = ai - log(rand) / lam;
            if rand <= exp(-(z - lam)^2 / 2)
                x(i) = z;
                break
            end
        end
    end
end


function L = chol_seguro(V)
% Factor inferior con jitter relativo creciente si V no es numericamente PD.
    [R, flag] = chol(V);
    j = 1e-12 * mean(diag(V));
    while flag ~= 0
        [R, flag] = chol(V + j * eye(size(V, 1)));
        j = j * 10;
    end
    L = R';
end
