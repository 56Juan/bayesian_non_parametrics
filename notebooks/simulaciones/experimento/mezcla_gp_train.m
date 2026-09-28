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
%
% PRIORS
%   A_h | C_h ~ MN(0, V0, C_h),  V0 = (n/g) (X'X)^{-1}      (g-prior matricial)
%   (a, b, c) uniforme sobre la grilla  |  C_h ~ IW(nu0, Lam0)
%   alpha_h ~ N(mu_a, 1),  mu_a ~ N(mu_mu, 1/tau_mu)
%   omega_h ~ N(0, (s_omega^2/p) I)
%
% GIBBS (exacto, sin Metropolis)
%   1. S_t        : discreta, pi_h(x_t) N_K(c_t; A_h'x_t, C_h)
%   2. atomo h    : con A_h INTEGRADO, Pn = V0^{-1} + X_h'X_h, B = Pn^{-1} X_h'Y_h,
%                   Q = Y_h'Y_h - B'Pn B:
%                     log p(C | Y_h) = -(n_h/2) log|C| - (1/2) tr(C^{-1} Q)
%                   'gp': con C = U_a diag(d) U_a', la traza y el determinante
%                         cuestan O(K) por punto de la grilla -> se muestrea
%                         (a, b, c) EXACTO sobre la grilla completa.
%                   'iw': C_h ~ IW(nu0 + n_h, Lam0 + Q).
%                   Luego A_h | C_h ~ MN(B, Pn^{-1}, C_h).
%   3. Z*         : normales truncadas (Albert-Chib), por inversa de erfc en
%                   la cola, con el algoritmo de Robert (1995) cuando |eta| > 25.
%   4. gating     : (alpha_h, omega_h) | Z* normal conjugada; mu_a | alpha normal.
%
% ENTRADAS
%   Y      (n,K)      respuesta: coeficientes blanqueados de la curva (SOLO train)
%   X      (n,q)      [1, covariables estandarizadas]
%   hp     struct     .covarianza .SIG .NUG .g .s_omega .mu_mu .tau_mu .nu0 .Lam0
%   mcmc   struct     .nsim .burn .thin .N .n_inicial
%   S_lam  (nE,K)     autovalores de S(ell) por ell
%   S_U    (K,K,nE)   autovectores de S(ell) por ell
%   out_path char     .mat de salida
%   seed   int        semilla (unica por cadena)
%
% SALIDA (.mat -v7). Los indices (idx_cov, S) son BASE 1.
%   A (nk,N,q,K) · idx_cov (nk,N,3) · C (nk,N,K,K, solo 'iw') · alpha (nk,N-1)
%   omega (nk,N-1,p) · mu_a (nk,1) · S (nk,n) · loglik, n_ocupados, masa_max (nsim,1)
%   ms_por_iter · ms_por_paso (1,4) = [S, atomos, Z*, gating] · seed

    rng(seed, 'twister');

    [n, K] = size(Y);
    q = size(X, 2);
    p = q - 1;
    N = mcmc.N;
    gp = strcmp(char(hp.covarianza), 'gp');

    SIG = hp.SIG(:);  NUG = hp.NUG(:);
    nE = size(S_lam, 1);  nS = numel(SIG);  nN = numel(NUG);
    assert(size(S_lam, 2) == K && size(S_U, 1) == K && size(S_U, 3) == nE, ...
        'S_lam / S_U no coinciden con K=%d.', K);

    % ── g-prior matricial ────────────────────────────────────────────────────
    V0  = (n / hp.g) * inv(X' * X);
    V0  = (V0 + V0') / 2;
    V0i = inv(V0);
    Lam0 = hp.Lam0;
    nu0  = hp.nu0;

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
    A     = zeros(q, K, N);
    idx   = repmat([floor(nE/2) + 1, floor(nS/2) + 1, floor(nN/2) + 1], N, 1);
    Csig  = repmat(diag(var(Y, 1, 1)), 1, 1, N);
    alpha = zeros(N - 1, 1);
    om    = zeros(N - 1, p);
    mu_a  = 0;

    % ── trazas ───────────────────────────────────────────────────────────────
    guardar = (mcmc.burn + 1):mcmc.thin:mcmc.nsim;
    nk      = numel(guardar);
    trA     = zeros(nk, N, q, K, 'single');
    trIdx   = zeros(nk, N, 3, 'int16');
    if ~gp, trC = zeros(nk, N, K, K, 'single'); end
    trAlpha = zeros(nk, N - 1);
    trOm    = zeros(nk, N - 1, p);
    trMu    = zeros(nk, 1);
    trS     = zeros(nk, n, 'int16');
    loglik     = zeros(mcmc.nsim, 1);
    n_ocupados = zeros(mcmc.nsim, 1);
    masa_max   = zeros(mcmc.nsim, 1);
    t_pasos    = zeros(1, 4);

    lidx   = 1:(N - 1);
    kg     = 1;
    t_ini  = tic;
    cada   = max(floor(mcmc.nsim / 10), 1);

    for it = 1:mcmc.nsim

        % ── 1. asignaciones S_t ──────────────────────────────────────────────
        t0  = tic;
        eta = alpha' + X(:, 2:end) * om';                     % (n, N-1)
        lp  = log_pesos(eta);                                 % (n, N)
        for h = 1:N
            [U, d] = factor_atomo(gp, idx(h, :), S_U, S_lam, SIG, NUG, Csig(:, :, h));
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
        for h = 1:N
            m  = (S == h);
            nh = cuenta(h);
            Xh = X(m, :);  Yh = Y(m, :);
            Pn = V0i + Xh' * Xh;
            Vn = inv(Pn);  Vn = (Vn + Vn') / 2;
            B  = Vn * (Xh' * Yh);
            Q  = Yh' * Yh - B' * Pn * B;  Q = (Q + Q') / 2;
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
                    pr = exp(lpg(:) - max(lpg(:)));
                    k  = find(cumsum(pr) >= rand * sum(pr), 1);
                    if isempty(k), k = numel(pr); end
                    [ia, ib, ic] = ind2sub([nE nS nN], k);
                    idx(h, :) = [ia, ib, ic];
                end
            else
                Csig(:, :, h) = iwishrnd(Lam0 + Q, nu0 + nh);
            end
            [U, d] = factor_atomo(gp, idx(h, :), S_U, S_lam, SIG, NUG, Csig(:, :, h));
            A(:, :, h) = B + chol_seguro(Vn) * randn(q, K) * (U .* sqrt(d)')';
        end
        t_pasos(2) = t_pasos(2) + toc(t0);

        % ── 3. latentes Z* ───────────────────────────────────────────────────
        t0 = tic;
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
        prec = hp.tau_mu + (N - 1);
        mu_a = (hp.tau_mu * hp.mu_mu + sum(alpha)) / prec + randn / sqrt(prec);
        t_pasos(4) = t_pasos(4) + toc(t0);

        % ── guardar ──────────────────────────────────────────────────────────
        if kg <= nk && it == guardar(kg)
            trA(kg, :, :, :)  = reshape(single(permute(A, [3 1 2])), [1 N q K]);
            trIdx(kg, :, :)   = reshape(int16(idx), [1 N 3]);
            if ~gp
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

    A = trA;  idx_cov = trIdx;  omega = trOm;  %#ok<NASGU>
    S = trS;  mu_a = trMu;      alpha = trAlpha;  %#ok<NASGU>
    vars = {'A', 'idx_cov', 'alpha', 'omega', 'mu_a', 'S', 'loglik', ...
            'n_ocupados', 'masa_max', 'ms_por_iter', 'ms_por_paso', 'seed'};
    if ~gp
        C = trC; %#ok<NASGU>
        vars{end + 1} = 'C';
    end
    save(out_path, vars{:}, '-v7');
    fprintf('  -> %s  (%.1f ms/it)\n', out_path, ms_por_iter);
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
