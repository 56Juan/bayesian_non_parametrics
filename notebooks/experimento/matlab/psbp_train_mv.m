function psbp_train_mv(Y, Xnoint, hp, mcmc, out_path, feature_names, seed)
% PSBP_TRAIN_MV  Muestreador Gibbs del PSBPM-FD CONJUNTO (respuesta vectorial).
%
%   psbp_train_mv(Y, Xnoint, hp, mcmc, out_path, feature_names, seed)
%
% ENTRADAS
%   Y            (n,q)   respuesta: coeficientes blanqueados theta_w
%   Xnoint       (n,p)   predictores sin columna de unos (rezagos de theta_w)
%   hp           struct  .ag .bg .mumu .taumu .pwj .nu0 .S0 (q,q)
%                        .apij .bpij .mupsij .taupsij  (p,1)
%   mcmc         struct  .nsim .burn .N .M   (M = tamano de la grilla G*)
%   out_path     char    .mat de salida
%   feature_names char   nombres de los predictores separados por coma
%   seed         int     semilla RNG
%
% MODELO. Un solo stick-breaking probit (gating compartido por las q
% coordenadas), con pesos v_h(x) = Phi(alpha_h - sum_j psi_hj |x_j - Gamma_hj|),
% Gamma_hj sobre la grilla por predictor G*, e inclusion gamma_hj. Atomo h:
%   y_t | S_t = h  ~  N_q( B_h [1; x_t], Sigma_h )
%   B_h' | Sigma_h ~ MN( 0, (n/g) (X_h'X_h)^{-1}, Sigma_h )   (g-prior de Zellner)
%   Sigma_h        ~ IW( nu0, S0 ),   g ~ Gamma(ag, bg)
% Con q = 1 reproduce psbp_train.m (misma estructura de pasos); con N = 1 es
% un VAR(p) sobre theta_w, es decir el FAR(p) con kn = K.
%
% Respecto de psbp_train.m se vectorizo la asignacion S_t y, en el paso de
% Gamma, la suma sobre los otros predictores se calcula una vez por (h, j) en
% vez de una vez por punto de la grilla (misma cantidad, 50 veces menos trabajo).

    if nargin < 7, seed = 1; end
    rng(seed, 'twister');

    [n, q] = size(Y);
    p = size(Xnoint, 2);
    X = [ones(n,1), Xnoint];

    nsim = mcmc.nsim;  burn = mcmc.burn;  N = mcmc.N;  M = mcmc.M;

    ag = hp.ag; bg = hp.bg; mumu = hp.mumu; taumu = hp.taumu; pwj = hp.pwj;
    nu0 = hp.nu0; S0 = hp.S0;
    apij = hp.apij(:); bpij = hp.bpij(:); mupsij = hp.mupsij(:); taupsij = hp.taupsij(:);
    assert(all(size(S0) == [q q]), 'S0 debe ser (q,q).');
    assert(numel(apij) == p && numel(taupsij) == p, 'apij/bpij/mupsij/taupsij deben tener p entradas.');

    % ── Grilla de localizacion G*: una columna por predictor, extremos incluidos
    xmin = min(Xnoint, [], 1);  xmax = max(Xnoint, [], 1);
    Gstar = xmin + ((0:M-1)' / (M-1)) .* (xmax - xmin);

    % ── Inicializacion ─────────────────────────────────────────────────────
    B      = zeros(N, q, p+1);                 % B(h,:,:) = [b0 | B_j]
    Sig    = repmat(reshape(S0, 1, q, q), N, 1, 1);
    Si     = randi(N, n, 1);
    alphah = zeros(N-1, 1);
    psijh  = repmat(mupsij', N-1, 1);
    gammajh = ones(N, p);
    Gloc   = randi(M, N-1, p);
    Gammajh = Gstar(sub2ind([M, p], Gloc, repmat(1:p, N-1, 1)));
    mu = 1; g = 1; pij = 0.5*ones(p,1); wj = ones(p,1);
    XtX_full = X' * X;

    % ── Trazas ─────────────────────────────────────────────────────────────
    Bout       = zeros(nsim, N, q, p+1, 'single');
    Sigout     = zeros(nsim, N, q, q,   'single');
    alphahout  = zeros(nsim, N-1,       'single');
    Gammajhout = zeros(nsim, N-1, p,    'single');
    psijhout   = zeros(nsim, N-1, p,    'single');
    gammajhout = zeros(nsim, N,   p,    'single');
    pijout     = zeros(nsim, p,         'single');
    wjout      = zeros(nsim, p,         'single');
    Sout       = zeros(nsim, n,         'int16');
    N1out      = zeros(nsim, 1,         'single');
    muout      = zeros(nsim, 1,         'single');
    gout       = zeros(nsim, 1,         'single');
    loglikout  = zeros(nsim, 1);        % log p(Y | estado), mezcla completa
    mse_inout  = zeros(nsim, 1);        % error cuadratico in-sample de E[y|x]
    entropout  = zeros(nsim, 1);        % entropia media de los pesos w_h(x_t)
    gamdiagout = zeros(nsim, 4);
    inE_media  = zeros(n, q);  n_post = 0;

    t_start = tic;
    for gt = 1:nsim

        % ── 1. Pesos del stick-breaking en cada x_t ─────────────────────────
        D    = dist_gating(Xnoint, psijh, Gammajh);              % (n, N-1)
        v    = normcdf(alphah' - D);                              % (n, N-1)
        v    = min(max(v, 1e-12), 1 - 1e-12);
        logw = [log(v), zeros(n,1)] + [zeros(n,1), cumsum(log(1 - v), 2)];   % (n, N)

        % ── 2. Log-densidad de cada atomo y asignacion S_t ──────────────────
        logf = zeros(n, N);
        for h = 1:N
            Bh = reshape(B(h,:,:), q, p+1);
            Sh = reshape(Sig(h,:,:), q, q);
            R  = Y - X * Bh';                                     % (n, q)
            Lh = chol(Sh, 'lower');
            Q  = Lh \ R';                                         % (q, n)
            logf(:,h) = -0.5*sum(Q.^2, 1)' - sum(log(diag(Lh))) - 0.5*q*log(2*pi);
        end
        lp   = logw + logf;
        mx   = max(lp, [], 2);
        loglikout(gt) = sum(mx + log(sum(exp(lp - mx), 2)));
        P    = exp(lp - mx);  P = P ./ sum(P, 2);
        U    = rand(n, 1);
        Si   = sum(cumsum(P, 2) < U, 2) + 1;  Si = min(Si, N);
        Wn   = exp(logw);  Wn = Wn ./ sum(Wn, 2);
        entropout(gt) = -mean(sum(Wn .* log(Wn + realmin), 2));

        % ── 3. Z latentes del probit | S (despues de S, FIX 4 del univariado) ─
        eta  = alphah' - D;
        cols = 1:N-1;
        mask = cols <= min(Si, N-1);
        pos  = (cols == Si) & (Si < N);
        neg  = mask & ~pos;
        Zc   = zeros(n, N-1);
        Zc(pos) =  rtn_pos( eta(pos), 1);
        Zc(neg) = -rtn_pos(-eta(neg), 1);
        Zil  = Zc;
        Wil  = (Zc + D) .* mask;

        % ── 4. B_h | Sigma_h, S  (matriz-normal conjugada con g-prior) ───────
        for h = 1:N
            inc = [true, gammajh(h,:) == 1];
            Xh  = X(:, inc);  idx = (Si == h);
            Vinv = (g/n) * (Xh' * Xh) + Xh(idx,:)' * Xh(idx,:);
            Vinv = (Vinv + Vinv')/2;
            Vh   = inv(Vinv);  Vh = (Vh + Vh')/2;
            Mh   = Vh * (Xh(idx,:)' * Y(idx,:));                  % (ph, q)
            Sh   = reshape(Sig(h,:,:), q, q);
            Bh_t = Mh + chol(Vh + 1e-10*eye(size(Vh)), 'lower') * randn(size(Mh)) * chol(Sh, 'upper');
            Bh   = zeros(q, p+1);  Bh(:, inc) = Bh_t';
            B(h,:,:) = reshape(Bh, 1, q, p+1);
        end

        % ── 5. Sigma_h | B_h, S  (Wishart inverso) ──────────────────────────
        for h = 1:N
            inc = [true, gammajh(h,:) == 1];
            Xh  = X(:, inc);  idx = (Si == h);
            Bh  = reshape(B(h,:,:), q, p+1);  C = Bh(:, inc)';   % (ph, q)
            E   = Y(idx,:) - Xh(idx,:) * C;
            Spost = S0 + E'*E + (g/n) * (C' * (Xh'*Xh) * C);
            Spost = (Spost + Spost')/2;
            nupost = nu0 + sum(idx) + sum(inc);
            Sig(h,:,:) = reshape(riwishart(nupost, Spost), 1, q, q);
        end

        % ── 6. g ────────────────────────────────────────────────────────────
        aghat = ag;  temp = 0;
        for h = 1:N
            inc = [true, gammajh(h,:) == 1];
            Xh  = X(:, inc);  Bh = reshape(B(h,:,:), q, p+1);  C = Bh(:, inc)';
            Sh  = reshape(Sig(h,:,:), q, q);
            aghat = aghat + 0.5 * q * sum(inc);
            temp  = temp + trace(Sh \ (C' * (Xh'*Xh) * C));
        end
        g = rgamma(aghat, 1/(bg + 0.5/n*temp), 1, 1);

        % ── 7. w_j ──────────────────────────────────────────────────────────
        for j = 1:p
            if sum(gammajh(:,j)) > 0
                wj(j) = 1;
            else
                b_val  = exp(gammaln(bpij(j)+N) + gammaln(apij(j)+bpij(j)) ...
                           - gammaln(bpij(j)) - gammaln(apij(j)+bpij(j)+N));
                pwjhat = pwj*b_val / ((1-pwj) + pwj*b_val);
                wj(j)  = double(rand < pwjhat);
            end
        end

        % ── 8. alpha_h, mu ──────────────────────────────────────────────────
        for h = 1:N-1
            v_val = 1 / (1 + sum(Si >= h));
            alphah(h) = v_val*(mu + sum(Wil(Si >= h, h))) + sqrt(v_val)*randn;
        end
        taumuhat = N + taumu;
        mu = (taumu*mumu + sum(alphah))/taumuhat + sqrt(1/taumuhat)*randn;

        % ── 9. pi_j ─────────────────────────────────────────────────────────
        for j = 1:p
            if wj(j) == 0, pij(j) = 0;
            else, pij(j) = rbeta(apij(j) + sum(gammajh(:,j)), bpij(j) + N - sum(gammajh(:,j)));
            end
        end

        % ── 10. Gamma_hj sobre la grilla (resto precalculado una vez por (h,j)) ─
        gd = zeros(1, 4);
        for h = 1:N-1
            sel = (Si >= h);  kh = sum(sel);
            Xs  = Xnoint(sel, :);  Zs = Zil(sel, h);
            Dfull = sum(psijh(h,:) .* abs(Xs - Gammajh(h,:)), 2);    % (kh,1)
            for j = 1:p
                if gammajh(h,j) == 1
                    rest = Dfull - psijh(h,j) * abs(Xs(:,j) - Gammajh(h,j));
                    base = alphah(h) - rest;                                    % (kh,1)
                    pm = zeros(M,1);
                    for m_idx = 1:M
                        mu_m = base - psijh(h,j) * abs(Xs(:,j) - Gstar(m_idx, j));
                        pm(m_idx) = -0.5 * sum((Zs - mu_m).^2);
                    end
                    rango = max(pm) - min(pm);
                    pm1 = exp(pm - max(pm));  pm1 = pm1 / sum(pm1);
                    gd(1) = gd(1) + 1;  gd(2) = gd(2) + any(~isfinite(pm1));
                    gd(3) = max(gd(3), abs(sum(pm1) - 1));  gd(4) = max(gd(4), rango);
                    Gammajh(h,j) = Gstar(rdiscrete(pm1), j);
                    Dfull = rest + psijh(h,j) * abs(Xs(:,j) - Gammajh(h,j));
                end
            end
        end

        % ── 11. psi_hj (normal truncada a (0, inf)) ─────────────────────────
        for h = 1:N-1
            sel = (Si >= h);  Xs = Xnoint(sel, :);  Zs = Zil(sel, h);
            Dfull = sum(psijh(h,:) .* abs(Xs - Gammajh(h,:)), 2);
            for j = 1:p
                if gammajh(h,j) == 0
                    psijh(h,j) = 0;
                else
                    dj   = abs(Xs(:,j) - Gammajh(h,j));
                    rest = Dfull - psijh(h,j) * dj;
                    Tijh = alphah(h) - Zs - rest;
                    v_val = 1 / (taupsij(j) + dj'*dj);
                    m_val = v_val * (taupsij(j)*mupsij(j) + sum(Tijh .* dj));
                    psi_new = rtn_pos(m_val, sqrt(v_val));
                    Dfull = rest + psi_new * dj;
                    psijh(h,j) = psi_new;
                end
            end
        end

        % ── 12. gamma_hj (inclusion): razon de marginales con la columna j
        %        integrada bajo su prior condicional (g-prior), mas el termino
        %        del gating del univariado. Los terminos que se cancelan entre
        %        a_jh y b_jh no se calculan.
        for h = 1:N
            idx = (Si == h);  Sh = reshape(Sig(h,:,:), q, q);  Lh = chol(Sh, 'lower');
            Bh  = reshape(B(h,:,:), q, p+1);
            for j = 1:p
                inc_rest = [true, gammajh(h,:) == 1];  inc_rest(j+1) = false;
                X1  = [Xnoint(:,j), X(:, inc_rest)];
                sb  = (n/g) * inv(X1'*X1);
                sbj = sb(1,1) - sb(1,2:end) * (sb(2:end,2:end) \ sb(1,2:end)');
                Brest = Bh(:, inc_rest)';                                    % (p_rest, q)
                m0  = (sb(1,2:end) * (sb(2:end,2:end) \ Brest));             % (1, q)
                Ystar = Y(idx,:) - X(idx, inc_rest) * Brest;                 % (nh, q)
                xj  = Xnoint(idx, j);
                s1  = 1 / (xj'*xj + 1/sbj);
                m1  = s1 * (xj' * Ystar + m0 / sbj);                         % (1, q)
                % log N_q(0; m0, sbj Sh) - log N_q(0; m1, s1 Sh)
                q0  = (Lh \ m0')' * (Lh \ m0') / sbj;
                q1  = (Lh \ m1')' * (Lh \ m1') / s1;
                lr  = -0.5*q*log(sbj) - 0.5*q0 + 0.5*q*log(s1) + 0.5*q1;
                lr  = lr + log(pij(j) + realmin) - log(1 - pij(j) + realmin);
                if h < N
                    sel = (Si >= h);  Xs = Xnoint(sel,:);  Zs = Zil(sel,h);
                    dj  = abs(Xs(:,j) - Gammajh(h,j));
                    rest = sum(psijh(h,:) .* abs(Xs - Gammajh(h,:)), 2) - psijh(h,j)*dj;
                    Tijh = alphah(h) - Zs - rest;
                    v_val = 1 / (taupsij(j) + dj'*dj);
                    m_val = v_val * (taupsij(j)*mupsij(j) + sum(Tijh .* dj));
                    lr = lr + log(normpdf(0, mupsij(j), 1/sqrt(taupsij(j))) + realmin) ...
                            - log(1 - normcdf(-mupsij(j)*sqrt(taupsij(j))) + realmin) ...
                            - log(normpdf(0, m_val, sqrt(v_val)) + realmin) ...
                            + log(1 - normcdf(-m_val/sqrt(v_val)) + realmin);
                end
                gammajh(h,j) = double(rand < 1/(1 + exp(-lr)));
                if gammajh(h,j) == 0
                    Bh(:, j+1) = 0;  B(h,:,:) = reshape(Bh, 1, q, p+1);
                end
            end
        end

        % ── 13. E[y|x] in-sample con el estado final ────────────────────────
        vfin = normcdf(alphah' - dist_gating(Xnoint, psijh, Gammajh));
        cfin = cumprod(1 - vfin, 2);
        wfin = [vfin(:,1), vfin(:,2:end) .* cfin(:,1:end-1), cfin(:,end)];   % (n, N)
        inE  = zeros(n, q);
        for h = 1:N
            inE = inE + wfin(:,h) .* (X * reshape(B(h,:,:), q, p+1)');
        end
        mse_inout(gt) = mean(sum((Y - inE).^2, 2));
        if gt > burn, inE_media = inE_media + inE; n_post = n_post + 1; end

        % ── Trazas ──────────────────────────────────────────────────────────
        Bout(gt,:,:,:)     = B;
        Sigout(gt,:,:,:)   = Sig;
        alphahout(gt,:)    = alphah';
        Gammajhout(gt,:,:) = Gammajh;
        psijhout(gt,:,:)   = psijh;
        gammajhout(gt,:,:) = gammajh;
        pijout(gt,:)       = pij';
        wjout(gt,:)        = wj';
        Sout(gt,:)         = int16(Si');
        N1out(gt)          = max(Si);
        muout(gt)          = mu;
        gout(gt)           = g;
        gamdiagout(gt,:)   = gd;

        if mod(gt, 100) == 0
            el = toc(t_start);
            fprintf('  [seed=%d] iter %4d/%d  N_act=%-2d  loglik=%.1f  mse_in=%.4f  (%.1f s/iter, eta %.0f min)\n', ...
                    seed, gt, nsim, max(Si), loglikout(gt), mse_inout(gt), el/gt, el/gt*(nsim-gt)/60);
        end
    end
    inE_media = inE_media / max(n_post, 1);

    save(out_path, '-v7', ...
        'Bout','Sigout','alphahout','Gammajhout','psijhout','gammajhout','pijout','wjout', ...
        'Sout','N1out','muout','gout','loglikout','mse_inout','entropout','gamdiagout','inE_media', ...
        'Gstar','nsim','burn','N','M','p','q','n','feature_names','seed');
    fprintf('  OK guardado: %s  (%.1f min)\n', out_path, toc(t_start)/60);
end

% ============================================================================
function D = dist_gating(Xnoint, psijh, Gammajh)
    [n, p] = size(Xnoint);  Nm1 = size(psijh, 1);
    D = sum(reshape(psijh, 1, Nm1, p) .* abs(reshape(Xnoint, n, 1, p) - reshape(Gammajh, 1, Nm1, p)), 3);
end

function S = riwishart(nu, Psi)
% S ~ IW(nu, Psi): S = inv(W), W ~ Wishart(nu, inv(Psi)) por Bartlett.
    q = size(Psi, 1);
    Pinv = inv(Psi);  Pinv = (Pinv + Pinv')/2;
    Lc = chol(Pinv + 1e-10*eye(q), 'lower');
    A = zeros(q);
    for i = 1:q
        A(i,i) = sqrt(rgamma((nu - i + 1)/2, 2, 1, 1));
        for jj = 1:i-1, A(i,jj) = randn; end
    end
    W = Lc * (A * A') * Lc';
    S = inv((W + W')/2);  S = (S + S')/2;
end

function x = rgamma(a, b, r, c)
    x = zeros(r, c);
    for k = 1:r*c
        if a >= 1
            d = a - 1/3; cv = 1/sqrt(9*d);
            while true
                z = randn; v = (1+cv*z)^3;
                if v > 0 && log(rand) < 0.5*z^2 + d - d*v + d*log(v), x(k) = d*v*b; break; end
            end
        else
            x(k) = rgamma(a+1, b, 1, 1) * rand^(1/a);
        end
    end
end

function x = rbeta(a, b)
    ga = rgamma(a, 1, 1, 1);  gb = rgamma(b, 1, 1, 1);  x = ga / (ga + gb);
end

function x = rtn_pos(m, s)
    a = -m(:) ./ s(:);  t = zeros(size(a));  usa_inv = a < 35;
    t(usa_inv) = -norminv(rand(nnz(usa_inv), 1) .* normcdf(-a(usa_inv)));
    for r = find(~usa_inv)'
        lam = (a(r) + sqrt(a(r)^2 + 4)) / 2;
        while true
            z = a(r) - log(rand) / lam;
            if log(rand) <= -(z - lam)^2 / 2, t(r) = z; break; end
        end
    end
    x = reshape(m(:) + s(:) .* t, size(m));  x = max(x, 0);
end

function k = rdiscrete(pr)
    pr = pr(:)' / sum(pr);
    k = find(rand <= cumsum(pr), 1, 'first');
    if isempty(k), k = length(pr); end
end
