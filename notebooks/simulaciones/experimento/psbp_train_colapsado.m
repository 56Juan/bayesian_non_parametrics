function psbp_train_colapsado(y, Xnoint, hp, mcmc, out_path, feature_names, seed)
% PSBP_TRAIN_COLAPSADO  psbp_train.m de la corrida 107 con UN cambio: el paso de
%   Gamma se hace con psi INTEGRADO y psi se muestrea a continuacion condicionado
%   al Gamma nuevo ([EXP-1]). Todo lo demas es identico a 107_sim_C1/psbp_train.m,
%   incluidos los [FIX] y las funciones auxiliares. EXPERIMENTO: no reemplaza al
%   psbp_train.m canonico de ninguna corrida.
%
% Motivo. En el original Gamma_hj | psi_hj y psi_hj | Gamma_hj se alternan. Con
% psi_hj ~ 0 la verosimilitud de Gamma es plana (Gamma deriva al azar), y con
% Gamma mal ubicado la condicional de psi lo empuja a 0: el gating queda
% atrapado en "dormido" aunque los datos lo sostengan (medido en 107/108: TV
% del gating 0.01-0.05 con un gating verdadero de TV 0.43).
%
% Sampler MCMC del modelo PSBP-FD para una cadena.
%
%   psbp_train(y, Xnoint, hp, mcmc, out_path, feature_names, seed)
%
% ENTRADAS
%   y            (n,1)   Respuesta (scores FPCA)
%   Xnoint       (n,p)   Predictores SIN columna de unos
%   hp           struct  Hiperparámetros del modelo
%   mcmc         struct  .nsim  .burn  .N  .M
%   out_path     char    Ruta completa del .mat de salida
%   feature_names char   Nombres de covariables separados por coma
%   seed         int     Semilla RNG (única por job)

    if nargin < 7, seed = 1; end
    rng(seed, 'twister');

    % ── Dimensiones ──────────────────────────────────────────────────────────
    n = size(Xnoint, 1);
    p = size(Xnoint, 2);
    X = horzcat(ones(n,1), Xnoint);

    % ── Parámetros MCMC ───────────────────────────────────────────────────────
    nsim = mcmc.nsim;
    burn = mcmc.burn;
    N    = mcmc.N;
    M    = mcmc.M;

    % ── Hiperparámetros ───────────────────────────────────────────────────────
    atau    = hp.atau;
    btau    = hp.btau;
    ag      = hp.ag;
    bg      = hp.bg;
    mumu    = hp.mumu;
    taumu   = hp.taumu;
    pwj     = hp.pwj;
    apij    = hp.apij(:);
    bpij    = hp.bpij(:);
    mupsij  = hp.mupsij(:);
    taupsij = hp.taupsij(:);

    % ── Grid de localización G* ───────────────────────────────────────────────
    % [FIX] El grid se construye sobre Xnoint: la columna de unos del
    % intercepto no es una covariable y contaminaba el rango [min, max].
    xmin  = min(Xnoint(:));
    xmax  = max(Xnoint(:));
    Gstar = (xmin + ((1:M)/M) .* (xmax - xmin))';

    % ── Inicialización ────────────────────────────────────────────────────────
    bjrange = [-4,-3,-2,-1.5,-1,0,1,1.5,2,3,4,5]';
    b0range = [-4,-3,-2,-1.5,-1,0,1,1.5,2,3,4,5]';
    
    betajh  = bjrange(randi(numel(bjrange), N, p));
    beta0h  = b0range(randi(numel(b0range), N, 1));
    tauh    = rgamma(atau, 1/btau, N, 1);
    Si      = randi(N, n, 1);
    alphah  = zeros(N-1, 1);

    psijh   = repmat(mupsij(:)', N-1, 1);
    gammajh = ones(N, p);
    Gloc    = randi(M, N-1, p);
    Gammajh = Gstar(Gloc);
    mu      = 1;
    g       = 1;
    pij     = 0.5 * ones(p, 1);
    wj      = ones(p, 1);

    % ── Pre-asignación de trazas ──────────────────────────────────────────────
    betajhout  = zeros(nsim, N,   p, 'single');
    beta0hout  = zeros(nsim, N,      'single');
    tauhout    = zeros(nsim, N,      'single');
    alphahout  = zeros(nsim, N-1,    'single');
    Gammajhout = zeros(nsim, N-1, p, 'single');
    psijhout   = zeros(nsim, N-1, p, 'single');
    gammajhout = zeros(nsim, N,   p, 'single');
    pijout     = zeros(nsim, p,      'single');
    wjout      = zeros(nsim, p,      'single');
    N1out      = zeros(nsim, 1,      'single');
    Nout       = zeros(nsim, 1,      'single');
    muout      = zeros(nsim, 1,      'single');
    osumout    = zeros(nsim, p,      'single');
    inEout     = zeros(nsim, n,      'single');

    t_start = tic;

    % ════════════════════════════════════════════════════════════════════════
    % BUCLE PRINCIPAL MCMC
    % ════════════════════════════════════════════════════════════════════════
    for gt = 1:nsim

        M_TRUNC = 6;
        Q_HI    = 1 - eps/2;
        % [EXP-0] Z* y S VECTORIZADOS. Mismas condicionales, mismo recorte
        % M_TRUNC y mismo clip de q que los bucles de 107_sim_C1/psbp_train.m
        % (verificado con test_equivalencia_vectorizacion.m); cambia solo el orden
        % en que se consumen los numeros aleatorios. Sin esto Octave tarda ~10 s
        % por iteracion por las ~20 mil llamadas escalares a normcdf/norminv.
        ETA = gating_eta(alphah, psijh, Gammajh, Xnoint);          % (n, N-1) sin recortar
        [Zil, Wil] = latentes_Z(ETA, alphah, Si, N, M_TRUNC, Q_HI, rand(n, N-1));

        % ── Actualizar Si ─────────────────────────────────────────────────────
        [Si, phxi] = asignaciones_S(ETA, y, X, beta0h, betajh, tauh, rand(n, 1));

        % ── E[y|x] in-sample ─────────────────────────────────────────────────
        inE = zeros(n, N);
        for h = 1:N
            inE(:,h) = phxi(:,h) .* (X(:,1)*beta0h(h,1) + X(:,2:end)*betajh(h,:)');
        end

        % ── Actualizar betah ──────────────────────────────────────────────────
        betajh = zeros(N, p);
        for h = 1:N
            Xh    = X(:, horzcat(1, gammajh(h,:)) == 1);
            Sh    = n/g * inv(Xh'*Xh) / tauh(h);
            Shhat = inv(inv(Sh) + tauh(h)*Xh(Si==h,:)'*Xh(Si==h,:));
            pgh   = size(Shhat, 1);
            for l = 1:pgh-1
                for kk = l+1:pgh
                    Shhat(l,kk) = Shhat(kk,l);
                end
            end
            muhhat    = Shhat * (tauh(h)*Xh(Si==h,:)'*y(Si==h) + inv(Sh)*zeros(pgh,1));
            betahtemp = rmvnorm(muhhat, Shhat)';
            beta0h(h) = betahtemp(1);
            count = 2;
            for j = 1:p
                if gammajh(h,j) == 1
                    betajh(h,j) = betahtemp(count);
                    count = count + 1;
                end
            end
        end

        % ── Actualizar tauh ───────────────────────────────────────────────────
        for h = 1:N
            betagh  = horzcat(beta0h(h), betajh(h, gammajh(h,:)==1));
            Xh      = X(:, horzcat(1, gammajh(h,1:p)) == 1);
            aa      = atau + 0.5*sum(Si==h) + 0.5*sum(gammajh(h,1:p)) + 0.5;
            bb      = btau + 0.5*(y(Si==h) - Xh(Si==h,:)*betagh')'*(y(Si==h) - Xh(Si==h,:)*betagh') ...
                          + 0.5/n*g*betagh*Xh'*Xh*betagh';
            tauh(h) = rgamma(aa, 1/bb, 1, 1);
        end

        % ── Actualizar g ──────────────────────────────────────────────────────
        aghat = ag + 0.5*(sum(sum(gammajh(:,1:p),1)) + N);
        temp  = 0;
        for h = 1:N
            betagh = horzcat(beta0h(h), betajh(h, gammajh(h,:)==1));
            Xh     = X(:, horzcat(1, gammajh(h,1:p)) == 1);
            temp   = temp + tauh(h)*betagh*Xh'*Xh*betagh';
        end
        bghat = bg + 0.5/n*temp;
        g     = rgamma(aghat, 1/bghat, 1, 1);

        % ── Actualizar wj ─────────────────────────────────────────────────────
        for j = 1:p
            if sum(gammajh(:,j)) > 0
                wj(j) = 1;
            else
                b_val  = exp(gammaln(bpij(j)+N) + gammaln(apij(j)+bpij(j)) ...
                           - gammaln(bpij(j)) - gammaln(apij(j)+bpij(j)+N));
                pwjhat = pwj*b_val / ((1-pwj)*1 + pwj*b_val);
                wj(j)  = double(rand(1,1) < pwjhat);
            end
        end

        % ── Actualizar alphah ─────────────────────────────────────────────────
        for h = 1:N-1
            v_val     = inv(1 + sum(Si >= h));
            m_val     = v_val * (mu + sum(Wil(Si >= h, h)));
            alphah(h) = m_val + sqrt(v_val)*randn(1,1);
        end

        % ── Actualizar mu ─────────────────────────────────────────────────────
        taumuhat = N + taumu;
        mumuhat  = (taumu*mumu + sum(alphah)) / taumuhat;
        mu       = mumuhat + sqrt(1/taumuhat)*randn(1,1);

        % ── Actualizar pij ────────────────────────────────────────────────────
        for j = 1:p
            if wj(j) == 0
                pij(j) = 0;
            else
                pij(j) = rbeta(apij(j) + sum(gammajh(:,j)), bpij(j) + N - sum(gammajh(:,j)));
            end
        end

        % ── Actualizar (Gammajh, psijh) en bloque [EXP-1] ───────────────────────
        % Con T_i = alpha_h - Z*_ih - sum_{k~=j} psi_hk |x_ik - Gamma_hk| se tiene
        % T = psi D + e, e ~ N(0,I), D_i = |x_ij - Gamma|, y psi ~ N+(mupsij, 1/taupsij).
        % Integrando psi:
        %   log p(T | Gamma) = m^2/(2v) + log(v)/2 + log Phi(m/sqrt(v)) + cte,
        %   v = 1/(taupsij + D'D),   m = v (taupsij mupsij + D'T),
        % para cada punto de G*. Se muestrea Gamma de ahi y psi | Gamma ~ N+(m, v).
        for h = 1:N-1
            idx_h = (Si >= h);
            for j = 1:p
                if gammajh(h,j) == 1
                    otros = [1:j-1, j+1:p];
                    Tj = alphah(h) - Zil(idx_h,h) ...
                         - sum(psijh(h,otros) .* abs(Xnoint(idx_h,otros) - Gammajh(h,otros)), 2);
                    Dm = abs(Xnoint(idx_h,j) - Gstar');                % (kh, M)
                    vG = 1 ./ (taupsij(j) + sum(Dm.^2, 1));            % (1, M)
                    mG = vG .* (taupsij(j)*mupsij(j) + Tj' * Dm);      % (1, M)
                    pm = 0.5*mG.^2 ./ vG + 0.5*log(vG) + log_Phi(mG ./ sqrt(vG));
                    pm = exp(pm - max(pm));
                    l  = rdiscrete(pm / sum(pm));
                    Gammajh(h,j) = Gstar(l);
                    psijh(h,j)   = rtnorm_pos(mG(l), vG(l));
                end
            end
        end

        % ── Actualizar psijh ──────────────────────────────────────────────────
        for h = 1:N-1
            kh = sum(Si >= h);
            for j = 1:p
                if gammajh(h,j) == 0
                    psijh(h,j) = 0;
                else
                    Tijh  = alphah(h) - Zil(Si>=h,h) ...
                            - sum(repmat(psijh(h,1:j-1),   kh,1) .* abs(Xnoint(Si>=h,1:j-1)   - repmat(Gammajh(h,1:j-1),   kh,1)), 2) ...
                            - sum(repmat(psijh(h,j+1:end), kh,1) .* abs(Xnoint(Si>=h,j+1:end) - repmat(Gammajh(h,j+1:end), kh,1)), 2);
                    v_val = inv(taupsij(j) + (Xnoint(Si>=h,j) - Gammajh(h,j))' * (Xnoint(Si>=h,j) - Gammajh(h,j)));
                    m_val = v_val * (taupsij(j)*mupsij(j) + sum(Tijh .* abs(Xnoint(Si>=h,j) - Gammajh(h,j))));
                    u     = rand(1,1);

                    a_val = (0 - m_val)/sqrt(v_val);
                    a_val = min(a_val, M_TRUNC);
                    m_eff = -a_val*sqrt(v_val);          % = m_val si no se recorto
                    q     = u+(1-u)*normcdf(a_val,0,1);
                    psijh(h,j) = m_eff + sqrt(v_val)*norminv(min(max(q, realmin), Q_HI),0,1);
                    psijh(h,j) = max(psijh(h,j), 0);     % soporte [0, inf)
                end
            end
        end

        % ── Actualizar gammajh ────────────────────────────────────────────────
        for h = 1:N
            for j = 1:p
                gammajh1   = gammajh(h,1:p); gammajh1(:,j) = [];
                Xh_tmp     = X;              Xh_tmp(:,j+1) = [];
                Xh1        = horzcat(Xnoint(:,j), Xh_tmp(:, horzcat(1,gammajh1)==1));
                sb         = n/g * inv(Xh1'*Xh1) / tauh(h);
                betagh_tmp = horzcat(beta0h(h), betajh(h,:)); betagh_tmp(:,j+1) = [];
                betagh2    = betagh_tmp(:, horzcat(1,gammajh1)==1);
                sbj        = sb(1,1) - sb(1,2:end)*inv(sb(2:end,2:end))*sb(1,2:end)';
                taubj      = 1/sbj;
                mubj       = sb(1,2:end)*inv(sb(2:end,2:end))*betagh2';
                ystar      = y(Si==h) - X(Si==h,1)*beta0h(h) ...
                             - Xnoint(Si==h,1:j-1)*betajh(h,1:j-1)' ...
                             - Xnoint(Si==h,j+1:end)*betajh(h,j+1:end)';

                if h < N
                    kh   = sum(Si >= h);
                    Tijh = alphah(h) - Zil(Si>=h,h) ...
                           - sum(repmat(psijh(h,1:j-1),   kh,1) .* abs(Xnoint(Si>=h,1:j-1)   - repmat(Gammajh(h,1:j-1),   kh,1)), 2) ...
                           - sum(repmat(psijh(h,j+1:end), kh,1) .* abs(Xnoint(Si>=h,j+1:end) - repmat(Gammajh(h,j+1:end), kh,1)), 2);
                    v_val = inv(taupsij(j) + (Xnoint(Si>=h,j) - Gammajh(h,j))' * (Xnoint(Si>=h,j) - Gammajh(h,j)));
                    m_val = v_val * (taupsij(j)*mupsij(j) + sum(Tijh .* abs(Xnoint(Si>=h,j) - Gammajh(h,j))));

                    bjhin = log(1-pij(j)+realmin) ...
                            + sum(log(normpdf(y(Si==h), X(Si==h,1)*beta0h(h) ...
                                + Xnoint(Si==h,1:j-1)*betajh(h,1:j-1)' ...
                                + Xnoint(Si==h,j+1:end)*betajh(h,j+1:end)', 1/sqrt(tauh(h))) + realmin)) ...
                            + sum(log(normpdf(Zil(Si>=h,h), ...
                                alphah(h) ...
                                - sum(repmat(psijh(h,1:j-1),   kh,1) .* abs(Xnoint(Si>=h,1:j-1)   - repmat(Gammajh(h,1:j-1),   kh,1)), 2) ...
                                - sum(repmat(psijh(h,j+1:end), kh,1) .* abs(Xnoint(Si>=h,j+1:end) - repmat(Gammajh(h,j+1:end), kh,1)), 2), ...
                                1) + realmin));

                    ajhin = log(pij(j)+realmin) ...
                            + sum(log(normpdf(ystar, 0, 1/sqrt(tauh(h))) + realmin)) ...
                            + log(normpdf(0, mubj, 1/sqrt(taubj)) + realmin) ...
                            - log(normpdf(0, inv(tauh(h)*Xnoint(Si==h,j)'*Xnoint(Si==h,j)+taubj) ...
                                         * (tauh(h)*Xnoint(Si==h,j)'*ystar + taubj*mubj), ...
                                         sqrt(inv(tauh(h)*Xnoint(Si==h,j)'*Xnoint(Si==h,j)+taubj))) + realmin) ...
                            + sum(log(normpdf(0, Tijh, 1) + realmin)) ...
                            + log(normpdf(0, mupsij(j), 1/sqrt(taupsij(j))) + realmin) ...
                            - log(1 - normcdf((0-mupsij(j))*sqrt(taupsij(j)),0,1) + realmin) ...
                            - log(normpdf(0, m_val, sqrt(v_val)) + realmin) ...
                            + log(1 - normcdf((0-m_val)/sqrt(v_val),0,1) + realmin);

                    gammajh(h,j) = double(rand(1,1) < 1/(1+exp(bjhin-ajhin)));

                else  % h == N
                    nh  = sum(Si == h);
                    % [FIX] eliminado el factor exp(1.2*nh) presente en ajh y
                    % bjh: se cancela en ajh/(ajh+bjh) y arriesga Inf/Inf=NaN.
                    bjh = exp(log(1-pij(j)+realmin) ...
                              + sum(log(normpdf(y(Si==h), ...
                                  X(Si==h,1)*beta0h(h) ...
                                  + Xnoint(Si==h,1:j-1)*betajh(h,1:j-1)' ...
                                  + Xnoint(Si==h,j+1:end)*betajh(h,j+1:end)', ...
                                  1/sqrt(tauh(h))) + realmin))) + realmin;
                    ajh = exp(log(pij(j)+realmin) ...
                              + sum(log(normpdf(ystar, 0, 1/sqrt(tauh(h))) + realmin)) ...
                              + log(normpdf(0, mubj, 1/sqrt(taubj)) + realmin) ...
                              - log(normpdf(0, inv(tauh(h)*Xnoint(Si==h,j)'*Xnoint(Si==h,j)+taubj) ...
                                           * (tauh(h)*Xnoint(Si==h,j)'*ystar + taubj*mubj), ...
                                           sqrt(inv(tauh(h)*Xnoint(Si==h,j)'*Xnoint(Si==h,j)+taubj))) + realmin));
                    gammajh(h,j) = double(rand(1,1) < ajh/(ajh+bjh));
                end
            end
        end

        % ── Registrar trazas ──────────────────────────────────────────────────
        osumout(gt,:)      = (sum(gammajh(1:max(Si),:)==1, 1) == 0);
        muout(gt)          = mu;
        tauhout(gt,:)      = tauh';
        beta0hout(gt,:)    = beta0h';
        betajhout(gt,:,:)  = betajh;
        alphahout(gt,:)    = alphah';
        psijhout(gt,:,:)   = psijh;
        Gammajhout(gt,:,:) = Gammajh;
        gammajhout(gt,:,:) = gammajh;
        pijout(gt,:)       = pij';
        wjout(gt,:)        = wj';
        N1out(gt)          = max(Si);
        Nout(gt)           = N;
        inEout(gt,:)       = sum(inE, 2);

        % ── Progreso (cada 200 iter para no saturar logs de parfor) ───────────
        if mod(gt, 200) == 0
            elapsed = toc(t_start);
            eta     = elapsed/gt * (nsim-gt);
            phase   = 'burn-in';  if gt > burn, phase = 'muestreo'; end
            fprintf('  [seed=%d] iter %4d/%d  N_act=%-2d  (%s, eta %.0fmin)\n', ...
                    seed, gt, nsim, max(Si), phase, eta/60);
        end
    end

    % ── Guardar ───────────────────────────────────────────────────────────────
    save(out_path, '-v7', ...
        'betajhout','beta0hout','tauhout','alphahout','psijhout','Gammajhout', ...
        'gammajhout','pijout','wjout','muout','osumout','N1out','Nout','inEout', ...
        'nsim','burn','N','M','p','n','feature_names','seed');

    fprintf('  ✓ Guardado: %s\n', out_path);
end

% ============================================================================
% FUNCIONES AUXILIARES
% ============================================================================

function x = rgamma(a, b, r, c)
    x = zeros(r, c);
    for k = 1:r*c
        if a >= 1
            d = a - 1/3; cv = 1/sqrt(9*d);
            while true
                z = randn; v = (1+cv*z)^3;
                if v > 0 && log(rand) < 0.5*z^2 + d - d*v + d*log(v)
                    x(k) = d*v*b; break
                end
            end
        else
            x(k) = rgamma(a+1, b, 1, 1) * rand^(1/a);
        end
    end
end

function x = rbeta(a, b)
    ga = rgamma(a, 1, 1, 1);
    gb = rgamma(b, 1, 1, 1);
    x  = ga / (ga + gb);
end

function x = rmvnorm(mu, Sigma)
    L = chol(Sigma, 'lower');
    x = (mu(:) + L * randn(length(mu), 1))';
end

function k = rdiscrete(p)
    p = p(:)' / sum(p);
    k = find(rand <= cumsum(p), 1, 'first');
    if isempty(k), k = length(p); end
end

function y = log_Phi(x)
% [EXP-1] log Phi(x) estable en ambas colas.
    y  = zeros(size(x));
    ng = x < 0;
    y(ng)  = log(0.5 * erfcx(-x(ng) / sqrt(2))) - x(ng).^2 / 2;
    y(~ng) = log1p(-0.5 * erfc(x(~ng) / sqrt(2)));
end

function x = rtnorm_pos(m, v)
% [EXP-1] x ~ N(m, v) truncada a (0, inf). Inversa por erfc; Robert (1995) si
% el corte queda a mas de 25 sd.
    s = sqrt(v);
    a = -m / s;
    if a < 25
        z = sqrt(2) * erfcinv(rand * erfc(a / sqrt(2)));
    else
        lam = (a + sqrt(a^2 + 4)) / 2;
        while true
            z = a - log(rand) / lam;
            if rand <= exp(-(z - lam)^2 / 2), break; end
        end
    end
    x = max(m + s * z, 0);
end
