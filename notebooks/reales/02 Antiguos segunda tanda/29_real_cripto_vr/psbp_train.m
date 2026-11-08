function psbp_train(y, Xnoint, hp, mcmc, out_path, feature_names, seed)
% PSBP_TRAIN  Sampler MCMC del modelo PSBP-FD para una cadena.
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
%
% Corrida 113 (sobre la 107):
%   [FIX 3] G* POR PREDICTOR: Gstar es (M, p), columna j equiespaciada en
%           [min x_j, max x_j] incluyendo ambos extremos. Antes era una sola
%           grilla con el min/max global de Xnoint y sin el extremo inferior.
%   [DIAG]  gamdiagout (nsim, 4) registra el paso de Gamma en cada iteracion:
%           [n de sorteos, n con pm1 no finito, max |sum(pm1)-1|,
%            max (max(log pm) - min(log pm))]. Se guardan tambien Gstar.
%   [FIX 4] Z | S se sortea DESPUES de actualizar S, como en el apendice del
%           paper. Antes (y en Case1_01.m) Z salia del S anterior.
%   [FIX 5] Normal truncada exacta (rtn_pos) para Z y psi; se elimina el
%           recorte de la media a [-6, 6], que cambiaba la condicional.
%   [FIX 6] inEout con el estado final de la iteracion (antes, el anterior).

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
    % [FIX 3] Una columna por predictor, con los dos extremos: (M, p).
    xmin  = min(Xnoint, [], 1);
    xmax  = max(Xnoint, [], 1);
    Gstar = xmin + ((0:M-1)' / (M-1)) .* (xmax - xmin);

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
    Gammajh = Gstar(sub2ind([M, p], Gloc, repmat(1:p, N-1, 1)));
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
    gamdiagout = zeros(nsim, 4);

    t_start = tic;

    % ════════════════════════════════════════════════════════════════════════
    % BUCLE PRINCIPAL MCMC
    % ════════════════════════════════════════════════════════════════════════
    for gt = 1:nsim

        % ── Actualizar Si ─────────────────────────────────────────────────────
        for i = 1:n
            vhx = ones(N-1, 1);
            phx = ones(N, 1);
            for h = 1:N-1
                vhx(h) = normcdf(alphah(h) - sum(psijh(h,:) .* abs(X(i,2:end) - Gammajh(h,:))), 0, 1);
                if h == 1
                    phx(h) = vhx(h);
                else
                    phx(h) = vhx(h) * prod(1 - vhx(1:h-1));
                end
            end
            phx(N)    = prod(1 - vhx);
            phx1  = exp(log(phx+realmin) + log(normpdf(y(i), X(i,1)*beta0h(:,1) + betajh(:,:)*X(i,2:end)', 1./sqrt(tauh)) + realmin));
            phx12 = phx1 / sum(phx1);
            Si(i) = rdiscrete(phx12);
        end

        % ── Actualizar Zil | Si ───────────────────────────────────────────────
        % [FIX 4] Z se sortea DESPUES de S (apendice: paso 1 = S, luego la
        % aumentacion). Antes se sorteaba con el S de la iteracion anterior y
        % alpha/psi/Gamma/gamma leian celdas vacias o con el signo cambiado.
        % [FIX 5] Normal truncada exacta (rtn_pos), sin recortar la media a
        % [-6, 6]. Z_il < 0 para l < S_i, Z_il > 0 para l = S_i < N.
        D    = dist_gating(Xnoint, psijh, Gammajh);          % (n, N-1)
        eta  = alphah' - D;
        cols = 1:N-1;
        mask = cols <= min(Si, N-1);
        pos  = (cols == Si) & (Si < N);
        neg  = mask & ~pos;
        Zc   = zeros(n, N-1);
        Zc(pos) =  rtn_pos( eta(pos), 1);
        Zc(neg) = -rtn_pos(-eta(neg), 1);
        Zil = [Zc, zeros(n, 1)];
        Wil = [(Zc + D) .* mask, zeros(n, 1)];

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

        % ── Actualizar Gammajh ────────────────────────────────────────────────
        gd = zeros(1, 4);
        for h = 1:N-1
            kh = sum(Si >= h);
            for j = 1:p
                if gammajh(h,j) == 1
                    pm = zeros(M, 1);
                    for m_idx = 1:M
                        % [FIX 2] log-verosimilitud, no exp(sum(log)): con kh >~ 500
                        % exp(.) vale 0 en toda la grilla, pm queda constante y
                        % Gamma se sortea sin mirar los datos. Se normaliza abajo.
                        pm(m_idx) = sum(log(normpdf(Zil(Si>=h,h), ...
                            alphah(h) ...
                            - sum(repmat(psijh(h,1:j-1),   kh,1) .* abs(Xnoint(Si>=h,1:j-1)   - repmat(Gammajh(h,1:j-1),   kh,1)), 2) ...
                            - sum(repmat(psijh(h,j+1:end), kh,1) .* abs(Xnoint(Si>=h,j+1:end) - repmat(Gammajh(h,j+1:end), kh,1)), 2) ...
                            - repmat(psijh(h,j), kh,1) .* abs(Xnoint(Si>=h,j) - Gstar(m_idx, j)), ...
                            1) + realmin));
                    end
                    rango        = max(pm) - min(pm);
                    pm           = exp(pm - max(pm));
                    pm1          = pm / sum(pm);
                    gd(1) = gd(1) + 1;
                    gd(2) = gd(2) + any(~isfinite(pm1));
                    gd(3) = max(gd(3), abs(sum(pm1) - 1));
                    gd(4) = max(gd(4), rango);
                    Gammajh(h,j) = Gstar(rdiscrete(pm1), j);
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
                    % [FIX 5] N+(m, v) exacta, sin recortar la media.
                    psijh(h,j) = rtn_pos(m_val, sqrt(v_val));
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

        % ── E[y|x] in-sample ─────────────────────────────────────────────────
        % [FIX 6] Con el estado FINAL de la iteracion (pesos y betas que se
        % registran abajo). Antes usaba los de la iteracion anterior.
        vfin = normcdf(alphah' - dist_gating(Xnoint, psijh, Gammajh));   % (n, N-1)
        cfin = cumprod(1 - vfin, 2);
        wfin = [vfin(:,1), vfin(:,2:end) .* cfin(:,1:end-1), cfin(:,end)];
        inE  = wfin .* (beta0h' + Xnoint * betajh');                     % (n, N)

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
        gamdiagout(gt,:)   = gd;

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
        'gamdiagout','Gstar', ...
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

function D = dist_gating(Xnoint, psijh, Gammajh)
% D(i,h) = sum_j psi_hj |x_ij - Gamma_hj|, (n, N-1).
    [n, p] = size(Xnoint);
    Nm1    = size(psijh, 1);
    D = sum(reshape(psijh, 1, Nm1, p) .* abs(reshape(Xnoint, n, 1, p) ...
            - reshape(Gammajh, 1, Nm1, p)), 3);
end

function x = rtn_pos(m, s)
% x ~ N(m, s^2) truncada a (0, inf), elemento a elemento, sin recortar m.
% Inversion por la cola inferior, t = -Phi^-1(u Phi(-a)), a = -m/s, que es
% exacta mientras Phi(-a) no subdesborde (a < 35); mas alla, rechazo
% exponencial de Robert (1995).
    a = -m(:) ./ s(:);
    t = zeros(size(a));
    usa_inv = a < 35;
    t(usa_inv) = -norminv(rand(nnz(usa_inv), 1) .* normcdf(-a(usa_inv)));
    for r = find(~usa_inv)'
        lam = (a(r) + sqrt(a(r)^2 + 4)) / 2;
        while true
            z = a(r) - log(rand) / lam;
            if log(rand) <= -(z - lam)^2 / 2, t(r) = z; break; end
        end
    end
    x = reshape(m(:) + s(:) .* t, size(m));
    x = max(x, 0);          % solo redondeo: t >= a por construccion
end

function k = rdiscrete(p)
    p = p(:)' / sum(p);
    k = find(rand <= cumsum(p), 1, 'first');
    if isempty(k), k = length(p); end
end
