function psbp_train(y, Xnoint, hp, mcmc, out_path, feature_names)
% PSBP_TRAIN  Sampler MCMC del modelo PSBP-FD para UNA regresión.
%
%   psbp_train(y, Xnoint, hp, mcmc, out_path, feature_names)
%
% ENTRADAS
%   y            (n,1)   Respuesta (scores FPCA, ya en su escala final)
%   Xnoint       (n,p)   Predictores SIN columna de unos
%   hp           struct  Hiperparámetros del modelo:
%                          .atau .btau .ag .bg .mumu .taumu .pwj
%                          .apij   (p,1)
%                          .bpij   (p,1)
%                          .mupsij (p,1)
%                          .taupsij(p,1)
%   mcmc         struct  Configuración MCMC:
%                          .nsim  .burn  .N  .M
%   out_path     char    Ruta completa del archivo .mat de salida
%   feature_names char   Nombres de las covariables separados por coma
%
% SALIDA
%   Archivo .mat en out_path con las trazas MCMC, consumible desde Python
%   (mismos nombres que PSBP_FD_v2.traces).
%
% NOTA: inv() se conserva verbatim del original para mantener paridad
%       con el sampler Python v2.

    % ── Dimensiones ──────────────────────────────────────────────────────────
    n = size(Xnoint, 1);
    p = size(Xnoint, 2);
    X = horzcat(ones(n,1), Xnoint);     % (n, p+1)

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
    apij    = hp.apij(:);       % (p,1)
    bpij    = hp.bpij(:);
    mupsij  = hp.mupsij(:);
    taupsij = hp.taupsij(:);

    % ── Grid de localización G* ───────────────────────────────────────────────
    Gstar = (min(min(X)) + ((1:M)/M) .* (max(max(X)) - min(min(X))))';

    % ── Inicialización ────────────────────────────────────────────────────────
    bjrange  = [-4,-3,-2,-1.5,-1,0,1,1.5,2,3,4,5]';
    b0range  = [-4,-3,-2,-1.5,-1,0,1,1.5,2,3,4,5]';
    betajh   = bjrange(randi(7, N, p));
    beta0h   = b0range(randi(7, N, 1));
    tauh     = rgamma(atau, 1/btau, N, 1);
    Si       = randi(N, n, 1);
    alphah   = zeros(N-1, 1);
    psijh    = zeros(N-1, p);
    gammajh  = ones(N, p);
    Gloc     = randi(M, N-1, p);
    Gammajh  = Gstar(Gloc);
    mu       = 1;
    g        = 1;
    pij      = 0.5 * ones(p, 1);
    wj       = ones(p, 1);

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

    % ════════════════════════════════════════════════════════════════════════
    % BUCLE PRINCIPAL MCMC
    % ════════════════════════════════════════════════════════════════════════
    for gt = 1:nsim

        % ── Actualizar Zil, Wil ──────────────────────────────────────────────
        Zil = zeros(n, N);
        Wil = zeros(n, N);
        for i = 1:n
            lmax = Si(i);
            for l = 1:min(lmax, N-1+(lmax==N))
                u = rand(1,1);
                m_val = alphah(l) - sum(psijh(l,:) .* abs(Xnoint(i,:) - Gammajh(l,:)), 2);
                v_val = 1;
                if Si(i) < N
                    if l < Si(i)
                        Zil(i,l) = m_val + sqrt(v_val)*norminv(u*normcdf((0-m_val)/sqrt(v_val),0,1), 0,1);
                    else  % l == Si(i)
                        Zil(i,l) = m_val + sqrt(v_val)*norminv(u+(1-u)*normcdf((0-m_val)/sqrt(v_val),0,1), 0,1);
                    end
                else  % Si(i) == N
                    Zil(i,l) = m_val + sqrt(v_val)*norminv(u*normcdf((0-m_val)/sqrt(v_val),0,1), 0,1);
                end
                Wil(i,l) = Zil(i,l) + sum(psijh(l,:) .* abs(X(i,2:end) - Gammajh(l,:)));
            end
        end

        % ── Actualizar Si ─────────────────────────────────────────────────────
        phxi = zeros(n, N);
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
            phx(N) = prod(1 - vhx);
            phxi(i,:) = phx';
            phx1  = exp(log(phx+realmin) + log(normpdf(y(i), X(i,1)*beta0h(:,1) + betajh(:,:)*X(i,2:end)', 1./sqrt(tauh)) + realmin));
            phx12 = phx1 / sum(phx1);
            Si(i) = rdiscrete(phx12);
        end

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
                for k = l+1:pgh
                    Shhat(l,k) = Shhat(k,l);
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
            betagh = horzcat(beta0h(h), betajh(h, gammajh(h,:)==1));
            Xh     = X(:, horzcat(1, gammajh(h,1:p)) == 1);
            aa     = atau + 0.5*sum(Si==h) + 0.5*sum(gammajh(h,1:p)) + 0.5;
            bb     = btau + 0.5*(y(Si==h) - Xh(Si==h,:)*betagh')'*(y(Si==h) - Xh(Si==h,:)*betagh') ...
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
                b_val    = exp(gammaln(bpij(j)+N) + gammaln(apij(j)+bpij(j)) ...
                             - gammaln(bpij(j)) - gammaln(apij(j)+bpij(j)+N));
                pwjhat   = pwj*b_val / ((1-pwj)*1 + pwj*b_val);
                wj(j)    = double(rand(1,1) < pwjhat);
            end
        end

        % ── Actualizar alphah ─────────────────────────────────────────────────
        for h = 1:N-1
            v_val    = inv(1 + sum(Si >= h));
            m_val    = v_val * (mu + sum(Wil(Si >= h, h)));
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
        for h = 1:N-1
            kh = sum(Si >= h);
            for j = 1:p
                if gammajh(h,j) == 1
                    pm = zeros(M, 1);
                    for m_idx = 1:M
                        pm(m_idx) = exp(1.2*kh + sum(log(normpdf(Zil(Si>=h,h), ...
                            alphah(h) ...
                            - sum(repmat(psijh(h,1:j-1),   kh,1) .* abs(Xnoint(Si>=h,1:j-1)   - repmat(Gammajh(h,1:j-1),   kh,1)), 2) ...
                            - sum(repmat(psijh(h,j+1:end), kh,1) .* abs(Xnoint(Si>=h,j+1:end) - repmat(Gammajh(h,j+1:end), kh,1)), 2) ...
                            - repmat(psijh(h,j), kh,1) .* abs(Xnoint(Si>=h,j) - Gstar(m_idx)), ...
                            1) + realmin))) + realmin;
                    end
                    pm1          = pm / sum(pm);
                    Gammajh(h,j) = Gstar(rdiscrete(pm1));
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
                    Tijh = alphah(h) - Zil(Si>=h,h) ...
                           - sum(repmat(psijh(h,1:j-1),   kh,1) .* abs(Xnoint(Si>=h,1:j-1)   - repmat(Gammajh(h,1:j-1),   kh,1)), 2) ...
                           - sum(repmat(psijh(h,j+1:end), kh,1) .* abs(Xnoint(Si>=h,j+1:end) - repmat(Gammajh(h,j+1:end), kh,1)), 2);
                    v_val = inv(taupsij(j) + (Xnoint(Si>=h,j) - Gammajh(h,j))' * (Xnoint(Si>=h,j) - Gammajh(h,j)));
                    m_val = v_val * (taupsij(j)*mupsij(j) + sum(Tijh .* abs(Xnoint(Si>=h,j) - Gammajh(h,j))));
                    u     = rand(1,1);
                    psijh(h,j) = m_val + sqrt(v_val)*norminv(u + (1-u)*normcdf((0-m_val)/sqrt(v_val),0,1), 0,1);
                    if psijh(h,j) == inf
                        psijh(h,j) = 0.01;
                    end
                end
            end
        end

        % ── Actualizar gammajh ────────────────────────────────────────────────
        for h = 1:N
            for j = 1:p
                gammajh1       = gammajh(h,1:p); gammajh1(:,j) = [];
                Xh_tmp         = X;              Xh_tmp(:,j+1) = [];
                Xh1            = horzcat(Xnoint(:,j), Xh_tmp(:, horzcat(1,gammajh1)==1));
                sb             = n/g * inv(Xh1'*Xh1) / tauh(h);
                betagh_tmp     = horzcat(beta0h(h), betajh(h,:)); betagh_tmp(:,j+1) = [];
                betagh2        = betagh_tmp(:, horzcat(1,gammajh1)==1);
                sbj            = sb(1,1) - sb(1,2:end)*inv(sb(2:end,2:end))*sb(1,2:end)';
                taubj          = 1/sbj;
                mubj           = sb(1,2:end)*inv(sb(2:end,2:end))*betagh2';
                ystar          = y(Si==h) - X(Si==h,1)*beta0h(h) ...
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
                    bjh = exp(1.2*nh + log(1-pij(j)+realmin) ...
                              + sum(log(normpdf(y(Si==h), ...
                                  X(Si==h,1)*beta0h(h) ...
                                  + Xnoint(Si==h,1:j-1)*betajh(h,1:j-1)' ...
                                  + Xnoint(Si==h,j+1:end)*betajh(h,j+1:end)', ...
                                  1/sqrt(tauh(h))) + realmin))) + realmin;
                    ajh = exp(1.2*nh + log(pij(j)+realmin) ...
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
        inEout(gt,:)       = sum(inE, 2)';

        if mod(gt, 100) == 0
            fprintf('    iter %4d / %d   (N_activo=%d)\n', gt, nsim, max(Si));
        end
    end

    % ── Guardar traza ─────────────────────────────────────────────────────────
    save(out_path, '-v7', ...
        'betajhout','beta0hout','tauhout','alphahout','psijhout','Gammajhout', ...
        'gammajhout','pijout','wjout','muout','osumout','N1out','Nout','inEout', ...
        'nsim','burn','N','M','p','n','feature_names');

    fprintf('  ✓ Traza guardada: %s  (nsim=%d, burn=%d)\n', out_path, nsim, burn);
end

% ============================================================================
% FUNCIONES AUXILIARES — reemplazos sin Statistics Toolbox
% ============================================================================

function x = rgamma(a, b, r, c)
% Gamma(a, b) donde b es la escala (= 1/rate).
% Usa el algoritmo de Marsaglia-Tsang (2000).
    x = zeros(r, c);
    for k = 1:r*c
        if a >= 1
            d = a - 1/3;  cv = 1/sqrt(9*d);
            while true
                z = randn; v = (1 + cv*z)^3;
                if v > 0 && log(rand) < 0.5*z^2 + d - d*v + d*log(v)
                    x(k) = d*v*b;  break
                end
            end
        else
            x(k) = rgamma(a+1, b, 1, 1) * rand^(1/a);
        end
    end
end

function x = rbeta(a, b)
% Beta(a,b) via cociente de Gammas.
    ga = rgamma(a, 1, 1, 1);
    gb = rgamma(b, 1, 1, 1);
    x  = ga / (ga + gb);
end

function x = rmvnorm(mu, Sigma)
% Normal multivariada via Cholesky.
    L = chol(Sigma, 'lower');
    x = (mu(:) + L * randn(length(mu), 1))';
end

function k = rdiscrete(p)
% Muestreo de una categoría discreta con probabilidades p.
    p  = p(:)' / sum(p);
    k  = find(rand <= cumsum(p), 1, 'first');
    if isempty(k), k = length(p); end
end
