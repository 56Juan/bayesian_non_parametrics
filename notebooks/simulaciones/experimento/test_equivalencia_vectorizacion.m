% test_equivalencia_vectorizacion.m — [EXP-0]
%
% Verifica que latentes_Z.m y asignaciones_S.m reproducen EXACTAMENTE los
% bucles de 107_sim_C1/psbp_train.m cuando reciben los mismos numeros
% aleatorios. Los bucles de referencia de abajo son copia literal del original,
% con rand reemplazado por U(i,l) y u(i). Correr desde esta carpeta.

clear; rng(123, 'twister');
n = 300; p = 5; N = 12;
Xnoint  = randn(n, p);  X = [ones(n,1), Xnoint];  y = randn(n, 1);
alphah  = 2 * randn(N-1, 1);
psijh   = abs(randn(N-1, p)) .* (rand(N-1, p) < 0.6);
Gammajh = randn(N-1, p);
beta0h  = randn(N, 1);  betajh = randn(N, p);  tauh = 0.5 + rand(N, 1);
Si      = randi(N, n, 1);
M_TRUNC = 6;  Q_HI = 1 - eps/2;
U = rand(n, N-1);  u = rand(n, 1);

% ── referencia: bucle de Z* del original ────────────────────────────────────
Zr = zeros(n, N);  Wr = zeros(n, N);
for i = 1:n
    if Si(i) < N, L_ = Si(i); else, L_ = N-1; end
    for l = 1:L_
        m_val = alphah(l) - sum(psijh(l,:) .* abs(Xnoint(i,:) - Gammajh(l,:)), 2);
        m_val = min(max(m_val, -M_TRUNC), M_TRUNC);
        if Si(i) < N && l == Si(i)
            q = U(i,l) + (1-U(i,l))*normcdf(-m_val, 0, 1);
        else
            q = U(i,l)*normcdf(-m_val, 0, 1);
        end
        Zr(i,l) = m_val + norminv(min(max(q, realmin), Q_HI), 0, 1);
        Wr(i,l) = Zr(i,l) + sum(psijh(l,:) .* abs(X(i,2:end) - Gammajh(l,:)));
    end
end

% ── referencia: bucle de S del original ─────────────────────────────────────
Sr = zeros(n, 1);  Pr = zeros(n, N);
for i = 1:n
    vhx = ones(N-1, 1);  phx = ones(N, 1);
    for h = 1:N-1
        vhx(h) = normcdf(alphah(h) - sum(psijh(h,:) .* abs(X(i,2:end) - Gammajh(h,:))), 0, 1);
        if h == 1, phx(h) = vhx(h); else, phx(h) = vhx(h) * prod(1 - vhx(1:h-1)); end
    end
    phx(N) = prod(1 - vhx);  Pr(i,:) = phx';
    phx1 = exp(log(phx+realmin) + log(normpdf(y(i), X(i,1)*beta0h(:,1) + betajh(:,:)*X(i,2:end)', 1./sqrt(tauh)) + realmin));
    pr = phx1 / sum(phx1);
    k = find(u(i) <= cumsum(pr'), 1, 'first');  if isempty(k), k = N; end
    Sr(i) = k;
end

% ── vectorizado ─────────────────────────────────────────────────────────────
ETA = gating_eta(alphah, psijh, Gammajh, Xnoint);
[Zv, Wv]  = latentes_Z(ETA, alphah, Si, N, M_TRUNC, Q_HI, U);
[Sv, Pv]  = asignaciones_S(ETA, y, X, beta0h, betajh, tauh, u);

errZ = max(abs(Zv(:) - Zr(:)));  errW = max(abs(Wv(:) - Wr(:)));
errP = max(abs(Pv(:) - Pr(:)));  nS = sum(Sv ~= Sr);
fprintf('max|Z* vec - bucle| = %.2e\nmax|W vec - bucle|  = %.2e\n', errZ, errW);
fprintf('max|pi vec - bucle| = %.2e\nS distintos         = %d de %d\n', errP, nS, n);
assert(errZ < 1e-10 && errW < 1e-10 && errP < 1e-12 && nS == 0, 'La vectorizacion NO reproduce los bucles.');
fprintf('OK: la vectorizacion reproduce los bucles de psbp_train.m\n');
