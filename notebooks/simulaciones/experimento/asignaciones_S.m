function [Si, phxi] = asignaciones_S(ETA, y, X, beta0h, betajh, tauh, u)
% [EXP-0] Version vectorizada del bucle de S_i de psbp_train.m.
    V    = normcdf(ETA, 0, 1);                                   % (n, N-1)
    cp   = cumprod(1 - V, 2);
    phxi = [V .* [ones(size(V, 1), 1), cp(:, 1:end-1)], cp(:, end)];
    media = X(:, 1) * beta0h(:)' + X(:, 2:end) * betajh';        % (n, N)
    sd    = 1 ./ sqrt(tauh(:)');                                 % (1, N)
    % densidad normal escrita a mano: normpdf de Octave no expande dimensiones
    lik   = exp(-0.5 * ((y(:) - media) ./ sd).^2) ./ (sqrt(2*pi) * sd);
    ph1   = exp(log(phxi + realmin) + log(lik + realmin));
    ph1   = ph1 ./ sum(ph1, 2);
    C     = cumsum(ph1, 2);
    C(:, end) = 1;
    [~, Si] = max(C >= u, [], 2);
end
