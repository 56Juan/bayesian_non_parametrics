function [Zil, Wil] = latentes_Z(ETA, alphah, Si, N, M_TRUNC, Q_HI, U)
% [EXP-0] Version vectorizada del bucle de Z* de psbp_train.m: para l <= S_i
% (l <= N-1 si S_i = N), Z*_il ~ N(m_il, 1) truncada a (-inf,0) si l < S_i y a
% (0,inf) si l = S_i, con m recortada a [-M_TRUNC, M_TRUNC]; W = Z* + sum psi|x-Gamma|.
    n  = size(ETA, 1);
    l  = 1:(N - 1);
    activo   = l <= min(Si(:), N - 1);
    positivo = l == Si(:);
    m  = min(max(ETA, -M_TRUNC), M_TRUNC);
    c0 = normcdf(-m, 0, 1);
    q  = U .* c0;
    q(positivo) = U(positivo) + (1 - U(positivo)) .* c0(positivo);
    Z  = m + norminv(min(max(q, realmin), Q_HI), 0, 1);
    Zil = zeros(n, N);
    Wil = zeros(n, N);
    Zil(:, 1:N-1) = Z .* activo;
    Wil(:, 1:N-1) = (Z + (alphah(:)' - ETA)) .* activo;
end
