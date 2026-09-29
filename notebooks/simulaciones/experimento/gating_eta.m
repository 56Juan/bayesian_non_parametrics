function ETA = gating_eta(alphah, psijh, Gammajh, Xnoint)
% [EXP-0] ETA(i,h) = alpha_h - sum_j psi_hj |x_ij - Gamma_hj|, h = 1..N-1.
    ETA = repmat(alphah(:)', size(Xnoint, 1), 1);
    for j = 1:size(Xnoint, 2)
        ETA = ETA - psijh(:, j)' .* abs(Xnoint(:, j) - Gammajh(:, j)');
    end
end
