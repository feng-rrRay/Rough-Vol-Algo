function result = CFArray(H, rho, nu, lambda, theta, V0, u, t, r, d, n)
    [hs, Fs] = AdamsFractionalEquationArray(H, rho, nu, lambda, u, t, n);

    t = t.';
    hs(:, :, 1) = hs(:, :, 1) / 2;
    hs(:, :, n+1) = hs(:, :, n+1) / 2;
    G = sum(hs, 3) .* t / n;

    Fs(:, :, 1) = Fs(:, :, 1) / 2;
    Fs(:, :, n+1) = Fs(:, :, n+1) / 2;
    H = sum(Fs, 3) .* t / n;

    %disp(size(t_3d));
    %disp(size(u_3d));
    %disp(size((r - d) .* t_3d .* u_3d .* 1i));
    %disp(size(lambda * theta * G + V0 * H));
    result = exp((r - d) .* t .* u .* 1i + lambda * theta * G + V0 * H);
end