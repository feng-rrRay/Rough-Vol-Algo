function C = CallPriceArray(S, K, H, rho, nu, lambda, theta, V0, t, r, d, n, M, N)

    k = log(K ./ S);

    [Pi1, Pi2] = PiArray(k, H, rho, nu, lambda, theta, V0, t, r, d, n, M, N);

    t = t.';

    C = S .* Pi1 - exp(-r * t) .* K .* Pi2;

end