function output = IVCurveArray(logStrikes, H, rho, nu, lambda, theta, V0, t, r, d, n, M, N)

    S = ones(size(logStrikes));
    K = exp(logStrikes);
    callPrices = CallPriceArray(S, K, H, rho, nu, lambda, theta, V0, t, r, d, n, M, N);
    
    idx = (isinf(callPrices) | isnan(callPrices));

    sigma1 = ones(size(S)) * 1e-6;
    sigma2 = ones(size(S)) * 10;

    sigma1(idx) = 10;
    sigma2(idx) = 10;
    
    while any(sigma2 - sigma1 > 1e-6, 'all')
        
        sigma = (sigma1 + sigma2) / 2;
        callPricesNew = callPrice(S, K, t, r, sigma);
        idx1 = (callPricesNew < callPrices);
        idx2 = (callPricesNew >= callPrices);
        sigma1(idx1) = sigma(idx1);
        sigma2(idx2) = sigma(idx2);

    end
    output = (sigma1 + sigma2) / 2;


end


function output = callPrice(S, K, T, r, sigma)
    T = T.';
    d1 = (log(S ./ K) + (r + sigma.^2 / 2) .* T) ./ (sigma .* sqrt(T));
    d2 = (log(S ./ K) + (r - sigma.^2 / 2) .* T) ./ (sigma .* sqrt(T));
    output = S .* normcdf(d1) - exp(-r * T) .* K .* normcdf(d2);
end