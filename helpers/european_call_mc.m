function price = european_call_mc(logS_mc,K)
    %EUROPEAN_CALL_MC Calculate European call option price based on Monte Carlo results
    % INPUT
    %   logS_mc  -  (# mc paths x 1) array
    %   K        -  strike price (scalar)
    
    price = mean(max(exp(logS_mc) - K, 0));
end

