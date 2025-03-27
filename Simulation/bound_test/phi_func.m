function [meanVal, statError] = phi_func(log_S_T_mc, K)
%PHI_FUNC Calculate payoff according to Monte Carlo results
    num_sample = size(log_S_T_mc,1);
    val_mc = max(exp(log_S_T_mc) - K, 0);
    meanVal = mean(val_mc);
    statError = std(val_mc,0) / sqrt(num_sample);
end

