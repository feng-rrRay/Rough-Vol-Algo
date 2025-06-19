function [rmse_val,standard_error,bias] = rmse_mc(logS_mc,K,refPrice)
    %RMSE_MC Compute rooted mean squared error(RMSE) for Monte Carlo estimators
    %   RMSE = sqrt(standard error^2 + bias^2)
    num_path_mc = size(logS_mc, 1);
    payoffs = max(exp(logS_mc) - K, 0);
    % standard error
    var_mc_estimator = var(payoffs) / num_path_mc;
    standard_error = sqrt(var_mc_estimator);
    % bias
    mean_payoffs = mean(payoffs);
    bias = abs(mean_payoffs - refPrice);   % AKA weak error in discretization methods
    rmse_val = sqrt(var_mc_estimator + bias^2);
end

