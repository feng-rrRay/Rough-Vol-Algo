function func_val = phi_func(V_T_mc, log_S_T_mc, K)
%PHI_FUNC a local Lipschitz continuous function
    val_mc = max(exp(log_S_T_mc) - K, 0);
    func_val = mean(val_mc);
    % func_val = mean(log_S_T_mc);
end

