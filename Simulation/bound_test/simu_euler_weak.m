function [logS_mc_final,t] = simu_euler_weak(spotV, num_maturity, dt_maturity,...
    num_moneyness, num_path_mc, steps_per_maturity, u_current, forward_curve, cs, xs, ...,
    theta, rho, lambda, nu, H, noise_mat_3d_mc, J, r, eps)
    
    tic;
    num_steps_mc = num_maturity * steps_per_maturity;
    dt = dt_maturity / steps_per_maturity;
    % num_factor = size(cs, 2);
    % u_current_mc = ones(num_path_mc,1) * u_current;
    V_mat_mc = spotV * ones(num_path_mc, num_steps_mc+1); % initialize variance paths
    logS_mc_current = zeros(num_path_mc,1);
    % option_price_mat = zeros(num_maturity, num_moneyness);
    % moneyness_mat = zeros(num_maturity, num_moneyness);
    % tau_vec = zeros(num_maturity,1);
    gamma_val = gamma(H+1/2);
    
    for i = 1:num_steps_mc
        f_mc_current = sqrt(max(V_mat_mc(:,i),eps));
        logS_mc_current = logS_mc_current + (r - f_mc_current.^2/2) * dt + ...
            f_mc_current .* (rho*noise_mat_3d_mc(:,i,1) + sqrt(1-rho^2)*noise_mat_3d_mc(:,i,2));
        for j = 1:i
            f_mc_j = sqrt(max(V_mat_mc(:,j),eps));
            inner = lambda * (theta - f_mc_j.^2) * dt + nu * f_mc_j .* noise_mat_3d_mc(:,j,1);
            V_mat_mc(:,i+1) = V_mat_mc(:,i+1) + ((i-j+1)*dt)^(H-1/2) * inner / gamma_val;
        end
    end
    logS_mc_final = logS_mc_current;
    t = toc;
end

