function V_mat_mc = simu_hyb(spotV, num_maturity, dt_maturity,...
    num_moneyness, num_path_mc, steps_per_maturity, forward_curve,...
    theta, rho, lambda, nu, H, noise_mat_3d_mc, J, r, eps)
%IV_SIMU_HYBRID_SCHEME Hybrid scheme iv simulation under risk-neutral
%measure (benchmark)
    num_steps_mc = num_maturity * steps_per_maturity;
    dt = dt_maturity / steps_per_maturity;
    V_mat_mc = spotV * ones(num_path_mc, num_steps_mc+1) + [0, forward_curve]; % initialize variance paths
    logS_mc_current = zeros(num_path_mc,1); % current stock price = 1

    kernel_length = num_steps_mc*2;
    kernel_dt = (linspace(1,kernel_length,kernel_length).^(H+1/2) - ...
        linspace(0,kernel_length-1,kernel_length).^(H+1/2)) / (H+1/2);
    gamma_val = gamma(H+1/2);

    for i = 1:num_steps_mc
        f_mc_current = sqrt(max(V_mat_mc(:,i), eps));
        logS_mc_current = logS_mc_current + (r - f_mc_current.^2/2) * dt...
            + f_mc_current .* (rho * noise_mat_3d_mc(:,i,1) + sqrt(1-rho^2) * noise_mat_3d_mc(:,i,2));
        
        % update volatility path
        j_vec = i+1 : num_steps_mc+1;
        adj_J = min(J, num_steps_mc+1-i);
        tmpd = nu / gamma_val * f_mc_current;
        tmpc = -dt^(H+1/2) * lambda / gamma_val * V_mat_mc(:,i); 
        V_mat_mc(:,j_vec) = V_mat_mc(:,j_vec) + tmpc * kernel_dt(j_vec-i)...
            + tmpd * dt^(H-1/2) .* [zeros(1,adj_J), kernel_dt(J+1:num_steps_mc+1-i)] .* noise_mat_3d_mc(:,i,1); 
        for j = 1:adj_J
            V_mat_mc(:,i+j) = V_mat_mc(:,i+j) + tmpd .* noise_mat_3d_mc(:,i,j+2); % dt^(H-1/2) not applicable here
        end

        disp(i);
    end
end

