function V_mat_mc = simu_acc(spotV, num_maturity, dt_maturity,...
    num_moneyness, num_path_mc, steps_per_maturity, u_current, forward_curve, cs, xs, ...,
    theta, rho, lambda, nu, H, noise_mat_3d_mc, J, r, eps)
    %IV_SIMU_PC Predictor-Corrector sampler
    %   Predictor: implicit-explicit scheme for multi-factor approximation
    %   Corrector: hybrid scheme with fft
    %   Overall complexity: O( N * (nf + logN) )
    num_steps_mc = num_maturity * steps_per_maturity;
    dt = dt_maturity / steps_per_maturity;
    num_factor = size(cs, 2);
    u_current_mc = ones(num_path_mc,1) * u_current;
    V_mat_mc = spotV * ones(num_path_mc, num_steps_mc+1) + [0, forward_curve]; % initialize variance paths
    gamma_val = gamma(H+1/2);

    % Markovian
    for i = 1:num_steps_mc
        f_mc_current = max(V_mat_mc(:,i), eps);
        delta_u = -lambda*V_mat_mc(:,i)*dt + nu*sqrt(f_mc_current).*noise_mat_3d_mc(:,i,1);
        u_current_mc = (u_current_mc + delta_u) ./ (1 + xs*dt);
        V_mat_mc(:,i+1) = V_mat_mc(:,i+1) + sum(cs .* u_current_mc,2); 
    end

    % Hybrid-like
    kernel_dt = ones(num_path_mc,1) * [0, linspace(1,num_steps_mc,num_steps_mc).^(H+1/2) - ...
        linspace(0,num_steps_mc-1,num_steps_mc).^(H+1/2)] / (H+1/2);
    f_mat_mc = sqrt(max(V_mat_mc, eps));
    noise_path_mc = squeeze(noise_mat_3d_mc(:,:,1));
    var_path_diff = nu * dt^(-1/2) * f_mat_mc .* [noise_path_mc,zeros(num_path_mc,1)];
    var_path_drift = lambda * dt^(1/2) * V_mat_mc;
    tmp_N = 2^nextpow2(2*num_steps_mc-1);
    kernel_fft = fft(kernel_dt,tmp_N,2);
    var_fft = fft(var_path_diff - var_path_drift,tmp_N,2);
    res = ifft(kernel_fft.*var_fft,tmp_N,2) * dt^H / gamma_val;
    V_mat_mc = spotV * ones(num_path_mc, num_steps_mc+1) + [0, forward_curve] ...
        + res(:,1:num_steps_mc+1);
    for i = 1:num_steps_mc
        f_mc_current = sqrt(max(V_mat_mc(:,i), eps));
        adj_J = min(J, num_steps_mc+1-i);
        for j = 1 : adj_J
            % notice `f_mc_current` and `f_mat_mc` here, `f_mat_mc` are
            % previous values to deduct
            V_mat_mc(:,i+j) = V_mat_mc(:,i+j) + (nu * f_mc_current .* noise_mat_3d_mc(:,i,j+2) - ...
                dt^(H-1/2) * nu * f_mat_mc(:,i) .* kernel_dt(:,j+1) .* noise_mat_3d_mc(:,i,1)) / gamma_val;
        end
    end
    
end