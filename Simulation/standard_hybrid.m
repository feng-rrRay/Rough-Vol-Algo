function results = standard_hybrid(spotV, num_maturity, dt_maturity,...
    num_moneyness, num_path_mc, steps_per_maturity, forward_curve,...
    theta, rho, lambda, nu, H, noise_mat_3d_mc, J, r, eps, mode)
    %STANDARD_HYBRID standard hybrid scheme by Bennedsen et al. (2017)
    %   INPUTS:
    %     spotV             – initial variance level (scalar)
    %     num_maturity      – number of maturities under 'iv' mode (integer)
    %     dt_maturity       – maturity time interval under 'iv' mode (scalar)
    %     num_moneyness     – # of strike/moneyness points per maturity for 'iv' mode
    %     num_path_mc       – number of Monte-Carlo paths
    %     steps_per_maturity– time‐steps per maturity
    %     forward_curve     – forward variance curve (1×num_steps)
    %     theta             – rough Heston model parameter (scalar)
    %     rho               – rough Heston model correlation (scalar)
    %     lambda            – rough Heston model mean‐reversion rate
    %     nu                – rough Heston model diffusion term multiplier (scalar)
    %     H                 – Hurst index
    %     noise_mat_3d_mc   – pre-generated noise: [paths × steps × (J+2)]
    %     J                 – number of additional noises for hybrid scheme
    %     r                 – risk-free rate
    %     eps               – small floor for variance
    %     mode              – output mode, one of:
    %                          'std'    : return logS_mc_final + timings
    %                          'iv'     : return implied‐vol matrix + maturities + moneyness
    %                          'full'   : return full V_mat paths
    %
    %   OUTPUT:
    %     RESULTS – struct with these possible fields:
    %       .logS_mc_final   – accelerated hybrid final log‐price
    %       .V_mat_mc        – full variance paths (only if mode='path')
    %       .iv_mat          – implied‐volatility surface (mode='iv')
    %       .tau_vec         – vector of maturities (mode='iv')
    %       .moneyness_mat   – grid of moneyness levels (mode='iv')
    %       .t               – CPU time

    tic;
    num_steps_mc = num_maturity * steps_per_maturity;
    dt = dt_maturity / steps_per_maturity;
    V_mat_mc = spotV * ones(num_path_mc, num_steps_mc+1) + [0, forward_curve]; % initialize variance paths
    logS_mc_current = zeros(num_path_mc,1); % current stock price = 1
    if strcmp(mode, 'iv')
        option_price_mat = zeros(num_maturity, num_moneyness);
        moneyness_mat = zeros(num_maturity, num_moneyness);
        tau_vec = zeros(num_maturity,1);
    end
    % resize noise matrix (coupling strategy)
    noise_mat_3d_mc = resize_noise_mat(noise_mat_3d_mc, num_steps_mc);


    kernel_length = num_steps_mc*2;
    kernel_dt = (linspace(1,kernel_length,kernel_length).^(H+1/2) - ...
        linspace(0,kernel_length-1,kernel_length).^(H+1/2)) / (H+1/2);
    gamma_val = gamma(H+1/2);

    for i = 1:num_steps_mc
        f_mc_current = sqrt(max(V_mat_mc(:,i), eps));
        logS_mc_current = logS_mc_current + (r - f_mc_current.^2/2) * dt...
            + dt^(1/2) * f_mc_current .* (rho * noise_mat_3d_mc(:,i,1) + ...
            sqrt(1-rho^2) * noise_mat_3d_mc(:,i,2));
        % update the whole variance path
        j_vec = i+1 : num_steps_mc+1;
        adj_J = min(J, num_steps_mc+1-i);
        tmpd = nu / gamma_val * f_mc_current;
        tmpc = -dt^(H+1/2) * lambda / gamma_val * V_mat_mc(:,i); 
        V_mat_mc(:,j_vec) = V_mat_mc(:,j_vec) + tmpc * kernel_dt(j_vec-i)...
            + tmpd * dt^H .* [zeros(1,adj_J), kernel_dt(J+1:num_steps_mc+1-i)] .* noise_mat_3d_mc(:,i,1); 
        for j = 1:adj_J  % adjustment to nearby terms (more accurate noises)
            V_mat_mc(:,i+j) = V_mat_mc(:,i+j) + dt^H * tmpd .* noise_mat_3d_mc(:,i,j+2);
        end
        if strcmp(mode, 'iv') && mod(i, steps_per_maturity) == 0
            index_maturity = i / steps_per_maturity;
            tau_vec(index_maturity) = dt_maturity * index_maturity;
            moneyness_mat(index_maturity, :) = exp(linspace(-2*sqrt(spotV)*sqrt(tau_vec(index_maturity)),...
                2*sqrt(spotV)*sqrt(tau_vec(index_maturity)), num_moneyness));
            option_price_mat(index_maturity, :) = mean( (exp(logS_mc_current) - moneyness_mat(index_maturity, :))...
                .* (exp(logS_mc_current) - moneyness_mat(index_maturity, :) >0), 1 );
        end
    end

    t = toc;
    fprintf("CPU time (standard hybrid scheme): %.5f\n", t);

    results.logS_mc_final = logS_mc_current;
    results.t = t;
    if strcmp(mode, 'iv')
        results.tau_vec = tau_vec;
        results.moneyness_mat = moneyness_mat;
        iv_mat = zeros(size(option_price_mat));
        for i = 1:size(option_price_mat,1)
            for j = 1:size(option_price_mat,2)
                iv_mat(i, j) = blsimpv(1, moneyness_mat(i, j), r, tau_vec(i), option_price_mat(i, j),...
                    'Class',{'call'});
            end
        end
        results.iv_mat = iv_mat;
    elseif strcmp(mode, 'full')
        results.V_mat_mc = V_mat_mc;
    end
end

