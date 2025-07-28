function results = euler_scheme(spotV, num_maturity, dt_maturity,...
    num_moneyness, num_path_mc, steps_per_maturity, ...,
    theta, rho, lambda, nu, H, noise_mat_3d_mc, r, eps, mode)
    %EULER_SCHEME Euler scheme simulation of rough Heston model
    % INPUT and OUTPUT align with those in `standard_hybrid.m`

    num_steps_mc = num_maturity * steps_per_maturity;
    dt = dt_maturity / steps_per_maturity;
    V_mat_mc = spotV * ones(num_path_mc, num_steps_mc+1); % initialize variance paths
    logS_mc_current = zeros(num_path_mc,1);
    if strcmp(mode, 'iv')
        option_price_mat = zeros(num_maturity, num_moneyness);
        moneyness_mat = zeros(num_maturity, num_moneyness);
        tau_vec = zeros(num_maturity,1);
    end
    gamma_val = gamma(H+1/2);
    % resize noise matrix (coupling strategy)
    noise_mat_3d_mc = resize_noise_mat(noise_mat_3d_mc, num_steps_mc);

    
    tic;
    for i = 1:num_steps_mc
        f_mc_current = sqrt(max(V_mat_mc(:,i),eps));
        logS_mc_current = logS_mc_current + (r - f_mc_current.^2/2) * dt + ...
            dt^(1/2) * f_mc_current .* (rho*noise_mat_3d_mc(:,i,1) + sqrt(1-rho^2)*noise_mat_3d_mc(:,i,2));
        for j = 1:i
            f_mc_j = sqrt(max(V_mat_mc(:,j),eps));
            inner = lambda * (theta - f_mc_j.^2) * dt + dt^(1/2) * nu * f_mc_j .* noise_mat_3d_mc(:,j,1);
            V_mat_mc(:,i+1) = V_mat_mc(:,i+1) + ((i-j+1)*dt)^(H-1/2) * inner / gamma_val;
        end
        if mod(i, steps_per_maturity) == 0
            index_maturity = i / steps_per_maturity;
            tau_vec(index_maturity) = dt_maturity * index_maturity;
            moneyness_mat(index_maturity, :) = exp(linspace(-2*sqrt(spotV)*sqrt(tau_vec(index_maturity)),...
                2*sqrt(spotV)*sqrt(tau_vec(index_maturity)), num_moneyness));
            option_price_mat(index_maturity, :) = mean( (exp(logS_mc_current) - moneyness_mat(index_maturity, :))...
                .* (exp(logS_mc_current) - moneyness_mat(index_maturity, :) >0), 1 );
        end
    end
    results.t = toc;
    fprintf("CPU time (Euler scheme): %.5f\n", results.t);
    results.logS_mc_final = logS_mc_current;

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

