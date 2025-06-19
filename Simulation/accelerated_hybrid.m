function results = accelerated_hybrid(spotV, num_maturity, dt_maturity,...
    num_moneyness, num_path_mc, steps_per_maturity, u_current, forward_curve, cs, xs, ...,
    theta, rho, lambda, nu, H, noise_mat_3d_mc, J, r, eps, mode)
    %ACCELERATED_HYBRID Multi-factor accelerated hybrid SDE simulation
    %   INPUTS:
    %     spotV             – initial variance level (scalar)
    %     num_maturity      – number of maturities under 'iv' mode (integer)
    %     dt_maturity       – maturity time interval under 'iv' mode (scalar)
    %     num_moneyness     – # of strike/moneyness points per maturity for 'iv' mode
    %     num_path_mc       – number of Monte-Carlo paths
    %     steps_per_maturity– time‐steps per maturity
    %     u_current         – current factor state (vector length = # paths)
    %     forward_curve     – forward variance curve (1×num_steps)
    %     cs                – factor loading matrix (# factors)
    %     xs                – multi-factor approximation (# factors)
    %     theta             – rough Heston model parameter (scalar)
    %     rho               – rough Heston model correlation (scalar)
    %     lambda            – rough Heston model mean‐reversion rate
    %     nu                – rough Heston model diffusion term multiplier (scalar)
    %     H                 – Hurst index
    %     noise_mat_3d_mc   – pre-generated noise: [paths × steps × (J+2)]
    %     J                 – number of additional noises for hybrid scheme, see Bennedsen et al. (2017)
    %     r                 – risk-free rate
    %     eps               – small floor for variance
    %     mode              – output mode, one of:
    %                          'std'    : return logS_mc_final + logS_mc_markov + timings
    %                          'iv'     : return implied‐vol matrix + maturities + moneyness
    %                          'full'   : return full V_mat paths
    %
    %   OUTPUT:
    %     RESULTS – struct with these possible fields:
    %       .logS_mc_markov  – Markovian log‐price result
    %       .logS_mc_final   – accelerated hybrid final log‐price
    %       .V_mat_mc        – full variance paths (only if mode='path')
    %       .iv_mat          – implied‐volatility surface (mode='iv')
    %       .tau_vec         – vector of maturities (mode='iv')
    %       .moneyness_mat   – grid of moneyness levels (mode='iv')
    %       .t_markov        – CPU time for predictor step
    %       .t_acchyb        – CPU time for predictor + hybrid steps
    %

    num_steps_mc = num_maturity * steps_per_maturity;
    dt = dt_maturity / steps_per_maturity;
    num_factor = size(cs, 2);
    u_current_mc = ones(num_path_mc,1) * u_current;
    V_mat_mc = spotV * ones(num_path_mc, num_steps_mc+1) + [0, forward_curve]; % initialize variance paths
    logS_mc_current = zeros(num_path_mc,1); % current stock price = 1
    logS_mc_markov = zeros(num_path_mc,1);
    gamma_val = gamma(H+1/2);
    results = struct();
    if strcmp(mode, 'iv')
        results.moneyness_mat = zeros(num_maturity, num_moneyness);
        results.tau_vec = zeros(num_maturity,1);
        option_price_mat = zeros(num_maturity, num_moneyness);
    end

    % resize noise matrix (coupling strategy)
    noise_mat_3d_mc = resize_noise_mat(noise_mat_3d_mc, num_steps_mc);

    % Markovian precomputation
    tic;
    for i = 1:num_steps_mc
        f_mc_current = max(V_mat_mc(:,i), eps);
        delta_u = -lambda*V_mat_mc(:,i)*dt + nu*sqrt(f_mc_current)*dt^(1/2).*noise_mat_3d_mc(:,i,1);
        % u_current_mc = (u_current_mc + delta_u) ./ (1 + xs*dt);
        u_current_mc = (u_current_mc + delta_u) .* exp(-xs*dt);
        V_mat_mc(:,i+1) = V_mat_mc(:,i+1) + sum(cs .* u_current_mc,2); 
        % test Markovian approximation at the meantime
        logS_mc_markov = logS_mc_markov + (r - f_mc_current/2) * dt + ...
            sqrt(f_mc_current)*dt^(1/2) .* (rho*noise_mat_3d_mc(:,i,1) + sqrt(1-rho^2)*noise_mat_3d_mc(:,i,2));
    end
    t1=toc;
    results.logS_mc_markov = logS_mc_markov; % output results by Markovian approximation by default

    % Accelerated Hybrid
    tic;
    kernel_dt = ones(num_path_mc,1) * [0, linspace(1,num_steps_mc,num_steps_mc).^(H+1/2) - ...
        linspace(0,num_steps_mc-1,num_steps_mc).^(H+1/2)] / (H+1/2);
    f_mat_mc = sqrt(max(V_mat_mc, eps));
    noise_path_mc = squeeze(noise_mat_3d_mc(:,1:num_steps_mc,1));
    var_path_diff = nu * f_mat_mc .* [noise_path_mc,zeros(num_path_mc,1)];
    var_path_drift = lambda * dt^(1/2) * V_mat_mc;
    tmp_N = 2^nextpow2(2*num_steps_mc-1);
    % Fast Fourier Transform
    kernel_fft = fft(kernel_dt,tmp_N,2);
    var_fft = fft(var_path_diff - var_path_drift,tmp_N,2);
    res = ifft(kernel_fft.*var_fft,tmp_N,2) * dt^H / gamma_val;
    V_mat_mc = spotV * ones(num_path_mc, num_steps_mc+1) + [0, forward_curve] ...
        + res(:,1:num_steps_mc+1); % variance path after FFT
    for i = 1:num_steps_mc
        f_mc_current = sqrt(max(V_mat_mc(:,i), eps));
        logS_mc_current = logS_mc_current + (r - f_mc_current.^2/2) * dt...
            + dt^(1/2) * f_mc_current .* (rho * noise_mat_3d_mc(:,i,1) + sqrt(1-rho^2) * noise_mat_3d_mc(:,i,2));
        adj_J = min(J, num_steps_mc+1-i);
        for j = 1 : adj_J
            % notice `f_mc_current` and `f_mat_mc` here, `f_mat_mc` are
            % previous values to deduct
            V_mat_mc(:,i+j) = V_mat_mc(:,i+j) + (nu * dt^H * f_mat_mc(:,i) .* noise_mat_3d_mc(:,i,j+2) - ...
                dt^H * nu * f_mat_mc(:,i) .* kernel_dt(:,j+1) .* noise_mat_3d_mc(:,i,1)) / gamma_val;
            % V_mat_mc(:,i+j) = V_mat_mc(:,i+j) + (nu * f_mc_current .* noise_mat_3d_mc(:,i,j+2) - ...
            %     dt^(H-1/2) * nu * f_mat_mc(:,i) .* kernel_dt(:,j+1) .* noise_mat_3d_mc(:,i,1)) / gamma_val;
        end
        if strcmp(mode, 'iv') && mod(i, steps_per_maturity) == 0
            index_maturity = i / steps_per_maturity;
            results.tau_vec(index_maturity) = dt_maturity * index_maturity;
            results.moneyness_mat(index_maturity, :) = exp(linspace(-2*sqrt(spotV)*sqrt(results.tau_vec(index_maturity)),...
                2*sqrt(spotV)*sqrt(results.tau_vec(index_maturity)), num_moneyness));
            option_price_mat(index_maturity, :) = mean( (exp(logS_mc_current) - results.moneyness_mat(index_maturity, :))...
                .* (exp(logS_mc_current) - results.moneyness_mat(index_maturity, :) >0), 1 ); % European call price for iv computation
        end
    end
    t2=toc;
    % func_val = phi_func(V_mat_mc(:,num_steps_mc+1), logS_mc_current, 1);
    if strcmp(mode, 'iv')
        results.iv_mat = zeros(size(option_price_mat));
        for i = 1:size(option_price_mat,1)
            for j = 1:size(option_price_mat,2)
                results.iv_mat(i, j) = blsimpv(1, results.moneyness_mat(i, j), r, results.tau_vec(i), option_price_mat(i, j),...
                    'Class',{'call'});
            end
        end
    end
    results.logS_mc_final = logS_mc_current; % output logS estimate by default
    if strcmp(mode, 'full')
        results.V_mat_mc = V_mat_mc; % output V_mat under `path` mode
    end
    fprintf("CPU time (precomputing): %.5f\n", t1);
    fprintf("CPU time (FFT Hybrid):   %.5f\n", t2);
    results.t_markov = t1;
    results.t = t1 + t2;
end

