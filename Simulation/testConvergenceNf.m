function result_struct = testConvergenceNf(nfList, H, T, n)
    theta = 0.05;
    rho = -0.1;
    lambda = 0.1;
    nu = 0.3;
    r = 0; d = 0;
    V0 = 0.05;
    eps = 0.0001;
    dt_maturity = T;
    num_maturity = 1;    % number of maturities for IV surface
    step_maturity = n;   % number of time steps
    num_steps_mc = num_maturity * step_maturity; % number of steps in total
    num_path_mc = 3e5;    % Monte Carlo paths
    num_moneyness = 41;
    dt = dt_maturity / step_maturity; 
    J = 2; 
    alpha = H + 0.5;

    % get noise matrix
    noise_mat_3d_mc = get_noise_mat(J, H, num_path_mc, num_steps_mc);
    
    % use hybrid scheme forward curve
    forward_curve_hyb = lambda * theta * ((1:num_steps_mc)*dt).^(H+1/2) / (H+1/2) / gamma(H+1/2);
    
    % loop over nf
    num_instances = numel(nfList);
    result_struct = repmat(struct(...
        'nf',[],...
        't_markov',[],...
        't_acchyb',[],...
        'logS_mc_markov',[],...
        'logS_mc_final',[]), ...
        num_instances, 1);
    for k = 1 : num_instances
        nf = nfList(k);
        fprintf("Testing nf = %d\n", nf);
        rn = 1 + 10 * nf^(-0.9);
        cs = ones(1,nf) .* ((rn^(1-alpha) - 1)*rn^((alpha-1)*(1+nf/2))) ./ (gamma(alpha) * gamma(2-alpha));
        xs = ones(1,nf) .* ((1-alpha)/(2-alpha)*(rn^(2-alpha)-1)/(rn^(1-alpha)-1));
        for i = 1:nf
            cs(i) = cs(i) * rn^((1-alpha)*i);
            xs(i) = xs(i) * rn^(i-1-nf/2);
        end
        % run accelerated_hybrid algorithm to get logS_mc_final and logS_mc_markov
        u_current = zeros(1,nf);
        results = accelerated_hybrid(V0, num_maturity, dt_maturity,...
            num_moneyness, num_path_mc, step_maturity, u_current, forward_curve_hyb, cs, xs, ...,
            theta, rho, lambda, nu, H, noise_mat_3d_mc, J, r, eps, 'std');
        % save results to result_struct
        result_struct(k).nf              = nf;
        result_struct(k).t_markov        = results.t_markov;
        result_struct(k).t_acchyb        = results.t;
        % result_struct(k).logS_mc_markov  = results.logS_mc_markov;
        % result_struct(k).logS_mc_final   = results.logS_mc_final;
        result_struct(k).atm_call_markov = european_call_mc(results.logS_mc_markov, 1);
        result_struct(k).atm_call_final  = european_call_mc(results.logS_mc_final, 1);
    end
end