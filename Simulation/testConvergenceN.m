function result_struct = testConvergenceN(nList, H, T, nf, J)
    theta = 0.05;
    rho = -0.1;
    lambda = 0.1;
    nu = 0.3;
    r = 0; d = 0;
    V0 = 0.05;
    eps = 0.0001;
    dt_maturity = T;
    num_maturity = 1; 
    num_path_mc = 3e5;    % # Monte Carlo paths
    num_moneyness = 41;
    alpha = H + 0.5;

    % transform inverse pricing as reference
    addpath("Fourier_Transform_Inversion/");
    left = 1e-12;
    right = 400;
    n = 1000; N = 20000;
    KList = [0.9, 1, 1.1];
    refList = [0, 0, 0];
    for ii = 1 : numel(KList)
        k = log(KList(ii)); 
        S = 1;
        refList(ii) = CallPriceArray(S,k,H,rho,nu,lambda,theta,V0,T,r,d,n,left,right,N);
        fprintf("Ref European call price: %.6f, S=%d, K=%.2f\n", refList(ii),S,KList(ii));
    end
    % refList = [0.102132261667885, 0.024095334125073, 0.002281439730902];

    % get noise matrix
    noise_mat_3d_mc = get_noise_mat(J, H, num_path_mc, max(nList));

    % initialize markovian approximation factors
    rn = 1 + 10 * nf^(-0.9);
    cs = ones(1,nf) .* ((rn^(1-alpha) - 1)*rn^((alpha-1)*(1+nf/2))) ./ (gamma(alpha) * gamma(2-alpha));
    xs = ones(1,nf) .* ((1-alpha)/(2-alpha)*(rn^(2-alpha)-1)/(rn^(1-alpha)-1));
    for i = 1:nf
        cs(i) = cs(i) * rn^((1-alpha)*i);
        xs(i) = xs(i) * rn^(i-1-nf/2);
    end
    u_current = zeros(1, nf);

    % loop over n
    num_instances = numel(nList);
    result_struct = repmat(struct(...
        'n',[],...
        't_markov',[],...
        't_acchyb',[],...
        't_euler',[],...
        't_hyb',[],...
        'ref_atm',[], 'ref_otm',[], 'ref_itm',[],...
        'rmse_markov_atm',[],...
        'rmse_acchyb_atm',[],...
        'rmse_euler_atm',[],...
        'rmse_hyb_atm',[],...
        'rmse_markov_itm',[],...
        'rmse_acchyb_itm',[],...
        'rmse_euler_itm',[],...
        'rmse_hyb_itm',[],...
        'rmse_markov_otm',[],...
        'rmse_acchyb_otm',[],...
        'rmse_euler_otm',[],...
        'rmse_hyb_otm',[],...
        'price_markov_atm',[],...
        'price_acchyb_atm',[],...
        'price_euler_atm',[],...
        'price_hyb_atm',[],...
        'price_markov_itm',[],...
        'price_acchyb_itm',[],...
        'price_euler_itm',[],...
        'price_hyb_itm',[],...
        'price_markov_otm',[],...
        'price_acchyb_otm',[],...
        'price_euler_otm',[],...
        'price_hyb_otm',[]), ...
        num_instances, 1);
    for k = 1 : num_instances
        step_maturity = nList(k);
        dt = dt_maturity / step_maturity; 
        result_struct(k).n = step_maturity;
        result_struct(k).ref_itm = refList(1);
        result_struct(k).ref_atm = refList(2);
        result_struct(k).ref_otm = refList(3);
        num_steps_mc = num_maturity * step_maturity;
        fprintf("Testing n = %d...\n", step_maturity);

        forward_curve_hyb = lambda * theta * ((1:num_steps_mc)*dt).^(H+1/2) / (H+1/2) / gamma(H+1/2);
        
        % Running Markovian approximation method and accelerated hybrid scheme
        results_1 = accelerated_hybrid(V0, num_maturity, dt_maturity,...
            num_moneyness, num_path_mc, step_maturity, u_current, forward_curve_hyb, cs, xs, ...,
            theta, rho, lambda, nu, H, noise_mat_3d_mc, J, r, eps, 'std');
        result_struct(k).t_acchyb = results_1.t;
        result_struct(k).t_markov = results_1.t_markov;
        [result_struct(k).rmse_markov_itm,result_struct(k).se_markov_itm,result_struct(k).bias_markov_itm] = rmse_mc(results_1.logS_mc_markov, KList(1), refList(1));
        [result_struct(k).rmse_acchyb_itm,result_struct(k).se_acchyb_itm,result_struct(k).bias_acchyb_itm] = rmse_mc(results_1.logS_mc_final, KList(1), refList(1));
        [result_struct(k).rmse_markov_atm,result_struct(k).se_markov_atm,result_struct(k).bias_markov_atm] = rmse_mc(results_1.logS_mc_markov, KList(2), refList(2));
        [result_struct(k).rmse_acchyb_atm,result_struct(k).se_acchyb_atm,result_struct(k).bias_acchyb_atm] = rmse_mc(results_1.logS_mc_final, KList(2), refList(2));
        [result_struct(k).rmse_markov_otm,result_struct(k).se_markov_otm,result_struct(k).bias_markov_otm] = rmse_mc(results_1.logS_mc_markov, KList(3), refList(3));
        [result_struct(k).rmse_acchyb_otm,result_struct(k).se_acchyb_otm,result_struct(k).bias_acchyb_otm] = rmse_mc(results_1.logS_mc_final, KList(3), refList(3));
        result_struct(k).price_markov_itm = european_call_mc(results_1.logS_mc_markov, KList(1));
        result_struct(k).price_acchyb_itm = european_call_mc(results_1.logS_mc_final, KList(1));
        result_struct(k).price_markov_atm = european_call_mc(results_1.logS_mc_markov, KList(2));
        result_struct(k).price_acchyb_atm = european_call_mc(results_1.logS_mc_final, KList(2));
        result_struct(k).price_markov_otm = european_call_mc(results_1.logS_mc_markov, KList(3));
        result_struct(k).price_acchyb_otm = european_call_mc(results_1.logS_mc_final, KList(3));
        fprintf("ATM option price (Accelerated Hybrid):      %.6f\n", result_struct(k).price_acchyb_atm);
        fprintf("ATM option price (Markovian Approximation): %.6f\n", result_struct(k).price_markov_atm);

        % Running classical hybrid scheme
        results_2 = standard_hybrid(V0, num_maturity, dt_maturity,...
            num_moneyness, num_path_mc, step_maturity, forward_curve_hyb,...
            theta, rho, lambda, nu, H, noise_mat_3d_mc, J, r, eps, 'std');
        result_struct(k).t_hyb = results_2.t;
        [result_struct(k).rmse_hyb_itm,result_struct(k).se_hyb_itm,result_struct(k).bias_hyb_itm] = rmse_mc(results_2.logS_mc_final, KList(1), refList(1));
        [result_struct(k).rmse_hyb_atm,result_struct(k).se_hyb_atm,result_struct(k).bias_hyb_atm] = rmse_mc(results_2.logS_mc_final, KList(2), refList(2));
        [result_struct(k).rmse_hyb_otm,result_struct(k).se_hyb_otm,result_struct(k).bias_hyb_otm] = rmse_mc(results_2.logS_mc_final, KList(3), refList(3));
        result_struct(k).price_hyb_itm = european_call_mc(results_2.logS_mc_final, KList(1));
        result_struct(k).price_hyb_atm = european_call_mc(results_2.logS_mc_final, KList(2));
        result_struct(k).price_hyb_otm = european_call_mc(results_2.logS_mc_final, KList(3));
        fprintf("ATM option price (Hybrid Scheme Classic):   %.6f\n", result_struct(k).price_hyb_atm);

        % Running Euler scheme
        results_3 = euler_scheme(V0, num_maturity, dt_maturity, num_moneyness, num_path_mc, step_maturity,...
            theta, rho, lambda, nu, H, noise_mat_3d_mc, r, eps, 'std');
        result_struct(k).t_euler = results_3.t;
        [result_struct(k).rmse_euler_itm,result_struct(k).se_euler_itm,result_struct(k).bias_euler_itm] = rmse_mc(results_3.logS_mc_final, KList(1), refList(1));
        [result_struct(k).rmse_euler_atm,result_struct(k).se_euler_atm,result_struct(k).bias_euler_atm] = rmse_mc(results_3.logS_mc_final, KList(2), refList(2));
        [result_struct(k).rmse_euler_otm,result_struct(k).se_euler_otm,result_struct(k).bias_euler_otm] = rmse_mc(results_3.logS_mc_final, KList(3), refList(3));
        result_struct(k).price_euler_itm = european_call_mc(results_3.logS_mc_final, KList(1));
        result_struct(k).price_euler_atm = european_call_mc(results_3.logS_mc_final, KList(2));
        result_struct(k).price_euler_otm = european_call_mc(results_3.logS_mc_final, KList(3));
        fprintf("ATM option price (Euler Scheme):            %.6f\n", result_struct(k).price_euler_atm);
    end
end