function result_struct = testConvergenceAcchyb(nList, nfList, H, T, J, refPrice)
    num_n = numel(nList);
    num_nf = numel(nfList);
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
    noise_mat_3d_mc = get_noise_mat(J, H, num_path_mc, max(nList));
    if refPrice == inf  % get the ATM European call price
        addpath("Fourier_Transform_Inversion/");
        left = 1e-12;
        right = 400;
        n = 1000; N = 20000;
        S = 1;
        k = 0;
        refPrice = CallPriceArray(S,k,H,rho,nu,lambda,theta,V0,T,r,d,n,left,right,N);
        fprintf("Reference price via transform inversion: %.6f\n", refPrice);
    end
    result_struct = repmat(struct('n', [], 'nf', [], ...
        't', [], 'ref', [], 'price', [], 'rmse', [], 'se', [], 'bias', []), num_n * num_nf, 1);
    for k1 = 1 : numel(nfList)
        nf = nfList(k1);
        fprintf("Testing nf = %d...\n", nf);
        % initialize markovian approximation factors
        rn = 1 + 10 * nf^(-0.9);
        cs = ones(1,nf) .* ((rn^(1-alpha) - 1)*rn^((alpha-1)*(1+nf/2))) ./ (gamma(alpha) * gamma(2-alpha));
        xs = ones(1,nf) .* ((1-alpha)/(2-alpha)*(rn^(2-alpha)-1)/(rn^(1-alpha)-1));
        for i = 1:nf
            cs(i) = cs(i) * rn^((1-alpha)*i);
            xs(i) = xs(i) * rn^(i-1-nf/2);
        end
        u_current = zeros(1, nf);
        for k2 = 1 : numel(nList)
            step_maturity = nList(k2);
            dt = dt_maturity / step_maturity; 
            num_steps_mc = num_maturity * step_maturity;
            fprintf("  Testing n = %d...\n", step_maturity);
            
            ii = (k1 - 1) * num_n + k2;
            result_struct(ii).n = step_maturity;
            result_struct(ii).nf = nf;
            result_struct(ii).ref = refPrice;
    
            forward_curve_hyb = lambda * theta * ((1:num_steps_mc)*dt).^(H+1/2) / (H+1/2) / gamma(H+1/2);
            
            % Running Markovian approximation method and accelerated hybrid scheme
            results_1 = accelerated_hybrid(V0, num_maturity, dt_maturity,...
                num_moneyness, num_path_mc, step_maturity, u_current, forward_curve_hyb, cs, xs, ...,
                theta, rho, lambda, nu, H, noise_mat_3d_mc, J, r, eps, 'std');
            result_struct(ii).t = results_1.t;
            result_struct(ii).price = european_call_mc(results_1.logS_mc_final, 1); % atm call option price
            [result_struct(ii).rmse, result_struct(ii).se, result_struct(ii).bias] = rmse_mc(results_1.logS_mc_final, ...
                1, refPrice);

        end
    end
end