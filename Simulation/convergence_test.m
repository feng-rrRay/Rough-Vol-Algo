clear; clc;
rng(1);
addpath("helpers/");
%% begin test
nList = [25, 50, 100, 250, 500];
H = 0.1;
T = 1/252;
nf = 200;
J = 2;

results = testConvergenceN(nList, H, T, nf, J);
instanceName = "new_pc_test_n_1Day_point1H_200nf";
json_add("assets/"+instanceName+".json", results);

%% plot
width = 4;     % Width in inches
height = 3;    % Height in inches
alw = 0.75;    % AxesLineWidth
fsz = 11;      % Fontsize
lw = 1.5;      % LineWidth
msz = 8;       % MarkerSize

figure;
pos = get(gcf, 'Position');
set(gcf, 'Position', [pos(1) pos(2) width*100, height*100]); %<- Set size
set(gca, 'FontSize', fsz, 'LineWidth', alw); %<- Set properties

hold on;
% plot([results.nf], [results.atm_call_final], 'DisplayName', 'Accelerated Hybrid');
% plot([results.nf], [results.atm_call_markov], 'DisplayName', 'Markov Approximation');
plot(log([results.n]), log([results.rmse_acchyb_atm]), 'Marker', '.', 'DisplayName', 'Accelerated Hybrid');
plot(log([results.n]), log([results.rmse_markov_atm]), 'Marker', '+', 'DisplayName', 'Markov Approximation');
plot(log([results.n]), log([results.rmse_hyb_atm]), 'Marker', 'x', 'DisplayName', 'Classical Hybrid');
plot(log([results.n]), log([results.rmse_euler_atm]), 'Marker', '*', 'DisplayName', 'Euler Scheme');
title("$H = 0.1, T = 1/252$", 'Interpreter', 'latex');
xlabel("$\log(n)$", 'Interpreter', 'latex');
ylabel("$\log(RMSE)$", 'Interpreter', 'latex');
legend('Interpreter','latex', 'FontSize', 8, 'Location','best');
grid off;
box on;
hold off;
% save plot
print(gcf, '-dpng', '-r300', "assets/"+instanceName+".png");

%% functions
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
    % refList = [0, 0, 0];
    % for ii = 1 : numel(KList)
    %     k = log(KList(ii)); 
    %     S = 1;
    %     refList(ii) = CallPriceArray(S,k,H,rho,nu,lambda,theta,V0,T,r,d,n,left,right,N);
    %     fprintf("Ref European call price: %.6f, S=%d, K=%.2f\n", refList(ii),S,KList(ii));
    % end
    refList = [0.100000, 0.005404, -0.000001];

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
        result_struct(k).rmse_markov_itm = sqrt(mean((max(exp(results_1.logS_mc_markov) - KList(1), 0) - refList(1)).^2));
        result_struct(k).rmse_acchyb_itm = sqrt(mean((max(exp(results_1.logS_mc_final) - KList(1), 0) - refList(1)).^2));
        result_struct(k).rmse_markov_atm = sqrt(mean((max(exp(results_1.logS_mc_markov) - KList(2), 0) - refList(2)).^2));
        result_struct(k).rmse_acchyb_atm = sqrt(mean((max(exp(results_1.logS_mc_final) - KList(2), 0) - refList(2)).^2));
        result_struct(k).rmse_markov_otm = sqrt(mean((max(exp(results_1.logS_mc_markov) - KList(3), 0) - refList(3)).^2));
        result_struct(k).rmse_acchyb_otm = sqrt(mean((max(exp(results_1.logS_mc_final) - KList(3), 0) - refList(3)).^2));
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
        result_struct(k).rmse_hyb_itm = sqrt(mean((max(exp(results_2.logS_mc_final) - KList(1), 0) - refList(1)).^2));
        result_struct(k).rmse_hyb_atm = sqrt(mean((max(exp(results_2.logS_mc_final) - KList(2), 0) - refList(2)).^2));
        result_struct(k).rmse_hyb_otm = sqrt(mean((max(exp(results_2.logS_mc_final) - KList(3), 0) - refList(3)).^2));
        result_struct(k).price_hyb_itm = european_call_mc(results_2.logS_mc_final, KList(1));
        result_struct(k).price_hyb_atm = european_call_mc(results_2.logS_mc_final, KList(2));
        result_struct(k).price_hyb_otm = european_call_mc(results_2.logS_mc_final, KList(3));
        fprintf("ATM option price (Hybrid Scheme Classic):   %.6f\n", result_struct(k).price_hyb_atm);

        % Running Euler scheme
        results_3 = euler_scheme(V0, num_maturity, dt_maturity, num_moneyness, num_path_mc, step_maturity,...
            theta, rho, lambda, nu, H, noise_mat_3d_mc, r, eps, 'std');
        result_struct(k).t_euler = results_3.t;
        result_struct(k).rmse_euler_itm = sqrt(mean((max(exp(results_3.logS_mc_final) - KList(1), 0) - refList(1)).^2));
        result_struct(k).rmse_euler_atm = sqrt(mean((max(exp(results_3.logS_mc_final) - KList(2), 0) - refList(2)).^2));
        result_struct(k).rmse_euler_otm = sqrt(mean((max(exp(results_3.logS_mc_final) - KList(3), 0) - refList(3)).^2));
        result_struct(k).price_euler_itm = european_call_mc(results_3.logS_mc_final, KList(1));
        result_struct(k).price_euler_atm = european_call_mc(results_3.logS_mc_final, KList(2));
        result_struct(k).price_euler_otm = european_call_mc(results_3.logS_mc_final, KList(3));
        fprintf("ATM option price (Euler Scheme):            %.6f\n", result_struct(k).price_euler_atm);
    end


end