clear; clc;
rng(1);
addpath("helpers/");
%% begin test
nfList = [5:10:300, 300:50:1000];
H = 0.1;
T = 1/252;
n = 200;

results = testConvergenceNf(nfList, H, T, n);

%% plot
% json_add("assets/test_nf_intraDay_point1H_200n.json", results);
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
plot([results.nf], [results.atm_call_final], 'DisplayName', 'Accelerated Hybrid');
plot([results.nf], [results.atm_call_markov], 'DisplayName', 'Markov Approximation');
title("$T = 1/252/10$", 'Interpreter', 'latex');
xlabel("Number of Factors $n_f$", 'Interpreter', 'latex');
ylabel("Price", 'Interpreter', 'latex');
legend('Interpreter','latex');
grid off;
box on;
hold off;
% save plot
print(gcf, '-dpng', '-r300', "assets/test_nf_intraDay_point1H_200n.png");

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

    Sigma = zeros(J+2,J+2);
    j_vec = 2:J+1; k_vec = 2:J+1;
    Sigma(1,3:J+2) = ((j_vec-1).^(H+1/2) - (j_vec-2).^(H+1/2)) / (H+1/2) * dt^(H+1/2); % dt term absorbed in path generation in reference code
    F1mat = hypergeom([-H+1/2,1],H+3/2,[1,j_vec]'./[1,k_vec].*([1,j_vec]'<[1,k_vec]));
    Sigma(3:J+2,3:J+2) = ((j_vec'<k_vec).*((j_vec-1)'.^(H+1/2).* (k_vec-1).^(H-1/2) .* F1mat(j_vec-1,k_vec-1) ... 
        -(j_vec-2)'.^(H+1/2).* ([1,1:J-1]).^(H-1/2) .* F1mat([1,1:J-1],[1,1:J-1]))/(H+1/2))* dt^(2*H);
    Sigma = Sigma + Sigma';
    Sigma(logical(eye(J+2))) = [dt, dt, ((j_vec-1).^(2*H) - (j_vec-2).^(2*H)) / (2*H) * dt^(2*H)]';
    
    
    noise_mat_3d_mc = normrnd(0,1,[num_path_mc,num_steps_mc,J+2]);
    L = chol(Sigma)'; % lower triangular matrix; **numerical problem here if H=1/2**
    for j = J+2:-1:1
        noise_mat_3d_mc(:,:,j) = L(j,j) * noise_mat_3d_mc(:,:,j);
        for i = 1:j-1
            noise_mat_3d_mc(:,:,j) = noise_mat_3d_mc(:,:,j) + L(j,i) * noise_mat_3d_mc(:,:,i);
        end
    end
    
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