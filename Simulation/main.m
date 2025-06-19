% MAIN script for discretization scheme implementation--Lifted model
% Markovian approximation for rough Heston model
clear; clc;
rng(1);
addpath("helpers/");

%% Parameters
%  Rough Heston model:  (8.1.3)-(8.1.4) in (Abi Jaber, 2018)
H = 0.15;
theta = 0.05;
rho = -0.1;
lambda = 0.1;
nu = 0.3;
r = 0; d = 0;
V0 = 0.05;
eps = 0.0001;

%  Lifted Heston model: (8.2.1)-(8.2.3) in (Abi Jaber, 2018)
nf = 100; % number of factors (50 would yield reasonable results for 1 day to maturity)
alpha = H + 0.5;
rn = 1 + 10 * nf^(-0.9);
cs = ones(1,nf) .* ((rn^(1-alpha) - 1)*rn^((alpha-1)*(1+nf/2))) ./ (gamma(alpha) * gamma(2-alpha));
xs = ones(1,nf) .* ((1-alpha)/(2-alpha)*(rn^(2-alpha)-1)/(rn^(1-alpha)-1));
for i = 1:nf
    cs(i) = cs(i) * rn^((1-alpha)*i);
    xs(i) = xs(i) * rn^(i-1-nf/2);
end

% time grid definition
dt_maturity = 1/252;  % gap between maturities in IV surface
num_maturity = 4;    % number of maturities for IV surface
num_moneyness = 41;
step_maturity = 10;  % steps in-fill (100 for reasonable results for 1/10 day to maturity)
num_steps_mc = num_maturity * step_maturity; % number of steps in total
num_path_mc = 3e5;    % Monte Carlo paths
dt = dt_maturity / step_maturity; 
J = 2; % hybrid scheme #(Gaussian noise)

%% Kernel coefficients and random matrices
% reference: Sec~3.1, Bennedsen et al. (2017)
noise_mat_3d_mc = get_noise_mat(J, H, num_path_mc, num_steps_mc);

%% Implicit-explicit scheme lifted Heston (Markov Approximation)
u_current = zeros(1,nf);
num_steps_mc = num_maturity * step_maturity;
forward_curve = lambda * theta * sum(cs./xs .* (1-exp(-(1:num_steps_mc)'*dt*xs)),2)'; % func g_0(t)
results_1 = markov_approx(V0, num_maturity, dt_maturity,...
    num_moneyness, num_path_mc, step_maturity, u_current, forward_curve, cs, xs, ...,
    theta, rho, lambda, nu, H, noise_mat_3d_mc, r, eps, 'iv');
t1 = results_1.t;
moneyness_mat_1 = results_1.moneyness_mat;
iv_mat_1 = results_1.iv_mat;

%% Hybrid scheme rough Heston
forward_curve_hyb = lambda * theta * ((1:num_steps_mc)*dt).^(H+1/2) / (H+1/2) / gamma(H+1/2);
results_2 = standard_hybrid(V0, num_maturity, dt_maturity,...
    num_moneyness, num_path_mc, step_maturity, forward_curve_hyb,...
    theta, rho, lambda, nu, H, noise_mat_3d_mc, J, r, eps, 'iv');
t2 = results_2.t;
moneyness_mat_2 = results_2.moneyness_mat;
iv_mat_2 = results_2.iv_mat;


%% Inverse transform (benchmark by Chenyu Wang)
addpath('Fourier_Transform_Inversion');
n = 1000; N = 20000;
left = 1e-12; right = 400;
iv_mat_3 = zeros(num_maturity, num_moneyness);
log_strikes = linspace(-2*sqrt(V0*dt_maturity),2*sqrt(V0*dt_maturity),num_moneyness);
% iv_mat_3(1,:) = IVCurveArray(log_strikes, H, rho, nu, lambda, theta, V0, dt_maturity, r, d, n, left, right, N);
tic;
for k = 1:num_maturity
    log_strikes = linspace(-2*sqrt(V0*dt_maturity*k),2*sqrt(V0*dt_maturity*k),num_moneyness);
    iv_mat_3(k,:) = IVCurveArray(log_strikes, H, rho, nu, lambda, theta, V0, dt_maturity*k, r, d, n, left, right, N);
    disp(k);
end
t3 = toc;

%% Accelerated Hybrid Scheme
u_current = zeros(1,nf);
num_steps_mc = num_maturity * step_maturity;
% forward_curve = lambda * theta * sum(cs./xs .* (1-exp(-(1:num_steps_mc)'*dt*xs)),2)'; % func g_0(t)
forward_curve_hyb = lambda * theta * ((1:num_steps_mc)*dt).^(H+1/2) / (H+1/2) / gamma(H+1/2);
results_4 = accelerated_hybrid(V0, num_maturity, dt_maturity,...
    num_moneyness, num_path_mc, step_maturity, u_current, forward_curve_hyb, cs, xs, ...,
    theta, rho, lambda, nu, H, noise_mat_3d_mc, J, r, eps, 'iv');
t4=results_4.t;
iv_mat_4 = results_4.iv_mat;
moneyness_mat_4 = results_4.moneyness_mat;
tau_vec_4 = results_4.tau_vec;

%% Euler scheme
results_5 = euler_scheme(V0, num_maturity, dt_maturity, num_moneyness, num_path_mc, step_maturity,...
    theta, rho, lambda, nu, H, noise_mat_3d_mc, r, eps, 'iv');
t5 = results_5.t;
moneyness_mat_5 = results_5.moneyness_mat;
iv_mat_5 = results_5.iv_mat;

%% Comparison
figure;
% indices = [1,2,3,4];
indices = 1:4;
for i = 4:4
    % subplot(1, 4, i);  % Creates a 2x2 grid and selects the i-th subplot.
    hold on;
    plot(log(moneyness_mat_1(indices(i),:)), iv_mat_1(indices(i),:), "DisplayName", "Markovian approx");
    plot(log(moneyness_mat_2(indices(i),:)), iv_mat_2(indices(i),:), "DisplayName", "Hybrid scheme");
    plot(log(moneyness_mat_2(indices(i),:)), iv_mat_3(indices(i),:), "DisplayName", "Inverse transform");
    plot(log(moneyness_mat_4(indices(i),:)), iv_mat_4(indices(i),:), "DisplayName", "Hybrid Acceleration");
    plot(log(moneyness_mat_5(indices(i),:)), iv_mat_5(indices(i),:), "DisplayName", "Euler Scheme");
    hold off;
    
    legend('Location','best');
    title(sprintf('IV: t = %d/252, H = %.2f, n = %d', indices(i), H, num_maturity*step_maturity));
end

%% Report
err_exact_1 = sqrt(mean((iv_mat_1 - iv_mat_3).^2,'all')); err_exact_2 = sqrt(mean((iv_mat_2 - iv_mat_3).^2,'all')); err_exact_3 = 0; err_exact_4 = sqrt(mean((iv_mat_4 - iv_mat_3).^2,'all'));
err_hyb_1 = sqrt(mean((iv_mat_1 - iv_mat_2).^2,'all')); err_hyb_2 = 0; err_hyb_3 = sqrt(mean((iv_mat_3 - iv_mat_2).^2,'all')); err_hyb_4 = sqrt(mean((iv_mat_4 - iv_mat_2).^2,'all'));
fprintf("Parameters: nf = %d, steps_per_maturity = %d.\n", nf, step_maturity);
fprintf("Markov Approximation: cpu time: %5.2f, err-to-exact: %3.2E, err-to-hybrid: %3.2E.\n", t1, err_exact_1, err_hyb_1);
fprintf("       Hybrid Scheme: cpu time: %5.2f, err-to-exact: %3.2E, err-to-hybrid: %3.2E.\n", t2, err_exact_2, err_hyb_2);
fprintf("             Inverse Transform: cpu time: %5.2f, err-to-exact: %3.2E, err-to-hybrid: %3.2E.\n", t3, err_exact_3, err_hyb_3);
fprintf("        Acceleration: cpu time: %5.2f, err-to-exact: %3.2E, err-to-hybrid: %3.2E.\n", t4, err_exact_4, err_hyb_4);
