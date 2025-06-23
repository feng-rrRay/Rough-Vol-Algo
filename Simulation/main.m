% MAIN script for discretization scheme implementation--Lifted model
% Markovian approximation for rough Heston model
clear; clc;
rng(42);
addpath("helpers/");

%% Parameters
%  Rough Heston model:  (8.1.3)-(8.1.4) in (Abi Jaber, 2018)
H = 0.15;
theta = 0.05;
rho = -0.7;
lambda = 0.1;
nu = 0.3;
r = 0; d = 0;
V0 = 0.05;
eps = 0.0001;

%  Lifted Heston model: (8.2.1)-(8.2.3) in (Abi Jaber, 2018)
nf = 400; % number of factors 
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
num_maturity = 20;    % number of maturities for IV surface
num_moneyness = 41;
step_maturity = 50;  % steps in-fill (100 for reasonable results for 1/10 day to maturity)
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
% log_strikes = linspace(-2*sqrt(V0*dt_maturity),2*sqrt(V0*dt_maturity),num_moneyness);
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

%% Plot
alw = 0.75;    % AxesLineWidth
fsz = 11;      % Fontsize
lw = 1.5;      % LineWidth
msz = 8;       % MarkerSize
if ismac
    width = 3;     % Width in inches
    height = 2.25;    % Height in inches
elseif ispc
    width = 4;     
    height = 3;   
end

indices = [1,5,10,15,20];
for i = 1 : numel(indices)
    figure;
    pos = get(gcf, 'Position');
    set(gcf, 'Position', [pos(1) pos(2) width*100, height*100]); %<- Set size
    set(gca, 'FontSize', fsz, 'LineWidth', alw); %<- Set properties
    hold on;
    plot(log(moneyness_mat_1(indices(i),:)), iv_mat_1(indices(i),:), "DisplayName", "Markov Approximation");
    plot(log(moneyness_mat_2(indices(i),:)), iv_mat_2(indices(i),:), "DisplayName", "Classical Hybrid");
    plot(log(moneyness_mat_2(indices(i),:)), iv_mat_3(indices(i),:), "DisplayName", "Transform Inversion");
    plot(log(moneyness_mat_4(indices(i),:)), iv_mat_4(indices(i),:), "DisplayName", "Accelerated Hybrid");
    plot(log(moneyness_mat_5(indices(i),:)), iv_mat_5(indices(i),:), "DisplayName", "Euler Scheme");
    grid off;
    box on;
    hold off;

    xlabel('$\log(K/S)$', 'Interpreter','latex');
    ylabel('Implied Volatility', 'Interpreter','latex');
    legend('Location','best','Interpreter','latex','FontSize',8);
    title(sprintf('IV: $T = %d/252$, $H = %.2f$', indices(i), H), 'Interpreter','latex');
    print(gcf, '-dpng', '-r300', sprintf('assets/iv/iv_%d_Day_point15H_400nf_50n_neg7rho.png', indices(i)));
end

%% plot implied volatility surface
figure;
pos = get(gcf, 'Position');
set(gcf, 'Position', [pos(1) pos(2) width*100, height*100]); %<- Set size
set(gca, 'FontSize', fsz, 'LineWidth', alw); %<- Set properties

tc = colormapToTrueColor(winter(), iv_mat_4);
surf(moneyness_mat_4, results_4.tau_vec, iv_mat_4, tc, 'FaceColor','interp', ...
    'EdgeAlpha',0.3,'DisplayName','Accelerated Hybrid');

hold on;
tc = colormapToTrueColor(autumn(), iv_mat_3);
surf(moneyness_mat_4, results_4.tau_vec, iv_mat_3, tc, 'FaceColor','interp', ...
    'EdgeAlpha',0.3,'DisplayName','Transform Inversion');

xlabel("$K$", "Interpreter", "latex");
ylabel("$T$", "Interpreter", "latex");
zlabel("Implied Volatility", "Interpreter", "latex");
title("$H=0.15, \rho=-0.7$", 'Interpreter','latex');

% colorbar;
% view(45,30);
legend('Location','northeast','Interpreter','latex');
grid on;
box off;
hold off;
print(gcf, '-dpng', '-r300', 'assets/iv/iv_surface_point15H_400nf_50n_neg7rho.png');

%% Report
% err_exact_1 = sqrt(mean((iv_mat_1 - iv_mat_3).^2,'all')); err_exact_2 = sqrt(mean((iv_mat_2 - iv_mat_3).^2,'all')); err_exact_3 = 0; err_exact_4 = sqrt(mean((iv_mat_4 - iv_mat_3).^2,'all'));
% err_hyb_1 = sqrt(mean((iv_mat_1 - iv_mat_2).^2,'all')); err_hyb_2 = 0; err_hyb_3 = sqrt(mean((iv_mat_3 - iv_mat_2).^2,'all')); err_hyb_4 = sqrt(mean((iv_mat_4 - iv_mat_2).^2,'all'));
fprintf("Parameters: nf = %d, steps_per_maturity = %d, H = %.2f, rho = %.1f\n", nf, step_maturity, H, rho);
fprintf("Markov Approximation: cpu time: %5.2f\n", t1);
fprintf("    Classical Hybrid: cpu time: %5.2f\n", t2);
fprintf(" Transform Inversion: cpu time: %5.2f\n", t3);
fprintf("  Accelerated Hybrid: cpu time: %5.2f\n", t4);
fprintf("        Euler Scheme: cpu time: %5.2f\n", t5);