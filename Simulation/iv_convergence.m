clear; clc;
rng(1);

%% Parameters
%  Rough Heston model:  (8.1.3)-(8.1.4) in (Jaber, 2018)
H = 0.15;
theta = 0.05;
rho = -0.1;
lambda = 0.1;
nu = 0.3;
r = 0; d = 0;
V0 = 0.05;
eps = 0.0001;

%  Lifted Heston model: (8.2.1)-(8.2.3) in (Jaber, 2018)
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
dt_maturity = 1/12;  % T
num_maturity = 1;    % FIXED TO 1
num_moneyness = 41;
J = 2; % hybrid scheme #(Gaussian noise)
num_path_mc = 3e5;    % Monte Carlo paths

%% Fourier Inverse Transform
addpath('Fourier_Transform_Inversion');
n = 1000; N = 20000;
left = 1e-12; right = 400;
iv_mat_ref = zeros(1, num_moneyness);
log_strikes = linspace(-2*sqrt(V0*dt_maturity),2*sqrt(V0*dt_maturity),num_moneyness);
iv_mat_ref(1,:) = IVCurveArray(log_strikes, H, rho, nu, lambda, theta, V0, dt_maturity, r, d, n, left, right, N);

n_vec = 2.^(4:10);
iv_mat_simu = zeros(size(n_vec,2),num_moneyness);

for ii = 1 : size(n_vec,2)
    fprintf("n = %d\n", n_vec(ii));
    step_maturity = n_vec(ii);  % n VARYING HERE
    num_steps_mc = num_maturity * step_maturity; % number of steps in total
    dt = dt_maturity / step_maturity; 

    Sigma = zeros(J+2,J+2);
    j_vec = 2:J+1; k_vec = 2:J+1;
    Sigma(1,3:J+2) = ((j_vec-1).^(H+1/2) - (j_vec-2).^(H+1/2)) / (H+1/2) * dt^(H+1/2); % dt term absorbed in path generation in reference code
    F1mat = hypergeom([-H+1/2,1],H+3/2,[1,j_vec]'./[1,k_vec].*([1,j_vec]'<[1,k_vec]));
    Sigma(3:J+2,3:J+2) = ((j_vec'<k_vec).*((j_vec-1)'.^(H+1/2).* (k_vec-1).^(H-1/2) .* F1mat(j_vec-1,k_vec-1) ... 
        -(j_vec-2)'.^(H+1/2).* ([1,1:J-1]).^(H-1/2) .* F1mat([1,1:J-1],[1,1:J-1]))/(H+1/2))* dt^(2*H);
    Sigma = Sigma + Sigma';
    Sigma(logical(eye(J+2))) = [dt, dt, ((j_vec-1).^(2*H) - (j_vec-2).^(2*H)) / (2*H) * dt^(2*H)]';
    
    % generate noise matrix (W_i, W_i', W_{i,1},...,W_{i,J}), i = 0,1,...,nT,
    % n = 1/dt, T = num_maturity * dt_maturity
    noise_mat_3d_mc = normrnd(0,1,[num_path_mc,num_steps_mc,J+2]);
    L = chol(Sigma)'; % lower triangular matrix; **numerical problem here if H=1/2**
    
    for j = J+2:-1:1
        noise_mat_3d_mc(:,:,j) = L(j,j) * noise_mat_3d_mc(:,:,j);
        for i = 1:j-1
            noise_mat_3d_mc(:,:,j) = noise_mat_3d_mc(:,:,j) + L(j,i) * noise_mat_3d_mc(:,:,i);
        end
    end
    
    u_current = zeros(1,nf);
    num_steps_mc = num_maturity * step_maturity;
    % forward_curve = lambda * theta * sum(cs./xs .* (1-exp(-(1:num_steps_mc)'*dt*xs)),2)'; % func g_0(t)
    forward_curve_hyb = lambda * theta * ((1:num_steps_mc)*dt).^(H+1/2) / (H+1/2) / gamma(H+1/2);
    [iv_mat, tau_vec, moneyness_mat] = iv_simu_pc(V0, num_maturity, dt_maturity,...
        num_moneyness, num_path_mc, step_maturity, u_current, forward_curve_hyb, cs, xs, ...,
        theta, rho, lambda, nu, H, noise_mat_3d_mc, J, r, eps);
    iv_mat_simu(ii,:) = iv_mat;
end

%% save to json
s_out = struct();
s_out.H = H; s_out.lambda = lambda; s_out.rho = rho; s_out.theta = theta; s_out.nu = nu; s_out.V0 = V0;
s_out.nf = nf; s_out.T = dt_maturity; s_out.kappa = J;
s_out.timesteps = n_vec;
s_out.kVec = log_strikes;
s_out.moneyness = moneyness_mat;
s_out.ivRef = iv_mat_ref;
s_out.ivSimu = iv_mat_simu;
json_add("IVConvergeN.json",s_out);

%% plot
width = 6;     % Width in inches
height = 4.5;    % Height in inches
alw = 0.75;    % AxesLineWidth
fsz = 11;      % Fontsize
lw = 1.5;      % LineWidth
msz = 8;       % MarkerSize

figure; 
pos = get(gcf, 'Position');
set(gcf, 'Position', [pos(1) pos(2) width*100, height*100]); %<- Set size
set(gca, 'FontSize', fsz, 'LineWidth', alw); %<- Set properties
colors = lines(size(n_vec,2)+1);
hold on;
plot(moneyness_mat, iv_mat_ref, 'LineStyle','--', 'Color', colors(1,:), 'LineWidth',lw*1.5,'MarkerSize',msz,'DisplayName', 'Fourier Inverse');
for j = 1 : size(n_vec,2)
    legend_name = sprintf("$n=%d$", n_vec(j));
    plot(moneyness_mat, iv_mat_simu(j,:), 'Color', colors(j+1,:), 'LineWidth',lw,'MarkerSize',...
        msz,'DisplayName', legend_name);
end
xlabel('Moneyness ($K/S$)','Interpreter','latex','FontName', 'Times New Roman');
ylabel('Implied Volatility','FontName', 'Times New Roman');
legend('Location','best','FontName', 'Times New Roman', 'Interpreter','latex');
grid off;
box on;
hold off;

%% save figure
fileName = sprintf("iv_fig/iv_converge_1M_H_%d.png",H*100);
print(gcf, '-dpng', '-r300', fileName);