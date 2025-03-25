clear; clc;
rng(1);

%% Parameters
%  Rough Heston model:  (8.1.3)-(8.1.4) in (Jaber, 2018)
H = 0.15;
theta = 0.09;
rho = 0;
lambda = 0.05;
nu = 0.3;
r = 0;
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
dt_maturity = 1/252;  % gap between maturities in IV surface
num_maturity = 20;    % number of maturities for IV surface
num_moneyness = 41;
J = 2; % hybrid scheme #(Gaussian noise)
steps_per_maturity_vec = [5,10,20,30,40,50,75];

total_steps_vec = num_maturity .* steps_per_maturity_vec;
res_table = zeros(1,size(steps_per_maturity_vec,2));

for ii = 1:size(steps_per_maturity_vec,2)
    fprintf("Total steps N = %d\n", total_steps_vec(ii));
    
    step_maturity = steps_per_maturity_vec(ii);
    num_steps_mc = num_maturity * step_maturity; % number of steps in total
    num_path_mc = 3e5;    % Monte Carlo paths
    dt = dt_maturity / step_maturity; 

    % Kernel coefficients and random matrices
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
    % L = cholcov(Sigma);
    % disp(L);
    
    for j = J+2:-1:1
        noise_mat_3d_mc(:,:,j) = L(j,j) * noise_mat_3d_mc(:,:,j);
        for i = 1:j-1
            noise_mat_3d_mc(:,:,j) = noise_mat_3d_mc(:,:,j) + L(j,i) * noise_mat_3d_mc(:,:,i);
        end
    end

    forward_curve_hyb = lambda * theta * ((1:num_steps_mc)*dt).^(H+1/2) / (H+1/2) / gamma(H+1/2);
    u_current = zeros(1,nf);
    
    V_mat_mc_hyb = simu_hyb(V0, num_maturity, dt_maturity,...
        num_moneyness, num_path_mc, step_maturity, forward_curve_hyb,...
        theta, rho, lambda, nu, H, noise_mat_3d_mc, J, r, eps);

    V_mat_mc_acc = simu_acc(V0, num_maturity, dt_maturity,...
        num_moneyness, num_path_mc, step_maturity, u_current, forward_curve_hyb, cs, xs, ...
        theta, rho, lambda, nu, H, noise_mat_3d_mc, J, r, eps);

    res_table(ii) = mean((V_mat_mc_acc(:,num_steps_mc+1) - V_mat_mc_hyb(:,num_steps_mc+1)).^2);

end

%% plot
plot(steps_per_maturity_vec, res_table);