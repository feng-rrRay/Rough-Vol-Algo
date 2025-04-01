clear; clc;
rng(1);
addpath("../Fourier_Transform_Inversion");

%% Parameters
%  Rough Heston model:  (8.1.3)-(8.1.4) in (Jaber, 2018)
Hs = [0.1,0.15,0.2,0.3,0.4];
for Hindex = 1:5
    rng(1);
    H = Hs(Hindex);
    theta = 0.05;
    rho = -0.1;
    lambda = 0.1;
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
    dt_maturity = 1/12;  % gap between maturities in IV surface
    num_maturity = 1;    % number of maturities for IV surface
    num_moneyness = 41;
    J = 2; % hybrid scheme #(Gaussian noise)
    steps_per_maturity_vec = [2:5,10:5:100,110:10:320];
    
    total_steps_vec = num_maturity .* steps_per_maturity_vec;
    call_prices_mat = zeros(4,size(steps_per_maturity_vec,2));
    time_mat = zeros(4,size(steps_per_maturity_vec,2));
    stat_error_mat = zeros(4,size(steps_per_maturity_vec,2));
    
    % Fourier inverse transform
    left = 1e-12;
    right = 400;
    n = 1000; N = 20000;
    k = 0; S = 1; d = r;
    phi_fourier = CallPriceArray(S,k,H,rho,nu,lambda,theta,V0,dt_maturity*num_maturity,r,d,n,left,right,N);
    fprintf("Ref European call price: %.6f, S=%d, k=%.2f\n", phi_fourier,S,k);
    
    %% ---
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
        
        [logS_final_mc_hyb,t_hyb] = simu_hyb_weak(V0, num_maturity, dt_maturity,...
            num_moneyness, num_path_mc, step_maturity, forward_curve_hyb,...
            theta, rho, lambda, nu, H, noise_mat_3d_mc, J, r, eps);
    
        [logS_final_mc_markov,logS_final_mc_acc,t_markov,t_acc] = simu_acc_weak(V0, num_maturity, dt_maturity,...
            num_moneyness, num_path_mc, step_maturity, u_current, forward_curve_hyb, cs, xs, ...
            theta, rho, lambda, nu, H, noise_mat_3d_mc, J, r, eps);
    
        [logS_final_mc_euler,t_euler] = simu_euler_weak(V0, num_maturity, dt_maturity,...
            num_moneyness, num_path_mc, step_maturity, u_current, forward_curve_hyb, cs, xs, ...
            theta, rho, lambda, nu, H, noise_mat_3d_mc, J, r, eps);
    
        [phi_val_hyb, stat_error_hyb] = phi_func(logS_final_mc_hyb, exp(k));
        [phi_val_acc, stat_error_acc] = phi_func(logS_final_mc_acc, exp(k));
        [phi_val_markov, stat_error_markov] = phi_func(logS_final_mc_markov,exp(k));
        [phi_val_euler, stat_error_euler] = phi_func(logS_final_mc_euler,exp(k));
    
        call_prices_mat(1,ii) = phi_val_hyb;
        call_prices_mat(2,ii) = phi_val_acc;
        call_prices_mat(3,ii) = phi_val_markov;
        call_prices_mat(4,ii) = phi_val_euler;
    
        time_mat(1,ii) = t_hyb;
        time_mat(2,ii) = t_acc;
        time_mat(3,ii) = t_markov;
        time_mat(4,ii) = t_euler;
    
        stat_error_mat(1,ii) = stat_error_hyb;
        stat_error_mat(2,ii) = stat_error_acc;
        stat_error_mat(3,ii) = stat_error_markov;
        stat_error_mat(4,ii) = stat_error_euler;
    
    end
    
    %% save result to file
    s_out = struct();
    s_out.H = H; s_out.lambda = lambda; s_out.rho = rho; s_out.theta = theta; s_out.nu = nu; s_out.V0 = V0;
    s_out.nf = nf; s_out.T = dt_maturity*num_maturity; s_out.kappa = J;
    s_out.timesteps = steps_per_maturity_vec*num_maturity;
    s_out.refPrice = phi_fourier;
    s_out.callPrices = call_prices_mat;
    s_out.computeTime = time_mat;
    s_out.statErrors = stat_error_mat;
    s_out.displayNames = ["Classical Hybrid", "Accelerated Hybrid", "Markov Approximate", "Euler Scheme"];
    json_add("EuropeanCallConvergeN_1M.json",s_out);
    
    %% plot
    figure;
    hold on;
    yline(phi_fourier,'-','Reference',"DisplayName","Reference")
    plot(steps_per_maturity_vec*num_maturity, call_prices_mat(1,:),"Color","g","DisplayName","Classical Hybrid");
    plot(steps_per_maturity_vec*num_maturity, call_prices_mat(2,:),"Color","b","DisplayName","Accelerated Hybrid");
    plot(steps_per_maturity_vec*num_maturity, call_prices_mat(3,:),"Color","r","DisplayName","Markov Approximate");
    plot(steps_per_maturity_vec*num_maturity, call_prices_mat(4,:),"Color","y","DisplayName","Euler Scheme");
    hold off;
    legend;

end

%% debug zigzag problem
% clear; clc;
% rng(1);
% addpath("../Fourier_Transform_Inversion");
% 
% H = 0.2;
% theta = 0.05;
% rho = -0.1;
% lambda = 0.1;
% nu = 0.3;
% r = 0;
% V0 = 0.05;
% eps = 0.0001;
% 
% %  Lifted Heston model: (8.2.1)-(8.2.3) in (Jaber, 2018)
% nf = 100; % number of factors (50 would yield reasonable results for 1 day to maturity)
% alpha = H + 0.5;
% rn = 1 + 10 * nf^(-0.9);
% cs = ones(1,nf) .* ((rn^(1-alpha) - 1)*rn^((alpha-1)*(1+nf/2))) ./ (gamma(alpha) * gamma(2-alpha));
% xs = ones(1,nf) .* ((1-alpha)/(2-alpha)*(rn^(2-alpha)-1)/(rn^(1-alpha)-1));
% for i = 1:nf
%     cs(i) = cs(i) * rn^((1-alpha)*i);
%     xs(i) = xs(i) * rn^(i-1-nf/2);
% end
% 
% % time grid definition
% dt_maturity = 1/252;  % gap between maturities in IV surface
% num_maturity = 1;    % number of maturities for IV surface
% num_moneyness = 41;
% J = 2; % hybrid scheme #(Gaussian noise)
% steps_per_maturity_vec = [2:5,10:5:100,110:10:320];
% 
% total_steps_vec = num_maturity .* steps_per_maturity_vec;
% call_prices_mat = zeros(1,size(steps_per_maturity_vec,2));
% % time_mat = zeros(1,size(steps_per_maturity_vec,2));
% stat_error_mat = zeros(1,size(steps_per_maturity_vec,2));
% 
% for ii = 1:size(steps_per_maturity_vec,2)
%     disp(ii);
%     rng(1);
%     step_maturity = steps_per_maturity_vec(ii);
%     num_steps_mc = num_maturity * step_maturity; % number of steps in total
%     num_path_mc = 3e5;    % Monte Carlo paths
%     dt = dt_maturity / step_maturity; 
%     forward_curve_hyb = lambda * theta * ((1:num_steps_mc)*dt).^(H+1/2) / (H+1/2) / gamma(H+1/2);
%     u_current = zeros(1,nf);
% 
%     k = 0;
%     noise_mat_3d_mc = normrnd(0,sqrt(dt),[num_path_mc,num_steps_mc,2]);
%     [logS_final_mc_euler,t_euler] = simu_euler_weak(V0, num_maturity, dt_maturity,...
%             num_moneyness, num_path_mc, step_maturity, u_current, forward_curve_hyb, cs, xs, ...
%             theta, rho, lambda, nu, H, noise_mat_3d_mc, J, r, eps);
%     [phi_val_euler, stat_error_euler] = phi_func(logS_final_mc_euler,exp(k));
%     call_prices_mat(1,ii) = phi_val_euler;
%     stat_error_mat(1,ii) = stat_error_euler;
% end
% figure;
% plot(total_steps_vec,call_prices_mat);