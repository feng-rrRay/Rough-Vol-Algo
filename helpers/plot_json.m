clear;clc;
addpath("../");
addpath("../../Fourier_Transform_Inversion");
data = jsondecode(fileread("EuropeanCallConvergeN_1M.json"));
% data = jsondecode(fileread("EuropeanCallConvergeNf.json"));

%% Defaults for the plot
width = 4;     % Width in inches
height = 3;    % Height in inches
alw = 0.75;    % AxesLineWidth
fsz = 11;      % Fontsize
lw = 1.5;      % LineWidth
msz = 8;       % MarkerSize

%% loop
for i = 1 : numel(data)
    wrapper = data(i);
    if ~isfield(wrapper, "refPrice")
        % Fourier inverse transform
        left = 1e-12;
        right = 400;
        n = 1000; N = 20000;
        k = 0; S = 1; r = 0; d = r;
        refPrice = CallPriceArray(S, k, wrapper.H, wrapper.rho, wrapper.nu,...
            wrapper.lambda, wrapper.theta, wrapper.V0, wrapper.T, r, d,...
            n, left, right, N);
        wrapper.refPrice = refPrice;
        fprintf("Ref European call price: %.6f, S=%d, k=%.2f\n", refPrice,S,k);
    end
    %% plot
    x = wrapper.timesteps';  % x is a 1x45 vector
    % x = wrapper.nfs';
    
    figure; 
    pos = get(gcf, 'Position');
    set(gcf, 'Position', [pos(1) pos(2) width*100, height*100]); %<- Set size
    set(gca, 'FontSize', fsz, 'LineWidth', alw); %<- Set properties

    hold on;
    yline(wrapper.refPrice,'-','Reference','LineWidth',lw,"DisplayName","Reference");
    colors = lines(4);
    for j = 1 : size(wrapper.displayNames, 1)
        % y = wrapper.computeTime(j, :);
        y = wrapper.callPrices(j, :);
        err = wrapper.statErrors(j, :);

        % Calculate 95% confidence intervals
        y_lower = y - 1.96 * err;
        y_upper = y + 1.96 * err;

        % Create a vector for the filled area: first the upper bound, then the lower bound in reverse.
        x_fill = [x, fliplr(x)];
        y_fill = [y_upper, fliplr(y_lower)];

        % Plot the confidence interval as a filled area (with 20% opacity)
        fill(x_fill, y_fill, colors(j,:), 'FaceAlpha', 0.2, 'EdgeColor', 'none', ...
            'DisplayName',"CI:"+wrapper.displayNames{j},'HandleVisibility', 'off');

        % Plot the mean line on top of the filled area
        plot(x, y, 'Color', colors(j,:), 'LineWidth',lw,'MarkerSize',msz,'DisplayName', wrapper.displayNames{j});
    end
    
    xlabel('$n$','Interpreter','latex','FontName', 'Times New Roman');
    % xlabel('$n_f$','Interpreter','latex','FontName', 'Times New Roman');
    % ylabel('CPU Time (s)','FontName', 'Times New Roman');
    ylabel('European Call Price','FontName', 'Times New Roman');
    
    legend('Location','best','FontName', 'Times New Roman');
    grid off;
    box on;
    hold off;
    % save figure
    fileName = sprintf("convergence_n_1M_H_%d.png",wrapper.H*100);
    print(gcf, '-dpng', '-r300', fileName);
end