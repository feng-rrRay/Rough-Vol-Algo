clear;clc;
addpath("../");
addpath("../../Fourier_Transform_Inversion");
data = jsondecode(fileread("EuropeanCallConvergeN_1M.json"));

width = 4;     % Width in inches
height = 3;    % Height in inches
alw = 0.75;    % AxesLineWidth
fsz = 11;      % Fontsize
lw = 1.5;      % LineWidth
msz = 8;       % MarkerSize

for i = 1 : numel(data)
    wrapper = data(i);
    % plot
    figure;
    hold on;
    errors = abs(wrapper.callPrices - wrapper.refPrice);
    for j = 1 : size(wrapper.computeTime,1)
        plot(wrapper.computeTime(j,:), errors(j,:), ...
            'LineWidth',lw, 'DisplayName',wrapper.displayNames{j});
    end
    legend;
    title(sprintf("$H = %.2f$", wrapper.H), 'Interpreter','latex');
    hold off;
end