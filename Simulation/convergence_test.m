clear; clc;
rng(42);
addpath("helpers/");
%% begin test
nList = [25, 50, 100, 250, 500];
nfList = [50, 100, 200, 400, 800];
H = 0.1;
T = 1/252;
% nf = 200;
J = 2;
titleStr = "$H = 0.1, T = 1/252$";

taskId = 1;


if taskId == 1
    results = testConvergenceAcchyb(nList, nfList, H, T, J, inf);
elseif taskId == 2
    results = testConvergenceN(nList, H, T, nf, J);
end

%% save
dirNameList = {"conv1", "conv2"};
instanceName = "42_conv1_1Day_point1H";
filePath = fullfile('assets', dirNameList{taskId}, instanceName);

% save results
json_add(filePath+'.json', results);

%% plot
% plot parameters
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
    
if taskId == 1
    figure;
    pos = get(gcf, 'Position');
    set(gcf, 'Position', [pos(1) pos(2) width*100, height*100]); %<- Set size
    set(gca, 'FontSize', fsz, 'LineWidth', alw); %<- Set properties
    
    hold on;
    for k = 1 : numel(nfList)
        mask = [results.nf] == nfList(k);
        data = results(mask);
        plot(log([data.n]), log([data.rmse]), 'DisplayName', ...
            sprintf("$n_f$ = %d", nfList(k)), 'Marker', 'x');
    end
    title(titleStr, 'Interpreter', 'latex');
    xlabel("$\log(n)$", 'Interpreter', 'latex');
    ylabel("$\log(RMSE)$", 'Interpreter', 'latex');
    grid off;
    box on;
    legend('Interpreter','latex','Location','best');
    print(gcf, '-dpng', '-r300', filePath+'.png');

elseif taskId == 2
    json_add("assets/"+instanceName+".json", results);
    
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
    title(titleStr, 'Interpreter', 'latex');
    xlabel("$\log(n)$", 'Interpreter', 'latex');
    ylabel("$\log(RMSE)$", 'Interpreter', 'latex');
    legend('Interpreter','latex', 'FontSize', 8, 'Location','best');
    grid off;
    box on;
    hold off;
    % save plot
    print(gcf, '-dpng', '-r300', "assets/"+instanceName+".png");
end

%% manual plot
