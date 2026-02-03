%% 第一阶段：基础几何形态的“物理有效性”验证
clear; clc; close all;

% --- 1. 固定参数 ---
m = 7; 
area_size = 1000;
N_base = 400; 
N_grid = 1600;
V_sync = [10, 5]; % 初始验证：赋予完全同步的速度 (Gamma=1)

% 场景列表 (严格对应你的 generate_swarm_data.m)
scenarios = {
    {'line',    [], '1. 直线', N_base},...
    {'circle',  [], '2. 圆环', N_base},...
    {'spiral',  [], '3. 螺旋', N_base},...
    {'grid',    [], '4. 网格', N_grid},...
    {'random',  [], '5. 随机', N_base},...
    {'cluster', 1,  '6. 聚类:有序中心+随机簇内', N_grid},...
    {'cluster', 2,  '7. 聚类:随机中心+随机簇内', N_grid},...
    {'cluster', 3,  '8. 聚类:有序中心+有序簇内', N_grid},...
    {'cluster', 4,  '9. 聚类:随机中心+有序簇内', N_grid}
};

res_all = zeros(length(scenarios), 6); 

figure('Color', 'w', 'Name', 'Stage 1: 全场景物理有效性验证', 'Position', [50, 50, 1400, 900]);

for i = 1:length(scenarios)
    item = scenarios{i};
    % 生成坐标
    [X, Y] = generate_swarm_data(item{1}, item{4}, area_size, item{2});
    
    % 赋予全同步运动 (用于验证向下兼容性)
    VX = ones(size(X)) * V_sync(1);
    VY = ones(size(Y)) * V_sync(2);
    
    % 调用修正后的算法引擎
    [orig, imp, G] = Improved_Algorithm_Engine(X, Y, VX, VY, m);
    res_all(i, :) = [orig, imp];
    
    % 可视化
    subplot(3, 3, i);
    plot(X, Y, 'b.', 'MarkerSize', 4); hold on;
    % 绘制少量箭头展示运动方向
    q = round(linspace(1, length(X), 25));
    quiver(X(q), Y(q), VX(q), VY(q), 0.5, 'r', 'LineWidth', 1);
    axis equal; grid on;
    % 重点显示 M2 的得分变化
    title({item{3}, sprintf('\\rho_{M2}=%.3f', orig(2))}, 'FontSize', 10);
end

% --- 结果汇总 ---
VarNames = {'Orig_M1','Orig_M2','Orig_M3','New_W1','New_W2','New_W3'};
T = table(res_all(:,1), res_all(:,2), res_all(:,3), ...
          res_all(:,4), res_all(:,5), res_all(:,6), ...
          'VariableNames', VarNames, 'RowNames', cellfun(@(x) x{3}, scenarios, 'UniformOutput', false));

fprintf('\n>>> 物理有效性验证结果 (应观察到场景 5,6,7 的 M2 处于低位) <<<\n');
disp(T);