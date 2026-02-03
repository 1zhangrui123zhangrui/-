%% 第一阶段：基于原论文场景的“物理有效性”验证
clear; clc; close all;

% 1. 全局参数（沿用原论文设置）
m = 7;
area_size = 1000;
N = 400; 

% 定义需要测试的经典场景 (调用你原有的 generate_swarm_data)
scenarios = {'line', 'circle', 'spiral', 'grid', 'random'};
scene_names = {'直线', '圆环', '螺旋', '网格', '随机'};

results = [];

figure('Color', 'w', 'Name', '原论文图形验证 (运动同步状态)');

for i = 1:length(scenarios)
    % 2. 生成你代码中的标准数据
    [X, Y] = generate_swarm_data(scenarios{i}, N, area_size);
    
    % 3. 赋予理想状态：全同步速度 (1, 1)
    VX = ones(N, 1);
    VY = ones(N, 1);
    
    % 4. 计算对比指标
    [s_orig, s_imp, G] = calc_engine_v2(X, Y, VX, VY, m);
    results = [results; [s_orig, s_imp]];
    
    % 5. 可视化原论文图形
    subplot(2, 3, i);
    plot(X, Y, 'b.', 'MarkerSize', 5);
    axis equal; grid on;
    title(sprintf('%s (\\Gamma=%.2f)', scene_names{i}, G));
end

% 6. 数据对比展示
fprintf('\n--- 理想同步状态下的指标对比 (向下兼容性验证) ---\n');
VarNames = {'原_M1', '原_M2', '原_M3', '新_W1', '新_W2', '新_W3'};
T = table(results(:,1), results(:,2), results(:,3), ...
          results(:,4), results(:,5), results(:,6), ...
          'VariableNames', VarNames, 'RowNames', scenarios);
disp(T);

% 7. 柱状图展示
figure('Color', 'w', 'Position', [100, 100, 800, 400]);
bar(results);
set(gca, 'XTickLabel', scene_names);
legend(VarNames, 'Location', 'eastoutside');
title('原算法与新算法评分对比 (同步运动场景)');
ylabel('规律性评分 \rho');
grid on;