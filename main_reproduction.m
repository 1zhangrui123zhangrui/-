%% 复现论文主程序 (main_reproduction.m) [全语法修复版]
clear; clc; close all;

% --- 1. 全局参数 ---
m = 7;               
area_size = 1000;    
base_N = 400;        
grid_N = 1600;       

% --- 2. 场景定义 ---
scenarios = {
    {'line',    [], '1. 直线', base_N};
    {'circle',  [], '2. 圆', base_N};
    {'spiral',  [], '3. 螺旋', base_N};
    {'grid',    [], '4. 网格', grid_N}; 
    {'random',  [], '5. 随机', base_N};
    {'cluster', 1,  '6. 聚类: 有序+随机', grid_N}; 
    {'cluster', 2,  '7. 聚类: 随机+随机', grid_N}; 
    {'cluster', 3,  '8. 聚类: 有序+有序', grid_N}; 
    {'cluster', 4,  '9. 聚类: 随机+有序', grid_N}; 
};

% --- 3. 运行 ---
% 预分配表格
results = table('Size', [length(scenarios), 4], ...
    'VariableTypes', {'string', 'double', 'double', 'double'}, ...
    'VariableNames', {'Structure', 'Method_1', 'Method_2', 'Method_3'});

figure('Position', [50, 50, 1400, 1000], 'Color', 'w');

for i = 1:length(scenarios)
    item = scenarios{i};
    current_N = item{4}; 
    
    % 生成数据
    [X, Y] = generate_swarm_data(item{1}, current_N, area_size, item{2});
    
    % 计算指标 (确保 calculate_swarm_regularity.m 已按下方代码修复)
    s1 = calculate_swarm_regularity(X, Y, m, 1);
    s2 = calculate_swarm_regularity(X, Y, m, 2);
    s3 = calculate_swarm_regularity(X, Y, m, 3);
    
    % 记录结果
    results.Structure(i) = item{3};
    results.Method_1(i) = s1;
    results.Method_2(i) = s2;
    results.Method_3(i) = s3;
    
    % 绘图
    subplot(3, 3, i);
    pt_size = 15;
    if current_N > 1000
        pt_size = 5;
    end
    
    scatter(X, Y, pt_size, 'filled'); 
    axis equal; 
    
    % 设置标题颜色：绿色表示符合预期 (参考论文 Table 5 逻辑)
    t_color = 'k';
    if (i==4 || i==8 || i==9) && s2 > 0.7 % 有序结构 M2/M3 应为高分 
         t_color = [0 0.5 0];
    end
    
    title(sprintf('%s\nM2=%.2f | M3=%.2f', item{3}, s2, s3), 'Color', t_color);
end

disp(results);