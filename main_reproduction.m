%% 复现论文主程序 (main_reproduction.m)
% 功能：调用 generate_swarm_data 和 calculate_swarm_regularity
% 复现论文中的所有空间结构（含4种聚类模式）
%
% 注意：请确保您已经更新了 generate_swarm_data.m 文件！

clear; clc; close all;

% --- 1. 全局参数设置 ---
m = 7;               % 邻居数量
area_size = 1000;    % 区域大小
base_N = 400;        % 基础点数 (用于大多数结构)
grid_N = 1600;       % 网格专用大点数 (消除边缘效应)

% --- 2. 定义9种测试场景 ---
% 格式: {类型, 聚类模式(可选), 显示名称, 点数N}
scenarios = {
    {'line',    [], '1. 直线 (Line)', base_N};
    {'circle',  [], '2. 圆 (Circle - 带缺口)', base_N};
    {'spiral',  [], '3. 螺旋 (Spiral - 等弧长)', base_N};
    {'grid',    [], '4. 网格 (Grid - 完美正方)', grid_N};  % 使用大N
    {'random',  [], '5. 随机 (Random)', base_N};
    {'cluster', 1,  '6. 聚类: 宏观有序+微观随机', base_N};
    {'cluster', 2,  '7. 聚类: 宏观随机+微观随机', base_N};
    {'cluster', 3,  '8. 聚类: 宏观有序+微观有序', base_N};
    {'cluster', 4,  '9. 聚类: 宏观随机+微观有序', base_N};
};

% --- 3. 初始化结果表和画布 ---
results = table('Size', [length(scenarios), 4], ...
    'VariableTypes', {'string', 'double', 'double', 'double'}, ...
    'VariableNames', {'Structure', 'Method_1_Dist', 'Method_2_ModDist', 'Method_3_Diff'});

figure('Name', '无人机群空间结构复现全集', 'Position', [50, 50, 1400, 1000], 'Color', 'w');

% --- 4. 循环测试所有场景 ---
for i = 1:length(scenarios)
    % 解析当前场景参数
    item = scenarios{i};
    type = item{1};
    c_mode = item{2};  % 聚类模式 (仅 cluster 类型有效)
    name = item{3};
    current_N = item{4};
    
    % Step A: 生成数据
    % 注意：这里调用的是您的 generate_swarm_data.m 文件
    % 如果该文件没更新，cluster_mode 参数可能无效
    [X, Y] = generate_swarm_data(type, current_N, area_size, c_mode);
    
    % Step B: 计算三种自相关指标
    s1 = calculate_swarm_regularity(X, Y, m, 1); % 距离法
    s2 = calculate_swarm_regularity(X, Y, m, 2); % 修正距离法 (M2)
    s3 = calculate_swarm_regularity(X, Y, m, 3); % 差值法 (M3)
    
    % Step C: 记录结果
    results.Structure(i) = name;
    results.Method_1_Dist(i) = s1;
    results.Method_2_ModDist(i) = s2;
    results.Method_3_Diff(i) = s3;
    
    % Step D: 可视化绘制
    subplot(3, 3, i);
    
    % 根据点数自动调整点的大小，避免过密
    pt_size = 15;
    if current_N > 1000, pt_size = 5; end
    
    scatter(X, Y, pt_size, 'filled', 'MarkerFaceColor', [0.2 0.5 0.8], 'MarkerEdgeColor', 'none');
    axis equal; grid on; box on;
    
    % 设置坐标轴范围，保持视觉一致性
    xlim([-area_size/1.3, area_size/1.3]);
    ylim([-area_size/1.3, area_size/1.3]);
    set(gca, 'XTickLabel', [], 'YTickLabel', []); % 隐藏坐标数值
    
    % 标题显示核心指标 (M2 和 M3)
    title_str = sprintf('%s\nM2=%.2f | M3=%.2f', name, s2, s3);
    
    % 简单的自动颜色标记 (绿色表示符合预期的特征)
    t_color = 'k'; 
    % 如果是伪蜂群(随机或聚类内部乱)，M3应该低(<0.45) -> 绿色
    if (strcmp(type, 'random') || (strcmp(type, 'cluster') && ismember(c_mode, [1, 2]))) && s3 < 0.45
        t_color = [0 0.6 0]; 
    % 如果是真蜂群(网格或聚类内部有序)，M3应该高(>0.8) -> 绿色
    elseif (strcmp(type, 'grid') || (strcmp(type, 'cluster') && c_mode == 3)) && s3 > 0.8
        t_color = [0 0.6 0];
    end
    
    title(title_str, 'FontSize', 11, 'FontWeight', 'bold', 'Color', t_color);
    
    fprintf('完成: %s (N=%d)\n', name, current_N);
end

% --- 5. 显示最终表格 ---
disp('==========================================================');
disp('复现结果汇总 (注意观察 Cluster 的 M3 变化)');
disp('==========================================================');
disp(results);

sgtitle('论文复现完整结果：空间结构 vs 自相关指标', 'FontSize', 16, 'FontWeight', 'bold');