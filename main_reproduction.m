%% 复现论文《基于自相关分析的无人机群空间结构规律性评估》
% 对应论文解析中的“第三阶段：合成数据验证”
clear; clc; close all;

% 1.1 参数初始化
N = 100;            % 无人机数量
m = 7;              % 邻居数量 (建议 [6,10])
area_size = 1000;   % 区域大小

% 定义要测试的结构类型
struct_types = {'line', 'circle', 'spiral', 'grid', 'random', 'cluster'};
struct_names = {'直线 (Line)', '圆 (Circle)', '螺旋 (Spiral)', '网格 (Grid)', '随机 (Random)', '聚类 (Cluster)'};

% 存储结果
results = table('Size', [length(struct_types), 4], ...
    'VariableTypes', {'string', 'double', 'double', 'double'}, ...
    'VariableNames', {'Structure', 'Method_1_Dist', 'Method_2_ModDist', 'Method_3_Diff'});

figure('Name', '无人机群空间结构可视化', 'Position', [100, 100, 1200, 800]);

%% 循环测试每种结构
for i = 1:length(struct_types)
    type = struct_types{i};
    name = struct_names{i};
    
    % 1. 生成数据
    [X, Y] = generate_swarm_data(type, N, area_size);
    
    % 2. 计算三种方法的自相关值
    score1 = calculate_swarm_regularity(X, Y, m, 1);
    score2 = calculate_swarm_regularity(X, Y, m, 2);
    score3 = calculate_swarm_regularity(X, Y, m, 3);
    
    % 存储结果
    results.Structure(i) = name;
    results.Method_1_Dist(i) = score1;
    results.Method_2_ModDist(i) = score2;
    results.Method_3_Diff(i) = score3;
    
    % 可视化绘制
    subplot(2, 3, i);
    scatter(X, Y, 20, 'filled');
    title(name);
    axis equal; grid on;
    xlabel(['M3 Score: ', num2str(score3, '%.3f')]);
end

%% 显示最终结果表格
disp('==========================================================');
disp('复现结果 (对应论文表格 2, 3, 4, 5)');
disp('==========================================================');
disp(results);

%% 结果分析与自动判定
disp('--- 自动分析 ---');

% 验证随机结构的方法1失效
rand_idx = find(results.Structure == "随机 (Random)");
if results.Method_1_Dist(rand_idx) > 0.9
    disp(['[验证成功] 方法 1 在随机结构下失效 (得分 > 0.9): ', num2str(results.Method_1_Dist(rand_idx))]);
else
    disp('[验证失败] 方法 1 未表现出预期的高分失效。');
end

% 验证随机结构的方法3有效性
if results.Method_3_Diff(rand_idx) < 0.45
    disp(['[验证成功] 方法 3 正确识别随机结构 (得分 < 0.45): ', num2str(results.Method_3_Diff(rand_idx))]);
end

% 验证聚类结构的方法3敏感性
cluster_idx = find(results.Structure == "聚类 (Cluster)");
if results.Method_3_Diff(cluster_idx) < 0.35
    disp(['[验证成功] 方法 3 对聚类结构极其敏感 (得分 < 0.35): ', num2str(results.Method_3_Diff(cluster_idx))]);
else
    disp(['[注意] 方法 3 对聚类得分偏高，可能是簇内过于规律或簇间距不够远。']);
end