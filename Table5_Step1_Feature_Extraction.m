%% Table5_Step1_Feature_Extraction.m
% ======================================================================
% 表5仿真 - 第一步: 特征提取
% 严格按照论文设定，从 UCI Swarm Behaviour 数据集提取三种变体的自相关特征
% 
% 论文设定:
%   - 数据集: 24000个样本, 每个样本200架无人机
%   - 每架无人机12个字段: x,y, xVel,yVel, xSm,ySm, xAm,yAm, xCm,yCm, nACm, nSm
%   - m = 8 (第二篇论文 Section 7.1)
%   - 三种变体: corr(1), corr(2), corr(3)
%   - 数据划分: 7:1.5:1.5 (训练:验证:测试)
%
% 输出: 
%   catboost_features_table5.csv  - 包含特征和标签的CSV文件
% ======================================================================
clear; clc; close all;

%% 1. 加载数据集
csv_path = 'Swarm_Behaviour_Data.csv';
if ~exist(csv_path, 'file')
    error(['找不到数据文件: %s\n' ...
           '请从 https://archive.ics.uci.edu/dataset/524/swarm+behaviour 下载\n' ...
           '并将 CSV 文件放在当前目录下。'], csv_path);
end

fprintf('========================================\n');
fprintf('表5仿真 - 特征提取阶段\n');
fprintf('========================================\n\n');

fprintf('[1/4] 正在加载数据集...\n');
raw = readmatrix(csv_path);

% 处理可能的表头行 (NaN)
if isnan(raw(1,1))
    raw(1,:) = [];
end

labels = raw(:, end);          % 最后一列: 标签 (0=非蜂群, 1=蜂群)
data = raw(:, 1:end-1);        % 前面所有列: 特征数据
num_samples = size(data, 1);
num_cols = size(data, 2);

fprintf('  样本总数: %d\n', num_samples);
fprintf('  每样本列数: %d\n', num_cols);
fprintf('  蜂群样本: %d, 非蜂群样本: %d\n', sum(labels==1), sum(labels==0));

% 验证数据格式: 200架无人机 × 12字段 = 2400列
N_drones = 200;     % 每个样本的无人机数量
fields_per_drone = 12;  % 每架无人机的字段数
expected_cols = N_drones * fields_per_drone;
fprintf('  预期列数: %d (200架×12字段)\n', expected_cols);

if num_cols ~= expected_cols
    warning('实际列数 (%d) 与预期 (%d) 不符，请检查数据集格式！', num_cols, expected_cols);
end

%% 2. 特征提取参数设定
% 第二篇论文 Section 7.1: m = 8
m = 8;
fprintf('\n[2/4] 开始特征提取 (m=%d)...\n', m);
fprintf('  变体1: 距离自相关\n');
fprintf('  变体2: 修正距离自相关\n');
fprintf('  变体3: 差分自相关\n\n');

%% 3. 提取三种变体的自相关特征
features = zeros(num_samples, 3);  % [corr(1), corr(2), corr(3)]

tic;
for i = 1:num_samples
    row = data(i, :);
    
    % 提取坐标: 每架无人机的第1,2个字段是 x, y
    % 数据格式: [x1,y1,vx1,vy1,...其他字段..., x2,y2,vx2,vy2,...其他字段..., ...]
    % 每架无人机占12列，x在第1列(偏移0)，y在第2列(偏移1)
    X = row(1:fields_per_drone:end)';   % x坐标, N×1
    Y = row(2:fields_per_drone:end)';   % y坐标, N×1
    
    % 确保正确提取了200个坐标
    if length(X) ~= N_drones || length(Y) ~= N_drones
        warning('样本 %d: 提取了 %d/%d 个坐标，预期 %d', i, length(X), length(Y), N_drones);
        continue;
    end
    
    % 计算三种变体
    features(i, 1) = calculate_swarm_regularity_paper(X, Y, m, 1);  % corr(1)
    features(i, 2) = calculate_swarm_regularity_paper(X, Y, m, 2);  % corr(2)
    features(i, 3) = calculate_swarm_regularity_paper(X, Y, m, 3);  % corr(3)
    
    % 进度显示
    if mod(i, 2000) == 0
        elapsed = toc;
        eta = elapsed / i * (num_samples - i);
        fprintf('  进度: %d/%d (%.1f%%) | 已用时: %.0fs | 预计剩余: %.0fs\n', ...
            i, num_samples, i/num_samples*100, elapsed, eta);
    end
end
total_time = toc;
fprintf('  特征提取完成! 总耗时: %.1f秒 (%.2f秒/样本)\n', total_time, total_time/num_samples);

%% 4. 同时提取速度信息 (第二篇论文SCDAM需要，此处预留)
fprintf('\n[3/4] 提取速度数据 (为后续SCDAM预留)...\n');
velocity_data = zeros(num_samples, N_drones * 2);  % [vx1,vy1, vx2,vy2, ...]
for i = 1:num_samples
    row = data(i, :);
    VX = row(3:fields_per_drone:end);  % vx在第3列(偏移2)
    VY = row(4:fields_per_drone:end);  % vy在第4列(偏移3)
    velocity_data(i, :) = reshape([VX; VY], 1, []);
end
fprintf('  速度数据提取完成\n');

%% 5. 导出CSV供Python CatBoost使用
fprintf('\n[4/4] 导出特征CSV...\n');

% 导出格式: corr1, corr2, corr3, label
T = table(features(:,1), features(:,2), features(:,3), labels, ...
    'VariableNames', {'corr1', 'corr2', 'corr3', 'label'});
writetable(T, 'catboost_features_table5.csv');
fprintf('  已保存: catboost_features_table5.csv (%d行)\n', num_samples);

% 同时保存 MAT 文件 (包含更多信息，供后续SCDAM使用)
save('table5_features.mat', 'features', 'labels', 'velocity_data', 'data', 'm');
fprintf('  已保存: table5_features.mat\n');

%% 6. 快速统计验证
fprintf('\n========================================\n');
fprintf('特征统计摘要:\n');
fprintf('========================================\n');
fprintf('           corr(1)    corr(2)    corr(3)\n');
fprintf('  均值:    %.4f     %.4f     %.4f\n', mean(features));
fprintf('  标准差:  %.4f     %.4f     %.4f\n', std(features));
fprintf('  最小值:  %.4f     %.4f     %.4f\n', min(features));
fprintf('  最大值:  %.4f     %.4f     %.4f\n', max(features));

fprintf('\n--- 按标签分类 ---\n');
idx_swarm = labels == 1;
idx_non = labels == 0;
fprintf('蜂群(label=1):\n');
fprintf('  corr(1): %.4f ± %.4f\n', mean(features(idx_swarm,1)), std(features(idx_swarm,1)));
fprintf('  corr(2): %.4f ± %.4f\n', mean(features(idx_swarm,2)), std(features(idx_swarm,2)));
fprintf('  corr(3): %.4f ± %.4f\n', mean(features(idx_swarm,3)), std(features(idx_swarm,3)));
fprintf('非蜂群(label=0):\n');
fprintf('  corr(1): %.4f ± %.4f\n', mean(features(idx_non,1)), std(features(idx_non,1)));
fprintf('  corr(2): %.4f ± %.4f\n', mean(features(idx_non,2)), std(features(idx_non,2)));
fprintf('  corr(3): %.4f ± %.4f\n', mean(features(idx_non,3)), std(features(idx_non,3)));

%% 7. 特征分布可视化
figure('Name', '表5特征分布', 'Position', [100, 100, 1200, 400], 'Color', 'w');

method_names = {'corr^{(1)} 距离', 'corr^{(2)} 修正距离', 'corr^{(3)} 差分'};
for j = 1:3
    subplot(1, 3, j);
    histogram(features(idx_non, j), 50, 'FaceColor', [0.8 0.2 0.2], 'FaceAlpha', 0.5, 'EdgeColor', 'none');
    hold on;
    histogram(features(idx_swarm, j), 50, 'FaceColor', [0.2 0.2 0.8], 'FaceAlpha', 0.5, 'EdgeColor', 'none');
    legend('非蜂群', '蜂群');
    xlabel(method_names{j});
    ylabel('样本数');
    title(sprintf('变体%d分布', j));
    grid on;
end
sgtitle('三种自相关变体的特征分布 (m=8)', 'FontSize', 14, 'FontWeight', 'bold');

fprintf('\n特征提取阶段完成!\n');
fprintf('下一步: 运行 Python 脚本 train_catboost_table5.py 进行CatBoost训练\n');
fprintf('或运行 Table5_Step2_MATLAB_Classifier.m 使用MATLAB梯度提升树\n');
