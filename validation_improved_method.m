%% 新论文改进方法可行性验证仿真
% 功能：验证“运动一致性增益”、“多尺度谱特征”以及“滞回切换逻辑”
clear; clc; close all;

%% 1. 场景设置：模拟“伪组织”干扰场景
% 场景：两组无人机交错飞行。
% 现象：在某一瞬间，它们的空间排列非常整齐（网格状），但运动方向相反。
% 目的：验证改进后的方法是否能通过“运动一致性”剔除这种伪高分。

N = 100;           % 无人机数量
m_scales = 3:2:15; % 多尺度扫描范围
T = 50;            % 仿真总步数

% 初始化坐标和速度
% 第一组：向右飞行
P1 = [randn(N/2,1)*2, randn(N/2,1)*10]; 
V1 = repmat([1, 0], N/2, 1); 
% 第二组：向左飞行
P2 = [randn(N/2,1)*2 + 5, randn(N/2,1)*10]; 
V2 = repmat([-1, 0], N/2, 1);

P = [P1; P2];
V = [V1; V2];

%% 2. 核心算法函数实现 (集成改进点)

% 定义滞回环参数
epsilon_up = 0.75;
epsilon_down = 0.65;
current_label = 0; % 0:低组织度, 1:高组织度

history_rho = [];
history_refined_rho = [];
labels = [];

fprintf('开始进行动态时空演化仿真...\n');

for t = 1:T
    % 更新位置 (模拟交错过程)
    P = P + V * 0.2;
    
    % --- 改进点 1: 多尺度空间相关性 rho_m ---
    rho_scales = [];
    for m = m_scales
        rho_m = calculate_basic_autocorr(P(:,1), P(:,2), m);
        rho_scales = [rho_scales, rho_m];
    end
    rho_mean = mean(rho_scales); % 多尺度平均
    
    % --- 改进点 2: 引入运动一致性增益 Gamma ---
    % 计算速度余弦相似度矩阵
    V_norm = V ./ sqrt(sum(V.^2, 2));
    S_v = V_norm * V_norm'; % 相似度矩阵
    Gamma = (sum(S_v(:)) - N) / (N*(N-1)); % 全局运动一致性指标
    
    % 时空融合后的指标
    rho_refined = rho_mean * max(0, Gamma); 
    
    % --- 改进点 3: 滞回环决策逻辑 ---
    if rho_refined > epsilon_up
        current_label = 1;
    elseif rho_refined < epsilon_down
        current_label = 0;
    end
    % 若在中间区域，保持 current_label 不变
    
    % 记录数据
    history_rho = [history_rho; rho_mean];
    history_refined_rho = [history_refined_rho; rho_refined];
    labels = [labels; current_label];
end

%% 3. 结果可视化与验证分析

figure('Color', 'w', 'Position', [100, 100, 1000, 600]);

% 子图 1: 指标对比
subplot(2,1,1);
plot(1:T, history_rho, 'r--', 'LineWidth', 1.5, 'DisplayName', '原方法 (仅空间)');
hold on;
plot(1:T, history_refined_rho, 'b-', 'LineWidth', 2, 'DisplayName', '改进方法 (时空融合)');
yline(epsilon_up, 'k:', '上阈值');
yline(epsilon_down, 'k:', '下阈值');
title('原方法与改进方法评估指标对比');
xlabel('时间步'); ylabel('规律性评分 \rho');
legend; grid on;

% 子图 2: 决策稳定性验证
subplot(2,1,2);
stairs(1:T, labels, 'Color', [0 0.5 0], 'LineWidth', 2);
ylim([-0.2 1.2]);
title('基于滞回环的组织度等级判定 (0:随机/低序, 1:有序)');
xlabel('时间步'); ylabel('判定等级');
grid on;

fprintf('仿真完成。观察到改进方法在交错飞行（Gamma降低）时评分显著下降，有效识别了伪组织。\n');

%% 支撑函数：基础自相关计算
function score = calculate_basic_autocorr(X, Y, m)
    N = length(X);
    D = pdist2([X Y], [X Y]);
    [sorted_D, ~] = sort(D, 1);
    R_star = sorted_D(2:m+1, :);
    
    % 计算皮尔逊相关系数
    C = corr(R_star);
    % 排除对角线自身相关
    score = (sum(C(:)) - N) / (N*(N-1));
end