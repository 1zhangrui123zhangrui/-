%% 论文全空间结构可视化与评估 (终极整合版)
% 功能：一键复现论文中提到的所有空间结构及其自相关评估指标
% 包含：Line, Circle, Spiral, Grid, Random 以及 4种 Cluster 模式
clear; clc; close all;

% --- 全局设置 ---
m = 7;               % 邻居数量
area_size = 1000;    % 区域大小
base_N = 400;        % 基础点数 (用于非网格结构)
grid_N = 1600;       % 网格专用点数 (为了消除边缘效应，需更大)

% --- 定义要展示的9种结构 ---
% 格式: {类型, 聚类模式(可选), 显示名称, 推荐N}
scenarios = {
    {'line',    [], '1. 直线 (Line)', base_N};
    {'circle',  [], '2. 圆 (Circle - 带缺口)', base_N};
    {'spiral',  [], '3. 螺旋 (Spiral - 等弧长)', base_N};
    {'grid',    [], '4. 网格 (Grid - 完美正方)', grid_N};
    {'random',  [], '5. 随机 (Random)', base_N};
    {'cluster', 1,  '6. 聚类: 宏观有序+微观随机', base_N};
    {'cluster', 2,  '7. 聚类: 宏观随机+微观随机', base_N};
    {'cluster', 3,  '8. 聚类: 宏观有序+微观有序', base_N};
    {'cluster', 4,  '9. 聚类: 宏观随机+微观有序', base_N};
};

% 创建画布
figure('Name', '无人机群空间结构全集', 'Position', [50, 50, 1400, 1000], 'Color', 'w');

% --- 循环生成与绘制 ---
for i = 1:length(scenarios)
    item = scenarios{i};
    type = item{1};
    c_mode = item{2};
    name = item{3};
    current_N = item{4};
    
    % 1. 生成数据 (调用内嵌优化版函数)
    [X, Y] = local_generate_data(type, current_N, area_size, c_mode);
    
    % 2. 计算指标
    s1 = local_calculate_metric(X, Y, m, 1); % 距离法
    s2 = local_calculate_metric(X, Y, m, 2); % 修正距离法 (M2)
    s3 = local_calculate_metric(X, Y, m, 3); % 差值法 (M3)
    
    % 3. 绘图
    subplot(3, 3, i);
    
    % 根据点数调整点的大小
    pt_size = 15;
    if current_N > 1000, pt_size = 5; end
    
    scatter(X, Y, pt_size, 'filled', 'MarkerFaceColor', [0.2 0.5 0.8], 'MarkerEdgeColor', 'none');
    axis equal; grid on; box on;
    
    % 美化坐标轴
    xlim([-area_size/1.3, area_size/1.3]);
    ylim([-area_size/1.3, area_size/1.3]);
    set(gca, 'XTickLabel', [], 'YTickLabel', []); % 隐藏坐标数值，保持整洁
    
    % 标题与结果展示
    % 重点展示 M2 和 M3，因为 M1 在随机情况下已失效
    title_str = sprintf('%s\nM2=%.2f | M3=%.2f', name, s2, s3);
    
    % 根据 M3 进行自动判定着色
    title_color = 'k';
    if contains(name, '随机') || (contains(name, '微观随机') && ~contains(name, '宏观'))
        % 预期低分
        if s3 < 0.45, title_color = [0 0.6 0]; else, title_color = 'r'; end
    elseif contains(name, 'Grid') || contains(name, '微观有序')
        % 预期高分
        if s3 > 0.8, title_color = [0 0.6 0]; else, title_color = 'r'; end
    end
    
    title(title_str, 'FontSize', 11, 'FontWeight', 'bold', 'Color', title_color);
end

sgtitle('论文空间结构复现全集：M2(修正距离) 与 M3(距离差值) 指标对比', 'FontSize', 16, 'FontWeight', 'bold');


%% --- 核心算法函数 (内嵌版) ---

% 1. 数据生成器 (包含所有优化)
function [X, Y] = local_generate_data(type, N, area_size, cluster_mode)
    if nargin < 4, cluster_mode = 1; end
    X = zeros(N, 1); Y = zeros(N, 1);
    
    switch type
        case 'line'
            X = linspace(-area_size/2, area_size/2, N)';
            Y = X * 0.0; 
        case 'circle'
            % 优化: 制造微小缺口，打破完美对称引起的零方差
            theta = linspace(0, 2*pi * 0.98, N)';
            R = area_size * 0.35;
            X = R * cos(theta);
            Y = R * sin(theta);
        case 'spiral'
            % 优化: 等弧长 + 适中密度
            num_turns = 4; 
            max_theta = num_turns * 2 * pi;
            t = linspace(0, 1, N)';
            theta = max_theta * sqrt(t); % sqrt(t) 保证外圈不稀疏
            a = 0; 
            b = (area_size * 0.4) / max_theta; 
            r = a + b * theta;
            X = r .* cos(theta);
            Y = r .* sin(theta);
        case 'grid'
            % 优化: 完美正方 + 整数坐标
            side_num = round(sqrt(N));
            [x_grid, y_grid] = meshgrid(1:side_num, 1:side_num);
            scale = area_size / side_num;
            X = x_grid(:) * scale;
            Y = y_grid(:) * scale;
            % 截断多余点 (虽然推荐传入 N=side^2)
            if length(X) > N, X = X(1:N); Y = Y(1:N); end
        case 'random'
            X = (rand(N, 1) - 0.5) * area_size;
            Y = (rand(N, 1) - 0.5) * area_size;
        case 'cluster'
            % 聚类逻辑
            num_clusters = 5;
            points_per_cluster = floor(N / num_clusters);
            X = []; Y = [];
            
            % 确定中心
            if cluster_mode == 1 || cluster_mode == 3 % 宏观有序
                R_c = area_size * 0.3;
                ang = linspace(0, 2*pi, num_clusters+1);
                cx = R_c * cos(ang(1:end-1))';
                cy = R_c * sin(ang(1:end-1))';
            else % 宏观随机
                cx = (rand(num_clusters,1)-0.5)*area_size*0.8;
                cy = (rand(num_clusters,1)-0.5)*area_size*0.8;
            end
            
            % 生成内部
            for i=1:num_clusters
                cnt = points_per_cluster;
                if i==num_clusters, cnt = N - length(X); end
                
                if cluster_mode == 3 || cluster_mode == 4 % 微观有序
                    side = ceil(sqrt(cnt));
                    sp = area_size * 0.025;
                    [lx, ly] = meshgrid(linspace(-sp*side/2, sp*side/2, side));
                    loc_x = lx(1:cnt)' + cx(i);
                    loc_y = ly(1:cnt)' + cy(i);
                else % 微观随机
                    sig = area_size * 0.04;
                    loc_x = cx(i) + randn(cnt,1)*sig;
                    loc_y = cy(i) + randn(cnt,1)*sig;
                end
                X = [X; loc_x]; Y = [Y; loc_y];
            end
    end
end

% 2. 评估算法 (基础版)
function score = local_calculate_metric(X, Y, m, method)
    N = length(X);
    if N < m+1, score=0; return; end
    
    % 计算距离矩阵并排序
    D = pdist2([X Y], [X Y]);
    D_sort = sort(D, 1);
    
    % 取最近m个邻居 (排除自身)
    R_star = D_sort(2:m+1, :);
    
    % 构建特征矩阵 B
    switch method
        case 1 % 距离
            B = R_star;
        case 2 % 修正距离
            r_bar = mean(R_star, 2);
            B = R_star - r_bar;
        case 3 % 差值
            B = zeros(size(R_star));
            B(1,:) = R_star(1,:);
            B(2:end,:) = diff(R_star);
    end
    
    % 计算皮尔逊相关系数
    C = corr(B);
    
    % 处理 NaN (防止完美对称导致的零方差问题)
    C(isnan(C)) = 0; 
    
    score = mean(C(:));
end