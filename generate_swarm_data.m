function [X, Y] = generate_swarm_data(type, N, area_size, cluster_mode)
% GENERATE_SWARM_DATA 生成器 - 包含高级聚类模式
% cluster_mode 参数定义 (仅当 type='cluster' 时有效):
%   1: 宏观有序 + 微观随机 (Org Center, Rnd Inner)
%   2: 宏观随机 + 微观随机 (Rnd Center, Rnd Inner)
%   3: 宏观有序 + 微观有序 (Org Center, Org Inner)
%   4: 宏观随机 + 微观有序 (Rnd Center, Org Inner)

    if nargin < 4, cluster_mode = 1; end
    
    X = zeros(N, 1);
    Y = zeros(N, 1);
    
    switch type
        case 'line'
            X = linspace(-area_size/2, area_size/2, N)';
            Y = X * 0.0; 
        case 'circle'
            % 制造微小缺口的圆，避免零方差
            theta = linspace(0, 2*pi * 0.98, N)';
            R = area_size * 0.35;
            X = R * cos(theta);
            Y = R * sin(theta);
        case 'spiral'
            num_turns = 4; 
            max_theta = num_turns * 2 * pi;
            t = linspace(0, 1, N)';
            theta = max_theta * sqrt(t);
            a = 0; b = (area_size * 0.4) / max_theta; 
            r = a + b * theta;
            X = r .* cos(theta);
            Y = r .* sin(theta);
        case 'grid'
            side_num = round(sqrt(N));
            [x_grid, y_grid] = meshgrid(1:side_num, 1:side_num);
            scale = area_size / side_num;
            X = x_grid(:) * scale;
            Y = y_grid(:) * scale;
            if length(X) > N, X = X(1:N); Y = Y(1:N); end
            
            case 'cluster' % 聚类结构 (深度优化版)
            % 参数定义
            % cluster_mode:
            % 1: 宏观有序 + 微观随机
            % 2: 宏观随机 + 微观随机
            % 3: 宏观有序 + 微观有序
            % 4: 宏观随机 + 微观有序
            
            num_clusters = 5; 
            points_per_cluster = floor(N / num_clusters);
            X = []; Y = [];
            
            % --- A. 生成簇中心 (Centers) ---
            if cluster_mode == 1 || cluster_mode == 3
                % [宏观有序]: 五点均匀分布 (正五边形)
                % 论文图5显示的是比较开阔的分布
                R_center = area_size * 0.35; 
                angles = linspace(0, 2*pi, num_clusters+1);
                centers_x = R_center * cos(angles(1:end-1))';
                centers_y = R_center * sin(angles(1:end-1))';
            else
                % [宏观随机]: 随机撒点
                % 限制在区域内部，防止太靠边
                centers_x = (rand(num_clusters, 1) - 0.5) * area_size * 0.7;
                centers_y = (rand(num_clusters, 1) - 0.5) * area_size * 0.7;
            end
            
            % --- B. 生成簇内点 (Inner Points) ---
            for i = 1:num_clusters
                % 处理最后一个簇可能多几个点的情况
                if i == num_clusters
                    count = N - length(X); 
                else
                    count = points_per_cluster; 
                end
                
                if cluster_mode == 3 || cluster_mode == 4
                    % [微观有序]: 局部微型网格
                    % 这会让 M2 (修正距离) 分数很高 (>0.8)
                    side = ceil(sqrt(count));
                    
                    % 间距要小，体现"簇"的紧凑性
                    space = area_size * 0.025; 
                    
                    % 生成局部网格
                    [lx, ly] = meshgrid(linspace(-space*side/2, space*side/2, side));
                    
                    % 截取需要的点数
                    lx = lx(1:count)';
                    ly = ly(1:count)';
                    
                    % 加上中心偏移
                    local_x = lx + centers_x(i);
                    local_y = ly + centers_y(i);
                else
                    % [微观随机]: 高斯分布 (正态分布)
                    % 这会让 M2 分数处于中等水平 (~0.5)
                    % sigma 不能太大(这就散了)，也不能太小(就缩成一点了)
                    sigma = area_size * 0.03; 
                    
                    local_x = centers_x(i) + randn(count, 1) * sigma;
                    local_y = centers_y(i) + randn(count, 1) * sigma;
                end
                
                X = [X; local_x];
                Y = [Y; local_y];
            end
            
        case 'random'
            X = (rand(N, 1) - 0.5) * area_size;
            Y = (rand(N, 1) - 0.5) * area_size;
            
        otherwise
            error('Unknown type');
    end
end