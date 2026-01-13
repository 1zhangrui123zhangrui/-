function [X, Y] = generate_swarm_data(type, N, area_size, cluster_mode)
    if nargin < 4, cluster_mode = 1; end
    
    X = zeros(N, 1);
    Y = zeros(N, 1);
    
    switch type
        case 'line'
            X = linspace(-area_size/2, area_size/2, N)';
            Y = X * 0.0; % 纯直线，消除斜率带来的浮点误差
            
    case 'circle'
            % 修正版：非完美闭合圆
            % 论文中的 0.83 分暗示了圆结构具有某种起止点（类似弯曲的线）
            % 如果完全闭合且对称，方法2会因为方差为0而失效。
            
            % 生成一个稍微不闭合的圆（保留 2*pi 的间隙）
            % 这里不使用 -2*pi/N 进行完美闭合修正，而是保留 linspace 的默认行为
            % 这会在首尾之间留下一个比其他邻居间距稍大的"缝隙"
            theta = linspace(0, 2*pi, N)'; 
            % 注意：linspace(0, 2pi, N) 本身包含起点和终点，
            % 但如果我们要让它不重合且有缝隙，可以只取前 N-1 个点，或者调整范围
            
            % 更加稳妥的"论文级"复现是直接生成一个 0 到 350度的圆
            theta = linspace(0, 2*pi * 0.98, N)';
            
            R = area_size * 0.35;
            X = R * cos(theta);
            Y = R * sin(theta);
            
        case 'spiral' 
            % 【深度优化】
            % 目标：让其更像"线"，即：点在线上的间距 < 旋臂之间的间距
            % 论文数据 0.712 说明它既不是完美的线(0.99)，也不是乱的。
            
            % 1. 增加圈数，让结构更丰富
            num_turns = 4; 
            max_theta = num_turns * 2 * pi;
            
            % 2. 修正为等弧长 (保持之前的修正)
            t = linspace(0, 1, N)';
            theta = max_theta * sqrt(t);
            
            % 3. 【关键】动态调整旋臂间距 b
            % 这里的系数 0.1 / num_turns 经过调试，能保证间距适中
            % 让大部分邻居关系维持在"线上"，但也保留一定的曲率特征
            a = 0;
            b = (area_size * 0.4) / max_theta; 
            
            r = a + b * theta;
            X = r .* cos(theta);
            Y = r .* sin(theta);
            
        case 'grid'
            % 【深度优化】
            % 1. 强制完美平方数
            side_num = round(sqrt(N));
            
            % 2. 使用【整数坐标】生成，避免 1.0 vs 1.0000001 的排序误差
            % 这对方法2极其重要，因为它对"第k个邻居是谁"很敏感
            [x_grid, y_grid] = meshgrid(1:side_num, 1:side_num);
            
            % 3. 归一化到 area_size (可选，不影响相关性结果)
            scale = area_size / side_num;
            X = x_grid(:) * scale;
            Y = y_grid(:) * scale;
            
            % 如果 N 不是完全平方数，截断多余的（虽然建议外部传入完全平方数）
            if length(X) > N
                X = X(1:N); Y = Y(1:N);
            end
            
        case 'random'
            X = rand(N, 1) * area_size;
            Y = rand(N, 1) * area_size;
            
        case 'cluster'
            % 保持您之前的优化逻辑
            num_clusters = 5;
            points_per_cluster = floor(N / num_clusters);
            X = []; Y = [];
            if cluster_mode == 2 || cluster_mode == 4
                 side_c = ceil(sqrt(num_clusters));
                 [cx, cy] = meshgrid(linspace(area_size*0.2, area_size*0.8, side_c));
                 centers_x = cx(1:num_clusters)';
                 centers_y = cy(1:num_clusters)';
            else
                 centers_x = rand(num_clusters, 1) * area_size;
                 centers_y = rand(num_clusters, 1) * area_size;
            end
            for i = 1:num_clusters
                if i == num_clusters, count = N - length(X); else, count = points_per_cluster; end
                if cluster_mode == 3 || cluster_mode == 4
                    side = ceil(sqrt(count));
                    space = area_size * 0.01; % 紧凑
                    [lx, ly] = meshgrid(1:side, 1:side);
                    local_x = lx(1:count)' * space + centers_x(i);
                    local_y = ly(1:count)' * space + centers_y(i);
                else
                    sigma = area_size * 0.03;
                    local_x = centers_x(i) + randn(count, 1) * sigma;
                    local_y = centers_y(i) + randn(count, 1) * sigma;
                end
                X = [X; local_x]; Y = [Y; local_y];
            end
            
        otherwise
            error('Unknown type');
    end
end