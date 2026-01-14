function [X, Y] = generate_swarm_data(type, N, area_size, cluster_mode)
% GENERATE_SWARM_DATA 生成符合物理规律的无人机群空间结构
% 
% 修改记录：2024-01-14
% 修复了"模具克隆"过于完美导致方法2得分异常低的问题
% 现在微观有序的簇结构相似但不完全相同，避免方差为零的病态情况
%
% 输入参数:
%   type - 结构类型: 'line', 'circle', 'spiral', 'grid', 'random', 'cluster'
%   N - 总点数
%   area_size - 区域大小
%   cluster_mode - 聚类模式 (仅当type='cluster'时有效):
%                 1: 宏观有序 + 微观随机
%                 2: 宏观随机 + 微观随机
%                 3: 宏观有序 + 微观有序
%                 4: 宏观随机 + 微观有序
%
% 输出:
%   X, Y - 坐标向量 (N×1)

    if nargin < 4, cluster_mode = 1; end
    
    X = zeros(N, 1);
    Y = zeros(N, 1);
    
    switch type
        case 'line' 
            % 直线结构
            X = linspace(-area_size/2, area_size/2, N)';
            Y = X * 0.0; 
            
        case 'circle'
            % 圆形结构 (带0.98缺口避免完美对称)
            theta = linspace(0, 2*pi * 0.98, N)';
            R = area_size * 0.35;
            X = R * cos(theta);
            Y = R * sin(theta);
            
        case 'spiral'
            % 阿基米德螺旋线
            num_turns = 4; 
            max_theta = num_turns * 2 * pi;
            t = linspace(0, 1, N)';
            theta = max_theta * sqrt(t);  % 使用sqrt保证外圈密度适中
            a = 0; 
            b = (area_size * 0.4) / max_theta; 
            r = a + b * theta;
            X = r .* cos(theta);
            Y = r .* sin(theta);
            
        case 'grid'
            % 正方形网格
            side_num = round(sqrt(N));
            [x_grid, y_grid] = meshgrid(1:side_num, 1:side_num);
            scale = area_size / side_num;
            X = x_grid(:) * scale;
            Y = y_grid(:) * scale;
            if length(X) > N
                X = X(1:N); 
                Y = Y(1:N); 
            end

        case 'random' 
            % 均匀随机分布
            X = (rand(N, 1) - 0.5) * area_size;
            Y = (rand(N, 1) - 0.5) * area_size;
            
        case 'cluster'
            % =============== 聚类结构生成 ===============
            % 关键修改: 微观有序的簇不再完美复制，而是相似但有差异
            num_clusters = 5; 
            
            % 每个簇的点数 (为保持结构可比性，尽量均匀分配)
            base_points_per_cluster = floor(N / num_clusters);
            remainder = mod(N, num_clusters);
            
            % 创建数组记录每个簇的实际点数
            cluster_sizes = ones(num_clusters, 1) * base_points_per_cluster;
            cluster_sizes(1:remainder) = cluster_sizes(1:remainder) + 1;
            
            X = [];
            Y = [];
            
            % A. 生成簇中心坐标
            if cluster_mode == 1 || cluster_mode == 3
                % [宏观有序]: 中心点呈正五边形排列
                R_center = area_size * 0.3; 
                angles = linspace(0, 2*pi, num_clusters+1);
                centers_x = R_center * cos(angles(1:end-1))';
                centers_y = R_center * sin(angles(1:end-1))';
            else
                % [宏观随机]: 中心点随机分布
                centers_x = (rand(num_clusters, 1) - 0.5) * area_size * 0.6;
                centers_y = (rand(num_clusters, 1) - 0.5) * area_size * 0.6;
            end
            
            % B. 生成每个簇的内部点
            for i = 1:num_clusters
                cnt = cluster_sizes(i);
                
                if cluster_mode == 3 || cluster_mode == 4
                    % =============== 微观有序 (有组织) ===============
                    % 关键: 为每个簇生成相似但不完全相同的内部结构
                    % 避免完美复制导致方差为零的病态情况
                    
                    % 基础间距参数
                    base_spacing = area_size * 0.03;
                    
                    % 1. 生成基础网格 (类似但不完全相同)
                    % 每个簇的网格大小略有差异
                    grid_density = 0.8 + 0.2 * rand(); % 0.8-1.0之间随机
                    effective_spacing = base_spacing * grid_density;
                    
                    % 计算网格边长
                    side_len = ceil(sqrt(cnt * 1.2)); % 稍大一些，然后裁剪
                    
                    % 创建基础网格 (六边形交错排列 - 最密堆积)
                    [gx, gy] = meshgrid(1:side_len, 1:side_len);
                    gx = gx * effective_spacing;
                    gy = gy * (effective_spacing * sqrt(3)/2);
                    
                    % 交错行偏移 (形成六边形网格)
                    gx(2:2:end, :) = gx(2:2:end, :) + effective_spacing/2;
                    
                    % 展平并中心化
                    local_points = [gx(:), gy(:)];
                    local_points(:,1) = local_points(:,1) - mean(local_points(:,1));
                    local_points(:,2) = local_points(:,2) - mean(local_points(:,2));
                    
                    % 2. 为每个簇添加独特的轻微旋转
                    % 论文中的"高度有组织"不是完美复制，而是相似
                    rotate_angle = (i-1) * (pi/20) + (rand()-0.5)*0.1; % 基础旋转 + 随机扰动
                    R = [cos(rotate_angle), -sin(rotate_angle); 
                         sin(rotate_angle), cos(rotate_angle)];
                    local_points = (R * local_points')';
                    
                    % 3. 按距离中心远近排序，选取最近的cnt个点
                    dist_to_center = sqrt(local_points(:,1).^2 + local_points(:,2).^2);
                    [~, sort_idx] = sort(dist_to_center);
                    selected_idx = sort_idx(1:min(cnt, length(local_points)));
                    
                    local_x = local_points(selected_idx, 1);
                    local_y = local_points(selected_idx, 2);
                    
                    % 4. 添加微小但明显的物理扰动
                    % 比完美复制时的噪声更大，确保有足够的方差
                    position_noise = effective_spacing * 0.08; % 8%的间距作为噪声
                    local_x = local_x + randn(size(local_x)) * position_noise;
                    local_y = local_y + randn(size(local_y)) * position_noise;
                    
                else
                    % =============== 微观随机 (无组织) ===============
                    % 高斯分布簇
                    sigma = area_size * 0.04; 
                    local_x = centers_x(i) + randn(cnt, 1) * sigma;
                    local_y = centers_y(i) + randn(cnt, 1) * sigma;
                end
                
                % 平移到簇中心
                local_x = local_x + centers_x(i);
                local_y = local_y + centers_y(i);
                
                X = [X; local_x];
                Y = [Y; local_y];
            end
            
            % 确保总点数正确
            if length(X) > N
                X = X(1:N);
                Y = Y(1:N);
            end
            
        otherwise
            error('未知结构类型: %s', type);
    end
    
    % 最终验证：确保没有NaN值
    if any(isnan(X)) || any(isnan(Y))
        error('生成的数据包含NaN值');
    end
end