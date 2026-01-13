function [X, Y] = generate_swarm_data(type, N, area_size)
% GENERATE_SWARM_DATA 生成论文描述的四类空间结构
%
%
% 输入:
%   type:string 'line', 'circle', 'spiral', 'grid', 'random', 'cluster'
%   N:         无人机数量
%   area_size: 区域边长 (例如 1000)

    X = zeros(N, 1);
    Y = zeros(N, 1);
    
    switch type
        case 'line' % 线状结构 - 直线
            X = linspace(-area_size/2, area_size/2, N)';
            Y = X * 0.5 + 10; % y = kx + b
            
        case 'circle' % 线状结构 - 圆
            theta = linspace(0, 2*pi, N)';
            R = area_size * 0.3;
            X = R * cos(theta);
            Y = R * sin(theta);
            
       case 'spiral' % 线状结构 - 螺旋 (修正为等弧长采样)
            % 论文中的螺旋线点密度是均匀的，不能用 linspace 生成角度
            % 阿基米德螺旋线弧长近似公式：s proportional to theta^2
            % 因此，要使 s 均匀，theta 应该正比于 sqrt(t)
            
            % 1. 生成总圈数对应的最大角度
            max_theta = 4 * pi; 
            
            % 2. 使用平方根分布生成角度，以抵消半径增大带来的弧长增加
            % 这样生成的点在曲线上是近似等间距的
            t = linspace(0, 1, N)';
            theta = max_theta * sqrt(t); 
            
            % 3. 生成坐标
            a = 0; 
            b = area_size * 0.05; % 调整间距系数
            r = a + b * theta;
            
            X = r .* cos(theta);
            Y = r .* sin(theta);
            
        case 'grid' % 高组织 - 正方形网格
            side_num = ceil(sqrt(N));
            [x_grid, y_grid] = meshgrid(linspace(-area_size/2, area_size/2, side_num));
            X = x_grid(1:N)';
            Y = y_grid(1:N)';
            
        case 'random' % 随机结构
            % 均匀分布
            X = (rand(N, 1) - 0.5) * area_size;
            Y = (rand(N, 1) - 0.5) * area_size;
            
        case 'cluster' % 聚类结构
            % 设定 K 个簇中心
            num_clusters = 5;
            points_per_cluster = floor(N / num_clusters);
            
            % 随机生成簇中心
            centers_x = (rand(num_clusters, 1) - 0.5) * area_size * 0.8;
            centers_y = (rand(num_clusters, 1) - 0.5) * area_size * 0.8;
            
            idx = 1;
            for i = 1:num_clusters
                % 在每个簇中心周围生成高斯分布
                sigma = area_size * 0.05; % 簇的紧密度
                count = points_per_cluster;
                if i == num_clusters, count = N - idx + 1; end % 补齐剩余点
                
                X(idx:idx+count-1) = centers_x(i) + randn(count, 1) * sigma;
                Y(idx:idx+count-1) = centers_y(i) + randn(count, 1) * sigma;
                idx = idx + count;
            end
            
        otherwise
            error('Unknown structure type');
    end
end