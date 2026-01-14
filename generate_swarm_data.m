function [X, Y] = generate_swarm_data(type, N, area_size, cluster_mode)
% GENERATE_SWARM_DATA [最终去噪版]
% 核心修正：对于"微观有序"结构，移除所有随机抖动(Jitter)，生成数学上完美的晶格

    if nargin < 4, cluster_mode = 1; end
    
    X = zeros(N, 1); Y = zeros(N, 1);
    
    switch type
        case 'line' 
            X = linspace(-area_size/2, area_size/2, N)'; Y = X * 0.0; 
        case 'circle'
            theta = linspace(0, 2*pi * 0.98, N)'; R = area_size * 0.35;
            X = R * cos(theta); Y = R * sin(theta);
        case 'spiral'
            t = linspace(0, 1, N)'; theta = 4 * 2 * pi * sqrt(t);
            r = ((area_size * 0.4) / (8*pi)) * theta;
            X = r .* cos(theta); Y = r .* sin(theta);
        case 'grid'
            side_num = round(sqrt(N));
            [x_grid, y_grid] = meshgrid(1:side_num, 1:side_num);
            scale = area_size / side_num;
            X = x_grid(:) * scale; Y = y_grid(:) * scale;
            if length(X) > N, X = X(1:N); Y = Y(1:N); end
        case 'random' 
            X = (rand(N, 1) - 0.5) * area_size; Y = (rand(N, 1) - 0.5) * area_size;
            
        case 'cluster'
            num_clusters = 5; 
            points_per_cluster = floor(N / num_clusters);
            
            % 1. 簇中心
            if cluster_mode == 1 || cluster_mode == 3 
                R_c = area_size * 0.35; ang = linspace(0, 2*pi, num_clusters+1);
                cx = R_c * cos(ang(1:end-1))'; cy = R_c * sin(ang(1:end-1))';
            else 
                cx = (rand(num_clusters,1)-0.5)*area_size*0.8; cy = (rand(num_clusters,1)-0.5)*area_size*0.8;
            end
            
            % 2. 簇内部
            idx = 1;
            for i = 1:num_clusters
                if i == num_clusters, cnt = N - (i-1)*points_per_cluster; else, cnt = points_per_cluster; end
                
                if cluster_mode == 3 || cluster_mode == 4
                    % === [关键修正] 微观有序：生成完美晶格，不加任何噪声 ===
                    side = ceil(sqrt(cnt));
                    base_spacing = area_size * 0.03; 
                    
                    % 六边形网格逻辑 (Hexagonal) - 保持结构丰富性
                    [hx, hy] = meshgrid(1:side, 1:side);
                    hx = hx * base_spacing;
                    hy = hy * (base_spacing * sqrt(3)/2);
                    hx(2:2:end, :) = hx(2:2:end, :) + base_spacing/2;
                    lx = hx(:); ly = hy(:);
                    if length(lx) > cnt, lx = lx(1:cnt); ly = ly(1:cnt); end
                    
                    % 中心化
                    lx = lx - mean(lx); ly = ly - mean(ly);
                    
                    % 仅保留旋转 (Rotation)，这不会破坏距离的一致性
                    rand_angle = rand() * 2 * pi;
                    R_rot = [cos(rand_angle), -sin(rand_angle); sin(rand_angle), cos(rand_angle)];
                    coords = R_rot * [lx'; ly'];
                    local_x = coords(1, :)'; local_y = coords(2, :)';
                    
                    % 【绝对不要加 Jitter/randn！】
                    % 之前这里加了 randn，导致偏差变成了噪声，相关性变成0
                else
                    % 微观随机：保留噪声
                    sigma = area_size * 0.03; 
                    local_x = randn(cnt, 1) * sigma; local_y = randn(cnt, 1) * sigma;
                end
                
                X(idx : idx+cnt-1) = local_x + cx(i);
                Y(idx : idx+cnt-1) = local_y + cy(i);
                idx = idx + cnt;
            end
    end
end