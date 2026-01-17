function [X, Y] = generate_swarm_data(type, N, area_size, cluster_mode)
% GENERATE_SWARM_DATA [旋转增强修复版]
% 修正：场景 9 使用随机旋转降低 M3 评分 [cite: 1]

    if nargin < 4, cluster_mode = 1; end
    X = zeros(N, 1); Y = zeros(N, 1);
    
    switch type
        case 'line', X = linspace(-area_size/2, area_size/2, N)'; Y = X * 0.0;
        case 'circle', theta = linspace(0, 2*pi*0.98, N)'; R = area_size*0.35; X = R*cos(theta); Y = R*sin(theta);
        case 'spiral', t = linspace(0, 1, N)'; theta = 4*2*pi*sqrt(t); r = (area_size*0.4/(8*pi))*theta; X = r.*cos(theta); Y = r.*sin(theta);
        case 'grid', side = round(sqrt(N)); [hx, hy] = meshgrid(1:side); s = area_size/side; X = hx(:)*s; Y = hy(:)*s; if length(X)>N, X=X(1:N); Y=Y(1:N); end
        case 'random', X = (rand(N,1)-0.5)*area_size; Y = (rand(N,1)-0.5)*area_size;
        
        case 'cluster'
            num_clusters = 5; points_per_cluster = floor(N/num_clusters);
            cx = []; cy = [];
            if cluster_mode == 1 || cluster_mode == 3 % 有序中心
                R_c = area_size * 0.35; ang = linspace(0, 2*pi, num_clusters+1);
                cx = R_c * cos(ang(1:end-1))'; cy = R_c * sin(ang(1:end-1))';
            else % 随机中心
                cx = (rand(num_clusters,1)-0.5)*area_size*0.8; cy = (rand(num_clusters,1)-0.5)*area_size*0.8;
            end
            
            idx = 1;
            for i = 1:num_clusters
                cnt = (i==num_clusters)*(N-(i-1)*points_per_cluster) + (i~=num_clusters)*points_per_cluster;
                if cluster_mode == 3 || cluster_mode == 4
                    side = ceil(sqrt(cnt)); s = area_size * 0.012;
                    [hx, hy] = meshgrid(1:side, 1:side);
                    hx = hx*s; hy = hy*(s*sqrt(3)/2); hx(2:2:end,:) = hx(2:2:end,:) + s/2;
                    lx = hx(:); ly = hy(:); 
                    lx = lx(1:cnt)-mean(lx(1:cnt)); ly = ly(1:cnt)-mean(ly(1:cnt));
                    
                    % 关键修复：场景 9 每个簇独立随机旋转，降低 M3 全局相关性 [cite: 1]
                    rot = (cluster_mode == 4) * rand() * 2 * pi; 
                    R_rot = [cos(rot) -sin(rot); sin(rot) cos(rot)];
                    pos = R_rot * [lx'; ly'];
                    X(idx:idx+cnt-1) = pos(1,:)' + cx(i);
                    Y(idx:idx+cnt-1) = pos(2,:)' + cy(i);
                else
                    sig = area_size * 0.025;
                    X(idx:idx+cnt-1) = randn(cnt,1)*sig + cx(i); Y(idx:idx+cnt-1) = randn(cnt,1)*sig + cy(i);
                end
                idx = idx + cnt;
            end
    end
end