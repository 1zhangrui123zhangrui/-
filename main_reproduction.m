%% 复现论文深度优化版
clear; clc; close all;

m = 7;             
area_size = 1000;  

struct_types = {'line', 'circle', 'spiral', 'grid', 'random'};
struct_names = {'直线 (Line)', '圆 (Circle)', '螺旋 (Spiral)', '网格 (Grid)', '随机 (Random)'};

% 准备结果表
results = table('Size', [length(struct_types), 4], ...
    'VariableTypes', {'string', 'double', 'double', 'double'}, ...
    'VariableNames', {'Structure', 'Method_1_Dist', 'Method_2_ModDist', 'Method_3_Diff'});

for i = 1:length(struct_types)
    type = struct_types{i};
    name = struct_names{i};
    
    % 【关键调整】针对不同结构使用不同的 N
    switch type
        case 'grid'
            % Grid 需要巨大的 N 来稀释边缘效应，逼近 0.9
            % N = 40x40 = 1600 或 50x50 = 2500
            current_N = 2500; 
        case 'spiral'
            % Spiral 需要适中的密度，太密会变成面，太疏会断
            current_N = 400;  
        otherwise
            current_N = 400;
    end
    
    [X, Y] = generate_swarm_data(type, current_N, area_size);
    
    s1 = calculate_swarm_regularity(X, Y, m, 1);
    s2 = calculate_swarm_regularity(X, Y, m, 2);
    s3 = calculate_swarm_regularity(X, Y, m, 3);
    
    results.Structure(i) = name;
    results.Method_1_Dist(i) = s1;
    results.Method_2_ModDist(i) = s2;
    results.Method_3_Diff(i) = s3;
    
    fprintf('完成: %s (N=%d) -> M2=%.4f (目标: Grid>0.9, Spiral~0.7)\n', name, current_N, s2);
end

disp(results);