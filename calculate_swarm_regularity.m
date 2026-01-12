function [autocorr_score, B] = calculate_swarm_regularity(X, Y, m, method_type)
% CALCULATE_SWARM_REGULARITY 计算无人机群空间结构的自相关值
% 复现论文《基于自相关分析的无人机群空间结构规律性评估方法及系统》的核心算法
%
% 输入:
%   X, Y:        N x 1 的坐标向量
%   m:           邻居数量 (论文建议 [6, 10])
%   method_type: 1 (原始距离), 2 (修正距离), 3 (距离差值)
%
% 输出:
%   autocorr_score: 最终的标量自相关评分 (0-1之间)
%   B:              相对位置矩阵 (调试用)

    N = length(X);
    
    % 1. 计算距离矩阵
    % 使用欧几里得距离
    dist_mat = pdist2([X, Y], [X, Y]);
    
    % 2. 排序与截取
    % 对每一列进行升序排序
    sorted_dist = sort(dist_mat, 1, 'ascend');
    
    % 截取第 2 到 m+1 个元素 (排除自身距离0，取最近的 m 个邻居)
    % R_star 维度: m x N
    if m >= N
        error('邻居数量 m 必须小于无人机总数 N');
    end
    R_star = sorted_dist(2:m+1, :);
    
    % 3. 构建相对位置矩阵 B (MOP)
    B = zeros(size(R_star));
    
    switch method_type
        case 1 % 方法 1：原始距离法
            % 直接使用排序后的矩阵
            B = R_star;
            
        case 2 % 方法 2：修正距离法
            % 步骤 A: 计算每一行的平均值 (即第 j 个邻居的平均距离)
            r_bar = mean(R_star, 2); 
            
            % 步骤 B: 每个元素减去对应的行均值
            % 利用广播机制或循环减去
            B = R_star - r_bar;
            
        case 3 % 方法 3：距离差值法
            % 第一行保持不变
            B(1, :) = R_star(1, :);
            % 后续行计算差分 (增量)
            % B(j,k) = R*(j,k) - R*(j-1,k)
            B(2:end, :) = diff(R_star);
            
        otherwise
            error('Method type must be 1, 2, or 3');
    end
    
    % 4. 计算自相关值
    % 计算 B 的列向量之间的皮尔逊相关系数
    % corr_mat 维度: N x N
    corr_mat = corr(B, 'Type', 'Pearson');
    
    % 处理可能的 NaN (当向量标准差为0时发生，虽然在距离中很少见)
    corr_mat(isnan(corr_mat)) = 0; 
    
    % 计算矩阵所有元素的平均值 (包含对角线的1)
    % 公式: (1/N^2) * sum(sum(corr_k_l))
    autocorr_score = mean(corr_mat(:));
end