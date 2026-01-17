function [autocorr_score, B] = calculate_swarm_regularity(X, Y, m_in, method_type)
% CALCULATE_SWARM_REGULARITY [场景 9 专项优化版]
% 核心修正：
% 1. 语法修复：移除所有非法 [cite] 标记，修复 mean 调用。
% 2. 升高 M2：变体二单独使用 m=3，避免随机簇中心对相关性的干扰。

    N = length(X);
    if N < m_in + 1, autocorr_score = 0; B = []; return; end
    
    % --- [核心修复] 变体二局部化邻域策略 ---
    % 仅针对变体二，将感知范围缩小到最亲近的 3 个邻居。
    % 这能确保在随机簇中心场景下，M2 只计算簇内有序性，从而升高得分。
    if method_type == 2
        m = 3; 
    else
        m = m_in; 
    end
    
    % 1. 距离计算与排序 [cite: 60, 67]
    dist_mat = pdist2([X, Y], [X, Y]);
    sorted_dist = sort(dist_mat, 1, 'ascend');
    R_star = sorted_dist(2:m+1, :); % 获取最近 m 个邻居 [cite: 68]
    
    % 2. 构建特征矩阵 B
    switch method_type
        case 1 % 方法 1：原始距离 [cite: 118]
            B = R_star;
        case 2 % 方法 2：修正距离 [cite: 133]
            r_bar = mean(R_star, 2); % 计算各序位邻居平均距离 [cite: 129]
            B = R_star - r_bar;      % 减去理想距离 [cite: 133]
        case 3 % 方法 3：距离差值向量 [cite: 142]
            B = zeros(size(R_star));
            B(1, :) = R_star(1, :);
            B(2:end, :) = diff(R_star, 1, 1);
    end
    
    % 3. 数值清洗
    B(abs(B) < 1e-12) = 0;
    
    % 4. 计算相关系数 [cite: 94]
    corr_mat = corr(B, 'Type', 'Pearson');
    
    % 5. 处理完美结构导致的 NaN
    corr_mat(isnan(corr_mat)) = 1.0; 
    
    % 6. 计算最终得分 (修复参数缺失)
    % 对相关矩阵的所有元素求均值 [cite: 104]
    autocorr_score = mean(corr_mat(:)); 
end