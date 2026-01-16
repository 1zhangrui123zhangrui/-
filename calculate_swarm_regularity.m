function [autocorr_score, B] = calculate_swarm_regularity(X, Y, m, method_type)
% CALCULATE_SWARM_REGULARITY [标准化+精度修复版]
% 
% 改进点：
% 1. 输入坐标 Z-score 标准化：消除数据尺寸(Scaling)对相关性计算的潜在干扰。
% 2. 完美结构 NaN -> 1.0 逻辑保留。

    N = length(X);
    if N < m + 1, autocorr_score = 0; B = []; return; end
    
    % === [新增] 1. 数据标准化 (Z-score Normalization) ===
    % 这一步对于 Kaggle 真实数据集至关重要，因为不同样本的坐标范围差异巨大
    if std(X) > 1e-6, X = (X - mean(X)) / std(X); end
    if std(Y) > 1e-6, Y = (Y - mean(Y)) / std(Y); end
    
    % 2. 距离计算
    dist_mat = pdist2([X, Y], [X, Y]);
    sorted_dist = sort(dist_mat, 1, 'ascend');
    
    % 动态容错：确保 m 不越界
    real_m = min(m, N-1);
    R_star = sorted_dist(2:real_m+1, :); 
    
    % 3. 构建特征矩阵
    switch method_type
        case 1, B = R_star;
        case 2, r_bar = mean(R_star, 2); B = R_star - r_bar;
        case 3, B = zeros(size(R_star)); B(1,:) = R_star(1,:); B(2:end,:) = diff(R_star);
    end
    
    % 4. 数值清洗
    B(abs(B) < 1e-9) = 0;
    
    % 5. 计算相关系数
    corr_mat = corr(B, 'Type', 'Pearson');
    
    % === NaN 处理逻辑 ===
    if method_type == 2 || method_type == 3
        % 完美结构 (方差为0) -> 相关性为 1
        corr_mat(isnan(corr_mat)) = 1; 
    else
        corr_mat(isnan(corr_mat)) = 0;
    end
    
    autocorr_score = mean(corr_mat(:));
end