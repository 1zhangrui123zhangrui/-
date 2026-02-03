function [orig_scores, imp_scores, Gamma] = Improved_Algorithm_Engine(X, Y, VX, VY, m_in)
    % 严格复刻：变体2(局部+标准化), 变体3(差分原样), 变体1(全局)
    N = length(X);
    
    % 0. 坐标微扰 (解决 NaN，不再强制转0)
    X = X + randn(size(X)) * 1e-10;
    Y = Y + randn(size(Y)) * 1e-10;
    
    P = [X, Y]; V = [VX, VY];
    
    % 1. 运动一致性增益 Gamma
    v_mag = sqrt(VX.^2 + VY.^2) + 1e-9;
    V_unit = V ./ v_mag;
    S_v = V_unit * V_unit'; 
    Gamma = (sum(S_v(:)) - N) / (N * (N - 1));
    Gamma = max(0, min(1, Gamma)); 
    
    % 2. 原论文 3 种变体计算
    orig_scores = zeros(1, 3);
    D = pdist2(P, P);
    D_sort = sort(D, 1);
    
    for method_type = 1:3
        % 变体 2 采用局部化邻域 m=3
        m = (method_type == 2) * 3 + (method_type ~= 2) * m_in;
        R_star = D_sort(2:m+1, :); % 提取邻居距离
        
        switch method_type
            case 1 % 变体1: 距离 (需要行去趋势)
                B = R_star - mean(R_star, 1); % 减去该点自己的邻居均值
                
            case 2 % 变体2: 距离倒数 (最关键：局部+去均值)
                B_raw = 1 ./ (R_star + 1e-6);
                % 消除 1/r 的非线性增长趋势，使随机场景的线性相关性降低
                B = B_raw - mean(B_raw, 1); 
                
            case 3 % 变体3: 距离差 (差分本身就是去趋势，不再二次修改)
                B = diff(D_sort(1:m+1, :), 1, 1);
        end
        
        % 3. 计算 Pearson 自相关
        if size(B, 1) > 1
            C = corr(B);
            rho = (sum(C(:)) - N) / (N * (N - 1));
        else
            rho = 0;
        end
        orig_scores(method_type) = rho;
    end
    
    imp_scores = orig_scores * Gamma;
end