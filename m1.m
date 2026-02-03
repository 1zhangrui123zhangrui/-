function score = calc_M1(P, m)
    N = size(P, 1);
    D = pdist2(P, P);
    [sorted_D, ~] = sort(D, 1);
    R_star = sorted_D(2:m+1, :); % 原论文变体一：邻居距离矩阵
    C = corr(R_star);
    score = (sum(C(:)) - N) / (N*(N-1));
end