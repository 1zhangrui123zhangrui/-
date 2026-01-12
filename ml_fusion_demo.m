%% 第四阶段：机器学习融合 (对应论文第5节)
% 注意：此代码需要加载实际的 Swarm Behavior CSV 数据集才能运行
% 这里展示特征提取和模型训练的逻辑结构

function ml_fusion_demo()
    % 假设已加载数据，格式为: [SampleID, X, Y, IsSwarm]
    % 这里模拟生成一些训练数据来演示流程
    num_samples = 200;
    features = zeros(num_samples, 3); % [Score1, Score2, Score3]
    labels = zeros(num_samples, 1);   % 0 或 1
    
    disp('正在生成模拟训练数据并提取特征...');
    
    for i = 1:num_samples
        % 模拟：随机生成一半是网格(Swarm)，一半是随机(Non-Swarm)
        if rand > 0.5
            [X, Y] = generate_swarm_data('grid', 100, 1000);
            labels(i) = 1;
        else
            [X, Y] = generate_swarm_data('random', 100, 1000);
            labels(i) = 0;
        end
        
        % 特征工程：提取三个方法的得分
        m = 7;
        f1 = calculate_swarm_regularity(X, Y, m, 1);
        f2 = calculate_swarm_regularity(X, Y, m, 2);
        f3 = calculate_swarm_regularity(X, Y, m, 3);
        
        features(i, :) = [f1, f2, f3];
    end
    
    % 模型训练
    % 论文使用 CatBoost，MATLAB 中可用 fitcensemble (Bag/Boosting) 近似
    disp('正在训练集成学习模型...');
    
    % 使用 TreeBagger (随机森林) 或 LSBoost
    t = templateTree('MaxNumSplits', 6); % 树深约等于6
    model = fitcensemble(features, labels, ...
        'Method', 'Bag', ... % 或 'LSBoost'
        'NumLearningCycles', 200, ...
        'Learners', t);
        
    % 验证
    cv_model = crossval(model, 'KFold', 5);
    acc = 1 - kfoldLoss(cv_model);
    
    disp(['模型交叉验证准确率: ', num2str(acc * 100), '%']);
    
    % 查看特征重要性 (如果算法支持)
    % plot(predictorImportance(model));
end