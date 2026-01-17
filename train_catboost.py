import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ================= 配置 =================
# 1. 这里填写你在 MATLAB 第一步里生成的那个新 CSV 的文件名
DATA_FILE = 'catboost_training_data.csv' 

# 2. 这里的列名必须和你 MATLAB 生成的 CSV 表头完全一致！
# 请打开 CSV 确认一下列名
FEATURE_COLS = ['Method1', 'Method2', 'Method3'] 
TARGET_COL = 'Label'
# =======================================

def main():
    # 1. 读取 MATLAB 处理好的数据
    try:
        df = pd.read_csv(DATA_FILE)
        print(f"成功加载数据，共 {len(df)} 条样本")
    except FileNotFoundError:
        print(f"找不到文件 {DATA_FILE}，请先在 MATLAB 中运行 run_kaggle_validation.m 生成数据！")
        return

    # 2. 准备训练
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. CatBoost 训练
    print("开始训练模型...")
    model = CatBoostClassifier(
        iterations=1000, 
        learning_rate=0.05, 
        depth=6, 
        verbose=100
    )
    model.fit(X_train, y_train, eval_set=(X_test, y_test))

    # 4. 结果验证
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"\n最终准确率: {acc*100:.2f}%")
    
    # 5. 保存模型供 C++ 使用
    model.save_model("drone_swarm_model.cbm")
    print("模型已保存为 drone_swarm_model.cbm")

if __name__ == "__main__":
    main()