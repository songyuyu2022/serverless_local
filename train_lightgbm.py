# lightGBM 离线训练脚本
import lightgbm as lgb
import pandas as pd
import numpy as np

# ===== 1. 读取你的训练数据（来自 metrics.csv）=====
df = pd.read_csv("metrics.csv")

# 只保留训练阶段的数据
df = df[df["phase"] == "train"]
df = df.replace([np.inf, -np.inf], np.nan).dropna()

# ===== 2. 选择特征 X 和目标 y =====
# 典型用于通信预测的特征：
feature_cols = [
    "grad_bytes",
    "hot_ratio",
    "cold_skip_ratio",
    "dispatch_count",
    "mode_hot_frac",
    "mode_cold_frac",
    "mode_http_frac",
    "inst_entropy",
    "tokens",
]

# 目标：专家通信时间（也可以改成 step_time_ms 等）
target_col = "expert_comm_ms"

X = df[feature_cols]
y = df[target_col]

# ===== 3. LightGBM Dataset =====
train_data = lgb.Dataset(X, label=y)

# ===== 4. LightGBM 参数 =====
params = {
    "objective": "regression",
    "metric": "l2",
    "num_leaves": 64,
    "learning_rate": 0.05,
    "num_iterations": 300,
}

def main():
    print("Training LightGBM...")
    model = lgb.train(params, train_data)
    # ===== 5. 保存模型到文件 =====
    model_path = "lgb_instance_selector.txt"
    model.save_model(model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()
