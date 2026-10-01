import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from src.utils import set_seed, setup_logger

def generate_anomaly_test_data(T=2000, N=5, seed=42):
    np.random.seed(seed)
    data = np.random.randn(T, N) * 0.5 + 10.0
    anomaly_window = range(1200, 1250)
    data[anomaly_window, 1] += 5.0  
    
    labels = np.zeros(T, dtype=int)
    labels[anomaly_window] = 1
    return data, labels

def evaluate_anomaly_detection(true_labels, pred_labels):
    tp = np.sum((true_labels == 1) & (pred_labels == 1))
    fp = np.sum((true_labels == 0) & (pred_labels == 1))
    fn = np.sum((true_labels == 1) & (pred_labels == 0))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {"Precision": round(precision, 4), "Recall": round(recall, 4), "F1": round(f1, 4)}

def main():
    logger = setup_logger("anomaly_tuned")
    set_seed(42)
    
    logger.info("============== 模块三：Isolation Forest 调参寻优实验 ==============")
    
    data, true_labels = generate_anomaly_test_data()
    
    train_end = 1000
    train_data = data[:train_end]
    test_data = data[train_end:]
    test_labels = true_labels[train_end:]
    
    # 防泄漏标准化
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_data)
    test_scaled = scaler.transform(test_data)
    
    # 模型训练 (使用适中的 contamination=0.03)
    iso_forest = IsolationForest(contamination=0.03, random_state=42)
    iso_forest.fit(train_scaled)
    
    test_scores = iso_forest.decision_function(test_scaled)
    train_scores = iso_forest.decision_function(train_scaled)
    
    # 【调参点 1】放宽分位数到下 3%
    dynamic_threshold = np.percentile(train_scores, 3.0) 
    pred_labels = np.where(test_scores < dynamic_threshold, 1, 0)
    
    # 【调参点 2】缩短平滑窗口到 3 步，降低对瞬时异常的滞后性
    df_preds = pd.Series(pred_labels)
    smoothed_preds = df_preds.rolling(window=3, min_periods=1).mean()
    final_pred_labels = np.where(smoothed_preds >= 0.33, 1, 0)
    
    metrics = evaluate_anomaly_detection(test_labels, final_pred_labels)
    logger.info(f"调参后异常检测表现: Precision={metrics['Precision']} | Recall={metrics['Recall']} | F1={metrics['F1']}")
    logger.info("============== 调参实验执行完毕 ==============")

if __name__ == "__main__":
    main()