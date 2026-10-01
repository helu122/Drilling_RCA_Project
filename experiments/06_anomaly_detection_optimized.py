import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from src.utils import set_seed, setup_logger

def generate_anomaly_test_data(T=2000, N=5, seed=42):
    """生成带有突发异常事件的模拟钻井时序数据用于基线测试"""
    np.random.seed(seed)
    data = np.random.randn(T, N) * 0.5 + 10.0
    
    # 注入人工已知异常事件 (1200~1250 步)
    anomaly_window = range(1200, 1250)
    data[anomaly_window, 1] += 5.0  
    
    labels = np.zeros(T, dtype=int)
    labels[anomaly_window] = 1
    return data, labels

def evaluate_anomaly_detection(true_labels, pred_labels):
    """计算事件级的异常检测指标"""
    tp = np.sum((true_labels == 1) & (pred_labels == 1))
    fp = np.sum((true_labels == 0) & (pred_labels == 1))
    fn = np.sum((true_labels == 1) & (pred_labels == 0))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {"Precision": round(precision, 4), "Recall": round(recall, 4), "F1": round(f1, 4)}

def main():
    logger = setup_logger("anomaly_optimized")
    set_seed(42)
    
    logger.info("============== 模块三：异常检测阈值调优与平滑实验 ==============")
    
    data, true_labels = generate_anomaly_test_data()
    
    train_end = 1000
    train_data = data[:train_end]
    test_data = data[train_end:]
    test_labels = true_labels[train_end:]
    
    # 防泄漏：仅在训练集拟合标准化
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_data)
    test_scaled = scaler.transform(test_data)
    
    # 【优化点 1】收紧污染率 contamination，减少误报基数
    logger.info("正在训练优化后的 Isolation Forest (降低 contamination=0.01)...")
    iso_forest = IsolationForest(contamination=0.01, random_state=42)
    iso_forest.fit(train_scaled)
    
    # 获取连续的异常得分 (decision_function 越低代表越异常)
    test_scores = iso_forest.decision_function(test_scaled)
    
    # 【优化点 2】利用训练集正常数据的得分分布确定严格的分位数阈值（严禁使用测试集标签！）
    train_scores = iso_forest.decision_function(train_scaled)
    # 取训练集得分的下 1% 作为告警红线
    dynamic_threshold = np.percentile(train_scores, 1.0) 
    
    # 生成原始预测
    pred_labels = np.where(test_scores < dynamic_threshold, 1, 0)
    
    # 【优化点 3】加入时间窗口滑动平滑（连续多步触发才算真实异常，过滤毛刺）
    df_preds = pd.Series(pred_labels)
    # 比如要求连续 3 步中有 2 步被判为异常才触发
    smoothed_preds = df_preds.rolling(window=5, min_periods=1).mean()
    final_pred_labels = np.where(smoothed_preds >= 0.4, 1, 0)
    
    # 评估指标
    metrics = evaluate_anomaly_detection(test_labels, final_pred_labels)
    logger.info(f"优化后异常检测表现: Precision={metrics['Precision']} | Recall={metrics['Recall']} | F1={metrics['F1']}")
    logger.info("============== 优化实验执行完毕 ==============")

if __name__ == "__main__":
    main()