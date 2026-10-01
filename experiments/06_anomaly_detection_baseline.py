# 内置一个轻量级的isolated forest 与Z-score统计基线
# 并在模拟的时序数据上跑通整个异常事件

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
    time = np.arange(T)
    # 模拟平稳的正常钻井状态 (如立管压力、扭矩)
    data = np.random.randn(T, N) * 0.5 + 10.0
    
    # 注入人工已知异常事件 (模拟第 1200~1250 时间步的钻井异常，如局部压力突变)
    anomaly_window = range(1200, 1250)
    data[anomaly_window, 1] += 5.0  # 传感器 1 发生剧烈漂移
    
    # 构造二分类真实标签 (1代表异常，0代表正常)
    labels = np.zeros(T, dtype=int)
    labels[anomaly_window] = 1
    
    return data, labels

def evaluate_anomaly_detection(true_labels, pred_labels):
    """计算事件级的异常检测指标"""
    # 简单的混淆矩阵指标
    tp = np.sum((true_labels == 1) & (pred_labels == 1))
    fp = np.sum((true_labels == 0) & (pred_labels == 1))
    fn = np.sum((true_labels == 1) & (pred_labels == 0))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {"Precision": round(precision, 4), "Recall": round(recall, 4), "F1": round(f1, 4)}

def main():
    logger = setup_logger("anomaly_baseline")
    set_seed(42)
    
    logger.info("============== 模块三：异常检测基线跑通实验 ==============")
    
    # 1. 获取带有异常标签的时序数据
    data, true_labels = generate_anomaly_test_data()
    
    # 2. 严格按防泄漏协议：前 1000 步作为纯净训练集（只用来拟合标准化与正常分布）
    train_end = 1000
    train_data = data[:train_end]
    test_data = data[train_end:]
    test_labels = true_labels[train_end:]
    
    # 3. 拟合标准化参数 (严格禁止使用测试集)
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_data)
    test_scaled = scaler.transform(test_data)
    
    # 4. 训练传统机器学习基线: Isolation Forest
    logger.info("正在训练 Isolation Forest 异常检测基线...")
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    iso_forest.fit(train_scaled)
    
    # 预测测试集 (-1 代表异常，1 代表正常，转换为 1 和 0)
    test_preds_raw = iso_forest.predict(test_scaled)
    pred_labels = np.where(test_preds_raw == -1, 1, 0)
    
    # 5. 评估指标
    metrics = evaluate_anomaly_detection(test_labels, pred_labels)
    logger.info(f"异常检测基线表现: Precision={metrics['Precision']} | Recall={metrics['Recall']} | F1={metrics['F1']}")
    logger.info("============== 异常检测管线打通完毕 ==============")

if __name__ == "__main__":
    main()