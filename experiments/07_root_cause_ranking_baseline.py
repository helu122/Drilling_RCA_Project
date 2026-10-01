import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.utils import set_seed, setup_logger

def generate_rca_test_data(T=2000, N=5, seed=42):
    """生成带有连锁反应的异常数据，用于根因溯源测试"""
    np.random.seed(seed)
    data = np.random.randn(T, N) * 0.5 + 10.0
    
    # 定义真实根因
    true_root_cause = 1
    
    # 模拟真实钻井的连锁异常反应：
    # 1. 根因传感器1 在 1200 步率先发生严重突变 (真实根因)
    data[1200:1250, true_root_cause] += 5.0  
    
    # 2. 传感器3 在 5 步后(1205)受到物理传导，发生联动突变 (衍生异常，非根因)
    data[1205:1250, 3] += 3.5  
    
    # 模拟异常检测模块抛出的告警窗口 (假设 Isolation Forest 成功框出了这段异常)
    detected_window = range(1190, 1260) 
    
    return data, detected_window, true_root_cause

def rca_magnitude_ranking(train_data, anomaly_window_data):
    """基线 1：残差漂移幅度排序 (Z-score Magnitude)"""
    mean = np.mean(train_data, axis=0)
    std = np.std(train_data, axis=0)
    
    # 计算异常窗口内各传感器的绝对 Z-score
    z_scores = np.abs((anomaly_window_data - mean) / std)
    
    # 取每个传感器在窗口内的最大偏离度作为根因嫌疑分
    sensor_scores = np.max(z_scores, axis=0)
    
    # 降序排列：偏离越大的排在越前面
    ranked_sensors = np.argsort(sensor_scores)[::-1]
    return ranked_sensors, sensor_scores

def rca_earliest_change_ranking(train_data, anomaly_window_data, z_threshold=3.0):
    """基线 2：最早变化时间排序 (Earliest Change-Point)"""
    mean = np.mean(train_data, axis=0)
    std = np.std(train_data, axis=0)
    
    z_scores = np.abs((anomaly_window_data - mean) / std)
    N = z_scores.shape[1]
    
    first_change_times = np.full(N, np.inf)
    
    for i in range(N):
        # 找到该传感器第一次超过 3-sigma 阈值的时间步
        exceed_idx = np.where(z_scores[:, i] > z_threshold)[0]
        if len(exceed_idx) > 0:
            first_change_times[i] = exceed_idx[0]
            
    # 升序排列：越早发生突变的排在越前面
    ranked_sensors = np.argsort(first_change_times)
    return ranked_sensors, first_change_times

def evaluate_rca(ranked_list, true_root_cause):
    """计算 RCA 核心评价指标: Top-1 准确率与 MRR"""
    # 找到真实根因在推荐列表中的排名 (0-indexed，需要 +1)
    rank_idx = np.where(ranked_list == true_root_cause)[0][0] + 1
    
    hit_at_1 = 1 if rank_idx == 1 else 0
    mrr = 1.0 / rank_idx
    
    return rank_idx, hit_at_1, round(mrr, 4)

def main():
    logger = setup_logger("rca_baseline")
    set_seed(42)
    
    logger.info("============== 模块三：根因排序 (RCA) 基线跑通 ==============")
    
    # 1. 加载数据
    data, detected_window, true_root_cause = generate_rca_test_data()
    logger.info(f"真实根因 (Ground Truth) 传感器编号: {true_root_cause}")
    
    # 纯净训练集 (用于计算正常分布)
    train_data = data[:1000]
    # 取出被异常检测器框定的异常数据段
    anomaly_window_data = data[detected_window]
    
    # 2. 执行 基线 1：残差幅度排序
    logger.info("\n--- 执行 基线 1: 残差幅度排序 (Magnitude) ---")
    rank_mag, scores_mag = rca_magnitude_ranking(train_data, anomaly_window_data)
    rank_idx, hit1, mrr = evaluate_rca(rank_mag, true_root_cause)
    logger.info(f"排序结果: {rank_mag} | 各传感器最大偏差: {np.round(scores_mag, 2)}")
    logger.info(f"指标 -> 真实根因最终排名: 第 {rank_idx} 名 | Precision@1: {hit1} | MRR: {mrr}")
    
    # 3. 执行 基线 2：最早变化时间排序
    logger.info("\n--- 执行 基线 2: 最早变化时间排序 (Earliest Change) ---")
    rank_time, times = rca_earliest_change_ranking(train_data, anomaly_window_data, z_threshold=3.0)
    rank_idx_t, hit1_t, mrr_t = evaluate_rca(rank_time, true_root_cause)
    # 把无限大的时间用 -1 表示方便阅读
    print_times = [int(t) if t != np.inf else -1 for t in times]
    logger.info(f"排序结果: {rank_time} | 各传感器触发时间步(相对): {print_times}")
    logger.info(f"指标 -> 真实根因最终排名: 第 {rank_idx_t} 名 | Precision@1: {hit1_t} | MRR: {mrr_t}")
    
    logger.info("\n============== RCA 基线评估完毕 ==============")

if __name__ == "__main__":
    main()