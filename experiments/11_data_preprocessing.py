import sys
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import setup_logger, set_seed

def generate_raw_sensor_data(logger):
    """生成一段带高频噪声和缺失值的模拟原始钻井数据"""
    set_seed(42)
    time_idx = pd.date_range(start="2014-01-01 00:00:00", periods=15, freq="S")
    
    # 模拟真实物理趋势 + 高频机械波动
    wob = np.linspace(15.0, 16.5, 15) + np.random.normal(0, 0.5, 15)
    spp = np.linspace(3000, 3050, 15) + np.random.normal(0, 5, 15)
    
    df = pd.DataFrame({
        'TIME': time_idx,
        'WOB': wob,
        'SPP': spp
    })
    
    # 人工注入网络空洞 (缺失值)
    df.loc[3:5, 'WOB'] = np.nan  # 连续缺失
    df.loc[10, 'SPP'] = np.nan   # 单点缺失
    
    logger.info("\n--- 原始数据 (包含缺失值和高频毛刺) 前12行 ---")
    logger.info(f"\n{df.head(12)}")
    return df

def preprocess_drilling_data(df, logger):
    """数据预处理管线：插值填补 -> 滑动平滑 -> Z-score归一化"""
    
    cols_to_process = ['WOB', 'SPP']
    
    # 1. 缺失值填补 (Imputation)
    logger.info("\n[步骤 1] 正在执行缺失值线性插值 (Linear Interpolation)...")
    df_imputed = df.copy()
    # 时序插值能完美顺应前后的物理趋势
    df_imputed[cols_to_process] = df_imputed[cols_to_process].interpolate(method='linear')
    # 边缘兜底填充 (防止头尾恰好有 NaN)
    df_imputed[cols_to_process] = df_imputed[cols_to_process].bfill().ffill()
    logger.info(f"填补后缺失值统计:\n{df_imputed.isnull().sum()}")
    
    # 2. 滤波平滑 (Smoothing)
    logger.info("\n[步骤 2] 正在执行滑动窗口平滑 (Window=3) 压制机械噪声...")
    df_smoothed = df_imputed.copy()
    df_smoothed[cols_to_process] = df_smoothed[cols_to_process].rolling(window=3, min_periods=1).mean()
    
    # 3. 归一化/标准化 (Standardization)
    logger.info("\n[步骤 3] 正在执行 Z-Score 标准化，消除量纲差异...")
    scaler = StandardScaler()
    df_scaled = df_smoothed.copy()
    df_scaled[cols_to_process] = scaler.fit_transform(df_smoothed[cols_to_process])
    
    logger.info("\n--- 最终预处理完成的 DataFrame (可直接输入模型) ---")
    logger.info(f"\n{df_scaled.head(8)}")
    
    return df_scaled, scaler

def main():
    logger = setup_logger("data_preprocessing")
    logger.info("============== 模块四：数据清洗与预处理管线 ==============")
    
    df_raw = generate_raw_sensor_data(logger)
    df_ready, scaler = preprocess_drilling_data(df_raw, logger)
    
    logger.info("============== 预处理管线测试完毕 ==============")

if __name__ == "__main__":
    main()