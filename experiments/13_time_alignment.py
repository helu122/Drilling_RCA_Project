import sys
import os
import glob
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import setup_logger

def align_and_clean_timeseries(raw_data_dir, output_dir, logger):
    # 1. 精准锁定 time.csv
    csv_files = glob.glob(os.path.join(raw_data_dir, "**", "*time*.csv"), recursive=True)
    if not csv_files:
        logger.error("未找到基于时间的 CSV 数据！请检查文件名是否包含 'time'。")
        return

    target_file = csv_files[0]
    logger.info(f"==> 锁定目标数据: {os.path.basename(target_file)}")

    # 2. 定义特征漏斗：白名单字典映射
    # 规则：一旦列名中包含列表里的任何一个关键词，就被提取并赋予标准简写
    mapping_rules = {
        'WOB': ['weight on bit', 'wob'],
        'RPM': ['rotary speed', 'rpm', 'surf_rpm'],
        'SPP': ['standpipe', 'press', 'spp'],
        'TORQUE': ['torque', 'stor'],
        'ROP': ['rate of penetration', 'rop'],
        'HKLD': ['hook load', 'hkld'],
        'Q': ['flow rate', 'pump', 'mud flow in']
    }

    # 3. 极速加载原始数据
    logger.info("[-] 正在加载原始工业数据 (视文件大小可能需要数秒)...")
    df_raw = pd.read_csv(target_file, low_memory=False)
    
    # 4. 识别时间列
    time_col = next((col for col in df_raw.columns if 'time' in col.lower() or 'date' in col.lower()), None)
    if not time_col:
        logger.error("未在数据集中找到时间戳列！")
        return

    # 5. 执行第一层漏斗：提取黄金特征
    logger.info("[-] 正在执行降维提取：剔除冗余变量...")
    extracted_cols = {time_col: 'TIME'}
    for standard_name, keywords in mapping_rules.items():
        for col in df_raw.columns:
            if any(kw in col.lower() for kw in keywords):
                extracted_cols[col] = standard_name
                break  # 找到信号最靠前的代表特征即跳出

    df_filtered = df_raw[list(extracted_cols.keys())].rename(columns=extracted_cols)
    logger.info(f"成功从 {len(df_raw.columns)} 个变量中提炼出 {len(df_filtered.columns)-1} 个核心特征: {list(df_filtered.columns)[1:]}")

    # 6. 时间对齐与重采样
    logger.info("[-] 正在执行时间轴对齐与重采样 (Resampling to 10S)...")
    # 强制转换标准时间格式，并设为索引
    df_filtered['TIME'] = pd.to_datetime(df_filtered['TIME'], errors='coerce', utc=True)
    df_filtered = df_filtered.dropna(subset=['TIME'])
    df_filtered.set_index('TIME', inplace=True)

    # 强制将所有特征列转为数值型 (剔除字符串干扰)
    df_filtered = df_filtered.apply(pd.to_numeric, errors='coerce')

    # 核心降噪对齐：10秒一个窗口，取平均值
    df_resampled = df_filtered.resample('10S').mean()

    # 7. 填补微小空洞
    logger.info("[-] 正在填补重采样造成的物理空洞 (线性插值)...")
    # limit=3 表示最多只自动补齐连续断联的 30 秒数据，再长就保留 NaN 交给后续分段逻辑处理
    df_resampled = df_resampled.interpolate(method='linear', limit=3)

    # 8. 序列化输出为 Parquet 格式
    logger.info("[-] 正在将标准时序矩阵序列化为 Parquet 格式...")
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "aligned_timeseries.parquet")
    
    # 默认使用 pyarrow 引擎保存
    df_resampled.to_parquet(out_path, engine='pyarrow')

    logger.info(f"==> 对齐完成！输出文件已保存至: {out_path}")
    logger.info(f"最终矩阵维度 (时间步数, 特征数): {df_resampled.shape}")
    logger.info(f"\n--- 最终生成的纯净时序矩阵头部 ---\n{df_resampled.head()}")

def main():
    logger = setup_logger("time_alignment")
    logger.info("============== 第二阶段：多源数据时间对齐与清洗 ==============")
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_data_dir = os.path.join(base_dir, "data", "raw")
    output_dir = os.path.join(base_dir, "results_archive", "06_time_alignment")
    
    align_and_clean_timeseries(raw_data_dir, output_dir, logger)
    logger.info("============== 管线执行完毕 ==============")

if __name__ == "__main__":
    main()