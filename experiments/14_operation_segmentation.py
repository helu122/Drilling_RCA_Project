import sys
import os
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import setup_logger

def extract_drilling_segments(input_path, output_dir, logger):
    logger.info("[-] 正在加载对齐后的时序 Parquet 矩阵...")
    df = pd.read_parquet(input_path)
    initial_rows = len(df)
    
    # 1. 核心过滤放宽：只要有进尺和转速，就认为是钻进
    thresholds = {
        'RPM': 10.0,    
        'ROP': 0.1      
    }
    
    logger.info(f"[-] 应用基础物理阈值进行过滤: {thresholds}")
    
    mask = pd.Series(True, index=df.index)
    for col, min_val in thresholds.items():
        if col in df.columns:
            mask = mask & (df[col].notna()) & (df[col] > min_val)
            
    df_drilling = df[mask].copy()
    
    # 增加调试信息：看看究竟有多少行满足了基础条件
    logger.info(f"[Debug] 满足阈值的离散点数据共有: {len(df_drilling)} 行")
    
    if len(df_drilling) == 0:
        logger.error("没有任何数据满足基础阈值，工况提取失败！")
        return
        
    logger.info("[-] 正在切割连续钻进片段...")
    df_drilling['time_diff'] = df_drilling.index.to_series().diff().dt.total_seconds()
    
    # 2. 容忍度放宽：如果两次有效信号间隔超过 10 分钟 (600秒)，才切断片段
    df_drilling['segment_id'] = (df_drilling['time_diff'] > 600).cumsum()
    df_drilling.drop(columns=['time_diff'], inplace=True)
    
    segment_stats = df_drilling.groupby('segment_id').agg(
        start_time=('RPM', lambda x: x.index.min()),
        end_time=('RPM', lambda x: x.index.max()),
        duration_minutes=('RPM', lambda x: len(x) * 10 / 60.0),
        row_count=('RPM', 'count')
    ).reset_index()
    
    # 3. 碎片容忍放宽：保留大于 5 分钟的有效片段 (之前是 30 分钟)
    valid_segments = segment_stats[segment_stats['duration_minutes'] >= 5]
    
    df_final = df_drilling[df_drilling['segment_id'].isin(valid_segments['segment_id'])]
    
    final_rows = len(df_final)
    retention_rate = (final_rows / initial_rows) * 100
    
    logger.info(f"==> 工况切分完成！")
    logger.info(f"原始数据总时长: ~{initial_rows * 10 / 3600:.1f} 小时 ({initial_rows} 行)")
    logger.info(f"提取出纯钻进有效时长: ~{final_rows * 10 / 3600:.1f} 小时 ({final_rows} 行)")
    logger.info(f"有效数据保留率: {retention_rate:.2f}%")
    logger.info(f"共发现 {len(valid_segments)} 个高质量连续纯钻进片段 (>= 5分钟)。")
    
    os.makedirs(output_dir, exist_ok=True)
    out_parquet = os.path.join(output_dir, "pure_drilling_segments.parquet")
    out_csv = os.path.join(output_dir, "segment_catalog.csv")
    
    df_final.to_parquet(out_parquet, engine='pyarrow')
    valid_segments.to_csv(out_csv, index=False, encoding='utf-8-sig')
    
    if len(valid_segments) > 0:
        logger.info("\n--- 最长的前 5 个高质量钻进片段 ---")
        pd.set_option('display.max_columns', None)
        logger.info(f"\n{valid_segments.sort_values('duration_minutes', ascending=False).head()}")

def main():
    logger = setup_logger("operation_segmentation")
    logger.info("============== 第二阶段：钻井工况自动筛选 (降级包容版) ==============")
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_parquet = os.path.join(base_dir, "results_archive", "06_time_alignment", "aligned_timeseries.parquet")
    output_dir = os.path.join(base_dir, "results_archive", "07_operation_segmentation")
    
    extract_drilling_segments(input_parquet, output_dir, logger)
    logger.info("============== 管线执行完毕 ==============")

if __name__ == "__main__":
    main()