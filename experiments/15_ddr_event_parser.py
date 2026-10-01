import sys
import os
import pandas as pd
import re

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import setup_logger

def create_mock_ddr_reports():
    # 模拟从 Volve 官方 DDR 报告中 OCR/解析出来的文本记录
    # 时间点特意对齐了你提取出来的纯钻进片段
    return [
        {
            "report_time": "2009-07-03 20:00:00+00:00", 
            "description": "Drilling ahead smoothly. No issues observed. Connection made at 20:45."
        },
        {
            "report_time": "2009-07-04 01:15:00+00:00", 
            "description": "Observed total loss of returns at shakers. SPP dropped suddenly. Pumping LCM pills."
        },
        {
            "report_time": "2009-07-04 06:10:00+00:00", 
            "description": "Drilling normal. Gas peaks observed but controlled with ECD."
        },
        {
            "report_time": "2009-07-07 13:45:00+00:00", 
            "description": "High erratic torque. Overpull on connection. Pipe stuck at 3450m. Attempting to jar down."
        }
    ]

def parse_ddr_events(mock_ddr, logger):
    logger.info("[-] 正在使用正则 NLP 引擎解析 DDR 文本报告...")
    
    # 弱标签异常字典 (关键词)
    event_keywords = {
        "Stuck Pipe (卡钻)": r"\b(stuck|overpull|pack off|tight hole)\b",
        "Loss (井漏)": r"\b(loss|losses|no returns)\b",
        "Kick (井涌)": r"\b(kick|influx|flow|pit gain)\b"
    }
    
    parsed_events = []
    for entry in mock_ddr:
        desc_lower = entry['description'].lower()
        detected_fault = "Normal"
        confidence = "High"
        
        for fault_type, pattern in event_keywords.items():
            if re.search(pattern, desc_lower):
                detected_fault = fault_type
                break
                
        parsed_events.append({
            "event_time": pd.to_datetime(entry['report_time']),
            "fault_type": detected_fault,
            "evidence": entry['description'],
            "confidence": confidence
        })
        
    return pd.DataFrame(parsed_events)

def match_labels_to_segments(segments_df, events_df, output_dir, logger):
    logger.info("[-] 正在将异常事件时间戳映射至钻进片段 (Ground Truth 绑定)...")
    
    # 确保时间列格式统一 (带时区)
    segments_df['start_time'] = pd.to_datetime(segments_df['start_time'])
    segments_df['end_time'] = pd.to_datetime(segments_df['end_time'])
    
    # 初始化标签列
    segments_df['label'] = 'Normal'
    segments_df['fault_evidence'] = 'None'
    
    # 交叉匹配逻辑：如果报告的异常时间落在这个钻进片段的区间内，就给该片段打标
    match_count = 0
    for idx, segment in segments_df.iterrows():
        for e_idx, event in events_df.iterrows():
            if segment['start_time'] <= event['event_time'] <= segment['end_time']:
                if event['fault_type'] != 'Normal':
                    segments_df.at[idx, 'label'] = event['fault_type']
                    segments_df.at[idx, 'fault_evidence'] = event['evidence']
                    match_count += 1
                    
    logger.info(f"==> 标签绑定完成！成功将 {match_count} 个异常事件挂载到有效纯钻片段上。")
    
    os.makedirs(output_dir, exist_ok=True)
    out_csv = os.path.join(output_dir, "labeled_segment_catalog.csv")
    segments_df.to_csv(out_csv, index=False, encoding='utf-8-sig')
    
    logger.info(f"带标签的片段名录已保存至: {out_csv}")
    
    # 打印最终带有异常标签的数据
    faulty_segments = segments_df[segments_df['label'] != 'Normal']
    logger.info("\n--- 被诊断为异常的钻进片段 (RCA 分析的重点目标) ---")
    if not faulty_segments.empty:
        pd.set_option('display.max_columns', None)
        logger.info(f"\n{faulty_segments[['segment_id', 'start_time', 'duration_minutes', 'label', 'fault_evidence']]}")
    else:
        logger.info("当前所有片段均为 Normal。")

def main():
    logger = setup_logger("ddr_parser")
    logger.info("============== 第二阶段：DDR 异常事件弱标签库构建 ==============")
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    catalog_path = os.path.join(base_dir, "results_archive", "07_operation_segmentation", "segment_catalog.csv")
    output_dir = os.path.join(base_dir, "results_archive", "08_event_catalog")
    
    if not os.path.exists(catalog_path):
        logger.error(f"未找到输入片段名录: {catalog_path}，请先运行 14_operation_segmentation.py")
        return
        
    segments_df = pd.read_csv(catalog_path)
    mock_ddr = create_mock_ddr_reports()
    
    events_df = parse_ddr_events(mock_ddr, logger)
    match_labels_to_segments(segments_df, events_df, output_dir, logger)
    
    logger.info("============== 管线执行完毕 ==============")

if __name__ == "__main__":
    main()