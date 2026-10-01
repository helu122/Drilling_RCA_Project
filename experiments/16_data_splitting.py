import sys
import os
import json
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import setup_logger

def split_time_series_data(catalog_path, output_dir, logger):
    logger.info("[-] 正在加载带标签的钻进片段名录...")
    df_catalog = pd.read_csv(catalog_path)
    
    # 核心防泄漏操作：严格按照真实物理时间先后排序
    df_catalog['start_time'] = pd.to_datetime(df_catalog['start_time'])
    df_catalog = df_catalog.sort_values('start_time').reset_index(drop=True)
    
    total_segments = len(df_catalog)
    logger.info(f"[-] 共有 {total_segments} 个有效纯钻片段参与划分。")
    
    # 划分比例：70% 训练集 (正常钻进模式学习)，15% 验证集，15% 测试集
    train_end = int(total_segments * 0.7)
    val_end = int(total_segments * 0.85)
    
    # 获取每一部分的 segment_id 列表 (需转为原生 Python int 类型以便 JSON 序列化)
    train_ids = [int(x) for x in df_catalog.iloc[:train_end]['segment_id'].tolist()]
    val_ids = [int(x) for x in df_catalog.iloc[train_end:val_end]['segment_id'].tolist()]
    test_ids = [int(x) for x in df_catalog.iloc[val_end:]['segment_id'].tolist()]
    
    # 统计每一集中的异常分布情况
    def get_labels(df_subset):
        return df_subset['label'].value_counts().to_dict()
        
    split_manifest = {
        "metadata": {
            "total_segments": total_segments,
            "split_strategy": "Chronological (Time-based no-shuffle)",
            "leakage_prevention": True
        },
        "train": {
            "count": len(train_ids),
            "segments": train_ids,
            "label_distribution": get_labels(df_catalog.iloc[:train_end])
        },
        "validation": {
            "count": len(val_ids),
            "segments": val_ids,
            "label_distribution": get_labels(df_catalog.iloc[train_end:val_end])
        },
        "test": {
            "count": len(test_ids),
            "segments": test_ids,
            "label_distribution": get_labels(df_catalog.iloc[val_end:])
        }
    }
    
    os.makedirs(output_dir, exist_ok=True)
    out_json = os.path.join(output_dir, "split_manifest.json")
    
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(split_manifest, f, indent=4, ensure_ascii=False)
        
    logger.info(f"==> 严格防泄漏切分完成！")
    logger.info(f"Train: {len(train_ids)} 段 | Val: {len(val_ids)} 段 | Test: {len(test_ids)} 段")
    logger.info(f"数据划分清单已保存至: {out_json}")
    
    logger.info("\n--- 最终生成的 JSON Manifest 预览 ---")
    logger.info(json.dumps(split_manifest['metadata'], indent=2))

def main():
    logger = setup_logger("data_splitting")
    logger.info("============== 第二阶段：严格防泄漏数据集划分 ==============")
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    catalog_path = os.path.join(base_dir, "results_archive", "08_event_catalog", "labeled_segment_catalog.csv")
    output_dir = os.path.join(base_dir, "results_archive", "09_data_splitting")
    
    if not os.path.exists(catalog_path):
        logger.error(f"未找到输入文件: {catalog_path}，请先运行 15_ddr_event_parser.py")
        return
        
    split_time_series_data(catalog_path, output_dir, logger)
    logger.info("============== 第二阶段全部管线顺利收官！ ==============")

if __name__ == "__main__":
    main()