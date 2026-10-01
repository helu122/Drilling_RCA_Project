import sys
import os
import glob
import xml.etree.ElementTree as ET
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import setup_logger

def scan_witsml_directory(raw_data_dir, logger):
    namespaces = {'witsml': 'http://www.witsml.org/schemas/1series'}
    
    # 你的 RCA 诊断黄金变量雷达名单 (可以随时增加)
    golden_vars = ['WOB', 'RPM', 'Q', 'SPP', 'TORQUE', 'HKLD', 'ROP', 'ECD', 'GAS']
    
    inventory_records = []
    global_var_dict = {}
    
    xml_files = glob.glob(os.path.join(raw_data_dir, "*.xml"))
    logger.info(f"在 {raw_data_dir} 中发现 {len(xml_files)} 个 XML 文件。")
    
    for file_path in xml_files:
        file_name = os.path.basename(file_path)
        logger.info(f"正在扫描盲盒: {file_name}...")
        
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            log_node = root.find('witsml:log', namespaces)
            if log_node is None:
                continue
                
            # 提取井号与日志名
            well_name = log_node.attrib.get('uid', 'Unknown_Well')
            log_name_node = log_node.find('witsml:name', namespaces)
            log_name = log_name_node.text if log_name_node is not None else "Unknown_Log"
            
            # 解析变量字典
            curve_infos = log_node.findall('witsml:logCurveInfo', namespaces)
            current_file_vars = []
            
            for curve in curve_infos:
                mnemonic_node = curve.find('witsml:mnemonic', namespaces)
                mnemonic = mnemonic_node.text if mnemonic_node is not None else ""
                
                unit_node = curve.find('witsml:unit', namespaces)
                unit = unit_node.text if unit_node is not None else ""
                
                desc_node = curve.find('witsml:curveDescription', namespaces)
                desc = desc_node.text if desc_node is not None else ""
                
                current_file_vars.append(mnemonic)
                
                # 汇总到全局大字典中
                if mnemonic and mnemonic not in global_var_dict:
                    global_var_dict[mnemonic] = {
                        "Mnemonic": mnemonic,
                        "Unit": unit,
                        "Description": desc,
                        "Is_Golden_Var": "Yes" if mnemonic.upper() in golden_vars else "No",
                        "Found_In_Files_Count": 1
                    }
                elif mnemonic:
                    global_var_dict[mnemonic]["Found_In_Files_Count"] += 1
            
            # 提取时间跨度 (直接读取时序数据的首尾)
            start_time, end_time = "Unknown", "Unknown"
            log_data_node = log_node.find('witsml:logData', namespaces)
            if log_data_node is not None:
                data_nodes = log_data_node.findall('witsml:data', namespaces)
                if len(data_nodes) > 0:
                    start_time = data_nodes[0].text.split(',')[0]
                    end_time = data_nodes[-1].text.split(',')[0]
            
            # 生成该文件的资产档案
            record = {
                "File_Name": file_name,
                "Well_ID": well_name,
                "Log_Name": log_name,
                "Start_Time": start_time,
                "End_Time": end_time,
                "Total_Variables": len(current_file_vars),
                "Golden_Vars_Found": ", ".join([v for v in current_file_vars if v.upper() in golden_vars])
            }
            inventory_records.append(record)
            
        except Exception as e:
            logger.error(f"解析 {file_name} 失败: {e}")
            
    df_inventory = pd.DataFrame(inventory_records)
    df_dict = pd.DataFrame(global_var_dict.values())
    return df_inventory, df_dict

def main():
    logger = setup_logger("data_inventory")
    logger.info("============== 第二阶段：数据盘点与特征映射雷达 ==============")
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_data_dir = os.path.join(base_dir, "data", "raw")
    
    # 建立输出归档目录
    output_dir = os.path.join(base_dir, "results_archive", "05_data_inventory")
    os.makedirs(output_dir, exist_ok=True)
    
    df_inventory, df_dict = scan_witsml_directory(raw_data_dir, logger)
    
    if not df_inventory.empty:
        inv_path = os.path.join(output_dir, "inventory.csv")
        dict_path = os.path.join(output_dir, "variable_dictionary.csv")
        
        # 存盘并输出 (utf-8-sig 防止 Excel 打开中文乱码)
        df_inventory.to_csv(inv_path, index=False, encoding='utf-8-sig')
        df_dict.to_csv(dict_path, index=False, encoding='utf-8-sig')
        
        logger.info(f"\n--- 扫描完毕，成果已生成 ---")
        logger.info(f"1. 数据资产清单已保存: {inv_path}")
        logger.info(f"2. 全局变量字典已保存: {dict_path}")
        
        logger.info("\n预览 1: Inventory (文件资产盘点)")
        logger.info(f"\n{df_inventory[['File_Name', 'Well_ID', 'Start_Time', 'Golden_Vars_Found']]}")
        
        logger.info("\n预览 2: Variable Dictionary (特征字典盘点)")
        logger.info(f"\n{df_dict[['Mnemonic', 'Unit', 'Is_Golden_Var', 'Description']]}")
    else:
        logger.warning("未扫描到任何有效的 WITSML 数据。")
        
    logger.info("============== 盘点管线就绪 ==============")

if __name__ == "__main__":
    main()