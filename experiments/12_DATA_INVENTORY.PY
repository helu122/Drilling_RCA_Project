import sys
import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import setup_logger

def scan_data_directory(raw_data_dir, logger):
    namespaces = {'witsml': 'http://www.witsml.org/schemas/1series'}
    
    # RCA 诊断黄金变量雷达名单
    golden_vars = ['WOB', 'RPM', 'Q', 'SPP', 'TORQUE', 'HKLD', 'ROP', 'ECD', 'GAS', 'FLOW']
    
    inventory_records = []
    global_var_dict = {}
    
    # 核心升级：加入 recursive=True，自动穿透扫描所有子文件夹中的数据
    xml_files = glob.glob(os.path.join(raw_data_dir, "**", "*.xml"), recursive=True)
    csv_files = glob.glob(os.path.join(raw_data_dir, "**", "*.csv"), recursive=True)
    all_files = xml_files + csv_files
    
    logger.info(f"在 {raw_data_dir} 及其子目录中发现 {len(xml_files)} 个 XML 和 {len(csv_files)} 个 CSV 文件。")
    
    for file_path in all_files:
        file_name = os.path.basename(file_path)
        logger.info(f"正在扫描盲盒: {file_name}...")
        ext = os.path.splitext(file_name)[1].lower()
        
        try:
            if ext == '.xml':
                # --- WITSML 解析逻辑 ---
                tree = ET.parse(file_path)
                root = tree.getroot()
                log_node = root.find('witsml:log', namespaces)
                if log_node is None:
                    continue
                    
                well_name = log_node.attrib.get('uid', 'Unknown_Well')
                log_name = log_node.find('witsml:name', namespaces).text if log_node.find('witsml:name', namespaces) is not None else "Unknown_Log"
                
                curve_infos = log_node.findall('witsml:logCurveInfo', namespaces)
                current_file_vars = []
                
                for curve in curve_infos:
                    mnemonic = curve.find('witsml:mnemonic', namespaces).text if curve.find('witsml:mnemonic', namespaces) is not None else ""
                    unit = curve.find('witsml:unit', namespaces).text if curve.find('witsml:unit', namespaces) is not None else ""
                    desc = curve.find('witsml:curveDescription', namespaces).text if curve.find('witsml:curveDescription', namespaces) is not None else ""
                    
                    if mnemonic:
                        current_file_vars.append(mnemonic)
                        if mnemonic not in global_var_dict:
                            global_var_dict[mnemonic] = {
                                "Mnemonic": mnemonic, "Unit": unit, "Description": desc,
                                "Is_Golden_Var": "Yes" if any(g in mnemonic.upper() for g in golden_vars) else "No",
                                "Found_In_Files_Count": 1
                            }
                        else:
                            global_var_dict[mnemonic]["Found_In_Files_Count"] += 1
                            
                start_time, end_time = "Unknown", "Unknown"
                log_data_node = log_node.find('witsml:logData', namespaces)
                if log_data_node is not None and len(log_data_node.findall('witsml:data', namespaces)) > 0:
                    data_nodes = log_data_node.findall('witsml:data', namespaces)
                    start_time = data_nodes[0].text.split(',')[0]
                    end_time = data_nodes[-1].text.split(',')[0]

            elif ext == '.csv':
                # --- CSV 极速解析逻辑 ---
                df_head = pd.read_csv(file_path, nrows=5)
                current_file_vars = df_head.columns.tolist()
                
                time_col = None
                for col in current_file_vars:
                    if 'time' in col.lower() or 'date' in col.lower():
                        time_col = col
                        break
                
                start_time, end_time = "Unknown", "Unknown"
                if time_col:
                    df_time = pd.read_csv(file_path, usecols=[time_col]).dropna()
                    if not df_time.empty:
                        start_time = str(df_time.iloc[0][time_col])
                        end_time = str(df_time.iloc[-1][time_col])
                
                well_name = "Volve_F9A" if "F9" in file_name.upper() else "Unknown_Well_CSV"
                log_name = "Realtime_Drilling_CSV"
                
                for col in current_file_vars:
                    mnemonic = col.strip()
                    if mnemonic:
                        if mnemonic not in global_var_dict:
                            global_var_dict[mnemonic] = {
                                "Mnemonic": mnemonic, "Unit": "Check CSV", "Description": "Extracted from CSV",
                                "Is_Golden_Var": "Yes" if any(g in mnemonic.upper() for g in golden_vars) else "No",
                                "Found_In_Files_Count": 1
                            }
                        else:
                            global_var_dict[mnemonic]["Found_In_Files_Count"] += 1
                            
            # 汇总资产档案
            golden_found = [v for v in current_file_vars if any(g in v.upper() for g in golden_vars)]
            record = {
                "File_Name": file_name,
                "Well_ID": well_name,
                "Log_Name": log_name,
                "Start_Time": start_time,
                "End_Time": end_time,
                "Total_Variables": len(current_file_vars),
                "Golden_Vars_Found": ", ".join(golden_found)
            }
            inventory_records.append(record)
            
        except Exception as e:
            logger.error(f"解析 {file_name} 失败: {e}")
            
    return pd.DataFrame(inventory_records), pd.DataFrame(global_var_dict.values())

def main():
    logger = setup_logger("data_inventory")
    logger.info("============== 第二阶段：双引擎数据盘点雷达 (XML+CSV) ==============")
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_data_dir = os.path.join(base_dir, "data", "raw")
    output_dir = os.path.join(base_dir, "results_archive", "05_data_inventory")
    os.makedirs(output_dir, exist_ok=True)
    
    df_inventory, df_dict = scan_data_directory(raw_data_dir, logger)
    
    if not df_inventory.empty:
        df_inventory.to_csv(os.path.join(output_dir, "inventory.csv"), index=False, encoding='utf-8-sig')
        df_dict.to_csv(os.path.join(output_dir, "variable_dictionary.csv"), index=False, encoding='utf-8-sig')
        
        logger.info("\n--- 预览 1: Inventory (文件资产盘点) ---")
        pd.set_option('display.max_columns', None)
        logger.info(f"\n{df_inventory[['File_Name', 'Start_Time', 'Total_Variables', 'Golden_Vars_Found']]}")
    else:
        logger.warning("未扫描到任何有效的测井数据文件。")

if __name__ == "__main__":
    main()