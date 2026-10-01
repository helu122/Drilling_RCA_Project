import sys
import os
import xml.etree.ElementTree as ET
import pandas as pd

# 确保能导入 src 里的 setup_logger
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import setup_logger

def generate_mock_witsml_file(filepath):
    """
    生成一个微缩版的 Volve WITSML 格式 XML 文件，用于解析测试。
    真实的 Volve WITSML 也是这种结构，只是数据量大几百万倍。
    """
    witsml_content = """<?xml version="1.0" encoding="UTF-8"?>
    <logs xmlns="http://www.witsml.org/schemas/1series" version="1.4.1.1">
        <log uid="Volve_Well_1">
            <name>Drilling_Time_Log</name>
            <indexType>date time</indexType>
            <!-- 变量字典 (Mnemonics) -->
            <logCurveInfo uid="TIME"><mnemonic>TIME</mnemonic><unit>s</unit></logCurveInfo>
            <logCurveInfo uid="WOB"><mnemonic>WOB</mnemonic><unit>klbf</unit><curveDescription>Weight on Bit</curveDescription></logCurveInfo>
            <logCurveInfo uid="ROP"><mnemonic>ROP</mnemonic><unit>m/h</unit><curveDescription>Rate of Penetration</curveDescription></logCurveInfo>
            <logCurveInfo uid="SPP"><mnemonic>SPP</mnemonic><unit>psi</unit><curveDescription>Standpipe Pressure</curveDescription></logCurveInfo>
            <logCurveInfo uid="RPM"><mnemonic>RPM</mnemonic><unit>rpm</unit><curveDescription>Rotary Speed</curveDescription></logCurveInfo>
            <!-- 真实传感器数据 (逗号分隔) -->
            <logData>
                <mnemonicList>TIME,WOB,ROP,SPP,RPM</mnemonicList>
                <unitList>s,klbf,m/h,psi,rpm</unitList>
                <data>2014-01-01T00:00:01Z,15.2,20.5,3005.1,120</data>
                <data>2014-01-01T00:00:02Z,15.4,20.1,3010.2,120</data>
                <data>2014-01-01T00:00:03Z,16.0,19.8,3015.0,121</data>
                <data>2014-01-01T00:00:04Z,16.1,19.5,3012.5,121</data>
                <!-- 模拟传感器缺失值空洞 -->
                <data>2014-01-01T00:00:05Z,15.8,,3008.0,120</data>
            </logData>
        </log>
    </logs>
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(witsml_content)

def parse_witsml_to_dataframe(xml_file, logger):
    """
    解析 WITSML 文件，提取变量字典与时序数据。
    """
    logger.info(f"正在解析 WITSML 文件: {xml_file}")
    
    # 注册命名空间，WITSML 标准里通常带有 xmlns
    namespaces = {'witsml': 'http://www.witsml.org/schemas/1series'}
    
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    # 定位到 log 节点
    log_node = root.find('witsml:log', namespaces)
    if log_node is None:
        logger.error("未找到 log 节点！")
        return None, None
        
    # 1. 提取变量字典 (Curve Info)
    curve_infos = log_node.findall('witsml:logCurveInfo', namespaces)
    variable_dict = {}
    for curve in curve_infos:
        mnemonic = curve.find('witsml:mnemonic', namespaces).text if curve.find('witsml:mnemonic', namespaces) is not None else ""
        unit = curve.find('witsml:unit', namespaces).text if curve.find('witsml:unit', namespaces) is not None else ""
        desc = curve.find('witsml:curveDescription', namespaces).text if curve.find('witsml:curveDescription', namespaces) is not None else ""
        variable_dict[mnemonic] = {"unit": unit, "description": desc}
        
    logger.info(f"成功提取 {len(variable_dict)} 个变量特征。")
    
    # 2. 提取时间序列数据
    log_data_node = log_node.find('witsml:logData', namespaces)
    mnemonic_list = log_data_node.find('witsml:mnemonicList', namespaces).text.split(',')
    
    data_nodes = log_data_node.findall('witsml:data', namespaces)
    parsed_rows = []
    
    for data_node in data_nodes:
        # WITSML 的数据是用逗号分隔的纯文本字符串，缺失值通常是连续的逗号 (如 15.8,,3008)
        row_values = data_node.text.split(',')
        parsed_rows.append(row_values)
        
    # 3. 组装为 Pandas DataFrame
    df = pd.DataFrame(parsed_rows, columns=mnemonic_list)
    
    # 将 TIME 列设为 datetime 格式，并将其他列转为数值型 (强制转换，无法转换的变 NaN)
    if 'TIME' in df.columns:
        df['TIME'] = pd.to_datetime(df['TIME'])
        
    for col in df.columns:
        if col != 'TIME':
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    return df, variable_dict

def main():
    logger = setup_logger("witsml_parser")
    logger.info("============== 模块四：Volve WITSML 数据解析器 ==============")
    
    # 创建一个临时目录存放 mock 数据
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw")
    os.makedirs(data_dir, exist_ok=True)
    mock_file_path = os.path.join(data_dir, "sample_volve_log.xml")
    
    # 生成 mock 文件
    generate_mock_witsml_file(mock_file_path)
    
    # 执行解析
    df, variable_dict = parse_witsml_to_dataframe(mock_file_path, logger)
    
    logger.info("\n--- 提取到的变量字典 ---")
    for var, info in variable_dict.items():
        logger.info(f"{var}: {info['description']} ({info['unit']})")
        
    logger.info("\n--- 解析得到的 DataFrame 前5行 ---")
    logger.info(f"\n{df.head()}")
    
    logger.info("\n--- 数据缺失值统计 (验证空洞解析是否成功) ---")
    logger.info(f"\n{df.isnull().sum()}")
    
    logger.info("============== 解析器测试完毕 ==============")

if __name__ == "__main__":
    main()