import os
import xml.etree.ElementTree as ET
import pandas as pd
from sklearn.preprocessing import StandardScaler

class VolveDataLoader:
    """
    Volve 钻井数据集标准加载与预处理工具类
    负责 WITSML (XML) 解析、特征提取、缺失值填补、平滑降噪与标准化。
    """
    def __init__(self, logger):
        self.logger = logger
        self.scaler = StandardScaler()
        self.variable_dict = {}

    def parse_witsml(self, xml_file):
        """解析 WITSML 文件，返回 DataFrame 和特征字典"""
        self.logger.info(f"==> 开始加载并解析 WITSML: {os.path.basename(xml_file)}")
        namespaces = {'witsml': 'http://www.witsml.org/schemas/1series'}
        
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
        except Exception as e:
            self.logger.error(f"XML 文件读取失败: {e}")
            return None

        log_node = root.find('witsml:log', namespaces)
        if log_node is None:
            self.logger.error("未找到有效的 <log> 节点，文件格式可能不匹配。")
            return None
            
        # 提取变量字典
        curve_infos = log_node.findall('witsml:logCurveInfo', namespaces)
        self.variable_dict = {}
        for curve in curve_infos:
            mnemonic = curve.find('witsml:mnemonic', namespaces).text if curve.find('witsml:mnemonic', namespaces) is not None else ""
            unit = curve.find('witsml:unit', namespaces).text if curve.find('witsml:unit', namespaces) is not None else ""
            desc = curve.find('witsml:curveDescription', namespaces).text if curve.find('witsml:curveDescription', namespaces) is not None else ""
            self.variable_dict[mnemonic] = {"unit": unit, "description": desc}
            
        # 提取时序数据
        log_data_node = log_node.find('witsml:logData', namespaces)
        if log_data_node is None:
             self.logger.error("未找到 <logData> 节点！")
             return None

        mnemonic_list = log_data_node.find('witsml:mnemonicList', namespaces).text.split(',')
        data_nodes = log_data_node.findall('witsml:data', namespaces)
        
        parsed_rows = [node.text.split(',') for node in data_nodes]
        df = pd.DataFrame(parsed_rows, columns=mnemonic_list)
        
        if 'TIME' in df.columns:
            df['TIME'] = pd.to_datetime(df['TIME'])
            
        # 转换为数值型，非数值转为 NaN
        for col in df.columns:
            if col != 'TIME':
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
        self.logger.info(f"解析完成，成功加载 {df.shape[0]} 行, {df.shape[1]} 列数据。")
        return df

    def preprocess(self, df, cols_to_process=None, window_size=3):
        """执行插值、平滑和 Z-Score 标准化"""
        self.logger.info("==> 开始执行数据清洗与预处理管线...")
        
        # 如果未指定列，则处理除了 TIME 以外的所有数值列
        if cols_to_process is None:
            cols_to_process = [col for col in df.columns if col != 'TIME']
            
        df_processed = df.copy()
        
        # 1. 线性插值填补空洞
        self.logger.info(f"[-] 填补缺失值 (Linear Interpolation)... (填补前缺失总数: {df_processed[cols_to_process].isnull().sum().sum()})")
        df_processed[cols_to_process] = df_processed[cols_to_process].interpolate(method='linear')
        df_processed[cols_to_process] = df_processed[cols_to_process].bfill().ffill()
        
        # 2. 滑动窗口滤波降噪
        self.logger.info(f"[-] 压制高频机械噪声 (Rolling Mean, window={window_size})...")
        df_processed[cols_to_process] = df_processed[cols_to_process].rolling(window=window_size, min_periods=1).mean()
        
        # 3. Z-Score 标准化
        self.logger.info("[-] 消除量纲差异 (Z-Score Standardization)...")
        df_processed[cols_to_process] = self.scaler.fit_transform(df_processed[cols_to_process])
        
        self.logger.info("==> 预处理管线执行完毕。")
        return df_processed

    def load_and_preprocess(self, xml_file, cols_to_process=None, window_size=3):
        """一键式端到端加载并处理数据"""
        df_raw = self.parse_witsml(xml_file)
        if df_raw is None:
            return None, None
            
        df_clean = self.preprocess(df_raw, cols_to_process, window_size)
        return df_clean, self.variable_dict

# 简单的模块自测逻辑 (仅在直接运行此文件时触发)
if __name__ == "__main__":
    from utils import setup_logger
    logger = setup_logger("data_loader_test")
    
    # 指向我们刚刚在 EXP-11 中生成的模拟 WITSML 文件
    test_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw", "sample_volve_log.xml")
    
    if os.path.exists(test_file):
        loader = VolveDataLoader(logger)
        # 一键端到端调用
        df_clean, var_dict = loader.load_and_preprocess(test_file, window_size=2)
        logger.info(f"\n清洗后的最终数据头部:\n{df_clean.head()}")
    else:
        logger.error("未找到测试文件，请先运行实验 10 生成 sample_volve_log.xml")