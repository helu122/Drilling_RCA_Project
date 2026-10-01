import os
import random
import numpy as np
import logging
from datetime import datetime

def set_seed(seed: int = 42):
    """全局锁定随机种子，确保结果绝对可复现"""
    random.seed(seed)
    np.random.seed(seed)

def setup_logger(exp_name: str):
    """自动将日志存入 results 文件夹，并在终端同步输出"""
    os.makedirs("results/logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"results/logs/{exp_name}_{timestamp}.log"
    
    logger = logging.getLogger(exp_name)
    logger.setLevel(logging.INFO)
    
    # 避免日志重复打印
    if not logger.handlers:
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        fh = logging.FileHandler(log_file, encoding='utf-8')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
    return logger