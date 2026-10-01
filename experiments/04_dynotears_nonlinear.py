import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np
import pandas as pd
from causalnex.structure.dynotears import from_pandas_dynamic
from src.utils import set_seed, setup_logger
from src.metrics import evaluate_causal_graph

def generate_nonlinear_ts_data(N, T, max_lag=2, seed=42):
    """
    带非线性反馈的时序管网生成器 (模拟真实钻井中的摩阻、扭矩的非线性突变)
    """
    np.random.seed(seed)
    data = np.random.randn(T, N) * 0.1
    true_links = {i: [] for i in range(N)}
    
    for i in range(N):
        # 自身历史惯性
        true_links[i].append(((i, -1), np.random.uniform(0.2, 0.4)))
        
        # 随机指定 1~2 个外部原因节点
        num_parents = np.random.randint(1, 3)
        possible_parents = list(range(N))
        possible_parents.remove(i)
        parents = np.random.choice(possible_parents, num_parents, replace=False)
        
        for p in parents:
            lag = -np.random.randint(1, max_lag + 1)
            w = np.random.uniform(0.1, 0.3) * np.random.choice([-1, 1])
            true_links[i].append(((p, lag), round(w, 3)))
            
    # 按时间步滚动生成 (注入非线性映射)
    for t in range(max_lag, T):
        for i in range(N):
            for (c, lag), w in true_links[i]:
                raw_val = data[t + lag, c]
                # 核心非线性注入: 叠加正弦波动与二次方衰减，模拟钻柱粘滑振动(Stick-Slip)的非线性特征
                nonlinear_val = raw_val + 0.5 * np.sin(raw_val) - 0.1 * (raw_val ** 2)
                data[t, i] += w * nonlinear_val
                
    return data, true_links

def adapt_dynotears_to_tigramite(sm, N, tau_max):
    """把 causalnex 的输出适配为你写的 metrics.py 格式"""
    pred_graph = np.empty((N, N, tau_max + 1), dtype=object)
    pred_graph.fill("")
    
    # sm.edges 格式示例: ('Sensor_1_lag1', 'Sensor_2_lag0')
    for u, v in sm.edges():
        if "_lag" in u and "_lag" in v:
            src_str, src_lag_str = u.split("_lag")
            tgt_str, tgt_lag_str = v.split("_lag")
            
            src_idx = int(src_str.split("_")[1])
            tgt_idx = int(tgt_str.split("_")[1])
            lag = int(src_lag_str)
            tgt_lag = int(tgt_lag_str)
            
            # 我们只评估指向当前时刻 (lag0) 的滞后因果边
            if tgt_lag == 0 and lag <= tau_max:
                pred_graph[src_idx, tgt_idx, lag] = "-->"
                
    return pred_graph

def main():
    logger = setup_logger("dynotears_nonlinear")
    set_seed(42)
    
    node_scales = [5, 10, 15]
    T = 1000
    tau_max = 2
    
    logger.info("============== DYNOTEARS 非线性管网挑战赛 ==============")
    
    for N in node_scales:
        logger.info(f"\n---> [阶段测试] 正在解析 {N} 节点非线性网络...")
        
        # 1. 生成非线性数据
        data, true_links = generate_nonlinear_ts_data(N, T, max_lag=tau_max)
        var_names = [f"Sensor_{i}" for i in range(N)]
        df = pd.DataFrame(data, columns=var_names)
        
        # 2. 运行 DYNOTEARS (连续最优化寻优)
        start_time = time.time()
        # p 代表历史滞后阶数, w_threshold 是过滤微弱连接的权重截断阈值
        sm = from_pandas_dynamic(df, p=tau_max, w_threshold=0.05) 
        end_time = time.time()
        
        # 3. 结构翻译与阅卷
        pred_graph = adapt_dynotears_to_tigramite(sm, N, tau_max)
        metrics = evaluate_causal_graph(true_links, pred_graph)
        
        logger.info(f"算法计算耗时: {end_time - start_time:.4f} 秒")
        logger.info(f"拓扑恢复指标: SHD={metrics['SHD']} | Precision={metrics['Precision']} | Recall={metrics['Recall']} | F1={metrics['F1_score']}")

    logger.info("\n============== 官方示例闭环验证完毕 ==============")

if __name__ == "__main__":
    main()