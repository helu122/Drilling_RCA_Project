import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from causalnex.structure.dynotears import from_pandas_dynamic
from src.utils import set_seed, setup_logger
from src.metrics import evaluate_causal_graph

def generate_linear_ts_data(N, T, max_lag=2, seed=42):
    """纯线性时序管网生成器 (回归 DYNOTEARS 的理论主场)"""
    np.random.seed(seed)
    data = np.random.randn(T, N) * 0.1
    true_links = {i: [] for i in range(N)}
    
    for i in range(N):
        true_links[i].append(((i, -1), np.random.uniform(0.2, 0.4)))
        
        num_parents = np.random.randint(1, 3)
        possible_parents = list(range(N))
        possible_parents.remove(i)
        parents = np.random.choice(possible_parents, num_parents, replace=False)
        
        for p in parents:
            lag = -np.random.randint(1, max_lag + 1)
            w = np.random.uniform(0.1, 0.3) * np.random.choice([-1, 1])
            true_links[i].append(((p, lag), round(w, 3)))
            
    # 纯线性滚动生成 (移除之前的 sin 和二次方项)
    for t in range(max_lag, T):
        for i in range(N):
            for (c, lag), w in true_links[i]:
                data[t, i] += w * data[t + lag, c]
                
    return data, true_links

def adapt_dynotears_to_tigramite(sm, N, tau_max):
    """结构翻译器"""
    pred_graph = np.empty((N, N, tau_max + 1), dtype=object)
    pred_graph.fill("")
    
    for u, v in sm.edges():
        if "_lag" in u and "_lag" in v:
            src_str, src_lag_str = u.split("_lag")
            tgt_str, tgt_lag_str = v.split("_lag")
            
            src_idx = int(src_str.split("_")[1])
            tgt_idx = int(tgt_str.split("_")[1])
            lag = int(src_lag_str)
            tgt_lag = int(tgt_lag_str)
            
            if tgt_lag == 0 and lag <= tau_max:
                pred_graph[src_idx, tgt_idx, lag] = "-->"
                
    return pred_graph

def main():
    logger = setup_logger("dynotears_linear")
    set_seed(42)
    
    node_scales = [5, 10, 15]
    T = 1000
    tau_max = 2
    
    logger.info("============== DYNOTEARS 线性主场回归测试 ==============")
    
    for N in node_scales:
        logger.info(f"\n---> [阶段测试] 正在解析 {N} 节点纯线性网络...")
        
        # 1. 生成纯线性数据
        data, true_links = generate_linear_ts_data(N, T, max_lag=tau_max)
        
        # 2. 【核心修复】连续优化算法极其依赖数据标准化
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        
        var_names = [f"Sensor_{i}" for i in range(N)]
        df = pd.DataFrame(data_scaled, columns=var_names)
        
        # 3. 运行 DYNOTEARS
        start_time = time.time()
        # 降低 w_threshold，防止标准化后的微弱权重被误杀
        sm = from_pandas_dynamic(df, p=tau_max, w_threshold=0.01) 
        end_time = time.time()
        
        # 4. 结构翻译与阅卷
        pred_graph = adapt_dynotears_to_tigramite(sm, N, tau_max)
        metrics = evaluate_causal_graph(true_links, pred_graph)
        
        logger.info(f"算法计算耗时: {end_time - start_time:.4f} 秒")
        logger.info(f"拓扑恢复指标: SHD={metrics['SHD']} | Precision={metrics['Precision']} | Recall={metrics['Recall']} | F1={metrics['F1_score']}")

    logger.info("\n============== 测试全部结束 ==============")

if __name__ == "__main__":
    main()