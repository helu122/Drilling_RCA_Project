import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np
from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr
from src.utils import set_seed, setup_logger
from src.metrics import evaluate_causal_graph

def generate_stable_ts_data(N, T, max_lag=2, seed=42):
    """复用之前的稳定时序生成器"""
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
    for t in range(max_lag, T):
        for i in range(N):
            for (c, lag), w in true_links[i]:
                data[t, i] += w * data[t + lag, c]
    return data, true_links

def main():
    logger = setup_logger("pcmci_alpha_tuning")
    set_seed(42)
    
    N = 15
    T = 1000
    
    # 我们测试四个不同严苛程度的阈值
    alphas_to_test = [0.1, 0.05, 0.01, 0.001]
    
    logger.info(f"============== 开始执行 15 节点 pc_alpha 敏感度测试 ==============")
    
    # 保证所有阈值测试使用的是同一套底层数据和同一套物理规律
    data, true_links = generate_stable_ts_data(N, T)
    var_names = [f"Sensor_{i}" for i in range(N)]
    dataframe = pp.DataFrame(data, var_names=var_names)
    
    for alpha in alphas_to_test:
        logger.info(f"\n---> [超参数测试] 当前 pc_alpha = {alpha}")
        
        pcmci = PCMCI(dataframe=dataframe, cond_ind_test=ParCorr(significance='analytic'))
        results = pcmci.run_pcmci(tau_max=2, pc_alpha=alpha)
        
        graph = results['graph']
        metrics = evaluate_causal_graph(true_links, graph)
        
        logger.info(f"评估指标: SHD={metrics['SHD']} | Precision={metrics['Precision']:.4f} | Recall={metrics['Recall']:.4f} | F1={metrics['F1_score']:.4f}")

    logger.info("\n============== 测试全部结束 ==============")

if __name__ == "__main__":
    main()