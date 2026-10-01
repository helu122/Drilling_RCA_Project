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
    """
    自动生成 N 节点的稳定时序网络结构
    """
    np.random.seed(seed)
    # 初始化带有基础高斯噪声的数据矩阵
    data = np.random.randn(T, N) * 0.1
    true_links = {i: [] for i in range(N)}

    for i in range(N):
        # 1. 注入自相关性 (AR=1)，权重控制在 0.2~0.4，保证序列自身稳定
        true_links[i].append(((i, -1), np.random.uniform(0.2, 0.4)))
        
        # 2. 随机指定 1 到 2 个其他节点作为“原因节点”
        num_parents = np.random.randint(1, 3)
        possible_parents = list(range(N))
        possible_parents.remove(i)
        parents = np.random.choice(possible_parents, num_parents, replace=False)
        
        for p in parents:
            lag = -np.random.randint(1, max_lag + 1)
            # 严格压制外部权重在 0.1~0.3 之间，防止多节点累加后导致时序发散
            w = np.random.uniform(0.1, 0.3) * np.random.choice([-1, 1])
            true_links[i].append(((p, lag), round(w, 3)))

    # 3. 按时间步滚动生成时序数据
    for t in range(max_lag, T):
        for i in range(N):
            for (c, lag), w in true_links[i]:
                data[t, i] += w * data[t + lag, c]

    return data, true_links

def main():
    logger = setup_logger("pcmci_scalability")
    set_seed(42)
    
    # 我们依次测试 5, 10, 15 三个规模的节点集合
    node_scales = [5, 10, 15]
    T = 1000
    
    logger.info("============== 开始执行网络规模扩容测试 ==============")
    
    for N in node_scales:
        logger.info(f"\n---> [阶段测试] 正在构建与解析 {N} 节点网络...")
        
        # 生成数据与标准答案
        data, true_links = generate_stable_ts_data(N, T)
        var_names = [f"Sensor_{i}" for i in range(N)]
        dataframe = pp.DataFrame(data, var_names=var_names)
        
        # 初始化算法
        pcmci = PCMCI(dataframe=dataframe, cond_ind_test=ParCorr(significance='analytic'))
        
        # 记录运算耗时
        start_time = time.time()
        results = pcmci.run_pcmci(tau_max=2, pc_alpha=0.05)
        end_time = time.time()
        
        # 提取图矩阵并评分
        graph = results['graph']
        metrics = evaluate_causal_graph(true_links, graph)
        
        logger.info(f"算法计算耗时: {end_time - start_time:.4f} 秒")
        logger.info(f"拓扑恢复指标: SHD={metrics['SHD']} | Precision={metrics['Precision']} | Recall={metrics['Recall']} | F1={metrics['F1_score']}")

    logger.info("\n============== 测试全部结束 ==============")

if __name__ == "__main__":
    main()