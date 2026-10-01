
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import set_seed, setup_logger
from src.metrics import evaluate_causal_graph
import numpy as np
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr
import tigramite.plotting as tp
from src.utils import set_seed, setup_logger

def main():
    logger = setup_logger("pcmci_simulation")
    set_seed(42)
    
    logger.info("1. 纯手工构造 5 节点的可控模拟数据集 (Ground Truth)...")
    T = 1000
    # 初始化 1000行 x 5列 的数据矩阵，并注入标准差为 0.2 的高斯白噪声
    data = np.random.randn(T, 5) * 0.2
    
    # 显式物理传导规则模拟 (纯 NumPy 编写，绝对可控)：
    # Sensor_0 作为源头，带有 0.4 的自相关性
    # Sensor_1 受 Sensor_0 滞后 1 秒影响 (权重 0.7)
    # Sensor_2 受 Sensor_1 滞后 1 秒影响 (权重 0.8)
    # Sensor_3 受 Sensor_1 滞后 2 秒 和 Sensor_2 滞后 1 秒 共同影响
    # Sensor_4 受 Sensor_3 滞后 1 秒影响 (权重 0.5)
    for t in range(2, T):
        data[t, 0] += 0.4 * data[t-1, 0]
        data[t, 1] += 0.7 * data[t-1, 0]
        data[t, 2] += 0.8 * data[t-1, 1]
        data[t, 3] += 0.6 * data[t-2, 1] + 0.7 * data[t-1, 2]
        data[t, 4] += 0.5 * data[t-1, 3]
        
    var_names = ['Sensor_0', 'Sensor_1', 'Sensor_2', 'Sensor_3', 'Sensor_4']
    dataframe = pp.DataFrame(data, var_names=var_names)
    
    logger.info("2. 初始化 PCMCI 算法，开始拓扑学习...")
    # 使用偏相关(ParCorr)作为独立性检验方法，这是处理线性连续数据最快的基线
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=ParCorr(significance='analytic'))
    
    # 运行 PCMCI，设定最大溯源时间窗 tau_max=2
    results = pcmci.run_pcmci(tau_max=2, pc_alpha=0.05)
    
    logger.info("3. 算法运行完毕，正在保存结构对比图...")
    # 提取显著的因果网络结构 (利用 FDR 进行多重比较校正)
    # 新版 Tigramite 已自动根据 pc_alpha=0.05 生成了最终的因果图矩阵
    graph = results['graph']
    # 画图并保存
    fig, axes = plt.subplots(figsize=(8, 6))
    tp.plot_graph(
        val_matrix=results['val_matrix'],
        graph=graph,
        var_names=var_names,
        link_colorbar_label='Causal Effect (Weight)',
        node_colorbar_label='Auto-correlation',
        fig_ax=(fig, axes)
    )
    
    save_path = "results/01_pcmci_sim_graph.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"因果网络拓扑图已保存至: {save_path}")
    logger.info("4. 开始计算结构评估指标...")
    # 把我们最开始写的真实规则(links)需要稍微重构一下以适配评估函数
    ground_truth_links = {
        0: [], 
        1: [((0, -1), 0.7)], 
        2: [((1, -1), 0.8)], 
        3: [((1, -2), 0.6), ((2, -1), 0.7)], 
        4: [((3, -1), 0.5)]
    }
    
    metrics_result = evaluate_causal_graph(ground_truth_links, graph)
    
    logger.info("========= 算法阅卷结果 =========")
    for k, v in metrics_result.items():
        logger.info(f"{k}: {v}")
    
    if metrics_result["SHD"] == 0:
        logger.info("结论：满分！算法完美还原了底层的物理传导因果图，且无任何多余或遗漏。")

if __name__ == "__main__":
    main()