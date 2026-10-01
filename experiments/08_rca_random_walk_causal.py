import sys
import os
import numpy as np
import pandas as pd
import networkx as nx

# 尝试导入 PyRCA (如果不报错的话)
try:
    from pyrca.analyzers.random_walk import RandomWalk, RandomWalkConfig
    HAS_PYRCA = True
except ImportError:
    HAS_PYRCA = False

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import set_seed, setup_logger

def get_phase1_causal_graph():
    """
    模拟我们在阶段一（PCMCI/DYNOTEARS）学出来的真实钻井因果拓扑图
    因果链条： Sensor_1 (如排量) -> Sensor_3 (如立压) -> Sensor_4
    """
    G = nx.DiGraph()
    sensors = [f"Sensor_{i}" for i in range(5)]
    G.add_nodes_from(sensors)
    
    # 注入阶段一学到的因果先验边
    G.add_edge("Sensor_1", "Sensor_3")
    G.add_edge("Sensor_3", "Sensor_4")
    
    return G, sensors

def random_walk_rca_custom(causal_graph, anomalous_nodes, alpha=0.85):
    """
    用 NetworkX 纯手工完美复现 PyRCA 的 Random Walk 底层逻辑 
    (Personalized PageRank)
    """
    # 1. 反转因果图 (溯源需要逆流而上)
    reversed_G = causal_graph.reverse(copy=True)
    
    # 2. 个性化重启概率 (只从触发异常告警的节点发起游走)
    personalization = {node: 0.0 for node in reversed_G.nodes()}
    for node in anomalous_nodes:
        if node in personalization:
            personalization[node] = 1.0 / len(anomalous_nodes)
            
    # 3. 运行 PageRank 随机游走
    # alpha: 阻尼系数，代表继续沿着边游走的概率。1-alpha 代表回到告警节点的概率。
    root_cause_scores = nx.pagerank(reversed_G, alpha=alpha, personalization=personalization)
    
    # 4. 按嫌疑分降序排列
    ranked_sensors = sorted(root_cause_scores.items(), key=lambda item: item[1], reverse=True)
    return ranked_sensors

def main():
    logger = setup_logger("rca_random_walk")
    set_seed(42)
    
    logger.info("============== 模块三：基于因果图的随机游走根因排序 (RCA) ==============")
    
    # 1. 模拟输入数据
    G, sensors = get_phase1_causal_graph()
    logger.info(f"导入阶段一因果图边: {list(G.edges())}")
    
    # 模拟异常检测模块抛出的警报 (假设系统发现 Sensor_3 和 Sensor_1 都有异常)
    # 在真实场景中，往往下游受害节点(Sensor_3)的警报可能比源头还响亮
    detected_anomalies = ["Sensor_1", "Sensor_3"]
    logger.info(f"异常检测模块上报了异常节点集: {detected_anomalies}")
    
    # 2. 执行自定义图游走底层实现
    logger.info("\n--- 执行 NetworkX 底层随机游走 RCA ---")
    ranked_results = random_walk_rca_custom(G, detected_anomalies)
    
    for rank, (node, score) in enumerate(ranked_results):
        logger.info(f"Rank {rank+1}: {node} (根因嫌疑分: {score:.4f})")
        
    # 3. 如果安装了 PyRCA，演示其标准接口调用
    if HAS_PYRCA:
        logger.info("\n--- 执行 Salesforce PyRCA 标准库调用 ---")
        # PyRCA 通常要求将图转为 Pandas 邻接矩阵
        adj_matrix = nx.to_pandas_adjacency(G)
        config = RandomWalkConfig(graph=adj_matrix, alpha=0.85)
        rw_model = RandomWalk(config)
        # 传入异常节点列表
        pyrca_results = rw_model.find_root_causes(detected_anomalies).to_dict()
        logger.info(f"PyRCA 官方库计算结果: {pyrca_results}")
    else:
        logger.info("\n(未检测到 PyRCA 库，跳过官方接口调用，底层重写版已实现等价功能)")

    logger.info("\n============== 因果图 RCA 溯源完毕 ==============")

if __name__ == "__main__":
    main()