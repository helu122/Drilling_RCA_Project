import numpy as np

def evaluate_causal_graph(true_links, pred_graph):
    """
    计算因果图推断的结构指标 (SHD, Precision, Recall, F1)
    true_links: 字典格式的标准答案
    pred_graph: Tigramite 预测出的图矩阵, 形状为 (N, N, tau_max+1)
    """
    N = pred_graph.shape[0]
    tau_max = pred_graph.shape[2] - 1
    
    # 1. 提取真实边集合: 格式为 (source_node, target_node, lag)
    true_edges = set()
    for target, causes in true_links.items():
        for cause_info in causes:
            source, lag = cause_info[0]
            true_edges.add((source, target, abs(lag)))
            
    # 2. 提取预测边集合
    pred_edges = set()
    for i in range(N):
        for j in range(N):
            for tau in range(tau_max + 1):
                # Tigramite 中 "-->" 代表 i 的过去(t-tau) 导致了 j 的现在(t)
                if pred_graph[i, j, tau] == "-->":
                    pred_edges.add((i, j, tau))
                    
    # 3. 计算混淆矩阵元素
    tp = len(true_edges.intersection(pred_edges)) # 真正例 (找对的边)
    fp = len(pred_edges - true_edges)             # 假正例 (多画的边)
    fn = len(true_edges - pred_edges)             # 假负例 (漏画的边)
    
    # 4. 计算量化指标
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # 结构汉明距离 (SHD): 多画的边 + 漏画的边 + 方向画反的边
    # 这里为了适配时序滞后，简化为 FP + FN
    shd = fp + fn
    
    return {
        "SHD": shd,
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1_score": round(f1, 4)
    }