import sys
import os
import networkx as nx
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
save_dir = os.path.join(os.path.dirname(__file__), "results_archive", "04_visualizations")
os.makedirs(save_dir, exist_ok=True)

def get_phase1_causal_graph():
    G = nx.DiGraph()
    sensors = [f"Sensor_{i}" for i in range(5)]
    G.add_nodes_from(sensors)
    G.add_edge("Sensor_1", "Sensor_3")
    G.add_edge("Sensor_3", "Sensor_4")
    return G

def plot_causal_graph(G, root_cause, symptoms, save_path):
    plt.figure(figsize=(9, 5))
    
    # 【优化 1】手动指定完美坐标：1->3->4 在上一排，0和2在下一排
    pos = {
        "Sensor_1": (0, 1),
        "Sensor_3": (1, 1),
        "Sensor_4": (2, 1),
        "Sensor_0": (0.5, 0),
        "Sensor_2": (1.5, 0)
    }
    
    node_colors = []
    for node in G.nodes():
        if node == root_cause:
            node_colors.append('#ff4d4d')  
        elif node in symptoms:
            node_colors.append('#ffa64d')  
        else:
            node_colors.append('#80bfff')  
            
    # 【优化 2】缩小节点，同时在 edges 中传入 node_size，让箭头刚好在节点边缘结束
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                           node_size=1500, edgecolors='black', linewidths=1.5)
    
    nx.draw_networkx_edges(G, pos, edge_color='gray', 
                           arrows=True, arrowsize=20, width=2, alpha=0.8,
                           node_size=1500)
    
    nx.draw_networkx_labels(G, pos, font_size=11, font_weight='bold')
    
    import matplotlib.lines as mlines
    legend_elements = [
        mlines.Line2D([0], [0], marker='o', color='w', markerfacecolor='#ff4d4d', markersize=10, label='Root Cause (Sensor 1)'),
        mlines.Line2D([0], [0], marker='o', color='w', markerfacecolor='#ffa64d', markersize=10, label='Symptom (Sensor 3)'),
        mlines.Line2D([0], [0], marker='o', color='w', markerfacecolor='#80bfff', markersize=10, label='Normal')
    ]
    
    # 【优化 3】将图例移出主画幅上方，绝对不遮挡节点
    plt.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, fontsize=10)
    
    # 【优化 4】增加边缘空白，防止节点被切断
    plt.margins(0.2)
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"因果图已成功保存至: {save_path}")
    plt.show()

def main():
    G = get_phase1_causal_graph()
    root_cause = "Sensor_1"
    symptoms = ["Sensor_3"] 
    save_path = os.path.join(save_dir, "EXP-10_causal_graph.png")
    plot_causal_graph(G, root_cause, symptoms, save_path)

if __name__ == "__main__":
    main()