import numpy as np

def train_data(hp, node_num, G, labeled_nodes):
    D = np.zeros((node_num, ))
    A = np.eye(node_num)
    N = len(labeled_nodes)
    train_nodes = labeled_nodes[:int(N)*hp.ratio]
    xs = np.array(labeled_nodes)
    ys = np.ones((hp.labeled_node,))
    unlabeled_nodes = set(G.nodes())-set(labeled_nodes)
    xu = np.array(unlabeled_nodes)
    for edge in G.edges():
        A[int(edge[0])][int(edge[1])] += 1
        A[int(edge[1])][int(edge[0])] += 1
        D[int(edge[0])] += 1
        D[int(edge[1])] += 1
    D_ = np.diag(D**-0.5)
    L = np.matmul(np.matmul(D_, A), D_)
    return L, xs, ys, xu
