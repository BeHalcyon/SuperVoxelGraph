import numpy as np

def train_data(hp, node_num, G, labeled_nodes):
    D = np.zeros((node_num, ))
    A = np.eye(node_num)
    N = len(labeled_nodes)
    train_nodes = labeled_nodes[:int(N*hp.ratio)]
    test_nodes = labeled_nodes[int(N*hp.ratio):]
    xs = np.array(train_nodes)
    ys = np.ones((int(N*hp.ratio),))
    unlabeled_nodes = list(set(G.nodes())-set(labeled_nodes))

    xu = np.concatenate((np.array(unlabeled_nodes), np.array(test_nodes)))
    yu = np.concatenate((np.zeros(node_num-N,), np.ones(N-int(N*hp.ratio),)))

    for edge in G.edges():
        A[int(edge[0])][int(edge[1])] += 1
        A[int(edge[1])][int(edge[0])] += 1
        D[int(edge[0])] += 1
        D[int(edge[1])] += 1
    D_ = np.diag(D**-0.5)
    L = np.matmul(np.matmul(D_, A), D_)
    return L, xs, ys, xu, yu