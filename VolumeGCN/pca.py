
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

import sys
import os
import numpy as np
import time
from args import parse_args
from module import Graphs, metricMeasure
sys.path.append("/")
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

def main():
    time_start = time.time()

    # 1. read graph data
    hp = parse_args()
    G = Graphs(hp)
    # random label
    print("The gexf graph data has been loaded.")
    node_num = len(G.nodes())
    hp.node_num = node_num

    x = []
    y = []

    # 2. allocate data
    # all nodes feature vector
    print('The dimension of each node : {}'.format(hp.vec_dim))
    f_init = np.zeros((hp.node_num, hp.vec_dim), dtype=np.float32)

    for n in range(node_num):
        for i in range(hp.vec_dim):
            f_init[n][i] = G.node[str(n)][str(i)]
        if G.node[str(n)]['cls'] != -1:
            x.append(list(f_init[n]))
            y.append(G.node[str(n)]['cls'])

    hp.label = len(set(y))
    hp.labeled_node = len(y)
    print('Number of all nodes : ', hp.node_num)
    print('Number of labeled nodes : ', hp.labeled_node)
    print('Number of trained labeled nodes : ', int(hp.labeled_node * hp.ratio))
    print('Number of test labeled nodes : ', int(hp.labeled_node * (1 - hp.ratio)))
    x = np.array(x)
    y = np.array(y, dtype=np.int)

    print(f_init[:5])
    pca = PCA(n_components=0.99, whiten=True, svd_solver='auto')
    new_x = pca.fit_transform(f_init)
    print(new_x[:10])
    inv_x = pca.inverse_transform(new_x)
    print(inv_x[:10])
    print(pca.explained_variance_ratio_)
    print(pca.n_components_)

    time_end = time.time()
    all_time = int(time_end - time_start)

    hours = int(all_time / 3600)
    minute = int((all_time - 3600 * hours) / 60)
    print()
    print('totally cost  :  ', hours, 'h', minute, 'm', all_time - hours * 3600 - 60 * minute, 's')


if __name__ == '__main__':
    main()
