# -*- coding: utf-8 -*-
#/usr/bin/python3
from module import Graphs
import numpy as np
import argparse
import pickle
from args import parse_args
# os.environ['CUDA_VISIBLE_DEVICES'] = ""


import json
import time

def main():
    start = time.time()
    hp = parse_args()
    G = Graphs(hp)

    node_num = len(G.nodes())
    labeled_nodes = [0 for i in range(100)]
    hp.node_num = node_num
    hp.labeled_node = len(labeled_nodes)

    # print(hp)
    D = np.zeros((node_num,))
    A = np.eye(node_num)
    for edge in G.edges():
        A[int(edge[0])][int(edge[1])] += 1
        A[int(edge[1])][int(edge[0])] += 1
        D[int(edge[0])] += 1
        D[int(edge[1])] += 1
    D_ = np.diag(D ** -0.5)
    L = np.matmul(np.matmul(D_, A), D_)

    emb = np.zeros((node_num, hp.dim))

    for i in range(node_num):
        for j in range(hp.dim):
            emb[i][j] = G.node[str(i)][str(j)]
    w = (np.random.randn(hp.dim, hp.dim) / np.sqrt(hp.dim/2)).astype('float32')
    emb = np.matmul(np.matmul(L, np.matmul(np.matmul(L, emb), w)), w)
    f = open(hp.node_embedding, 'wb')
    pickle.dump(emb, f)

    print("Node embedding file has been saved. (Unsupervised)")
    elapsed = (time.time() - start)
    print("Time for unsupervised GCN : ", elapsed, "s.")


    # f = open(hp.node_embedding, 'rb')
    # emb = pickle.load(f)
    #
    # print(emb.shape)

if __name__ == '__main__':
    main()