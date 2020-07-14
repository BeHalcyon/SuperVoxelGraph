import numpy as np

def train_data(hp, node_num, G, labeled_nodes):
    D = np.zeros((node_num, ))
    # A为连接关系矩阵 D为度矩阵
    A = np.eye(node_num)
    N = len(labeled_nodes)
    node_id = [node[0] for node in labeled_nodes]
    # 训练节点和测试节点存储的数据都只是node原始的id
    train_nodes = node_id[:int(N*hp.ratio)]
    test_nodes = node_id[int(N*hp.ratio):]
    # xs为训练节点id组成的numpy数组
    xs = np.array(train_nodes)
    # ys为训练节点id对应的种类
    ys_ = []
    for i in range(int(N*hp.ratio)):
        ys_.append(labeled_nodes[i][1])
    ys = np.array(ys_)
    unlabeled_nodes = list(set(G.nodes())-set(node_id))
    # xu为测试节点id组成的numpy数组
    xu = np.array(test_nodes)
    # yu为测试节点对应的种类
    yu_ = []
    for i in range(int(N*hp.ratio), N):
        yu_.append(labeled_nodes[i][1])
    yu = np.array(yu_)


    for edge in G.edges():
        A[int(edge[0])][int(edge[1])] += 1
        A[int(edge[1])][int(edge[0])] += 1
        D[int(edge[0])] += 1
        D[int(edge[1])] += 1
    D_ = np.diag(D**-0.5)
    # # 拉普拉斯矩阵
    # L = np.matmul(np.matmul(D_, A), D_)

    import tensorflow as tf
    sess = tf.Session()
    T_D = tf.placeholder(dtype=tf.float32, shape=D_.shape)
    T_A = tf.placeholder(dtype=tf.float32, shape=A.shape)
    T_L = tf.matmul(tf.matmul(T_D, T_A), T_D)
    tensor_L = sess.run(T_L, feed_dict={T_D: D_, T_A: A})

    return tensor_L, xs, ys, xu, yu

def train_data_with_weight(hp, node_num, G, labeled_nodes):
    D = np.zeros((node_num, ))
    # A为连接关系矩阵 D为度矩阵
    A = np.eye(node_num)
    N = len(labeled_nodes)
    node_id = [node[0] for node in labeled_nodes]
    # 训练节点和测试节点存储的数据都只是node原始的id
    train_nodes = node_id[:int(N*hp.ratio)]
    test_nodes = node_id[int(N*hp.ratio):]
    # xs为训练节点id组成的numpy数组
    xs = np.array(train_nodes)
    # ys为训练节点id对应的种类
    ys_ = []
    for i in range(int(N*hp.ratio)):
        ys_.append(labeled_nodes[i][1])
    ys = np.array(ys_)
    unlabeled_nodes = list(set(G.nodes())-set(node_id))
    # xu为测试节点id组成的numpy数组
    xu = np.array(test_nodes)
    # yu为测试节点对应的种类
    yu_ = []
    for i in range(int(N*hp.ratio), N):
        yu_.append(labeled_nodes[i][1])
    yu = np.array(yu_)


    for edge in G.edges():
        A[int(edge[0])][int(edge[1])] += G[edge[0]][edge[1]]['weight']
        A[int(edge[1])][int(edge[0])] += G[edge[1]][edge[0]]['weight']
        D[int(edge[0])] += 1
        D[int(edge[1])] += 1
    D_ = np.diag(D**-0.5)


    import tensorflow as tf
    sess = tf.Session()
    T_D = tf.placeholder(dtype=tf.float32, shape=D_.shape)
    T_A = tf.placeholder(dtype=tf.float32, shape=A.shape)
    T_L = tf.matmul(tf.matmul(T_D, T_A), T_D)
    tensor_L = sess.run(T_L, feed_dict={T_D: D_, T_A: A})

    # # 拉普拉斯矩阵
    # L = np.matmul(np.matmul(D_, A), D_)
    # print(L)


    return tensor_L, xs, ys, xu, yu