import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

histogram_size = 256

label_histogram_array = np.loadtxt("label_histogram_array.txt") #缺省按照'%.18e'格式保存数据，以空格分隔
label_number = label_histogram_array.shape[0]

edge_array = np.loadtxt("edge_weight_array.txt")
print("label number: ", label_number)
print("edge information: ", edge_array.shape)

G = nx.Graph()

# print(G.nodes.data())

for i in range(label_number):
    G.add_node(i)

    for j in range(i+1, label_number):
        if edge_array[i][j] > 0:
            G.add_edge(i, j)

for i in range(label_number):
    for j in range(label_histogram_array[i].shape[0]):
        G.node[i][j] = str(label_histogram_array[i][j])


# print(G.graph)
# print(G.nodes.data())
# print(G.edges.data())

# nx.write_gpickle(G, "jet_mixfrac_0051_supervoxels.gpickle")
nx.write_gexf(G,'jet_mixfrac_0051_supervoxels.gexf')

print("SuperVoxel Graph has been calculated.")

#nx.write_gml(G,'jet_mixfrac_0051_supervoxels.gml')
# nx.write_pajek(G, "jet_mixfrac_0051_supervoxels.net")
# nx.draw(G, with_labels=True)
# plt.show()


